import torch
import active_adaptation
from active_adaptation.envs.mdp import Command
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    yaw_quat,
    quat_from_yaw,
    wrap_to_pi,
    normalize,
    sample_quat_yaw
)
from isaaclab.utils.warp import raycast_mesh

import warp as wp

wp.init()

@wp.kernel
def compute_avoidance_kernel(
    grid: wp.uint64,
    grid_points: wp.array(dtype=wp.vec3),     # 树点
    query_points: wp.array(dtype=wp.vec3),    # 机器人位置
    goal_dirs: wp.array(dtype=wp.vec3),       # 归一化目标方向（z=0）
    bias_sign: wp.array(dtype=wp.int32),      # 每env固定±1，打破对称
    output: wp.array(dtype=wp.vec3),
    radius: float,    # SENSE_RADIUS
    beta: float,      # BETA
    kappa: float,     # KAPPA
    inflation: float  # INFLATION
):
    tid = wp.tid()
    p = query_points[tid]
    g = goal_dirs[tid]
    g.z = 0.0

    rep = wp.vec3(0.0, 0.0, 0.0)  # 径向排斥
    tan = wp.vec3(0.0, 0.0, 0.0)  # 切向绕行

    q = wp.hash_grid_query(grid, p, radius)
    index = int(0)
    eps = 1.0e-6

    while wp.hash_grid_query_next(q, index):
        obs = grid_points[index]
        v = p - obs
        v.z = 0.0
        d = wp.length(v)
        if d < eps or d > radius:
            continue

        # 软膨胀
        d_eff = wp.max(d - inflation, 0.02)
        # 近距权重：靠近→权重大，并带 1/d_eff
        s = wp.max(1.0 - d_eff / wp.max(radius - inflation, 0.05), 0.0)
        w = wp.pow(s, beta) / d_eff

        # 径向排斥（obs->robot）
        ur = v / d
        rep += ur * w

        # 前方威胁筛选 + 切向
        ob_dir = -ur                               # robot->obs
        forward_threat = wp.max(wp.dot(g, ob_dir), 0.0)
        t_left = wp.vec3(-ur.y, ur.x, 0.0)         # 左手切向（+90°）

        z = g.x * ob_dir.y - g.y * ob_dir.x        # cross(g, ob_dir).z
        sgn = 1.0
        if z < -1.0e-6:
            sgn = -1.0
        elif wp.abs(z) <= 1.0e-6:
            sgn = float(bias_sign[tid])

        tan += t_left * sgn * (w * kappa * forward_threat)

    out = rep + tan
    out.z = 0.0
    output[tid] = out


@wp.kernel
def sample_target_kernel(
    env_ids: wp.array(dtype=wp.int64),
    root_pos_w: wp.array(dtype=wp.vec3),
    target_pos_w: wp.array(dtype=wp.vec3),
    terrain_types: wp.array(dtype=wp.int32),
    seed: wp.int32,
):
    env_id = env_ids[wp.tid()]
    seed_ = wp.rand_init(seed, wp.int32(env_id))

    terrain_type = terrain_types[env_id]
    if terrain_type == 5:
        # sample from [0, 1], [1, 0], [0, -1], [-1, 0]
        alpha = float(wp.randi(seed_, 0, 4)) * wp.PI / 2.0
        offset = wp.vec3(wp.cos(alpha), wp.sin(alpha), 0.0) * 4.0
        result = root_pos_w[env_id] + offset
    else:
        offset = wp.sample_unit_cube(seed_)
        offset.z = 0.0
        result = root_pos_w[env_id] + offset * 4.0
        x_border = wp.round(result.x / 8.0) * 8.0
        y_border = wp.round(result.y / 8.0) * 8.0
        x_dist = wp.abs(result.x - x_border)
        y_dist = wp.abs(result.y - y_border)
        if x_dist < y_dist:
            result.x = x_border
        elif x_dist > y_dist:
            result.y = y_border
        else:
            result.x = x_border
            result.y = y_border
    target_pos_w[env_id] = result


TEST_TERRAIN_ORIGINS = [
    [4.3, -0.3, 0.0],
    [7.5, 1.25, 0.0],
    [5.7, 2.5, 0.0],
    [0.35, 2.5, 0.0],
    [0.35, -2.0, 0.0],
    [-4.0, -2.0, 0.0],
    [4.1, 2.3, 0.0],
]


class LocoNavigation(Command):
    def __init__(
        self,
        env,
        feet_names: str,
        pelvis_names: str,
        robot_name: str,
        max_target_height: float,
        use_curriculum: bool = True,
        offset: float = 0.2,
        random_command: float = False,
        stand_prob: float = 0.05,
        reach_distance_thres: float = 0.2
    ):
        super().__init__(env)
        self.feet_names = feet_names
        self.use_curriculum = (
            use_curriculum
            and self.env.backend == "isaac"
            and self.env.training
        )
        self.resample_interval = int(self.env.cfg.allocate_time / self.env.step_dt)
        self.resample_distance_thres = reach_distance_thres
        self.random_command = random_command
        self.stand_prob = stand_prob
        self.allocate_time = self.env.cfg.allocate_time
        self.curri_delay_ratio = 0.0
        if self.env.backend == "isaac":
            from active_adaptation.envs.terrain import BetterTerrainImporter, BetterTerrainGenerator
            self.terrain_importer: BetterTerrainImporter = self.env.scene.terrain
            self.terrain_generator: BetterTerrainGenerator = self.terrain_importer.terrain_generator

            if self.use_curriculum and self.terrain_importer.cfg.terrain_type == "generator":
                assert self.terrain_generator.cfg.curriculum, "Curriculum must be enabled in the terrain generator."
            if self.terrain_importer.cfg.terrain_type == "usd":
                self._origins = torch.tensor(TEST_TERRAIN_ORIGINS, device=self.device)
                self._test_target = torch.zeros(self.num_envs, 3, device=self.device) # will be sampled from _origins

        with torch.device(self.device):
            self.target_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_direction = torch.zeros(self.num_envs, 3)
            self.toe_heel_height = torch.zeros(self.num_envs, 2)
            self.time_alloted = torch.zeros(self.num_envs, 1)
            self.time_elapsed = torch.zeros(self.num_envs, 1)
            self.origin_pos_w = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
            self.frame_marker = VisualizationMarkers(
                FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/target_pose",
                )
            )
            self.frame_marker.set_visibility(True)
        elif self.env.backend == "mujoco":
            self.target_marker = self.env.scene.create_sphere_marker(radius=0.05, rgba=(0., 0., 1., 1.))
        # preset values for acceleration
        self.contact_foot_ids = self.env.scene.sensors["contact_forces"].find_bodies(feet_names)[0]
        self.foot_ids = self.asset.find_bodies(feet_names)[0]
        self.body_pelvis_id = self.asset.find_bodies(pelvis_names)[0]
        self.robot_name = robot_name
        self.max_target_height = max_target_height
        self.knee_joint_ids = self.asset.find_joints(".*knee.*")[0]
        self.knee_action_min = self.asset.data.joint_pos_limits[:, self.knee_joint_ids, 0]
        self.knee_action_max = self.asset.data.joint_pos_limits[:, self.knee_joint_ids, 1]
        self.hip_pitch_joint_ids = self.asset.find_joints(".*hip_pitch.*")[0]
        self.hip_pitch_action_min = self.asset.data.joint_pos_limits[:, self.hip_pitch_joint_ids, 0]
        self.hip_pitch_action_max = self.asset.data.joint_pos_limits[:, self.hip_pitch_joint_ids, 1]

        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        self.head_vec =torch.tensor([0.0, 0.0, 0.55], device=self.device).expand(self.num_envs, 3)
        self.offsets = torch.tensor([
            [-0.5, -0.5, 0.0],
            [-0.5,  0.0, 0.0],
            [-0.5,  0.5, 0.0],
            [ 0.0, -0.5, 0.0],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.5, 0.0],
            [ 0.5, -0.5, 0.0],
            [ 0.5,  0.0, 0.0],
            [ 0.5,  0.5, 0.0],
        ], device=self.device).repeat(self.num_envs, 1, 1).reshape(self.num_envs * 9, -1)  # shape: [9, 3]
        
        self.torso_id = self.asset.find_bodies("torso_link")[0]
        self.head_offset = offset
        self.terrain_types = torch.zeros((self.num_envs, 3), dtype=torch.long, device=self.device)
        self.raw_terrain_types = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.move_mask = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        self.has_tree = False
        self.seed = wp.rand_init(0 + active_adaptation.get_local_rank())

        if self.env.backend == "isaac":
            from .terrain.hussar_terrain import TREE_INFOS
            if len(TREE_INFOS) > 0:
                self.has_tree = True
                self.grid = wp.HashGrid(512, 1024, 4, device=wp.get_device(str(self.device)))
                self.wp_tree = wp.from_torch(torch.tensor(TREE_INFOS, device=self.device, dtype=torch.float32), dtype=wp.vec3)
                self.grid.build(self.wp_tree, radius=1.5)
        self.update()
        self.last_pos = torch.zeros(self.num_envs, 2, device=self.device)


    def compute_direction(self,
                        root_pos: torch.Tensor,
                        goal_dir_xy: torch.Tensor,          
                        radius: float = 0.6,                
                        beta: float = 2.0,                  
                        kappa: float = 0.8,                 
                        inflation: float = 0.2,             
                        ema: float = 0.6                    
                        ) -> torch.Tensor:
        if not self.has_tree:
            return torch.zeros_like(root_pos)

        N = root_pos.shape[0]
        device = root_pos.device

        if not hasattr(self, "_bias_sign") or self._bias_sign.numel() != N:
            self._bias_sign = torch.randint(low=0, high=2, size=(N,), device=device).mul_(2).sub_(1).to(torch.int32)

        if ema > 0.0 and (not hasattr(self, "_avoid_ema") or self._avoid_ema.shape[0] != N):
            self._avoid_ema = torch.zeros_like(root_pos)

        out = torch.zeros_like(root_pos)
        root_pos_wp = wp.from_torch(root_pos, dtype=wp.vec3)
        goal_dir_wp = wp.from_torch(goal_dir_xy, dtype=wp.vec3)
        bias_wp     = wp.from_torch(self._bias_sign)
        out_wp      = wp.from_torch(out, dtype=wp.vec3)

        wp.launch(
            compute_avoidance_kernel,
            dim=N,
            inputs=[
                self.grid.id, self.wp_tree, root_pos_wp, goal_dir_wp, bias_wp,
                out_wp, float(radius), float(beta), float(kappa), float(inflation)
            ],
            device=self.grid.device
        )
        avoid = wp.to_torch(out_wp)  # [N,3]

        if ema > 0.0:
            self._avoid_ema = ema * self._avoid_ema + (1.0 - ema) * avoid
            avoid = self._avoid_ema

        return avoid

    @property
    def command(self):
        # target_pos_b = quat_rotate_inverse(
        #     self.asset.data.body_quat_w[:, self.torso_id].squeeze(1),
        #     self.target_pos_w - self.asset.data.root_pos_w
        # )    

        relative_pos = self.target_pos_w - self.asset.data.body_pos_w[:, self.torso_id].squeeze(1)
        relative_pos[:, 2] = 0.0
        quat = yaw_quat(self.asset.data.body_quat_w[:, self.torso_id])
        target_pos_b = quat_rotate_inverse(
            quat.squeeze(1),
            relative_pos
        )

        if self.random_command:
            noise_pos = torch.randn_like(target_pos_b, device=self.device, dtype=torch.float32) * 0.1    
            target_pos_b = target_pos_b + noise_pos
        
        time_elapsed = self.time_elapsed.clamp_max(self.time_alloted)
        time_rest = self.time_alloted - self.time_elapsed
        new_command = torch.cat([
            target_pos_b[:, :2], # 2
            time_elapsed, # 1
            time_rest
        ], dim=-1) # [num_envs, 5]
        old_command = torch.cat([
            self.last_pos,
            time_elapsed,
            time_rest
        ], dim=-1)
        command = torch.where(
            (self.env.episode_length_buf % 2 == 0).unsqueeze(1),
            new_command,
            old_command,
        )
        self.last_pos = target_pos_b[:, :2]
        return command
        
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(4),
            signs=torch.tensor([1, -1, 1, 1])
        )

    def reset(self, env_ids: torch.Tensor):
        self.time_alloted[env_ids] = self.resample_interval / 50.
        self.time_elapsed[env_ids] = 0.
        self.sample_target(env_ids)

    def update(self):
        resample = (self.time_elapsed >= (self.resample_interval / 50.))
        self.target_pos_w = torch.where(resample, self.origin_pos_w, self.target_pos_w)
        self.target_reached = torch.where(resample, 0, self.target_reached)
        self.time_elapsed = torch.where(resample, 0.0, self.time_elapsed)
        
        self.pos_diff = self.target_pos_w - self.asset.data.root_pos_w
        self.pos_diff_norm = self.pos_diff[:, :2].norm(dim=-1, keepdim=True)
        self.target_reached = (self.pos_diff_norm < self.resample_distance_thres)
        self.time_elapsed += self.env.step_dt
        self.compute_target_head_height()
    
    def compute_target_head_height(self):
        target_vec = self.target_pos_w[:, :2] - self.asset.data.root_pos_w[:, :2]
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        target_quat = quat_from_yaw(target_yaw)
        
        test_pos = self.asset.data.root_pos_w.clone()
        test_pos[:, :2] += 0.45 * normalize(target_vec)
        
        test_points = test_pos[:, None, :] + quat_rotate(
            target_quat.unsqueeze(1).repeat(1, 9, 1).reshape(self.num_envs * 9, -1),
            self.offsets
        ).reshape(self.num_envs, 9, 3)
        self.est_height = self.get_required_head_height_at(test_points, self.head_offset).clamp(max=self.max_target_height)

        
        head_position = self.asset.data.body_pos_w[:, self.torso_id].squeeze(1) + quat_rotate(
            self.asset.data.root_quat_w,
            self.head_vec
        )
        self.head_height = (head_position[:, 2] - self.env.get_ground_height_at(head_position)).reshape(self.num_envs, 1)

    def sample_init(self, env_ids):
        
        if self.use_curriculum and self.env.episode_count > 1 and self.env.training:
            # time_remaining = (self.env.max_episode_length - self.env.episode_length_buf[env_ids, None]) * self.env.step_dt
            # distance_commanded = (self.distance_commanded[env_ids] + self.command_speed[env_ids] * time_remaining)
            move_up = (self.env.stats['loco']['reaching_target'] > 2.0).reshape(self.num_envs)[env_ids]
            move_down = (self.env.stats['loco']['reaching_target'] < 1.0).reshape(self.num_envs)[env_ids]
            # move_up = move_up & ~move_down
            self.terrain_importer.update_env_origins(env_ids, move_up, move_down)
            self._origins = self.terrain_importer.env_origins.clone()
            self.env.extra["curriculum/terrain_level"] = self.terrain_importer.terrain_levels.float().mean()
        self.env.extra["curriculum/curri_delay_ratio"] = self.curri_delay_ratio

        init_root_state = self.init_root_state[env_ids]

        if self.env.backend == "isaac":
            if self.terrain_importer.cfg.terrain_type == "plane":
                origins = self.env.scene.env_origins[env_ids]
            elif self.terrain_importer.cfg.terrain_type == "generator": # generator
                idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
                origins = self._origins[idx]
            elif self.terrain_importer.cfg.terrain_type == "usd":
                idx = torch.multinomial(torch.ones(len(self._origins), device=self.device).expand(len(env_ids), -1), 2, replacement=False)
                origins = self._origins[idx[:, 0]]
                self._test_target[env_ids] = self._origins[idx[:, 1]]
            else:
                raise ValueError(f"Unsupported terrain type: {self.terrain.cfg.terrain_type}")
            
            if self.terrain_importer.cfg.terrain_type == "generator":# and self.env.training:
                num_cols, num_rows = self.terrain_generator.cfg.num_cols, self.terrain_generator.cfg.num_rows
                sub_terrain_size = self.terrain_generator.cfg.size[0]
                sub_terrain_types = self.terrain_generator.sub_terrain_types.reshape(num_rows, num_cols).clone().to(self.device)
                env_x = (torch.floor(origins[:, 0] / sub_terrain_size) + num_rows // 2).to(torch.long)
                env_y = (torch.floor(origins[:, 1] / sub_terrain_size) + num_cols // 2).to(torch.long)
                self.raw_terrain_types[env_ids] = sub_terrain_types[env_x, env_y]
                self.terrain_types[env_ids, :] = (self.raw_terrain_types[env_ids].unsqueeze(-1) >> torch.arange(2, -1, -1).to(self.device)) & 1
            
            orientations = quat_mul(
                init_root_state[:, 3:7],
                sample_quat_yaw(len(env_ids), device=self.device)
            )
        
        elif self.env.backend == "mujoco":
            origins = torch.zeros(len(env_ids), 3, device=self.device)
            orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(env_ids), 4)
        
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = orientations
        self.origin_pos_w[env_ids] = origins
        return init_root_state

    def mask2id(self, mask: torch.Tensor):
        return mask.nonzero().squeeze(-1)
    
    def sample_target(self, env_ids: torch.Tensor):
        if self.env.backend == "isaac" and self.terrain_importer.cfg.terrain_type == "usd":
            self.target_pos_w[env_ids] = self._test_target[env_ids]
            return
        if self.env.backend == "mujoco":
            self.target_pos_w[env_ids] = torch.tensor([4.5, 0.0, 1.0], device=self.device)
            return
        wp.launch(
            sample_target_kernel,
            dim=[len(env_ids),],
            inputs=[
                wp.from_torch(env_ids, dtype=wp.int64, return_ctype=True),
                wp.from_torch(self.asset.data.root_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.target_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.raw_terrain_types, dtype=wp.int32, return_ctype=True),
                self.seed,
            ],
            device=wp.get_device(str(self.device))
        )
        self.seed = wp.rand_init(int(self.seed))
        self.target_reached[env_ids] = 0
    
    def debug_draw(self):
        # self.env.sim.set_camera_view(
        #     self.asset.data.root_pos_w[0].cpu() + torch.tensor([2., 2., 1.]),
        #     self.asset.data.root_pos_w[0].cpu()
        # )
        if self.env.backend == "isaac":
            # pelvis axis
            quat = self.asset.data.root_quat_w
            scales = torch.tensor([0.2, 0.2, 0.2]).expand(len(self.asset.data.root_pos_w), 3)
            # self.frame_marker.visualize(self.asset.data.root_pos_w, quat, scales=scales)
            
            diff = self.target_pos_w - self.asset.data.root_pos_w
            diff[:, 2] = 0.0
            # line from current position to target position
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                diff,
                color=(1, 0, 0, 1)
            )
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w,
                self.target_direction,
                color=(0, 1, 0, 1)
            )
        elif self.env.backend == "mujoco":
            self.target_marker.geom.pos = self.target_pos_w[0]
            
    def get_required_head_height_at(self, pos: torch.Tensor, offset: float) -> torch.Tensor:
        base_height = pos[:, :, 2] - self.env.get_ground_height_at(pos)
        
        bshape = pos.shape[:-1]
        ray_starts = pos.clone().reshape(-1, 3)
        ray_directions = torch.tensor([0., 0., 1.], device=self.device)
        ray_hits = raycast_mesh(
            ray_starts=ray_starts.reshape(-1, 3),
            ray_directions=ray_directions.expand(bshape.numel(), 3),
            max_dist=100.,
            mesh=self.env.ground_mesh,
            return_distance=False,
        )[0]
        base_upper = (ray_hits - ray_starts).norm(dim=-1)
        base_upper = base_upper.nan_to_num(10.).reshape(*bshape)
        assert not base_upper.isnan().any()
        required_height = (base_upper + base_height - offset)
        return torch.min(required_height, dim=1, keepdim=True)[0]

