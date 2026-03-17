import torch
import numpy as np
import einops
import warp as wp
from typing import Tuple, TYPE_CHECKING, Optional

import active_adaptation
from active_adaptation.envs.mdp import Observation
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    quat_from_euler_xyz
)
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaacsim.core.utils.stage import get_current_stage
try:
    from simple_raycaster.raycaster import MultiMeshRaycaster
    from simple_raycaster.helpers import voxelize_wp
except ImportError:
    raise ImportError(
        "simple-raycaster is not installed."
        "Please install it via `pip install git+https://github.com/btx0424/simple-raycaster.git`"
    )

from .command import LocoNavigation

class target_head_height(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return (self.command_manager.est_height - self.command_manager.head_height).reshape(self.num_envs, 1)
    
    def symmetry_transforms(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([0]),
            signs=torch.tensor([1.]),
        )


class toe_heel_height(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return self.command_manager.toe_heel_height.reshape(self.num_envs, 2)
    
    def symmetry_transforms(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([1, 0]),
            signs=torch.tensor([1., 1.]),
        )
        

class target_direction_b(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return quat_rotate_inverse(
            self.command_manager.asset.data.root_quat_w,
            self.command_manager.target_direction
        )[:, :2].reshape(self.num_envs, 2)
    
    def symmetry_transforms(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([0, 1]),
            signs=torch.tensor([1,-1]),
        )

class terrain_types(Observation[LocoNavigation]):
    def __init__(self, env):
        super().__init__(env)
    
    def compute(self) -> torch.Tensor:
        return self.command_manager.terrain_types.float()


class lidar_voxel_map(Observation):
    def __init__(
        self,
        env,
        pattern: str = "n 1 x z y",
        noise_std: float = 0.0,
        include_self: bool = False,
        num_scan: int = 128,
        random_offset: bool = False,
        resolution: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        hole_prob: float = 0.0,
        obs_delay_range: Tuple[int, int] = (5, 10),  # 单位：仿真步（steps）
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_id = self.asset.find_bodies("torso_link")[0]

    # -------------------------------
        # 体素网格设置
        # -------------------------------
        self.resolution = tuple(resolution)
        if len(self.resolution) != 3:
            raise ValueError("resolution must be a sequence of 3 elements")
        # self.larger_fov = False
        # if self.larger_fov:
        self.shape = (32, 32, 40)
        # else:
        # self.shape = (32, 32, 32)  # (Dx, Dy, Dz)
        self.pattern = f"n x y z -> {pattern}"  # einops 输出维度顺序

        self.noise_std = noise_std
        self.include_self = include_self
        self.num_scan = num_scan
        self.random_offset = random_offset
        self.hole_prob = hole_prob
        self.obs_delay_range = obs_delay_range  # 步数范围（稍后换算为帧）
    
        # 主设备/环境信息来自 Observation 基类
        self.grid = torch.zeros(
            self.num_envs, *self.shape, device=self.device, dtype=torch.bool
        )
        self.grid_obs = torch.zeros_like(self.grid)    # 对外发布的“延迟后”占据栅格
        self.left_grid = torch.zeros_like(self.grid)   # 前雷达
        self.right_grid = torch.zeros_like(self.grid)  # 后雷达
        self.force_full_scan = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 预计算体素中心（机器人躯干局部系）
        self.grid_shape = torch.as_tensor(self.shape, device=self.device)
        self.grid_half_shape = self.grid_shape // 2
        self.grid_half_size = self.grid_half_shape * torch.tensor(self.resolution, device=self.device)

        grid_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(n, device=self.device) for n in self.shape],
                indexing="ij",
            ),
            dim=-1,
        )  # [Dx, Dy, Dz, 3]
        self.grid_centers = (
            grid_coords * torch.tensor(self.resolution, device=self.device)
            - self.grid_half_size
            + torch.tensor(self.resolution, device=self.device) / 2
        )
        del grid_coords

        self.lidar_pos = torch.tensor(
            [
                [0.12734, 0.00007, 0.17622],   # front
                [-0.11284, -0.0004, 0.17493], # back
            ],
            device=self.device,
        )

        angle = (96 / 180.0) * torch.pi
        vangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
        hangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
        vgrid, hgrid = torch.meshgrid(vangles, hangles, indexing="ij")
        vsin, vcos = vgrid.sin(), vgrid.cos()
        hsin, hcos = hgrid.sin(), hgrid.cos()

        lidar_dirs_front = torch.stack([vcos * hcos, vcos * hsin, vsin], dim=-1)
        lidar_dirs_back  = torch.stack([-vcos * hcos, -vcos * hsin, vsin], dim=-1)
        self.lidar_ray_dirs = torch.stack([lidar_dirs_front, lidar_dirs_back], dim=0)  # [2, S, S, 3]
        del vangles, hangles, vgrid, hgrid, vsin, vcos, hsin, hcos, lidar_dirs_front, lidar_dirs_back

        # 位姿噪声（在 reset 时重采样）
        self.pos_noise = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.rot_noise = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.rot_noise[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # -------------------------------
        # 扫描/发布节拍：10 Hz（每 5 步）
        # -------------------------------
        self.scan_period_steps = 5  # 固定 10 Hz

        # 前/后雷达相位偏置（单位：步；后雷达随机偏置）
        self.scan_time_offset = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.int64
        )
        self.scan_time_offset[:, 1] = torch.randint(
            -4, 5, (self.num_envs,), device=self.device
        )

        # -------------------------------
        # Raycaster 初始化（Isaac 或 Mujoco）
        # -------------------------------
        if self.env.backend == "isaac":
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/grid_map",
                    markers={
                        "griddot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.8, 0.0, 0.8)
                            ),
                        ),
                    },
                )
            )
            self.marker.set_visibility(True)

            paths = []
            if self.include_self:
                for body_name in self.asset.body_names:
                    paths.append(f"/World/envs/env_0/Robot/{body_name}/visuals")
            self.raycaster = MultiMeshRaycaster.from_prim_paths(
                paths,
                stage=get_current_stage(),
                device=wp.get_device(str(self.device)),
                simplify_factor=0.0,
            )
            self.raycaster.add_mesh(self.env.ground_mesh)

        elif self.env.backend == "mujoco":
            self.marker_front = self.env.scene.create_sphere_marker(
                radius=0.05, rgba=(1.0, 0.0, 0.0, 1.0)
            )
            self.marker_back = self.env.scene.create_sphere_marker(
                radius=0.05, rgba=(0.0, 1.0, 0.0, 1.0)
            )
            body_names = self.asset.body_names
            self.raycaster = MultiMeshRaycaster.from_MjModel(
                body_names=body_names,
                model=self.asset.mj_model,
                device=wp.get_device(str(self.device)),
                simplify_factor=0.0,
            )
            self.raycaster.add_mesh(self.env.ground_mesh)

        print(self.raycaster)

        # -------------------------------
        # 环形缓冲（按“帧”长度）
        # -------------------------------
        # 1) obs_delay_range 为“步”，转换为“帧”上界
        self.max_delay_steps  = int(max(self.obs_delay_range))
        self.max_delay_frames = (self.max_delay_steps + self.scan_period_steps - 1) // self.scan_period_steps  # ceil(steps/5)

        # 2) 缓冲长度：只需容纳最大延迟帧 + 小余量
        self.buffer_len = max(4, self.max_delay_frames + 2)

        # 每个 env/雷达的扫描帧号（本雷达本次扫描 +1）
        self.scan_frame_id = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )

        # 缓冲内容与指针
        self.buf_grids = torch.zeros(
            (self.num_envs, 2, self.buffer_len, *self.shape),
            dtype=torch.bool, device=self.device
        )
        self.buf_frame_ids = torch.full(
            (self.num_envs, 2, self.buffer_len),
            fill_value=-1, dtype=torch.int32, device=self.device
        )
        self.buf_head = torch.full(
            (self.num_envs, 2), fill_value=-1, dtype=torch.int32, device=self.device
        )
        self.buf_fill_count = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )

        # 每个 env 的观测延迟（单位：帧），前/后雷达各一列 → [E, 2]
        self.obs_delay_frames = torch.zeros(
            self.num_envs, 2, dtype=torch.int32, device=self.device
        )

        # 调试标志：本步是否发布
        self.should_publish_obs = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.saved_array = []
        self.count = 0

    # -------------------------------
    # reset：重置噪声、缓冲与“独立延迟”（步→帧）
    # -------------------------------
    def reset(self, env_ids: torch.Tensor):
        pos_sigma = 0.005
        ang_sigma = 1.0 * torch.pi / 180.0
        self.pos_noise[env_ids] = torch.randn(len(env_ids), 1, 3, device=self.device) * pos_sigma
        ang_noise = torch.randn(len(env_ids), 1, 3, device=self.device) * ang_sigma
        self.rot_noise[env_ids] = quat_from_euler_xyz(ang_noise)

        self.grid[env_ids] = False
        self.grid_obs[env_ids] = False
        self.left_grid[env_ids] = False
        self.right_grid[env_ids] = False

        low_steps, high_steps = self.obs_delay_range
        low_steps = int(low_steps)
        high_steps = int(high_steps)
        delay_steps_front = torch.randint(
            low=low_steps, high=high_steps + 1, size=(len(env_ids),),
            device=self.device, dtype=torch.int32
        )
        delay_steps_back = torch.randint(
            low=low_steps, high=high_steps + 1, size=(len(env_ids),),
            device=self.device, dtype=torch.int32
        )
        delay_frames_front = (delay_steps_front + (self.scan_period_steps - 1)) // self.scan_period_steps
        delay_frames_back  = (delay_steps_back  + (self.scan_period_steps - 1)) // self.scan_period_steps

        self.obs_delay_frames[env_ids, 0] = delay_frames_front.clamp_min(0)  # front
        self.obs_delay_frames[env_ids, 1] = delay_frames_back.clamp_min(0)   # back

        self.buf_grids[env_ids] = False
        self.buf_frame_ids[env_ids] = -1
        self.buf_head[env_ids] = -1
        self.buf_fill_count[env_ids] = 0
        self.scan_frame_id[env_ids] = 0

        self.scan_time_offset[env_ids, 1] = torch.randint(-5, 5, (len(env_ids),), device=self.device)
        self.force_full_scan[env_ids] = True
    # -------------------------------
    # 环形缓冲：写入（仅本步扫描的 env）
    # -------------------------------
    def _enqueue_lidar_frames(self, env_mask: torch.Tensor, lidar_idx: int, grids: torch.Tensor):
        if not env_mask.any():
            return
        idxs = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)

        new_heads = (self.buf_head[idxs, lidar_idx] + 1) % self.buffer_len
        self.buf_head[idxs, lidar_idx] = new_heads

        self.buf_grids[idxs, lidar_idx, new_heads] = grids[idxs]
        self.buf_frame_ids[idxs, lidar_idx, new_heads] = self.scan_frame_id[idxs, lidar_idx]

        self.buf_fill_count[idxs, lidar_idx] = torch.clamp(
            self.buf_fill_count[idxs, lidar_idx] + 1, max=self.buffer_len
        )

    # -------------------------------
    # 环形缓冲：回读“该雷达”的延迟帧
    # -------------------------------
    def _read_delayed_grid(self, lidar_idx: int) -> torch.Tensor:
        """
        返回 [E, Dx, Dy, Dz]，每个 env 从该 lidar 的环形缓冲中回读
        obs_delay_frames[:, lidar_idx] 帧之前的数据。
        若缓冲尚未累积足够深，则回退至“最老可用”的帧；若仍无数据，则返回全 False。
        """
        E = self.num_envs
        out = torch.zeros((E, *self.shape), dtype=torch.bool, device=self.device)

        heads = self.buf_head[:, lidar_idx]       # [-1 或 0..B-1]
        fills = self.buf_fill_count[:, lidar_idx] # [0..B]
        delay = torch.clamp(self.obs_delay_frames[:, lidar_idx], min=0, max=self.buffer_len - 1)

        has_any = fills > 0
        if not has_any.any():
            return out

        idxs = torch.nonzero(has_any, as_tuple=False).squeeze(-1)
        want = (heads[idxs] - delay[idxs]) % self.buffer_len

        too_deep = delay[idxs] >= fills[idxs]
        if too_deep.any():
            td_idx = idxs[too_deep]
            oldest = (heads[td_idx] - (fills[td_idx] - 1)) % self.buffer_len
            want[too_deep] = oldest

        out[idxs] = self.buf_grids[idxs, lidar_idx, want]
        return out

    # -------------------------------
    # 每步更新：扫描→入队；到节拍→回读并发布
    # -------------------------------
    def update(self):
        # 躯干世界位姿
        self.torso_quat_w = self.asset.data.body_quat_w[:, self.body_id]  # [E,4]
        self.torso_pos_w  = self.asset.data.body_pos_w[:, self.body_id]   # [E,3]

        # 噪声后位姿
        noise_torso_pos  = self.torso_pos_w + self.pos_noise  # [E,3]
        noise_torso_quat = quat_mul(self.rot_noise, self.torso_quat_w)  # [E,4]

        # 两雷达世界坐标
        lidar_pos_w = noise_torso_pos + quat_rotate(
            noise_torso_quat, 
            self.lidar_pos.unsqueeze(0)
        )  # [E,2,3]

        # 射线世界方向与起点
        ray_dirs_w = quat_rotate(
            self.torso_quat_w.reshape(self.num_envs, 1, 1, 4),
            self.lidar_ray_dirs.reshape(1, 2, -1, 3),
        )  # [E,2,R/2,3]
        ray_starts_w = lidar_pos_w.unsqueeze(2).expand_as(ray_dirs_w)  # [E,2,R/2,3]

        # 参与碰撞的网格（是否包含自身）
        if self.include_self:
            mesh_pos_w = torch.cat(
                [self.asset.data.body_pos_w, torch.zeros(self.num_envs, 1, 3, device=self.device)], dim=1
            )
            mesh_quat_w = torch.cat(
                [self.asset.data.body_quat_w,
                 torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 1, 4)],
                dim=1,
            )
        else:
            mesh_pos_w = torch.zeros(self.num_envs, 1, 3, device=self.device)
            mesh_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 1, 4)

        # 本步需扫描的 env/lidar（考虑相位）
        step = self.env.episode_length_buf  # [E]
        should_update_scan = ((step.unsqueeze(1) + self.scan_time_offset) % self.scan_period_steps) == 0  # [E,2]

        # Raycast（仅 enabled 生效；需确保 API 对 disabled 也返回占位以便 reshape）
        if self.force_full_scan.any():
            should_update_scan |= self.force_full_scan.unsqueeze(1)
        enabled = should_update_scan.flatten(0, 1)  # [E*2]
        hit_positions, hit_distances = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w.repeat_interleave(2, dim=0),
            mesh_quat_w=mesh_quat_w.repeat_interleave(2, dim=0),
            ray_starts_w=ray_starts_w.flatten(0, 1),
            ray_dirs_w=ray_dirs_w.flatten(0, 1),
            enabled=enabled, max_dist=100.0,
        )

        # 命中点回到躯干系并拆分前/后
        hit_positions = hit_positions.reshape(self.num_envs, -1, 3)  # [E,R,3]
        hit_positions_b = quat_rotate_inverse(
            self.torso_quat_w.reshape(self.num_envs, 1, 4),
            hit_positions - self.torso_pos_w.reshape(self.num_envs, 1, 3),
        ).reshape(self.num_envs, 2, -1, 3)  # [E,2,R/2,3]

        front_hit_b = hit_positions_b[:, 0]
        back_hit_b  = hit_positions_b[:, 1]

        front_hit_b += torch.randn_like(front_hit_b) * self.noise_std
        back_hit_b  += torch.randn_like(back_hit_b)  * self.noise_std

        # 体素化
        hardware_offset = torch.tensor([0.0, 0.0, 0.2], device=self.device)
        front_grid = voxelize_wp(self.shape, self.resolution, front_hit_b + hardware_offset)  # [E,Dx,Dy,Dz]
        back_grid  = voxelize_wp(self.shape, self.resolution, back_hit_b + hardware_offset)   # [E,Dx,Dy,Dz]
        
        # 随机“空洞”丢失（可选）
        if self.hole_prob > 0.0:
            hole_mask_front = (torch.rand(front_grid.shape, device=self.device) < self.hole_prob)
            hole_mask_back  = (torch.rand(back_grid.shape,  device=self.device)  < self.hole_prob)
            front_grid = torch.where(hole_mask_front, torch.zeros(1, dtype=torch.bool, device=self.device), front_grid)
            back_grid  = torch.where(hole_mask_back,  torch.zeros(1, dtype=torch.bool, device=self.device),  back_grid)

        # 仅对本步扫描的 env/lidar 入队，并推进帧号
        front_mask = should_update_scan[:, 0]
        back_mask  = should_update_scan[:, 1]
        self.scan_frame_id[front_mask, 0] += 1
        self.scan_frame_id[back_mask,  1] += 1

        self._enqueue_lidar_frames(front_mask, 0, front_grid)
        self._enqueue_lidar_frames(back_mask,  1, back_grid)

        # 到发布节拍（每 5 步）→ 从各自缓冲按“各自帧延迟”回读并发布
        should_publish_obs = (step % self.scan_period_steps) == 0  # [E]
        self.should_publish_obs = should_publish_obs
        if self.force_full_scan.any():
            should_publish_obs |= self.force_full_scan
            self.force_full_scan[self.force_full_scan] = False
        if should_publish_obs.any():
            delayed_front = self._read_delayed_grid(lidar_idx=0)  # front 的延迟
            delayed_back  = self._read_delayed_grid(lidar_idx=1)  # back  的延迟

            self.left_grid  = delayed_front
            self.right_grid = delayed_back
            self.grid       = self.left_grid | self.right_grid
            self.grid_obs   = self.grid

    # -------------------------------
    # 对外观测：einops 重排以适配下游
    # -------------------------------
    def compute(self) -> torch.Tensor:
        return einops.rearrange(self.grid_obs, self.pattern)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            hardware_offset = torch.tensor([0.0, 0.0, 0.2], device=self.device)
            pos = quat_rotate(
                self.torso_quat_w.reshape(self.num_envs, 1, 1, 1, 4),
                self.grid_centers.expand(self.num_envs, *self.shape, 3) - hardware_offset
            ) + self.torso_pos_w.reshape(self.num_envs, 1, 1, 1, 3)
            pos = pos[self.grid.bool()]
            self.marker.visualize(pos.reshape(-1, 3))
        elif self.env.backend == "mujoco":
            self.marker_front.geom.pos = self.lidar_pos_w[0, 0]
            self.marker_back.geom.pos = self.lidar_pos_w[0, 1]

    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0),
            signs=torch.ones(self.shape[1]),
        )