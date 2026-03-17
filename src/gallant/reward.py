import torch
import warp as wp

from active_adaptation.envs.mdp import Reward
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    quat_from_yaw,
    wrap_to_pi,
    normalize
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation

from isaaclab.utils.warp import raycast_mesh
from .command import LocoNavigation


terrain_dict= {
    "flat": 0,
    "ceil": 1,
    "tree": 2,
    "door": 3,
    "platform": 4,
    "pillar": 5,
    "slope_up": 6,
    "slope_down": 7
    # "tree": 1,
    # "door": 2,
    # "platform": 3,
    # "pillar": 4,
    # "slope_up": 5,
    # "slope_down": 6
}


class torques_scaled(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def compute(self) -> torch.Tensor:
        torques = self.asset.data.applied_torque / self.asset.data.joint_stiffness
        rew = torques.square().sum(1)
        return -rew.reshape(self.num_envs, 1)


class feet_stumble(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids, :]
        force_xy = torch.norm(contact_forces[:, :, :2], dim=-1)
        force_z = torch.abs(contact_forces[:, :, 2])
        rew = torch.any(force_xy > 3 * force_z, dim=1).float()
        # scale_env_ids = (self.command_manager.raw_terrain_types >= terrain_dict["platform"]).nonzero().squeeze()
        # rew[scale_env_ids] *= 5.0
        return -rew.reshape(self.num_envs, 1).float()

class angvel_xy_l2_torso(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_names: str = None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        body_names = "torso_link"
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
    def update(self):

        angvel = quat_rotate_inverse(
            self.asset.data.body_quat_w[:, self.body_ids],
            self.asset.data.body_ang_vel_w[:, self.body_ids]
        )
        self.angvel = angvel

    def compute(self) -> torch.Tensor:
        r = -self.angvel[:, :, :2].square().sum(-1).mean(1)
        return r.reshape(self.num_envs, 1)


class feet_contact_forces(Reward):
    def __init__(self, env, body_names, max_contact_force: float,weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.max_contact_force = max_contact_force
    
    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids, :].norm(dim=-1)
        rew = -torch.sum(contact_forces - self.max_contact_force, dim=1, keepdim=True).clamp_min(0.0)
        # scale_env_ids = ((self.command_manager.raw_terrain_types == terrain_dict["platform"]) | (self.command_manager.raw_terrain_types == terrain_dict["ceil"])).nonzero().squeeze()
        # rew[scale_env_ids] *= 0.01
        return -torch.sum(contact_forces - self.max_contact_force, dim=1, keepdim=True).clamp_min(0.0)

class feet_ground_parallel_bin(Reward): # how many points are on the ground when 
    def __init__(self, env, body_names, weight: float, enabled: bool = True, threshold: float = 0.035):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids_a, body_names_a = self.asset.find_bodies(body_names)
        self.body_ids_c, body_names_c = self.contact_forces.find_bodies(body_names)
        for name_a, name_c in zip(body_names_a, body_names_c):
            assert name_a == name_c
        self.threshold = threshold
        self.thres = self.env.step_dt * 3

    def compute(self):
        feet_fwd_vec = quat_rotate(
            self.asset.data.body_quat_w[:, self.body_ids_a],
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        feet_side_vec = quat_rotate(
            self.asset.data.body_quat_w[:, self.body_ids_a],
            torch.tensor([0., 1., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        toe_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] + feet_fwd_vec * 0.08
        heel_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] - feet_fwd_vec * 0.02
        in_contact = self.contact_forces.data.current_contact_time[:, self.body_ids_c] > 0.02
        left_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] - feet_side_vec * 0.025
        right_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] + feet_side_vec * 0.025
        toe_height = toe_pos_w[:, :, 2] - self.env.get_ground_height_at(toe_pos_w)
        heel_height = heel_pos_w[:, :, 2] - self.env.get_ground_height_at(heel_pos_w)
        left_height = left_pos_w[:, :, 2] - self.env.get_ground_height_at(left_pos_w)
        right_height = right_pos_w[:, :, 2] - self.env.get_ground_height_at(right_pos_w)
        ankle_height = self.asset.data.body_pos_w[:, self.body_ids_a][:, :, 2] - self.env.get_ground_height_at(self.asset.data.body_pos_w[:, self.body_ids_a])
        if torch.any(torch.isnan(toe_height)) or torch.any(torch.isnan(heel_height)) or torch.any(torch.isnan(ankle_height)):
            import ipdb; ipdb.set_trace()
       
        mask_toe = (toe_height <= self.threshold) & (toe_height >= 0)
        mask_heel = (heel_height <= self.threshold) & (heel_height >= 0)
        mask_ankle = (ankle_height <= self.threshold) & (ankle_height >= 0)
        mask_right = (right_height <= self.threshold) & (right_height >= 0)
        mask_left = (left_height <= self.threshold) & (left_height >= 0)
        rew = ((mask_toe + mask_ankle + mask_heel + mask_right + mask_left) * in_contact).sum(dim=-1).float()
        rew = torch.nan_to_num(rew, nan=0.0)
        # stair_env_id = (self.command_manager.raw_terrain_types >= terrain_dict["platform"]).nonzero().squeeze(1) 
        # rew[stair_env_id] *= 10.0

        return rew.reshape(self.num_envs, 1)

class penalty_fast_ang_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, threshold: float = 1.0):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.threshold = threshold

    def compute(self):
        ang_vel = torch.abs(self.asset.data.root_ang_vel_b[:, 2])
        return -(ang_vel - self.threshold).clamp_min(0.0).reshape(self.num_envs, 1)

# class feet_ground_parallel_bin(Reward):
#     def __init__(self, env, body_names, weight: float, enabled: bool = True):
#         super().__init__(env, weight, enabled)
#         self.asset: Articulation = self.env.scene.articulations["robot"]
#         self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]
#         self.body_ids_a, body_names_a = self.asset.find_bodies(body_names)
#         self.body_ids_c, body_names_c = self.contact_forces.find_bodies(body_names)
#         for name_a, name_c in zip(body_names_a, body_names_c):
#             assert name_a == name_c

#         self.thres = self.env.step_dt * 3

#         # --- build (num_envs, 60, 3) local grid -----------------------------
#         # x: 12 bins in [-0.06, 0.137], y: 5 bins in [-0.032, 0.032], z = -0.03
#         num_envs = self.env.num_envs
#         device = self.device

#         xs = torch.linspace(-0.06, 0.137, 12, device=device)
#         ys = torch.linspace(-0.032, 0.032, 5, device=device)
#         grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (5,12) each
#         xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (60,2)
#         z = torch.full((xy.shape[0], 1), -0.03, device=device)     # (60,1)
#         local_grid = torch.cat([xy, z], dim=-1)                    # (60,3)

#         # store with per-env shape as requested: (num_envs, 60, 3)
#         self.local_grid = local_grid.unsqueeze(0).expand(num_envs, -1, -1).contiguous()

#     def compute(self):
#         # feet world poses
#         feet_quat = self.asset.data.body_quat_w[:, self.body_ids_a]   # (E,2,4)
#         feet_pos  = self.asset.data.body_pos_w[:,  self.body_ids_a]   # (E,2,3)
#         E = feet_pos.shape[0]
#         device = self.device

#         in_contact = self.contact_forces.data.current_contact_time[:, self.body_ids_c] > 0.02

#         rew = torch.zeros(E, device=device)

#         if not torch.any(in_contact):
#             return rew.reshape(E, 1)

#         active_pairs = in_contact.nonzero(as_tuple=False)  # (K, 2): [env_id, foot_idx]
#         env_ids_active = active_pairs[:, 0]
#         foot_ids_active = active_pairs[:, 1]
#         feet_quat_active = feet_quat[env_ids_active, foot_ids_active]  # (K,4)
#         feet_pos_active  = feet_pos[env_ids_active,  foot_ids_active]  # (K,3)

#         local_grid_active = self.local_grid[env_ids_active]  # (K,60,3)

#         world_offsets = quat_rotate(feet_quat_active.unsqueeze(1), local_grid_active)  # (K,60,3)
#         points_w = feet_pos_active.unsqueeze(1) + world_offsets                        # (K,60,3)

#         ground_h = self.env.get_ground_height_at(points_w)  # (K,60)
#         dz = points_w[..., 2] - ground_h                    # (K,60)

#         mask = (dz >= 0.0) & (dz <= 0.04)                   # (K,60)
#         counts_per_pair = mask.sum(dim=1).float()           # (K,)

#         per_env_counts = torch.zeros(E, 2, device=device)   # (E,2)
#         per_env_counts[env_ids_active, foot_ids_active] = counts_per_pair

#         per_env_min = per_env_counts.min(dim=1).values      # (E,)

#         rew = per_env_min
#         # 无接触的 env 默认保持 0
#         pillar_mask = (self.command_manager.raw_terrain_types == terrain_dict["pillar"])
#         # pillar_env_id = pillar_mask.nonzero().squeeze(1) 
#         # rew[pillar_env_id] *= 10.0

#         feet_z = feet_pos[..., 2]                    # (E, 2)
#         any_foot_below0 = (feet_z < 0.0).any(dim=1) # (E,)
#         try:
#             violate_mask = pillar_mask & any_foot_below0 # (E,)
#         except:
#             import ipdb; ipdb.set_trace()
#         rew[violate_mask] = 0.0

#         return rew.reshape(E, 1) / 60.0

class contact_momentum(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.sensor_ids, self.sensor_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = self.asset.find_bodies(body_names)[0]

    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_ids, 2]
        feet_vel = self.asset.data.body_vel_w[:, self.body_ids, 2]
        contact_momentum = torch.clip(feet_vel, max=0.0) * torch.clip(contact_forces-50.0, min=0.0)
        return contact_momentum.sum(1, keepdim=True)

class penalize_contact(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.sensor_ids, self.sensor_names = self.contact_sensor.find_bodies(body_names)

    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_ids, :]
        contact_forces = contact_forces.norm(dim=-1)
        # tree_env_id = (self.command_manager.raw_terrain_types == terrain_dict["tree"]).nonzero().squeeze()
        # contact_forces[tree_env_id] *= 5.0
        return -contact_forces.sum(1, keepdim=True)

class action_vanish(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.action_min = (self.asset.data.joint_pos_limits[:, :, 0] - self.asset.data.default_joint_pos) / self.env.action_manager.action_scaling
        self.action_max = (self.asset.data.joint_pos_limits[:, :, 1] - self.asset.data.default_joint_pos) / self.env.action_manager.action_scaling

    def compute(self) -> torch.Tensor:
        action = self.asset.data.applied_action[:, :, 0]
        upper_error = torch.clip(action - self.action_max, min=0.0)
        lower_error = torch.clip(-action + self.action_min, min=0.0)
        return -torch.sum(upper_error + lower_error, dim=1, keepdim=True)


class dof_vel(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
    
    def compute(self) -> torch.Tensor:
        dof_vel = self.asset.data.joint_vel
        # env_ids = self.command_manager.target_reached.nonzero().squeeze(1)
        # dof_vel[env_ids] *= 100.0
        return -dof_vel.square().sum(1, True)


class dof_vel_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.dof_vel_limits = self.asset.data.joint_velocity_limits[:, :] * 0.80
        
    def compute(self) -> torch.Tensor:
        dof_vel = self.asset.data.joint_vel
        return -((dof_vel - self.dof_vel_limits).clamp_min(0.0)).sum(1, True)


class torque_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.torque_limits = self.asset.data.joint_effort_limits[:, :] * 0.95

    def compute(self) -> torch.Tensor:
        torque = self.asset.data.applied_torque
        return -((torque - self.torque_limits).clamp_min(0.0)).sum(1, True)


class dof_pos_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.soft_joint_pos_limits = self.asset.data.soft_joint_pos_limits
    
    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos
        violation_min = (joint_pos - self.soft_joint_pos_limits[:, :, 0]).clamp_max(0.0)
        violation_max = (self.soft_joint_pos_limits[:, :, 1] - joint_pos).clamp_max(0.0)
        return (violation_min + violation_max).sum(1, keepdim=True)


class feet_distance_lateral(Reward):
    def __init__(self, env, body_names, least_distance: float, most_distance: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.least_distance = least_distance    
        self.most_distance = most_distance
    
    def compute(self):
        cur_feetpos_translated = self.asset.data.body_link_pos_w[:, self.body_ids, :] - self.asset.data.root_pos_w.unsqueeze(1)
        feetpos_in_body_frame = torch.zeros(self.env.num_envs, 2, 3, device=self.env.device)
        for i in range(2):
            feetpos_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetpos_translated[:, i, :])
        foot_lateral_dis = torch.abs(feetpos_in_body_frame[:, 0, 1] - feetpos_in_body_frame[:, 1, 1])
        rew = (torch.clamp(foot_lateral_dis - self.least_distance, max=0) + torch.clamp(-foot_lateral_dis + self.most_distance, max=0)).reshape(self.env.num_envs, -1)
        # pillar_id = ((self.command_manager.raw_terrain_types == terrain_dict["pillar"])).nonzero().squeeze()
        # rew[pillar_id] *= 0.01
        return rew

class knee_distance_lateral(Reward):
    def __init__(self, env, body_names, least_distance: float, most_distance: float, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.least_distance = least_distance    
        self.most_distance = most_distance
    
    def compute(self):
        cur_feetpos_translated = self.asset.data.body_link_pos_w[:, self.body_ids, :] - self.asset.data.root_pos_w.unsqueeze(1)
        kneepos_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            kneepos_in_body_frame[:, i, :] = quat_rotate_inverse(yaw_quat(self.asset.data.root_link_quat_w), cur_feetpos_translated[:, i, :])
        foot_lateral_dis = torch.abs(kneepos_in_body_frame[:, 0, 1] - kneepos_in_body_frame[:, 1, 1]) + torch.abs(kneepos_in_body_frame[:, 2, 1] - kneepos_in_body_frame[:, 3, 1])
        
        rew = (torch.clamp(foot_lateral_dis - 2 * self.least_distance, max=0) + torch.clamp(-foot_lateral_dis + 2 * self.most_distance, max=0)).reshape(self.env.num_envs, -1)
        # pillar_id = ((self.command_manager.raw_terrain_types == terrain_dict["pillar"])).nonzero().squeeze()
        # rew[pillar_id] *= 0.01
        return rew

class feet_ground_parallel(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids_a, body_names_a = self.asset.find_bodies(body_names)
        self.body_ids_c, body_names_c = self.contact_forces.find_bodies(body_names)
        for name_a, name_c in zip(body_names_a, body_names_c):
            assert name_a == name_c

        self.thres = self.env.step_dt * 3

    def compute(self):
        feet_fwd_vec = quat_rotate(
            self.asset.data.body_quat_w[:, self.body_ids_a],
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        toe_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] + feet_fwd_vec * 0.12#0.1
        heel_pos_w = self.asset.data.body_pos_w[:, self.body_ids_a] - feet_fwd_vec * 0.06#0.02
        first_contact = self.contact_forces.compute_first_air(self.thres)[:, self.body_ids_c]
        toe_height = toe_pos_w[:, :, 2] #- self.env.get_ground_height_at(toe_pos_w)
        heel_height = heel_pos_w[:, :, 2] #- self.env.get_ground_height_at(heel_pos_w)
        if torch.any(torch.isnan(toe_height)) or torch.any(torch.isnan(heel_height)):
            import ipdb; ipdb.set_trace()
        # print(toe_height - heel_height)
        rew = torch.sum((toe_height - heel_height).square() * first_contact, dim=1)
        ground_toe_height = toe_pos_w[:, :, 2] - self.env.get_ground_height_at(toe_pos_w)
        ground_heel_height = heel_pos_w[:, :, 2] - self.env.get_ground_height_at(heel_pos_w)
        self.command_manager.toe_heel_height = ground_toe_height - ground_heel_height
        rew += torch.sum((ground_toe_height - ground_heel_height).square() * first_contact, dim=1)
        rew = torch.nan_to_num(rew, nan=0.0)
        # stair_env_id = (self.command_manager.raw_terrain_types >= terrain_dict["platform"]).nonzero().squeeze(1) 
        # rew[stair_env_id] *= 10.0
        # ceil_env_id = (self.command_manager.raw_terrain_types == terrain_dict["ceil"]).nonzero().squeeze(1)
        # rew[ceil_env_id] *= 0.2
        return -rew.reshape(self.num_envs, 1)


class feet_parallel(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        assert len(self.body_ids) == 2

    def compute(self):
        self.feet_fwd_vec = quat_rotate(
            yaw_quat(self.asset.data.body_quat_w[:, self.body_ids]),
            torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 2, 3)
        )
        dot = torch.sum(self.feet_fwd_vec[:, 0] * self.feet_fwd_vec[:, 1], dim=1, keepdim=True)
        return dot - 1.
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_pos_w[:, self.body_ids].reshape(-1, 3),
            self.feet_fwd_vec.reshape(-1, 3),
            color=(0, 0, 1, 1),
        )


class feet_clearance_simple(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.target_height = 0.08
    
    def compute(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        feet_vel_w = self.asset.data.body_vel_w[:, self.body_ids]
        feet_speed = torch.norm(feet_vel_w[:, :, :2], dim=2).square()
        feet_height = feet_pos_w[:, :, 2] - self.env.get_ground_height_at(feet_pos_w)
        error = (feet_height - self.target_height).clamp_max(0.0)
        return (feet_speed * error).sum(dim=1).reshape(self.num_envs, 1)


class feet_ground_slip(Reward):
    def __init__(
        self, env, body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

    def compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_ids] >= 0.02
        )
        feet_vel = self.asset.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        # stair_env_id = (self.command_manager.raw_terrain_types >= terrain_dict["platform"]).nonzero().squeeze(1)
        # slip[stair_env_id] *= 10.0
        return -slip

class waist_deviation_l2(Reward):
    def __init__(self, env, joint_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def compute(self):
        dev = self.asset.data.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return -dev.square().sum(1, True)


class orientation_torso(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
    
    def compute(self) -> torch.Tensor:
        torso_quat = self.asset.data.body_quat_w[:, self.body_ids].reshape(self.num_envs, 4)
        projected_gravity = quat_rotate_inverse(
            torso_quat,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1))
        env_ids = (projected_gravity[:, 0] < 0.) #& (self.command_manager.raw_terrain_types == terrain_dict["ceil"])
        projected_gravity[env_ids, 0] *= 10.0
        projected_gravity[:, 1] *= 5.0
        if "ceil" in terrain_dict:
            ceil_env_id = ((self.command_manager.raw_terrain_types == terrain_dict["ceil"]) & (projected_gravity[:, 0] >= 0.)).nonzero().squeeze(1)
            projected_gravity[ceil_env_id] *= 0.2
        return -torch.sum(projected_gravity[:, :2].square(), dim=1, keepdim=True)

class orientation_pelvis(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]

    def compute(self) -> torch.Tensor:      
        pelvis_quat = self.asset.data.body_quat_w[:, self.body_ids].reshape(self.num_envs, 4)
        projected_gravity = quat_rotate_inverse(
            pelvis_quat,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1))
        env_ids = (projected_gravity[:, 0] < 0.) #& (self.command_manager.raw_terrain_types == terrain_dict["ceil"])
        projected_gravity[env_ids, 0] *= 10.0
        projected_gravity[:, 1] *= 5.0
        return -torch.sum(projected_gravity[:, :2].square(), dim=1, keepdim=True)


class linvel_x_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.command_manager = self.env.command_manager
    
    def compute(self):
        linvel_x = self.asset.data.root_lin_vel_b[:, 0]
        error = torch.square(self.command_manager.command_linvel[:, 0] - linvel_x)
        return torch.exp(-error / 0.25).reshape(self.num_envs, 1)


class linvel_y_exp(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.command_manager = self.env.command_manager
    
    def compute(self):
        linvel_y = self.asset.data.root_lin_vel_b[:, 1]
        error = torch.square(self.command_manager.command_linvel[:, 1] - linvel_y)
        return torch.exp(-error / 0.25).reshape(self.num_envs, 1)


class velocity_direction(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        g = self.command_manager.pos_diff / (1e-8 + self.command_manager.pos_diff_norm)
        g[:, 2] = 0.0
        g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)

        avoid = self.command_manager.compute_direction(
            self.asset.data.root_pos_w, goal_dir_xy=g,
            radius=1.0, beta=2.0, kappa=0.8, inflation=0.2, ema=0.6
        ) # TODO sweep radius here

        avoid_dir = avoid / (avoid.norm(dim=-1, keepdim=True) + 1e-8)
        lam = 2.0
        self.command_manager.target_direction = g + lam * avoid_dir
        self.command_manager.target_direction[:, 2] = 0.0
        self.command_manager.target_direction = self.command_manager.target_direction / (self.command_manager.target_direction.norm(dim=-1, keepdim=True) + 1e-8)
        speed_xy = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        rew = torch.sum(self.command_manager.target_direction[:, :2] * self.asset.data.root_lin_vel_w[:, :2], dim=-1, keepdim=True).div(speed_xy)
        rew = torch.max(rew.reshape(self.num_envs, 1), (1. * self.command_manager.target_reached.reshape(self.num_envs, 1)).reshape(self.num_envs, 1) )
        # tree_env_id = (self.command_manager.raw_terrain_types == terrain_dict["tree"]).nonzero().squeeze(1)
        # rew[tree_env_id] *= 20.0
        # pillar_env_id = (self.command_manager.raw_terrain_types == terrain_dict["pillar"]).nonzero().squeeze(1)
        # rew[pillar_env_id] *= 10.0
        # rew = torch.where(rew<0., rew * 10.0, rew)
        # rew *= self.command_manager.target_pos_scale
        return rew


class velocity_projection(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        target_direction = normalize(self.command_manager.target_direction[:, :2])
        velocity = self.asset.data.root_lin_vel_w[:, :2]
        rew = torch.sum(velocity * target_direction, dim=-1, keepdim=True).clamp_max(1.0)
        return rew.reshape(self.num_envs, 1)


class reach_target_pos(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        Tr = 2
        diff = (torch.abs(self.command_manager.target_pos_w[:, :2] - self.asset.data.root_pos_w[:, :2]) - 0.2).clip(min=0.)
        rew = 1 / ((1 + diff.square().sum(dim=-1, keepdim=True)))
        rew = torch.where(
            self.command_manager.time_elapsed > self.command_manager.time_alloted - Tr,
            rew,
            torch.zeros_like(self.command_manager.time_alloted)
        )
        return rew.reshape(self.num_envs, 1)


class feet_clearance_height(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.target_height = 0.05
    
    def update(self):
        self.feet_pos_w = self.asset.data.body_pos_w[:, self.command_manager.foot_ids, :]
        self.feet_vel_w = self.asset.data.body_vel_w[:, self.command_manager.foot_ids, :3]

    def compute(self) -> torch.Tensor:
        target_vec = (self.command_manager.target_pos_w[:, :2] - self.asset.data.root_pos_w[:, :2])
        quat = yaw_quat(self.asset.data.root_link_quat_w)
        feetvel_in_body_frame = quat_rotate_inverse(
            quat.unsqueeze(1).expand(self.num_envs, 2, 4),
            self.feet_vel_w - self.asset.data.root_lin_vel_w.unsqueeze(1)
        )       
        feet_height_w = self.feet_pos_w[:, :, 2]
        test_pos = self.feet_pos_w.clone()
        platform_env_id = (self.command_manager.raw_terrain_types == terrain_dict["platform"]).nonzero().squeeze(1)

        test_pos[:, :, :2] += 0.15 * normalize(target_vec).unsqueeze(1) # used 0.2
        test_pos[platform_env_id, :, :2] += 0.05 * normalize(target_vec[platform_env_id]).unsqueeze(1)
        self.command_manager.feet_ground_height = feet_height_w - (self.env.get_ground_height_at(self.feet_pos_w) + self.env.get_ground_height_at(test_pos))/2
        height_error = torch.square((self.command_manager.feet_ground_height - self.target_height).clamp(max=0.))
        self.height_error = height_error
        feet_lateral_vel = feetvel_in_body_frame[:, :, :2].norm(dim=-1)
        rew = -torch.sum(height_error * feet_lateral_vel, dim=1, keepdim=True)
        # height_env_id = (self.command_manager.raw_terrain_types >= terrain_dict["platform"]).nonzero().squeeze(1)
        # rew[platform_env_id] *= 10.0
        return rew
    
class linvel_z_l2_new(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        linvel_z = self.asset.data.root_lin_vel_w[:, 2]
        return -linvel_z.square().reshape(self.num_envs, 1)


class reach_target_yaw(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        target_vec = self.command_manager.target_pos_w[:, :2] - self.asset.data.root_pos_w[:, :2]
        # target_vec = self.command_manager.target_direction[:, :2]
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        no_reach_diff = wrap_to_pi(target_yaw - self.asset.data.heading_w) / torch.pi
        no_reach_rew = -no_reach_diff.square().reshape(self.num_envs, 1)
        pelvis_quat = self.asset.data.body_quat_w[:, self.command_manager.body_pelvis_id].reshape(self.num_envs, 4)
        forward_vec = quat_rotate(pelvis_quat, self.command_manager.forward_vec)
        pelvis_heading_w = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        no_reach_rew += -(wrap_to_pi(target_yaw - pelvis_heading_w) / torch.pi).square().reshape(self.num_envs, 1)
        rew = no_reach_rew * (~self.command_manager.target_reached).float().reshape(self.num_envs, 1)

        # tree_env_id = (self.command_manager.raw_terrain_types == terrain_dict["tree"]).nonzero().squeeze()
        # rew[tree_env_id] *= 10.0
        return rew.reshape(self.num_envs, 1)


class num_foot_contact(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]

    def compute(self) -> torch.Tensor:
        force_z = self.contact_forces.data.net_forces_w[:, self.command_manager.contact_foot_ids, 2]
        contacts = torch.sum(force_z < 0.1, dim=-1).reshape(self.num_envs, 1)# num of no contact
        single_contacts = contacts == 1
        double_contacts = contacts == 0
        return single_contacts * (1. * ~self.command_manager.target_reached.reshape(self.num_envs, 1)) + double_contacts * (1. * self.command_manager.target_reached.reshape(self.num_envs, 1))
        # return (-contacts + 2.0) * (1. * self.command_manager.target_reached.reshape(self.num_envs, 1))

class pillar_middle(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        world_pos = self.asset.data.root_pos_w[:, :2]
        target_pos = self.command_manager.target_pos_w[:, :2]
        pos_diff = world_pos - target_pos
        x_diff, y_diff = pos_diff[:, 0], pos_diff[:, 1]
        using_diff = torch.min(torch.square(x_diff), torch.square(y_diff))
        pillar_env_id = (self.command_manager.raw_terrain_types == terrain_dict["pillar"]).nonzero().squeeze()
        rew = torch.exp(-using_diff * 4)
        rew[pillar_env_id] = 0.0
        return rew.reshape(self.num_envs, 1)

class stall_penalty(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, min_vel: float, max_vel: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.min_vel = min_vel
        self.max_vel = max_vel

    def compute(self) -> torch.Tensor:
        # vel_min = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True) <= 0.4
        vel_min = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True) <= self.min_vel
        vel_max = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True) >= self.max_vel
        # tree_env_id = (self.command_manager.raw_terrain_types == terrain_dict["tree"]).nonzero().squeeze()
        rew = -(vel_min+vel_max).float() * (1. * ~self.command_manager.target_reached.reshape(self.num_envs, 1))
        # rew[tree_env_id] *= 0.1
        return rew.reshape(self.num_envs, 1)


class single_foot_contact_hussar(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]

    def compute(self) -> torch.Tensor:
        force_z = self.contact_forces.data.net_forces_w[:, self.command_manager.contact_foot_ids, 2]
        in_contact = force_z > 0.5 # num of contact
        single_contact = torch.sum(1.*in_contact, dim=1) == 1
        single_contact = torch.max(self.command_manager.target_reached.reshape(self.num_envs, 1) * 1., single_contact.reshape(self.num_envs, 1)) - 1.
        # ceil_env_id = (self.command_manager.raw_terrain_types == terrain_dict["ceil"]).nonzero().squeeze(1)
        # single_contact[ceil_env_id] *= 0.2
        return single_contact


class reaching_target(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.award_time = 2.0

    def compute(self) -> torch.Tensor:
        rew_condition = self.command_manager.time_alloted - self.command_manager.time_elapsed < self.award_time
        return (self.command_manager.target_reached * rew_condition).reshape(self.num_envs, 1)


class tracking_head_height(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        est_height = self.command_manager.est_height
        head_height = self.command_manager.head_height
        error = (head_height - est_height).square()
        rew = torch.exp(-error * 4).reshape(self.num_envs, 1)
        # rew = -100 * error.square().reshape(self.num_envs, 1)
        # compute heuristic knee reward here
        height_error = head_height - est_height
        
        knee_deviation = (self.asset.data.joint_pos[:, self.command_manager.knee_joint_ids] - self.command_manager.knee_action_min) / (self.command_manager.knee_action_max - self.command_manager.knee_action_min)
        knee_heuristic_rew = -torch.sum(torch.abs((knee_deviation - 0.5) * height_error), dim=-1).reshape(self.num_envs, 1)
        rew += 5*knee_heuristic_rew

        hip_pitch_deviation = (-self.asset.data.joint_pos[:, self.command_manager.hip_pitch_joint_ids] + self.command_manager.hip_pitch_action_max) / (self.command_manager.hip_pitch_action_max - self.command_manager.hip_pitch_action_min)
        hip_pitch_heuristic_rew = -torch.sum(torch.abs((hip_pitch_deviation - 0.5) * height_error), dim=-1).reshape(self.num_envs, 1)
        rew += 5*hip_pitch_heuristic_rew

        # ceil_env_id = (self.command_manager.raw_terrain_types == terrain_dict["ceil"]).nonzero().squeeze(1)
        # rew[ceil_env_id] *= 10.0
        return rew.reshape(self.num_envs, 1)
    
    def get_required_head_height_at(self, pos: torch.Tensor, offset: float) -> torch.Tensor:
        if self.env.backend == "isaac":
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
        elif self.env.backend == "mujoco":
            # mujoco does not have a ground mesh, so we assume the ground is at z=0
            return pos[..., 2]  # to be implemented


class feet_air_time_hussar(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_forces.compute_first_contact(self.env.step_dt)[
            :, self.command_manager.contact_foot_ids
        ]
        last_air_time = self.contact_forces.data.last_air_time[:, self.command_manager.contact_foot_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1).reshape(self.num_envs, 1) * (1. * ~self.command_manager.target_reached.reshape(self.num_envs, 1))
        return air_time.reshape(self.num_envs, 1)


class no_reach(Reward[LocoNavigation]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        interval_reached = (self.env.episode_length_buf+1) % self.command_manager.resample_interval == 0
        penalize = interval_reached & ~self.command_manager.target_reached
        return -100.0 * penalize.float().reshape(self.num_envs, 1)

# class solid_step(Reward[LocoNavigation]):
#     left_foot_mesh_path = f"{PATH}/g1_29dof/meshes/left_ankle_roll_link.STL"
#     right_foot_mesh_path = f"{PATH}/g1_29dof/meshes/right_ankle_roll_link.STL"

#     def __init__(self, env, weight: float, thres: float = 0.01, enabled: bool = True):
#         super().__init__(env, weight, enabled)
#         self.thres = thres
#         self.asset = self.command_manager.asset
        
#         left_foot_mesh = trimesh.load(self.left_foot_mesh_path)
#         right_foot_mesh = trimesh.load(self.right_foot_mesh_path)

#         self.foot_bottom_vertices = torch.stack([
#             torch.from_numpy(self.get_bottom_vertices(left_foot_mesh)),
#             torch.from_numpy(self.get_bottom_vertices(right_foot_mesh))
#         ]).to(self.device) # [2, *, 3]

#         self.ground_mesh = self.env.ground_mesh
#         self.foot_ids = self.asset.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])[0]
#         self.foot_ids = torch.tensor(self.foot_ids, device=self.device)
#         self.kernel_dim = self.num_envs * self.foot_bottom_vertices.shape[:2].numel()
#         self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]
#         self.body_ids_c = self.contact_forces.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])[0]

#     def compute(self) -> torch.Tensor:
#         feet_pos_w = self.asset.data.body_pos_w[:, self.foot_ids].reshape(self.num_envs, 2, 1, 3)
#         feet_quat_w = self.asset.data.body_quat_w[:, self.foot_ids].reshape(self.num_envs, 2, 1, 4)
#         feet_bottom_vertices = (
#             feet_pos_w
#             + quat_rotate(feet_quat_w, self.foot_bottom_vertices.reshape(1, 2, -1, 3))
#         )
#         distances = torch.zeros(self.num_envs, 2, self.foot_bottom_vertices.shape[1], device=self.device)
#         wp.launch(
#             mesh_query_point,
#             dim=[self.kernel_dim],
#             inputs=[
#                 self.ground_mesh.id,
#                 feet_bottom_vertices.reshape(self.kernel_dim, 3),
#                 wp.from_torch(distances.reshape(self.kernel_dim), dtype=wp.float32, return_ctype=True)
#             ],
#             device=wp.get_device(str(self.device))
#         )
#         solid_contact = (distances < self.thres).sum(dim=-1) # [num_envs, 2]
#         in_contact = self.contact_forces.data.current_contact_time[:, self.body_ids_c] > 0.02
#         solid_contact = solid_contact * in_contact.float()
#         solid_number = torch.min(solid_contact, dim=1, keepdim=True)[0]
#         rew = solid_number.float() / self.foot_bottom_vertices.shape[1]
#         return rew.reshape(self.num_envs, 1)
    
#     def get_bottom_vertices(self, mesh: trimesh.Trimesh) -> torch.Tensor:
#         z_min = np.min(mesh.vertices[:, 2])
#         vertices = mesh.vertices[mesh.vertices[:, 2] <= z_min+5e-4]
#         return vertices.astype(np.float32)


@wp.kernel
def mesh_query_point(
    mesh: wp.uint64,
    points_in: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    result = wp.mesh_query_point_no_sign(mesh, points_in[tid], 10.0)
    if result.result:
        points_out = wp.mesh_eval_position(mesh, result.face, result.u, result.v)
        distances[tid] = wp.length(points_in[tid] - points_out)
    else:
        distances[tid] = wp.INF

