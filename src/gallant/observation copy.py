import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING

import active_adaptation
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    quat_from_euler_xyz
)
import active_adaptation.utils.symmetry as sym_utils
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaacsim.core.utils.stage import get_current_stage
import warp as wp
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

class feet_ground_height(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return self.command_manager.feet_ground_height.reshape(self.num_envs, 2)
    
    def symmetry_transforms(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([1, 0]),
            signs=torch.tensor([1., 1.]),
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
        pattern: str="n 1 x z y",
        noise_std: float=0.0,
        include_self: bool=False,
        num_scan: int=128,
        random_offset: bool=False,
        resolution: Tuple[float, float, float]=(0.1, 0.1, 0.1),
        hole_prob: float=0.0,
        obs_delay_range: Tuple[int, int]=(5, 10),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_id = self.asset.find_bodies("torso_link")[0]

        self.resolution = tuple(resolution)
        if not len(self.resolution) == 3:
            raise ValueError("resolution must be a sequence of 3 elements")

        # shape = int(32 * 0.1 / self.resolution)
        self.shape = (32, 32, 32)
        self.pattern = f"n x y z -> {pattern}"
        self.noise_std = noise_std
        self.include_self = include_self
        self.num_scan = num_scan
        self.random_offset = random_offset
        self.hole_prob = hole_prob
        self.obs_delay_range = obs_delay_range

        with torch.device(self.device):
            self.grid_shape = torch.as_tensor(self.shape)
            self.grid_half_shape = self.grid_shape // 2
            self.grid_half_size = self.grid_half_shape * torch.tensor(self.resolution)

            self.grid_centers = torch.stack(torch.meshgrid(*map(torch.arange, self.grid_shape)), dim=-1)
            self.grid_centers = self.grid_centers * torch.tensor(self.resolution)
            self.grid_centers = self.grid_centers - self.grid_half_size + torch.tensor(self.resolution) / 2

            self.lidar_pos = torch.tensor([[0.12734, 0.00007, 0.17622], [-0.11284, -0.0004, 0.17493]], device=self.device)
            angle = (94/180.0) * torch.pi
            vangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
            hangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
            vgrid, hgrid = torch.meshgrid(vangles, hangles, indexing="ij")
            vsin, vcos = vgrid.sin(), vgrid.cos()
            hsin, hcos = hgrid.sin(), hgrid.cos()
            self.lidar_ray_dirs = torch.stack([
                torch.stack([vcos * hcos, vcos * hsin, vsin], dim=-1),  # front
                torch.stack([-vcos * hcos, -vcos * hsin, vsin], dim=-1) # back
            ], dim=0)  # [2, num_scan, num_scan, 3]
            del vangles, hangles, vgrid, hgrid, vsin, vcos, hsin, hcos

            self.pos_noise = torch.zeros(self.num_envs, 1, 3)
            self.rot_noise = torch.zeros(self.num_envs, 1, 4)
            self.rot_noise[:] = torch.tensor([1., 0., 0., 0.])
            self.obs_delay_steps = torch.zeros(self.num_envs, dtype=torch.int32)

        self.num_rays = (self.num_scan * self.num_scan) * 2
            
        if self.env.backend == "isaac":
            from isaaclab.markers import (
                VisualizationMarkers,
                VisualizationMarkersCfg,
                sim_utils
            )
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/grid_map",
                    markers={
                        "griddot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)
        elif self.env.backend == "mujoco":
            self.marker_front = self.env.scene.create_sphere_marker(radius=0.05, rgba=(1., 0., 0., 1.))
            self.marker_back = self.env.scene.create_sphere_marker(radius=0.05, rgba=(0., 1., 0., 1.))
        
        if self.env.backend == "isaac":
            paths = []
            if self.include_self:
                for body_name in self.asset.body_names:
                    paths.append(f"/World/envs/env_0/Robot/{body_name}/visuals")
            self.raycaster = MultiMeshRaycaster.from_prim_paths(
                paths,
                stage=get_current_stage(),
                device=wp.get_device(str(self.device)),
                simplify_factor=0.0
            )
            self.raycaster.add_mesh(self.env.ground_mesh)
        elif self.env.backend == "mujoco": # TODO@btx0424: test this
            body_names = self.asset.body_names
            self.raycaster = MultiMeshRaycaster.from_MjModel(
                body_names=body_names,
                model=self.asset.mj_model,
                device=wp.get_device(str(self.device)),
                simplify_factor=0.0
            )
            self.raycaster.add_mesh(self.env.ground_mesh)
        print(self.raycaster)

        # the two lidars are NOT synchronized, so we need to add a time offset to the second lidar
        self.scan_time_offset = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.int32)
        self.scan_time_offset[:, 1] = torch.randint(-5, 5, (self.num_envs,), device=self.device)

        self.grid = torch.zeros((self.num_envs, *self.shape), device=self.device, dtype=torch.bool)
        self.grid_obs = torch.zeros((self.num_envs, *self.shape), device=self.device, dtype=torch.bool)
        self.left_grid = torch.zeros((self.num_envs, *self.shape), device=self.device, dtype=torch.bool)
        self.right_grid = torch.zeros((self.num_envs, *self.shape), device=self.device, dtype=torch.bool)
        self.last_update_obs = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.force_full_scan = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.count = 0
        self.saved_array = []
        self.update()
    
    def reset(self, env_ids: torch.Tensor):
        pos_sigma = 0.01
        ang_sigma = 3 * torch.pi / 180.0
        self.pos_noise[env_ids] = torch.randn(len(env_ids), 1, 3, device=self.device) * pos_sigma
        ang_noise = torch.randn(len(env_ids), 1, 3, device=self.device) * ang_sigma
        self.rot_noise[env_ids] = quat_from_euler_xyz(*ang_noise.unbind(-1))

        self.grid[env_ids] = False
        self.grid_obs[env_ids] = False
        self.last_update_obs[env_ids] = 0
        self.force_full_scan[env_ids] = True
        self.obs_delay_steps[env_ids] = torch.randint(*self.obs_delay_range, (len(env_ids),), dtype=torch.int32, device=self.device)
    
    def update(self):
        self.torso_quat_w = self.asset.data.body_quat_w[:, self.body_id]
        self.torso_pos_w = self.asset.data.body_pos_w[:, self.body_id]

        noise_torso_pos = self.torso_pos_w + self.pos_noise 
        noise_torso_quat = quat_mul(self.rot_noise, self.torso_quat_w)
        
        self.lidar_pos_w = noise_torso_pos + quat_rotate(
            noise_torso_quat,
            self.lidar_pos.unsqueeze(0)
        ) # [num_envs, 2, 3]
        self.ray_dirs_w = quat_rotate(
            self.torso_quat_w.reshape(self.num_envs, 1, 1, 4),
            self.lidar_ray_dirs.reshape(1, 2, -1, 3)
        ) # [num_envs, 2, self.num_rays/2, 3]

        ray_starts_w = self.lidar_pos_w.unsqueeze(2).expand_as(self.ray_dirs_w) # [num_envs, 2, self.num_rays/2, 3]
        
        if self.include_self:
            mesh_pos_w = torch.cat([
                self.asset.data.body_pos_w,
                torch.zeros(self.num_envs, 1, 3, device=self.device),
            ], dim=1)
            mesh_quat_w = torch.cat([
                self.asset.data.body_quat_w,
                torch.tensor([1., 0., 0., 0.], device=self.device).expand(self.num_envs, 1, 4),
            ], dim=1)
        else:
            mesh_pos_w = torch.zeros(self.num_envs, 1, 3, device=self.device)
            mesh_quat_w = torch.tensor([1., 0., 0., 0.], device=self.device).expand(self.num_envs, 1, 4)
        
        should_update_scan = (self.env.episode_length_buf.unsqueeze(1)+self.scan_time_offset) % 5 == 0 # [num_envs, 2]
        should_update_obs = (self.env.episode_length_buf - self.last_update_obs) > self.obs_delay_steps
        if self.force_full_scan.any():
            should_update_scan |= self.force_full_scan.unsqueeze(1)
            should_update_obs |= self.force_full_scan
            self.force_full_scan[self.force_full_scan] = False
        hit_positions, hit_distances = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w.repeat_interleave(2, dim=0), # [num_envs * 2, n_meshes, 3]
            mesh_quat_w=mesh_quat_w.repeat_interleave(2, dim=0), # [num_envs * 2, n_meshes, 4]
            ray_starts_w=ray_starts_w.flatten(0, 1), # [num_envs * 2, self.num_rays/2, 3]
            ray_dirs_w=self.ray_dirs_w.flatten(0, 1), # [num_envs * 2, self.num_rays/2, 3]
            enabled=should_update_scan.flatten(0, 1), # [num_envs * 2]
            max_dist=100.,
        )
        self.hit_positions = hit_positions.reshape(self.num_envs, -1, 3)
        # self.hit_positions = hit_positions.reshape(self.num_envs, -1, 3) # [num_envs, self.num_rays, 3]
        # self.hit_distances = hit_distances.reshape(self.num_envs, -1) # [num_envs, self.num_rays]
        hit_positions_b = quat_rotate_inverse(
            self.torso_quat_w.reshape(self.num_envs, 1, 4),
            self.hit_positions - self.torso_pos_w.reshape(self.num_envs, 1, 3)
        ).reshape(self.num_envs, 2, -1, 3)
        front_hit_positions_b = hit_positions_b[:, 0]
        back_hit_positions_b = hit_positions_b[:, 1]
        front_grid = voxelize_wp(
            voxel_shape=self.shape,
            resolution=self.resolution,
            hit_positions_b=front_hit_positions_b,
        )
        back_grid = voxelize_wp(
            voxel_shape=self.shape,
            resolution=self.resolution,
            hit_positions_b=back_hit_positions_b,
        )
        self.left_grid = torch.where(should_update_scan[:, 0].reshape(self.num_envs, 1, 1, 1), front_grid, self.left_grid)
        self.right_grid = torch.where(should_update_scan[:, 1].reshape(self.num_envs, 1, 1, 1), back_grid, self.right_grid)
        self.grid = self.left_grid | self.right_grid
        hole_mask = (torch.rand_like(self.grid_obs, device=self.device, dtype=torch.float32) < self.hole_prob)
        self.grid = torch.where(hole_mask, False, self.grid)
        self.grid_obs = torch.where(should_update_obs.reshape(self.num_envs, 1, 1, 1), self.grid, self.grid_obs)
        self.last_update_obs = torch.where(should_update_obs, self.env.episode_length_buf, self.last_update_obs)
    
    def compute(self) -> torch.Tensor:
        
        # if self.count < 200:
        #     self.saved_array.append(self.grid_obs[0].cpu().numpy())
        #     self.count += 1
        #     print(self.count)
        # elif self.count == 200:
        #     self.saved_array.append(self.grid_obs[0].cpu().numpy())
        #     np.save("grid.npy", np.array(self.saved_array))
        #     self.count += 1
        return einops.rearrange(self.grid_obs, self.pattern)
        # return self.grid
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            # starts = self.asset.data.root_pos_w[0].expand(self.num_rays, 3)
            # self.env.debug_draw.vector(
            #     starts.reshape(-1, 3),
            #     self.ray_dirs.reshape(-1, 3),
            #     color=(1., 0., 1., 1.)
            # )
            # return
            pos = quat_rotate(
                self.torso_quat_w.reshape(self.num_envs, 1, 1, 1, 4),
                self.grid_centers.expand(self.num_envs, *self.shape, 3)
            ) + self.torso_pos_w.reshape(self.num_envs, 1, 1, 1, 3)
            pos = pos[self.grid.bool()]
            self.marker.visualize(pos.reshape(-1, 3))
        elif self.env.backend == "mujoco":
            self.marker_front.geom.pos = self.lidar_pos_w[0, 0]
            self.marker_back.geom.pos = self.lidar_pos_w[0, 1]
        # marker_pos = (self.lidar_pos_w.unsqueeze(2) + self.ray_dirs)[:, 1].reshape(-1, 3)
        # self.marker.visualize(
        #     translations=marker_pos,
        #     scales=torch.ones(3, device=self.device).expand_as(marker_pos),
        # )
    def symmetry_transforms(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0),
            signs=torch.ones(self.shape[1]),
        )