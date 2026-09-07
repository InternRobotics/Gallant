import torch
import einops
import warp as wp
from typing import Tuple, TYPE_CHECKING

import active_adaptation
from active_adaptation.envs.mdp import Observation
from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    quat_from_euler_xyz
)
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaaclab":
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
    
    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([0]),
            signs=torch.tensor([1.]),
        )


class toe_heel_height(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return self.command_manager.toe_heel_height.reshape(self.num_envs, 2)
    
    def symmetry_transform(self):
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
    
    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.tensor([0, 1]),
            signs=torch.tensor([1,-1]),
        )

class terrain_types(Observation[LocoNavigation]):
    def compute(self) -> torch.Tensor:
        return self.command_manager.terrain_types.float()


class lidar_voxel_map(Observation):
    """Front/back lidar voxel occupancy map (Isaac Lab + simple-raycaster)."""

    supported_backends = ("isaaclab",)
    _SCAN_OFFSET_LOW = -4
    _SCAN_OFFSET_HIGH = 5  # randint high is exclusive → [-4, 4]
    _MAX_RAY_DIST = 100.0
    _HARDWARE_OFFSET_Z = 0.2

    def __init__(
        self,
        pattern: str = "n 1 x z y",
        noise_std: float = 0.0,
        include_self: bool = False,
        num_scan: int = 128,
        random_offset: bool = False,
        resolution: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        hole_prob: float = 0.0,
        obs_delay_range: Tuple[int, int] = (5, 10),  # simulation steps
    ):
        super().__init__()
        self.resolution = tuple(resolution)
        if len(self.resolution) != 3:
            raise ValueError("resolution must be a sequence of 3 elements")
        self.shape = (32, 32, 40)
        self.pattern = f"n x y z -> {pattern}"
        self.noise_std = noise_std
        self.include_self = include_self
        self.num_scan = num_scan
        self.random_offset = random_offset
        self.hole_prob = hole_prob
        self.obs_delay_range = obs_delay_range

    def _initialize(self, env):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        body_ids, _ = find_bodies(self.asset, "torso_link")
        self.body_id = body_ids[0]

        self.hardware_offset = torch.tensor(
            [0.0, 0.0, self._HARDWARE_OFFSET_Z], device=self.device
        )
        self.res_t = torch.tensor(self.resolution, device=self.device)

        self.grid = torch.zeros(
            self.num_envs, *self.shape, device=self.device, dtype=torch.bool
        )
        self.grid_obs = torch.zeros_like(self.grid)
        self.front_map = torch.zeros_like(self.grid)
        self.back_map = torch.zeros_like(self.grid)
        self.force_full_scan = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.grid_shape = torch.as_tensor(self.shape, device=self.device)
        self.grid_half_shape = self.grid_shape // 2
        self.grid_half_size = self.grid_half_shape * self.res_t

        grid_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(n, device=self.device) for n in self.shape],
                indexing="ij",
            ),
            dim=-1,
        )  # [Dx, Dy, Dz, 3]
        self.grid_centers = (
            grid_coords * self.res_t - self.grid_half_size + self.res_t / 2
        )
        del grid_coords

        # Per-env sub-voxel jitter (resampled each scan when random_offset=True)
        self.voxel_offset = torch.zeros(self.num_envs, 3, device=self.device)

        self.lidar_pos = torch.tensor(
            [
                [0.12734, 0.00007, 0.17622],   # front
                [-0.11284, -0.0004, 0.17493],  # back
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
        lidar_dirs_back = torch.stack([-vcos * hcos, -vcos * hsin, vsin], dim=-1)
        self.lidar_ray_dirs = torch.stack(
            [lidar_dirs_front, lidar_dirs_back], dim=0
        )  # [2, S, S, 3]
        del (
            vangles,
            hangles,
            vgrid,
            hgrid,
            vsin,
            vcos,
            hsin,
            hcos,
            lidar_dirs_front,
            lidar_dirs_back,
        )

        self.pos_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.rot_noise = torch.zeros(self.num_envs, 4, device=self.device)
        self.rot_noise[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.scan_period_steps = 5  # 10 Hz at step_dt=0.02

        self.scan_time_offset = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.int64
        )
        self.scan_time_offset[:, 1] = self._sample_back_scan_offset(self.num_envs)

        self.marker = None
        if self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/grid_map",
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

        self.max_delay_steps = int(max(self.obs_delay_range))
        self.max_delay_frames = (
            self.max_delay_steps + self.scan_period_steps - 1
        ) // self.scan_period_steps
        self.buffer_len = max(4, self.max_delay_frames + 2)

        self.scan_frame_id = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self.buf_grids = torch.zeros(
            (self.num_envs, 2, self.buffer_len, *self.shape),
            dtype=torch.bool,
            device=self.device,
        )
        self.buf_head = torch.full(
            (self.num_envs, 2), fill_value=-1, dtype=torch.int32, device=self.device
        )
        self.buf_fill_count = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self.obs_delay_frames = torch.zeros(
            self.num_envs, 2, dtype=torch.int32, device=self.device
        )

    def _sample_back_scan_offset(self, n: int) -> torch.Tensor:
        return torch.randint(
            self._SCAN_OFFSET_LOW,
            self._SCAN_OFFSET_HIGH,
            (n,),
            device=self.device,
            dtype=torch.int64,
        )

    def _sample_voxel_offset(self, env_ids: torch.Tensor) -> None:
        """Uniform sub-voxel jitter in [-res/2, res/2] per axis."""
        if not self.random_offset:
            self.voxel_offset[env_ids] = 0.0
            return
        u = torch.rand(len(env_ids), 3, device=self.device)
        self.voxel_offset[env_ids] = (u - 0.5) * self.res_t

    def reset(self, env_ids: torch.Tensor, tensordict=None):
        pos_sigma = 0.005
        ang_sigma = 1.0 * torch.pi / 180.0
        self.pos_noise[env_ids] = (
            torch.randn(len(env_ids), 3, device=self.device) * pos_sigma
        )
        ang_noise = torch.randn(len(env_ids), 3, device=self.device) * ang_sigma
        self.rot_noise[env_ids] = quat_from_euler_xyz(ang_noise)
        self._sample_voxel_offset(env_ids)

        self.grid[env_ids] = False
        self.grid_obs[env_ids] = False
        self.front_map[env_ids] = False
        self.back_map[env_ids] = False

        low_steps, high_steps = self.obs_delay_range
        low_steps = int(low_steps)
        high_steps = int(high_steps)
        delay_steps_front = torch.randint(
            low=low_steps,
            high=high_steps + 1,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.int32,
        )
        delay_steps_back = torch.randint(
            low=low_steps,
            high=high_steps + 1,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.int32,
        )
        delay_frames_front = (
            delay_steps_front + (self.scan_period_steps - 1)
        ) // self.scan_period_steps
        delay_frames_back = (
            delay_steps_back + (self.scan_period_steps - 1)
        ) // self.scan_period_steps

        self.obs_delay_frames[env_ids, 0] = delay_frames_front.clamp_min(0)
        self.obs_delay_frames[env_ids, 1] = delay_frames_back.clamp_min(0)

        self.buf_grids[env_ids] = False
        self.buf_head[env_ids] = -1
        self.buf_fill_count[env_ids] = 0
        self.scan_frame_id[env_ids] = 0

        self.scan_time_offset[env_ids, 1] = self._sample_back_scan_offset(len(env_ids))
        self.force_full_scan[env_ids] = True

    def _enqueue_lidar_frames(
        self, env_mask: torch.Tensor, lidar_idx: int, grids: torch.Tensor
    ):
        if not env_mask.any():
            return
        idxs = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)

        new_heads = (self.buf_head[idxs, lidar_idx] + 1) % self.buffer_len
        self.buf_head[idxs, lidar_idx] = new_heads

        self.buf_grids[idxs, lidar_idx, new_heads] = grids[idxs]
        self.buf_fill_count[idxs, lidar_idx] = torch.clamp(
            self.buf_fill_count[idxs, lidar_idx] + 1, max=self.buffer_len
        )

    def _read_delayed_grid(self, lidar_idx: int) -> torch.Tensor:
        """Delayed occupancy [E, Dx, Dy, Dz] for one lidar (oldest if underfilled)."""
        out = torch.zeros(
            (self.num_envs, *self.shape), dtype=torch.bool, device=self.device
        )

        heads = self.buf_head[:, lidar_idx]
        fills = self.buf_fill_count[:, lidar_idx]
        delay = torch.clamp(
            self.obs_delay_frames[:, lidar_idx], min=0, max=self.buffer_len - 1
        )

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

    def _voxelize_hits(
        self, env_mask: torch.Tensor, hits_b: torch.Tensor
    ) -> torch.Tensor:
        """Voxelize body-frame hits for scanning envs only; others stay empty."""
        grid = torch.zeros(
            self.num_envs, *self.shape, dtype=torch.bool, device=self.device
        )
        if not env_mask.any():
            return grid

        idxs = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        offset = self.hardware_offset + self.voxel_offset[idxs]
        grid[idxs] = voxelize_wp(
            self.shape, self.resolution, hits_b[idxs] + offset.unsqueeze(1)
        )
        if self.hole_prob > 0.0:
            hole = torch.rand(len(idxs), *self.shape, device=self.device) < self.hole_prob
            grid[idxs] = grid[idxs] & ~hole
        return grid

    def update(self):
        self.torso_quat_w = self.asset.data.body_quat_w[:, self.body_id]
        self.torso_pos_w = self.asset.data.body_pos_w[:, self.body_id]

        # Noisy torso frame used for rays, lidar mounts, and body-frame hits.
        noise_torso_pos = self.torso_pos_w + self.pos_noise
        noise_torso_quat = quat_mul(self.rot_noise, self.torso_quat_w)
        self._viz_torso_pos = noise_torso_pos
        self._viz_torso_quat = noise_torso_quat

        lidar_pos_w = noise_torso_pos.reshape(self.num_envs, 1, 3) + quat_rotate(
            noise_torso_quat.reshape(self.num_envs, 1, 4),
            self.lidar_pos.expand(self.num_envs, 2, 3),
        )  # [E, 2, 3]

        ray_dirs_w = quat_rotate(
            noise_torso_quat.reshape(self.num_envs, 1, 1, 4),
            self.lidar_ray_dirs.reshape(1, 2, -1, 3),
        )  # [E, 2, R/2, 3]
        ray_starts_w = lidar_pos_w.unsqueeze(2).expand_as(ray_dirs_w)

        if self.include_self:
            mesh_pos_w = torch.cat(
                [
                    self.asset.data.body_pos_w,
                    torch.zeros(self.num_envs, 1, 3, device=self.device),
                ],
                dim=1,
            )
            mesh_quat_w = torch.cat(
                [
                    self.asset.data.body_quat_w,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(
                        self.num_envs, 1, 4
                    ),
                ],
                dim=1,
            )
        else:
            mesh_pos_w = torch.zeros(self.num_envs, 1, 3, device=self.device)
            mesh_quat_w = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            ).expand(self.num_envs, 1, 4)

        step = self.env.episode_length_buf
        should_update_scan = (
            (step.unsqueeze(1) + self.scan_time_offset) % self.scan_period_steps
        ) == 0
        if self.force_full_scan.any():
            should_update_scan = should_update_scan | self.force_full_scan.unsqueeze(1)

        # Resample sub-voxel offset when either lidar scans this step.
        scanning = should_update_scan.any(dim=1)
        if self.random_offset and scanning.any():
            self._sample_voxel_offset(
                torch.nonzero(scanning, as_tuple=False).squeeze(-1)
            )

        enabled = should_update_scan.flatten(0, 1)
        hit_positions, hit_distances, _ = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w.repeat_interleave(2, dim=0),
            mesh_quat_w=mesh_quat_w.repeat_interleave(2, dim=0),
            ray_starts_w=ray_starts_w.flatten(0, 1),
            ray_dirs_w=ray_dirs_w.flatten(0, 1),
            enabled=enabled,
            max_dist=self._MAX_RAY_DIST,
        )

        hit_distances = hit_distances.reshape(self.num_envs, 2, -1)
        hit_positions = hit_positions.reshape(self.num_envs, 2, -1, 3)
        # Misses / disabled → max_dist or INF; push outside the voxel volume.
        valid = torch.isfinite(hit_distances) & (
            hit_distances < self._MAX_RAY_DIST - 1e-3
        )
        hit_positions = torch.where(
            valid.unsqueeze(-1),
            hit_positions,
            torch.full_like(hit_positions, 1.0e6),
        )

        hit_positions_b = quat_rotate_inverse(
            noise_torso_quat.reshape(self.num_envs, 1, 1, 4),
            hit_positions - noise_torso_pos.reshape(self.num_envs, 1, 1, 3),
        )  # [E, 2, R/2, 3]

        front_hit_b = hit_positions_b[:, 0]
        back_hit_b = hit_positions_b[:, 1]
        if self.noise_std > 0.0:
            front_hit_b = front_hit_b + torch.randn_like(front_hit_b) * self.noise_std
            back_hit_b = back_hit_b + torch.randn_like(back_hit_b) * self.noise_std

        front_mask = should_update_scan[:, 0]
        back_mask = should_update_scan[:, 1]
        front_grid = self._voxelize_hits(front_mask, front_hit_b)
        back_grid = self._voxelize_hits(back_mask, back_hit_b)

        self.scan_frame_id[front_mask, 0] += 1
        self.scan_frame_id[back_mask, 1] += 1
        self._enqueue_lidar_frames(front_mask, 0, front_grid)
        self._enqueue_lidar_frames(back_mask, 1, back_grid)

        should_publish_obs = (step % self.scan_period_steps) == 0
        if self.force_full_scan.any():
            should_publish_obs = should_publish_obs | self.force_full_scan
            self.force_full_scan[self.force_full_scan] = False
        if should_publish_obs.any():
            self.front_map = self._read_delayed_grid(lidar_idx=0)
            self.back_map = self._read_delayed_grid(lidar_idx=1)
            self.grid = self.front_map | self.back_map
            self.grid_obs = self.grid

    def compute(self) -> torch.Tensor:
        return einops.rearrange(self.grid_obs, self.pattern)

    def debug_draw(self):
        if self.marker is None or not hasattr(self, "_viz_torso_pos"):
            return
        pos = quat_rotate(
            self._viz_torso_quat.reshape(self.num_envs, 1, 1, 1, 4),
            self.grid_centers.expand(self.num_envs, *self.shape, 3)
            - self.hardware_offset
            - self.voxel_offset.reshape(self.num_envs, 1, 1, 1, 3),
        ) + self._viz_torso_pos.reshape(self.num_envs, 1, 1, 1, 3)
        occupied = self.grid.bool()
        if not occupied.any():
            return
        self.marker.visualize(pos[occupied].reshape(-1, 3))

    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0),
            signs=torch.ones(self.shape[1]),
        )


class lidar_voxel_map_v2(Observation):
    """Simplified dual-lidar voxel occupancy (Isaac Lab + simple-raycaster).

    Always raycasts the robot visual meshes plus the ground mesh.

    Frame contract
    --------------
    - ``torso_w``: true torso pose from the sim.
    - ``torso_est_w``: noisy torso used as the sensor frame (pose noise).
    - ``hit_b``: hits expressed in ``torso_est`` body frame.
    - ``hit_map``: ``hit_b + hardware_offset (+ voxel_offset)`` fed to voxelize.

    ``update`` is a short pipeline: estimate pose → schedule → raycast →
    voxelize → enqueue → publish delayed union.
    """

    supported_backends = ("isaaclab",)
    _SCAN_OFFSET_LOW = -4
    _SCAN_OFFSET_HIGH = 5  # randint high exclusive → [-4, 4]
    _MAX_RAY_DIST = 100.0
    _HARDWARE_OFFSET_Z = 0.2

    def __init__(
        self,
        pattern: str = "n 1 x z y",
        noise_std: float = 0.0,
        num_scan: int = 128,
        random_offset: bool = False,
        resolution: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        hole_prob: float = 0.0,
        obs_delay_range: Tuple[int, int] = (5, 10),
    ):
        super().__init__()
        self.resolution = tuple(resolution)
        if len(self.resolution) != 3:
            raise ValueError("resolution must be a sequence of 3 elements")
        self.shape = (32, 32, 40)
        self.pattern = f"n x y z -> {pattern}"
        self.noise_std = noise_std
        self.num_scan = num_scan
        self.random_offset = random_offset
        self.hole_prob = hole_prob
        self.obs_delay_range = obs_delay_range

    def _initialize(self, env):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        body_ids, _ = find_bodies(self.asset, "torso_link")
        self.body_id = body_ids[0]

        self.hardware_offset = torch.tensor(
            [0.0, 0.0, self._HARDWARE_OFFSET_Z], device=self.device
        )
        self.res_t = torch.tensor(self.resolution, device=self.device)
        self.grid_shape = torch.as_tensor(self.shape, device=self.device)
        self.grid_half_size = (self.grid_shape // 2) * self.res_t

        grid_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(n, device=self.device) for n in self.shape],
                indexing="ij",
            ),
            dim=-1,
        )
        self.grid_centers = (
            grid_coords * self.res_t - self.grid_half_size + self.res_t / 2
        )
        del grid_coords

        self.grid_obs = torch.zeros(
            self.num_envs, *self.shape, device=self.device, dtype=torch.bool
        )
        self.voxel_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_full_scan = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self._init_lidar_pattern()
        self._init_pose_noise()
        self._init_scheduler()
        self._init_delay_buffers()
        self._init_raycaster()
        self._init_markers()

        # Last undelayed scan (for debug alignment; not what the policy sees).
        self._last_scan_grid = torch.zeros_like(self.grid_obs)
        self._last_torso_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_torso_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_torso_quat[:, 0] = 1.0
        self._last_voxel_offset = torch.zeros(self.num_envs, 3, device=self.device)

    def _init_lidar_pattern(self):
        self.lidar_pos = torch.tensor(
            [
                [0.12734, 0.00007, 0.17622],   # front
                [-0.11284, -0.0004, 0.17493],  # back
            ],
            device=self.device,
        )
        angle = (96 / 180.0) * torch.pi
        vangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
        hangles = torch.linspace(-angle, angle, self.num_scan, device=self.device)
        vgrid, hgrid = torch.meshgrid(vangles, hangles, indexing="ij")
        vsin, vcos = vgrid.sin(), vgrid.cos()
        hsin, hcos = hgrid.sin(), hgrid.cos()
        front = torch.stack([vcos * hcos, vcos * hsin, vsin], dim=-1)
        back = torch.stack([-vcos * hcos, -vcos * hsin, vsin], dim=-1)
        self.lidar_ray_dirs = torch.stack([front, back], dim=0)  # [2, S, S, 3]

    def _init_pose_noise(self):
        self.pos_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.rot_noise = torch.zeros(self.num_envs, 4, device=self.device)
        self.rot_noise[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    def _init_scheduler(self):
        self.scan_period_steps = 5  # 10 Hz at step_dt=0.02
        self.scan_time_offset = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.int64
        )
        self.scan_time_offset[:, 1] = self._sample_back_scan_offset(self.num_envs)

    def _init_delay_buffers(self):
        self.max_delay_steps = int(max(self.obs_delay_range))
        self.max_delay_frames = (
            self.max_delay_steps + self.scan_period_steps - 1
        ) // self.scan_period_steps
        self.buffer_len = max(4, self.max_delay_frames + 2)
        self.buf_grids = torch.zeros(
            (self.num_envs, 2, self.buffer_len, *self.shape),
            dtype=torch.bool,
            device=self.device,
        )
        self.buf_head = torch.full(
            (self.num_envs, 2), fill_value=-1, dtype=torch.int32, device=self.device
        )
        self.buf_fill_count = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self.obs_delay_frames = torch.zeros(
            self.num_envs, 2, dtype=torch.int32, device=self.device
        )

    def _init_raycaster(self):
        paths = [
            f"/World/envs/env_0/Robot/{body_name}/visuals"
            for body_name in self.asset.body_names
        ]
        self.raycaster = MultiMeshRaycaster.from_prim_paths(
            paths,
            stage=get_current_stage(),
            device=wp.get_device(str(self.device)),
            simplify_factor=0.0,
        )
        self.raycaster.add_mesh(self.env.ground_mesh)

    def _init_markers(self):
        self.marker = None
        if not self.env.sim.has_gui():
            return
        self.marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/grid_map_v2",
                markers={
                    "griddot": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.7, 0.9)
                        ),
                    ),
                },
            )
        )
        self.marker.set_visibility(True)

    def _sample_back_scan_offset(self, n: int) -> torch.Tensor:
        return torch.randint(
            self._SCAN_OFFSET_LOW,
            self._SCAN_OFFSET_HIGH,
            (n,),
            device=self.device,
            dtype=torch.int64,
        )

    def _sample_voxel_offset(self, env_ids: torch.Tensor) -> None:
        if not self.random_offset:
            self.voxel_offset[env_ids] = 0.0
            return
        u = torch.rand(len(env_ids), 3, device=self.device)
        self.voxel_offset[env_ids] = (u - 0.5) * self.res_t

    def reset(self, env_ids: torch.Tensor, tensordict=None):
        pos_sigma = 0.005
        ang_sigma = 1.0 * torch.pi / 180.0
        self.pos_noise[env_ids] = (
            torch.randn(len(env_ids), 3, device=self.device) * pos_sigma
        )
        ang_noise = torch.randn(len(env_ids), 3, device=self.device) * ang_sigma
        self.rot_noise[env_ids] = quat_from_euler_xyz(ang_noise)
        self._sample_voxel_offset(env_ids)

        self.grid_obs[env_ids] = False
        self._last_scan_grid[env_ids] = False

        low_steps, high_steps = (int(x) for x in self.obs_delay_range)
        delay_steps = torch.randint(
            low=low_steps,
            high=high_steps + 1,
            size=(len(env_ids), 2),
            device=self.device,
            dtype=torch.int32,
        )
        delay_frames = (delay_steps + (self.scan_period_steps - 1)) // self.scan_period_steps
        self.obs_delay_frames[env_ids] = delay_frames.clamp_min(0)

        self.buf_grids[env_ids] = False
        self.buf_head[env_ids] = -1
        self.buf_fill_count[env_ids] = 0
        self.scan_time_offset[env_ids, 1] = self._sample_back_scan_offset(len(env_ids))
        self.force_full_scan[env_ids] = True

    # ---- update pipeline -------------------------------------------------

    def update(self):
        torso_pos, torso_quat = self._estimate_torso()
        schedule = self._scan_schedule()
        hits_b = self._raycast_hits_body(torso_pos, torso_quat, schedule)
        grids = self._voxelize_lidars(hits_b, schedule)
        self._enqueue_scans(grids, schedule)
        self._maybe_publish(torso_pos, torso_quat, grids, schedule)

    def _estimate_torso(self) -> Tuple[torch.Tensor, torch.Tensor]:
        torso_pos = self.asset.data.body_pos_w[:, self.body_id] + self.pos_noise
        torso_quat = quat_mul(self.rot_noise, self.asset.data.body_quat_w[:, self.body_id])
        return torso_pos, torso_quat

    def _scan_schedule(self) -> torch.Tensor:
        step = self.env.episode_length_buf
        schedule = (
            (step.unsqueeze(1) + self.scan_time_offset) % self.scan_period_steps
        ) == 0
        if self.force_full_scan.any():
            schedule = schedule | self.force_full_scan.unsqueeze(1)
        scanning = schedule.any(dim=1)
        if self.random_offset and scanning.any():
            self._sample_voxel_offset(
                torch.nonzero(scanning, as_tuple=False).squeeze(-1)
            )
        return schedule  # [E, 2]

    def _mesh_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Robot body poses + identity ground row → [E, B+1, …]."""
        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, 1, 4)
        mesh_pos = torch.cat(
            [
                self.asset.data.body_pos_w,
                torch.zeros(self.num_envs, 1, 3, device=self.device),
            ],
            dim=1,
        )
        mesh_quat = torch.cat([self.asset.data.body_quat_w, identity], dim=1)
        return mesh_pos, mesh_quat

    def _raycast_hits_body(
        self,
        torso_pos: torch.Tensor,
        torso_quat: torch.Tensor,
        schedule: torch.Tensor,
    ) -> torch.Tensor:
        lidar_pos_w = torso_pos.reshape(self.num_envs, 1, 3) + quat_rotate(
            torso_quat.reshape(self.num_envs, 1, 4),
            self.lidar_pos.expand(self.num_envs, 2, 3),
        )
        ray_dirs_w = quat_rotate(
            torso_quat.reshape(self.num_envs, 1, 1, 4),
            self.lidar_ray_dirs.reshape(1, 2, -1, 3),
        )
        ray_starts_w = lidar_pos_w.unsqueeze(2).expand_as(ray_dirs_w)

        mesh_pos, mesh_quat = self._mesh_poses()
        enabled = schedule.flatten(0, 1)
        hit_pos, hit_dist, _ = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos.repeat_interleave(2, dim=0),
            mesh_quat_w=mesh_quat.repeat_interleave(2, dim=0),
            ray_starts_w=ray_starts_w.flatten(0, 1),
            ray_dirs_w=ray_dirs_w.flatten(0, 1),
            enabled=enabled,
            max_dist=self._MAX_RAY_DIST,
        )

        hit_dist = hit_dist.reshape(self.num_envs, 2, -1)
        hit_pos = hit_pos.reshape(self.num_envs, 2, -1, 3)
        valid = torch.isfinite(hit_dist) & (hit_dist < self._MAX_RAY_DIST - 1e-3)
        hit_pos = torch.where(valid.unsqueeze(-1), hit_pos, torch.full_like(hit_pos, 1.0e6))

        hits_b = quat_rotate_inverse(
            torso_quat.reshape(self.num_envs, 1, 1, 4),
            hit_pos - torso_pos.reshape(self.num_envs, 1, 1, 3),
        )
        if self.noise_std > 0.0:
            hits_b = hits_b + torch.randn_like(hits_b) * self.noise_std
        return hits_b  # [E, 2, R, 3]

    def _voxelize_lidars(
        self, hits_b: torch.Tensor, schedule: torch.Tensor
    ) -> torch.Tensor:
        grids = torch.zeros(
            self.num_envs, 2, *self.shape, dtype=torch.bool, device=self.device
        )
        for lidar_idx in range(2):
            mask = schedule[:, lidar_idx]
            if not mask.any():
                continue
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            offset = self.hardware_offset + self.voxel_offset[idxs]
            grids[idxs, lidar_idx] = voxelize_wp(
                self.shape,
                self.resolution,
                hits_b[idxs, lidar_idx] + offset.unsqueeze(1),
            )
            if self.hole_prob > 0.0:
                hole = (
                    torch.rand(len(idxs), *self.shape, device=self.device)
                    < self.hole_prob
                )
                grids[idxs, lidar_idx] = grids[idxs, lidar_idx] & ~hole
        return grids  # [E, 2, Dx, Dy, Dz]

    def _enqueue_scans(self, grids: torch.Tensor, schedule: torch.Tensor) -> None:
        for lidar_idx in range(2):
            mask = schedule[:, lidar_idx]
            if not mask.any():
                continue
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            new_heads = (self.buf_head[idxs, lidar_idx] + 1) % self.buffer_len
            self.buf_head[idxs, lidar_idx] = new_heads
            self.buf_grids[idxs, lidar_idx, new_heads] = grids[idxs, lidar_idx]
            self.buf_fill_count[idxs, lidar_idx] = torch.clamp(
                self.buf_fill_count[idxs, lidar_idx] + 1, max=self.buffer_len
            )

    def _read_delayed(self, lidar_idx: int) -> torch.Tensor:
        out = torch.zeros(
            (self.num_envs, *self.shape), dtype=torch.bool, device=self.device
        )
        fills = self.buf_fill_count[:, lidar_idx]
        has_any = fills > 0
        if not has_any.any():
            return out

        idxs = torch.nonzero(has_any, as_tuple=False).squeeze(-1)
        heads = self.buf_head[idxs, lidar_idx]
        delay = torch.clamp(
            self.obs_delay_frames[idxs, lidar_idx], min=0, max=self.buffer_len - 1
        )
        want = (heads - delay) % self.buffer_len
        too_deep = delay >= fills[idxs]
        if too_deep.any():
            want[too_deep] = (heads[too_deep] - (fills[idxs][too_deep] - 1)) % self.buffer_len
        out[idxs] = self.buf_grids[idxs, lidar_idx, want]
        return out

    def _maybe_publish(
        self,
        torso_pos: torch.Tensor,
        torso_quat: torch.Tensor,
        grids: torch.Tensor,
        schedule: torch.Tensor,
    ) -> None:
        step = self.env.episode_length_buf
        publish = (step % self.scan_period_steps) == 0
        if self.force_full_scan.any():
            publish = publish | self.force_full_scan
            self.force_full_scan[self.force_full_scan] = False
        if not publish.any():
            return

        self.grid_obs = self._read_delayed(0) | self._read_delayed(1)

        # Cache undelayed union + capture pose for alignment debug_draw.
        scanned = schedule.any(dim=1)
        if scanned.any():
            idxs = torch.nonzero(scanned, as_tuple=False).squeeze(-1)
            self._last_scan_grid[idxs] = grids[idxs, 0] | grids[idxs, 1]
            self._last_torso_pos[idxs] = torso_pos[idxs]
            self._last_torso_quat[idxs] = torso_quat[idxs]
            self._last_voxel_offset[idxs] = self.voxel_offset[idxs]

    def compute(self) -> torch.Tensor:
        return einops.rearrange(self.grid_obs, self.pattern)

    def debug_draw(self):
        """Draw last undelayed scan at its capture pose (alignment check)."""
        if self.marker is None:
            return
        occupied = self._last_scan_grid.bool()
        if not occupied.any():
            return
        pos = quat_rotate(
            self._last_torso_quat.reshape(self.num_envs, 1, 1, 1, 4),
            self.grid_centers.expand(self.num_envs, *self.shape, 3)
            - self.hardware_offset
            - self._last_voxel_offset.reshape(self.num_envs, 1, 1, 1, 3),
        ) + self._last_torso_pos.reshape(self.num_envs, 1, 1, 1, 3)
        self.marker.visualize(pos[occupied].reshape(-1, 3))

    def symmetry_transform(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0),
            signs=torch.ones(self.shape[1]),
        )
