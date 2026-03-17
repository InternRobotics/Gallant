import torch
from active_adaptation.envs.mdp.terminations import Termination
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.assets import Articulation

from gallant.command import LocoNavigation


class feet_too_close(Termination):
    namespace = "gallant"

    def __init__(self, env, body_names: str, thres: float = 0.06):
        super().__init__(env)
        self.threshold = thres
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        assert len(self.body_ids) == 2, "Only support two bodies"

    def compute(self, termination: torch.Tensor):
        feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
        distance_xy = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
        return (distance_xy < self.threshold).reshape(-1, 1)


class pillar_fall(Termination[LocoNavigation]):
    namespace = "gallant"

    def __init__(self, env, body_names: str, threshold: float = -0.05):
        super().__init__(env)
        self.threshold = threshold
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_names = body_names
        self.body_ids = self.asset.find_bodies(body_names)[0]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        con1 = self.command_manager.raw_terrain_types == 5
        con2 = (
            self.asset.data.body_pos_w[:, self.body_ids][:, :, 2] < self.threshold
        ).any(1, True)
        return (con1.reshape(-1, 1) & con2).reshape(-1, 1)


class no_moving(Termination[LocoNavigation]):
    namespace = "gallant"

    def __init__(self, env, thres: float = 0.01):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        root_pos_w = self.asset.data.root_pos_w
        origin_pos_w = self.command_manager.origin_pos_w.clone()
        ellapsed_step = self.env.episode_length_buf
        dist = (root_pos_w - origin_pos_w)[:, :2].norm(dim=-1)
        return (dist < self.thres).reshape(-1, 1) & (
            (ellapsed_step > 200.0)
            & (ellapsed_step < self.command_manager.resample_interval)
        ).reshape(-1, 1)
