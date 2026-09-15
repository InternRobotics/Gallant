from __future__ import annotations

import torch
from typing_extensions import override

from active_adaptation.envs.mdp.terminations import Termination
from active_adaptation.envs.utils import find_bodies

from gallant.command import LocoNavigation


class feet_too_close(Termination):
    namespace = "gallant"

    def __init__(self, body_names: str, thres: float = 0.06):
        super().__init__()
        self.body_names = body_names
        self.threshold = thres

    @override
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]
        body_ids, _ = find_bodies(self.asset, self.body_names)
        self.body_ids = torch.tensor(body_ids, device=self.device)
        assert len(self.body_ids) == 2, "Only support two bodies"

    def compute(self, termination: torch.Tensor):
        feet_pos = self.asset.data.body_pos_w[:, self.body_ids]
        distance_xy = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
        return (distance_xy < self.threshold).reshape(-1, 1)


class pillar_fall(Termination[LocoNavigation]):
    namespace = "gallant"

    def __init__(self, body_names: str, threshold: float = -0.05):
        super().__init__()
        self.body_names = body_names
        self.threshold = threshold

    @override
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]
        body_ids, _ = find_bodies(self.asset, self.body_names)
        self.body_ids = body_ids

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        con1 = self.command_manager.raw_terrain_types == 5
        con2 = (
            self.asset.data.body_pos_w[:, self.body_ids][:, :, 2] < self.threshold
        ).any(1, True)
        return (con1.reshape(-1, 1) & con2).reshape(-1, 1)


class no_moving(Termination[LocoNavigation]):
    namespace = "gallant"

    def __init__(self, thres: float = 0.01):
        super().__init__()
        self.thres = thres

    @override
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        root_pos_w = self.asset.data.root_pos_w
        origin_pos_w = self.command_manager.origin_pos_w.clone()
        elapsed_t = self.command_manager.time_elapsed.squeeze(-1)
        alloted = self.command_manager.time_alloted.squeeze(-1)
        dist = (root_pos_w - origin_pos_w)[:, :2].norm(dim=-1)
        within_budget = elapsed_t < alloted
        return (dist < self.thres).reshape(-1, 1) & (
            (elapsed_t > 4.0) & within_budget
        ).reshape(-1, 1)
