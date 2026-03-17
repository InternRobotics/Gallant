# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools
import math
from einops.layers.torch import Rearrange

from torchrl.data import TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import IndependentNormal, VecNorm
from active_adaptation.learning.ppo.common import *
from active_adaptation.learning.utils.opt import OptimizerGroup

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP
from active_adaptation.utils.torchrl import EnsembleCritic


@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_gallant.PPOPolicy"
    name: str = "ppo_gallant"
    train_every: int = 48
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    desired_kl: Union[float, None] = 0.02
    entropy_coef: float = 0.003
    muon: bool = False

    multi_critic: bool = False
    encoder_type: str = "concat" # How to combine the CNN and MLP features, "concat" or "attn"
    
    compile: bool = False
    use_ddp: bool = True

    store_transitions: bool = False
    in_keys: Tuple[str] = (OBS_KEY, "height_scan", "grid_map_", "base_height")

cs = ConfigStore.instance()
cs.store("ppo_gallant_concat", node=PPOConfig(encoder_type="concat"), group="algo")
cs.store("ppo_gallant_attn", node=PPOConfig(encoder_type="attn"), group="algo")


class EncoderAttn(nn.Module):
    def __init__(self, conv3d: bool=False):
        super().__init__()
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.Mish(), nn.LayerNorm(256),
            nn.LazyLinear(256)
        )
        self.pos_emb = PositionEmbedding1D(embed_dim=32, seq_len=16)

        if conv3d:
            cnn_cls = nn.LazyConv3d
            data_dim = 4 # [C, X, Z, Y]
        else:
            cnn_cls = nn.LazyConv2d
            data_dim = 3 # [C, X, Y]

        cnn_encoder = nn.Sequential(
            nn.Sequential(
                cnn_cls(8, kernel_size=3, stride=2, padding=1), 
                nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                cnn_cls(16, kernel_size=3, stride=2, padding=1),
                nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                cnn_cls(32, kernel_size=3, stride=2, padding=1),
                nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
            ),
            Rearrange("n z x y -> n (x y) z"),
        )
        self.cnn_encoder = FlattenBatch(cnn_encoder, data_dim=data_dim) # [..., Z, X, Y]

        mha = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.mha = FlattenBatch(mha, data_dim=2) # [..., L, D]

        self.ln = nn.LayerNorm(32)
        self.out = nn.Sequential(nn.Mish(), nn.LazyLinear(256), nn.Mish())

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        # CNN input is [N, Z, X, Y]
        cnn_feature = self.cnn_encoder(cnn_inp.float())
        cnn_feature = cnn_feature + self.pos_emb()

        mlp_feature = self.mlp_encoder(mlp_inp)
        mlp_feature = torch.unflatten(mlp_feature, -1, (8, 32))
        attn_out, attn_weights = self.mha(mlp_feature, cnn_feature, cnn_feature)
        feature = self.ln(mlp_feature + attn_out)
        return self.out(feature.flatten(-2))


class EncoderConcat(nn.Module):
    def __init__(self, conv3d: bool=False):
        super().__init__()
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256)
        )

        if conv3d:
            cnn_cls = nn.LazyConv3d
            data_dim = 4 # [C, X, Z, Y]
        else:
            cnn_cls = nn.LazyConv2d
            data_dim = 3 # [C, X, Y]

        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    cnn_cls(8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
                    nn.Flatten(),
                ),
                data_dim=data_dim,
            ),
            nn.LazyLinear(64),
            nn.Mish(),
            nn.LayerNorm(64),
            nn.LazyLinear(64),
        )
        self.out = nn.Sequential(nn.Mish(), nn.LazyLinear(256), nn.Mish())

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        cnn_feature = self.cnn_encoder(cnn_inp.float())
        mlp_feature = self.mlp_encoder(mlp_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        feature = torch.cat([mlp_feature, cnn_feature], dim=-1)
        return self.out(feature)


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: TensorSpec, 
        action_spec: TensorSpec, 
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec

        # when multi_critic is False, aggregate (sum and clip) the rewards BEFORE computing the advantage
        self.multi_critic = self.cfg.multi_critic
        self.num_rewards = reward_spec["reward"].shape[-1]

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param

        # adaptive learning rate to stabilize long-term training
        self.desired_kl = self.cfg.desired_kl
        self.init_lr = self.cfg.lr

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()
        
        if "height_scan" in observation_spec.keys(True, True):
            self.terrain_key = "height_scan"
        else:
            self.terrain_key = "grid_map_"
        
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
        self.hsc_transform = env.observation_funcs[self.terrain_key].symmetry_transform().to(self.device)
        self.act_transform = env.action_manager.symmetry_transform().to(self.device)
        
        self.vecnorm = Mod(VecNorm(observation_spec[OBS_KEY].shape[-1:], decay=1.0), [OBS_KEY], ["_obs_normed"]).to(self.device)

        EncoderClass = {
            "attn": EncoderAttn,
            "concat": EncoderConcat,
        }[cfg.encoder_type]

        actor_module = Seq(
            Mod(EncoderClass(conv3d=False), ["_obs_normed", self.terrain_key, "mask"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = Seq(
            Mod(EncoderClass(conv3d=False), ["_obs_normed", self.terrain_key, "mask"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.)
            elif isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.)
            elif isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        if self.cfg.multi_critic:
            self.critic = EnsembleCritic(self.critic, num_copies=self.num_rewards, init_=init_)

        if active_adaptation.is_distributed():
            self.world_size = active_adaptation.get_world_size()
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic, static_graph=True)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        
        def is_matrix_shaped(param: torch.Tensor) -> bool:
            return param.dim() == 2

        if self.cfg.muon:
            muon = torch.optim.Muon([
                {"params": [p for p in self.actor.parameters() if is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if is_matrix_shaped(p)]},
            ], lr=cfg.lr, adjust_lr_fn="match_rms_adamw", weight_decay=0.01)

            adamw = torch.optim.AdamW([
                {"params": [p for p in self.actor.parameters() if not is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if not is_matrix_shaped(p)]},
            ], lr=cfg.lr, weight_decay=0.01)
            self.opt = OptimizerGroup([muon, adamw])
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=cfg.lr,
                weight_decay=0.01
            )
    
    def get_rollout_policy(self, mode: str="train", critic: bool = False):
        if critic:
            policy = Seq(self.vecnorm, self.critic, self.actor)
        else:
            policy = Seq(self.vecnorm, self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy, fullgraph=True)
        return policy
    
    def on_stage_start(self, stage: str):
        pass

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.copy()

        infos = []
        if self.multi_critic:
            # aggregate the rewards AFTER computing the advantage
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"].sum(-1, True), subtract_mean=True)
        else:
            # aggregate the rewards BEFORE computing the advantage
            tensordict[REWARD_KEY] = tensordict[REWARD_KEY].sum(-1, True).clip(min=0.)
            self._compute_advantage(tensordict, self.critic, "adv", "ret")
            tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(TensorDict(self.update_batch(minibatch), []))

                if self.desired_kl is not None: # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(self.init_lr, actor_lr * 1.1)
                    self.opt.param_groups[0]["lr"] = actor_lr
        
        with torch.no_grad(), torch.device(self.device):
            # check the difference between the output with and without the mask
            # this is used to check if CNN is working properly
            # if the difference is small, then CNN is NOT working properly
            self.vecnorm(tensordict)
            a = self.critic(tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))
            b = self.critic(tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))
            value_diff = F.mse_loss(a["state_value"], b["state_value"])
            critic_feature_norm = b["_critic_feature"].norm(dim=-1, keepdim=True).mean()
            a = self.actor(
                tensordict.replace(mask=torch.zeros(*tensordict.shape, 1)))
            b = self.actor(
                tensordict.replace(mask=torch.ones(*tensordict.shape, 1)))
            policy_diff = F.mse_loss(a["loc"], b["loc"])
            actor_feature_norm = b["_actor_feature"].norm(dim=-1, keepdim=True).mean()

        out = {}
        for k, v in torch.stack(infos).items():
            out[k] = v.detach().mean().item()
        out["actor/feature_norm"] = actor_feature_norm.item()
        out["actor/policy_diff"] = policy_diff.item()
        out["actor/kl"] = kl.item()
        out["actor/lr"] = self.opt.param_groups[0]["lr"]

        out["critic/value_mean"] = tensordict["ret"].mean().item()
        out["critic/value_std"] = tensordict["ret"].std().item()
        out["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        out["critic/feature_norm"] = critic_feature_norm.item()
        out["critic/value_diff"] = value_diff.item()
        if active_adaptation.is_distributed():
            self.vecnorm.module.synchronize(mode="broadcast")
        return sorted(out.items())
    
    def compute_value(self, tensordict: TensorDict):
        tensordict = self.vecnorm(tensordict)
        tensordict = self.critic(tensordict)
        return tensordict

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def update_batch(self, tensordict: TensorDict):
        
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict.pop("loc"), tensordict.pop("scale")

        symmetry = tensordict.empty()
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[self.terrain_key] = self.hsc_transform(tensordict[self.terrain_key])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["is_init"] = tensordict["is_init"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)

        self.vecnorm(tensordict)
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        assert values.shape == b_returns.shape
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed() and not self.cfg.use_ddp:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
        
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
            loc, scale = dist.loc[:bsize], dist.scale[:bsize]
            kl = IndependentNormal.kl(loc_old, scale_old, loc, scale).mean()
        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "actor/kl": kl,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)


class PositionEmbedding1D(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Create sinusoidal position embeddings
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        
        pos_emb = torch.zeros(1, self.seq_len, self.embed_dim)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        
        # Make it learnable by wrapping in nn.Parameter
        self.pos_emb = nn.Parameter(pos_emb)
    
    def forward(self):
        return self.pos_emb