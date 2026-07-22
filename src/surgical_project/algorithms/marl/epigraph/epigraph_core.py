"""
Epigraph Core Networks for Safe MARL.
Contains: ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder.
With sequence-based RNN training support (rMAPPO-aligned).

This file is already well-structured and aligned with the paper.
No major changes needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# RNNLayer (rMAPPO-style, supports both acting and sequence training)
# =============================================================================

class RNNLayer(nn.Module): # 初始化输入是
    """
    GRU + LayerNorm, supports:
    - acting mode: x [E,feat], hxs [E,H], masks [E,1] -> out [E,H], next_h [E,H]
    - training mode: x [L*B,feat], hxs [B,H], masks [L*B,1] -> out [L*B,H], next_h [B,H]
    where:
      E = num_envs
      B = mini-batch sequence count
      L = sequence length (data_chunk_length)
    """
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal=True):
        super().__init__()
        self._recurrent_N = recurrent_N
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        # for name, p in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(p, 0.0)
        #     elif 'weight' in name:
        #         if use_orthogonal:
        #             nn.init.orthogonal_(p)
        #         else:
        #             nn.init.xavier_uniform_(p)
        self._init_weights(use_orthogonal=use_orthogonal)
        self.norm = nn.LayerNorm(outputs_dim)

    def _init_weights(self, use_orthogonal=True):
        for name, p in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, hxs, masks): # 输入x h, 输出 
        """
        x:
          acting:   [E, feat]
          training: [L*B, feat]
        hxs:
          acting:   [E, H]
          training: [B, H]
        masks:
          acting:   [E, 1]
          training: [L*B, 1] # L是时间长度
        return:
          out:      same leading dim as x (E or L*B) x H
          hxs_out:  [E, H] or [B, H] (last layer hidden)
        """
        assert hxs.dim() == 2
        layers = self._recurrent_N
        B = hxs.size(0)

        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if masks.dtype not in (torch.float32, torch.float64):
            masks = masks.float()

        # Inference
        if x.dim() == 2 and x.size(0) == B:
            h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()
            m = masks.view(1, B, 1).expand(layers, B, 1).contiguous() # [layers, B, 1]
            h = h * m

            out, h = self.rnn(x.unsqueeze(0), h) # h.shape = [layers, E, H]
            out = out.squeeze(0)  # out.shape: [1, E, H] -> [E, H] here 1 is time step
            out = self.norm(out)
            return out, h[-1] # 都是 [E, H]

        # Training: sequence L×B
        assert x.dim() == 2 and x.size(0) % B == 0, f"x {x.shape} not divisible by batch {B}"
        L = x.size(0) // B
        assert masks.size(0) == L * B, f"masks {masks.shape} vs L*B {L*B}"

        x = x.view(L, B, x.size(1)) # [L, B, feat]
        m = masks.view(L, B, 1) # [L, B, 1]

        h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()

        outs = []
        for t in range(L):
            mt = m[t].view(1, B, 1).expand(layers, B, 1).contiguous() # 【B，1】 -> [1, B, 1] ->【layers，B，1】
            h = h * mt
            out_t, h = self.rnn(x[t].unsqueeze(0), h)
            outs.append(out_t) #   outs = [
                                        # tensor shape [1, B, H],
                                        # tensor shape [1, B, H],
                                        # ...
                                        # tensor shape [1, B, H],
                                        # ]  outs 里存了 L 个 [1, B, H]

        out = torch.cat(outs, dim=0).reshape(L * B, -1) # [L, B, H] -> [L*B, H]
        out = self.norm(out) # [L*B, H]
        return out, h[-1]  # [L * B, H] ->【B, H] out对应多个时间步的输出


# =============================================================================
# TanhGaussian distribution
# =============================================================================

class TanhGaussian: # 涵盖采样 求logprob 求entropy
    """
    tanh(Normal(mean, std)) distribution with log_prob Jacobian correction.
    """
    def __init__(self, mean, log_std, eps=1e-6):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.normal = Normal(mean, self.std)
        self.eps = eps

    def sample(self, generator: torch.Generator | None = None):
        if generator is None:
            z = self.normal.rsample()
        else:
            eps = torch.randn(
                self.mean.shape,
                dtype=self.mean.dtype,
                device="cpu",
                generator=generator, # pytorch随机数生成器
            ) # cpu上生成标准正态分布的噪声
            if self.mean.device.type != "cpu":
                eps = eps.to(self.mean.device) # 转移到与 mean 相同的设备上
            z = self.mean + self.std * eps
        return torch.tanh(z)

    def log_prob(self, action): # log数值更稳定，尤其是多维动作时，概率密度相乘会很小；取 log 后就变成相加，更安全。
        a = torch.clamp(action, -1 + self.eps, 1 - self.eps) # 防止除以0
        z = 0.5 * torch.log((1 + a) / (1 - a)) # z = atanh(a) = 0.5 * log((1+a)/(1-a))
        logp_z = self.normal.log_prob(z) # z∼N(μ,σ) logpZ(z) 用当前的 Normal(mean, std) 分布，计算 z 这个点的 log probability density。
        log_det = torch.log(1 - a.pow(2) + self.eps)
        logp = (logp_z - log_det).sum(dim=-1, keepdim=True) # 修正后的项 [B, action_dim] ->【B, 1】
        # logπ(a∣s)=i∑​logπi​(ai​∣s) 总log probability

        return logp 
    def entropy(self):
        ent = 0.5 * (1 + torch.log(2 * torch.tensor(3.141592653589793))) + self.log_std
        # 正态分布 entropy H(N(μ,σ2))=21​(1+log(2π))+logσ = 1/2​log(2πeσ2)
        return ent.sum(dim=-1, keepdim=True) # 因为 ent 一开始是每个动作维度各自的熵，而 PPO/SAC 通常需要的是整条 action vector 的总熵。 [B, action_dim] ->【B, 1】


# =============================================================================
# Initialization helper
# =============================================================================

def init_linear(module: nn.Linear, gain: float = 1.0, use_orthogonal: bool = True) -> nn.Linear:
    if use_orthogonal:
        nn.init.orthogonal_(module.weight, gain)
    else:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


# =============================================================================
# ZEncoder
# =============================================================================

class ZEncoder(nn.Module):
    """
    Map scalar z to z_enc vector.
    z: [B,1] -> z_enc: [B,nz]
    
    This is a learnable embedding that conditions the policy and critics on the risk budget.
    """
    def __init__(self, nz, z_mean=0.0, z_scale=0.2, use_orthogonal: bool = True):
        super().__init__()
        self.nz = nz
        self.z_mean = z_mean
        self.z_scale = z_scale
        self.fc = nn.Sequential(
            init_linear(nn.Linear(1, nz), gain=1.0, use_orthogonal=use_orthogonal),
            nn.Tanh(),
            init_linear(nn.Linear(nz, nz), gain=1.0, use_orthogonal=use_orthogonal),
        )

    def forward(self, z):
        z_norm = (z - self.z_mean) / (self.z_scale + 1e-8) # 先归一化，然后过网络
        return self.fc(z_norm)


# =============================================================================
# ActorRNN
# =============================================================================

class ActorRNN(nn.Module):
    """
    Policy π(a|o,z) with dual interfaces:
    - act_step(): single-step for environment interaction
    - evaluate_actions_seq(): sequence evaluation for PPO training
    """
    def __init__(self, obs_dim, act_dim, hidden_size, nz, recurrent_N, use_orthogonal: bool = True, gain: float = 0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = init_linear(nn.Linear(obs_dim + nz, hidden_size), gain=1.0, use_orthogonal=use_orthogonal)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=use_orthogonal)
        self.fc_mean = init_linear(nn.Linear(hidden_size, act_dim), gain=gain, use_orthogonal=use_orthogonal) # gain代表要把正则化后的weight和bias放大缩小多少
        self.log_std = nn.Parameter(torch.full((1, act_dim), -0.5))

    def _dist_from_latent(self, h):
        mean = self.fc_mean(h) # mean.shape = [E, act_dim]
        log_std = torch.clamp(self.log_std.expand_as(mean), -1.0, -0.2)
        return TanhGaussian(mean, log_std)

    def act_step(self, obs, z_enc, hxs, masks, deterministic=False, generator=None):
        """
        Single-step acting for rollout/eval.
        
        Args:
            obs: [E, obs_dim]
            z_enc: [E, nz]
            hxs: [E, H]
            masks: [E, 1]
        
        Returns:
            action: [E, act_dim]
            logp: [E, 1]
            next_h: [E, H]
            entropy: [E, 1]
        """
        x = torch.cat([obs, z_enc], dim=-1) # 【E, obs_dim + nz】
        x = F.relu(self.fc1(x)) # 【E, hidden_size】= 【E， H】
        rnn_out, next_h = self.rnn(x, hxs, masks) 

        dist = self._dist_from_latent(rnn_out) #！！！
        if deterministic:
            action = torch.tanh(dist.mean)
        else:
            action = dist.sample(generator=generator)
        action = action.clamp(-1.0 + dist.eps, 1.0 - dist.eps)

        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, next_h, entropy
                                            # 输出动作（mean是从mlp得来，不同环境不一样， logstd自行优化，不同环境同一个维度共享一个logstd）， logppiold(a|s,z，h) 以及下一个隐藏状态 next_h 和 (可以有的熵)
        # obs[t] # [E, obs_dim]      actions[t] # [E, act_dim]      log_probs[t] # [E, 1]    hxs[t] # [E, H]

    def evaluate_actions_seq(self, obs_seq, z_enc_seq, hxs_init, masks_seq, act_seq):
        """
        Sequence evaluation for PPO training.
        
        Args:
            obs_seq: [L*B, obs_dim]
            z_enc_seq: [L*B, nz]
            hxs_init: [B, H]
            masks_seq: [L*B, 1]
            act_seq: [L*B, act_dim]
        
        Returns:
            logp: [L*B, 1]
            entropy: [L*B, 1]
            last_h: [B, H]
        """
        x = torch.cat([obs_seq, z_enc_seq], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, last_h = self.rnn(x, hxs_init, masks_seq)

        dist = self._dist_from_latent(rnn_out)
        logp = dist.log_prob(act_seq)
        entropy = dist.entropy()
        return logp, entropy, last_h # 训练使用 的 logppinew(a|s,z，h) 基于推理得到的action ， entropy也是基于action


# =============================================================================
# CriticVlRNN (centralized performance value)
# =============================================================================

class CriticVlRNN(nn.Module):
    """
    Vl: team performance value with dual interfaces:
    - value_step(): single-step for rollout
    - value_seq(): sequence evaluation for PPO training
    """
    def __init__(self, share_obs_dim, hidden_size, nz, recurrent_N, use_orthogonal: bool = True):
        super().__init__()
        self.share_obs_dim = share_obs_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = init_linear(nn.Linear(share_obs_dim + nz, hidden_size), gain=1.0, use_orthogonal=use_orthogonal)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=use_orthogonal)
        self.fc_value = init_linear(nn.Linear(hidden_size, 1), gain=1.0, use_orthogonal=use_orthogonal)

    def value_step(self, share_obs, z_enc, hxs, masks):
        """
        Single-step value prediction for rollout.
        
        Args:
            share_obs: [E, share_obs_dim]
            z_enc: [E, nz]
            hxs: [E, H]
            masks: [E, 1]
        
        Returns:
            vl: [E, 1]
            next_h: [E, H]
        """
        x = torch.cat([share_obs, z_enc], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, next_h = self.rnn(x, hxs, masks)
        vl = self.fc_value(rnn_out) # ！！！
        return vl, next_h

    def value_seq(self, share_obs_seq, z_enc_seq, hxs_init, masks_seq):
        """
        Sequence value prediction for PPO training.
        
        Args:
            share_obs_seq: [L*B, share_obs_dim]
            z_enc_seq: [L*B, nz]
            hxs_init: [B, H]
            masks_seq: [L*B, 1]
        
        Returns:
            vl: [L*B, 1]
            last_h: [B, H]
        """
        x = torch.cat([share_obs_seq, z_enc_seq], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, last_h = self.rnn(x, hxs_init, masks_seq)
        vl = self.fc_value(rnn_out)
        return vl, last_h


# =============================================================================
# CriticVhRNN (per-agent safety value)
# =============================================================================

class CriticVhRNN(nn.Module):
    """
    Vh: per-agent safety value with dual interfaces:
    - value_step(): single-step for rollout
    - value_seq(): sequence evaluation for PPO training
    """
    def __init__(self, obs_dim, hidden_size, nz, recurrent_N, use_orthogonal: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = init_linear(nn.Linear(obs_dim + nz, hidden_size), gain=1.0, use_orthogonal=use_orthogonal)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=use_orthogonal)
        self.fc_value = init_linear(nn.Linear(hidden_size, 1), gain=1.0, use_orthogonal=use_orthogonal)

    def value_step(self, obs, z_enc, hxs, masks):
        """
        Single-step value prediction for rollout.
        
        Args:
            obs: [E, obs_dim]
            z_enc: [E, nz]
            hxs: [E, H]
            masks: [E, 1]
        
        Returns:
            vh: [E, 1]
            next_h: [E, H]
        """
        x = torch.cat([obs, z_enc], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, next_h = self.rnn(x, hxs, masks)
        vh = self.fc_value(rnn_out)
        return vh, next_h

    def value_seq(self, obs_seq, z_enc_seq, hxs_init, masks_seq):
        """
        Sequence value prediction for PPO training.
        
        Args:
            obs_seq: [L*B, obs_dim]
            z_enc_seq: [L*B, nz]
            hxs_init: [B, H]
            masks_seq: [L*B, 1]
        
        Returns:
            vh: [L*B, 1]
            last_h: [B, H]
        """
        x = torch.cat([obs_seq, z_enc_seq], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, last_h = self.rnn(x, hxs_init, masks_seq)
        vh = self.fc_value(rnn_out)
        return vh, last_h


# =============================================================================
# RootFinder (for inference-time safe z solving)
# =============================================================================

class RootFinder:
    """
    Binary search to find safe z values.
    Used during evaluation to compute the minimum safe risk budget.
    """
    def __init__(self, z_min=-0.6, z_max=0.6, h_tgt=0.0, max_iter=32, tol=1e-4):
        self.z_min = z_min
        self.z_max = z_max
        self.h_tgt = h_tgt
        self.max_iter = max_iter
        self.tol = tol

    @torch.no_grad()
    def solve(self, vh_eval_fn, obs, h_tgt=None, z_min=None, z_max=None):
        """
        Find z* such that Vh(obs, z*) ≈ h_tgt using binary search.
        
        Args:
            vh_eval_fn: Callable([B,1] z) -> [B,1] predicted risk Vh
            obs: [B, obs_dim] observations (for device and batch size)
            h_tgt: Safety threshold (default 0.0)
            z_min, z_max: Search interval (defaults to the configured final range)
        
        Returns:
            z_star: [B, 1] - Minimum safe z value per environment
        """
        h_tgt = self.h_tgt if h_tgt is None else h_tgt
        z_min = self.z_min if z_min is None else float(z_min)
        z_max = self.z_max if z_max is None else float(z_max)
        if z_min > z_max:
            raise ValueError(f"Invalid root search interval [{z_min}, {z_max}]")
        device = obs.device
        B = obs.shape[0]

        low = torch.full((B, 1), z_min, device=device)
        high = torch.full((B, 1), z_max, device=device)
        vh_low = vh_eval_fn(low)
        vh_high = vh_eval_fn(high)
        low_safe = vh_low <= h_tgt
        high_safe = vh_high <= h_tgt

        for _ in range(self.max_iter):
            mid = 0.5 * (low + high)
            vh_mid = vh_eval_fn(mid)
            safe_mask = (vh_mid <= h_tgt)
            high = torch.where(safe_mask, mid, high)
            low = torch.where(safe_mask, low, mid)

            if (high - low).abs().max() < self.tol:
                break

        z_root = 0.5 * (low + high)
        return torch.where(
            low_safe & high_safe,
            torch.full_like(z_root, z_min),
            torch.where(
                (~low_safe) & (~high_safe),
                torch.full_like(z_root, z_max),
                z_root,
            ),
        )
