"""
Epigraph Core Networks for Safe MARL.
Contains: ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder.
With sequence-based RNN training support (rMAPPO-aligned).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# RNNLayer (rMAPPO-style, supports both acting and sequence training)
# =============================================================================

class RNNLayer(nn.Module):
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
        for name, p in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        x:
          acting:   [E, feat]
          training: [L*B, feat]
        hxs:
          acting:   [E, H]
          training: [B, H]
        masks:
          acting:   [E, 1]
          training: [L*B, 1]
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

        # Acting single-step
        if x.dim() == 2 and x.size(0) == B:
            h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()
            m = masks.view(1, B, 1).expand(layers, B, 1).contiguous()
            h = h * m

            out, h = self.rnn(x.unsqueeze(0), h)
            out = out.squeeze(0)
            out = self.norm(out)
            return out, h[-1]

        # Training: sequence L×B
        assert x.dim() == 2 and x.size(0) % B == 0, f"x {x.shape} not divisible by batch {B}"
        L = x.size(0) // B
        assert masks.size(0) == L * B, f"masks {masks.shape} vs L*B {L*B}"

        x = x.view(L, B, x.size(1))
        m = masks.view(L, B, 1)

        h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()

        outs = []
        for t in range(L):
            mt = m[t].view(1, B, 1).expand(layers, B, 1).contiguous()
            h = h * mt
            out_t, h = self.rnn(x[t].unsqueeze(0), h)
            outs.append(out_t)

        out = torch.cat(outs, dim=0).reshape(L * B, -1)
        out = self.norm(out)
        return out, h[-1]


# =============================================================================
# TanhGaussian distribution
# =============================================================================

class TanhGaussian:
    """
    tanh(Normal(mean, std)) distribution with log_prob Jacobian correction.
    """
    def __init__(self, mean, log_std, eps=1e-6):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.normal = Normal(mean, self.std)
        self.eps = eps

    def sample(self):
        z = self.normal.rsample()
        return torch.tanh(z)

    def log_prob(self, action):
        a = torch.clamp(action, -1 + self.eps, 1 - self.eps)
        z = 0.5 * torch.log((1 + a) / (1 - a))
        logp_z = self.normal.log_prob(z)
        log_det = torch.log(1 - a.pow(2) + self.eps)
        logp = (logp_z - log_det).sum(dim=-1, keepdim=True)
        return logp

    def entropy(self):
        ent = 0.5 * (1 + torch.log(2 * torch.tensor(3.141592653589793))) + self.log_std
        return ent.sum(dim=-1, keepdim=True)


# =============================================================================
# Initialization helper
# =============================================================================

def ortho_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    return m


# =============================================================================
# ZEncoder
# =============================================================================

class ZEncoder(nn.Module):
    """
    Map scalar z to z_enc vector.
    z: [B,1] -> z_enc: [B,nz]
    """
    def __init__(self, nz, z_mean=0.0, z_scale=0.2):
        super().__init__()
        self.nz = nz
        self.z_mean = z_mean
        self.z_scale = z_scale
        self.fc = nn.Sequential(
            ortho_init(nn.Linear(1, nz), 1.0),
            nn.Tanh(),
            ortho_init(nn.Linear(nz, nz), 1.0),
        )

    def forward(self, z):
        z_norm = (z - self.z_mean) / (self.z_scale + 1e-8)
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
    def __init__(self, obs_dim, act_dim, hidden_size, nz, recurrent_N):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = ortho_init(nn.Linear(obs_dim + nz, hidden_size), 1.0)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=True)
        self.fc_mean = ortho_init(nn.Linear(hidden_size, act_dim), 0.01)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def _dist_from_latent(self, h):
        mean = self.fc_mean(h)
        log_std = self.log_std.expand_as(mean)
        return TanhGaussian(mean, log_std)

    def act_step(self, obs, z_enc, hxs, masks, deterministic=False):
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
        x = torch.cat([obs, z_enc], dim=-1)
        x = F.relu(self.fc1(x))
        rnn_out, next_h = self.rnn(x, hxs, masks)

        dist = self._dist_from_latent(rnn_out)
        if deterministic:
            action = torch.tanh(dist.mean)
        else:
            action = dist.sample()

        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, next_h, entropy

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
        return logp, entropy, last_h


# =============================================================================
# CriticVlRNN (centralized performance value)
# =============================================================================

class CriticVlRNN(nn.Module):
    """
    Vl: team performance value with dual interfaces:
    - value_step(): single-step for rollout
    - value_seq(): sequence evaluation for PPO training
    """
    def __init__(self, share_obs_dim, hidden_size, nz, recurrent_N):
        super().__init__()
        self.share_obs_dim = share_obs_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = ortho_init(nn.Linear(share_obs_dim + nz, hidden_size), 1.0)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=True)
        self.fc_value = ortho_init(nn.Linear(hidden_size, 1), 1.0)

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
        vl = self.fc_value(rnn_out)
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
    def __init__(self, obs_dim, hidden_size, nz, recurrent_N):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.nz = nz

        self.fc1 = ortho_init(nn.Linear(obs_dim + nz, hidden_size), 1.0)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_N, use_orthogonal=True)
        self.fc_value = ortho_init(nn.Linear(hidden_size, 1), 1.0)

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
    Optional utility for inference.
    """
    def __init__(self, z_min=-0.6, z_max=0.6, max_iter=32, tol=1e-4):
        self.z_min = z_min
        self.z_max = z_max
        self.max_iter = max_iter
        self.tol = tol

    @torch.no_grad()
    def solve(self, vh_eval_fn, obs, h_tgt=0.0):
        """
        vh_eval_fn: Callable([B,1] z) -> [B,1] predicted risk Vh
        obs: not used directly unless vh_eval_fn closes over it
        h_tgt: safety threshold
        """
        device = obs.device
        B = obs.shape[0]

        low = torch.full((B, 1), self.z_min, device=device)
        high = torch.full((B, 1), self.z_max, device=device)

        for _ in range(self.max_iter):
            mid = 0.5 * (low + high)
            vh_mid = vh_eval_fn(mid)
            safe_mask = (vh_mid <= h_tgt)
            high = torch.where(safe_mask, mid, high)
            low = torch.where(safe_mask, low, mid)

            if (high - low).abs().max() < self.tol:
                break

        return 0.5 * (low + high)