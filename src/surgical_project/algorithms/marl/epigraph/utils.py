"""
Epigraph algorithm utilities.
Contains GAE computation and advantage normalization.
"""

import torch
from typing import Dict, Any


def compute_epigraph_gae(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    z_traj: torch.Tensor,
    vl_preds: torch.Tensor,
    vh_preds: torch.Tensor,
    masks: torch.Tensor,
    term_masks: torch.Tensor,
    ov_mask_vl: torch.Tensor,
    ov_vl: torch.Tensor,
    ov_mask_vh: torch.Tensor,
    ov_vh: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    Compute Epigraph GAE with rMAPPO-aligned bootstrap logic.
    
    Args:
        rewards: [T, E] - Team task rewards
        costs: [T, E, A] - Safety costs per agent
        z_traj: [T, E] - Z trajectory
        vl_preds: [T+1, E] - Vl predictions
        vh_preds: [T+1, E, A] - Vh predictions per agent
        masks: [T, E] - RNN/episode continuation masks
        term_masks: [T, E] - Bootstrap allowance masks
        ov_mask_vl: [T, E] - Override mask for Vl
        ov_vl: [T, E] - Override value for Vl
        ov_mask_vh: [T, E, A] - Override mask for Vh
        ov_vh: [T, E, A] - Override value for Vh
        gamma: Discount factor
        gae_lambda: GAE lambda
    
    Returns:
        Q_perf: [T, E] - Performance returns
        Q_safe: [T, E, A] - Safety returns per agent
        advantages: [T, E, A] - Epigraph advantages
    """
    T, E = rewards.shape
    A = costs.shape[2]
    device = rewards.device
    
    Q_perf = torch.zeros(T, E, device=device)
    Q_safe = torch.zeros(T, E, A, device=device)
    
    gae_perf = torch.zeros(E, device=device)
    gae_safe = torch.zeros(E, A, device=device)
    
    for t in reversed(range(T)):
        m_t = masks[t]
        tm_t = term_masks[t]
        
        # Vl branch (centralized)
        vl_next_nominal = vl_preds[t + 1]
        vl_next_effective = torch.where(
            ov_mask_vl[t].bool(),
            ov_vl[t],
            vl_next_nominal
        )
        vl_next_allowed = tm_t * vl_next_effective
        
        delta_perf = rewards[t] + gamma * vl_next_allowed - vl_preds[t]
        gae_perf = delta_perf + gamma * gae_lambda * m_t * gae_perf
        Q_perf[t] = gae_perf + vl_preds[t]
        
        # Vh branch (per-agent)
        vh_next_nominal = vh_preds[t + 1]
        vh_next_effective = torch.where(
            ov_mask_vh[t].bool(),
            ov_vh[t],
            vh_next_nominal
        )
        vh_next_allowed = tm_t.unsqueeze(-1) * vh_next_effective
        
        delta_safe = costs[t] + gamma * vh_next_allowed - vh_preds[t]
        gae_safe = delta_safe + gamma * gae_lambda * m_t.unsqueeze(-1) * gae_safe
        Q_safe[t] = gae_safe + vh_preds[t]
    
    # Epigraph advantage computation
    vl_curr = vl_preds[:T]
    vh_curr = vh_preds[:T]
    
    vl_minus_z = (vl_curr - z_traj).unsqueeze(-1).expand(T, E, A)
    V_baseline = torch.max(vh_curr, vl_minus_z)
    
    Q_perf_broadcast = Q_perf.unsqueeze(-1).expand(T, E, A)
    Q_combined = torch.max(Q_safe, Q_perf_broadcast)
    
    advantages = Q_combined - V_baseline
    
    return Q_perf, Q_safe, advantages


def normalize_advantages(
    advantages: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages using masked mean and std.
    
    Args:
        advantages: [T, N, 1] - Raw advantages
        masks: [T, N, 1] - Valid step masks
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages [T, N, 1]
    """
    masked_adv = advantages * masks
    num_valid = masks.sum()
    
    if num_valid > 0:
        adv_mean = masked_adv.sum() / num_valid
        adv_var = ((masked_adv - adv_mean) ** 2 * masks).sum() / num_valid
        adv_std = torch.sqrt(adv_var + eps)
        
        normalized = (advantages - adv_mean) / (adv_std + eps)
        return normalized * masks
    else:
        return advantages