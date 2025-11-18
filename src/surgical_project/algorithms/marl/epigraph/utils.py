"""
Epigraph algorithm utilities.
Contains GAE computation and advantage normalization.
"""

import torch
from typing import Dict, Any


def compute_dec_efocp_gae_dp(
    rewards: torch.Tensor,        # [T, E]
    costs: torch.Tensor,          # [T, E, A]
    z_traj: torch.Tensor,         # [T, E]
    vl_preds: torch.Tensor,       # [T+1, E]
    vh_preds: torch.Tensor,       # [T+1, E, A]
    masks: torch.Tensor,          # [T, E]
    term_masks: torch.Tensor,     # [T, E]
    ov_mask_vl: torch.Tensor,     # [T, E]
    ov_vl: torch.Tensor,          # [T, E]
    ov_mask_vh: torch.Tensor,     # [T, E, A]
    ov_vh: torch.Tensor,          # [T, E, A]
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dec-EFOCP GAE with discount-to-max (PyTorch port of def-marl implementation)."""
    T, E = rewards.shape
    _, _, A = costs.shape

    device = rewards.device
    Q_perf = torch.zeros(T, E, device=device)
    Q_safe = torch.zeros(T, E, A, device=device)

    for env_idx in range(E):
        gae_perf = torch.zeros((), device=device)
        gae_safe = torch.zeros(A, device=device)

        for t in range(T - 1, -1, -1):
            m_t = masks[t, env_idx]
            tm_t = term_masks[t, env_idx]

            # performance branch (Vl)
            vl_next_nominal = vl_preds[t + 1, env_idx]
            vl_next_effective = torch.where(
                ov_mask_vl[t, env_idx],
                ov_vl[t, env_idx],
                vl_next_nominal
            )
            if ov_mask_vl[t, env_idx] and tm_t <= 0:
                raise ValueError(f"Vl override set where term_masks=0 at t={t}, env={env_idx}")
            vl_next_allowed = tm_t * vl_next_effective
            l_t = -rewards[t, env_idx]
            delta_perf = l_t + gamma * vl_next_allowed - vl_preds[t, env_idx]
            gae_perf = delta_perf + gamma * gae_lambda * m_t * gae_perf
            Q_perf[t, env_idx] = gae_perf + vl_preds[t, env_idx]

            # safety branch (Vh)
            vh_next_nominal = vh_preds[t + 1, env_idx]
            vh_next_effective = torch.where(
                ov_mask_vh[t, env_idx],
                ov_vh[t, env_idx],
                vh_next_nominal
            )
            if ov_mask_vh[t, env_idx].any() and tm_t <= 0:
                raise ValueError(f"Vh override set where term_masks=0 at t={t}, env={env_idx}")
            vh_next_allowed = tm_t * vh_next_effective
            h_t = costs[t, env_idx]
            disc_to_h = (1.0 - gamma) * h_t + gamma * vh_next_allowed
            target_vh = torch.maximum(h_t, disc_to_h)
            delta_safe = target_vh - vh_preds[t, env_idx]
            gae_safe = delta_safe + gamma * gae_lambda * m_t * gae_safe
            Q_safe[t, env_idx] = gae_safe + vh_preds[t, env_idx]

    vl_curr = vl_preds[:T]
    vh_curr = vh_preds[:T]
    vl_minus_z = vl_curr.unsqueeze(-1) - z_traj.unsqueeze(-1)
    V_baseline = torch.maximum(vh_curr, vl_minus_z)

    Q_perf_broadcast = Q_perf.unsqueeze(-1).expand_as(Q_safe)
    Q_mixed = torch.maximum(Q_safe, Q_perf_broadcast)
    advantages = Q_mixed - V_baseline

    return Q_perf, Q_safe, advantages


def normalize_advantages(
    advantages: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages using masked mean and std.
    
    Args:
        advantages: [T, E, A] - Raw advantages per agent
        masks: [T, E] - Valid step masks (1 if env alive, 0 after reset)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages [T, E, A]
    """
    # Shape assertions
    assert advantages.ndim == 3, f"advantages should be [T,E,A], got shape {advantages.shape}"
    T, E, A = advantages.shape
    
    if masks.ndim == 2:
        assert masks.shape == (T, E), f"masks shape mismatch: expected ({T},{E}), got {masks.shape}"
        masks_broadcast = masks.unsqueeze(-1).expand(T, E, A)
    elif masks.ndim == 3:
        assert masks.shape == (T, E, 1) or masks.shape == (T, E, A), \
            f"masks shape mismatch: expected ({T},{E},1) or ({T},{E},{A}), got {masks.shape}"
        masks_broadcast = masks.expand(T, E, A) if masks.shape[-1] == 1 else masks
    else:
        raise ValueError(f"masks should be 2D or 3D, got {masks.ndim}D with shape {masks.shape}")
    
    normalized = torch.zeros_like(advantages)
    for env_idx in range(E):
        mask_env = masks_broadcast[:, env_idx, :]  # [T, A]
        adv_env = advantages[:, env_idx, :]
        num_valid = mask_env.sum()
        if num_valid > 0:
            masked_adv_env = adv_env * mask_env
            mean_env = masked_adv_env.sum() / num_valid
            centered = adv_env - mean_env
            var_env = ((centered ** 2) * mask_env).sum() / num_valid
            std_env = torch.sqrt(var_env + eps)
            normalized_env = (centered / (std_env + eps)) * mask_env
            normalized[:, env_idx, :] = normalized_env
        else:
            normalized[:, env_idx, :] = 0.0
    
    return normalized
