"""
Epigraph algorithm utilities.
Contains GAE computation and advantage normalization.
"""

import torch
from typing import Dict, Any


def compute_epigraph_gae(
    rewards: torch.Tensor,        # [T, E]       team task reward per step (avg over agents)
    costs: torch.Tensor,          # [T, E, A]    per-agent safety "cost" (risk penalty, >=0 typically)
    z_traj: torch.Tensor,         # [T, E]       per-env z over time (same for all agents in that env)
    vl_preds: torch.Tensor,       # [T+1, E]     predicted Vl values (team performance critic)
    vh_preds: torch.Tensor,       # [T+1, E, A]  predicted Vh values (per-agent safety critic)
    masks: torch.Tensor,          # [T, E]       1 if env was alive at step t (used for GAE recursion reset)
    term_masks: torch.Tensor,     # [T, E]       1 if next state is valid for bootstrap (not terminal)
    ov_mask_vl: torch.Tensor,     # [T, E]       optional override mask for Vl bootstrap
    ov_vl: torch.Tensor,          # [T, E]       optional override value for Vl bootstrap
    ov_mask_vh: torch.Tensor,     # [T, E, A]    optional override mask for Vh bootstrap
    ov_vh: torch.Tensor,          # [T, E, A]    optional override value for Vh bootstrap
    gamma: float,
    gae_lambda: float,
    lambda_safe: float,
):
    """
    Compute Epigraph-style returns and advantages.
    
    This is the core GAE function aligned with the Epigraph paper:
    - Vl(s,z): team performance value
    - Vh_i(s,z): per-agent safety value
    - Epigraph backup: V_baseline(s) = max(Vh_i(s), Vl(s) - z)
    - Advantage: A_i(s,a) = max(lambda_safe * Q_safe_i, Q_perf) - V_baseline(s)
    
    === Input Semantics ===
    rewards: Team-level task reward [T,E] = average over agents of per-agent task reward.
             This feeds the Vl (team performance) critic.
    
    costs: Per-agent safety cost [T,E,A] where cost >= 0 (already relu(-r_safe_risk)).
           This feeds the Vh_i (per-agent safety) critics.
    
    z_traj: Risk budget per environment [T,E], shared by all agents in that env.
    
    masks: Standard RNN mask (1 if env alive, 0 after reset).
    
    term_masks: Bootstrap mask (1 if we can bootstrap from t+1, 0 if true terminal).
    
    override_bootstrap: Used for milestone truncation:
        - When milestone is reached, we want to truncate the rollout but NOT treat it as
          a true episode termination.
        - Set ov_mask_vl[t] = True, ov_vl[t] = Vl(s_{t+1}, z_{t+1})
        - Set ov_mask_vh[t,:,i] = True, ov_vh[t,:,i] = Vh_i(s_{t+1}, z_{t+1})
        - Keep term_masks[t] = 1 (allow bootstrap)
        - This allows GAE to use the override value instead of the nominal prediction,
          implementing the "truncation bootstrap" trick from rMAPPO Runner.

    Returns:
        Q_perf:     [T, E]       performance returns for Vl branch (used to train Vl critic)
        Q_safe:     [T, E, A]    safety returns for Vh branch (used to train Vh critics)
        advantages: [T, E, A]    policy advantages per agent (used in PPO actor loss)
                                  (already combined with lambda_safe)
    """
    # ========== Shape Assertions (Critical for debugging milestone truncation) ==========
    T, E = rewards.shape
    assert costs.ndim == 3, f"costs should be [T,E,A], got {costs.shape}"
    A = costs.shape[2]
    
    assert rewards.shape == (T, E), f"rewards shape mismatch: expected ({T},{E}), got {rewards.shape}"
    assert costs.shape == (T, E, A), f"costs shape mismatch: expected ({T},{E},{A}), got {costs.shape}"
    assert z_traj.shape == (T, E), f"z_traj shape mismatch: expected ({T},{E}), got {z_traj.shape}"
    assert vl_preds.shape == (T + 1, E), f"vl_preds shape mismatch: expected ({T+1},{E}), got {vl_preds.shape}"
    assert vh_preds.shape == (T + 1, E, A), f"vh_preds shape mismatch: expected ({T+1},{E},{A}), got {vh_preds.shape}"
    assert masks.shape == (T, E), f"masks shape mismatch: expected ({T},{E}), got {masks.shape}"
    assert term_masks.shape == (T, E), f"term_masks shape mismatch: expected ({T},{E}), got {term_masks.shape}"
    assert ov_mask_vl.shape == (T, E), f"ov_mask_vl shape mismatch: expected ({T},{E}), got {ov_mask_vl.shape}"
    assert ov_vl.shape == (T, E), f"ov_vl shape mismatch: expected ({T},{E}), got {ov_vl.shape}"
    assert ov_mask_vh.shape == (T, E, A), f"ov_mask_vh shape mismatch: expected ({T},{E},{A}), got {ov_mask_vh.shape}"
    assert ov_vh.shape == (T, E, A), f"ov_vh shape mismatch: expected ({T},{E},{A}), got {ov_vh.shape}"
    
    device = rewards.device

    # Storage for final returns (a.k.a. "Q") and running GAE buffers
    Q_perf = torch.zeros(T, E, device=device)
    Q_safe = torch.zeros(T, E, A, device=device)

    gae_perf = torch.zeros(E, device=device)        # [E]
    gae_safe = torch.zeros(E, A, device=device)     # [E, A]

    # Work backwards in time, classic GAE
    for t in reversed(range(T)):
        # masks[t] is 1 if the episode was ongoing at t
        # term_masks[t] is 1 if we are allowed to bootstrap at t+1
        m_t = masks[t]        # [E]
        tm_t = term_masks[t]  # [E]

        # ----- performance branch (Vl) -----
        vl_next_nominal = vl_preds[t + 1]  # [E]
        # allow override (e.g. truncation bootstrap)
        vl_next_effective = torch.where(
            ov_mask_vl[t].bool(),
            ov_vl[t],
            vl_next_nominal
        )  # [E]

        # only bootstrap if term_masks says next state is valid
        vl_next_allowed = tm_t * vl_next_effective  # [E]

        # TD residual (delta) for performance
        delta_perf = rewards[t] + gamma * vl_next_allowed - vl_preds[t]  # [E]

        # standard GAE recursion
        gae_perf = delta_perf + gamma * gae_lambda * m_t * gae_perf      # [E]
        Q_perf[t] = gae_perf + vl_preds[t]                               # [E]

        # ----- safety branch (Vh, per agent) -----
        vh_next_nominal = vh_preds[t + 1]  # [E, A]
        vh_next_effective = torch.where(
            ov_mask_vh[t].bool(),
            ov_vh[t],
            vh_next_nominal
        )  # [E, A]

        vh_next_allowed = tm_t.unsqueeze(-1) * vh_next_effective  # [E, A]

        delta_safe = costs[t] + gamma * vh_next_allowed - vh_preds[t]    # [E, A]

        gae_safe = delta_safe + gamma * gae_lambda * m_t.unsqueeze(-1) * gae_safe  # [E, A]
        Q_safe[t] = gae_safe + vh_preds[t]                                          # [E, A]

    # -------------------------------------------------------------------------
    # Compute policy advantages.
    #
    # Epigraph backup:
    #   V_baseline(s)   = max( Vh(s), Vl(s) - z )
    #   Q_perf_branch   = Q_perf(s,a)           (team return, broadcast to agents)
    #   Q_safe_branch   = Q_safe(s,a)           (per-agent safe return)
    #
    # Original "worst-case" advantage idea:
    #   A = max(Q_safe, Q_perf_broadcast) - V_baseline
    #
    # We now bias safety using lambda_safe:
    #   Q_combined = max(lambda_safe * Q_safe, Q_perf_broadcast)
    #   advantages = Q_combined - V_baseline
    #
    # lambda_safe > 1 makes safety dominate more often in max().
    # -------------------------------------------------------------------------

    vl_curr = vl_preds[:T]           # [T, E]
    vh_curr = vh_preds[:T]           # [T, E, A]

    # Vl - z is per-env; broadcast to [T,E,A]
    vl_minus_z = (vl_curr - z_traj).unsqueeze(-1).expand(T, E, A)  # [T,E,A]

    # Baseline is epigraph value V(s):
    # max( Vh , Vl - z )
    V_baseline = torch.max(vh_curr, vl_minus_z)  # [T,E,A]

    # Broadcast team performance return to every agent
    Q_perf_broadcast = Q_perf.unsqueeze(-1).expand(T, E, A)  # [T,E,A]

    # Safety-weighted combination
    Q_safe_weighted = lambda_safe * Q_safe                   # [T,E,A]
    Q_combined = torch.max(Q_safe_weighted, Q_perf_broadcast)  # [T,E,A]

    # Final per-agent advantage
    advantages = Q_combined - V_baseline  # [T,E,A]

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
        # Broadcast masks to [T,E,A]
        masks_broadcast = masks.unsqueeze(-1).expand(T, E, A)
    elif masks.ndim == 3:
        assert masks.shape == (T, E, 1) or masks.shape == (T, E, A), \
            f"masks shape mismatch: expected ({T},{E},1) or ({T},{E},{A}), got {masks.shape}"
        if masks.shape[-1] == 1:
            masks_broadcast = masks.expand(T, E, A)
        else:
            masks_broadcast = masks
    else:
        raise ValueError(f"masks should be 2D or 3D, got {masks.ndim}D with shape {masks.shape}")
    
    # Masked normalization
    masked_adv = advantages * masks_broadcast
    num_valid = masks_broadcast.sum()
    
    if num_valid > 0:
        adv_mean = masked_adv.sum() / num_valid
        adv_var = ((masked_adv - adv_mean) ** 2 * masks_broadcast).sum() / num_valid
        adv_std = torch.sqrt(adv_var + eps)
        
        normalized = (advantages - adv_mean) / (adv_std + eps)
        return normalized * masks_broadcast
    else:
        return advantages