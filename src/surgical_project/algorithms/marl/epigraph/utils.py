"""
Epigraph algorithm utilities.
Contains GAE computation and advantage normalization.
"""

import torch


def compute_dec_efocp_gae_dp(
    rewards: torch.Tensor,        # [T, E]
    costs: torch.Tensor,          # [T, E, A]
    z_traj: torch.Tensor,         # [T, E]
    vl_preds: torch.Tensor,       # [T+1, E]
    vh_preds: torch.Tensor,       # [T+1, E, A]
    continuation_masks: torch.Tensor,  # [T, E]
    bootstrap_masks: torch.Tensor,     # [T, E]
    ov_mask_vl: torch.Tensor,     # [T, E]
    ov_vl: torch.Tensor,          # [T, E]
    ov_mask_vh: torch.Tensor,     # [T, E, A]
    ov_vh: torch.Tensor,          # [T, E, A]
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference Dec-EFOCP lambda return with local rollout-boundary handling."""
    T, E = rewards.shape
    _, _, A = costs.shape

    device, dtype = rewards.device, rewards.dtype
    Q_perf = torch.zeros(T, E, device=device, dtype=dtype)
    Q_safe = torch.zeros(T, E, A, device=device, dtype=dtype)
    Q_total = torch.zeros(T, E, A, device=device, dtype=dtype)

    invalid_override = (
        (ov_mask_vl & (bootstrap_masks <= 0.5))
        | (ov_mask_vh.any(dim=-1) & (bootstrap_masks <= 0.5))
    )
    if invalid_override.any():
        raise ValueError("Bootstrap override set where bootstrap_masks=0")

    next_vh_rows = torch.zeros(T + 1, E, A, device=device, dtype=dtype)
    next_vl_rows = torch.zeros(T + 1, E, A, device=device, dtype=dtype)
    gae_coeffs = torch.zeros(T + 1, E, device=device, dtype=dtype)
    segment_age = torch.zeros(E, device=device, dtype=torch.long)
    row_ids = torch.arange(T + 1, device=device).unsqueeze(1)
    env_ids = torch.arange(E, device=device)
    lambda_tensor = torch.as_tensor(gae_lambda, device=device, dtype=dtype)

    for t in range(T - 1, -1, -1):
        reset = (continuation_masks[t] <= 0.5) | (t == T - 1)
        keep = (~reset).to(dtype)
        next_vh_rows = next_vh_rows * keep.view(1, E, 1)
        next_vl_rows = next_vl_rows * keep.view(1, E, 1)
        gae_coeffs = gae_coeffs * keep.view(1, E)
        segment_age = torch.where(reset, torch.zeros_like(segment_age), segment_age)

        can_bootstrap = bootstrap_masks[t] > 0.5
        final_vl = torch.where(ov_mask_vl[t], ov_vl[t], vl_preds[t + 1])
        final_vl = torch.where(can_bootstrap, final_vl, torch.zeros_like(final_vl))
        bootstrap_vh = torch.where(ov_mask_vh[t], ov_vh[t], vh_preds[t + 1])
        # Vl is an accumulated cost, so its no-bootstrap terminal value is 0.
        # Vh is a maximum-over-time constraint value, whose terminal boundary
        # is the terminal constraint h itself. Using 0 here would incorrectly
        # floor every safe (negative-h) terminal target at zero.
        final_vh = torch.where(
            can_bootstrap.unsqueeze(-1), bootstrap_vh, costs[t]
        )
        next_vl_rows[0] = torch.where(
            reset.unsqueeze(-1), final_vl.unsqueeze(-1).expand(E, A), next_vl_rows[0]
        )
        next_vh_rows[0] = torch.where(reset.unsqueeze(-1), final_vh, next_vh_rows[0])
        gae_coeffs[0] = torch.where(reset, torch.ones_like(gae_coeffs[0]), gae_coeffs[0])

        active = row_ids <= segment_age.unsqueeze(0)
        active_h = active.unsqueeze(-1)
        h_t = costs[t].unsqueeze(0)
        l_t = -rewards[t].view(1, E, 1)
        vh_rows = torch.where(
            active_h,
            torch.maximum(h_t, (1.0 - gamma) * h_t + gamma * next_vh_rows),
            torch.zeros_like(next_vh_rows),
        )
        vl_rows = torch.where(
            active_h,
            l_t + gamma * next_vl_rows,
            torch.zeros_like(next_vl_rows),
        )
        masked_z = torch.where(active, z_traj[t].unsqueeze(0), 0.0).unsqueeze(-1)
        total_rows = torch.maximum(vh_rows, vl_rows - masked_z)

        weights = gae_coeffs.unsqueeze(-1)
        Q_safe[t] = (vh_rows * weights).sum(dim=0)
        Q_perf[t] = (vl_rows[:, :, 0] * gae_coeffs).sum(dim=0)
        Q_total[t] = (total_rows * weights).sum(dim=0)

        insert_at = segment_age + 1
        vh_rows[insert_at, env_ids] = vh_preds[t]
        vl_rows[insert_at, env_ids] = vl_preds[t].unsqueeze(-1).expand(E, A)
        next_vh_rows = vh_rows
        next_vl_rows = vl_rows

        gae_coeffs = torch.roll(gae_coeffs, shifts=1, dims=0)
        age_float = segment_age.to(dtype)
        gae_coeffs[0] = torch.pow(lambda_tensor, age_float + 1.0)
        gae_coeffs[1] = torch.pow(lambda_tensor, age_float) * (1.0 - gae_lambda)
        segment_age = segment_age + 1

    vl_curr = vl_preds[:T]
    vh_curr = vh_preds[:T]
    vl_minus_z = vl_curr.unsqueeze(-1) - z_traj.unsqueeze(-1)
    V_baseline = torch.maximum(vh_curr, vl_minus_z)

    advantages = Q_total - V_baseline

    return Q_perf, Q_safe, advantages


def normalize_advantages(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize over time independently for each environment and agent.
    
    Args:
        advantages: [T, E, A] - Raw advantages per agent
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages [T, E, A]
    """
    # Shape assertions
    assert advantages.ndim == 3, f"advantages should be [T,E,A], got shape {advantages.shape}"
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Non-finite EPIGRAPH advantages")
    mean = advantages.mean(dim=0, keepdim=True)
    std = advantages.std(dim=0, keepdim=True, unbiased=False)
    return (advantages - mean) / (std + eps)
