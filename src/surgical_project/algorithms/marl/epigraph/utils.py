"""
Utility functions for Epigraph algorithm.
Contains: GAE computation, value loss, z initialization, and helper functions.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: [T, N, 1] - Rewards at each timestep
        values: [T, N, 1] - Value predictions at each timestep
        masks: [T, N, 1] - Continuation masks (0 at episode boundary, 1 otherwise)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: [T, N, 1] - Computed advantages
        returns: [T, N, 1] - TD(lambda) returns (advantages + values)
    """
    T, N, _ = rewards.shape
    device = rewards.device
    
    # Validate parameters
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma={gamma} not in [0,1]")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"gae_lambda={gae_lambda} not in [0,1]")
    
    advantages = torch.zeros(T, N, 1, device=device)
    gae = torch.zeros(N, 1, device=device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(N, 1, device=device)
        else:
            next_value = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * masks[t] * next_value - values[t]
        
        # GAE recursion
        gae = delta + gamma * gae_lambda * masks[t] * gae
        advantages[t] = gae
    
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages using mean and std of valid (non-masked) entries.
    
    Args:
        advantages: [T, N, 1] - Raw advantages
        masks: [T, N, 1] - Valid entry masks
        eps: Small constant for numerical stability
    
    Returns:
        normalized_advantages: [T, N, 1]
    """
    flat_adv = advantages.view(-1, 1)
    flat_mask = masks.view(-1, 1)
    valid = flat_mask > 0.5
    
    if valid.sum() == 0:
        raise ValueError("No valid advantages to normalize (all masks are zero)")
    
    valid_adv = flat_adv[valid]
    mean = valid_adv.mean()
    std = valid_adv.std().clamp(min=eps)
    
    # Handle pathological case: extremely small std
    if std < 1e-8:
        print(f"[WARNING] Very small advantage std: {float(std):.3e}. Skip normalization.")
        return advantages
    
    normalized = (advantages - mean) / std
    return normalized


def clip_value_loss(
    values_pred: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    clip_param: float,
    huber_delta: float = 1.0,
    use_huber: bool = True,
) -> torch.Tensor:
    """
    Compute clipped value loss with optional Huber loss.
    
    Args:
        values_pred: [*, 1] - Predicted values from current network
        values_old: [*, 1] - Old value predictions (from buffer)
        returns: [*, 1] - Target returns
        clip_param: PPO clipping parameter
        huber_delta: Huber loss delta
        use_huber: Whether to use Huber loss
    
    Returns:
        loss: Scalar loss value
    """
    # Clipped value predictions
    values_clipped = values_old + torch.clamp(
        values_pred - values_old,
        -clip_param,
        clip_param
    )
    
    if use_huber:
        # Huber loss for both unclipped and clipped
        loss_unclipped = F.huber_loss(values_pred, returns, delta=huber_delta, reduction='none')
        loss_clipped = F.huber_loss(values_clipped, returns, delta=huber_delta, reduction='none')
    else:
        # MSE loss
        loss_unclipped = (values_pred - returns) ** 2
        loss_clipped = (values_clipped - returns) ** 2
    
    # Take maximum of clipped and unclipped
    loss = torch.max(loss_unclipped, loss_clipped).mean()
    
    return loss


def init_z(
    mode: str,
    p_extreme: float,
    z_min: float,
    z_max: float,
    n_envs: int,
    n_agents: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Initialize z values for training start.
    
    Args:
        mode: Initialization mode - "uniform", "extreme", or "mixed"
        p_extreme: Probability of extreme values (for "mixed" mode)
        z_min: Lower bound of z
        z_max: Upper bound of z
        n_envs: Number of parallel environments
        n_agents: Number of agents per environment
        device: PyTorch device
    
    Returns:
        z: [n_envs, n_agents, 1] - Initialized z values
    """
    if mode == "uniform":
        # Uniform distribution in [z_min, z_max]
        z = torch.rand(n_envs, n_agents, 1, device=device) * (z_max - z_min) + z_min
    
    elif mode == "extreme":
        # Binary extremes: either z_min or z_max
        random_vals = torch.rand(n_envs, n_agents, 1, device=device)
        z = torch.where(
            random_vals > 0.5,
            torch.full((n_envs, n_agents, 1), z_max, device=device),
            torch.full((n_envs, n_agents, 1), z_min, device=device),
        )
    
    elif mode == "mixed":
        # Mix of extreme and uniform
        # p_extreme chance of extreme values, (1-p_extreme) chance of uniform
        random_choice = torch.rand(n_envs, n_agents, 1, device=device)
        random_extreme = torch.rand(n_envs, n_agents, 1, device=device)
        random_uniform = torch.rand(n_envs, n_agents, 1, device=device)
        
        # For extreme: 50-50 between z_min and z_max
        z_extreme = torch.where(
            random_extreme > 0.5,
            torch.full((n_envs, n_agents, 1), z_max, device=device),
            torch.full((n_envs, n_agents, 1), z_min, device=device),
        )
        
        # For uniform: random in [z_min, z_max]
        z_uniform = random_uniform * (z_max - z_min) + z_min
        
        # Mix based on p_extreme
        z = torch.where(
            random_choice < p_extreme,
            z_extreme,
            z_uniform,
        )
    
    else:
        raise ValueError(f"Unknown z initialization mode: {mode}")
    
    return z


def clip_z(z: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    """
    Clip z values to valid range.
    
    Args:
        z: [*, 1] - Z values to clip
        z_min: Lower bound
        z_max: Upper bound
    
    Returns:
        clipped_z: [*, 1] - Clipped z values
    """
    return torch.clamp(z, z_min, z_max)


def update_z_training(
    z_current: torch.Tensor,
    r_safe: torch.Tensor,
    gamma: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    """
    Update z values during training rollout (per-agent recursive update).
    
    Formula: z_{t+1} = clip((z_t + r_safe_t) / gamma, z_min, z_max)
    
    Args:
        z_current: [N, 1] - Current z values for one agent
        r_safe: [N, 1] - Safety rewards from current step
        gamma: Discount factor
        z_min: Lower bound
        z_max: Upper bound
    
    Returns:
        z_next: [N, 1] - Updated z values
    """
    z_next = (z_current + r_safe) / gamma
    z_next = clip_z(z_next, z_min, z_max)
    return z_next


def z_statistics(z: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics of z values for logging.
    
    Args:
        z: [N, n_agents, 1] or [N, 1] - Z values
    
    Returns:
        stats: Dictionary of statistics
    """
    z_flat = z.view(-1)
    
    stats = {
        "z_mean": float(z_flat.mean().item()),
        "z_std": float(z_flat.std().item()),
        "z_min": float(z_flat.min().item()),
        "z_max": float(z_flat.max().item()),
        "z_median": float(z_flat.median().item()),
    }
    
    return stats


def gradient_norm(parameters) -> float:
    """
    Compute total gradient norm across parameters.
    
    Args:
        parameters: Iterable of parameters
    
    Returns:
        total_norm: L2 norm of gradients
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm