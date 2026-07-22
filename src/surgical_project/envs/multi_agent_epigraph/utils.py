"""Small EPIGRAPH-specific reward decomposition helpers."""

from typing import Dict, Tuple

import torch


def _zeros(device, num_envs: int) -> torch.Tensor:
    return torch.zeros(num_envs, device=device)


def safe_get_rc(
    reward_components: Dict[str, torch.Tensor] | None,
    key: str,
    device,
    num_envs: int,
) -> torch.Tensor:
    if reward_components is None:
        return _zeros(device, num_envs)
    return reward_components.get(key, _zeros(device, num_envs))


def compose_task_safe_from_rc(
    rc: Dict[str, torch.Tensor],
    agent: str,
    device,
    num_envs: int,
    use_time_eff_in_task: bool = True,
    include_zpenalty_in_safe: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split shared task reward components into performance and safety signals."""
    component = lambda key: safe_get_rc(rc, key, device, num_envs)
    aware_key = "humanaware" if agent == "robot" else "robotaware"
    zone_a_active = component("zoneA_active_mask")
    zone_b_active = component("zoneB_active_mask")
    zone_c_active = component("zoneC_active_mask")
    zone_d_active = component("zoneD_active_mask")

    zone_b_weight = component(f"zoneB_weight_{agent}")
    zone_b_task = zone_b_active * zone_b_weight * (
        component(f"zoneB_gap_{agent}_contrib")
        + component(f"zoneB_surftangent_{agent}_contrib")
    )
    zone_d_weight = component(f"zoneD_weight_{agent}")
    zone_d_task = zone_d_active * zone_d_weight * (
        component(f"zoneD_progress_{agent}_contrib")
        + component(f"zoneD_deviation_{agent}_contrib")
    )

    task_reward = (
        zone_a_active * component(f"zoneA_total_{agent}")
        + zone_b_task
        + zone_d_task
        + component(f"global_potential_{agent}_contrib")
        + component(f"global_completion_{agent}_contrib")
        + (
            component(f"global_timeeff_{agent}_contrib")
            if use_time_eff_in_task
            else _zeros(device, num_envs)
        )
        + component(f"{agent}force_contrib")
        + component(f"{aware_key}_contrib")
    )

    safety_risk = (
        zone_c_active * component(f"zoneC_total_{agent}")
        + zone_b_active * zone_b_weight * component(f"zoneB_inward_{agent}_contrib")
        + zone_d_active * zone_d_weight * component(f"zoneD_inward_{agent}_contrib")
        + (
            component(f"global_zpenalty_{agent}_contrib")
            if include_zpenalty_in_safe
            else _zeros(device, num_envs)
        )
    )
    return task_reward, safety_risk
