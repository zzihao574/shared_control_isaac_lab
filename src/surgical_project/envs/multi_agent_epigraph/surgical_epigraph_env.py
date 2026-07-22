"""Shared surgical MARL environment with EPIGRAPH reward decomposition."""

from __future__ import annotations

from typing import Any, Dict

import torch

from surgical_project.envs.multi_agent.surgical_direct_marl_env import (
    SurgicalDirectMARLEnv,
)
from surgical_project.envs.multi_agent.utils import StepTracer

from .surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
from .utils import compose_task_safe_from_rc


class SurgicalEpigraphEnv(SurgicalDirectMARLEnv):
    """Add EPIGRAPH task/safety signals to the shared physical environment."""

    cfg: SurgicalEpigraphEnvCfg

    def __init__(
        self,
        cfg: SurgicalEpigraphEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        logging_cfg = self.params.get("logging", {})
        self.step_tracer = StepTracer(
            num_envs=self.num_envs,
            device=self.device,
            enable_console_logging=bool(
                logging_cfg.get("enable_console_logging", False)
            ),
            print_every_steps=int(logging_cfg.get("print_every_steps", 10)),
            max_envs_to_print=int(logging_cfg.get("max_envs_to_print", 2)),
        )
        self._last_z_snapshot = None
        constraint_geometry = self.params.get("constraint_geometry", {})
        self.constraint_h_scale = float(
            constraint_geometry.get("constraint_h_scale", 0.015)
        )

    def step(self, actions: Dict[str, torch.Tensor]) -> tuple:
        obs, rewards, terminated, truncated, info = super().step(actions)

        r_task: Dict[str, torch.Tensor] = {}
        r_safe: Dict[str, torch.Tensor] = {}
        safety_risk: Dict[str, torch.Tensor] = {}
        constraint_h: Dict[str, torch.Tensor] = {}
        threshold = max(float(self.collision_threshold), 1e-6)
        h_scale = max(float(self.constraint_h_scale), threshold)
        clearance = self.safety_distances_t1 - threshold
        unsafe = self.is_violating_t1 | (clearance < 0.0)
        safe_h = -(clearance.clamp(min=0.0) / h_scale).clamp(max=1.0)
        unsafe_h = (0.5 + (-clearance).clamp(min=0.0) / threshold).clamp(max=1.0)
        signed_h = torch.where(unsafe, unsafe_h, safe_h)
        for agent in self.cfg.possible_agents:
            task_reward, risk = compose_task_safe_from_rc(
                rc=self.reward_components,
                agent=agent,
                device=self.device,
                num_envs=self.num_envs,
            )
            r_task[agent] = task_reward.view(self.num_envs, 1)
            safety_risk[agent] = risk.view(self.num_envs, 1)
            r_safe[agent] = torch.relu(-risk).view(self.num_envs, 1)
            constraint_h[agent] = signed_h.view(self.num_envs, 1)

        info = dict(info) if info is not None else {}
        info.update(
            {
                "r_task": r_task,
                "r_safe": r_safe,
                "safety_risk": safety_risk,
                "constraint_h": constraint_h,
                "is_violating": self.is_violating_t1.clone(),
                "safety_distance": self.safety_distances_t1.clone(),
                "rejoin_streak": self.rejoin_streak.clone(),
                "progress_ratio": self.reward_components.get(
                    "progress_ratio",
                    torch.zeros(self.num_envs, device=self.device),
                ).clone(),
            }
        )
        return obs, rewards, terminated, truncated, info
