"""Human force models shared by the surgical MARL environments.

This module deliberately depends only on PyTorch so the force model can be
tested without launching Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


SUPPORTED_HUMAN_MODEL_TYPES = (
    "learnable",
    "fixed_impedance",
    "residual_impedance",
)


@dataclass(frozen=True)
class HumanForceResult:
    """Requested force components and the final saturated human force."""

    policy: torch.Tensor
    impedance: torch.Tensor
    residual: torch.Tensor
    total: torch.Tensor
    reference_position: torch.Tensor
    reference_velocity: torch.Tensor


class HumanForceController:
    """Compose learnable, fixed-impedance, and residual human forces."""

    def __init__(self, params: Dict, device: torch.device | str):
        self.device = torch.device(device)
        self.model_type = str(params.get("human_model_type", "learnable"))
        if self.model_type not in SUPPORTED_HUMAN_MODEL_TYPES:
            choices = ", ".join(SUPPORTED_HUMAN_MODEL_TYPES)
            raise ValueError(
                f"Unsupported human_model_type={self.model_type!r}; expected one of: {choices}"
            )

        impedance = params.get("human_impedance", {})
        self.kp = self._vector3(impedance.get("kp", [0.8, 0.8, 0.8]), "human_impedance.kp")
        self.kd = self._vector3(impedance.get("kd", [0.1, 0.1, 0.1]), "human_impedance.kd")
        self.lookahead_distance = float(impedance.get("lookahead_distance", 0.04))
        self.reference_speed = float(impedance.get("reference_speed", 0.02))
        self.max_human_force = float(
            params.get("constraints", {}).get("max_human_force", 0.04)
        )

        if self.lookahead_distance <= 0.0:
            raise ValueError("human_impedance.lookahead_distance must be positive")
        if self.reference_speed < 0.0:
            raise ValueError("human_impedance.reference_speed must be non-negative")
        if self.max_human_force <= 0.0:
            raise ValueError("constraints.max_human_force must be positive")

        trajectory = params.get("trajectory", {})
        self.start = self._vector3(trajectory.get("start_point"), "trajectory.start_point")
        self.end = self._vector3(trajectory.get("end_point"), "trajectory.end_point")
        delta = self.end - self.start
        self.path_length = torch.linalg.vector_norm(delta)
        if float(self.path_length.item()) <= 0.0:
            raise ValueError("trajectory start_point and end_point must differ")
        self.direction = delta / self.path_length

    def _vector3(self, value, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Missing required configuration: {name}")
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.shape != (3,):
            raise ValueError(f"{name} must contain exactly three values, got shape {tuple(tensor.shape)}")
        return tensor

    def compute_reference(
        self, eef_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return look-ahead position and tapered tangent velocity references."""
        relative = eef_position - self.start.unsqueeze(0)
        progress = torch.sum(relative * self.direction.unsqueeze(0), dim=-1)
        projection_progress = torch.clamp(progress, 0.0, float(self.path_length.item()))

        reference_progress = torch.clamp(
            projection_progress + self.lookahead_distance,
            0.0,
            float(self.path_length.item()),
        )
        reference_position = (
            self.start.unsqueeze(0)
            + reference_progress.unsqueeze(-1) * self.direction.unsqueeze(0)
        )

        remaining = torch.clamp(self.path_length - projection_progress, min=0.0)
        speed_taper = torch.clamp(remaining / self.lookahead_distance, 0.0, 1.0)
        reference_velocity = (
            self.reference_speed
            * speed_taper.unsqueeze(-1)
            * self.direction.unsqueeze(0)
        )
        return reference_position, reference_velocity

    def compute_impedance(
        self, eef_position: torch.Tensor, eef_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the per-axis bounded impedance prior."""
        reference_position, reference_velocity = self.compute_reference(eef_position)
        position_error = reference_position - eef_position
        velocity_error = reference_velocity - eef_velocity
        impedance = self.kp.unsqueeze(0) * position_error + self.kd.unsqueeze(0) * velocity_error
        impedance = torch.clamp(
            impedance,
            -self.max_human_force,
            self.max_human_force,
        )
        return impedance, reference_position, reference_velocity

    def compose(
        self,
        policy_force: torch.Tensor,
        eef_position: torch.Tensor,
        eef_velocity: torch.Tensor,
    ) -> HumanForceResult:
        """Interpret a policy command according to ``human_model_type``."""
        policy = policy_force
        impedance, reference_position, reference_velocity = self.compute_impedance(
            eef_position, eef_velocity
        )

        if self.model_type == "learnable":
            residual = torch.zeros_like(policy)
            requested_total = policy
        elif self.model_type == "fixed_impedance":
            residual = torch.zeros_like(policy)
            requested_total = impedance
        else:
            residual = policy
            requested_total = impedance + residual

        # The bounded impedance prior is composed first; this final clamp is the
        # physical human-force safety limit after an optional residual is added.
        total = torch.clamp(
            requested_total,
            -self.max_human_force,
            self.max_human_force,
        )

        return HumanForceResult(
            policy=policy,
            impedance=impedance,
            residual=residual,
            total=total,
            reference_position=reference_position,
            reference_velocity=reference_velocity,
        )

    def wandb_metadata(self) -> Dict[str, object]:
        """Return flat, query-friendly experiment metadata."""
        return {
            "experiment/human_model_type": self.model_type,
            "human/kp_x": float(self.kp[0].item()),
            "human/kp_y": float(self.kp[1].item()),
            "human/kp_z": float(self.kp[2].item()),
            "human/kd_x": float(self.kd[0].item()),
            "human/kd_y": float(self.kd[1].item()),
            "human/kd_z": float(self.kd[2].item()),
            "human/lookahead_distance": self.lookahead_distance,
            "human/reference_speed": self.reference_speed,
            "human/max_force_per_axis": self.max_human_force,
            "human/residual_limit_per_axis": self.max_human_force,
            "human/residual_can_override_impedance": True,
            "human/force_limit_semantics": "per_axis",
        }
