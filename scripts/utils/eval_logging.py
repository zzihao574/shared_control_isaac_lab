#!/usr/bin/env python3
"""
Shared evaluation logging utilities for play scripts.

This module provides a lightweight recorder that:
  * collects per-step metrics (progress, deviation, safety, rewards, forces)
  * aggregates per-episode statistics with algorithm-specific score formulas
  * persists summaries/step traces/force traces to disk
  * optionally interfaces with WandB (kept minimal to avoid hard dependency)

It deliberately avoids touching algorithm internals. Play scripts call:

    recorder = EvalRecorder(agent_ids, mode="rmappo", completion_threshold=0.01)
    for ep in range(num_episodes):
        recorder.start_episode(ep)
        ...
        recorder.record_step(step_idx, env, rewards, info, detail)
        ...
        summary = recorder.end_episode()
    recorder.save(save_dir)

All helper functions tolerate missing tensors/fields and fall back to None.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml

__all__ = [
    "EvalRecorder",
    "EvalWandBLogger",
    "print_paper_metrics",
    "resolve_best_checkpoint",
    "wandb_available",
]


_CHECKPOINT_SCORE_RE = re.compile(
    r"^ckpt_milestone_\d+_score_([-+]?\d+(?:\.\d+)?)\.(?:pt|pth)$"
)


def resolve_best_checkpoint(
    explicit_checkpoint: Optional[str],
    checkpoint_root: str,
) -> str:
    """Resolve an explicit checkpoint or the highest-score milestone below a root."""
    if explicit_checkpoint:
        checkpoint = os.path.abspath(explicit_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    root = os.path.abspath(checkpoint_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    candidates = []
    for current_dir, _, filenames in os.walk(root):
        for filename in filenames:
            match = _CHECKPOINT_SCORE_RE.match(filename)
            if match is None:
                continue
            candidates.append(
                (float(match.group(1)), os.path.join(current_dir, filename))
            )

    if not candidates:
        raise FileNotFoundError(
            f"No scored milestone checkpoint was found below: {root}"
        )

    score, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    print(f"[CHECKPOINT] Auto-selected best fixed checkpoint (score={score:.6f}):")
    print(f"[CHECKPOINT] {checkpoint}")
    return checkpoint


def _to_float(value, index: int = 0) -> Optional[float]:
    """Safely convert tensors/arrays/scalars to python float."""
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            # flatten and pick index
            if value.ndim == 0:
                return float(value.item())
            index = min(index, value.numel() - 1)
            return float(value.reshape(-1)[index].item())
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            try:
                return float(value[index])
            except Exception:
                pass
        return float(value)
    except Exception:
        return None


def _get_attr_value(obj, name: str) -> Optional[torch.Tensor]:
    return getattr(obj, name, None)


def _get_rc_value(reward_components: Dict[str, torch.Tensor], key: str) -> Optional[float]:
    if not isinstance(reward_components, dict):
        return None
    if key not in reward_components:
        return None
    return _to_float(reward_components[key], 0)


def _aggregate(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    if len(clean) == 1:
        single = clean[0]
        return {"mean": single, "std": 0.0, "min": single, "max": single}
    return {
        "mean": mean(clean),
        "std": pstdev(clean),
        "min": min(clean),
        "max": max(clean),
    }


@dataclass
class StepRecord:
    episode: int
    step: int
    reward_mean: Optional[float] = None
    task_reward_mean: Optional[float] = None
    safe_cost_mean: Optional[float] = None
    progress_ratio: Optional[float] = None
    deviation_m: Optional[float] = None
    distance_to_final: Optional[float] = None
    safety_distance: Optional[float] = None
    in_c_zone: Optional[bool] = None
    is_colliding: Optional[bool] = None


@dataclass
class ForceRecord:
    episode: int
    step: int
    fx: float
    fy: float
    fz: float
    f_norm: float


@dataclass
class PositionRecord:
    episode: int
    step: int
    x: float
    y: float
    z: float


@dataclass
class EpisodeSummary:
    episode: int
    episode_length: int
    progress_final: Optional[float]
    score: Optional[float]
    reward_per_step: Optional[float]
    task_return_total: Optional[float]
    safe_cost_total: Optional[float]
    on_track_ratio: Optional[float]
    c_zone_ratio: Optional[float]
    collision_ratio: Optional[float]
    time_to_completion_steps: Optional[int]


class EvalRecorder:
    """
    Collect and persist evaluation metrics.

    Args:
        agent_ids: sequence of agent identifiers
        mode: one of {"rmappo", "maddpg", "epigraph"} controlling score formula
        completion_threshold: distance-to-final threshold (meters) indicating completion
        c_zone_threshold: safety distance threshold (meters) for C zone classification
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        mode: str,
        completion_threshold: float = 0.01,
        c_zone_threshold: float = 0.0075,
    ):
        self.agent_ids = list(agent_ids)
        self.mode = mode.lower()
        self.completion_threshold = completion_threshold
        self.c_zone_threshold = c_zone_threshold

        self._episode_idx: Optional[int] = None
        self._current_steps: List[StepRecord] = []
        self._current_forces: Dict[str, List[ForceRecord]] = {aid: [] for aid in self.agent_ids}
        self._current_positions: List[PositionRecord] = []
        self._current_reward_sum = 0.0
        self._current_task_sum = 0.0
        self._current_safe_sum = 0.0
        self._completion_step: Optional[int] = None

        self.episode_summaries: List[EpisodeSummary] = []
        self.all_step_records: List[StepRecord] = []
        self.all_force_records: Dict[str, List[ForceRecord]] = {aid: [] for aid in self.agent_ids}
        self.all_position_records: List[PositionRecord] = []

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def start_episode(self, episode_idx: int):
        self._episode_idx = episode_idx
        self._current_steps = []
        self._current_forces = {aid: [] for aid in self.agent_ids}
        self._current_positions = []
        self._current_reward_sum = 0.0
        self._current_task_sum = 0.0
        self._current_safe_sum = 0.0
        self._completion_step = None

    def record_step(
        self,
        step_idx: int,
        env,
        rewards: Dict[str, torch.Tensor],
        info: Dict,
        detail: Dict,
    ):
        if self._episode_idx is None:
            raise RuntimeError("start_episode must be called before record_step.")

        actual_env = getattr(env, "unwrapped", env)
        rc = getattr(actual_env, "reward_components", {})

        progress_ratio = _get_rc_value(rc, "progress_ratio")
        deviation = _get_rc_value(rc, "deviation")
        distance_to_final = _get_rc_value(rc, "distance_to_final")

        safety_tensor = _get_attr_value(actual_env, "safety_distances_t1")
        safety_distance = _to_float(safety_tensor, 0)

        colliding_tensor = _get_attr_value(actual_env, "is_violating_t1")
        is_colliding = None
        if colliding_tensor is not None:
            val = _to_float(colliding_tensor, 0)
            if val is not None:
                is_colliding = bool(val > 0.5)

        in_c_zone = None
        if safety_distance is not None:
            in_c_zone = safety_distance <= self.c_zone_threshold

        # Rewards (per-agent)
        reward_vals = []
        for aid in self.agent_ids:
            if aid in rewards:
                reward_vals.append(_to_float(rewards[aid], 0))
        reward_mean = None
        clean_reward_vals = [v for v in reward_vals if v is not None]
        if clean_reward_vals:
            reward_mean = sum(clean_reward_vals) / len(clean_reward_vals)
            self._current_reward_sum += reward_mean

        # Task / safe decomposition (if available)
        task_mean = None
        safe_mean = None
        if isinstance(info, dict):
            if "r_task" in info:
                task_values = []
                for aid in self.agent_ids:
                    if aid in info["r_task"]:
                        task_values.append(_to_float(info["r_task"][aid], 0))
                clean_task = [v for v in task_values if v is not None]
                if clean_task:
                    task_mean = sum(clean_task) / len(clean_task)
                    self._current_task_sum += task_mean
            if "r_safe" in info:
                safe_values = []
                for aid in self.agent_ids:
                    if aid in info["r_safe"]:
                        safe_values.append(_to_float(info["r_safe"][aid], 0))
                clean_safe = [v for v in safe_values if v is not None]
                if clean_safe:
                    safe_mean = sum(clean_safe) / len(clean_safe)
                    self._current_safe_sum += safe_mean

        # Completion detection
        if (
            self._completion_step is None
            and distance_to_final is not None
            and distance_to_final < self.completion_threshold
        ):
            self._completion_step = step_idx

        record = StepRecord(
            episode=self._episode_idx,
            step=step_idx,
            reward_mean=reward_mean,
            task_reward_mean=task_mean,
            safe_cost_mean=safe_mean,
            progress_ratio=progress_ratio,
            deviation_m=deviation,
            distance_to_final=distance_to_final,
            safety_distance=safety_distance,
            in_c_zone=in_c_zone,
            is_colliding=is_colliding,
        )
        self._current_steps.append(record)

        # Force records
        forces_dict = (
            detail.get("force_breakdown")
            or detail.get("applied_forces")
            or detail.get("mean_actions")
            or {}
        )
        force_channels = list(dict.fromkeys([*self.agent_ids, *forces_dict.keys()]))
        for aid in force_channels:
            self._current_forces.setdefault(aid, [])
            fx = fy = fz = f_norm = 0.0
            if aid in forces_dict:
                tensor = forces_dict[aid]
                fx = _to_float(tensor[..., 0], 0) or 0.0
                fy = _to_float(tensor[..., 1], 0) or 0.0
                fz = _to_float(tensor[..., 2], 0) or 0.0
                f_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
            self._current_forces[aid].append(
                ForceRecord(
                    episode=self._episode_idx,
                    step=step_idx,
                    fx=fx,
                    fy=fy,
                    fz=fz,
                    f_norm=f_norm,
                )
            )

        stylus_pos = getattr(actual_env, "stylus_pos_t1", None)
        if isinstance(stylus_pos, torch.Tensor) and stylus_pos.numel() >= 3:
            x = _to_float(stylus_pos[..., 0], 0)
            y = _to_float(stylus_pos[..., 1], 0)
            z = _to_float(stylus_pos[..., 2], 0)
            if None not in (x, y, z):
                self._current_positions.append(
                    PositionRecord(
                        episode=self._episode_idx,
                        step=step_idx,
                        x=x,
                        y=y,
                        z=z,
                    )
                )

    def end_episode(self) -> EpisodeSummary:
        if self._episode_idx is None:
            raise RuntimeError("start_episode must preceed end_episode.")

        episode_length = len(self._current_steps)
        progress_final = None
        if episode_length > 0:
            progress_final = self._current_steps[-1].progress_ratio

        reward_per_step = None
        if episode_length > 0:
            reward_per_step = self._current_reward_sum / episode_length

        task_return_total = self._current_task_sum if self._current_task_sum != 0.0 else None
        safe_cost_total = self._current_safe_sum if self._current_safe_sum != 0.0 else None

        score = None
        if episode_length > 0:
            if self.mode == "epigraph":
                combined = 0.0
                for rec in self._current_steps:
                    task = rec.task_reward_mean or 0.0
                    safe = rec.safe_cost_mean or 0.0
                    combined += (task - safe)
                score = 1000.0 * combined / episode_length
            else:
                score = 1000.0 * self._current_reward_sum / episode_length

        def _ratio(predicate) -> Optional[float]:
            if episode_length == 0:
                return None
            matches = [1 for rec in self._current_steps if predicate(rec)]
            if not matches:
                return 0.0
            return len(matches) / episode_length

        on_track_ratio = _ratio(lambda rec: rec.deviation_m is not None and abs(rec.deviation_m) < 0.01)
        c_zone_ratio = _ratio(lambda rec: rec.in_c_zone is True)
        collision_ratio = _ratio(lambda rec: rec.is_colliding is True)

        summary = EpisodeSummary(
            episode=self._episode_idx,
            episode_length=episode_length,
            progress_final=progress_final,
            score=score,
            reward_per_step=reward_per_step,
            task_return_total=task_return_total,
            safe_cost_total=safe_cost_total,
            on_track_ratio=on_track_ratio,
            c_zone_ratio=c_zone_ratio,
            collision_ratio=collision_ratio,
            time_to_completion_steps=self._completion_step,
        )

        # Persist current episode data to global storage
        self.episode_summaries.append(summary)
        self.all_step_records.extend(self._current_steps)
        for aid, records in self._current_forces.items():
            self.all_force_records.setdefault(aid, []).extend(records)
        self.all_position_records.extend(self._current_positions)

        # Reset episode state
        self._episode_idx = None
        self._current_steps = []
        self._current_forces = {aid: [] for aid in self.agent_ids}
        self._current_positions = []
        self._current_reward_sum = 0.0
        self._current_task_sum = 0.0
        self._current_safe_sum = 0.0
        self._completion_step = None

        return summary

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, "eval_summary.yaml")
        steps_path = os.path.join(save_dir, "eval_steps.csv")

        summary_payload = {
            "mode": self.mode,
            "num_episodes": len(self.episode_summaries),
            "episodes": [summary.__dict__ for summary in self.episode_summaries],
            "aggregated": self._build_aggregates(),
            "paper_metrics": self.build_paper_metrics(),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary_payload, f, sort_keys=False, allow_unicode=True)

        with open(steps_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "step",
                    "reward_mean",
                    "task_reward_mean",
                    "safe_cost_mean",
                    "progress_ratio",
                    "deviation_m",
                    "distance_to_final",
                    "safety_distance",
                    "in_c_zone",
                    "is_colliding",
                ]
            )
            for rec in self.all_step_records:
                writer.writerow(
                    [
                        rec.episode,
                        rec.step,
                        _format(rec.reward_mean),
                        _format(rec.task_reward_mean),
                        _format(rec.safe_cost_mean),
                        _format(rec.progress_ratio),
                        _format(rec.deviation_m),
                        _format(rec.distance_to_final),
                        _format(rec.safety_distance),
                        _bool_to_int(rec.in_c_zone),
                        _bool_to_int(rec.is_colliding),
                    ]
                )

        for aid, force_records in self.all_force_records.items():
            if not force_records:
                continue
            force_path = os.path.join(save_dir, f"forces_{aid}.csv")
            with open(force_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "step", "fx", "fy", "fz", "f_norm"])
                for record in force_records:
                    writer.writerow(
                        [
                            record.episode,
                            record.step,
                            _format(record.fx),
                            _format(record.fy),
                            _format(record.fz),
                            _format(record.f_norm),
                        ]
                    )

        if self.all_position_records:
            positions_path = os.path.join(save_dir, "eval_positions.csv")
            with open(positions_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "step", "x", "y", "z"])
                for record in self.all_position_records:
                    writer.writerow(
                        [
                            record.episode,
                            record.step,
                            _format(record.x),
                            _format(record.y),
                            _format(record.z),
                        ]
                    )

    def _build_aggregates(self) -> Dict[str, Dict[str, float]]:
        keys = [
            "score",
            "episode_length",
            "progress_final",
            "reward_per_step",
            "task_return_total",
            "safe_cost_total",
            "on_track_ratio",
            "c_zone_ratio",
            "collision_ratio",
            "time_to_completion_steps",
        ]
        aggregates = {}
        for key in keys:
            values = [getattr(summary, key) for summary in self.episode_summaries]
            agg = _aggregate(values)
            if agg is not None:
                aggregates[key] = agg
        return aggregates

    def build_paper_metrics(self) -> Dict[str, float]:
        """Return the six scalar metrics used in the fixed-human comparison table."""
        metric_values = {
            "P_fin": [
                summary.progress_final for summary in self.episode_summaries
            ],
            "Steps": [
                float(summary.episode_length) for summary in self.episode_summaries
            ],
            "rho_track": [
                summary.on_track_ratio for summary in self.episode_summaries
            ],
            "bar_R": [summary.score for summary in self.episode_summaries],
            "rho_C": [summary.c_zone_ratio for summary in self.episode_summaries],
            "rho_col": [
                summary.collision_ratio for summary in self.episode_summaries
            ],
        }
        result: Dict[str, float] = {}
        for key, values in metric_values.items():
            stats = _aggregate(values)
            if stats is not None:
                result[key] = stats["mean"]
        return result


def print_paper_metrics(recorder: EvalRecorder):
    """Print the compact metric block used by the paper."""
    metrics = recorder.build_paper_metrics()
    if not metrics:
        return

    print("\n[PAPER METRICS]")
    labels = (
        ("P_fin", "P_fin   [up]"),
        ("Steps", "Steps   [down]"),
        ("rho_track", "rho_track [up]"),
        ("bar_R", "bar_R   [up]"),
        ("rho_C", "rho_C   [down]"),
        ("rho_col", "rho_col [down]"),
    )
    for key, label in labels:
        value = metrics.get(key)
        if value is None:
            continue
        print(f"  {label}: {value:.6f}")


def _format(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return float(value)


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return int(bool(value))


# ---------------------------------------------------------------------- #
# Optional WandB integration (keep lazy import to avoid hard dependency)
# ---------------------------------------------------------------------- #


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except Exception:
        return False
    return True


class EvalWandBLogger:
    """Minimal WandB helper for evaluation logging."""

    def __init__(self, project: str = "evaluation", run_name: Optional[str] = None, config: Optional[Dict] = None):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config, settings=wandb.Settings(start_method="thread"))

    def log_step(self, step_idx: int, payload: Dict[str, float]):
        self.wandb.log(payload, step=step_idx)

    def log_episode(self, episode_idx: int, payload: Dict[str, float]):
        self.wandb.log(payload, step=episode_idx)

    def log_table(self, name: str, columns: List[str], data: List[List]):
        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({name: table})

    def finish(self):
        self.wandb.finish()
