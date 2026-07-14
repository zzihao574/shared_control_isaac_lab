#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Features unified training execution, milestone evaluation, and optimized WandB logging.
"""

import argparse
import copy
import os
import yaml
import torch
import numpy as np
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, DefaultDict, Deque
from collections import defaultdict, deque


@dataclass(frozen=True)
class SeedPlan:
    """Single source of truth for MADDPG random-stream derivation."""

    base_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_seed", int(self.base_seed))

    def network_seed(self) -> int:
        return self.base_seed % (2**32)

    def replay_seed(self) -> int:
        return (self.base_seed + 424242) % (2**32)

    def exploration_seed(self, agent_index: int, env_id: int) -> int:
        return (
            self.base_seed
            + 1234567
            + int(agent_index) * 10007
            + int(env_id) * 97
        ) % (2**32)

    def apply_network_seed(self) -> None:
        """Seed PyTorch immediately before deterministic network construction."""
        torch.manual_seed(self.network_seed())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.network_seed())

    def make_replay_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.replay_seed())
        return generator

    def make_exploration_generator(
        self,
        device: torch.device | str,
        agent_index: int,
        env_id: int,
    ) -> torch.Generator:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(self.exploration_seed(agent_index, env_id))
        return generator


def setup_global_reproducibility(
    seed: int, strict_determinism: bool = True
) -> None:
    """Seed process-level RNGs before Isaac Sim is launched."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[SEED] Strict determinism enabled (may slow down training)")

    print(f"[SEED] Global reproducibility set: seed={seed}")


def resolve_startup_seed(args) -> int:
    """Resolve the seed before AppLauncher from checkpoint, CLI, or YAML."""
    cli_seed = getattr(args, "seed", None)
    checkpoint_path = getattr(args, "checkpoint", None)

    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_params = checkpoint.get("params", {})
        checkpoint_seed = int(checkpoint_params.get("seed", 42))
        if cli_seed is not None and int(cli_seed) != checkpoint_seed:
            raise ValueError(
                "Cannot resume with a different seed than the checkpoint: "
                f"checkpoint={checkpoint_seed}, CLI={cli_seed}"
            )
        return checkpoint_seed

    if cli_seed is not None:
        return int(cli_seed)

    with open(args.config, "r", encoding="utf-8") as config_file:
        params = yaml.safe_load(config_file) or {}
    return int(params.get("seed", 42))


def build_maddpg_wandb_config(
    resolved_config: Dict[str, Any],
    runtime: Dict[str, Any],
    human_metadata: Dict[str, Any],
    git_commit: str,
) -> Dict[str, Any]:
    """Build nested and query-friendly WandB metadata from resolved values."""
    run_config = copy.deepcopy(resolved_config)
    termination = resolved_config.get("termination_conditions", {})
    constraints = resolved_config.get("constraints", {})
    human_model_type = str(resolved_config.get("human_model_type", "learnable"))

    run_config.update(
        {
            "algorithm": "maddpg",
            "num_envs": int(runtime["num_envs"]),
            "max_global_steps": int(runtime["max_global_steps"]),
            "experiment/algorithm": "maddpg",
            "experiment/human_model_type": human_model_type,
            "experiment/seed": int(runtime["seed"]),
            "robot/max_force_per_axis": float(
                constraints.get("max_robot_force", 0.04)
            ),
            "termination/z_below_zero": bool(
                termination.get("z_below_zero", False)
            ),
            "termination/edge_collision": bool(
                termination.get("edge_collision", True)
            ),
            "termination/safety_distance_threshold": float(
                termination.get("safety_distance_threshold", 0.0)
            ),
            "git_commit": git_commit,
        }
    )
    run_config.update(copy.deepcopy(human_metadata))
    return run_config

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")


class TrainingRunner:
    """
    Unified training loop executor with noise scheduling and metrics collection.
    Features exponential noise decay and unified global step tracking.
    """
    
    def __init__(self, env, maddpg, metrics_hub, agent_ids, max_global_steps=None):
        self.env = env
        self.maddpg = maddpg
        self.metrics = metrics_hub
        self.agent_ids = agent_ids
        self.global_step = 0  # Current training step
        self.global_episodes = 0  # Total episodes completed
        self._skip_episode_once = False  # Flag to skip episode counting once
        self._current_obs = None  # Current observations cache
        
        # Load exploration parameters for noise scheduling
        expl = self.maddpg.params.get("exploration", {})
        self.sigma_start = float(expl.get("sigma_start", 0.7))
        self.sigma_end = float(expl.get("sigma_end", 0.1))
        self.decay_k = float(expl.get("decay_k", 6.0))
        
        # Use the max_global_steps passed from trainer (already resolved CLI vs YAML priority)
        if max_global_steps is not None and max_global_steps > 0:
            self.max_global_steps = int(max_global_steps)
        else:
            # Fallback to YAML config if not provided
            maddpg_cfg = self.maddpg.params.get('maddpg_config', {})
            self.max_global_steps = int(maddpg_cfg.get('max_global_steps', 200000))
        
        print(f"[NOISE SCHEDULE] Configured exponential decay:")
        print(f"  Start: {self.sigma_start}, End: {self.sigma_end}, k: {self.decay_k}")
        print(f"  Max steps for noise decay: {self.max_global_steps}")

    def _calculate_noise_scale(self) -> float:
        """Calculate current noise scaling factor using exponential decay."""
        if self.max_global_steps <= 0:
            return self.sigma_start
            
        ratio = min(1.0, float(self.global_step) / float(max(1, self.max_global_steps)))
        noise_scale = self.sigma_end + (self.sigma_start - self.sigma_end) * math.exp(-self.decay_k * ratio)
        
        return noise_scale

    def execute_training_step(self):
        """Execute one training step with noise scheduling and metrics collection."""
        # Use current observations
        current_obs = self._current_obs
        if current_obs is None:
            if hasattr(self.env, "_get_observations"):
                current_obs = self.env._get_observations()
            else:
                current_obs, _ = self.env.reset()
            self._current_obs = current_obs
            
        # Calculate noise scaling factor
        noise_scale = self._calculate_noise_scale()
        
        # Select actions with noise and global noise scheduling
        actions, detail = self.maddpg.select_actions(current_obs, add_noise=True, noise_scale=noise_scale)

        # Let environment record actor detail info
        self.env.unwrapped.set_detail_actor_info(detail)

        # Environment interaction
        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)

        # Store joint transitions
        done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
        
        self.maddpg.add_experience_to_buffer(
            obs=current_obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=done_any_dict
        )

        # Update networks
        stats = self.maddpg.update()

        # Count episodes using OR aggregation
        done_any = None
        for aid in self.agent_ids:
            d = done_any_dict[aid].to(torch.bool)
            done_any = d if done_any is None else (done_any | d)
        
        episode_increment = int(done_any.sum().item())
        if self._skip_episode_once:
            episode_increment = 0
            self._skip_episode_once = False
        self.global_episodes += episode_increment

        # Unified logging with noise scheduling information
        if stats and (stats.get("training/critic_updates", 0) > 0 or stats.get("training/actor_updates", 0) > 0):
            payload = {
                # Pass per-agent data directly to WandB (not averaged)
                "loss/actor": stats.get("loss/actor"),
                "loss/critic": stats.get("loss/critic"),
                "q_mean": stats.get("q_mean"),
                "q_std": stats.get("q_std"),
                "q_target_mean": stats.get("q_target_mean"),
                "q_target_std": stats.get("q_target_std"),
                "grad_norm/actor": stats.get("grad_norm/actor"),
                "grad_norm/critic": stats.get("grad_norm/critic"),
                # Keep global metrics
                "replay/buffer_size": len(self.maddpg.replay) if hasattr(self.maddpg, "replay") else None,
                "train/episodes_done": self.global_episodes,
                "training/critic_updates": stats.get("training/critic_updates"),
                "training/actor_updates": stats.get("training/actor_updates"),
                "exploration/noise_scale": noise_scale,
            }
            # Clean None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            self.metrics.push_update(self.global_step, payload)

        # Push force statistics every 10 steps
        if self.global_step % 10 == 0:
            actual_env = getattr(self.env, "unwrapped", self.env)
            if hasattr(actual_env, "get_force_breakdown"):
                detail["force_breakdown"] = actual_env.get_force_breakdown()
            self._push_current_step_force_statistics(detail)

        # Step counting and environment synchronization
        self.global_step += 1
        actual_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(actual_env, "set_trainer_global_step"):
            actual_env.set_trainer_global_step(self.global_step)

        # Update current observations for next round
        self._current_obs = next_obs

        return next_obs
 
    def mark_skip_episode_once(self):
        """Mark to skip episode counting once for milestone evaluation."""
        self._skip_episode_once = True

    def _push_current_step_force_statistics(self, detail):
        """Push current step's force statistics to WandB."""
        force_payload = {}

        breakdown = detail.get("force_breakdown", {}) if isinstance(detail, dict) else {}
        if not breakdown:
            source = detail.get("mean_actions", {}) if isinstance(detail, dict) else {}
            breakdown = {k: v for k, v in source.items() if k in {"human", "robot"}}

        for channel in (
            "human_impedance",
            "human_residual",
            "human",
            "robot",
            "combined",
        ):
            forces = breakdown.get(channel)
            if forces is None:
                continue
            mean_xyz = forces.mean(dim=0)
            norms = torch.linalg.vector_norm(forces, dim=-1)
            force_payload.update({
                f"forces/{channel}_fx_mean": float(mean_xyz[0].item()),
                f"forces/{channel}_fy_mean": float(mean_xyz[1].item()),
                f"forces/{channel}_fz_mean": float(mean_xyz[2].item()),
                f"forces/{channel}_norm_mean": float(norms.mean().item()),
                f"forces/{channel}_norm_rms": float(torch.sqrt(torch.mean(norms.square())).item()),
            })

        residual = breakdown.get("human_residual")
        if residual is not None:
            residual_norm = torch.linalg.vector_norm(residual, dim=-1)
            force_payload["forces/human_residual_norm_p95"] = float(
                torch.quantile(residual_norm, 0.95).item()
            )

        human = breakdown.get("human")
        if human is not None:
            max_human_force = float(
                self.maddpg.params.get("constraints", {}).get("max_human_force", 0.04)
            )
            saturated = torch.isclose(
                human.abs(),
                torch.tensor(max_human_force, device=human.device, dtype=human.dtype),
                atol=1e-6,
                rtol=0.0,
            )
            force_payload["forces/human_saturation_fraction"] = float(
                saturated.float().mean().item()
            )

        if all(name in breakdown for name in ("human", "robot", "combined")):
            composition_error = breakdown["combined"] - (
                breakdown["human"] + breakdown["robot"]
            )
            force_payload["forces/composition_error_max"] = float(
                composition_error.abs().max().item()
            )
            force_payload["forces/composition_error_mean"] = float(
                composition_error.abs().mean().item()
            )

        if self.maddpg.human_model_type == "fixed_impedance":
            force_payload["human/fixed_actor_checksum"] = self.maddpg.human_actor_checksum()
        
        # Push to MetricsHub for WandB logging
        if force_payload:
            self.metrics.push_update(self.global_step, force_payload)

class MilestoneEvaluator:
    """
    Milestone evaluator with single environment evaluation.
    Saves unified checkpoints compatible with resume/play.
    """
    
    def __init__(self, env, maddpg, metrics_hub, log_dir, agent_ids, runner=None):
        self.env = env
        self.maddpg = maddpg
        self.metrics = metrics_hub
        self.log_dir = log_dir
        self.agent_ids = agent_ids
        self.runner = runner
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run_evaluation(self, milestone: int, global_step: int) -> dict:
        """Handle milestone evaluation and model saving."""
        # In-place evaluation with return normalization
        return_norm, num_eps = self._run_single_evaluation_episode()
        
        # Scale return_norm by 1000 for milestone tracking
        milestone_return = return_norm * 1000
        
        print(f"[EVAL] Milestone {milestone}: return_norm={return_norm:.4f}, scaled={milestone_return:.2f}")

        ckpt_path = save_milestone_checkpoint_maddpg(
            self.maddpg,
            self.runner,
            milestone_return,
            milestone,
            self.checkpoint_dir,
        )

        # Push milestone logs with return_norm*1000 to replace previous return metrics
        payload = {
            "eval/return_mean": float(milestone_return),  # This maps to milestone/actor_return
            "eval/num_episodes": int(num_eps),
            "milestone/latest_completed": int(milestone),
            # Additional debug info
            "eval/return_norm": float(return_norm),  # Original normalized return for reference
            "milestone/checkpoint_path": ckpt_path,
        }
        self.metrics.push_milestone(global_step, milestone, payload)
        
        print(f"[EVAL] Uploaded milestone metrics: scaled_return={milestone_return:.2f}")

        return {"skip_episode_once": True}

    def _run_single_evaluation_episode(self):
        """Run env0 evaluation and always restore normal parallel physics."""
        active_env = 0
        env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(env, "set_evaluation_active_env"):
            raise RuntimeError(
                "Environment does not support single-environment physical evaluation"
            )

        had_trainer_step = hasattr(env, "_trainer_global_step")
        training_global_step = getattr(env, "_trainer_global_step", None)
        env.set_evaluation_active_env(active_env)
        try:
            return self._run_active_evaluation_episode(env, active_env)
        finally:
            if had_trainer_step:
                env.set_trainer_global_step(training_global_step)
            env.clear_evaluation_active_env()
            env.reset()
            print("[EVAL] Environment reset back to parallel training mode")

    def _run_active_evaluation_episode(self, env, active_env: int):
        """Evaluate and record only the selected physical environment."""
        target_episodes = 1

        print(f"[EVAL] Starting in-place evaluation (env{active_env} only, 1 episode)...")

        obs, _ = env.reset()
        print(f"[EVAL] Environment reset for independent evaluation")
        
        num_envs = len(obs[self.agent_ids[0]])
        ep_returns = torch.zeros(num_envs, device='cuda' if torch.cuda.is_available() else 'cpu')
        ep_steps = torch.zeros(num_envs, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
        completed_return_norms = []
        
        eval_step_counter = 0
        
        with torch.no_grad():
            while len(completed_return_norms) < target_episodes:
                # Get current observations
                if hasattr(env, '_get_observations'):
                    current_obs = env._get_observations()
                elif hasattr(env, 'observation_manager'):
                    current_obs = env.observation_manager.compute()
                else:
                    current_obs = obs
                
                # Select actions deterministically (no noise during evaluation)
                actions, detail_info = self.maddpg.select_actions(current_obs, add_noise=False, noise_scale=0.0)

                # The environment owns the physical single-env force mask. This
                # keeps fixed/residual impedance and policy forces consistent.
                env.set_detail_actor_info(detail_info)

                # Reuse the environment's normal StepTracer path. Starting from
                # zero matches the training cadence (0, 10, 20, ... by default).
                env.set_trainer_global_step(eval_step_counter)
                obs, rewards, terminated, truncated, _ = env.step(actions)
                eval_step_counter += 1
                
                # Accumulate env0 rewards and steps
                step_rewards = torch.stack([rewards[aid] for aid in self.agent_ids])
                avg_step_rewards = step_rewards.mean(dim=0)
                ep_returns[active_env] += avg_step_rewards[active_env]
                ep_steps[active_env] += 1
                
                # Check if env0 is complete
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                done_any = None
                for aid in self.agent_ids:
                    d = done_any_dict[aid].to(torch.bool)
                    done_any = d if done_any is None else (done_any | d)
                
                if done_any[active_env]:
                    total_reward = float(ep_returns[active_env].item())
                    total_steps = int(ep_steps[active_env].item())
                    
                    # Calculate return_norm = total_reward / total_steps
                    return_norm = total_reward / max(1, total_steps)
                    completed_return_norms.append(return_norm)
                    
                    print(f"[EVAL] Episode completed: steps={total_steps}, total_reward={total_reward:.3f}, return_norm={return_norm:.4f}")
                    
                    # Reset counters for next episode (though we only do 1)
                    ep_returns[active_env] = 0.0
                    ep_steps[active_env] = 0
                    
                    if len(completed_return_norms) >= target_episodes:
                        break
        
        final_return_norms = completed_return_norms[:target_episodes]
        avg_return_norm = sum(final_return_norms) / max(1, len(final_return_norms))
        
        print(f"[EVAL] Completed: {len(final_return_norms)} episodes, Average return_norm: {avg_return_norm:.4f}")

        return avg_return_norm, len(final_return_norms)

class MetricsHub:
    """
    Simplified single-exit metrics bus for unified data pipeline.
    Features event-based subscription system and ring buffer for update history.
    """
    
    def __init__(self, ring: int = 100):
        self.subs: DefaultDict[str, list[Callable[[dict], None]]] = defaultdict(list)  # Event subscribers
        self.update_ring: Deque[dict] = deque(maxlen=ring)  # Update history ring buffer

    def subscribe(self, event: str, handler: Callable[[dict], None]) -> None:
        """Subscribe to an event type with a handler function."""
        self.subs[event].append(handler)

    def _emit(self, event: str, payload: dict) -> None:
        """Emit an event to all subscribers."""
        for h in self.subs.get(event, []):
            h(payload)

    def push_update(self, step: int, stats: dict) -> None:
        """Push training update statistics."""
        if not stats:
            return
        data = {"step": step, **stats}
        self.update_ring.append(data)
        self._emit("update", data)

    def push_milestone(self, step: int, milestone: int, summary: dict) -> None:
        """Push milestone completion summary."""
        self._emit("milestone_summary", {"step": step, "milestone": milestone, **summary})


class WandBLogger:
    """
    Optimized WandB logger with consolidated metric mapping dictionaries.
    Features network configuration logging and milestone tracking.
    """
    
    # Class-level mapping dictionaries for cleaner code
    AGENT_METRICS_MAP = {
        'loss/actor': 'train/{}/actor_loss',
        'loss/critic': 'train/{}/critic_loss', 
        'q_mean': 'model/{}/q_mean',
        'q_std': 'model/{}/q_std',
        'q_target_mean': 'model/{}/q_target_mean',
        'q_target_std': 'model/{}/q_target_std',
        'grad_norm/actor': 'model/{}/grad_norm_actor',
        'grad_norm/critic': 'model/{}/grad_norm_critic',
    }
    
    GLOBAL_METRICS_MAP = {
        "exploration/noise_scale": "exploration/noise_scale",
        "train/episodes_done": "train/global_episodes", 
        "replay/buffer_size": "replay/buffer_size",
        "training/critic_updates": "train/critic_updates",
        "training/actor_updates": "train/actor_updates",
        "eval/return_mean": "milestone/actor_return",
        "milestone/latest_completed": "milestone/latest_completed",
        # Force statistics
        "forces/robot_fx_mean": "forces/robot_fx_mean",
        "forces/robot_fy_mean": "forces/robot_fy_mean",
        "forces/robot_fz_mean": "forces/robot_fz_mean",
        "forces/human_fx_mean": "forces/human_fx_mean",
        "forces/human_fy_mean": "forces/human_fy_mean", 
        "forces/human_fz_mean": "forces/human_fz_mean",
    }
    
    def __init__(self, project_name: str = "surgical_robot_maddpg", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE  # WandB availability flag
        self.project_name = project_name
        self.run = None
        
        if not self.enabled:
            print("[WANDB] Disabled")

    def initialize_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """Initialize WandB run with enhanced configuration tracking."""
        if not self.enabled:
            return
        
        # Direct initialization - fail fast if there's a problem
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config,
            tags=[
                "maddpg",
                "multi-agent",
                "surgical-robot",
                str(config.get("human_model_type", "learnable")),
                "async-updates",
            ],
            notes="MADDPG shared-control experiment with configurable human force model",
            settings=wandb.Settings(start_method="thread")
        )
        
        # Log key configuration for dashboard
        networks_cfg = config.get("networks", {})
        exploration_cfg = config.get("exploration", {})
        maddpg_cfg = config.get("maddpg_config", {})
        
        wandb.config.update({
            "update_interval": maddpg_cfg.get("update_interval", 100),
            "critic_update_interval": maddpg_cfg.get("update_interval", 100),
            "actor_update_interval": maddpg_cfg.get("update_interval", 100) * 2,  # Actor updates 2x slower
            "agent_mode": (
                "robot_only" if config.get("human_model_type") == "fixed_impedance"
                else "human_and_robot"
            ),
            "reward_components": "trajectory+progress+potential_field",
            "completion_threshold": config.get("reward_parameters", {}).get("completion_threshold", 0.01),
            # Network configuration
            "actor_layers": networks_cfg.get("actor", {}).get("hidden_layers", []),
            "critic_layers": networks_cfg.get("critic", {}).get("hidden_layers", []),
            "actor_bypass_layers": networks_cfg.get("actor", {}).get("input_bypass_layers", []),
            "critic_bypass_layers": networks_cfg.get("critic", {}).get("input_bypass_layers", []),
            "orthogonal_init": networks_cfg.get("actor", {}).get("orthogonal_init", False),
            # Exploration configuration
            "noise_sigma_start": exploration_cfg.get("sigma_start", 0.7),
            "noise_sigma_end": exploration_cfg.get("sigma_end", 0.1),
            "noise_decay_k": exploration_cfg.get("decay_k", 6.0),
        })
        
        print(f"[WANDB] Successfully initialized: {self.run.name}")

    def attach_metrics_hub(self, hub: "MetricsHub"):
        """Attach to MetricsHub for unified data pipeline."""
        if not self.enabled:
            return

        # Subscribe to training update events
        hub.subscribe("update", lambda data: self.log_metrics(data, data["step"]))

        # Subscribe to milestone completion events
        def _on_ms(ms):
            step = ms.get("step", 0)
            payload_to_log = {}
            if "eval/return_mean" in ms:
                payload_to_log["eval/return_mean"] = ms["eval/return_mean"]
            if "milestone" in ms:
                payload_to_log["milestone/latest_completed"] = ms["milestone"]

            if payload_to_log:
                self.log_metrics(payload_to_log, step)

        hub.subscribe("milestone_summary", _on_ms)
        print("[WANDB] Attached to MetricsHub with optimized metric mapping.")

    def log_metrics(self, metrics_data: Dict[str, Any], step: int) -> None:
        """Log metrics with consolidated mapping dictionaries."""
        if not self.enabled or not metrics_data:
            return

        log_data = {}

        # Handle Per-Agent training metrics using consolidated mapping
        if any(key in metrics_data and isinstance(metrics_data.get(key), dict) 
               for key in self.AGENT_METRICS_MAP.keys()):
            
            # Get agent IDs from first available per-agent metric
            agent_ids = None
            for source_key in self.AGENT_METRICS_MAP.keys():
                if source_key in metrics_data and isinstance(metrics_data.get(source_key), dict):
                    agent_ids = list(metrics_data[source_key].keys())
                    break
            
            if agent_ids:
                # Apply consolidated per-agent mapping
                for source_key, target_pattern in self.AGENT_METRICS_MAP.items():
                    if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                        for agent_id in agent_ids:
                            if agent_id in metrics_data[source_key]:
                                log_data[target_pattern.format(agent_id)] = metrics_data[source_key][agent_id]

        # Handle global metrics using consolidated mapping
        for src_key, dest_key in self.GLOBAL_METRICS_MAP.items():
            if src_key in metrics_data and metrics_data[src_key] is not None:
                log_data[dest_key] = metrics_data[src_key]

        # Human-model force channels are intentionally extensible. Passing
        # them through avoids maintaining a brittle one-key-at-a-time map.
        for key, value in metrics_data.items():
            if key.startswith(("forces/", "human/")) and value is not None:
                try:
                    log_data.setdefault(key, float(value))
                except (TypeError, ValueError):
                    pass

        if log_data:
            wandb.log(log_data, step=step)

    def finalize_run(self) -> None:
        """Finalize WandB run."""
        if self.enabled and self.run:
            wandb.finish()
            print("[WANDB] Run finished")
            

class TrainingConfiguration:
    """Training configuration loader and parameter manager."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.params = yaml.safe_load(f)
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Create configuration from YAML file."""
        return cls(config_path)
    
    def get_compute_device(self) -> str:
        """Get compute device (CUDA if available)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_milestone_checkpoint_maddpg(
    maddpg,
    runner,
    score: float,
    milestone: int,
    ckpt_dir: str,
):
    """Save flat dual-network checkpoint for MADDPG (resume + evaluation)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = f"ckpt_milestone_{milestone:06d}_score_{score:.6f}.pth"
    fpath = os.path.join(ckpt_dir, fname)

    checkpoint = _build_maddpg_checkpoint(
        maddpg,
        runner,
        score=float(score),
        milestone=int(milestone),
    )

    torch.save(checkpoint, fpath)
    _append_milestone_index_maddpg(
        os.path.join(ckpt_dir, "milestones_index.txt"),
        milestone,
        score,
        fname,
    )
    print(f"[CKPT] Saved milestone {milestone} (score={score:.4f}) -> {fpath}")
    return fpath


def save_final_checkpoint_maddpg(maddpg, runner, ckpt_dir: str) -> str:
    """Save a resume-compatible checkpoint at the configured training limit."""
    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = int(getattr(runner, "global_step", 0))
    fpath = os.path.join(ckpt_dir, f"final_step_{global_step:09d}.pth")
    checkpoint = _build_maddpg_checkpoint(maddpg, runner)
    torch.save(checkpoint, fpath)
    print(f"[CKPT] Saved final checkpoint -> {fpath}")
    return fpath


def _build_maddpg_checkpoint(maddpg, runner, **extra_fields) -> Dict[str, Any]:
    """Build the common resume-compatible MADDPG checkpoint payload."""
    checkpoint = {
        "algorithm": "maddpg_shared",
        "agent_ids": maddpg.agent_ids,
        "params": maddpg.params,
        "global_steps_total": int(getattr(runner, "global_step", 0)),
        "episodes_done_total": int(getattr(runner, "global_episodes", 0)),
        "training_steps_total": int(getattr(maddpg, "training_steps", 0)),
        "actor_update_count": int(getattr(maddpg, "actor_update_count", 0)),
        "critic_update_count": int(getattr(maddpg, "critic_update_count", 0)),
        "human_model_type": getattr(maddpg, "human_model_type", "learnable"),
        "human_impedance": maddpg.params.get("human_impedance", {}),
        "force_limit_semantics": "per_axis",
        "git_commit": maddpg.params.get("git_commit", "unknown"),
        "human_actor_checksum": maddpg.human_actor_checksum(),
    }
    checkpoint.update(extra_fields)

    optim_state = {}
    for agent_id in maddpg.agent_ids:
        agent = maddpg.agents[agent_id]
        checkpoint[f"{agent_id}_actor"] = agent.actor.state_dict()
        checkpoint[f"{agent_id}_critic"] = agent.critic.state_dict()
        checkpoint[f"{agent_id}_actor_target"] = agent.actor_target.state_dict()
        checkpoint[f"{agent_id}_critic_target"] = agent.critic_target.state_dict()
        optim_state[f"{agent_id}_actor"] = agent.actor_optimizer.state_dict()
        optim_state[f"{agent_id}_critic"] = agent.critic_optimizer.state_dict()
    checkpoint["optim_state"] = optim_state
    checkpoint["rng_state"] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return checkpoint


def _append_milestone_index_maddpg(index_path: str, milestone: int, score: float, fname: str):
    record = f"{datetime.now().isoformat()}\tmilestone={milestone:06d}\tscore={score:.6f}\tpath={fname}"
    records: List[str] = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)
    records.append(record)

    def _milestone_key(entry: str) -> int:
        for part in entry.split("\t"):
            if part.startswith("milestone="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return 0
        return 0

    records_sorted = sorted(records, key=_milestone_key)
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records_sorted:
            f.write(rec + "\n")


def resume_from_checkpoint_maddpg(path: str, maddpg, runner=None, device: Optional[str] = None):
    """Resume MADDPG training from flat dual checkpoint."""
    print(f"[RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    agent_ids = maddpg.agent_ids
    for agent_id in agent_ids:
        agent = maddpg.agents[agent_id]
        agent.actor.load_state_dict(ckpt[f"{agent_id}_actor"])
        agent.critic.load_state_dict(ckpt[f"{agent_id}_critic"])
        if f"{agent_id}_actor_target" in ckpt and f"{agent_id}_critic_target" in ckpt:
            agent.actor_target.load_state_dict(ckpt[f"{agent_id}_actor_target"])
            agent.critic_target.load_state_dict(ckpt[f"{agent_id}_critic_target"])
        if device is not None:
            agent.actor.to(device)
            agent.critic.to(device)
            agent.actor_target.to(device)
            agent.critic_target.to(device)
    print("[RESUME] Loaded actor/critic weights (including targets).")

    if "optim_state" in ckpt:
        for agent_id in agent_ids:
            agent = maddpg.agents[agent_id]
            if f"{agent_id}_actor" in ckpt["optim_state"]:
                agent.actor_optimizer.load_state_dict(ckpt["optim_state"][f"{agent_id}_actor"])
            if f"{agent_id}_critic" in ckpt["optim_state"]:
                agent.critic_optimizer.load_state_dict(ckpt["optim_state"][f"{agent_id}_critic"])
        print("[RESUME] Optimizer states restored.")

    if "params" in ckpt and isinstance(ckpt["params"], dict):
        checkpoint_mode = str(ckpt["params"].get("human_model_type", "learnable"))
        current_mode = str(maddpg.params.get("human_model_type", "learnable"))
        if checkpoint_mode != current_mode:
            raise ValueError(
                "Checkpoint human_model_type does not match the resolved training "
                f"configuration: checkpoint={checkpoint_mode}, current={current_mode}"
            )
        # The trainer reconstructs and resolves checkpoint params before creating
        # MADDPG.  Keep that resolved object here so explicit runtime overrides
        # (for example a larger max_global_steps) are not silently discarded.
        print("[RESUME] Checkpoint human model metadata validated.")

    maddpg.training_steps = int(ckpt.get("training_steps_total", getattr(maddpg, "training_steps", 0)))
    maddpg.actor_update_count = int(ckpt.get("actor_update_count", getattr(maddpg, "actor_update_count", 0)))
    maddpg.critic_update_count = int(ckpt.get("critic_update_count", getattr(maddpg, "critic_update_count", 0)))

    if runner is not None:
        runner.global_step = int(ckpt.get("global_steps_total", runner.global_step))
        runner.global_episodes = int(ckpt.get("episodes_done_total", runner.global_episodes))
        print(f"[RESUME] Runner counters -> steps={runner.global_step} episodes={runner.global_episodes}")

    if "rng_state" in ckpt:
        random.setstate(ckpt["rng_state"]["py"])
        np.random.set_state(ckpt["rng_state"]["np"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        if torch.cuda.is_available() and ckpt["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
        print("[RESUME] RNG states restored.")

    print("[RESUME] MADDPG checkpoint restoration complete.")


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser with residual networks support."""
    if config_path is None:
        # Adjust path based on actual file location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params_maddpg.yaml')

    parser = argparse.ArgumentParser(description="MADDPG multi-environment parallel training with residual networks")
    parser.add_argument("--config", type=str, default=config_path)
    
    # Environment configuration
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (new-run default: 512; resume: checkpoint value)",
    )
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: use resolved config/checkpoint seed)")
    parser.add_argument(
        "--human_model_type",
        choices=("learnable", "fixed_impedance", "residual_impedance"),
        default=None,
        help="Override the human force model for a new run",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Explicit output directory for resolved config, checkpoints, and logs",
    )
    
    # Training termination - removed default value to allow proper priority handling in trainer
    parser.add_argument(
        "--max_global_steps", 
        type=int, 
        default=0,  # 0 means unspecified, will use YAML config
        help="Stop after this many global training steps; if >0, it becomes the primary stop condition."
    )
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    
    # Resume
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resume training.")
    
    return parser
