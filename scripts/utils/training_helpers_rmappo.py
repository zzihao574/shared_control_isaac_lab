#!/usr/bin/env python3

"""
Training helper utilities for dual-network rMAPPO multi-environment parallel training.
Features unified training execution, milestone evaluation, and optimized WandB logging.
MODIFIED: Added mid-rollout evaluation trigger with override bootstrap mechanism.
OPTIMIZED: Removed TopK redundancy, fixed evaluation bugs, optimizer path compatibility.
FIXED: Evaluation returns with env.reset(), simplified action selection interface.
"""

import argparse
import copy
import os
import yaml
import torch
import numpy as np
import random
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any
from collections import defaultdict, deque


SUPPORTED_HUMAN_MODEL_TYPES = {
    "learnable",
    "fixed_impedance",
    "residual_impedance",
}


@dataclass(frozen=True)
class RMAPPOSeedPlan:
    """Single source of truth for rMAPPO random-stream derivation."""

    base_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_seed", int(self.base_seed))

    def network_seed(self) -> int:
        return self.base_seed % (2**32)

    def minibatch_seed(self) -> int:
        return (self.base_seed + 424242) % (2**32)

    def apply_network_seed(self) -> None:
        torch.manual_seed(self.network_seed())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.network_seed())

    def make_minibatch_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.minibatch_seed())
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
    print(f"[SEED] Reproducibility set. seed={seed}, strict={strict_determinism}")


def validate_rmappo_config(params: Dict[str, Any]) -> None:
    """Validate resolved human-model and rMAPPO configuration."""
    model_type = str(params.get("human_model_type", "learnable"))
    if model_type not in SUPPORTED_HUMAN_MODEL_TYPES:
        choices = ", ".join(sorted(SUPPORTED_HUMAN_MODEL_TYPES))
        raise ValueError(f"Unsupported human_model_type={model_type!r}; expected {choices}")

    rmappo_cfg = params.get("algorithms", {}).get("rmappo")
    if not isinstance(rmappo_cfg, dict):
        raise ValueError("Missing algorithms.rmappo configuration")

    if model_type != "learnable":
        impedance = params.get("human_impedance", {})
        for name in ("kp", "kd"):
            values = impedance.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError(f"human_impedance.{name} must contain three values")

    constraints = params.get("constraints", {})
    for name in ("max_human_force", "max_robot_force"):
        if float(constraints.get(name, 0.0)) <= 0.0:
            raise ValueError(f"constraints.{name} must be positive")

    force_scaling = params.get("force_scaling", {})
    for agent_name in ("human", "robot"):
        factor_name = f"{agent_name}_factor"
        limit_name = f"max_{agent_name}_force"
        factor = float(force_scaling.get(factor_name, 0.0))
        limit = float(constraints.get(limit_name, 0.0))
        if factor <= 0.0:
            raise ValueError(f"force_scaling.{factor_name} must be positive")
        if not np.isclose(factor * limit, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"force_scaling.{factor_name} must equal 1/{limit_name}; "
                f"got {factor} * {limit}"
            )

    obs_factors = params.get("obs_scaling", {}).get("factors", [])
    if len(obs_factors) != 9:
        raise ValueError("obs_scaling.factors must contain 9 values")
    human_factor = float(force_scaling["human_factor"])
    robot_factor = float(force_scaling["robot_factor"])
    if not np.isclose(human_factor, robot_factor, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "The shared observation scaler requires equal human/robot force factors"
        )
    if not np.allclose(obs_factors[-3:], human_factor, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "The final three obs_scaling factors must match force_scaling"
        )


def resolve_rmappo_config(args) -> "TrainingConfiguration":
    """Resolve YAML/checkpoint configuration and CLI overrides before launch."""
    checkpoint = None
    if getattr(args, "checkpoint", None):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if checkpoint is not None and isinstance(checkpoint.get("params"), dict):
        params = copy.deepcopy(checkpoint["params"])
    else:
        with open(args.config, "r", encoding="utf-8") as config_file:
            params = yaml.safe_load(config_file) or {}

    training_cfg = params.setdefault("training", {})
    stored_seed = int(params.get("seed", training_cfg.get("seed", 42)))
    if checkpoint is not None and args.seed is not None and int(args.seed) != stored_seed:
        raise ValueError(
            f"Cannot resume with seed={args.seed}; checkpoint seed is {stored_seed}"
        )
    seed = stored_seed if args.seed is None else int(args.seed)
    params["seed"] = seed
    training_cfg["seed"] = seed

    if checkpoint is not None:
        checkpoint_model = str(params.get("human_model_type", "learnable"))
        if args.human_model_type is not None and args.human_model_type != checkpoint_model:
            raise ValueError(
                "Cannot resume with a different human model: "
                f"checkpoint={checkpoint_model}, CLI={args.human_model_type}"
            )
    elif args.human_model_type is not None:
        params["human_model_type"] = args.human_model_type
    params.setdefault("human_model_type", "learnable")

    stored_num_envs = int(params.get("num_envs", 512))
    num_envs = stored_num_envs if args.num_envs is None else int(args.num_envs)
    params["num_envs"] = num_envs

    rmappo_cfg = params.setdefault("algorithms", {}).setdefault("rmappo", {})
    if int(getattr(args, "max_global_steps", 0)) > 0:
        rmappo_cfg["max_global_steps"] = int(args.max_global_steps)

    validate_rmappo_config(params)
    args.seed = seed
    args.num_envs = num_envs
    return TrainingConfiguration(args.config, params=params)


def build_rmappo_wandb_config(
    resolved_config: Dict[str, Any],
    runtime: Dict[str, Any],
    human_metadata: Dict[str, Any],
    git_commit: str,
) -> Dict[str, Any]:
    """Build complete and query-friendly WandB metadata."""
    config = copy.deepcopy(resolved_config)
    constraints = resolved_config.get("constraints", {})
    force_scaling = resolved_config.get("force_scaling", {})
    config.update(
        {
            "algorithm": "rmappo",
            "num_envs": int(runtime["num_envs"]),
            "max_global_steps": int(runtime["max_global_steps"]),
            "experiment/algorithm": "rmappo",
            "experiment/human_model_type": str(
                resolved_config.get("human_model_type", "learnable")
            ),
            "experiment/seed": int(runtime["seed"]),
            "robot/max_force_per_axis": float(
                constraints.get("max_robot_force", 0.04)
            ),
            "human/force_input_factor": float(
                force_scaling.get("human_factor", 25.0)
            ),
            "robot/force_input_factor": float(
                force_scaling.get("robot_factor", 25.0)
            ),
            "observation/base_dim": 6,
            "observation/opponent_action_dim": 3,
            "observation/total_dim": 9,
            "observation/opponent_action_source": "previous_actual_applied_force",
            "git_commit": git_commit,
        }
    )
    config.update(copy.deepcopy(human_metadata))
    return config

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")


# ============================================================================
# UTILITY: Multi-path attribute accessor for optimizer compatibility
# ============================================================================

def _get_attr_chain(obj, *paths):
    """
    Try to get attribute from obj using multiple dot-separated paths.
    Returns first existing non-None attribute, or None if all fail.
    
    Example:
        _get_attr_chain(rmappo, "policies.human.actor_optimizer", 
                               "trainers.human.optimizer_actor")
    
    This handles different optimizer storage patterns across implementations.
    """
    for p in paths:
        cur = obj
        ok = True
        for name in p.split('.'):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and cur is not None:
            return cur
    return None


class RMAPPOTrainingRunner:
    """Unified training loop executor for dual-network rMAPPO with mid-rollout evaluation support."""
    
    def __init__(self, env, rmappo_wrapper, metrics_hub, agent_ids, max_global_steps=None, evaluator=None):
        self.env = env
        self.rmappo = rmappo_wrapper
        self.metrics = metrics_hub
        self.agent_ids = agent_ids
        self.evaluator = evaluator  # NEW: evaluator reference for mid-rollout eval
        
        # Clear separation of step counting
        self.global_step = 0
        self.train_updates = 0
        self.global_episodes = 0
        self._skip_episode_once = False
        self._current_obs = None
        
        # Evaluation mode control
        self.is_eval_mode = False
        
        # ============ NEW: Milestone tracking for mid-rollout evaluation ============
        self.milestone_episodes = rmappo_wrapper.params.get('training_monitor', {}).get('milestone_episodes', [])
        self.max_milestone_triggered = 0
        self._pending_eval_milestone = None
        self._has_eval_this_window = False  # Prevent multiple evals in same window
        print(f"[RUNNER] Milestone episodes: {self.milestone_episodes}")
        # ===========================================================================
        
        # rMAPPO specific parameters
        self.T = rmappo_wrapper.T
        self.num_mini_batch = rmappo_wrapper.params.get('algorithms', {}).get('rmappo', {}).get('num_mini_batch', 4)
        self.data_chunk_length = rmappo_wrapper.params.get('algorithms', {}).get('rmappo', {}).get('data_chunk_length', 16)
        
        seed_plan = RMAPPOSeedPlan(rmappo_wrapper.params["seed"])
        self._train_generator = seed_plan.make_minibatch_generator()
        print(
            "[RUNNER] Global training RNG initialized with seed: "
            f"{seed_plan.minibatch_seed()}"
        )
        
        # Inject RNG into Wrapper
        self.rmappo.train_generator = self._train_generator
        
        # Use the max_global_steps passed from trainer
        if max_global_steps is not None and max_global_steps > 0:
            self.max_global_steps = int(max_global_steps)
        else:
            rmappo_args = rmappo_wrapper.params.get('algorithms', {}).get('rmappo', {})
            if not rmappo_args:
                raise ValueError("[CONFIG ERROR] 'algorithms.rmappo' is missing from params.")
            self.max_global_steps = int(rmappo_args.get('max_global_steps', 200000))
        
        print(f"[DUAL RMAPPO RUNNER] Configured:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  Max collection steps: {self.max_global_steps}")
        print(f"  Step counting: global_step (collection) + train_updates (training rounds)")
        print(f"  Networks: independent human & robot")
        print(f"  Reproducibility: Global CPU RNG for minibatch shuffling")
        print(f"  Mid-rollout evaluation: ENABLED")

    def set_eval_mode(self, flag: bool):
        """Set evaluation mode flag."""
        self.is_eval_mode = bool(flag)
        self.rmappo.set_eval_mode(bool(flag))

    def _check_milestone_crossed(self, future_episodes: int) -> Optional[int]:
        """Check if any milestone is crossed by future episode count."""
        if not self.milestone_episodes:
            return None
        
        for milestone in sorted(self.milestone_episodes):
            if future_episodes >= milestone > self.max_milestone_triggered:
                return milestone
        return None

    def _handle_mid_rollout_evaluation(self, rollout_step: int, current_obs, next_obs, milestone: int):
        """
        Handle artificial truncation and bootstrap override for mid-rollout evaluation.
        
        Key steps:
        1. Identify ongoing environments (not naturally done)
        2. Apply artificial truncation: masks[t]=0, term_masks[t]=1 (truncated semantics)
        3. Compute V(s_{t+1}) using CORRECT RNN state (BEFORE masking)
        4. Store override bootstrap values
        
        CRITICAL FIX: Do NOT multiply critic hidden state by masks[t] before computing V(s_{t+1}).
        We want V from the "pre-evaluation" context, not a zeroed state.
        """
        t = rollout_step
        
        print(f"\n[MID-ROLLOUT EVAL] Handling evaluation trigger at t={t}, milestone={milestone}")
        
        for aid in self.rmappo.trainable_agent_ids:
            buf = self.rmappo.buffers[aid]
            
            # ============ KEY POINT 1: Strict ongoing identification ============
            # ongoing = environments where:
            #   - masks[t] > 0.5 (not done by environment)
            #   - term_masks[t] > 0.5 (not terminated, could continue)
            ongoing = (buf.masks[t] > 0.5) & (buf.term_masks[t] > 0.5)
            # ====================================================================
            
            if not ongoing.any():
                print(f"[TRUNCATE] Agent {aid}: No ongoing envs at t={t}")
                continue
            
            ongoing_count = ongoing.sum().item()
            print(f"[TRUNCATE] Agent {aid}: {ongoing_count} ongoing envs at t={t}")
            
            # ============ KEY POINT 2: Compute V(s_{t+1}) BEFORE masking ============
            # CRITICAL: Use the ORIGINAL critic hidden state, not masked by masks[t]
            # We want the value estimate from the "pre-evaluation" RNN context
            obs_scaled = self.rmappo.build_obs_scaled(next_obs)
            _, share_obs = self.rmappo.build_obs_tensors(obs_scaled, aid)
            
            # ✅ FIXED: Clone but do NOT multiply by masks[t]
            # The RNN state should reflect the "pre-evaluation" trajectory
            critic_h = self.rmappo.rnn_states[aid]["critic"].clone()
            # ❌ OLD BUGGY CODE: critic_h = critic_h * buf.masks[t].squeeze(-1).unsqueeze(-1)
            
            # FIXED: Explicit dtype for critic mask
            masks_for_critic = torch.ones(
                share_obs.shape[0], 1, 
                device=self.rmappo.device, 
                dtype=torch.float32
            )
            
            with torch.no_grad():
                v_next_pre_eval = self.rmappo.policies[aid].get_values(
                    share_obs,
                    critic_h,  # Use unmasked critic state
                    masks_for_critic
                )
            # ==========================================================================
            
            # ============ Apply artificial truncation AFTER computing V ============
            # Break advantage recursion but allow bootstrap
            buf.masks[t][ongoing] = 0.0  # Break GAE recursion
            
            # FIXED: Explicitly set term_masks to allow bootstrap (truncated semantics)
            buf.term_masks[t][ongoing] = 1.0  # Allow bootstrap, prevent treating as terminal
            # =======================================================================
            
            # Store override bootstrap values
            buf.override_bootstrap_values[t][ongoing] = v_next_pre_eval[ongoing]
            buf.override_bootstrap_mask[t][ongoing] = True
            
            print(f"[OVERRIDE] Agent {aid}: Stored V(s_{{t+1}}) for {ongoing_count} envs at t={t}")
            
            # FIXED: Proper sample index extraction
            if ongoing_count > 0:
                # Get env indices where ongoing is True
                env_ids = torch.nonzero(ongoing.view(-1), as_tuple=True)[0]
                sample_idx = int(env_ids[0])
                sample_value = v_next_pre_eval[sample_idx, 0].item()
                print(f"  Sample env {sample_idx}: V(s_{{t+1}}) = {sample_value:.4f}")

    def execute_training_step(self):
        """
        Execute one complete rollout and training update for both networks.
        MODIFIED: Added mid-rollout evaluation trigger.
        FIXED: Evaluation returns with env.reset() instead of reusing stale next_obs.
        """
        current_obs = self._current_obs
        if current_obs is None:
            if hasattr(self.env, "_get_observations"):
                current_obs = self.env._get_observations()
            else:
                current_obs, _ = self.env.reset()
            self._current_obs = current_obs
        
        # ============ NEW: Buffer double-check cleanup ============
        # Ensure override fields are clean before starting new rollout
        for aid in self.agent_ids:
            self.rmappo.buffers[aid].override_bootstrap_values.zero_()
            self.rmappo.buffers[aid].override_bootstrap_mask.zero_()
        # ==========================================================
        
        # Reset window-level flags
        self._has_eval_this_window = False
        self._pending_eval_milestone = None
        
        episode_count = 0
        
        # Collect complete rollout
        for rollout_step in range(self.T):
            if not self.is_eval_mode:
                self.global_step += 1
                self.env.unwrapped.set_trainer_global_step(self.global_step)
                
                for aid in self.rmappo.trainable_agent_ids:
                    self.rmappo.trainers[aid].global_step = self.global_step
            
            # ============ FIXED: Simplified action selection interface ============
            actions, detail = self.rmappo.select_actions(
                current_obs, 
                deterministic=self.is_eval_mode
            )
            # =======================================================================

            self.env.unwrapped.set_detail_actor_info(detail)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            if not self.is_eval_mode and self.global_step % 10 == 0:
                actual_env = getattr(self.env, "unwrapped", self.env)
                if hasattr(actual_env, "get_force_breakdown"):
                    detail["force_breakdown"] = actual_env.get_force_breakdown()
                self._push_current_step_force_statistics(detail)

            # Calculate done status
            done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
            done_any = None
            for aid in self.agent_ids:
                d = done_any_dict[aid].to(torch.bool)
                done_any = d if done_any is None else (done_any | d)
            
            episode_increment = int(done_any.sum().item())
            
            # ============ NEW: Mid-rollout evaluation trigger ============
            if not self.is_eval_mode and episode_increment > 0 and not self._has_eval_this_window:
                # Check if we've crossed a milestone
                future_episode_count = self.global_episodes + episode_count + episode_increment
                crossed_milestone = self._check_milestone_crossed(future_episode_count)
                
                if crossed_milestone is not None:
                    print(f"\n{'='*80}")
                    print(f"[MID-ROLLOUT EVAL] Milestone {crossed_milestone} triggered at rollout_step={rollout_step}")
                    print(f"  Current episodes: {self.global_episodes}")
                    print(f"  Episodes this window: {episode_count}")
                    print(f"  Future episodes: {future_episode_count}")
                    print(f"{'='*80}\n")
                    
                    # First, add current experience to buffer (before evaluation)
                    if not self.is_eval_mode:
                        self.rmappo.add_experience_to_buffer(
                            obs=current_obs,
                            actions=actions,
                            rewards=rewards,
                            next_obs=next_obs,
                            dones=done_any_dict,
                            terminated=terminated,
                            truncated=truncated,
                            infos=infos
                        )
                    
                    # Handle artificial truncation and bootstrap override
                    self._handle_mid_rollout_evaluation(
                        rollout_step=rollout_step,
                        current_obs=current_obs,
                        next_obs=next_obs,
                        milestone=crossed_milestone
                    )
                    
                    # Mark that we've handled eval this window
                    self._has_eval_this_window = True
                    self._pending_eval_milestone = crossed_milestone
                    
                    # Update episode count BEFORE evaluation
                    episode_count += episode_increment
                    
                    # Set eval mode and trigger evaluation
                    if self.evaluator is not None:
                        print(f"[MID-ROLLOUT EVAL] Switching to eval mode...")
                        self.set_eval_mode(True)
                        
                        eval_result = self.evaluator.run_evaluation(
                            crossed_milestone, 
                            self.global_step
                        )
                        
                        self.set_eval_mode(False)
                        print(f"[MID-ROLLOUT EVAL] Returned to training mode")
                        
                        # Update max milestone
                        self.max_milestone_triggered = crossed_milestone
                        
                        # ============ FIXED: Reset environment after evaluation ============
                        # Use reset observation (t1) instead of stale next_obs (t1')
                        obs_reset, _ = self.env.reset()
                        current_obs = obs_reset
                        self._skip_episode_once = True  # Evaluation后的第一步不计入上一段
                        # ==================================================================
                    
                    # Continue to next rollout step after evaluation
                    continue
            # =============================================================
            
            # Normal buffer insertion (if not in eval mode and not just handled above)
            if not self.is_eval_mode:
                self.rmappo.add_experience_to_buffer(
                    obs=current_obs,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=done_any_dict,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos
                )
            
            # Episode counting
            if self.is_eval_mode or self._skip_episode_once:
                episode_increment = 0
                if self._skip_episode_once:
                    self._skip_episode_once = False
                    
            episode_count += episode_increment
            current_obs = next_obs

        self.rmappo.store_next_obs(next_obs)
        
        if not self.is_eval_mode:
            stats = self.rmappo.update()
            self.train_updates += 1
            self.global_episodes += episode_count

            if stats:
                payload = {
                    "loss/actor": {aid: stats.get(f"policy_loss/{aid}", 0.0) for aid in self.rmappo.trainable_agent_ids},
                    "loss/critic": {aid: stats.get(f"value_loss/{aid}", 0.0) for aid in self.rmappo.trainable_agent_ids},
                    "grad_norm/actor": {aid: stats.get(f"actor_grad_norm/{aid}", 0.0) for aid in self.rmappo.trainable_agent_ids},
                    "grad_norm/critic": {aid: stats.get(f"critic_grad_norm/{aid}", 0.0) for aid in self.rmappo.trainable_agent_ids},
                    "policy/entropy": np.mean([stats.get(f"dist_entropy/{aid}", 0.0) for aid in self.rmappo.trainable_agent_ids]),
                    "ppo/ratio_mean": np.mean([stats.get(f"ratio/{aid}", 1.0) for aid in self.rmappo.trainable_agent_ids]),
                    "train/collection_steps": self.global_step,
                    "train/training_rounds": self.train_updates,
                    "train/global_episodes": self.global_episodes,
                }
                payload = {k: v for k, v in payload.items() if v is not None}
                self.metrics.push_update(self.global_step, payload)

            if self.train_updates % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            stats = {}

        self._current_obs = next_obs
        
        # Return pending milestone if any
        return self._pending_eval_milestone

    def mark_skip_episode_once(self):
        """Mark to skip episode counting once for milestone evaluation."""
        self._skip_episode_once = True

    def _push_current_step_force_statistics(self, detail):
        """Push actual environment force channels to WandB."""
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
                f"forces/{channel}_norm_rms": float(
                    torch.sqrt(torch.mean(norms.square())).item()
                ),
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
                self.rmappo.params.get("constraints", {}).get("max_human_force", 0.04)
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

        if self.rmappo.human_model_type == "fixed_impedance":
            force_payload["human/fixed_actor_checksum"] = (
                self.rmappo.human_actor_checksum()
            )

        if force_payload:
            self.metrics.push_update(self.global_step, force_payload)

    def run_until(self, max_global_steps: int):
        """Run training until reaching maximum collection steps."""
        obs, _ = self.env.reset()
        self._current_obs = obs
        while self.global_step < max_global_steps:
            self.execute_training_step()


class RMAPPOMilestoneEvaluator:
    """
    Milestone evaluator for dual-network rMAPPO with checkpoint saving.
    MODIFIED: Removed topk_mgr parameter (TopKManager deleted).
    """
    
    def __init__(self, env, rmappo_wrapper, metrics_hub, log_dir, agent_ids, runner=None):
        """
        Initialize milestone evaluator.
        
        Args:
            env: Environment instance
            rmappo_wrapper: RMAPPO wrapper
            metrics_hub: Metrics hub for logging
            log_dir: Log directory
            agent_ids: List of agent IDs
            runner: Training runner reference (for checkpoint saving)
        """
        self.env = env
        self.rmappo = rmappo_wrapper
        self.metrics = metrics_hub
        self.log_dir = log_dir
        self.agent_ids = agent_ids
        self.runner = runner

    def run_evaluation(self, milestone: int, global_step: int) -> dict:
        """
        Handle milestone evaluation and model saving for dual networks.
        
        STRICT EVALUATION ISOLATION:
        - torch.no_grad() context
        - networks in .eval() mode
        - deterministic actions only
        - no buffer writes
        - no step/episode counting
        
        OPTIMIZED: Removed TopK redundancy, score tracked via WandB milestone metrics.
        """
        for aid in self.agent_ids:
            self.rmappo.policies[aid].actor.eval()
            self.rmappo.policies[aid].critic.eval()

        try:
            with torch.no_grad():
                return_norm, num_eps = self._run_single_evaluation_episode()
        finally:
            for aid in self.agent_ids:
                self.rmappo.policies[aid].actor.train()
                self.rmappo.policies[aid].critic.train()
        
        milestone_return = return_norm * 1000
        
        print(f"[EVAL] Milestone {milestone}: return_norm={return_norm:.4f}, scaled={milestone_return:.2f}")

        # ============ NEW: Save milestone checkpoint ============
        if self.runner is not None:
            ckpt_dir = os.path.join(self.log_dir, "checkpoints")
            ckpt_path = save_milestone_checkpoint(
                ckpt_dir=ckpt_dir,
                rmappo=self.rmappo,
                runner=self.runner,
                score=milestone_return,
                milestone=milestone
            )
            print(f"[EVAL] Checkpoint saved: {ckpt_path}")
        # ========================================================

        # ============ OPTIMIZED: Removed TopK update ============
        # TopK is no longer used - each milestone gets its own checkpoint
        # Score is tracked via WandB metrics below
        # ========================================================

        # Push metrics to WandB
        payload = {
            "eval/return_mean": float(milestone_return),      # Maps to milestone/actor_return
            "eval/num_episodes": int(num_eps),
            "milestone/latest_completed": int(milestone),
            "eval/return_norm": float(return_norm),
        }
        self.metrics.push_milestone(global_step, milestone, payload)
        
        print(f"[EVAL] Uploaded milestone metrics: scaled_return={milestone_return:.2f}")

        # Clear RNN states after evaluation to avoid training contamination
        for aid in self.agent_ids:
            self.rmappo.rnn_states[aid]["actor"].zero_()
            self.rmappo.rnn_states[aid]["critic"].zero_()
        
        print(f"[EVAL] RNN states cleared after milestone {milestone}")

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

        print(f"[EVAL] Starting in-place dual rMAPPO evaluation (env{active_env} only, 1 episode)...")

        obs, _ = env.reset()

        num_envs = len(obs[self.agent_ids[0]])
        ep_returns = torch.zeros(num_envs, device=self.rmappo.device)
        ep_steps = torch.zeros(num_envs, dtype=torch.int64, device=self.rmappo.device)
        completed_return_norms = []
        
        for aid in self.agent_ids:
            rmappo_args = self.rmappo.params.get('algorithms', {}).get('rmappo', {})
            H = rmappo_args.get('hidden_size', 256)
            self.rmappo.rnn_states[aid]["actor"] = torch.zeros(num_envs, H, device=self.rmappo.device)
            self.rmappo.rnn_states[aid]["critic"] = torch.zeros(num_envs, H, device=self.rmappo.device)
        
        eval_step_counter = 0

        with torch.no_grad():
            while len(completed_return_norms) < target_episodes:
                if hasattr(env, '_get_observations'):
                    current_obs = env._get_observations()
                elif hasattr(env, 'observation_manager'):
                    current_obs = env.observation_manager.compute()
                else:
                    current_obs = obs
                
                actions_dict, detail_info = self.rmappo.select_actions(
                    current_obs, deterministic=True
                )
                env.set_detail_actor_info(detail_info)

                env.set_trainer_global_step(eval_step_counter)
                obs, rewards, terminated, truncated, infos = env.step(actions_dict)
                eval_step_counter += 1

                step_rewards = torch.stack([rewards[aid] for aid in self.agent_ids])
                avg_step_rewards = step_rewards.mean(dim=0)
                ep_returns[active_env] += avg_step_rewards[active_env]
                ep_steps[active_env] += 1
                
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                done_any = None
                for aid in self.agent_ids:
                    d = done_any_dict[aid].to(torch.bool)
                    done_any = d if done_any is None else (done_any | d)
                
                if done_any[active_env]:
                    total_reward = float(ep_returns[active_env].item())
                    total_steps = int(ep_steps[active_env].item())
                    return_norm = total_reward / max(1, total_steps)
                    completed_return_norms.append(return_norm)
                    
                    print(f"[EVAL] Episode completed: steps={total_steps}, total_reward={total_reward:.3f}, return_norm={return_norm:.4f}")
                    
                    ep_returns[active_env] = 0.0
                    ep_steps[active_env] = 0
                    
                    if len(completed_return_norms) >= target_episodes:
                        break
                
                if eval_step_counter % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_return_norms = completed_return_norms[:target_episodes]
        avg_return_norm = sum(final_return_norms) / max(1, len(final_return_norms))
        
        print(f"[EVAL] Completed: {len(final_return_norms)} episodes, Average return_norm: {avg_return_norm:.4f}")
        
        return avg_return_norm, len(final_return_norms)

    def _extract_dual_model_state(self):
        """Extract model state for dual networks checkpoint saving."""
        model_state = {}
        for aid in self.agent_ids:
            policy = self.rmappo.policies[aid]
            model_state.update({
                f'{aid}_actor': policy.actor.state_dict(),
                f'{aid}_critic': policy.critic.state_dict(),
            })
        return model_state


class MetricsHub:
    """Simplified single-exit metrics bus for unified data pipeline."""
    
    def __init__(self, ring: int = 100):
        self.subs = defaultdict(list)
        self.update_ring = deque(maxlen=ring)

    def subscribe(self, event: str, handler) -> None:
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

    def push_scalars(self, stats: dict, step: int = None) -> None:
        """Push scalar metrics."""
        if not stats:
            return
        if step is not None:
            data = {"step": step, **stats}
        else:
            data = stats
        self._emit("update", data)

    def push_milestone(self, step: int, milestone: int, summary: dict) -> None:
        """Push milestone completion summary."""
        self._emit("milestone_summary", {"step": step, "milestone": milestone, **summary})


class WandBLogger:
    """Optimized WandB logger with safe scalar conversion and step protection."""
    
    AGENT_METRICS_MAP = {
        'loss/actor': 'train/actor_loss_{}',
        'loss/critic': 'train/critic_loss_{}',
        'grad_norm/actor': 'grad/{}_actor',
        'grad_norm/critic': 'grad/{}_critic',
    }
    
    GLOBAL_METRICS_MAP = {
        "policy/entropy": "policy/entropy",
        "ppo/ratio_mean": "ppo/ratio_mean",
        "train/collection_steps": "train/collection_steps",
        "train/training_rounds": "train/training_rounds",
        "train/global_episodes": "train/global_episodes",
        "eval/return_mean": "milestone/actor_return",  # Milestone scores via this mapping
        "milestone/latest_completed": "milestone/latest_completed",
        "eval/num_episodes": "eval/num_episodes",
        "forces/robot_fx_mean": "forces/robot_fx_mean",
        "forces/robot_fy_mean": "forces/robot_fy_mean",
        "forces/robot_fz_mean": "forces/robot_fz_mean",
        "forces/human_fx_mean": "forces/human_fx_mean",
        "forces/human_fy_mean": "forces/human_fy_mean", 
        "forces/human_fz_mean": "forces/human_fz_mean",
        "lr/actor": "lr/actor",
        "lr/critic": "lr/critic",
        "ppo/kl_mean": "ppo/kl_mean",
        "ppo/clip_fraction": "ppo/clip_fraction",
        "ppo/adv_mean_norm": "ppo/adv_mean_norm",
        "ppo/adv_std_norm": "ppo/adv_std_norm",
        "value/ret_abs_mean": "value/ret_abs_mean",
        "value/v_abs_mean": "value/v_abs_mean",
        "value/ret_absmax": "value/ret_absmax",
        "value/v_absmax": "value/v_absmax",
        "policy/logstd_mean": "policy/logstd_mean",
        "policy/saturation": "policy/saturation",
        "rnn/actor_h_norm": "rnn/actor_h_norm",
        "rnn/critic_h_norm": "rnn/critic_h_norm",
        "grad/actor": "grad/actor",
        "grad/critic": "grad/critic",
    }
    
    def __init__(self, project_name: str = "surgical_robot_rmappo", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.run = None
        self._last_step = -1
        
        if not self.enabled:
            print("[WANDB] Disabled")

    def initialize_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """Initialize WandB run."""
        if not self.enabled:
            return
        
        if wandb.run is not None:
            print("[WANDB] Run already initialized")
            wandb.config.update(config, allow_val_change=True)
            self.run = wandb.run
            return
        
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config,
            tags=["rmappo", "multi-agent", "surgical-robot", "rnn", "dual-network", "reproducible"],
        )
        
        rmappo_cfg = config.get("algorithms", {}).get("rmappo", {})
        wandb.config.update({
            "rollout_horizon": rmappo_cfg.get("rollout_horizon", 256),
            "ppo_epoch": rmappo_cfg.get("ppo_epoch", 10),
            "num_mini_batch": rmappo_cfg.get("num_mini_batch", 4),
            "clip_param": rmappo_cfg.get("clip_param", 0.2),
            "hidden_size": rmappo_cfg.get("hidden_size", 256),
            "network_architecture": "dual_independent",
            "action_distribution": "tanh_gaussian",
            "reproducibility_mode": "global_rng_fixed_num_envs",
        })
        
        print(f"[WANDB] Successfully initialized: {self.run.name}")

    def attach_metrics_hub(self, hub: "MetricsHub"):
        """Attach to MetricsHub for unified data pipeline."""
        if not self.enabled:
            return

        hub.subscribe("update", lambda data: self.log_metrics(data, data.get("step", 0)))

        def _on_ms(ms):
            if not hasattr(self, "_printed_ms_once"):
                print(f"[WANDB] milestone_summary received once: keys={list(ms.keys())}, step={ms.get('step')}")
                self._printed_ms_once = True
            
            step = ms.get("step", 0)
            payload_to_log = {}
            
            # Log milestone score via eval/return_mean (maps to milestone/actor_return)
            if "eval/return_mean" in ms:
                payload_to_log["eval/return_mean"] = ms["eval/return_mean"]
            
            # Log milestone completion
            if "milestone" in ms:
                payload_to_log["milestone/latest_completed"] = ms["milestone"]
            
            # Log episode count
            if "eval/num_episodes" in ms:
                payload_to_log["eval/num_episodes"] = ms["eval/num_episodes"]

            if payload_to_log:
                self.log_metrics(payload_to_log, step)

        hub.subscribe("milestone_summary", _on_ms)
        print("[WANDB] Attached to MetricsHub")

    def log_metrics(self, metrics_data: Dict[str, Any], step: int) -> None:
        """Log metrics with safe scalar conversion and step protection."""
        if not self.enabled or not metrics_data:
            return

        # Step monotonic protection
        if step < self._last_step:
            step = self._last_step + 1
        self._last_step = step

        # Safe scalar conversion
        def _to_scalar_safe(v):
            try:
                if isinstance(v, (int, float, bool)):
                    x = float(v)
                elif isinstance(v, np.generic):
                    x = float(v)
                elif isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        return None
                    x = float(v.detach().cpu().item())
                else:
                    x = float(v)
                if math.isnan(x) or math.isinf(x):
                    return None
                return x
            except Exception:
                return None

        log_data = {}

        # Per-agent metrics
        if any(key in metrics_data and isinstance(metrics_data.get(key), dict) 
               for key in self.AGENT_METRICS_MAP.keys()):
            agent_ids = None
            for source_key in self.AGENT_METRICS_MAP.keys():
                if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                    agent_ids = list(metrics_data[source_key].keys())
                    break
            
            if agent_ids:
                for source_key, target_pattern in self.AGENT_METRICS_MAP.items():
                    if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                        for agent_id in agent_ids:
                            if agent_id in metrics_data[source_key]:
                                val = _to_scalar_safe(metrics_data[source_key][agent_id])
                                if val is not None:
                                    log_data[target_pattern.format(agent_id)] = val

        # Global metrics
        for src_key, dest_key in self.GLOBAL_METRICS_MAP.items():
            if src_key in metrics_data and metrics_data[src_key] is not None:
                val = _to_scalar_safe(metrics_data[src_key])
                if val is not None:
                    log_data[dest_key] = val

        # Passthrough fallback
        passthrough_prefixes = (
            "train/", "forces/", "lr/", "eval/",
            "ppo/", "value/", "policy/", "rnn/", "grad/", "milestone/",
            "lifecycle/", "human/"
        )
        for k, v in metrics_data.items():
            if any(k.startswith(p) for p in passthrough_prefixes):
                val = _to_scalar_safe(v)
                if val is not None and k not in log_data:
                    log_data[k] = val

        if log_data:
            wandb.log(log_data, step=step)

    def finalize_run(self) -> None:
        """Finalize WandB run."""
        if self.enabled and self.run:
            wandb.finish()
            print("[WANDB] Run finished")


class TrainingConfiguration:
    """Training configuration loader."""
    
    def __init__(self, config_path: str, params: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        if params is None:
            with open(self.config_path, 'r') as f:
                params = yaml.safe_load(f)
        self.params = params or {}
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Create configuration from YAML file."""
        return cls(config_path)
    
    def get_compute_device(self) -> str:
        """Get compute device."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_flat_dual_checkpoint(
    rmappo,
    runner,
    score: Optional[float] = None,
    milestone: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build flat checkpoint with dual networks for milestone saving and resume.
    
    Contains:
    - Four network state_dicts (human/robot x actor/critic)
    - Optimizer states for training resume (multi-path compatible)
    - RNG states for reproducibility
    - Counters (steps/episodes/updates) for continuous training
    """
    checkpoint = {
        "algorithm": "rmappo_dual",
        "agent_ids": rmappo.agent_ids,
        "params": rmappo.params,
        "human_model_type": getattr(rmappo, "human_model_type", "learnable"),
        "human_impedance": rmappo.params.get("human_impedance", {}),
        "force_limit_semantics": "per_axis",
        "human_actor_checksum": rmappo.human_actor_checksum(),
        "git_commit": rmappo.params.get("git_commit", "unknown"),
        
        # Counters: ensure LR schedule and evaluation continuity
        "global_steps_total": int(runner.global_step),
        "training_rounds_total": int(rmappo.train_updates),
        "episodes_done_total": int(runner.global_episodes),
        "max_milestone_triggered": int(runner.max_milestone_triggered),
        
        # Four networks (flat keys)
        "human_actor": rmappo.policies["human"].actor.state_dict(),
        "human_critic": rmappo.policies["human"].critic.state_dict(),
        "robot_actor": rmappo.policies["robot"].actor.state_dict(),
        "robot_critic": rmappo.policies["robot"].critic.state_dict(),
    }
    if score is not None:
        checkpoint["score"] = float(score)
    if milestone is not None:
        checkpoint["milestone"] = int(milestone)
    
    # ============ OPTIMIZED: Multi-path optimizer state extraction ============
    # Support different optimizer storage patterns:
    #   - policies[aid].actor_optimizer / critic_optimizer
    #   - trainers[aid].actor_optimizer / critic_optimizer  
    #   - trainers[aid].optimizer_actor / optimizer_critic
    #   - trainers[aid].optimizer (fallback: single optimizer)
    optim_state = {}
    for aid in rmappo.agent_ids:
        opt_actor = _get_attr_chain(
            rmappo,
            f"policies.{aid}.actor_optimizer",
            f"trainers.{aid}.actor_optimizer",
            f"trainers.{aid}.optimizer_actor",
            f"trainers.{aid}.optimizer",  # fallback
        )
        opt_critic = _get_attr_chain(
            rmappo,
            f"policies.{aid}.critic_optimizer",
            f"trainers.{aid}.critic_optimizer",
            f"trainers.{aid}.optimizer_critic",
            f"trainers.{aid}.optimizer",  # fallback
        )
        
        if opt_actor is not None:
            optim_state[f"{aid}_actor"] = opt_actor.state_dict()
        if opt_critic is not None:
            optim_state[f"{aid}_critic"] = opt_critic.state_dict()
    
    checkpoint["optim_state"] = optim_state
    # ===========================================================================
    
    # RNG states: ensure reproducibility
    checkpoint["rng_state"] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    return checkpoint


def save_milestone_checkpoint(ckpt_dir: str, rmappo, runner, score: float, milestone: int) -> str:
    """
    Save milestone checkpoint and append to index file.
    
    Each milestone gets its own file with naming:
        ckpt_milestone_{milestone:06d}_score_{score:.6f}.pth
    
    Index file (milestones_index.txt) is appended with:
        timestamp    milestone=X    score=Y.YYY    path=filename.pth
    
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    fname = f"ckpt_milestone_{milestone:06d}_score_{score:.6f}.pth"
    fpath = os.path.join(ckpt_dir, fname)
    
    # Build checkpoint
    ckpt = build_flat_dual_checkpoint(rmappo, runner, score, milestone)
    
    # Atomic write: .tmp -> replace
    tmp_path = fpath + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, fpath)
    
    # Append to index file
    index_path = os.path.join(ckpt_dir, "milestones_index.txt")
    with open(index_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp}\tmilestone={milestone}\tscore={score:.6f}\tpath={fname}\n")
    
    print(f"[CKPT] Saved milestone {milestone} (score={score:.4f}) -> {fname}")
    
    return fpath


def save_final_checkpoint_rmappo(ckpt_dir: str, rmappo, runner) -> str:
    """Save a resume-compatible checkpoint at the configured training limit."""
    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = int(getattr(runner, "global_step", 0))
    fpath = os.path.join(ckpt_dir, f"final_step_{global_step:09d}.pth")
    torch.save(build_flat_dual_checkpoint(rmappo, runner), fpath)
    print(f"[CKPT] Saved final checkpoint -> {fpath}")
    return fpath


def resume_from_checkpoint(path: str, rmappo, runner, device=None) -> None:
    """
    Resume training from checkpoint with full state restoration.
    
    Restores:
    - Network weights (4 networks)
    - Optimizer states (4 optimizers, multi-path compatible)
    - RNG states (Python/NumPy/PyTorch/CUDA)
    - Counters (steps/episodes/updates)
    """
    print(f"[RESUME] Loading checkpoint: {path}")
    
    # FIXED: PyTorch 2.6+ requires weights_only=False for non-weight data (RNG states, etc.)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # 1) Load network weights
    for aid in rmappo.agent_ids:
        rmappo.policies[aid].actor.load_state_dict(ckpt[f"{aid}_actor"])
        rmappo.policies[aid].critic.load_state_dict(ckpt[f"{aid}_critic"])
        print(f"[RESUME] Loaded {aid} actor/critic weights")
    
    # 2) Move to target device
    if device is not None:
        for aid in rmappo.agent_ids:
            rmappo.policies[aid].actor.to(device)
            rmappo.policies[aid].critic.to(device)
    
    # 3) Load optimizer states (multi-path compatible)
    if "optim_state" in ckpt:
        for aid in rmappo.agent_ids:
            # ============ OPTIMIZED: Multi-path optimizer restoration ============
            opt_actor = _get_attr_chain(
                rmappo,
                f"policies.{aid}.actor_optimizer",
                f"trainers.{aid}.actor_optimizer",
                f"trainers.{aid}.optimizer_actor",
                f"trainers.{aid}.optimizer",
            )
            opt_critic = _get_attr_chain(
                rmappo,
                f"policies.{aid}.critic_optimizer",
                f"trainers.{aid}.critic_optimizer",
                f"trainers.{aid}.optimizer_critic",
                f"trainers.{aid}.optimizer",
            )
            
            if opt_actor is not None and f"{aid}_actor" in ckpt["optim_state"]:
                opt_actor.load_state_dict(ckpt["optim_state"][f"{aid}_actor"])
            if opt_critic is not None and f"{aid}_critic" in ckpt["optim_state"]:
                opt_critic.load_state_dict(ckpt["optim_state"][f"{aid}_critic"])
            # ======================================================================
        print(f"[RESUME] Loaded optimizer states")
    
    # 4) Restore counters (critical for LR schedule and evaluation)
    runner.global_step = int(ckpt.get("global_steps_total", runner.global_step))
    rmappo.train_updates = int(ckpt.get("training_rounds_total", rmappo.train_updates))
    runner.global_episodes = int(ckpt.get("episodes_done_total", runner.global_episodes))
    
    # Also update trainer counters for LR decay
    for aid in rmappo.agent_ids:
        rmappo.trainers[aid].global_update_step = rmappo.train_updates
    
    print(f"[RESUME] Restored counters:")
    print(f"  global_step: {runner.global_step}")
    print(f"  train_updates: {rmappo.train_updates}")
    print(f"  global_episodes: {runner.global_episodes}")
    
    # 5) Restore RNG states
    if "rng_state" in ckpt:
        random.setstate(ckpt["rng_state"]["py"])
        np.random.set_state(ckpt["rng_state"]["np"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        
        if torch.cuda.is_available() and ckpt["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
        
        print(f"[RESUME] Restored RNG states")
    
    # 6) Optional: restore milestone tracking
    restored_milestone = ckpt.get(
        "max_milestone_triggered", ckpt.get("milestone")
    )
    if restored_milestone is not None:
        runner.max_milestone_triggered = int(restored_milestone)
        print(f"[RESUME] Last milestone: {runner.max_milestone_triggered}")
    
    print(f"[RESUME] Checkpoint restoration complete\n")


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create argument parser with unified checkpoint interface."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml')

    parser = argparse.ArgumentParser(description="Dual rMAPPO training")
    parser.add_argument("--config", type=str, default=config_path,
                       help="Path to training configuration YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (.pth) for resume training or play")
    parser.add_argument("--num_envs", type=int, default=None,
                       help="Number of parallel environments (default: resolved config or 512)")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0",
                       help="Environment task name")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: resolved config/checkpoint)")
    parser.add_argument(
        "--human_model_type",
        choices=tuple(sorted(SUPPORTED_HUMAN_MODEL_TYPES)),
        default=None,
        help="Override the human force model for a new run",
    )
    parser.add_argument("--run_dir", type=str, default=None,
                       help="Explicit run directory")
    parser.add_argument("--max_global_steps", type=int, default=0,
                       help="Maximum global steps (0=use config value)")
    parser.add_argument("--top_k_models", type=int, default=10,
                       help="Number of top models to keep (deprecated)")
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    
    return parser
