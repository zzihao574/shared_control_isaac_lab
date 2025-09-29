#!/usr/bin/env python3

"""
Training helper utilities for dual-network rMAPPO multi-environment parallel training.
Features unified training execution, milestone evaluation, and optimized WandB logging.
Adapted for independent human and robot networks with synchronized training.
MODIFIED: Added obs scaling support and term_masks generation for proper GAE bootstrap.
FAIL-FAST: Removed all NaN/Inf repair mechanisms, emergency fallbacks, and skip logic.
MODIFIED: Added lifecycle markers for eval->train transitions in gradient monitoring system.
MODIFIED: Updated WandB logging keys to 7-prefix structure (policy/ppo/value/rnn/grad/train/lifecycle).
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pickle
import math
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict, deque

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")


# ============================================================================
# FAIL-FAST CHECKING FUNCTIONS
# ============================================================================

def finite_check(name: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    """Check for NaN/Inf in tensor - fail fast, no repair"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name}: expected Tensor, got {type(x)}")
    if not torch.is_floating_point(x):
        return True
    ok = torch.isfinite(x).all().item()
    if ok:
        return True
    bad_ratio = (~torch.isfinite(x)).float().mean().item()
    try:
        min_v = torch.nanmin(x).item()
        max_v = torch.nanmax(x).item()
    except Exception:
        min_v, max_v = float("nan"), float("nan")
    msg = (f"[NUMERIC ERROR] {name}: non-finite values detected\n"
           f"  - bad_ratio={bad_ratio*100:.2f}%\n"
           f"  - range=[{min_v:.3e}, {max_v:.3e}]\n"
           f"  - shape={tuple(x.shape)}, device={x.device}, dtype={x.dtype}")
    if raise_on_fail:
        raise ValueError(msg)
    else:
        print("[WARNING]", msg)
        return False


class RMAPPOTrainingRunner:
    """Unified training loop executor for dual-network rMAPPO with rollout collection."""
    
    def __init__(self, env, rmappo_wrapper, metrics_hub, agent_ids, max_global_steps=None):
        self.env = env
        self.rmappo = rmappo_wrapper
        self.metrics = metrics_hub
        self.agent_ids = agent_ids
        
        # Clear separation of step counting
        self.global_step = 0  # Environment collection steps (training mode only)
        self.train_updates = 0  # Completed training rounds
        self.global_episodes = 0  # Total episodes completed
        self._skip_episode_once = False  # Flag to skip episode counting once
        self._current_obs = None  # Current observations cache
        
        # Evaluation mode control
        self.is_eval_mode = False
        
        # rMAPPO specific parameters
        self.T = rmappo_wrapper.T  # Rollout horizon
        
        # Use the max_global_steps passed from trainer
        if max_global_steps is not None and max_global_steps > 0:
            self.max_global_steps = int(max_global_steps)
        else:
            mappo_args = rmappo_wrapper.params.get('mappo_args', {})
            if not mappo_args:
                raise ValueError("[CONFIG ERROR] 'mappo_args' is missing from params. Ensure unified configuration loading is working.")
            
            self.max_global_steps = int(mappo_args.get('max_global_steps', 200000))
        
        print(f"[DUAL RMAPPO RUNNER] Configured:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  Max collection steps: {self.max_global_steps}")
        print(f"  Step counting: global_step (collection) + train_updates (training rounds)")
        print(f"  Networks: independent human & robot")
        print(f"  FAIL-FAST: enabled - no emergency fallbacks")
        print(f"  Gradient monitoring: enabled with lifecycle markers")

    def set_eval_mode(self, flag: bool):
        """Set evaluation mode flag."""
        self.is_eval_mode = bool(flag)

    def execute_training_step(self):
        """Execute one complete rollout and training update for both networks."""
        # Use current observations
        current_obs = self._current_obs
        if current_obs is None:
            if hasattr(self.env, "_get_observations"):
                current_obs = self.env._get_observations()
            else:
                current_obs, _ = self.env.reset()
            self._current_obs = current_obs
        
        # Initial observation validation - fail immediately on bad data
        for agent_id in self.agent_ids:
            if agent_id in current_obs:
                finite_check(f"initial_obs_{agent_id}", current_obs[agent_id])
            
        episode_count = 0
        
        # Collect complete rollout (T steps) - dual networks in parallel
        for rollout_step in range(self.T):
            # Increment collection step counter and sync to environment (training only)
            if not self.is_eval_mode:
                self.global_step += 1
                self.env.unwrapped.set_trainer_global_step(self.global_step)
                
                # Sync global step to rmappo for gradient monitoring
                for aid in self.agent_ids:
                    self.rmappo.trainers[aid].global_step = self.global_step
            
            # Pre-action observation validation
            for agent_id in self.agent_ids:
                if agent_id in current_obs:
                    finite_check(f"pre_action_obs_{agent_id}_step_{rollout_step}", current_obs[agent_id])

            # select_actions handles obs scaling internally
            actions, detail = self.rmappo.select_actions(
                current_obs, 
                add_noise=(not self.is_eval_mode),
                deterministic=self.is_eval_mode
            )
            
            # Action validation
            for agent_id in self.agent_ids:
                if agent_id in actions:
                    finite_check(f"action_{agent_id}_step_{rollout_step}", actions[agent_id])

            # Environment interaction
            self.env.unwrapped.set_detail_actor_info(detail)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            
            # Environment output validation
            for agent_id in self.agent_ids:
                if agent_id in next_obs:
                    finite_check(f"next_obs_{agent_id}_step_{rollout_step}", next_obs[agent_id])
                if agent_id in rewards:
                    finite_check(f"reward_{agent_id}_step_{rollout_step}", rewards[agent_id])

            # Only store transitions during training, not evaluation
            if not self.is_eval_mode:
                # Store transitions for both agents with proper term_masks
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                
                # Pass terminated and truncated info for proper term_masks generation
                self.rmappo.add_experience_to_buffer(
                    obs=current_obs,  # Raw obs - scaling handled inside rmappo
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=done_any_dict,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos
                )

            # Count episodes using OR aggregation
            done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
            done_any = None
            for aid in self.agent_ids:
                d = done_any_dict[aid].to(torch.bool)
                done_any = d if done_any is None else (done_any | d)
            
            episode_increment = int(done_any.sum().item())
            
            # Skip episode counting during evaluation or when flag is set
            if self.is_eval_mode or self._skip_episode_once:
                episode_increment = 0
                if self._skip_episode_once:
                    self._skip_episode_once = False
                    
            episode_count += episode_increment

            # Update current observations for next round
            current_obs = next_obs

        # Store final observations for bootstrapping
        self.rmappo.store_next_obs(next_obs)
        
        # Only update networks during training, not evaluation
        if not self.is_eval_mode:
            # Update both networks (complete rollout collected)
            stats = self.rmappo.update()

            # Increment training round counter
            self.train_updates += 1
            self.global_episodes += episode_count

            # Dual network logging - separate loss tracking per agent (UPDATED KEYS)
            if stats:
                payload = {
                    # Per-agent metrics from dual networks (UPDATED KEYS)
                    "loss/actor": {aid: stats.get(f"policy_loss/{aid}", 0.0) for aid in self.agent_ids},
                    "loss/critic": {aid: stats.get(f"value_loss/{aid}", 0.0) for aid in self.agent_ids},
                    "grad_norm/actor": {aid: stats.get(f"actor_grad_norm/{aid}", 0.0) for aid in self.agent_ids},
                    "grad_norm/critic": {aid: stats.get(f"critic_grad_norm/{aid}", 0.0) for aid in self.agent_ids},
                    
                    # Shared metrics (averaged from both networks where applicable) - REMOVED model/ prefix
                    "policy/entropy": np.mean([stats.get(f"dist_entropy/{aid}", 0.0) for aid in self.agent_ids]),
                    "ppo/ratio_mean": np.mean([stats.get(f"ratio/{aid}", 1.0) for aid in self.agent_ids]),
                    
                    # Clear separation of step types in logging (UPDATED KEYS)
                    "train/collection_steps": self.global_step,
                    "train/training_rounds": self.train_updates,
                    "train/global_episodes": self.global_episodes,
                }
                # Clean None values
                payload = {k: v for k, v in payload.items() if v is not None}
                
                # Use collection steps for x-axis, but include training rounds info
                self.metrics.push_update(self.global_step, payload)

            # Push force statistics every rollout
            self._push_current_rollout_force_statistics(detail)
            
            # Memory cleanup every 10 rollouts
            if self.train_updates % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Return empty stats during evaluation
            stats = {}

        # Update current observations for next rollout
        self._current_obs = next_obs

        return next_obs

    def mark_skip_episode_once(self):
        """Mark to skip episode counting once for milestone evaluation."""
        self._skip_episode_once = True

    def _push_current_rollout_force_statistics(self, detail):
        """Push current rollout's force statistics to WandB."""
        if "mean_actions" not in detail:
            return
        
        force_payload = {}
        
        # Push per-agent force statistics
        for aid in self.agent_ids:
            if aid in detail["mean_actions"]:
                forces = detail["mean_actions"][aid]  # [num_envs, 3]
                
                # Force validation
                finite_check(f"force_stats_{aid}", forces)
                mean_forces = forces.mean(dim=0)  # [3] - cross-environment mean
                force_payload.update({
                    f"forces/{aid}_fx_mean": float(mean_forces[0].item()),
                    f"forces/{aid}_fy_mean": float(mean_forces[1].item()),
                    f"forces/{aid}_fz_mean": float(mean_forces[2].item()),
                })
        
        # Push to MetricsHub for WandB logging
        if force_payload:
            self.metrics.push_update(self.global_step, force_payload)

    def run_until(self, max_global_steps: int):
        """Run training until reaching maximum collection steps."""
        obs, _ = self.env.reset()
        self._current_obs = obs
        while self.global_step < max_global_steps:
            self.execute_training_step()


class RMAPPOMilestoneEvaluator:
    """Milestone evaluator adapted for dual-network rMAPPO with coordinated evaluation and lifecycle markers."""
    
    def __init__(self, env, rmappo_wrapper, topk_mgr, metrics_hub, log_dir, agent_ids):
        self.env = env
        self.rmappo = rmappo_wrapper
        self.topk = topk_mgr
        self.metrics = metrics_hub
        self.log_dir = log_dir
        self.agent_ids = agent_ids

    def run_evaluation(self, milestone: int, global_step: int) -> dict:
        """Handle milestone evaluation and model saving for dual networks with lifecycle markers."""
        # In-place evaluation with return normalization
        return_norm, num_eps = self._run_single_evaluation_episode()
        
        # Scale return_norm by 1000 for milestone tracking
        milestone_return = return_norm * 1000
        
        print(f"[EVAL] Milestone {milestone}: return_norm={return_norm:.4f}, scaled={milestone_return:.2f}")

        # Extract model weights from both networks
        model_state = self._extract_dual_model_state()

        # Update TopK and save (using scaled return)
        self.topk.update(milestone_return, model_state, milestone)
        ckpt_path = os.path.join(self.log_dir, f"topk_milestone_{milestone}.pth")
        self.topk.save_checkpoint(ckpt_path, self.agent_ids)

        # Push milestone logs with return_norm*1000
        payload = {
            "eval/return_mean": float(milestone_return),
            "eval/num_episodes": int(num_eps),
            "milestone/topk_best_score": float(milestone_return),
            "milestone/topk_avg_score": float(milestone_return),
            "milestone/topk_count": 1,
            "milestone/latest_completed": int(milestone),
            # Additional debug info
            "eval/return_norm": float(return_norm),  # Original normalized return
        }
        self.metrics.push_milestone(global_step, milestone, payload)
        
        print(f"[EVAL] Uploaded milestone metrics: scaled_return={milestone_return:.2f}")

        # Reset RNN states after evaluation for both networks
        print(f"[EVAL] Resetting RNN states for both networks after evaluation...")
        for aid in self.agent_ids:
            self.rmappo.rnn_states[aid]["actor"].zero_()
            self.rmappo.rnn_states[aid]["critic"].zero_()

        return {"skip_episode_once": True}

    def _run_single_evaluation_episode(self):
        """Run single environment evaluation episode with dual network coordination."""
        active_env = 0
        target_episodes = 1
        
        print(f"[EVAL] Starting in-place dual rMAPPO evaluation (env0 only, 1 episode)...")
        
        env = getattr(self.env, "unwrapped", self.env)
        obs, _ = env.reset()
        print(f"[EVAL] Environment reset for independent evaluation")
        
        num_envs = len(obs[self.agent_ids[0]])
        ep_returns = torch.zeros(num_envs, device=self.rmappo.device)
        ep_steps = torch.zeros(num_envs, dtype=torch.int64, device=self.rmappo.device)
        completed_return_norms = []
        
        # Reset RNN states for evaluation (both networks) using 2D format
        for aid in self.agent_ids:
            mappo_args = self.rmappo.params.get('mappo_args', {})
            H = mappo_args.get('hidden_size', 256)
            self.rmappo.rnn_states[aid]["actor"] = torch.zeros(num_envs, H, device=self.rmappo.device)
            self.rmappo.rnn_states[aid]["critic"] = torch.zeros(num_envs, H, device=self.rmappo.device)
        
        # Get current global step for StepTracer (frozen during evaluation)
        training_global_step = getattr(env, '_trainer_global_step', 0)
        eval_step_counter = 0
        
        with torch.no_grad():
            while len(completed_return_norms) < target_episodes:
                eval_step_counter += 1
                
                # Get current observations
                if hasattr(env, '_get_observations'):
                    current_obs = env._get_observations()
                elif hasattr(env, 'observation_manager'):
                    current_obs = env.observation_manager.compute()
                else:
                    current_obs = obs
                
                # Observation validation during evaluation
                for agent_id in self.agent_ids:
                    if agent_id in current_obs:
                        finite_check(f"eval_obs_{agent_id}_step_{eval_step_counter}", current_obs[agent_id])

                # Use rmappo.select_actions which handles obs scaling internally
                actions_dict, detail_info = self.rmappo.select_actions(
                    current_obs,
                    add_noise=False,
                    deterministic=True
                )
                
                # Action validation during evaluation
                for agent_id in self.agent_ids:
                    if agent_id in actions_dict:
                        finite_check(f"eval_action_{agent_id}_step_{eval_step_counter}", actions_dict[agent_id])
                
                # Apply complete action masking - only env0 executes real actions
                for aid, act in actions_dict.items():
                    if act.ndim == 2:
                        masked_actions = torch.zeros_like(act)
                        masked_actions[active_env] = act[active_env]
                        actions_dict[aid] = masked_actions
                
                # Update detail info with same masking for consistency
                detail_info = {
                    "mean_actions": {aid: actions_dict[aid].clone() for aid in self.agent_ids},
                    "noise_actions": {aid: torch.zeros_like(actions_dict[aid]) for aid in self.agent_ids}
                }
                env.set_detail_actor_info(detail_info)
                
                obs, rewards, terminated, truncated, infos = env.step(actions_dict)
                
                # Environment output validation during evaluation
                for agent_id in self.agent_ids:
                    if agent_id in rewards:
                        finite_check(f"eval_reward_{agent_id}_step_{eval_step_counter}", rewards[agent_id])
                
                # StepTracer with forced console logging every 10 steps
                if (eval_step_counter % 10 == 0 and hasattr(env, 'step_tracer') and 
                    env.step_tracer is not None):
                    # Temporarily enable console logging for evaluation visibility
                    original_logging = env.step_tracer.enable_console_logging
                    env.step_tracer.enable_console_logging = True
                    # Use frozen training step (stable during evaluation)
                    env.step_tracer.maybe_print_step(env, rewards, training_global_step, force_print=True)
                    env.step_tracer.enable_console_logging = original_logging
                
                # Accumulate env0 rewards and steps (sum of both agents)
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
                    
                    # Reset counters
                    ep_returns[active_env] = 0.0
                    ep_steps[active_env] = 0
                    
                    if len(completed_return_norms) >= target_episodes:
                        break
                
                # Memory cleanup every 100 steps
                if eval_step_counter % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_return_norms = completed_return_norms[:target_episodes]
        avg_return_norm = sum(final_return_norms) / max(1, len(final_return_norms))
        
        print(f"[EVAL] Completed: {len(final_return_norms)} episodes, Average return_norm: {avg_return_norm:.4f}")
        
        # Reset environment back to training state
        _, _ = env.reset()
        print(f"[EVAL] Environment reset back to training mode")
        
        return avg_return_norm, len(final_return_norms)

    def _extract_dual_model_state(self):
        """Extract model state for dual networks checkpoint saving."""
        model_state = {}
        
        # Extract state dicts from both networks
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
        self.subs = defaultdict(list)  # Event subscribers
        self.update_ring = deque(maxlen=ring)  # Update history ring buffer

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

    def push_milestone(self, step: int, milestone: int, summary: dict) -> None:
        """Push milestone completion summary."""
        self._emit("milestone_summary", {"step": step, "milestone": milestone, **summary})


class WandBLogger:
    """Optimized WandB logger adapted for dual-network rMAPPO metrics with 7-prefix structure."""
    
    # UPDATED: Dual network agent metrics mapping with new prefix structure
    AGENT_METRICS_MAP = {
        'loss/actor': 'train/actor_loss_{}',      # Updated key format
        'loss/critic': 'train/critic_loss_{}',    # Updated key format
        'grad_norm/actor': 'grad/{}_actor',       # Updated prefix
        'grad_norm/critic': 'grad/{}_critic',     # Updated prefix
    }
    
    # UPDATED: Global metrics with new 7-prefix structure
    GLOBAL_METRICS_MAP = {
        "policy/entropy": "policy/entropy",              # Updated prefix
        "ppo/ratio_mean": "ppo/ratio_mean",              # Already correct
        "train/collection_steps": "train/collection_steps",  # Already correct
        "train/training_rounds": "train/training_rounds",    # Already correct
        "train/global_episodes": "train/global_episodes",    # Already correct
        "eval/return_mean": "milestone/actor_return",
        "milestone/topk_best_score": "milestone/topk_best_return",
        "milestone/latest_completed": "milestone/latest_completed",
        # Per-agent force statistics
        "forces/robot_fx_mean": "forces/robot_fx_mean",
        "forces/robot_fy_mean": "forces/robot_fy_mean",
        "forces/robot_fz_mean": "forces/robot_fz_mean",
        "forces/human_fx_mean": "forces/human_fx_mean",
        "forces/human_fy_mean": "forces/human_fy_mean", 
        "forces/human_fz_mean": "forces/human_fz_mean",
    }
    
    def __init__(self, project_name: str = "surgical_robot_rmappo", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.run = None
        
        if not self.enabled:
            print("[WANDB] Disabled")

    def initialize_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """Initialize WandB run with enhanced dual-network rMAPPO configuration tracking."""
        if not self.enabled:
            return
        
        # Direct initialization
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config,
            tags=["rmappo", "multi-agent", "surgical-robot", "rnn", "on-policy", "dual-network", "tanh-gaussian", "obs-scaling", "fail-fast", "gradient-monitoring", "7-prefix-structure"],
            notes="Multi-environment parallel dual-network rMAPPO training with Tanh-Gaussian action domain, obs scaling, term_masks GAE support, FAIL-FAST NaN/Inf detection, 12-core gradient monitoring system, and unified 7-prefix WandB logging structure",
            settings=wandb.Settings(start_method="thread")
        )
        
        # Log dual-network rMAPPO-specific configuration using unified mappo_args
        mappo_cfg = config.get("mappo_args", config.get("mappo", {}))  # Fallback for compatibility
        
        wandb.config.update({
            "rollout_horizon": config.get("rollout_horizon", 256),
            "ppo_epoch": mappo_cfg.get("ppo_epoch", 10),
            "num_mini_batch": mappo_cfg.get("num_mini_batch", 4),
            "clip_param": mappo_cfg.get("clip_param", 0.2),
            "value_loss_coef": mappo_cfg.get("value_loss_coef", 0.5),
            "entropy_coef": mappo_cfg.get("entropy_coef", 0.01),
            "reward_scale": 0.01,
            "agent_mode": "robot_human_dual_network",
            "reward_components": "trajectory+progress+potential_field",
            "termination_mode": "direct_obstacle_collision",
            "completion_threshold": config.get("reward_parameters", {}).get("completion_threshold", 0.01),
            # RNN configuration
            "hidden_size": mappo_cfg.get("hidden_size", 256),
            "recurrent_N": mappo_cfg.get("recurrent_N", 1),
            "data_chunk_length": mappo_cfg.get("data_chunk_length", 16),
            # Loss configuration
            "huber_delta": mappo_cfg.get("huber_delta", 1.0),
            "use_popart": mappo_cfg.get("use_popart", False),
            "use_valuenorm": mappo_cfg.get("use_valuenorm", False),
            "use_clipped_value_loss": mappo_cfg.get("use_clipped_value_loss", False),
            # Dual network info
            "network_architecture": "dual_independent",
            "network_init": "robot_copy_from_human",
            "rnn_state_format": "external_2d_internal_3d",
            # New features
            "action_distribution": "tanh_gaussian",
            "action_domain": "bounded_minus_one_to_one",
            "obs_scaling_enabled": bool(config.get("obs_scaling", {}).get("factors")),
            "obs_scaling_factors": config.get("obs_scaling", {}).get("factors", "none"),
            "gae_term_masks_enabled": True,
            "step_counting_method": "collection_steps_separate_from_training_rounds",
            "evaluation_mode": "read_only_deterministic_with_masking",
            "fail_fast_enabled": True,  # Document FAIL-FAST mode
            "gradient_clipping_disabled": True,  # Document no clipping
            "gradient_monitoring_enabled": True,  # Document 12-core monitoring
            "lifecycle_markers_enabled": True,  # Document eval->train markers
            "wandb_logging_structure": "7_prefix_unified",  # Document new structure
        })
        
        print(f"[WANDB] Successfully initialized with 7-prefix structure: {self.run.name}")

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
            if "milestone/topk_best_score" in ms:
                payload_to_log["milestone/topk_best_score"] = ms["milestone/topk_best_score"]
            if "milestone" in ms:
                payload_to_log["milestone/latest_completed"] = ms["milestone"]

            if payload_to_log:
                self.log_metrics(payload_to_log, step)

        hub.subscribe("milestone_summary", _on_ms)
        print("[WANDB] Attached to MetricsHub with 7-prefix dual-network rMAPPO metric mapping, FAIL-FAST feature tracking, and gradient monitoring system.")

    def log_metrics(self, metrics_data: Dict[str, Any], step: int) -> None:
        """Log metrics with dual-network rMAPPO-specific mapping using 7-prefix structure."""
        if not self.enabled or not metrics_data:
            return

        log_data = {}

        # Handle Per-Agent training metrics (for dual networks)
        if any(key in metrics_data and isinstance(metrics_data.get(key), dict) 
               for key in self.AGENT_METRICS_MAP.keys()):
            
            # Get agent IDs from first available per-agent metric
            agent_ids = None
            for source_key in self.AGENT_METRICS_MAP.keys():
                if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                    agent_ids = list(metrics_data[source_key].keys())
                    break
            
            if agent_ids:
                # Apply per-agent mapping with updated key format
                for source_key, target_pattern in self.AGENT_METRICS_MAP.items():
                    if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                        for agent_id in agent_ids:
                            if agent_id in metrics_data[source_key]:
                                log_data[target_pattern.format(agent_id)] = metrics_data[source_key][agent_id]

        # Handle global metrics with 7-prefix structure
        for src_key, dest_key in self.GLOBAL_METRICS_MAP.items():
            if src_key in metrics_data and metrics_data[src_key] is not None:
                log_data[dest_key] = metrics_data[src_key]

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
        # Preserve backward compatibility while allowing mappo config access
        self.mappo_cfg = self.params.get('mappo', {})
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Create configuration from YAML file."""
        return cls(config_path)
    
    def get_compute_device(self) -> str:
        """Get compute device (CUDA if available)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class TopKModelManager:
    """Manages top-K model collection and checkpoint saving."""
    
    def __init__(self, k: int = 10, mode: str = "max"):
        self.k = k  # Number of top models to keep
        self.mode = mode  # Ranking mode
        self.top_models: List[Tuple[float, Dict, int]] = []  # (performance, model_state, milestone)
    
    def update(self, performance: float, model_state: Dict[str, Any], milestone: int) -> None:
        """Update top-K models with new performance data."""
        # Add new model
        if len(self.top_models) < self.k:
            self.top_models.append((performance, model_state, milestone))
        elif performance > self.top_models[-1][0]:
            self.top_models[-1] = (performance, model_state, milestone)
        
        # Sort by performance descending
        self.top_models.sort(key=lambda x: x[0], reverse=True)
    
    def get_top_models(self) -> List[Tuple[float, Dict, int]]:
        """Get list of top-K models."""
        return self.top_models
    
    def save_checkpoint(self, filepath: str, agent_ids: List[str]) -> None:
        """Save checkpoint with top-K models (dual network support)."""
        checkpoint = {
            'agent_ids': agent_ids,
            'algorithm': 'rmappo_dual',
            'network_architecture': 'dual_independent',
            'rnn_state_format': 'external_2d_internal_3d',
            'action_distribution': 'tanh_gaussian',
            'obs_scaling_enabled': True,
            'fail_fast_enabled': True,  # Document FAIL-FAST mode
            'gradient_monitoring_enabled': True,  # Document gradient monitoring
            'wandb_logging_structure': '7_prefix_unified',  # Document new structure
            'step_counting_method': 'collection_steps_separate_from_training_rounds',
            'top_k_count': len(self.top_models),
            'top_k_models': []
        }
        
        for i, (performance, model_state, milestone) in enumerate(self.top_models):
            model_info = {
                'rank': i + 1,
                'performance': performance,
                'milestone': milestone
            }
            checkpoint['top_k_models'].append(model_info)
            
            # Save model state with rank prefix
            for key, value in model_state.items():
                checkpoint[f'rank_{i+1}_{key}'] = value
        
        torch.save(checkpoint, filepath)
        if self.top_models:
            scores = [m[0] for m in self.top_models]
            print(f"[TOP-K] Saved {len(self.top_models)} best dual models, scores: "
                  f"{scores[0]:.2f} ~ {scores[-1]:.2f}")


def save_final_rmappo_networks(log_directory: str, rmappo_wrapper, global_step: int, global_episodes: int, max_milestone_triggered: Optional[int]) -> None:
    """Save final dual rMAPPO networks to checkpoint file."""
    final_path = os.path.join(log_directory, "final_rmappo_dual_networks.pth")
    
    final_checkpoint = {
        'params': rmappo_wrapper.params,
        'agent_ids': rmappo_wrapper.agent_ids,
        'algorithm': 'rmappo_dual',
        'network_architecture': 'dual_independent',
        'rnn_state_format': 'external_2d_internal_3d',
        'action_distribution': 'tanh_gaussian',
        'obs_scaling_enabled': bool(rmappo_wrapper.params.get('obs_scaling', {}).get('factors')),
        'fail_fast_enabled': True,  # Document FAIL-FAST mode
        'gradient_monitoring_enabled': True,  # Document gradient monitoring
        'wandb_logging_structure': '7_prefix_unified',  # Document new structure
        'step_counting_method': 'collection_steps_separate_from_training_rounds',
        'global_steps_total': global_step,
        'training_rounds_total': getattr(rmappo_wrapper, 'train_updates', 0),
        'episodes_done_total': global_episodes,
        'max_milestone_triggered': max_milestone_triggered or 0,
        'rollout_horizon': rmappo_wrapper.T,
        # Use unified mappo_args for config
        'mappo_config': rmappo_wrapper.params.get('mappo_args', {}),
    }
    
    # Save both policy networks
    for aid in rmappo_wrapper.agent_ids:
        policy = rmappo_wrapper.policies[aid]
        final_checkpoint.update({
            f'{aid}_actor': policy.actor.state_dict(),
            f'{aid}_critic': policy.critic.state_dict(),
        })
    
    torch.save(final_checkpoint, final_path)
    print(f"[CHECKPOINT] Final dual rMAPPO networks saved: {final_path}")
    print(f"[CHECKPOINT] Final stats: collection_steps={global_step}, episodes={global_episodes}")
    print(f"[CHECKPOINT] Gradient monitoring: enabled with dump mechanism")


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser for dual rMAPPO training."""
    if config_path is None:
        # Adjust path based on actual file location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml')

    parser = argparse.ArgumentParser(description="Dual rMAPPO multi-environment parallel training with FAIL-FAST NaN/Inf detection, 12-core gradient monitoring, and unified 7-prefix WandB logging")
    parser.add_argument("--config", type=str, default=config_path)
    
    # Environment configuration
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
    parser.add_argument("--seed", type=int, default=42)
    
    # Training termination
    parser.add_argument(
        "--max_global_steps", 
        type=int, 
        default=0,  # 0 means unspecified, will use YAML config
        help="Stop after this many environment collection steps; if >0, it becomes the primary stop condition."
    )
    
    # Model management
    parser.add_argument("--top_k_models", type=int, default=10)
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable WandB logging with gradient monitoring and 7-prefix structure")
    
    return parser