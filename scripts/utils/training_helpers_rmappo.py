#!/usr/bin/env python3

"""
Training helper utilities for rMAPPO multi-environment parallel training.
Features unified training execution, milestone evaluation, and optimized WandB logging.
Adapted specifically for on-policy rollout collection and PPO updates.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pickle
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, DefaultDict, Deque
from collections import defaultdict, deque

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")


class RMAPPOTrainingRunner:
    """
    Unified training loop executor for rMAPPO with rollout collection.
    Features on-policy trajectory collection and unified global step tracking.
    """
    
    def __init__(self, env, rmappo_wrapper, metrics_hub, agent_ids, max_global_steps=None):
        self.env = env
        self.rmappo = rmappo_wrapper
        self.metrics = metrics_hub
        self.agent_ids = agent_ids
        self.global_step = 0  # Current training step
        self.global_episodes = 0  # Total episodes completed
        self._skip_episode_once = False  # Flag to skip episode counting once
        self._current_obs = None  # Current observations cache
        
        # rMAPPO specific parameters
        self.T = rmappo_wrapper.T  # Rollout horizon
        
        # Use the max_global_steps passed from trainer
        if max_global_steps is not None and max_global_steps > 0:
            self.max_global_steps = int(max_global_steps)
        else:
            self.max_global_steps = int(rmappo_wrapper.params.get('ppo', {}).get('max_global_steps', 200000))
        
        print(f"[RMAPPO RUNNER] Configured:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  Max steps: {self.max_global_steps}")

    def execute_training_step(self):
        """Execute one complete rollout and training update."""
        # Use current observations
        current_obs = self._current_obs
        if current_obs is None:
            if hasattr(self.env, "_get_observations"):
                current_obs = self.env._get_observations()
            else:
                current_obs, _ = self.env.reset()
            self._current_obs = current_obs
            
        episode_count = 0
        
        # Collect complete rollout (T steps) - no noise scheduling for rMAPPO
        for rollout_step in range(self.T):
            # Select actions (on-policy, no noise scaling)
            actions, detail = self.rmappo.select_actions(current_obs, add_noise=True, noise_scale=1.0)

            # Environment interaction
            self.env.unwrapped.set_detail_actor_info(detail)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            # Store joint transitions
            done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
            
            self.rmappo.add_experience_to_buffer(
                obs=current_obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=done_any_dict
            )

            # Count episodes using OR aggregation
            done_any = None
            for aid in self.agent_ids:
                d = done_any_dict[aid].to(torch.bool)
                done_any = d if done_any is None else (done_any | d)
            
            episode_increment = int(done_any.sum().item())
            if self._skip_episode_once:
                episode_increment = 0
                self._skip_episode_once = False
            episode_count += episode_increment

            # Update current observations for next round
            current_obs = next_obs

        # Store final observations for bootstrapping
        self.rmappo.store_next_obs(next_obs)
        
        # Update networks (complete rollout collected)
        stats = self.rmappo.update()

        # Update global counters
        self.global_step += self.T * self.rmappo.num_envs
        self.global_episodes += episode_count

        # Unified logging (removed noise_scale)
        if stats and stats.get("training/policy_updates", 0) > 0:
            payload = {
                # rMAPPO specific metrics
                "loss/actor": stats.get("loss/actor"),
                "loss/critic": stats.get("loss/critic"),
                "model/entropy": stats.get("model/entropy", 0.0),
                "model/ratio": stats.get("model/ratio", 1.0),
                "grad_norm/actor": stats.get("grad_norm/actor"),
                "grad_norm/critic": stats.get("grad_norm/critic"),
                # Global metrics
                "train/episodes_done": self.global_episodes,
                "training/policy_updates": stats.get("training/policy_updates", 0),
                "training/value_updates": stats.get("training/value_updates", 0),
            }
            # Clean None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            self.metrics.push_update(self.global_step, payload)

        # Push force statistics every rollout
        self._push_current_rollout_force_statistics(detail)

        # Step counting and environment synchronization
        actual_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(actual_env, "set_trainer_global_step"):
            actual_env.set_trainer_global_step(self.global_step)

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
        
        # Robot forces - current rollout all environments average
        if "robot" in detail["mean_actions"]:
            robot_forces = detail["mean_actions"]["robot"]  # [num_envs, 3]
            robot_mean = robot_forces.mean(dim=0)  # [3] - cross-environment mean
            force_payload.update({
                "forces/robot_fx_mean": float(robot_mean[0].item()),
                "forces/robot_fy_mean": float(robot_mean[1].item()),
                "forces/robot_fz_mean": float(robot_mean[2].item()),
            })
        
        # Human forces - current rollout all environments average  
        if "human" in detail["mean_actions"]:
            human_forces = detail["mean_actions"]["human"]  # [num_envs, 3]
            human_mean = human_forces.mean(dim=0)  # [3] - cross-environment mean
            force_payload.update({
                "forces/human_fx_mean": float(human_mean[0].item()),
                "forces/human_fy_mean": float(human_mean[1].item()),
                "forces/human_fz_mean": float(human_mean[2].item()),
            })
        
        # Push to MetricsHub for WandB logging
        if force_payload:
            self.metrics.push_update(self.global_step, force_payload)

    def run_until(self, max_global_steps: int):
        """Run training until reaching maximum steps."""
        obs, _ = self.env.reset()
        self._current_obs = obs
        while self.global_step < max_global_steps:
            self.execute_training_step()


class RMAPPOMilestoneEvaluator:
    """
    Milestone evaluator adapted for rMAPPO with single environment evaluation.
    Features normalized return calculation and proper action masking.
    """
    
    def __init__(self, env, rmappo_wrapper, topk_mgr, metrics_hub, log_dir, agent_ids):
        self.env = env
        self.rmappo = rmappo_wrapper
        self.topk = topk_mgr
        self.metrics = metrics_hub
        self.log_dir = log_dir
        self.agent_ids = agent_ids

    def run_evaluation(self, milestone: int, global_step: int) -> dict:
        """Handle milestone evaluation and model saving."""
        # In-place evaluation with return normalization
        return_norm, num_eps = self._run_single_evaluation_episode()
        
        # Scale return_norm by 1000 for milestone tracking
        milestone_return = return_norm * 1000
        
        print(f"[EVAL] Milestone {milestone}: return_norm={return_norm:.4f}, scaled={milestone_return:.2f}")

        # Extract model weights
        model_state = self._extract_model_state()

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

        # CRITICAL FIX: Reset RNN states after evaluation
        print(f"[EVAL] Resetting RNN states after evaluation...")
        self.rmappo.rnn_states_actor.zero_()
        self.rmappo.rnn_states_critic.zero_()

        return {"skip_episode_once": True}

    def _run_single_evaluation_episode(self):
        """Run single environment evaluation episode with proper action masking."""
        active_env = 0
        target_episodes = 1
        
        print(f"[EVAL] Starting in-place rMAPPO evaluation (env0 only, 1 episode)...")
        
        env = getattr(self.env, "unwrapped", self.env)
        obs, _ = env.reset()
        print(f"[EVAL] Environment reset for independent evaluation")
        
        num_envs = len(obs[self.agent_ids[0]])
        ep_returns = torch.zeros(num_envs, device=self.rmappo.device)
        ep_steps = torch.zeros(num_envs, dtype=torch.int64, device=self.rmappo.device)
        completed_return_norms = []
        
        # Reset RNN states for evaluation
        H = self.rmappo.params.get('ppo', {}).get('hidden_size', 256)
        N = num_envs * 2
        rnn_a = torch.zeros(N, H, device=self.rmappo.device)
        rnn_c = torch.zeros(N, H, device=self.rmappo.device)
        
        # Get current global step for StepTracer
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
                
                # Build obs tensors for rMAPPO
                obs_tensor, share_obs_tensor = self.rmappo.build_obs_tensors(current_obs)
                masks = torch.ones(obs_tensor.shape[0], 1, device=self.rmappo.device)
                
                # Select actions deterministically
                values, actions_norm, action_log_probs, rnn_a_new, rnn_c_new = self.rmappo.policy.get_actions(
                    share_obs_tensor, obs_tensor, rnn_a, rnn_c, masks, deterministic=True
                )
                
                # Convert to environment format
                env_actions = self.rmappo.actions_to_env_format(actions_norm)
                
                # Apply complete action masking - only env0 executes real actions
                for aid, act in env_actions.items():
                    if act.ndim == 2:
                        masked_actions = torch.zeros_like(act)
                        masked_actions[active_env] = act[active_env]
                        env_actions[aid] = masked_actions
                
                # Create detail info for StepTracer (after masking)
                detail_info = {
                    "mean_actions": {
                        "human": env_actions["human"],
                        "robot": env_actions["robot"],
                    },
                    "noise_actions": {
                        "human": torch.zeros_like(env_actions["human"]),
                        "robot": torch.zeros_like(env_actions["robot"]),
                    }
                }
                env.set_detail_actor_info(detail_info)
                
                obs, rewards, terminated, truncated, infos = env.step(env_actions)
                
                # Call StepTracer every 10 eval steps
                if (hasattr(env, 'step_tracer') and env.step_tracer is not None and 
                    eval_step_counter % 10 == 0):
                    original_logging = env.step_tracer.enable_console_logging
                    env.step_tracer.enable_console_logging = True
                    env.step_tracer.maybe_print_step(env, rewards, training_global_step, force_print=True)
                    env.step_tracer.enable_console_logging = original_logging
                
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
                    
                    # Reset counters
                    ep_returns[active_env] = 0.0
                    ep_steps[active_env] = 0
                    
                    if len(completed_return_norms) >= target_episodes:
                        break
                
                # RNN state rollover (evaluation no reset unless done)
                rnn_a, rnn_c = rnn_a_new, rnn_c_new
        
        final_return_norms = completed_return_norms[:target_episodes]
        avg_return_norm = sum(final_return_norms) / max(1, len(final_return_norms))
        
        print(f"[EVAL] Completed: {len(final_return_norms)} episodes, Average return_norm: {avg_return_norm:.4f}")
        
        # Reset environment back to training state
        _, _ = env.reset()
        print(f"[EVAL] Environment reset back to training mode")
        
        return avg_return_norm, len(final_return_norms)

    def _extract_model_state(self):
        """Extract model state for checkpoint saving."""
        model_state = {}
        
        # rMAPPO uses policy wrapper, extract underlying networks
        policy = self.rmappo.policy
        prefix = 'shared'  # rMAPPO uses shared policy
        
        model_state.update({
            f'{prefix}_actor': policy.actor.state_dict(),
            f'{prefix}_critic': policy.critic.state_dict(),
        })
        
        return model_state


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
    Optimized WandB logger adapted for rMAPPO metrics.
    Features network configuration logging and milestone tracking.
    """
    
    # rMAPPO-specific metric mappings
    AGENT_METRICS_MAP = {
        'loss/actor': 'train/{}/actor_loss',
        'loss/critic': 'train/{}/critic_loss',
        'grad_norm/actor': 'model/{}/grad_norm_actor',
        'grad_norm/critic': 'model/{}/grad_norm_critic',
    }
    
    # Removed exploration/noise_scale from global metrics
    GLOBAL_METRICS_MAP = {
        "model/entropy": "model/entropy",
        "model/ratio": "model/ppo_ratio",
        "train/episodes_done": "train/global_episodes", 
        "training/policy_updates": "train/policy_updates",
        "training/value_updates": "train/value_updates",
        "eval/return_mean": "milestone/actor_return",
        "milestone/topk_best_score": "milestone/topk_best_return",
        "milestone/latest_completed": "milestone/latest_completed",
        # Force statistics
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
        """Initialize WandB run with enhanced rMAPPO configuration tracking."""
        if not self.enabled:
            return
        
        # Direct initialization
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config,
            tags=["rmappo", "multi-agent", "surgical-robot", "rnn", "on-policy"],
            notes="Multi-environment parallel rMAPPO training with RNN, rollout collection, and PPO updates",
            settings=wandb.Settings(start_method="thread")
        )
        
        # Log rMAPPO-specific configuration
        ppo_cfg = config.get("ppo", {})
        
        wandb.config.update({
            "rollout_horizon": config.get("rollout_horizon", 256),
            "ppo_epoch": ppo_cfg.get("ppo_epoch", 10),
            "num_mini_batch": ppo_cfg.get("num_mini_batch", 4),
            "clip_param": ppo_cfg.get("clip_param", 0.2),
            "value_loss_coef": ppo_cfg.get("value_loss_coef", 0.5),
            "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
            "reward_scale": 0.01,
            "agent_mode": "robot_human_shared",
            "reward_components": "trajectory+progress+potential_field",
            "termination_mode": "direct_obstacle_collision",
            "completion_threshold": config.get("reward_parameters", {}).get("completion_threshold", 0.01),
            # RNN configuration
            "hidden_size": ppo_cfg.get("hidden_size", 256),
            "recurrent_N": ppo_cfg.get("recurrent_N", 1),
            "data_chunk_length": ppo_cfg.get("data_chunk_length", 16),
            # Loss configuration
            "huber_delta": ppo_cfg.get("huber_delta", 1.0),
            "use_popart": ppo_cfg.get("use_popart", False),
            "use_valuenorm": ppo_cfg.get("use_valuenorm", False),
            "use_clipped_value_loss": ppo_cfg.get("use_clipped_value_loss", False),
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
            if "milestone/topk_best_score" in ms:
                payload_to_log["milestone/topk_best_score"] = ms["milestone/topk_best_score"]
            if "milestone" in ms:
                payload_to_log["milestone/latest_completed"] = ms["milestone"]

            if payload_to_log:
                self.log_metrics(payload_to_log, step)

        hub.subscribe("milestone_summary", _on_ms)
        print("[WANDB] Attached to MetricsHub with rMAPPO metric mapping.")

    def log_metrics(self, metrics_data: Dict[str, Any], step: int) -> None:
        """Log metrics with rMAPPO-specific mapping."""
        if not self.enabled or not metrics_data:
            return

        log_data = {}

        # Handle Per-Agent training metrics (for rMAPPO, we have shared policy)
        if any(key in metrics_data and isinstance(metrics_data.get(key), dict) 
               for key in self.AGENT_METRICS_MAP.keys()):
            
            # Get agent IDs from first available per-agent metric
            agent_ids = None
            for source_key in self.AGENT_METRICS_MAP.keys():
                if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                    agent_ids = list(metrics_data[source_key].keys())
                    break
            
            if agent_ids:
                # Apply per-agent mapping
                for source_key, target_pattern in self.AGENT_METRICS_MAP.items():
                    if source_key in metrics_data and isinstance(metrics_data[source_key], dict):
                        for agent_id in agent_ids:
                            if agent_id in metrics_data[source_key]:
                                log_data[target_pattern.format(agent_id)] = metrics_data[source_key][agent_id]

        # Handle global metrics
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
        self.ppo_cfg = self.params.get('ppo', {})
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Create configuration from YAML file."""
        return cls(config_path)
    
    def get_compute_device(self) -> str:
        """Get compute device (CUDA if available)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class TopKModelManager:
    """
    Manages top-K model collection and checkpoint saving.
    Features raw performance tracking and automatic sorting by performance.
    """
    
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
        """Save checkpoint with top-K models."""
        checkpoint = {
            'agent_ids': agent_ids,
            'algorithm': 'rmappo',
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
            print(f"[TOP-K] Saved {len(self.top_models)} best models, scores: "
                  f"{scores[0]:.2f} ~ {scores[-1]:.2f}")


def save_final_rmappo_networks(log_directory: str, rmappo_wrapper, global_step: int, global_episodes: int, max_milestone_triggered: Optional[int]) -> None:
    """Save final rMAPPO networks to checkpoint file."""
    final_path = os.path.join(log_directory, "final_rmappo_networks.pth")
    
    final_checkpoint = {
        'params': rmappo_wrapper.params,
        'agent_ids': rmappo_wrapper.agent_ids,
        'algorithm': 'rmappo',
        'global_steps_total': global_step,
        'episodes_done_total': global_episodes,
        'max_milestone_triggered': max_milestone_triggered or 0,
        'rollout_horizon': rmappo_wrapper.T,
        'ppo_config': rmappo_wrapper.params.get('ppo', {}),
    }
    
    # Save policy networks
    policy = rmappo_wrapper.policy
    final_checkpoint.update({
        'shared_actor': policy.actor.state_dict(),
        'shared_critic': policy.critic.state_dict(),
    })
    
    torch.save(final_checkpoint, final_path)
    print(f"[CHECKPOINT] Final rMAPPO networks saved: {final_path}")


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser for rMAPPO training."""
    if config_path is None:
        # Adjust path based on actual file location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml')

    parser = argparse.ArgumentParser(description="rMAPPO multi-environment parallel training")
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
        help="Stop after this many global training steps; if >0, it becomes the primary stop condition."
    )
    
    # Model management
    parser.add_argument("--top_k_models", type=int, default=10)
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    
    return parser