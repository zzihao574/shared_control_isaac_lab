#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Enhanced version with configurable network architecture and exponential noise decay support.

Features:
- TrainingRunner: Unified training loop execution with noise scheduling (no old reward_logger dependency)
- MilestoneEvaluator: Milestone evaluation and TopK model management  
- MetricsHub: Unified data pipeline for logging
- Configurable network support and intelligent noise scheduling
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


class TrainingRunner:
    """
    Unified training loop executor: rollout→replay→update→log→count with noise scheduling.
    MODIFIED: Removed dependency on reward_logger (can be None)
    
    Features:
    - Exponential noise decay scheduling (fast early, slow later)
    - Unified global step tracking
    - Episode counting with skip mechanism for milestone evaluation
    - Metrics collection and WandB logging
    """
    
    def __init__(self, env, maddpg, replay, metrics_hub, reward_logger, agent_ids):
        self.env = env
        self.maddpg = maddpg
        self.replay = replay
        self.metrics = metrics_hub
        self.reward_logger = reward_logger  # Can be None now
        self.agent_ids = agent_ids
        self.global_step = 0
        self.global_episodes = 0
        self._skip_episode_once = False
        self._current_obs = None
        
        # Load exploration parameters for noise scheduling
        expl = self.maddpg.params.get("exploration", {})
        self.sigma_start = float(expl.get("sigma_start", 0.7))
        self.sigma_end = float(expl.get("sigma_end", 0.1))
        self.decay_k = float(expl.get("decay_k", 6.0))
        
        # Get max_global_steps for noise scheduling
        maddpg_cfg = self.maddpg.params.get('maddpg_config', {})
        self.max_global_steps = int(maddpg_cfg.get('max_global_steps', 200000))
        
        print(f"[NOISE SCHEDULE] Configured exponential decay:")
        print(f"  Start: {self.sigma_start}, End: {self.sigma_end}, k: {self.decay_k}")
        print(f"  Max steps: {self.max_global_steps}")

    def _calculate_noise_scale(self) -> float:
        """Calculate current noise scaling factor (exponential decay: fast early, slow later)."""
        if self.max_global_steps <= 0:
            return self.sigma_start
            
        # ratio ∈ [0,1]
        ratio = min(1.0, float(self.global_step) / float(max(1, self.max_global_steps)))
        
        # Exponential decay: noise_scale = sigma_end + (sigma_start - sigma_end) * exp(-k * ratio)
        noise_scale = self.sigma_end + (self.sigma_start - self.sigma_end) * math.exp(-self.decay_k * ratio)
        
        return noise_scale

    def run_step(self):
        """Execute one training step with noise scheduling."""
        # Use current observations
        current_obs = self._current_obs
        if current_obs is None:
            # Prioritize environment's real-time observation function; reset if none
            if hasattr(self.env, "_get_observations"):
                current_obs = self.env._get_observations()
            else:
                current_obs, _ = self.env.reset()
            self._current_obs = current_obs
            
        # 1) Calculate noise scaling factor
        noise_scale = self._calculate_noise_scale()
        
        # 2) Select actions (with noise, training mode, using global noise scheduling)
        actions, detail = self.maddpg.select_actions(current_obs, add_noise=True, noise_scale=noise_scale)

        # 3) Let environment record actor detail info
        self.env.unwrapped.set_detail_actor_info(detail)

        # 4) Environment interaction
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # 5) Store joint transitions
        done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}

        
        self.maddpg.store_joint_transitions(
            obs=current_obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=done_any_dict
        )

        # 6) Update networks
        stats = self.maddpg.update()

        # 7) Count episodes (OR aggregation)
        done_any = None
        for aid in self.agent_ids:
            d = done_any_dict[aid].to(torch.bool)
            done_any = d if done_any is None else (done_any | d)
        
        episode_increment = int(done_any.sum().item())
        if self._skip_episode_once:
            episode_increment = 0
            self._skip_episode_once = False
        self.global_episodes += episode_increment

        # 8) Unified logging with noise scheduling information
        if stats and stats.get("training/updates", 0) > 0:
            payload = {
                "train/actor_loss": stats.get("loss/actor/avg"),
                "train/critic_loss": stats.get("loss/critic/avg"),
                "model/q_mean": stats.get("q_mean/avg"),
                "model/q_std": stats.get("q_std/avg"),
                "model/td_error_mean": stats.get("model/td_error_mean"),
                "model/td_rmse": stats.get("model/td_rmse"),
                "replay/buffer_size": len(self.maddpg.replay) if hasattr(self.maddpg, "replay") else None,
                "train/episodes_done": self.global_episodes,
                "train/updates": stats.get("training/updates"),
                "exploration/noise_scale": noise_scale,
                "exploration/sigma_ratio": noise_scale / self.sigma_start if self.sigma_start > 0 else 0,
            }
            # Clean None values
            payload = {k: v for k, v in payload.items() if v is not None}
            self.metrics.push_update(self.global_step, payload)

        # 9) Step counting + env synchronization
        self.global_step += 1
        actual_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(actual_env, "set_trainer_global_step"):
            actual_env.set_trainer_global_step(self.global_step)

        # 10) Update current observations for next round
        self._current_obs = next_obs

        return next_obs

    def mark_skip_episode_once(self):
        """Mark to skip episode counting once (for milestone evaluation)."""
        self._skip_episode_once = True

    def run_until(self, max_global_steps: int):
        """Run until reaching maximum steps."""
        obs, _ = self.env.reset()
        self._current_obs = obs
        while self.global_step < max_global_steps:
            self.run_step()


class MilestoneEvaluator:
    """
    Milestone evaluator: triggers evaluation→TopK→logging→return skip flag.
    
    Features:
    - Single environment evaluation for efficiency
    - TopK model management
    - Milestone logging to MetricsHub
    """
    
    def __init__(self, env, maddpg, topk_mgr, metrics_hub, log_dir, agent_ids):
        self.env = env
        self.maddpg = maddpg
        self.topk = topk_mgr
        self.metrics = metrics_hub
        self.log_dir = log_dir
        self.agent_ids = agent_ids

    def handle(self, milestone: int, global_step: int) -> dict:
        """Handle milestone trigger."""
        # 1) In-place evaluation
        avg_return, num_eps = self._evaluate_env0_once()

        # 2) Extract model weights
        model_state = self._extract_model_state()

        # 3) Update TopK and save
        self.topk.update(avg_return, model_state, milestone)
        ckpt_path = os.path.join(self.log_dir, f"topk_milestone_{milestone}.pth")
        self.topk.save_checkpoint(ckpt_path, self.agent_ids)

        # 4) Push milestone logs
        payload = {
            "eval/return_mean": float(avg_return),
            "eval/num_episodes": int(num_eps),
            "milestone/topk_best_score": float(avg_return),
            "milestone/topk_avg_score": float(avg_return),
            "milestone/topk_count": 1,
            "milestone/latest_completed": int(milestone),
        }
        self.metrics.push_milestone(global_step, milestone, payload)

        # 5) Notify training loop to skip episode counting once
        return {"skip_episode_once": True}

    def _evaluate_env0_once(self):
        """In-place evaluation (env0 single episode)."""
        active_env = 0
        target_episodes = 1
        
        print(f"[EVAL] Starting in-place evaluation (env0 only, 1 episode)...")
        
        env = getattr(self.env, "unwrapped", self.env)
        obs, _ = env.reset()
        print(f"[EVAL] Environment reset for independent evaluation")
        
        num_envs = len(obs[self.agent_ids[0]])
        ep_returns = torch.zeros(num_envs, device='cuda' if torch.cuda.is_available() else 'cpu')
        completed_returns = []
        
        with torch.no_grad():
            while len(completed_returns) < target_episodes:
                # Get current observations
                if hasattr(env, '_get_observations'):
                    current_obs = env._get_observations()
                elif hasattr(env, 'observation_manager'):
                    current_obs = env.observation_manager.compute()
                else:
                    current_obs = obs
                
                # Select actions (deterministic, no noise)
                actions, _ = self.maddpg.select_actions(current_obs, add_noise=False, noise_scale=0.0)
                
                # Only env0 executes real actions, others are masked to zero
                for aid, act in actions.items():
                    if act.ndim == 2:
                        masked = torch.zeros_like(act)
                        masked[active_env] = act[active_env]
                        actions[aid] = masked
                
                obs, rewards, terminated, truncated, infos = env.step(actions)
                
                # Only accumulate env0 rewards
                step_rewards = torch.stack([rewards[aid] for aid in self.agent_ids])
                avg_step_rewards = step_rewards.mean(dim=0)
                ep_returns[active_env] += avg_step_rewards[active_env]
                
                # Check if env0 is complete
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                done_any = None
                for aid in self.agent_ids:
                    d = done_any_dict[aid].to(torch.bool)
                    done_any = d if done_any is None else (done_any | d)
                
                if done_any[active_env]:
                    ret0 = float(ep_returns[active_env].item())
                    completed_returns.append(ret0)
                    ep_returns[active_env] = 0.0
                    
                    if len(completed_returns) >= target_episodes:
                        break
        
        final_returns = completed_returns[:target_episodes]
        avg_return = sum(final_returns) / max(1, len(final_returns))
        
        print(f"[EVAL] Completed: {len(final_returns)} episodes, Average return: {avg_return:.3f}")
        
        # Reset environment back to training state
        _, _ = env.reset()
        print(f"[EVAL] Environment reset back to training mode")
        
        return avg_return, len(final_returns)

    def _extract_model_state(self):
        """Extract model state for checkpoint saving."""
        model_state = {}
        for agent_id in self.agent_ids:
            agent = self.maddpg.agents[agent_id]
            prefix = f'{agent_id}'
            model_state.update({
                f'{prefix}_actor': agent.actor.state_dict(),
                f'{prefix}_critic': agent.critic.state_dict(),
                f'{prefix}_actor_target': agent.actor_target.state_dict(),
                f'{prefix}_critic_target': agent.critic_target.state_dict()
            })
        return model_state


class MetricsHub:
    """
    Single-exit metrics bus for unified data pipeline.
    
    Features:
    - Event-based subscription system
    - Ring buffer for update history
    - Support for step, episode, milestone, and buffer status events
    """
    
    def __init__(self, ring: int = 100):
        self.subs: DefaultDict[str, list[Callable[[dict], None]]] = defaultdict(list)
        self.episode_state: dict[int, dict] = {}
        self.milestone_scores: DefaultDict[int, dict] = defaultdict(dict)
        self.update_ring: Deque[dict] = deque(maxlen=ring)

    def subscribe(self, event: str, handler: Callable[[dict], None]) -> None:
        """Subscribe to an event type with a handler function."""
        self.subs[event].append(handler)

    def _emit(self, event: str, payload: dict) -> None:
        """Emit an event to all subscribers."""
        for h in self.subs.get(event, []):
            h(payload)

    def push_step(self, step: int, env_ids: list[int], payload: dict) -> None:
        """Push step-level data."""
        self._emit("step", {"step": step, "env_ids": env_ids, **payload})

    def push_episode(self, step: int, env_id: int, summary: dict) -> None:
        """Push episode completion data."""
        self.episode_state[env_id] = summary
        self._emit("episode", {"step": step, "env_id": env_id, **summary})

    def push_update(self, step: int, stats: dict) -> None:
        """Push training update statistics."""
        if not stats:
            return
        data = {"step": step, **stats}
        self.update_ring.append(data)
        self._emit("update", data)

    def push_milestone(self, step: int, milestone: int, summary: dict) -> None:
        """Push milestone completion summary."""
        scores = summary.get("scores", {})
        for eid, sc in scores.items():
            self.milestone_scores[milestone][eid] = sc
        self._emit("milestone_summary", {"step": step, "milestone": milestone, **summary})

    def push_buffer_status(self, step: int, status: dict) -> None:
        """Push buffer status information."""
        self._emit("buffer_status", {"step": step, **status})

    def get_episode_state(self, env_id: int) -> dict:
        """Get the latest episode state for an environment."""
        return self.episode_state.get(env_id, {})

    def get_milestone_scores(self) -> dict:
        """Get all milestone scores."""
        return self.milestone_scores


class WandBLogger:
    """
    Enhanced WandB logger with configurable networks and noise scheduling support.
    
    Features:
    - Network configuration logging
    - Exploration metrics tracking
    - Milestone and evaluation logging
    - Fail-fast initialization
    """
    
    def __init__(self, project_name: str = "surgical_robot_maddpg", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
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
            tags=["maddpg", "multi-agent", "surgical-robot", "configurable-networks"],
            notes="Multi-environment parallel MADDPG training with configurable networks and noise scheduling",
            settings=wandb.Settings(start_method="thread")
        )
        
        # Log key configuration for dashboard
        networks_cfg = config.get("networks", {})
        exploration_cfg = config.get("exploration", {})
        
        wandb.config.update({
            "update_interval": config.get("maddpg_config", {}).get("update_interval", 100),
            "reward_scale": 0.01,
            "agent_mode": "robot_only",
            "reward_components": "trajectory+progress+potential_field",
            "termination_mode": "direct_obstacle_collision",
            "completion_threshold": config.get("reward_parameters", {}).get("completion_threshold", 0.01),
            # Network configuration
            "actor_layers": networks_cfg.get("actor", {}).get("hidden_layers", []),
            "critic_layers": networks_cfg.get("critic", {}).get("hidden_layers", []),
            "actor_dropout": networks_cfg.get("actor", {}).get("dropout_p", 0.0),
            "critic_dropout": networks_cfg.get("critic", {}).get("dropout_p", 0.0),
            "orthogonal_init": networks_cfg.get("actor", {}).get("orthogonal_init", False),
            "std_scale": networks_cfg.get("actor", {}).get("std_scale", 1.0),
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

        # Training update level: direct logging with exploration metrics
        hub.subscribe("update", lambda data: self.log_algorithm_statistics(data, data["step"]))

        # Episode level: available for future use
        def _on_episode(ep):
            pass
        hub.subscribe("episode", _on_episode)

        # Milestone completion: calculate and upload milestone metrics
        def _on_ms(ms):
            scores = ms.get("scores", {})
            step = ms.get("step", 0)
            
            if scores:
                score_values = list(scores.values())
                best = max(score_values) if score_values else 0.0
                avg = sum(score_values) / len(score_values) if score_values else 0.0
                count = len(score_values)
            else:
                best = ms.get("milestone/topk_best_score", ms.get("eval/return_mean", 0.0))
                avg = ms.get("milestone/topk_avg_score", ms.get("eval/return_mean", 0.0))
                count = ms.get("milestone/topk_count", 1)
            
            log_data = {
                "milestone/topk_best_score": best,
                "milestone/topk_avg_score": avg,
                "milestone/topk_count": count,
                "milestone/latest_completed": ms.get("milestone", ms.get("milestone/latest_completed", 0)),
            }
            
            if "eval/return_mean" in ms:
                log_data["eval/return_mean"] = ms["eval/return_mean"]
            if "eval/num_episodes" in ms:
                log_data["eval/num_episodes"] = ms["eval/num_episodes"]
            
            self.log_algorithm_statistics(log_data, step)
            
        hub.subscribe("milestone_summary", _on_ms)
        
        print("[WANDB] Attached to MetricsHub with exploration metrics support")

    def log_algorithm_statistics(self, algorithm_stats: Dict[str, Any], step: int) -> None:
        """Log algorithm diagnostics with WHITELIST filtering."""
        if not self.enabled or not algorithm_stats:
            return

        # WHITELIST with exploration metrics
        WHITELIST = {
            # Training
            "train/actor_loss", "train/critic_loss",
            "train/action_std/robot", "train/action_std/human",
            "train/episodes_done", "train/updates",
            
            # Model statistics
            "model/q_mean", "model/q_std",
            "model/q_target_mean", "model/q_target_std",
            "model/td_error_mean", "model/td_rmse", "model/q_qt_corr",
            "model/grad_norm/actor", "model/grad_norm/critic",
            
            # Exploration metrics
            "exploration/noise_scale", "exploration/sigma_ratio",
            
            # Replay buffer
            "replay/buffer_size",
            
            # Evaluation
            "eval/return_mean", "eval/num_episodes",
            
            # Milestones
            "milestone/topk_best_score", "milestone/topk_avg_score",
            "milestone/topk_count", "milestone/latest_completed",
        }

        # Allow milestone/eval data to bypass training check
        has_milestone_data = any(key.startswith(("milestone/", "eval/")) 
                               for key in algorithm_stats.keys())
        
        # Only log when actual update occurred, except for milestone/eval data
        has_training_data = algorithm_stats.get("train/updates", 0) > 0 or algorithm_stats.get("train/episodes_done", 0) > 0
        if not has_milestone_data and not has_training_data:
            return

        log_data = {k: v for k, v in algorithm_stats.items() if k in WHITELIST}
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
        self.maddpg_cfg = self.params.get('maddpg_config', {})
    
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
    
    Features:
    - Raw performance tracking (no clipping)
    - Automatic sorting by performance
    - Checkpoint saving with model metadata
    """
    
    def __init__(self, k: int = 10, mode: str = "max"):
        self.k = k
        self.mode = mode
        self.top_models: List[Tuple[float, Dict, int]] = []
    
    def update(self, performance: float, model_state: Dict[str, Any], milestone: int) -> None:
        """Update top-K models with new performance data."""
        # Use raw performance value - no clipping to preserve true performance
        
        # Add new model
        if len(self.top_models) < self.k:
            self.top_models.append((performance, model_state, milestone))
        elif performance > self.top_models[-1][0]:
            self.top_models[-1] = (performance, model_state, milestone)
        
        # Sort by performance (descending)
        self.top_models.sort(key=lambda x: x[0], reverse=True)
    
    def get_top_models(self) -> List[Tuple[float, Dict, int]]:
        """Get list of top-K models."""
        return self.top_models
    
    def save_checkpoint(self, filepath: str, agent_ids: List[str]) -> None:
        """Save checkpoint with top-K models."""
        checkpoint = {
            'agent_ids': agent_ids,
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


class TrainingLogger:
    """
    Handles training progress logging and file operations.
    
    Features:
    - Progress tracking with unified global step
    - Configuration logging with network and exploration info
    - Final results persistence
    """
    
    def __init__(self, log_directory: str):
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
        self.global_step = 0
    
    def log_message(self, message: str) -> None:
        """Log a message to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.log_directory, "training.log")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_training_start(self, args: argparse.Namespace, params: Dict[str, Any]) -> None:
        """Log training start information with configuration details."""
        print("=" * 70, "\nMADDPG Multi-Environment Parallel Training (Configurable Networks)")
        
        # Read CLI or YAML max_global_steps for display
        cfg_steps = params.get('maddpg_config', {}).get('max_global_steps', 0) or 0
        cli_steps = getattr(args, "max_global_steps", 0) or 0
        max_steps = int(cli_steps if cli_steps > 0 else cfg_steps) or float('inf')
        target_str = f"{max_steps} steps" if max_steps != float('inf') else "∞ steps"
        
        print(f"Environments: {args.num_envs}, Target: {target_str}")
        print(f"Log Directory: {self.log_directory}")
        
        # Log network configuration
        networks_cfg = params.get('networks', {})
        if networks_cfg:
            actor_cfg = networks_cfg.get('actor', {})
            critic_cfg = networks_cfg.get('critic', {})
            print(f"Network Configuration:")
            print(f"  Actor layers: {actor_cfg.get('hidden_layers', 'default')}")
            print(f"  Critic layers: {critic_cfg.get('hidden_layers', 'default')}")
            print(f"  Dropout: Actor {actor_cfg.get('dropout_p', 0.0)}, Critic {critic_cfg.get('dropout_p', 0.0)}")
            print(f"  Orthogonal init: {actor_cfg.get('orthogonal_init', False)}")
            print(f"  Std scale: {actor_cfg.get('std_scale', 1.0)}")
        
        # Log exploration configuration
        exploration_cfg = params.get('exploration', {})
        if exploration_cfg:
            print(f"Exploration Configuration:")
            print(f"  Noise range: {exploration_cfg.get('sigma_start', 0.7)} → {exploration_cfg.get('sigma_end', 0.1)}")
            print(f"  Decay rate: {exploration_cfg.get('decay_k', 6.0)}")
        
        # Log unified completion threshold
        completion_threshold = params.get('reward_parameters', {}).get('completion_threshold', 0.01)
        print(f"Unified completion threshold: {completion_threshold}m")
        print(f"\n", "="*70)    
    
    def log_training_progress(self, global_step: int, global_episodes: int, top_k_manager: TopKModelManager) -> None:
        """Log periodic training progress."""
        self.global_step = global_step
        print(f"[Step {global_step}] Episodes so far: {global_episodes}")
        if top_k_manager.top_models:
            scores = [m[0] for m in top_k_manager.top_models]
            print(f"  Top-{len(scores)} Score Range: {min(scores):.2f} ~ {max(scores):.2f}")
    
    def log_training_complete(self, top_k_manager: TopKModelManager) -> None:
        """Log training completion summary."""
        print("\n" + "=" * 70, "\nTraining Complete!\n" + "=" * 70)
        print("\nFinal Top-K Models:")
        for i, (performance, _, milestone) in enumerate(top_k_manager.get_top_models()):
            print(f"  #{i+1} Milestone {milestone}: {performance:.2f}")
        print(f"\nResults saved in: {self.log_directory}")
    
    def save_final_results(self, global_step: int, global_episodes: int, 
                          top_k_manager: TopKModelManager, params: Dict[str, Any], args: argparse.Namespace) -> None:
        """Save final training results to disk."""
        stats = {
            'total_steps': global_step, 
            'total_episodes': global_episodes,
            'top_k_scores': [(p, m) for p, _, m in top_k_manager.get_top_models()],
            'final_config': {
                'networks': params.get('networks', {}),
                'exploration': params.get('exploration', {}),
            }
        }
        with open(os.path.join(self.log_directory, "training_stats.pkl"), 'wb') as f:
            pickle.dump(stats, f)
        params['command_line_args'] = vars(args)
        with open(os.path.join(self.log_directory, "used_config.yaml"), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)


def save_final_shared_networks(log_directory: str, maddpg, global_step: int, global_episodes: int, max_milestone_triggered: Optional[int]) -> None:
    """Save final shared networks to checkpoint file."""
    final_path = os.path.join(log_directory, "final_shared_networks.pth")
    
    final_checkpoint = {
        'params': maddpg.params,
        'agent_ids': maddpg.agent_ids,
        'global_steps_total': global_step,
        'episodes_done_total': global_episodes,
        'max_milestone_triggered': max_milestone_triggered or 0,
        'shared_networks': True,
        'network_config': maddpg.params.get('networks', {}),
        'exploration_config': maddpg.params.get('exploration', {}),
    }
    
    for agent_id in maddpg.agent_ids:
        agent = maddpg.agents[agent_id]
        final_checkpoint.update({
            f'{agent_id}_actor': agent.actor.state_dict(),
            f'{agent_id}_critic': agent.critic.state_dict(),
            f'{agent_id}_actor_target': agent.actor_target.state_dict(),
            f'{agent_id}_critic_target': agent.critic_target.state_dict()
        })
    
    torch.save(final_checkpoint, final_path)
    print(f"[CHECKPOINT] Final shared networks saved: {final_path}")


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser with configurable networks support."""
    if config_path is None:
        # Adjust path based on actual file location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params.yaml')

    config = TrainingConfiguration(config_path)
    parser = argparse.ArgumentParser(description="MADDPG multi-environment parallel training with configurable networks")
    parser.add_argument("--config", type=str, default=config_path)
    
    # Environment configuration
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
    parser.add_argument("--seed", type=int, default=config.params.get('seed', 42))
    
    parser.add_argument(
        "--max_global_steps", 
        type=int, 
        default=config.maddpg_cfg.get('max_global_steps', 0),
        help="Stop after this many global training steps; if >0, it becomes the sole stop condition."
    )
    
    # Model management
    parser.add_argument("--top_k_models", type=int, default=10)
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    
    return parser