#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Enhanced version with configurable network architecture and exponential noise decay support.

Features:
- TrainingRunner: Unified training loop execution with noise scheduling
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
    Unified training loop executor with noise scheduling and metrics collection.
    
    Features:
    - Exponential noise decay scheduling (fast early, slow later)
    - Unified global step tracking
    - Episode counting with skip mechanism for milestone evaluation
    - Metrics collection and WandB logging
    """
    
    def __init__(self, env, maddpg, replay, metrics_hub, reward_logger, agent_ids, max_global_steps=None):
        self.env = env
        self.maddpg = maddpg
        self.replay = replay
        self.metrics = metrics_hub
        self.reward_logger = reward_logger  # Can be None now
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
        
        # Get max_global_steps for noise scheduling - CLI takes priority over YAML
        maddpg_cfg = self.maddpg.params.get('maddpg_config', {})
        yaml_max_steps = int(maddpg_cfg.get('max_global_steps', 200000))
        
        # CLI parameter takes priority
        if max_global_steps is not None and max_global_steps > 0:
            self.max_global_steps = int(max_global_steps)
        else:
            self.max_global_steps = yaml_max_steps
        
        print(f"[NOISE SCHEDULE] Configured exponential decay:")
        print(f"  Start: {self.sigma_start}, End: {self.sigma_end}, k: {self.decay_k}")
        print(f"  Max steps: {self.max_global_steps}")

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
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

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
        if stats and stats.get("training/updates", 0) > 0:
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
                "train/updates": stats.get("training/updates"),
                "exploration/noise_scale": noise_scale,
            }
            # Clean None values
            payload = {k: v for k, v in payload.items() if v is not None}
            self.metrics.push_update(self.global_step, payload)

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

    def run_until(self, max_global_steps: int):
        """Run training until reaching maximum steps."""
        obs, _ = self.env.reset()
        self._current_obs = obs
        while self.global_step < max_global_steps:
            self.execute_training_step()


class MilestoneEvaluator:
    """
    Milestone evaluator with single environment evaluation and TopK model management.
    
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

    def run_evaluation(self, milestone: int, global_step: int) -> dict:
        """Handle milestone evaluation and model saving."""
        # In-place evaluation
        avg_return, num_eps = self._run_single_evaluation_episode()

        # Extract model weights
        model_state = self._extract_model_state()

        # Update TopK and save
        self.topk.update(avg_return, model_state, milestone)
        ckpt_path = os.path.join(self.log_dir, f"topk_milestone_{milestone}.pth")
        self.topk.save_checkpoint(ckpt_path, self.agent_ids)

        # Push milestone logs
        payload = {
            "eval/return_mean": float(avg_return),
            "eval/num_episodes": int(num_eps),
            "milestone/topk_best_score": float(avg_return),
            "milestone/topk_avg_score": float(avg_return),
            "milestone/topk_count": 1,
            "milestone/latest_completed": int(milestone),
        }
        self.metrics.push_milestone(global_step, milestone, payload)

        return {"skip_episode_once": True}

    def _run_single_evaluation_episode(self):
        """Run single environment evaluation episode."""
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
                
                # Select actions deterministically
                actions, _ = self.maddpg.select_actions(current_obs, add_noise=False, noise_scale=0.0)
                
                # Only env0 executes real actions
                for aid, act in actions.items():
                    if act.ndim == 2:
                        masked = torch.zeros_like(act)
                        masked[active_env] = act[active_env]
                        actions[aid] = masked
                
                obs, rewards, terminated, truncated, infos = env.step(actions)
                
                # Accumulate env0 rewards only
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
        self.subs: DefaultDict[str, list[Callable[[dict], None]]] = defaultdict(list)  # Event subscribers
        self.episode_state: dict[int, dict] = {}  # Episode state cache
        self.milestone_scores: DefaultDict[int, dict] = defaultdict(dict)  # Milestone score cache
        self.update_ring: Deque[dict] = deque(maxlen=ring)  # Update history ring buffer

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
    Enhanced WandB logger with per-agent metrics and configurable networks support.
    
    Features:
    - Per-agent training metrics
    - Network configuration logging
    - Exploration metrics tracking
    - Milestone and evaluation logging
    """
    
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
        print("[WANDB] Attached to MetricsHub with new focused logging.")

    def log_metrics(self, metrics_data: Dict[str, Any], step: int) -> None:
        """Log metrics with per-agent structure and global metrics."""
        if not self.enabled or not metrics_data:
            return

        log_data = {}

        # Handle Per-Agent training metrics
        if "loss/actor" in metrics_data and isinstance(metrics_data.get('loss/actor'), dict):
            agent_ids = list(metrics_data['loss/actor'].keys())
            
            for agent_id in agent_ids:
                # Loss metrics
                log_data[f'train/{agent_id}/actor_loss'] = metrics_data['loss/actor'][agent_id]
                log_data[f'train/{agent_id}/critic_loss'] = metrics_data['loss/critic'][agent_id]
                # Q-Value metrics
                log_data[f'model/{agent_id}/q_mean'] = metrics_data['q_mean'][agent_id]
                log_data[f'model/{agent_id}/q_std'] = metrics_data['q_std'][agent_id]
                # Target Q-Value metrics
                if 'q_target_mean' in metrics_data and agent_id in metrics_data['q_target_mean']:
                     log_data[f'model/{agent_id}/q_target_mean'] = metrics_data['q_target_mean'][agent_id]
                if 'q_target_std' in metrics_data and agent_id in metrics_data['q_target_std']:
                     log_data[f'model/{agent_id}/q_target_std'] = metrics_data['q_target_std'][agent_id]
                # Gradient norm metrics
                log_data[f'model/{agent_id}/grad_norm_actor'] = metrics_data['grad_norm/actor'][agent_id]
                log_data[f'model/{agent_id}/grad_norm_critic'] = metrics_data['grad_norm/critic'][agent_id]

        # Handle global and milestone metrics
        global_keys = {
            "exploration/noise_scale": "exploration/noise_scale",
            "train/episodes_done": "train/global_episodes",
            "replay/buffer_size": "replay/buffer_size",
            "train/updates": "train/updates",
            "eval/return_mean": "milestone/actor_return",
            "milestone/topk_best_score": "milestone/topk_best_return",
            "milestone/latest_completed": "milestone/latest_completed",
        }
        for src_key, dest_key in global_keys.items():
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