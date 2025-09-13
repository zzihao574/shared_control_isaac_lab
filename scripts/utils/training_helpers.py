#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Clean version - removes YAML duplicate dependencies, gets dimensions from environment cfg.
MODIFIED: Added MetricsHub for unified data pipeline, removed unused WandB methods
MODIFIED: Unified global step tracking and completion threshold consistency
MODIFIED: Removed UnifiedProgressManager and all active_env related functionality
MODIFIED: Updated WHITELIST and simplified WandB logging to match new key structure
MODIFIED: Changed from max_episodes to max_global_steps as primary termination condition
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pickle
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

# Hardware monitoring - DISABLED for minimal logging
HARDWARE_MONITORING_AVAILABLE = False


# === NEW CLASS: MetricsHub ===
class MetricsHub:
    """Single-exit metrics bus. Modules push; sinks subscribe. No IO here."""
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
            try:
                h(payload)
            except Exception as e:
                print(f"[MetricsHub] handler error on '{event}': {e}")

    # ---- Pushers ----
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

    # ---- Getters ----
    def get_episode_state(self, env_id: int) -> dict:
        """Get the latest episode state for an environment."""
        return self.episode_state.get(env_id, {})

    def get_milestone_scores(self) -> dict:
        """Get all milestone scores."""
        return self.milestone_scores


class CheckpointManager:
    """
    Manages checkpoint loading and agent initialization with fallback mechanisms.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, load_strategy: str = "distribute_topk"):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_data: Optional[Dict] = None
        self.load_strategy = load_strategy
        
    def load_checkpoint(self) -> bool:
        """Load checkpoint from file if exists."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print(f"[CHECKPOINT] Checkpoint file not found: {self.checkpoint_path}")
            return False
        
        try:
            self.checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu')
            print(f"[CHECKPOINT] Successfully loaded: {self.checkpoint_path}")
            
            if 'top_k_envs' in self.checkpoint_data:
                top_k_envs = self.checkpoint_data['top_k_envs']
                print(f"[CHECKPOINT] Contains {len(top_k_envs)} Top-K models")
                if top_k_envs:
                    scores = [env['performance'] for env in top_k_envs]
                    print(f"[CHECKPOINT] Score range: {min(scores):.2f} ~ {max(scores):.2f}/100")
            
            return True
        except Exception as e:
            print(f"[ERROR] Checkpoint loading failed: {e}")
            return False
    
    def initialize_agents_from_checkpoint(self, maddpg_trainer) -> None:
        """Initialize agents from checkpoint data using specified strategy."""
        # æ·»åŠ å…±äº«ç½'ç»œæž¶æž„æ£€æŸ¥
        if hasattr(maddpg_trainer, 'agents') and not hasattr(maddpg_trainer, 'env_agents'):
            print("[CHECKPOINT] Shared network architecture detected, skipping per-env initialization")
            return
            
        if not (self.checkpoint_data and 'top_k_envs' in self.checkpoint_data and self.checkpoint_data['top_k_envs']):
            print("[CHECKPOINT] No valid data, using random initialization")
            return
        
        top_k_envs = self.checkpoint_data['top_k_envs']
        print(f"[CHECKPOINT] Using {self.load_strategy} strategy for {maddpg_trainer.num_envs} environments")
        
        loading_strategies = {
            "distribute_topk": lambda n, t: {i: t[i % len(t)]['env_id'] for i in range(n)},
            "top1_all": lambda n, t: {i: t[0]['env_id'] for i in range(n)},
            "random": lambda n, t: {i: np.random.choice([e['env_id'] for e in t]) for i in range(n)}
        }
        
        strategy_func = loading_strategies.get(self.load_strategy, loading_strategies["distribute_topk"])
        source_mapping = strategy_func(maddpg_trainer.num_envs, top_k_envs)
        
        success_count = 0
        for target_env_id, source_env_id in source_mapping.items():
            try:
                for agent_id in maddpg_trainer.agent_ids:
                    target_agent = maddpg_trainer.env_agents[target_env_id][agent_id]
                    network_keys = {
                        'actor': f'env_{source_env_id}_{agent_id}_actor',
                        'critic': f'env_{source_env_id}_{agent_id}_critic',
                        'actor_target': f'env_{source_env_id}_{agent_id}_actor_target',
                        'critic_target': f'env_{source_env_id}_{agent_id}_critic_target'
                    }
                    if any(key not in self.checkpoint_data for key in network_keys.values()):
                        raise KeyError(f"Missing keys for agent {agent_id} from source env {source_env_id}")

                    target_agent.actor.load_state_dict(self.checkpoint_data[network_keys['actor']])
                    target_agent.critic.load_state_dict(self.checkpoint_data[network_keys['critic']])
                    target_agent.actor_target.load_state_dict(self.checkpoint_data[network_keys['actor_target']])
                    target_agent.critic_target.load_state_dict(self.checkpoint_data[network_keys['critic_target']])
                success_count += 1
            except Exception as e:
                print(f"[WARNING] Env {target_env_id} loading failed, using random: {e}")
        
        print(f"[CHECKPOINT] Successfully initialized {success_count}/{maddpg_trainer.num_envs} environments")


class WandBLogger:
    """
    Enhanced WandB logger with simplified key structure.
    MODIFIED: Updated to use new WHITELIST and simplified logging pipeline
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
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=config,
                tags=["maddpg", "multi-agent", "surgical-robot", "single-agent-training"],
                notes="Multi-environment parallel MADDPG training (Dual Protection + Single Agent)",
                settings=wandb.Settings(start_method="thread")
            )
            
            # Log key configuration for dashboard
            wandb.config.update({
                "update_interval": config.get("maddpg_config", {}).get("update_interval", 100),
                "reward_scale": 0.01,  # Our reward scaling factor
                "agent_mode": "robot_only",  # Single agent training mode
                "reward_components": "trajectory+progress+potential_field",  # Active components
                "termination_mode": "direct_obstacle_collision",
                "completion_threshold": config.get("reward_parameters", {}).get("completion_threshold", 0.01),
            })
            
            print(f"[WANDB] Successfully initialized: {self.run.name}")
        except Exception as e:
            print(f"[WANDB] Initialization failed: {e}")
            self.enabled = False

    def attach_metrics_hub(self, hub: "MetricsHub"):
        """Attach to MetricsHub for unified data pipeline with new key structure"""
        if not self.enabled:
            return

        # Training update level: direct logging with new keys
        hub.subscribe("update", lambda data: self.log_algorithm_statistics(data, data["step"]))

        # Episode level: currently not used but available for future
        def _on_episode(ep):
            # Could aggregate episode-level metrics here if needed
            pass
        hub.subscribe("episode", _on_episode)

        # Milestone completion: calculate and upload with new milestone/ keys
        def _on_ms(ms):
            scores = ms.get("scores", {})
            step = ms.get("step", 0)
            
            # Calculate milestone metrics from scores if available, otherwise use direct values
            if scores:
                score_values = list(scores.values())
                best = max(score_values) if score_values else 0.0
                avg = sum(score_values) / len(score_values) if score_values else 0.0
                count = len(score_values)
            else:
                # Fallback to direct values from milestone data
                best = ms.get("milestone/topk_best_score", ms.get("eval/return_mean", 0.0))
                avg = ms.get("milestone/topk_avg_score", ms.get("eval/return_mean", 0.0))
                count = ms.get("milestone/topk_count", 1)
            
            # Construct log data with milestone metrics
            log_data = {
                "milestone/topk_best_score": best,
                "milestone/topk_avg_score": avg,
                "milestone/topk_count": count,
                "milestone/latest_completed": ms.get("milestone", ms.get("milestone/latest_completed", 0)),
            }
            
            # Forward eval/ data if present
            if "eval/return_mean" in ms:
                log_data["eval/return_mean"] = ms["eval/return_mean"]
            if "eval/num_episodes" in ms:
                log_data["eval/num_episodes"] = ms["eval/num_episodes"]
            
            # Use new milestone/ prefix
            self.log_algorithm_statistics(log_data, step)
            
        hub.subscribe("milestone_summary", _on_ms)
        
        print("[WANDB] Attached to MetricsHub with new key structure")
        print("[WANDB] Using unified global step for consistent x-axis")

    def log_algorithm_statistics(self, algorithm_stats: Dict[str, Any], step: int) -> None:
        """Log algorithm diagnostics with new WHITELIST filtering."""
        if not self.enabled or not algorithm_stats:
            return

        # New simplified WHITELIST
        WHITELIST = {
            # --- train ---
            "train/actor_loss", "train/critic_loss",
            "train/action_std/robot", "train/action_std/human",
            "train/episodes_done", "train/updates",
            
            # --- model (q/td/gradç­‰ç»Ÿä¸€åœ¨è¿™) ---
            "model/q_mean", "model/q_std",
            "model/q_target_mean", "model/q_target_std",
            "model/td_error_mean", "model/td_rmse", "model/q_qt_corr",
            "model/grad_norm/actor", "model/grad_norm/critic",
            
            # --- replay ---
            "replay/buffer_size",
            
            # --- eval ---
            "eval/return_mean", "eval/num_episodes",
            
            # --- milestone ---
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

        try:
            log_data = {k: v for k, v in algorithm_stats.items() if k in WHITELIST}
            if log_data:
                # Always use the provided unified step for consistent x-axis
                wandb.log(log_data, step=step)
        except Exception as e:
            print(f"[WANDB] Failed to log algorithm statistics: {e}")

    def finalize_run(self) -> None:
        """Finalize WandB run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                print("[WANDB] Run finished")
            except Exception as e:
                print(f"[WANDB] Failed to finish: {e}")


class TrainingConfiguration:
    """Training configuration loader and parameter manager."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        self.maddpg_cfg = self.params.get('maddpg_config', {})
    
    def get_compute_device(self) -> str:
        """Get compute device (CUDA if available)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class TopKModelManager:
    """Manages top-K model collection and checkpoint saving."""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.top_models: List[Tuple[float, Dict, int]] = []
    
    def update(self, performance: float, model_state: Dict[str, Any], milestone: int) -> None:
        """Update top-K models with new performance data."""
        # Use raw performance value without clipping
        
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
    """Handles training progress logging and file operations with unified global step."""
    
    def __init__(self, log_directory: str):
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
        
        # UNIFIED: Global step tracking for metrics hub
        self.global_step = 0
    
    def log_message(self, message: str) -> None:
        """Log a message to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.log_directory, "training.log")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_training_start(self, args: argparse.Namespace, params: Dict[str, Any]) -> None:
        """Log training start information with step-based limits."""
        print("=" * 70, "\nMADDPG Multi-Environment Parallel Training (Shared Networks)")
        
        # Read CLI or YAML max_global_steps for display
        cfg_steps = params.get('maddpg_config', {}).get('max_global_steps', 0) or 0
        cli_steps = getattr(args, "max_global_steps", 0) or 0
        max_steps = int(cli_steps if cli_steps > 0 else cfg_steps) or float('inf')
        target_str = f"{max_steps} steps" if max_steps != float('inf') else "∞ steps"
        
        print(f"Environments: {args.num_envs}, Target: {target_str}")
        print(f"Log Directory: {self.log_directory}")
        
        # Log unified completion threshold
        completion_threshold = params.get('reward_parameters', {}).get('completion_threshold', 0.01)
        print(f"Unified completion threshold: {completion_threshold}m")
        print(f"\n", "="*70)    
    
    def log_training_progress(self, global_step: int, global_episodes: int, top_k_manager: TopKModelManager) -> None:
        """Log periodic training progress with unified global step."""
        self.global_step = global_step  # Update global step
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
            'top_k_scores': [(p, m) for p, _, m in top_k_manager.get_top_models()]
        }
        with open(os.path.join(self.log_directory, "training_stats.pkl"), 'wb') as f:
            pickle.dump(stats, f)
        params['command_line_args'] = vars(args)
        with open(os.path.join(self.log_directory, "used_config.yaml"), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser with step-based configuration."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params.yaml')

    config = TrainingConfiguration(config_path)
    parser = argparse.ArgumentParser(description="MADDPG multi-environment parallel training with shared networks")
    parser.add_argument("--config", type=str, default=config_path)
    
    # Environment count from command line overrides cfg default
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
    parser.add_argument("--seed", type=int, default=config.params.get('seed', 42))
    
    # MODIFIED: Removed --max_episodes, added --max_global_steps
    parser.add_argument(
        "--max_global_steps", 
        type=int, 
        default=config.maddpg_cfg.get('max_global_steps', 0),
        help="Stop after this many global training steps; if >0, it becomes the sole stop condition."
    )
    
    parser.add_argument("--top_k_models", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--load_strategy", type=str, default="distribute_topk", 
                       choices=["distribute_topk", "top1_all", "random"])
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser