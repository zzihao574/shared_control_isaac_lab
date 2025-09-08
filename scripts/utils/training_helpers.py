#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Clean version - removes YAML duplicate dependencies, gets dimensions from environment cfg.
MODIFIED: Added MetricsHub for unified data pipeline
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

    def push_milestone_summary(self, milestone: int, summary: dict) -> None:
        """Push milestone completion summary."""
        scores = summary.get("scores", {})
        for eid, sc in scores.items():
            self.milestone_scores[milestone][eid] = sc
        self._emit("milestone_summary", {"milestone": milestone, **summary})

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


class UnifiedProgressManager:
    """Unified progress manager - single source of truth for all progress tracking."""
    
    def __init__(self, num_envs: int, max_episodes: int, device="cpu"):
        self.num_envs = num_envs
        self.max_episodes = max_episodes
        self.device = torch.device(device)
        
        # Core state tensors
        self.env_episode_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_step_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_active_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.hard_disabled_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._episode_started = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Completion tracking
        self.num_completed_envs = 0
        
        # Optional training logger reference for file logging
        self.training_logger = None
        
        # Environment closure callbacks for dual-layer protection
        self.env_closure_callbacks: List[Callable[[int], None]] = []

    def register_closure_callback(self, callback: Callable[[int], None]):
        """Register callback function to be called when environments are closed."""
        self.env_closure_callbacks.append(callback)
        print(f"[PROGRESS] Registered environment closure callback")

    def on_step(self, env_ids):
        """Single step entry point: only count steps for specified environment IDs."""
        ids = self._normalize_ids(env_ids)
        if not ids: 
            return
            
        idx = torch.tensor(ids, device=self.device, dtype=torch.long)
        live = self.env_active_mask[idx] & (~self.hard_disabled_mask[idx])
        
        if torch.any(live):
            live_idx = idx[live]
            self.env_step_counts[live_idx] += 1
            self._episode_started[live_idx] = True

    def on_episode_end(self, env_ids, reason: str = "env_reset"):
        """Single episode settlement entry point: handle episode completion."""
        ids = self._normalize_ids(env_ids)
        if not ids: 
            return
            
        idx = torch.tensor(ids, device=self.device, dtype=torch.long)
        
        # Update episode count and reset step count
        self.env_episode_counts[idx] += 1
        self.env_step_counts[idx] = 0
        self._episode_started[idx] = False
        
        # Check if any environments reached max episodes
        newly_done = (self.env_episode_counts[idx] >= self.max_episodes) & self.env_active_mask[idx]
        done_idx = torch.nonzero(newly_done, as_tuple=False).squeeze(-1)
        
        if done_idx.numel() > 0:
            self.env_active_mask[idx[done_idx]] = False
            self.num_completed_envs += int(done_idx.numel())
            
            # Console logging for environment closures
            closed = [int(i) for i in idx[done_idx].tolist()]
            print(f"[PROGRESS] Newly closed this step: {closed}")
            print(f"[PROGRESS] Total closed: {self.num_completed_envs}/{self.num_envs}")
            
            # Optional: also stream to TrainingLogger if available
            if getattr(self, "training_logger", None) is not None:
                try:
                    self.training_logger.log_message(
                        f"Environments closed: {closed} at episodes "
                        f"{[int(self.env_episode_counts[i].item()) for i in idx[done_idx]]}"
                    )
                except Exception:
                    pass
            
            # First layer protection: immediately clear buffers via callbacks
            for callback in self.env_closure_callbacks:
                for env_id in closed:
                    try:
                        callback(env_id)
                    except Exception as e:
                        print(f"[WARNING] Environment {env_id} closure callback failed: {e}")

    def filter_valid_for_episode_end(self, env_ids=None, min_steps: int = 1) -> list[int]:
        """Unified filtering: only episodes that actually ran can be settled."""
        ids = self._normalize_ids(env_ids)
        if not ids: 
            return []
            
        idx = torch.tensor(ids, device=self.device, dtype=torch.long)
        started = self._episode_started[idx]
        enough = self.env_step_counts[idx] >= min_steps
        
        return [i for i, ok in zip(ids, (started & enough).tolist()) if ok]

    def get_active_environments(self) -> list[int]:
        """Get list of environments still in training."""
        mask = self.env_active_mask & (~self.hard_disabled_mask)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()

    def is_training_complete(self) -> bool:
        """Check if training is complete."""
        return (self.env_active_mask & (~self.hard_disabled_mask)).sum().item() == 0

    def get_progress_statistics(self) -> Dict[str, Union[int, float]]:
        """Get training progress statistics."""
        active_count = len(self.get_active_environments())
        return {
            'active_count': active_count, 
            'completed_count': self.num_completed_envs
        }

    @property
    def episode_counts(self): 
        return self.env_episode_counts
    
    @property
    def step_counts(self):    
        return self.env_step_counts

    def _normalize_ids(self, env_ids):
        """Normalize environment ID input."""
        if env_ids is None: 
            return list(range(self.num_envs))
        if torch.is_tensor(env_ids): 
            return env_ids.detach().cpu().tolist()
        if isinstance(env_ids, int): 
            return [env_ids]
        return list(env_ids)


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
    Enhanced WandB logger with layered, frequency-based monitoring system.
    MODIFIED: Added attach_metrics_hub method for unified data pipeline
    """
    def __init__(self, project_name: str = "surgical_robot_maddpg", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.run = None
        
        # Performance tracking for WandB dashboard
        self.topk = 20  # Top-K model tracking
        self._leaderboard = []  # Store (score, step, env_id) tuples
        self._latest_completed = 0
        self._completed_percent = 0.0
        
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
                notes="Multi-environment parallel MADDPG training (Dual Protection + Single Agent)"
            )
            
            # Log key configuration for dashboard
            wandb.config.update({
                "update_interval": config.get("maddpg_config", {}).get("update_interval", 100),
                "reward_scale": 0.01,  # Our reward scaling factor
                "agent_mode": "robot_only",  # Single agent training mode
                "reward_components": "trajectory+progress+potential_field",  # Active components
                "termination_mode": "direct_obstacle_collision",
            })
            
            print(f"[WANDB] Successfully initialized: {self.run.name}")
        except Exception as e:
            print(f"[WANDB] Initialization failed: {e}")
            self.enabled = False

    def attach_metrics_hub(self, hub: "MetricsHub"):
        """NEW METHOD: Attach to MetricsHub for unified data pipeline"""
        if not self.enabled:
            return

        # 1) Training update level: use existing 19 whitelist items
        hub.subscribe("update", lambda data: self.log_algorithm_statistics(data, data["step"]))

        # 2) Episode level: currently not used but available for future
        def _on_episode(ep):
            # Could aggregate episode-level metrics here if needed
            pass
        hub.subscribe("episode", _on_episode)

        # 3) Milestone completion: calculate and upload topk_best / topk_avg
        def _on_ms(ms):
            scores = ms.get("scores", {})
            if not scores:
                return
            score_values = list(scores.values())
            best = max(score_values) if score_values else 0.0
            avg = sum(score_values) / len(score_values) if score_values else 0.0
            self.log_algorithm_statistics({
                "performance/topk_best_score": best,
                "performance/topk_avg_score": avg,
                "performance/topk_count": len(score_values),
                "milestone/latest_completed": ms.get("milestone", 0),
            }, ms.get("step", 0))
        hub.subscribe("milestone_summary", _on_ms)
        
        print("[WANDB] Attached to MetricsHub with subscriptions: update, episode, milestone_summary")

    def log_algorithm_statistics(self, algorithm_stats: Dict[str, Any], step: int) -> None:
        """Log algorithm diagnostics with white-list filtering."""
        if not self.enabled or not algorithm_stats:
            return

        # Allow milestone/performance data to bypass training/updates check
        has_milestone_data = any(key.startswith(("milestone/", "performance/")) 
                               for key in algorithm_stats.keys())
        
        # Only log when actual update occurred, except for milestone/performance data
        if not has_milestone_data and algorithm_stats.get("training/updates", 0) <= 0:
            return

        WHITELIST = {
            # training (6)
            "training/actor_loss_mean", "training/actor_loss_std",
            "training/critic_loss_mean", "training/critic_loss_std",
            "training/avg_buffer_size", "training/updates",
            # algo global (5)
            "algo/q_mean", "algo/q_std",
            "algo/q_target_mean", "algo/q_target_std",
            "algo/td_error_mean",
            # env0 / env1 (8)
            "env0/algo/q_mean", "env0/algo/q_std",
            "env0/algo/q_target_mean", "env0/algo/q_target_std",
            "env0/algo/td_error_mean", "env0/algo/q_qt_corr", "env0/algo/td_rmse",
            "env1/algo/q_mean", "env1/algo/q_std",
            "env1/algo/q_target_mean", "env1/algo/q_target_std",
            "env1/algo/td_error_mean", "env1/algo/q_qt_corr", "env1/algo/td_rmse",
            # performance & milestone (4)
            "performance/topk_best_score", "performance/topk_avg_score",
            "performance/topk_count",
            "milestone/latest_completed",
        }

        try:
            log_data = {k: v for k, v in algorithm_stats.items() if k in WHITELIST}
            if log_data:
                wandb.log(log_data, step=step)
        except Exception as e:
            print(f"[WANDB] Failed to log algorithm statistics: {e}")

    def update_performance_and_progress(self, global_step: int, episode_data: Dict[int, Dict[str, Any]], 
                                      total_envs: int) -> None:
        """Update performance leaderboard and training progress metrics."""
        if not self.enabled or not episode_data:
            return
        try:
            log_data = {}
            
            # Collect scores from completed episodes
            for env_id, episode_info in episode_data.items():
                score = episode_info.get("episode_return", 0.0)
                self._leaderboard.append(score)
            
            # Maintain Top-K performance tracking
            if self._leaderboard:
                sorted_scores = sorted(self._leaderboard, reverse=True)
                topk_scores = sorted_scores[:self.topk]
                
                log_data["performance/topk_best_score"] = float(topk_scores[0])
                log_data["performance/topk_avg_score"] = float(np.mean(topk_scores))
                
                # Trim leaderboard to prevent memory growth
                if len(self._leaderboard) > self.topk * 10:
                    self._leaderboard = sorted_scores[:self.topk * 5]
            
            # Milestone tracking (cumulative episodes completed)
            self._latest_completed += len(episode_data)
            log_data["milestone/latest_completed"] = self._latest_completed
            
            if log_data:
                # Use same white-list filtering
                self.log_algorithm_statistics(log_data, global_step)
                
        except Exception as e:
            print(f"[WANDB] Failed to update performance and progress: {e}")

    def log_training_progress(self, global_step: int, progress_stats: Dict[str, Any], top_k_manager) -> None:
        """Log simplified, high-level training progress (kept for compatibility)."""
        if not self.enabled:
            return
        try:
            perf_stats = {}
            if top_k_manager.top_models:
                scores = [model[1] for model in top_k_manager.top_models]
                perf_stats.update({
                    "performance/topk_best_score": max(scores),
                    "performance/topk_avg_score": np.mean(scores),
                })
            
            if perf_stats:
                self.log_algorithm_statistics(perf_stats, global_step)
        except Exception as e:
            print(f"[WANDB] Failed to log simplified progress: {e}")

    def log_milestone_completion(self, global_step: int, milestone: int, performances: Dict[int, Dict[str, Any]]) -> None:
        """Log milestone statistics as a single point in a trend line."""
        if not self.enabled or not performances:
            return
        try:
            # Log the latest completed milestone against the global step
            milestone_stats = {"milestone/latest_completed": milestone}
            self.log_algorithm_statistics(milestone_stats, global_step)
            print(f"[WANDB] Logged trend for Milestone {milestone}")
        except Exception as e:
            print(f"[WANDB] Failed to log milestone trend: {e}")

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
        self.top_models: List[Tuple[int, float, Dict]] = []
    
    def update_model(self, env_id: int, performance: float, model_state: Dict[str, Any]) -> None:
        """Update top-K models with new performance data."""
        performance = np.clip(performance, 0, 100)
        existing_index = next((i for i, (e_id, _, _) in enumerate(self.top_models) if e_id == env_id), None)
        if existing_index is not None:
            self.top_models[existing_index] = (env_id, performance, model_state)
        elif len(self.top_models) < self.k:
            self.top_models.append((env_id, performance, model_state))
        elif performance > self.top_models[-1][1]:
            self.top_models[-1] = (env_id, performance, model_state)
        self.top_models.sort(key=lambda x: x[1], reverse=True)
    
    def get_top_models(self) -> List[Tuple[int, float, Dict]]:
        """Get list of top-K models."""
        return self.top_models
    
    def save_checkpoint(self, filepath: str, maddpg_trainer) -> None:
        """Save checkpoint with top-K models."""
        checkpoint = {
            'params': maddpg_trainer.params,
            'agent_ids': maddpg_trainer.agent_ids,
            'top_k_count': len(self.top_models),
            'top_k_envs': [{'env_id': eid, 'performance': p} for eid, p, _ in self.top_models]
        }
        for env_id, _, model_state in self.top_models:
            for key, value in model_state.items():
                checkpoint[f'env_{env_id}_{key}'] = value
        torch.save(checkpoint, filepath)
        if self.top_models:
            print(f"[TOP-K] Saved {len(self.top_models)} best models, scores: "
                  f"{self.top_models[0][1]:.2f} ~ {self.top_models[-1][1]:.2f}/100")


class TrainingLogger:
    """Handles training progress logging and file operations."""
    
    def __init__(self, log_directory: str):
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
        self.global_step = 0  # Track global step for metrics hub
    
    def log_message(self, message: str) -> None:
        """Log a message to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.log_directory, "training.log")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_training_start(self, args: argparse.Namespace, params: Dict[str, Any]) -> None:
        """Log training start information."""
        print("=" * 70, "\nMADDPG Multi-Environment Parallel Training (Dual Protection)")
        print(f"Environments: {args.num_envs}, Target: {args.max_episodes} episodes per env")
        print(f"Log Directory: {self.log_directory}\n", "="*70)    
    
    def log_training_progress(self, global_step: int, stats: Dict[str, Any], top_k_manager: TopKModelManager) -> None:
        """Log periodic training progress."""
        self.global_step = global_step  # Update global step
        total_envs = stats['active_count'] + stats['completed_count']
        print(f"[Step {global_step}] Completed: {stats['completed_count']}/{total_envs}")
        if top_k_manager.top_models:
            scores = [m[1] for m in top_k_manager.top_models]
            print(f"  Top-{len(scores)} Score Range: {min(scores):.2f} ~ {max(scores):.2f}/100")
    
    def log_training_complete(self, top_k_manager: TopKModelManager) -> None:
        """Log training completion summary."""
        print("\n" + "=" * 70, "\nTraining Complete!\n" + "=" * 70)
        print("\nFinal Top-K Models:")
        for i, (env_id, performance, _) in enumerate(top_k_manager.get_top_models()):
            print(f"  #{i+1} Environment {env_id}: {performance:.2f}/100")
        print(f"\nResults saved in: {self.log_directory}")
    
    def save_final_results(self, global_step: int, progress_manager: UnifiedProgressManager, 
                          top_k_manager: TopKModelManager, params: Dict[str, Any], args: argparse.Namespace) -> None:
        """Save final training results to disk."""
        stats = {'total_steps': global_step, 'top_k_scores': [(e, p) for e, p, _ in top_k_manager.get_top_models()]}
        with open(os.path.join(self.log_directory, "training_stats.pkl"), 'wb') as f:
            pickle.dump(stats, f)
        params['command_line_args'] = vars(args)
        with open(os.path.join(self.log_directory, "used_config.yaml"), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)


def create_argument_parser(config_path: str = None) -> argparse.ArgumentParser:
    """Create command line argument parser with default configuration."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../../src/surgical_project/envs/multi_agent/agents/training_params.yaml')

    config = TrainingConfiguration(config_path)
    parser = argparse.ArgumentParser(description="MADDPG multi-environment parallel training with dual protection")
    parser.add_argument("--config", type=str, default=config_path)
    
    # Environment count from command line overrides cfg default
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
    parser.add_argument("--seed", type=int, default=config.params.get('seed', 42))
    parser.add_argument("--max_episodes", type=int, default=config.maddpg_cfg.get('num_episodes', 600))
    parser.add_argument("--top_k_models", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--load_strategy", type=str, default="distribute_topk", 
                       choices=["distribute_topk", "top1_all", "random"])
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser