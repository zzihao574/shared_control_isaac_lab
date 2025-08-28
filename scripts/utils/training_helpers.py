#!/usr/bin/env python3

"""
Training helper utilities for MADDPG multi-environment parallel training.
Clean version - removes YAML duplicate dependencies, gets dimensions from environment cfg.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")

# Hardware monitoring
try:
    import GPUtil
    HARDWARE_MONITORING_AVAILABLE = True
except ImportError:
    HARDWARE_MONITORING_AVAILABLE = False
    print("[WARNING] Hardware monitoring not available. Install with: pip install psutil GPUtil")


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
    WandB logger with simplified, trend-focused monitoring.
    """
    def __init__(self, project_name: str = "surgical_robot_maddpg", enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.run = None
        if not self.enabled:
            print("[WANDB] Disabled")

    def initialize_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """Initialize WandB run with configuration."""
        if not self.enabled:
            return
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=config,
                tags=["maddpg", "multi-agent", "surgical-robot"],
                notes="Multi-environment parallel MADDPG training (Simplified Logging)"
            )
            print(f"[WANDB] Successfully initialized: {self.run.name}")
        except Exception as e:
            print(f"[WANDB] Initialization failed: {e}")
            self.enabled = False

    def log_algorithm_statistics(self, global_step: int, algorithm_stats: Dict[str, Any]) -> None:
        """Log simplified algorithm-specific metrics (loss and buffer size)."""
        if not self.enabled or not algorithm_stats:
            return
        try:
            log_data = {}
            if 'actor_losses' in algorithm_stats and algorithm_stats['actor_losses']:
                log_data["training/actor_loss_mean"] = np.mean(algorithm_stats['actor_losses'])
            if 'critic_losses' in algorithm_stats and algorithm_stats['critic_losses']:
                log_data["training/critic_loss_mean"] = np.mean(algorithm_stats['critic_losses'])
            if 'buffer_sizes' in algorithm_stats and algorithm_stats['buffer_sizes']:
                log_data["training/avg_buffer_size"] = np.mean(algorithm_stats['buffer_sizes'])
            
            if log_data:
                wandb.log(log_data, step=global_step)
        except Exception as e:
            print(f"[WANDB] Failed to log simplified algorithm metrics: {e}")

    def log_training_progress(self, global_step: int, progress_stats: Dict[str, Any], top_k_manager) -> None:
        """Log simplified, high-level training progress."""
        if not self.enabled:
            return
        try:
            total_envs = progress_stats['active_count'] + progress_stats['completed_count']
            completion_percent = (progress_stats['completed_count'] / total_envs) * 100 if total_envs > 0 else 0
            log_data = {"progress/completed_environments_percent": completion_percent}
            
            if HARDWARE_MONITORING_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        log_data["hardware/gpu_utilization"] = gpus[0].load * 100
                except:
                    pass
            
            if top_k_manager.top_models:
                scores = [model[1] for model in top_k_manager.top_models]
                log_data.update({
                    "performance/topk_best_score": max(scores),
                    "performance/topk_avg_score": np.mean(scores),
                })
            
            wandb.log(log_data, step=global_step)
        except Exception as e:
            print(f"[WANDB] Failed to log simplified progress: {e}")

    def log_milestone_completion(self, global_step: int, milestone: int, performances: Dict[int, Dict[str, Any]]) -> None:
        """Log milestone statistics as a single point in a trend line."""
        if not self.enabled or not performances:
            return
        try:
            scores = [p['score'] for p in performances.values()]
            completion_rate = sum(1 for p in performances.values() if p['completed']) / len(performances)
            collision_rate = sum(1 for p in performances.values() if p['collision']) / len(performances)

            # Log trend charts using the milestone number as the x-axis step
            wandb.log({
                "milestone/avg_score_trend": np.mean(scores),
                "milestone/completion_rate_trend": completion_rate,
                "milestone/collision_rate_trend": collision_rate,
            }, step=milestone)

            # Log the latest completed milestone against the global step
            wandb.log({"milestone/latest_completed": milestone}, step=global_step)
            print(f"[WANDB] Logged trend for Milestone {milestone}")
        except Exception as e:
            print(f"[WANDB] Failed to log milestone trend: {e}")

    def log_training_completion(self, total_steps: int, top_k_manager) -> None:
        """Log final training results, including a score histogram."""
        if not self.enabled:
            return
        try:
            final_scores = [model[1] for model in top_k_manager.top_models] if top_k_manager.top_models else [0]
            wandb.log({
                "final/total_steps": total_steps,
                "final/best_score": max(final_scores),
                "final/avg_topk_score": np.mean(final_scores),
            })
            if len(final_scores) > 1:
                wandb.log({"final/score_distribution": wandb.Histogram(final_scores)})
        except Exception as e:
            print(f"[WANDB] Failed to log training completion: {e}")

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
        self.hardware_cfg = self.params.get('hardware', {})
    
    def get_compute_device(self) -> str:
        """Get compute device (CUDA if available)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_milestone_episodes(self) -> List[int]:
        """Get milestone episode numbers."""
        return self.params.get('training_monitor', {}).get('milestone_episodes', [])


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


class TrainingProgressTracker:
    """Tracks training progress across multiple environments."""
    
    def __init__(self, num_envs: int, max_episodes: int):
        self.num_envs = num_envs
        self.max_episodes = max_episodes
        self.env_episode_counts = [0] * num_envs
        self.env_completed = [False] * num_envs
        self.num_completed_envs = 0
    
    def complete_episode(self, env_id: int) -> bool:
        """Mark episode completion and check if environment finished training."""
        self.env_episode_counts[env_id] += 1
        if self.env_episode_counts[env_id] >= self.max_episodes and not self.env_completed[env_id]:
            self.env_completed[env_id] = True
            self.num_completed_envs += 1
            return True
        return False
    
    def get_active_environments(self) -> List[int]:
        """Get list of environments still training."""
        return [i for i in range(self.num_envs) if not self.env_completed[i]]
    
    def is_training_complete(self) -> bool:
        """Check if all environments completed training."""
        return self.num_completed_envs >= self.num_envs
    
    def get_progress_statistics(self) -> Dict[str, Union[int, float]]:
        """Get training progress statistics."""
        active_count = len(self.get_active_environments())
        return {'active_count': active_count, 'completed_count': self.num_completed_envs}


class TrainingLogger:
    """Handles training progress logging and file operations."""
    
    def __init__(self, log_directory: str):
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
    
    def log_training_start(self, args: argparse.Namespace, params: Dict[str, Any]) -> None:
        """Log training start information."""
        print("=" * 70, "\nMADDPG Multi-Environment Parallel Training")
        print(f"Environments: {args.num_envs}, Target: {args.max_episodes} episodes per env")
        print(f"Log Directory: {self.log_directory}\n", "="*70)
    
    def log_environment_completion(self, env_id: int, episode_count: int, performance: float, completed_count: int, total_envs: int) -> None:
        """Log environment training completion."""
        print(f"[ENV {env_id}] Training Complete - {episode_count} episodes - Final Score: {performance:.2f}/100")
        print(f"[Progress] {completed_count}/{total_envs} environments completed\n")
    
    def log_training_progress(self, global_step: int, stats: Dict[str, Any], top_k_manager: TopKModelManager) -> None:
        """Log periodic training progress."""
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
    
    def save_final_results(self, global_step: int, progress_tracker: TrainingProgressTracker, 
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
    parser = argparse.ArgumentParser(description="MADDPG multi-environment parallel training")
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