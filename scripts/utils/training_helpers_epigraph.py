"""
Training helper utilities for Epigraph algorithm.
Includes: Configuration management, metrics tracking, and WandB logging.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict


# ============================================================================
# Configuration Management
# ============================================================================

class TrainingConfiguration:
    """
    Unified training configuration manager for Epigraph.
    Loads and validates YAML configuration files.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            params: Configuration dictionary loaded from YAML
        """
        self.params = params
        self._validate_config()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfiguration":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            TrainingConfiguration instance
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        
        return cls(params)
    
    def _validate_config(self):
        """Validate that all required configuration sections exist."""
        required_sections = [
            "training",
            "logging", 
            "algorithms",
            "epigraph",
            "constraints",
            "trajectory",
            "reward_parameters",
        ]
        
        for section in required_sections:
            if section not in self.params:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate algorithms.rmappo (NOT algorithms.epigraph)
        if "rmappo" not in self.params["algorithms"]:
            raise ValueError("Missing algorithms.rmappo configuration")
        
        # Validate training_monitor.milestone_episodes (CRITICAL for Route B)
        if "training_monitor" not in self.params:
            raise ValueError("Missing training_monitor section in config")
        if "milestone_episodes" not in self.params["training_monitor"]:
            raise ValueError("Missing training_monitor.milestone_episodes in config")
        
        print("[CONFIG] Configuration validated successfully")
    
    def get(self, key: str, default=None):
        """Get configuration value by key path (e.g., 'training.seed')."""
        keys = key.split('.')
        value = self.params
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


# ============================================================================
# Rollout Statistics (Required by Modification Plan)
# ============================================================================

def summarize_rollout_stats(
    r_task: torch.Tensor,
    r_safe_cost: torch.Tensor,
    z: torch.Tensor,
    dones: torch.Tensor,
    info: Dict[str, Any],
    agent_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute statistics from rollout data for logging.
    
    This is the key function required by the modification plan for trainer logging.
    
    Args:
        r_task: [T, E, num_agents] - Task rewards collected during rollout
        r_safe_cost: [T, E, num_agents] - Safety costs (>=0) collected during rollout
        z: [T, E, num_agents] - Risk budget values during rollout
        dones: [T, E] - Episode termination flags
        info: Additional information dictionary from rollout
        agent_labels: Optional list of agent identifiers (length must equal num_agents)
        
    Returns:
        stats: Dictionary of statistics including:
            - avg_episode_task_return: Average task return per episode
            - avg_episode_safe_cost_sum: Average safety cost sum per episode
            - unsafe_step_ratio: Ratio of unsafe steps
            - avg_progress_ratio: Average task progress
            - Per-agent metrics keyed by sanitized agent label (r_task_mean_*, r_safe_cost_mean_*, avg_z_*)
    """
    T, batch_dim, num_agents = r_task.shape

    if agent_labels is None:
        agent_labels = [f"agent{i}" for i in range(num_agents)]
    if len(agent_labels) != num_agents:
        raise ValueError(
            f"agent_labels length ({len(agent_labels)}) does not match num_agents ({num_agents})"
        )
    sanitized_labels = [label.replace(" ", "_") for label in agent_labels]

    stats = {}
    
    # ========== Task Return Statistics ==========
    # Total task rewards (sum over time)
    task_return_total = r_task.sum(dim=0)  # [batch_dim, num_agents]
    stats['avg_episode_task_return'] = float(task_return_total.mean().item())
    stats['r_task_mean_per_step'] = float(r_task.mean().item())
    for idx, label in enumerate(sanitized_labels):
        stats[f"r_task_mean_{label}"] = float(r_task[:, :, idx].mean().item())
    
    # ========== Safety Cost Statistics ==========
    # Total safety costs (sum over time)
    safe_cost_total = r_safe_cost.sum(dim=0)  # [batch_dim, num_agents]
    stats['avg_episode_safe_cost_sum'] = float(safe_cost_total.mean().item())
    stats['r_safe_cost_mean_per_step'] = float(r_safe_cost.mean().item())
    for idx, label in enumerate(sanitized_labels):
        stats[f"r_safe_cost_mean_{label}"] = float(r_safe_cost[:, :, idx].mean().item())
    
    # ========== Z Statistics (Per Agent) ==========
    stats['z_mean'] = float(z.mean().item())
    stats['z_std'] = float(z.std().item())
    stats['z_min'] = float(z.min().item())
    stats['z_max'] = float(z.max().item())
    for idx, label in enumerate(sanitized_labels):
        stats[f"avg_z_{label}"] = float(z[:, :, idx].mean().item())
    
    # ========== Episode Statistics ==========
    episodes_finished = int(dones.sum().item())
    stats['episodes_finished'] = episodes_finished
    
    # ========== Safety Violation Statistics ==========
    # From info dictionary (if available)
    if 'is_violating' in info and info['is_violating'] is not None:
        violations = info['is_violating']  # Should be [T, N] bool
        if isinstance(violations, torch.Tensor):
            total_steps = T * batch_dim
            unsafe_steps = violations.sum().item()
            stats['unsafe_step_ratio'] = unsafe_steps / total_steps if total_steps > 0 else 0.0
        else:
            stats['unsafe_step_ratio'] = 0.0
    else:
        stats['unsafe_step_ratio'] = 0.0
    
    # ========== Progress Statistics ==========
    # From info dictionary (if available)
    if 'progress_ratio' in info and info['progress_ratio'] is not None:
        progress = info['progress_ratio']
        if isinstance(progress, torch.Tensor):
            stats['avg_progress_ratio'] = float(progress.mean().item())
        else:
            stats['avg_progress_ratio'] = 0.0
    else:
        stats['avg_progress_ratio'] = 0.0
    
    # ========== Combined Return ==========
    # Total return = task - safe_cost (for reference)
    combined_return = task_return_total - safe_cost_total
    stats['avg_episode_combined_return'] = float(combined_return.mean().item())
    
    return stats


def summarize_eval_stats(
    episode_returns: List[float],
    episode_safe_costs: List[float],
    episode_success: List[bool],
    episode_lengths: List[int],
    z_values: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute statistics from evaluation episodes.
    
    Args:
        episode_returns: List of episode total task returns
        episode_safe_costs: List of episode total safety costs
        episode_success: List of episode success flags (bool)
        episode_lengths: List of episode lengths
        z_values: Optional list of z values used during evaluation
        
    Returns:
        stats: Dictionary of evaluation statistics
    """
    stats = {}
    
    # Return statistics
    if len(episode_returns) > 0:
        stats['eval_return_mean'] = float(np.mean(episode_returns))
        stats['eval_return_std'] = float(np.std(episode_returns))
        stats['eval_return_min'] = float(np.min(episode_returns))
        stats['eval_return_max'] = float(np.max(episode_returns))
    
    # Safety cost statistics
    if len(episode_safe_costs) > 0:
        stats['eval_safe_cost_mean'] = float(np.mean(episode_safe_costs))
        stats['eval_safe_cost_std'] = float(np.std(episode_safe_costs))
        stats['eval_safe_cost_sum'] = float(np.sum(episode_safe_costs))

    # Epigraph score (aligned with play_epigraph)
    if (
        len(episode_returns) > 0
        and len(episode_safe_costs) > 0
        and len(episode_lengths) > 0
    ):
        per_episode_score = [
            1000.0 * (ret - cost) / max(1, length)
            for ret, cost, length in zip(episode_returns, episode_safe_costs, episode_lengths)
        ]
        stats['eval_score_mean'] = float(np.mean(per_episode_score))
        stats['eval_score_std'] = float(np.std(per_episode_score))
    else:
        stats['eval_score_mean'] = 0.0
        stats['eval_score_std'] = 0.0
    
    # Success rate
    if len(episode_success) > 0:
        stats['eval_success_rate'] = float(np.mean(episode_success))
        stats['eval_num_success'] = int(np.sum(episode_success))
        stats['eval_num_episodes'] = len(episode_success)
    
    # Episode length
    if len(episode_lengths) > 0:
        stats['eval_episode_length_mean'] = float(np.mean(episode_lengths))
        stats['eval_episode_length_std'] = float(np.std(episode_lengths))
    
    # Z statistics (if provided)
    if z_values is not None and len(z_values) > 0:
        stats['eval_z_mean'] = float(np.mean(z_values))
        stats['eval_z_std'] = float(np.std(z_values))
    
    return stats


# ============================================================================
# WandB Logger
# ============================================================================

class WandBLogger:
    """
    Weights & Biases logger for training metrics.
    """
    
    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
    ):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            run_name: Name for this run
            config: Configuration dictionary to log
            entity: WandB entity (username or team)
        """
        try:
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            import wandb
            self.wandb = wandb
            self.enabled = True
            
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                entity=entity,
            )
            
            print(f"[WANDB] Initialized: {project}/{run_name}")
            
        except ImportError:
            print("[WARNING] WandB not installed, logging disabled")
            self.enabled = False
        except Exception as e:
            print(f"[WARNING] WandB initialization failed: {e}")
            self.enabled = False
    
    def log_rollout(self, step: int, stats: Dict[str, float]):
        """Log rollout statistics."""
        if not self.enabled:
            return
        
        log_dict = {"rollout/" + k: v for k, v in stats.items()}
        log_dict["global_step"] = step
        self.wandb.log(log_dict, step=step)
    
    def log_update(self, step: int, stats: Dict[str, float]):
        """Log update statistics."""
        if not self.enabled:
            return
        
        log_dict = {"train/" + k: v for k, v in stats.items()}
        log_dict["global_step"] = step
        self.wandb.log(log_dict, step=step)
    
    def log_eval(self, step: int, stats: Dict[str, float]):
        """Log evaluation statistics."""
        if not self.enabled:
            return
        
        log_dict = {"eval/" + k: v for k, v in stats.items()}
        log_dict["global_step"] = step
        self.wandb.log(log_dict, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled:
            self.wandb.finish()
            print("[WANDB] Run finished")


# ============================================================================
# Metric Tracking
# ============================================================================

class MetricTracker:
    """
    Track metrics over training with moving averages and history.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metric tracker.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
            # Keep only last window_size values
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_mean(self, key: str) -> Optional[float]:
        """Get moving average of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        return sum(self.metrics[key]) / len(self.metrics[key])
    
    def get_recent(self, key: str, n: int = 10) -> List[float]:
        """Get n most recent values of a metric."""
        if key not in self.metrics:
            return []
        return self.metrics[key][-n:]
    
    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics (moving averages)."""
        return {key: self.get_mean(key) for key in self.metrics.keys() 
                if self.get_mean(key) is not None}


# ============================================================================
# Utility Functions
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_training_progress(
    global_step: int,
    max_steps: int,
    rollout_stats: Dict[str, float],
    update_stats: Dict[str, float],
    agent_labels: Optional[List[str]] = None,
):
    """
    Print training progress to console.
    
    Args:
        global_step: Current global step
        max_steps: Maximum training steps
        rollout_stats: Statistics from summarize_rollout_stats()
        update_stats: Statistics from trainer update
    """
    progress_pct = global_step / max_steps * 100 if max_steps > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"Training Progress: {global_step}/{max_steps} ({progress_pct:.1f}%)")
    print("=" * 80)
    
    # Determine per-agent ordering
    if agent_labels is None:
        agent_labels = []
    sanitized_labels = [label.replace(" ", "_") for label in agent_labels]
    if not sanitized_labels:
        sanitized_labels = sorted(
            [k[len("avg_z_"):] for k in rollout_stats.keys() if k.startswith("avg_z_")]
        )
        agent_labels = sanitized_labels

    # Rollout statistics
    print("\n[Rollout Statistics]")
    print(f"  Task Return (avg)       : {rollout_stats.get('avg_episode_task_return', 0):.3f}")
    print(f"  Safe Cost Sum (avg)     : {rollout_stats.get('avg_episode_safe_cost_sum', 0):.3f}")
    print(f"  Combined Return (avg)   : {rollout_stats.get('avg_episode_combined_return', 0):.3f}")
    print(f"  Unsafe Step Ratio       : {rollout_stats.get('unsafe_step_ratio', 0):.2%}")
    print(f"  Progress Ratio (avg)    : {rollout_stats.get('avg_progress_ratio', 0):.2%}")
    print(f"  Episodes Finished       : {rollout_stats.get('episodes_finished', 0)}")
    
    if sanitized_labels:
        print("\n[Per-Agent Averages]")
        for label, sanitized in zip(agent_labels, sanitized_labels):
            task_mean = rollout_stats.get(f"r_task_mean_{sanitized}", 0.0)
            safe_mean = rollout_stats.get(f"r_safe_cost_mean_{sanitized}", 0.0)
            z_mean = rollout_stats.get(f"avg_z_{sanitized}", 0.0)
            print(f"  {label} -> task {task_mean:+.3f}, safe {safe_mean:+.3f}, z {z_mean:+.4f}")
    
    # Z statistics
    print("\n[Risk Budget (Z) Statistics]")
    print(f"  Z Range                 : [{rollout_stats.get('z_min', 0):.4f}, {rollout_stats.get('z_max', 0):.4f}]")
    
    # Update statistics
    print("\n[Update Statistics]")
    print(f"  Policy Loss             : {update_stats.get('loss_policy', 0):.4f}")
    print(f"  Value Loss (Vl)         : {update_stats.get('loss_value_vl', 0):.4f}")
    print(f"  Value Loss (Vh)         : {update_stats.get('loss_value_vh', 0):.4f}")
    print(f"  Entropy                 : {update_stats.get('entropy', 0):.4f}")
    
    print("=" * 80 + "\n")
