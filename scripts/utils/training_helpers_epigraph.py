"""
Training helper utilities for Epigraph algorithm.
Includes: Configuration management, metrics tracking, and WandB logging.
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field


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
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        
        return cls(params)
    
    def _validate_config(self):
        """Validate that required configuration sections exist."""
        required_sections = [
            "algorithms.rmappo",
            "epigraph",
            "constraints",
            "reward_parameters",
        ]
        
        for section in required_sections:
            parts = section.split('.')
            current = self.params
            
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config section: {section}")
                current = current[part]
        
        print("[CONFIG] Validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., "algorithms.rmappo.gamma")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        parts = key.split('.')
        current = self.params
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current


# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class MetricsHub:
    """
    Central metrics tracking hub for Epigraph training.
    Tracks rollout, update, and evaluation metrics.
    """
    
    # Rollout metrics
    rollout_return_task: list = field(default_factory=list)
    rollout_return_safe: list = field(default_factory=list)
    rollout_return_total: list = field(default_factory=list)
    rollout_episode_length: list = field(default_factory=list)
    
    # Update metrics
    update_loss_policy: list = field(default_factory=list)
    update_loss_value_vl: list = field(default_factory=list)
    update_loss_value_vh: list = field(default_factory=list)
    update_loss_entropy: list = field(default_factory=list)
    update_entropy: list = field(default_factory=list)
    update_approx_kl: list = field(default_factory=list)
    update_clip_fraction: list = field(default_factory=list)
    
    # Z statistics
    z_mean: list = field(default_factory=list)
    z_std: list = field(default_factory=list)
    z_min: list = field(default_factory=list)
    z_max: list = field(default_factory=list)
    
    # Evaluation metrics
    eval_return_mean: list = field(default_factory=list)
    eval_return_std: list = field(default_factory=list)
    eval_success_rate: list = field(default_factory=list)
    eval_episode_length: list = field(default_factory=list)
    eval_z_global_mean: list = field(default_factory=list)
    eval_z_global_std: list = field(default_factory=list)
    
    def record_rollout(self, info: Dict[str, Any]):
        """Record rollout metrics."""
        self.rollout_return_task.append(info.get("return_task_mean", 0.0))
        self.rollout_return_safe.append(info.get("return_safe_mean", 0.0))
        self.rollout_return_total.append(info.get("return_total_mean", 0.0))
        self.rollout_episode_length.append(info.get("episode_length_mean", 0.0))
    
    def record_update(self, info: Dict[str, Any]):
        """Record update metrics."""
        self.update_loss_policy.append(info.get("loss_policy", 0.0))
        self.update_loss_value_vl.append(info.get("loss_value_vl", 0.0))
        self.update_loss_value_vh.append(info.get("loss_value_vh", 0.0))
        self.update_loss_entropy.append(info.get("loss_entropy", 0.0))
        self.update_entropy.append(info.get("entropy", 0.0))
        self.update_approx_kl.append(info.get("approx_kl", 0.0))
        self.update_clip_fraction.append(info.get("clip_fraction", 0.0))
        
        # Z statistics
        self.z_mean.append(info.get("z_mean", 0.0))
        self.z_std.append(info.get("z_std", 0.0))
        self.z_min.append(info.get("z_min", 0.0))
        self.z_max.append(info.get("z_max", 0.0))
    
    def record_eval(self, info: Dict[str, Any]):
        """Record evaluation metrics."""
        self.eval_return_mean.append(info.get("return_mean", 0.0))
        self.eval_return_std.append(info.get("return_std", 0.0))
        self.eval_success_rate.append(info.get("success_rate", 0.0))
        self.eval_episode_length.append(info.get("episode_length", 0.0))
        self.eval_z_global_mean.append(info.get("z_global_mean", 0.0))
        self.eval_z_global_std.append(info.get("z_global_std", 0.0))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked metrics."""
        summary = {
            "rollout": {
                "return_task_mean": self._mean(self.rollout_return_task),
                "return_safe_mean": self._mean(self.rollout_return_safe),
                "return_total_mean": self._mean(self.rollout_return_total),
                "episode_length_mean": self._mean(self.rollout_episode_length),
            },
            "update": {
                "loss_policy_mean": self._mean(self.update_loss_policy),
                "loss_value_vl_mean": self._mean(self.update_loss_value_vl),
                "loss_value_vh_mean": self._mean(self.update_loss_value_vh),
                "entropy_mean": self._mean(self.update_entropy),
                "clip_fraction_mean": self._mean(self.update_clip_fraction),
            },
            "z": {
                "mean": self._mean(self.z_mean),
                "std": self._mean(self.z_std),
            },
            "eval": {
                "return_mean": self._mean(self.eval_return_mean),
                "success_rate": self._mean(self.eval_success_rate),
                "z_global_mean": self._mean(self.eval_z_global_mean),
            },
        }
        return summary
    
    @staticmethod
    def _mean(values: list) -> float:
        """Compute mean of list, handling empty lists."""
        return sum(values) / len(values) if values else 0.0


# ============================================================================
# WandB Logger
# ============================================================================

class WandBLogger:
    """
    Weights & Biases logger for Epigraph training.
    Logs rollout, update, and evaluation metrics with Epigraph-specific fields.
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
            run_name: Run name for this training session
            config: Configuration dictionary to log
            entity: WandB entity (team name)
        """
        try:
            import wandb
            self.wandb = wandb
            
            self.wandb.init(
                project=project,
                name=run_name,
                config=config,
                entity=entity,
            )
            
            print(f"[WANDB] Initialized: {project}/{run_name}")
            self.enabled = True
            
        except ImportError:
            print("[WARNING] wandb not installed, logging disabled")
            self.enabled = False
        except Exception as e:
            print(f"[WARNING] Failed to initialize wandb: {e}")
            self.enabled = False
    
    def log_rollout(self, global_step: int, info: Dict[str, Any]):
        """
        Log rollout metrics.
        
        Args:
            global_step: Current global training step
            info: Dictionary of rollout metrics
        """
        if not self.enabled:
            return
        
        log_data = {
            # Epigraph-specific: dual returns
            "rollout/return_task_mean": info.get("return_task_mean", 0.0),
            "rollout/return_safe_mean": info.get("return_safe_mean", 0.0),
            "rollout/return_total_mean": info.get("return_total_mean", 0.0),
            
            # Standard rollout metrics
            "rollout/episode_length_mean": info.get("episode_length_mean", 0.0),
            "rollout/success_rate": info.get("success_rate", 0.0),
            
            # Per-agent returns (if available)
            "rollout/return_human": info.get("return_human", 0.0),
            "rollout/return_robot": info.get("return_robot", 0.0),
        }
        
        self.wandb.log(log_data, step=global_step)
    
    def log_update(self, global_step: int, info: Dict[str, Any]):
        """
        Log update metrics.
        
        Args:
            global_step: Current global training step
            info: Dictionary of update metrics
        """
        if not self.enabled:
            return
        
        log_data = {
            # Policy loss
            "loss/policy": info.get("loss_policy", 0.0),
            
            # Epigraph-specific: dual value losses
            "loss/value_vl": info.get("loss_value_vl", 0.0),
            "loss/value_vh": info.get("loss_value_vh", 0.0),
            "loss/value_total": info.get("loss_value_total", 0.0),
            
            # Entropy
            "loss/entropy": info.get("loss_entropy", 0.0),
            "policy/entropy": info.get("entropy", 0.0),
            
            # PPO metrics
            "ppo/approx_kl": info.get("approx_kl", 0.0),
            "ppo/clip_fraction": info.get("clip_fraction", 0.0),
            
            # Explained variance
            "ppo/explained_var_vl": info.get("explained_var_vl", 0.0),
            "ppo/explained_var_vh": info.get("explained_var_vh", 0.0),
            
            # Gradient norms
            "grad/norm_actor": info.get("grad_norm_actor", 0.0),
            "grad/norm_vl": info.get("grad_norm_vl", 0.0),
            "grad/norm_vh": info.get("grad_norm_vh", 0.0),
            
            # Learning rates
            "lr/actor": info.get("lr_actor", 0.0),
            "lr/critic_vl": info.get("lr_vl", 0.0),
            "lr/critic_vh": info.get("lr_vh", 0.0),
            
            # Epigraph-specific: z statistics
            "epigraph/z_mean": info.get("z_mean", 0.0),
            "epigraph/z_std": info.get("z_std", 0.0),
            "epigraph/z_min": info.get("z_min", 0.0),
            "epigraph/z_max": info.get("z_max", 0.0),
            
            # Advantage statistics
            "advantage/mean_task": info.get("adv_task_mean", 0.0),
            "advantage/std_task": info.get("adv_task_std", 0.0),
            "advantage/mean_safe": info.get("adv_safe_mean", 0.0),
            "advantage/std_safe": info.get("adv_safe_std", 0.0),
            "advantage/mean_combined": info.get("adv_combined_mean", 0.0),
            "advantage/std_combined": info.get("adv_combined_std", 0.0),
        }
        
        self.wandb.log(log_data, step=global_step)
    
    def log_eval(self, global_step: int, info: Dict[str, Any]):
        """
        Log evaluation metrics.
        
        Args:
            global_step: Current global training step
            info: Dictionary of evaluation metrics
        """
        if not self.enabled:
            return
        
        log_data = {
            # Standard eval metrics
            "eval/return_mean": info.get("return_mean", 0.0),
            "eval/return_std": info.get("return_std", 0.0),
            "eval/success_rate": info.get("success_rate", 0.0),
            "eval/episode_length": info.get("episode_length", 0.0),
            
            # Epigraph-specific: z_global statistics from RootFinder
            "epigraph_eval/z_global_mean": info.get("z_global_mean", 0.0),
            "epigraph_eval/z_global_std": info.get("z_global_std", 0.0),
            "epigraph_eval/z_global_min": info.get("z_global_min", 0.0),
            "epigraph_eval/z_global_max": info.get("z_global_max", 0.0),
            
            # Per-agent z* statistics
            "epigraph_eval/z_human_mean": info.get("z_human_mean", 0.0),
            "epigraph_eval/z_robot_mean": info.get("z_robot_mean", 0.0),
            
            # Safety metrics
            "eval/collision_rate": info.get("collision_rate", 0.0),
            "eval/safety_violation_rate": info.get("safety_violation_rate", 0.0),
        }
        
        self.wandb.log(log_data, step=global_step)
    
    def log_custom(self, data: Dict[str, Any], step: int):
        """
        Log custom metrics.
        
        Args:
            data: Dictionary of custom metrics
            step: Step number for x-axis
        """
        if not self.enabled:
            return
        
        self.wandb.log(data, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled:
            self.wandb.finish()
            print("[WANDB] Run finished")


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of most recent checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.checkpoint_history = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[CKPT] Checkpoint directory: {checkpoint_dir}")
    
    def save(
        self,
        state_dict: Dict[str, Any],
        global_step: int,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            state_dict: Dictionary containing model and optimizer states
            global_step: Current global training step
            filename: Custom filename (auto-generated if not provided)
        
        Returns:
            path: Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_{global_step:08d}.pth"
        
        path = os.path.join(self.checkpoint_dir, filename)
        
        # Add metadata
        state_dict["global_step"] = global_step
        state_dict["checkpoint_path"] = path
        
        # Save
        torch.save(state_dict, path)
        
        # Track checkpoint
        self.checkpoint_history.append((global_step, path))
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"[CKPT] Saved checkpoint: {path}")
        return path
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            state_dict: Loaded checkpoint dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        
        print(f"[CKPT] Loaded checkpoint: {path}")
        print(f"[CKPT] Global step: {state_dict.get('global_step', 'unknown')}")
        
        return state_dict
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return
        
        # Sort by global step
        self.checkpoint_history.sort(key=lambda x: x[0])
        
        # Remove oldest checkpoints
        to_remove = self.checkpoint_history[:-self.keep_last_n]
        self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]
        
        for step, path in to_remove:
            if os.path.exists(path):
                os.remove(path)
                print(f"[CKPT] Removed old checkpoint: {path}")


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


def print_training_stats(
    global_step: int,
    max_steps: int,
    rollout_info: Dict[str, Any],
    update_info: Dict[str, Any],
):
    """
    Print training statistics to console.
    
    Args:
        global_step: Current global step
        max_steps: Maximum training steps
        rollout_info: Rollout metrics
        update_info: Update metrics
    """
    progress = global_step / max_steps * 100
    
    print("\n" + "=" * 80)
    print(f"Training Progress: {global_step}/{max_steps} ({progress:.1f}%)")
    print("=" * 80)
    
    # Rollout stats
    print("\nRollout Statistics:")
    print(f"  Task Return     : {rollout_info.get('return_task_mean', 0):.3f}")
    print(f"  Safe Return     : {rollout_info.get('return_safe_mean', 0):.3f}")
    print(f"  Total Return    : {rollout_info.get('return_total_mean', 0):.3f}")
    print(f"  Episode Length  : {rollout_info.get('episode_length_mean', 0):.1f}")
    
    # Update stats
    print("\nUpdate Statistics:")
    print(f"  Policy Loss     : {update_info.get('loss_policy', 0):.4f}")
    print(f"  Value Loss (Vl) : {update_info.get('loss_value_vl', 0):.4f}")
    print(f"  Value Loss (Vh) : {update_info.get('loss_value_vh', 0):.4f}")
    print(f"  Entropy         : {update_info.get('entropy', 0):.4f}")
    print(f"  Clip Fraction   : {update_info.get('clip_fraction', 0):.3f}")
    
    # Z statistics
    print("\nZ Statistics:")
    print(f"  Mean            : {update_info.get('z_mean', 0):.4f}")
    print(f"  Std             : {update_info.get('z_std', 0):.4f}")
    print(f"  Range           : [{update_info.get('z_min', 0):.4f}, {update_info.get('z_max', 0):.4f}]")
    
    print("=" * 80 + "\n")