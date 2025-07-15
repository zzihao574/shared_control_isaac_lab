#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train surgical robot with human-robot shared control."""

import argparse
import sys
import os
from datetime import datetime

# Add the src directory to Python path before importing anything
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train surgical robot with human-robot shared control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum training iterations.")
parser.add_argument("--save_interval", type=int, default=100, help="Model save interval.")
parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval.")
parser.add_argument("--log_interval", type=int, default=10, help="Logging interval.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is initialized."""

import gymnasium as gym
import torch
import yaml
import random
import numpy as np
from pathlib import Path

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# Import our custom environment and algorithms
import surgical_project.envs.single_agent  # This will register our environment
from surgical_project.algorithms.mbrl.shared_control import SharedControlTrainer
from surgical_project.algorithms.mbrl.actor_critic import SurgicalActorCritic


class TrainingConfig:
    """Training configuration class."""
    def __init__(self, config_path: str = None):
        # Load configuration from YAML file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.config = config_data['params']['config']
                
                # Ensure all numeric values are properly converted
                self._convert_numeric_values()
        else:
            # Default configuration
            self.config = self._get_default_config()
            
        # Override with command line arguments
        self._apply_cli_overrides()
        
    def _convert_numeric_values(self):
        """Convert string values to appropriate numeric types."""
        numeric_fields = {
            'learning_rate': float,
            'identifier_lr': float,
            'batch_size': int,
            'buffer_size': int,
            'min_buffer_size': int,
            'max_grad_norm': float,
            'q1_weight': float,
            'q2_weight': float,
            'q3_weight': float,
            'r_weight': float,
            'human_stiffness': float,
            'human_damping': float,
            'gamma': float,
            'tau': float,
            'entropy_weight': float,
            'actor_loss_weight': float,
            'critic_loss_weight': float,
            'identifier_loss_weight': float,
            'collaboration_weight': float,
            'reward_scale': float,
            'steps_per_epoch': int,
            'update_frequency': int,
            'save_frequency': int,
            'eval_frequency': int,
        }
        
        for field, field_type in numeric_fields.items():
            if field in self.config:
                try:
                    self.config[field] = field_type(self.config[field])
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] Failed to convert {field}: {e}, using default")
                    if field_type == float:
                        self.config[field] = 1e-4 if 'lr' in field else 1.0
                    else:
                        self.config[field] = 1000 if 'size' in field else 1
        
    def _get_default_config(self):
        """Get default training configuration."""
        return {
            'learning_rate': 3e-4,
            'identifier_lr': 1e-3,
            'batch_size': 512,
            'buffer_size': 100000,
            'min_buffer_size': 1000,
            'max_grad_norm': 0.5,
            'q1_weight': 10.0,
            'q2_weight': 5.0,
            'q3_weight': 15.0,
            'r_weight': 0.1,
            'human_stiffness': 201.0,
            'human_damping': 21.0,
            'gamma': 0.99,
            'tau': 0.005,
            'entropy_weight': 0.01,
            'actor_loss_weight': 1.0,
            'critic_loss_weight': 0.5,
            'identifier_loss_weight': 0.1,
            'collaboration_weight': 2.0,
            'reward_scale': 1.0,
        }
        
    def _apply_cli_overrides(self):
        """Apply command line argument overrides."""
        if hasattr(args_cli, 'num_envs') and args_cli.num_envs:
            self.num_envs = args_cli.num_envs
        if hasattr(args_cli, 'seed') and args_cli.seed:
            self.seed = args_cli.seed
        if hasattr(args_cli, 'max_iterations') and args_cli.max_iterations:
            self.max_iterations = args_cli.max_iterations
            
    def __getattr__(self, name):
        """Allow access to config values as attributes."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def print_config(self):
        """Print configuration for debugging."""
        print("\nConfiguration values:")
        for key, value in self.config.items():
            print(f"  {key}: {value} ({type(value).__name__})")


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] Random seeds set to {seed}")


def create_directories(log_root_path: str, log_dir: str):
    """Create necessary directories for logging."""
    full_log_path = os.path.join(log_root_path, log_dir)
    
    directories = [
        full_log_path,
        os.path.join(full_log_path, "checkpoints"),
        os.path.join(full_log_path, "configs"),
        os.path.join(full_log_path, "logs"),
    ]
    
    if args_cli.video:
        directories.append(os.path.join(full_log_path, "videos", "train"))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return full_log_path


def save_checkpoint(trainer, step: int, log_path: str, config: TrainingConfig, is_best: bool = False):
    """Save model checkpoint."""
    checkpoint_name = f"checkpoint_step_{step}.pth"
    if is_best:
        checkpoint_name = f"best_model_step_{step}.pth"
    
    checkpoint_path = os.path.join(log_path, "checkpoints", checkpoint_name)
    
    try:
        torch.save({
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.identifier_optimizer.state_dict(),
            'training_step': step,
            'config': config.config,
            'args': vars(args_cli),
        }, checkpoint_path)
        
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(trainer, checkpoint_path: str):
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        trainer.identifier_optimizer.load_state_dict(checkpoint['identifier_optimizer_state_dict'])
        
        start_step = checkpoint.get('training_step', 0)
        print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
        print(f"[INFO] Resuming from step: {start_step}")
        
        return start_step
        
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0


def fix_trainer_device_access(trainer, env):
    """Fix device access issue in trainer."""
    try:
        # Ensure trainer has correct device reference
        if hasattr(env, 'device'):
            trainer.device = env.device
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
            trainer.device = env.unwrapped.device
        else:
            # Fallback to CUDA if available
            trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure policy is on correct device
        trainer.policy = trainer.policy.to(trainer.device)
        
        print(f"[INFO] Trainer device set to: {trainer.device}")
        
    except Exception as e:
        print(f"[WARNING] Device setup issue: {e}")


def create_trainer_safely(env, config):
    """Create trainer with proper error handling."""
    try:
        # Get the unwrapped environment for device access
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Verify environment has required attributes
        if not hasattr(unwrapped_env, 'device'):
            print(f"[WARNING] Environment missing device attribute, using default")
            unwrapped_env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verify config has correct types
        print(f"[DEBUG] Config learning_rate: {config.learning_rate} ({type(config.learning_rate)})")
        print(f"[DEBUG] Config batch_size: {config.batch_size} ({type(config.batch_size)})")
        
        # Create trainer with unwrapped environment
        trainer = SharedControlTrainer(unwrapped_env, config)
        
        # Fix any device access issues
        fix_trainer_device_access(trainer, unwrapped_env)
        
        return trainer
        
    except Exception as e:
        print(f"[ERROR] Trainer creation failed: {e}")
        print("[INFO] Attempting alternative trainer creation...")
        
        # Alternative: Create trainer with minimal setup
        try:
            # Manual device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create a simplified config object with guaranteed correct types
            class SafeConfig:
                def __init__(self):
                    self.learning_rate = float(3e-4)
                    self.identifier_lr = float(1e-3)
                    self.batch_size = int(512)
                    self.buffer_size = int(100000)
                    self.min_buffer_size = int(1000)
                    self.max_grad_norm = float(0.5)
                    self.q1_weight = float(10.0)
                    self.q2_weight = float(5.0)
                    self.q3_weight = float(15.0)
                    self.r_weight = float(0.1)
            
            safe_config = SafeConfig()
            print(f"[INFO] Using safe config with learning_rate: {safe_config.learning_rate} ({type(safe_config.learning_rate)})")
            
            # Try to extract environment information
            unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            
            # Set device manually if missing
            if not hasattr(unwrapped_env, 'device'):
                unwrapped_env.device = device
                
            trainer = SharedControlTrainer(unwrapped_env, safe_config)
            trainer.device = device
            trainer.policy = trainer.policy.to(device)
            
            print(f"[INFO] Alternative trainer creation successful on {device}")
            return trainer
            
        except Exception as e2:
            print(f"[ERROR] Alternative trainer creation also failed: {e2}")
            raise e2


def evaluate_policy(env, trainer, num_eval_episodes: int = 10):
    """Evaluate the current policy."""
    print(f"[INFO] Evaluating policy over {num_eval_episodes} episodes...")
    
    eval_rewards = []
    eval_success_rate = 0
    
    try:
        for episode in range(num_eval_episodes):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]
            
            episode_reward = 0
            episode_length = 0
            episode_success = False
            
            while episode_length < 500:  # Max episode length
                with torch.no_grad():
                    action = trainer.policy.get_action(obs, deterministic=True)
                    action = torch.clamp(action, -1.0, 1.0)
                
                obs_dict, rewards, terminated, truncated, info = env.step(action)
                obs = obs_dict["policy"]
                
                episode_reward += rewards.mean().item()
                episode_length += 1
                
                # Check for task completion
                if hasattr(env.unwrapped, 'task_completed'):
                    if env.unwrapped.task_completed.any():
                        episode_success = True
                
                if terminated.any() or truncated.any():
                    break
            
            eval_rewards.append(episode_reward)
            if episode_success:
                eval_success_rate += 1
                
        avg_reward = np.mean(eval_rewards)
        success_rate = eval_success_rate / num_eval_episodes
        
        print(f"[EVAL] Average reward: {avg_reward:.3f}")
        print(f"[EVAL] Success rate: {success_rate:.1%}")
        
        return avg_reward, success_rate
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return 0.0, 0.0


def main():
    """Main training function."""
    
    # Set random seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    
    set_random_seeds(args_cli.seed)
    
    print("=" * 80)
    print("SURGICAL HUMAN-ROBOT SHARED CONTROL TRAINING")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    print(f"[INFO] Random seed: {args_cli.seed}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    # Create environment
    try:
        # Import and create configuration
        from surgical_project.envs.single_agent.surgical_direct_env_cfg import SurgicalDirectEnvCfg
        
        env_cfg = SurgicalDirectEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        
        # Create environment with configuration
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
        
        print(f"[INFO] Environment created successfully")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")
        print(f"[INFO] Simulation scale: {env.unwrapped.cfg.simulation_scale}x")
        print(f"[INFO] Network operates in real-world scale for direct deployment")
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Load training configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'surgical_project', 'envs', 'single_agent', 'agents', 'training_params.yaml')
    config = TrainingConfig(config_path)
    
    # Print configuration for debugging
    print(f"[INFO] Configuration loaded from: {config_path}")
    print(f"[INFO] Configuration file exists: {os.path.exists(config_path)}")
    if hasattr(config, 'print_config'):
        config.print_config()
    else:
        print(f"[INFO] Learning rate: {config.learning_rate} ({type(config.learning_rate)})")
        print(f"[INFO] Batch size: {config.batch_size} ({type(config.batch_size)})")
    
    # Create output directories
    log_root_path = "logs/surgical_shared_control"
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_log_path = create_directories(log_root_path, log_dir)
    
    print(f"[INFO] Logging to: {full_log_path}")
    
    # Save configurations
    dump_yaml(os.path.join(full_log_path, "configs", "env_config.yaml"), env.unwrapped.cfg)
    dump_yaml(os.path.join(full_log_path, "configs", "training_config.yaml"), config.config)
    dump_yaml(os.path.join(full_log_path, "configs", "args.yaml"), vars(args_cli))
    
    # Create trainer
    try:
        trainer = create_trainer_safely(env, config)
        print(f"[INFO] Trainer created successfully")
        print(f"[INFO] Network parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")
        print(f"[INFO] Training device: {trainer.device}")
        
        # Load checkpoint if specified
        start_step = 0
        if args_cli.checkpoint:
            if os.path.exists(args_cli.checkpoint):
                start_step = load_checkpoint(trainer, args_cli.checkpoint)
            else:
                print(f"[WARNING] Checkpoint file not found: {args_cli.checkpoint}")
            
    except Exception as e:
        print(f"[ERROR] Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # Training loop
    try:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        best_reward = float('-inf')
        training_rewards = []
        
        # Initial evaluation
        if start_step == 0:
            print("\n[INFO] Initial evaluation...")
            initial_reward, initial_success = evaluate_policy(env.unwrapped, trainer, num_eval_episodes=5)
            print(f"[INFO] Initial performance - Reward: {initial_reward:.3f}, Success: {initial_success:.1%}")
        
        print(f"\n[INFO] Training from step {start_step} to {args_cli.max_iterations}...")
        print("=" * 80)
        
        # Train the model
        trainer.train(total_steps=args_cli.max_iterations)
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully")
        
        # Final evaluation
        print("\n[INFO] Final evaluation...")
        final_reward, final_success = evaluate_policy(env.unwrapped, trainer, num_eval_episodes=10)
        print(f"[INFO] Final performance - Reward: {final_reward:.3f}, Success: {final_success:.1%}")
        
        # Save final model
        final_checkpoint_path = save_checkpoint(trainer, args_cli.max_iterations, full_log_path, config, is_best=True)
        
        # Training summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total training steps: {args_cli.max_iterations}")
        print(f"Final reward: {final_reward:.3f}")
        print(f"Final success rate: {final_success:.1%}")
        print(f"Model saved to: {final_checkpoint_path}")
        print(f"Logs saved to: {full_log_path}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_path = save_checkpoint(trainer, 0, full_log_path, config, is_best=False)
        if emergency_path:
            print(f"[INFO] Emergency checkpoint saved: {emergency_path}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save emergency checkpoint
        try:
            emergency_path = save_checkpoint(trainer, 0, full_log_path, config, is_best=False)
            if emergency_path:
                print(f"[INFO] Emergency checkpoint saved: {emergency_path}")
        except:
            print("[ERROR] Failed to save emergency checkpoint")
            
    finally:
        # Clean up
        try:
            env.close()
            print(f"[INFO] Environment closed")
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        try:
            simulation_app.close()
            print("[INFO] Simulation app closed")
        except:
            pass