#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified script to train surgical robot with human-robot shared control."""

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
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")

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


def load_config():
    """Load simplified training configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'surgical_project', 'envs', 'single_agent', 'agents', 'training_params.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            raw_config = config_data['params']['config']
            
            # Convert all numeric strings to proper types
            processed_config = {}
            for key, value in raw_config.items():
                if isinstance(value, str):
                    # Try to convert string to appropriate numeric type
                    try:
                        if '.' in value or 'e' in value.lower():
                            processed_config[key] = float(value)
                        else:
                            processed_config[key] = int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        processed_config[key] = value
                else:
                    processed_config[key] = value
            
            print(f"[INFO] Loaded config with {len(processed_config)} parameters")
            return processed_config
    else:
        print(f"[WARNING] Config file not found: {config_path}")
        # Default configuration with proper types
        return {
            'learning_rate': 3e-4,
            'identifier_lr': 1e-3,
            'batch_size': 256,
            'buffer_size': 50000,
            'min_buffer_size': 1000,
            'max_grad_norm': 0.5,
            'q1_weight': 100.0,
            'q2_weight': 50.0,
            'q3_weight': 200.0,
            'r_weight': 0.1,
            'human_stiffness': 201.0,
            'human_damping': 21.0,
            'gamma': 0.99,
            'tau': 0.005,
            'robot_action_weight': 0.7,
            'human_action_weight': 0.3,
        }


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


def save_checkpoint(trainer, step: int, log_path: str, config: dict):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"checkpoint_step_{step}.pth")
    
    try:
        checkpoint_data = {
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.identifier_optimizer.state_dict(),
            'training_step': step,
            'config': config,
            'args': vars(args_cli),
        }
        
        torch.save(checkpoint_data, checkpoint_path)
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


def evaluate_policy(env, trainer, num_eval_episodes: int = 5):
    """Evaluate the current policy."""
    print(f"[INFO] Evaluating policy over {num_eval_episodes} episodes...")
    
    eval_rewards = []
    
    for episode in range(num_eval_episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        
        episode_reward = 0
        episode_length = 0
        
        while episode_length < 200:  # Max episode length
            with torch.no_grad():
                action = trainer.policy.get_action(obs, deterministic=True)
                action = torch.clamp(action, -1.0, 1.0)
            
            obs_dict, rewards, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            
            episode_reward += rewards.mean().item()
            episode_length += 1
            
            if terminated.any() or truncated.any():
                break
        
        eval_rewards.append(episode_reward)
    
    avg_reward = np.mean(eval_rewards)
    print(f"[EVAL] Average reward: {avg_reward:.3f}")
    
    return avg_reward


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
        from surgical_project.envs.single_agent.surgical_direct_env_cfg import SurgicalDirectEnvCfg
        
        env_cfg = SurgicalDirectEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
        
        print(f"[INFO] Environment created successfully")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")
        
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
        print("[INFO] Recording training videos.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Load configuration
    config = load_config()
    
    # Create output directories
    log_root_path = "logs/surgical_shared_control"
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_log_path = create_directories(log_root_path, log_dir)
    
    print(f"[INFO] Logging to: {full_log_path}")
    
    # Save configurations
    dump_yaml(os.path.join(full_log_path, "configs", "env_config.yaml"), env_cfg)
    dump_yaml(os.path.join(full_log_path, "configs", "training_config.yaml"), config)
    dump_yaml(os.path.join(full_log_path, "configs", "args.yaml"), vars(args_cli))
    
    # Create trainer
    try:
        trainer = SharedControlTrainer(env, config)
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
        env.close()
        return
    
    # Training loop
    try:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        best_reward = float('-inf')
        
        # Initial evaluation
        if start_step == 0:
            initial_reward = evaluate_policy(env, trainer, num_eval_episodes=3)
            print(f"[INFO] Initial performance: {initial_reward:.3f}")
        
        print(f"\n[INFO] Training from step {start_step} to {args_cli.max_iterations}...")
        print("=" * 80)
        
        # Train the model
        trainer.train(total_steps=args_cli.max_iterations)
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully")
        
        # Final evaluation
        print("\n[INFO] Final evaluation...")
        final_reward = evaluate_policy(env, trainer, num_eval_episodes=5)
        
        # Save final model
        is_best = final_reward > best_reward
        final_checkpoint_path = save_checkpoint(trainer, args_cli.max_iterations, full_log_path, config)
        
        # Training summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total training steps: {args_cli.max_iterations}")
        print(f"Final reward: {final_reward:.3f}")
        print(f"Model saved to: {final_checkpoint_path}")
        print(f"Logs saved to: {full_log_path}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        
        # Save emergency checkpoint
        emergency_path = save_checkpoint(trainer, 0, full_log_path, config)
        if emergency_path:
            print(f"[INFO] Emergency checkpoint saved: {emergency_path}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
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
        print(f"\n[ERROR] Unexpected error in training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        try:
            simulation_app.close()
            print("[INFO] Simulation app closed")
        except:
            pass