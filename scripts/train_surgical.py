#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Paper-aligned surgical robot training script - simplified version with direct config loading"""

import argparse
import sys
import os
import yaml
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# Command line arguments
parser = argparse.ArgumentParser(description="Train surgical robot with paper-aligned human-robot shared control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum training episodes.")
parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All content after Isaac Sim initialization"""

import gymnasium as gym
import torch
import random
import numpy as np
from pathlib import Path

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle

# Import custom environments and algorithms
import surgical_project.envs.single_agent
from surgical_project.algorithms.mbrl.shared_control import SharedControlTrainer


def load_config() -> dict:
    """Load training configuration directly from specified path"""
    config_path = "/home/zzh/workspace/surgical_robot_project/src/surgical_project/envs/single_agent/agents/training_params.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Loaded config from: {config_path}")
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def set_random_seeds(seed: int):
    """Set random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(trainer, episode: int, log_path: str):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"checkpoint_episode_{episode}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        checkpoint_data = {
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.trainer.identifier_optimizer.state_dict(),
            'training_episode': episode,
            'config': trainer.config,
            'args': vars(args_cli),
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(trainer, checkpoint_path: str):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        trainer.trainer.identifier_optimizer.load_state_dict(checkpoint['identifier_optimizer_state_dict'])
        
        start_episode = checkpoint.get('training_episode', 0)
        print(f"[INFO] Checkpoint loaded from episode: {start_episode}")
        return start_episode
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0


def print_system_info():
    """Print system information"""
    print("=" * 80)
    print("PAPER-ALIGNED SURGICAL HUMAN-ROBOT SHARED CONTROL TRAINING")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Max episodes: {args_cli.max_episodes}")
    print(f"[INFO] Random seed: {args_cli.seed}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Wandb logging: {args_cli.wandb}")
    
    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("=" * 80)


def create_environment():
    """Create training environment"""
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
        
        # Test environment reset
        obs_dict, info = env.reset()
        obs_shape = obs_dict["policy"].shape
        print(f"[INFO] Observation shape: {obs_shape}")
        
        return env, env_cfg
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def setup_video_recording(env):
    """Setup video recording"""
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[INFO] Video recording enabled: every {args_cli.video_interval} steps")
    
    return env


def setup_logging():
    """Setup logging directory"""
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "surgical_shared_control"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    
    return log_dir


def main():
    """Main training function"""
    
    # Set random seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    set_random_seeds(args_cli.seed)
    
    # Load configuration directly from specified path
    agent_cfg = load_config()
    
    # Command line arguments override configuration
    agent_cfg['seed'] = args_cli.seed
    agent_cfg['max_episodes'] = args_cli.max_episodes
    agent_cfg['wandb_logging'] = args_cli.wandb
    
    # Print system information
    print_system_info()
    
    # Create environment
    env, env_cfg = create_environment()
    if env is None:
        return
    
    # Setup video recording
    env = setup_video_recording(env)
    
    # Setup logging
    log_dir = setup_logging()
    
    # Save configuration
    try:
        dump_yaml(os.path.join(log_dir, "params", "env_config.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent_config.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env_config.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent_config.pkl"), agent_cfg)
    except Exception as e:
        print(f"[WARNING] Failed to save configs: {e}")
    
    # Create trainer
    try:
        trainer = SharedControlTrainer(env, agent_cfg, log_dir)
        print(f"[INFO] Trainer created successfully")
        print(f"[INFO] Network parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")
        print(f"[INFO] Training device: {trainer.device}")
        
        # Load checkpoint if specified
        start_episode = 0
        if args_cli.checkpoint:
            if os.path.exists(args_cli.checkpoint):
                start_episode = load_checkpoint(trainer, args_cli.checkpoint)
            else:
                print(f"[WARNING] Checkpoint file not found: {args_cli.checkpoint}")
            
    except Exception as e:
        print(f"[ERROR] Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # Training
    try:
        print("\n" + "=" * 80)
        print("STARTING OFF-POLICY TRAINING")
        print("=" * 80)
        
        # Pre-training validation
        print("[INFO] Running pre-training validation...")
        obs_dict, _ = env.reset()
        test_action = torch.zeros(args_cli.num_envs, 3)
        obs_dict, reward, terminated, truncated, info = env.step(test_action)
        print(f"[INFO] Environment validation passed")
        print(f"[INFO] Reward shape: {reward.shape}, Obs shape: {obs_dict['policy'].shape}")
        
        # Train model using off-policy method
        return_list = trainer.train_off_policy(total_episodes=agent_cfg['max_episodes'])
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully")
        print(f"[INFO] Final average return: {np.mean(return_list[-10:]):.3f}")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        
        print(f"[INFO] All outputs saved to: {log_dir}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        save_checkpoint(trainer, 0, log_dir)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
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
        try:
            simulation_app.close()
        except:
            pass
        print("[INFO] Training script completed")