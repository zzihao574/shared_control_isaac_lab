#!/usr/bin/env python3

"""Surgical robot training script - Y-axis movement with optimized implementation"""

import argparse
import sys
import os
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train surgical robot with Y-axis movement")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_episodes", type=int, default=1000)
parser.add_argument("--wandb", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import random
import numpy as np

from isaaclab.utils.io import dump_yaml, dump_pickle

# Import surgical project modules
import surgical_project.envs.single_agent
from surgical_project.algorithms.mbrl.shared_control import SharedControlTrainer


def load_config() -> dict:
    """Load configuration"""
    config_path = "/home/zzh/workspace/surgical_robot_project/src/surgical_project/envs/single_agent/agents/training_params.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")


def set_random_seeds(seed: int):
    """Set random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_info():
    """Print training information"""
    print("=" * 80)
    print("SURGICAL HUMAN-ROBOT SHARED CONTROL TRAINING")
    print("OPTIMIZED IMPLEMENTATION WITH 21D OBSERVATIONS")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Environments: {args_cli.num_envs}")
    print(f"[INFO] Episodes: {args_cli.max_episodes}")
    print(f"[INFO] Seed: {args_cli.seed}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] Observation: 21D [x, ẋ, q, q̇, f]")
    print(f"[INFO] Trajectory: Y-axis (0.14,-0.2,0.03) → (0.14,0.2,0.03)")
    print(f"[INFO] Control: u(t) = Ŵa^T Sa(Za) - f(t) - e(t) - K2*ev(t) + PE")
    print(f"[INFO] State Management: Single z_true_t source (optimized)")
    print("=" * 80)


def create_environment():
    """Create environment"""
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
        
        # Test environment
        obs_dict, _ = env.reset()
        obs_shape = obs_dict['policy'].shape
        print(f"[INFO] Observation shape: {obs_shape}")
        
        if obs_shape[-1] != 21:
            print(f"[WARNING] Expected 21D observation, got {obs_shape[-1]}D")
        else:
            print(f"[INFO] ✅ Correct 21D observation space confirmed")
        
        return env, env_cfg
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def setup_video(env):
    """Setup video recording"""
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[INFO] Video recording enabled")
    
    return env


def setup_logging():
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "surgical_training", timestamp)
    
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    
    return log_dir


def save_checkpoint(trainer, episode: int, log_path: str):
    """Save checkpoint"""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"checkpoint_ep_{episode}.pth")
    
    try:
        trainer.save_model(checkpoint_path)
        return checkpoint_path
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(trainer, checkpoint_path: str):
    """Load checkpoint"""
    try:
        trainer.load_model(checkpoint_path)
        print(f"[INFO] Checkpoint loaded successfully")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0


def main():
    """Main function"""
    
    # Setup
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    set_random_seeds(args_cli.seed)
    
    # Load config
    params = load_config()
    params['seed'] = args_cli.seed
    params['max_episodes'] = args_cli.max_episodes
    if 'logging' not in params:
        params['logging'] = {}
    params['logging']['wandb_logging'] = args_cli.wandb
    
    print_info()
    
    # Create environment
    env, env_cfg = create_environment()
    if env is None:
        return
    
    env = setup_video(env)
    log_dir = setup_logging()
    
    # Save configs
    try:
        dump_yaml(os.path.join(log_dir, "params", "env_config.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent_config.yaml"), params)
        dump_pickle(os.path.join(log_dir, "params", "env_config.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent_config.pkl"), params)
        print(f"[INFO] Configuration files saved to {log_dir}/params/")
    except Exception as e:
        print(f"[WARNING] Failed to save configs: {e}")
    
    # Create trainer
    try:
        trainer = SharedControlTrainer(env, params, log_dir)
        
        total_params = sum(p.numel() for p in trainer.policy.parameters())
        print(f"[INFO] Trainer created successfully")
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Device: {trainer.device}")
        print(f"[INFO] Optimized features:")
        print(f"  - Single z_true_t state source (no redundant construction)")
        print(f"  - 21D observation: [x, ẋ, q, q̇, f]")
        print(f"  - Clear time notation: t vs t+1")
        print(f"  - Single point force limiting")
        print(f"  - Optimized exploration control")
        print(f"[INFO] Paper formulas:")
        print(f"  - Identifier: ż̂ = Ŵid^T Sid(ẑ,u) - Kid*z̃")
        print(f"  - Critic: Ŵ̇c = -σc(r(t) + Ŵc^T*Λ)Λ")
        print(f"  - Actor: Ŵ̇a,i = -σa(Ŵa,i^T*Sa + kΓ*Γ̂)Sa")
        print(f"  - Control: u(t) = Ŵa^T Sa(Za) - f(t) - e(t) - K2*ev(t) + PE")
        print(f"  - Updated params: K2={trainer.K2_gain}, Kid={trainer.Kid_gain}")
        
        # Load checkpoint
        if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
            load_checkpoint(trainer, args_cli.checkpoint)
            
    except Exception as e:
        print(f"[ERROR] Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # Training
    try:
        print("\n" + "=" * 80)
        print("STARTING OPTIMIZED TRAINING")
        print("=" * 80)
        
        # Validation
        print("[INFO] Validating environment and trainer...")
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        print(f"[INFO] Reset observation shape: {obs.shape}")
        
        # Initialize trainer state
        trainer._initialize_z_true_from_obs(obs)
        print(f"[INFO] z_true_t initialized: {trainer.z_true_t.shape}")
        
        # Test control computation
        u_t, Za_t, z_bar_t = trainer._compute_robot_control(obs)
        print(f"[INFO] Control computation successful:")
        print(f"  - u_t shape: {u_t.shape}")
        print(f"  - Za_t shape: {Za_t.shape}") 
        print(f"  - z_bar_t shape: {z_bar_t.shape}")
        
        # Test environment step
        obs_dict, reward, terminated, truncated, info = env.step(u_t)
        obs_new = obs_dict["policy"]
        print(f"[INFO] Environment step successful:")
        print(f"  - New observation shape: {obs_new.shape}")
        print(f"  - Reward: {reward.mean().item():.6f}")
        
        print(f"[INFO] ✅ Validation passed!")
        
        # Train
        print(f"[INFO] Training {params['max_episodes']} episodes...")
        print(f"[INFO] Human equilibrium: y≤0→(0.14,0.0,0.03), y>0→(0.14,0.2,0.03)")
        print(f"[INFO] Constraints: z>0, |ẋ|≤4cm/s, joints within limits")
        print(f"[INFO] State management: Single z_true_t source for efficiency")
        
        return_list = trainer.train_on_policy(total_episodes=params['max_episodes'])
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully!")
        print(f"[INFO] Final average return: {np.mean(return_list[-10:]):.3f}")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        print(f"[INFO] Final model saved to: {final_model_path}")
        
        # Evaluation
        print(f"[INFO] Running final evaluation...")
        eval_results = trainer.evaluate_policy(num_episodes=5)
        print(f"[INFO] Evaluation results:")
        print(f"  - Success rate: {eval_results['success_rate']:.1%}")
        print(f"  - Mean reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        
        print(f"[INFO] All outputs saved to: {log_dir}")
        print("=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        
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
            print("[INFO] Environment closed")
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Script interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
            print("[INFO] Simulation app closed")
        except:
            pass
        print("[INFO] Script completed")