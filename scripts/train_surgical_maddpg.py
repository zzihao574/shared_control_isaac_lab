#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train surgical robot multi-agent system using PyTorch MADDPG."""

import argparse
import sys
import os
from datetime import datetime
import yaml
import torch
import numpy as np
import random

# Add the src directory to Python path before importing anything
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train surgical robot MARL using PyTorch MADDPG")

# Environment parameters
parser.add_argument("--scenario", type=str, default="surgical", help="name of the scenario")
parser.add_argument("--max-episode-len", type=int, default=500, help="maximum episode length")
parser.add_argument("--num-episodes", type=int, default=25000, help="number of episodes")
parser.add_argument("--num-envs", type=int, default=512, help="number of parallel environments")

# Core training parameters  
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
parser.add_argument("--batch-size", type=int, default=1024, help="batch size for training")
parser.add_argument("--num-units", type=int, default=64, help="number of units in the MLP")

# Checkpointing
parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
parser.add_argument("--save-dir", type=str, default="./models/surgical_maddpg/", help="directory to save models")
parser.add_argument("--save-rate", type=int, default=1000, help="save model every N episodes")
parser.add_argument("--load-dir", type=str, default="", help="directory to load models from")

# Evaluation
parser.add_argument("--restore", action="store_true", default=False, help="restore from checkpoint")
parser.add_argument("--display", action="store_true", default=False, help="display environment")
parser.add_argument("--benchmark", action="store_true", default=False, help="run benchmark")
parser.add_argument("--benchmark-iters", type=int, default=100000, help="benchmark iterations")
parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="benchmark directory")
parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="plots directory")

# Isaac Lab specific
parser.add_argument("--seed", type=int, default=42, help="random seed")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is initialized."""

from pathlib import Path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# Import MADDPG components
from surgical_project.algorithms.marl.maddpg_trainer import MADDPGTrainer


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] Random seeds set to {seed}")


def parse_args_to_config(args: argparse.Namespace) -> dict:
    """Convert command line arguments to training configuration."""
    config = {
        # Environment
        "scenario": args.scenario,
        "max_episode_len": args.max_episode_len,
        "num_episodes": args.num_episodes,
        "num_envs": args.num_envs,
        "episode_length_s": 10.0,  # From environment config
        
        # Training parameters
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "num_units": args.num_units,
        
        # Checkpointing
        "exp_name": args.exp_name or f"surgical_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "save_dir": args.save_dir,
        "save_rate": args.save_rate,
        "load_dir": args.load_dir,
        
        # Evaluation
        "restore": args.restore,
        "display": args.display,
        "benchmark": args.benchmark,
        "benchmark_iters": args.benchmark_iters,
        "benchmark_dir": args.benchmark_dir,
        "plots_dir": args.plots_dir,
        
        # Other
        "seed": args.seed,
    }
    
    return config


def setup_directories(config: dict):
    """Setup logging and model directories."""
    # Create directories
    for dir_key in ["save_dir", "plots_dir", "benchmark_dir"]:
        os.makedirs(config[dir_key], exist_ok=True)
    
    # Create experiment subdirectories
    exp_name = config["exp_name"]
    exp_dir = os.path.join("logs", "surgical_maddpg", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    subdirs = ["configs", "models", "plots", "videos"]
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    # Update config with experiment paths
    config["exp_dir"] = exp_dir
    config["save_dir"] = os.path.join(exp_dir, "models")
    config["plots_dir"] = os.path.join(exp_dir, "plots")
    
    return exp_dir


def save_config(config: dict, exp_dir: str):
    """Save configuration to file."""
    config_path = os.path.join(exp_dir, "configs", "training_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[INFO] Configuration saved to: {config_path}")


def benchmark_agents(trainer: MADDPGTrainer, config: dict):
    """Run benchmark evaluation."""
    print(f"[INFO] Running benchmark for {config['benchmark_iters']} iterations")
    
    # Load models if specified
    if config["load_dir"]:
        checkpoint_name = "final"  # Or specify which checkpoint to load
        trainer.load_models(checkpoint_name)
    
    # Run evaluation
    results = trainer.evaluate(num_episodes=100, render=config["display"])
    
    # Save benchmark results
    import pickle
    benchmark_file = os.path.join(config["benchmark_dir"], f"{config['exp_name']}_benchmark.pkl")
    with open(benchmark_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[INFO] Benchmark results saved to: {benchmark_file}")
    return results


def main():
    """Main training function."""
    
    # Set random seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    
    set_random_seeds(args_cli.seed)
    
    print("=" * 80)
    print("SURGICAL HUMAN-ROBOT MADDPG TRAINING")
    print("=" * 80)
    print(f"[INFO] Scenario: {args_cli.scenario}")
    print(f"[INFO] Number of episodes: {args_cli.num_episodes}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Max episode length: {args_cli.max_episode_len}")
    print(f"[INFO] Learning rate: {args_cli.lr}")
    print(f"[INFO] Batch size: {args_cli.batch_size}")
    print(f"[INFO] Random seed: {args_cli.seed}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    try:
        # Parse configuration
        config = parse_args_to_config(args_cli)
        
        # Setup directories
        exp_dir = setup_directories(config)
        save_config(config, exp_dir)
        
        print(f"[INFO] Experiment: {config['exp_name']}")
        print(f"[INFO] Experiment directory: {exp_dir}")
        print(f"[INFO] Models will be saved to: {config['save_dir']}")
        
        # Create trainer
        print(f"\n[INFO] Creating MADDPG trainer...")
        trainer = MADDPGTrainer(config)
        
        if config["benchmark"]:
            # Run benchmark only
            benchmark_agents(trainer, config)
            
        elif config["display"]:
            # Display mode - load models and run with rendering
            print(f"[INFO] Running in display mode")
            if config["load_dir"]:
                trainer.load_models("final")
            trainer.evaluate(num_episodes=10, render=True)
            
        else:
            # Training mode
            if config["restore"] and config["load_dir"]:
                print(f"[INFO] Restoring from: {config['load_dir']}")
                trainer.load_models("latest")
            
            print(f"\n[INFO] Starting MADDPG training...")
            trainer.train()
            
            # Save final models
            trainer.save_models("final")
            
            print(f"\n[INFO] Training completed!")
            print(f"[INFO] Final models saved to: {config['save_dir']}")
            
            # Run final evaluation
            print(f"\n[INFO] Running final evaluation...")
            final_results = trainer.evaluate(num_episodes=20)
            
            # Save final results
            results_path = os.path.join(exp_dir, "final_results.yaml")
            with open(results_path, 'w') as f:
                yaml.dump(final_results, f, default_flow_style=False)
            
            print(f"[INFO] Final results saved to: {results_path}")
        
        print("\n" + "=" * 80)
        print("MADDPG TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            if 'trainer' in locals():
                trainer.close()
            print("[INFO] Environment closed")
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