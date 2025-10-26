#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_epigraph.py - Training script for Epigraph safe MARL algorithm
Features dual value functions (Vl/Vh), RootFinder evaluation, and z-recursive training.
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# === Step 1: Parse arguments BEFORE AppLauncher ===
parser = argparse.ArgumentParser(description="Train Epigraph agent.")
parser.add_argument("--config", type=str, required=True,
                    help="Path to training YAML config.")
parser.add_argument("--task", type=str,
                    default="Isaac-Surgical-MARL-Epigraph-v0",
                    help="Environment task name.")
parser.add_argument("--num_envs", type=int, default=512,
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
parser.add_argument("--run_name", type=str, default=None,
                    help="WandB run name (auto-generated if not provided).")
parser.add_argument("--project", type=str, default="surgical-epigraph",
                    help="WandB project name.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to checkpoint to resume from.")

# Import AppLauncher and add its args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# === Step 2: Launch Isaac App ===
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === Step 3: Import after Isaac is initialized ===
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from utils.training_helpers_epigraph import TrainingConfiguration, WandBLogger
import gymnasium as gym

# ----------------------------- #
# Setup Functions
# ----------------------------- #

def setup_global_reproducibility(seed: int, strict_determinism: bool = True):
    """Setup global random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[SETUP] Global seed set to: {seed}")


def setup_environment(args, config):
    """Create and configure the training environment."""
    print(f"[SETUP] Creating environment: {args.task}")
    
    # Create environment with specified number of parallel envs
    env = gym.make(args.task, num_envs=args.num_envs)
    
    # Inject config into environment
    if hasattr(env.unwrapped, "params"):
        env.unwrapped.params = config.params
        print(f"[SETUP] Config injected into environment")
    
    print(f"[SETUP] Environment created with {args.num_envs} parallel environments")
    return env


def create_trainer(env, config, args, device):
    """
    Create EpigraphTrainer instance.
    
    This imports the trainer from the epigraph algorithm package.
    """
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "src", 
            "surgical_project", "algorithms", "marl", "epigraph"
        ))
        from trainer import EpigraphTrainer
        
        trainer = EpigraphTrainer(
            env=env,
            device=device,
            algo_cfg=config.params["algorithms"]["rmappo"],  # Reuse rmappo config
            epi_cfg=config.params["epigraph"],  # Epigraph-specific config
        )
        
        print("[SETUP] EpigraphTrainer created successfully")
        return trainer
        
    except ImportError as e:
        print(f"[ERROR] Could not import EpigraphTrainer: {e}")
        print("[INFO] Make sure epigraph/trainer.py exists and is properly implemented")
        raise


def setup_checkpoint_dir(config, run_name):
    """Create checkpoint directory for saving models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"epigraph_{timestamp}"
    
    ckpt_dir = os.path.join("checkpoints", "epigraph", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"[SETUP] Checkpoint directory: {ckpt_dir}")
    return ckpt_dir


# ----------------------------- #
# Training Loop
# ----------------------------- #

def train_loop(trainer, config, args, logger, ckpt_dir):
    """
    Main training loop for Epigraph algorithm.
    
    Structure:
    1. Collect rollout (with per-agent z recursion)
    2. Update networks (PPO with dual GAE)
    3. Evaluate (with RootFinder)
    4. Save checkpoints
    """
    # Extract training parameters
    algo_cfg = config.params["algorithms"]["rmappo"]
    epi_cfg = config.params["epigraph"]
    
    rollout_horizon = algo_cfg["rollout_horizon"]
    ppo_epoch = algo_cfg["ppo_epoch"]
    num_mini_batch = algo_cfg["num_mini_batch"]
    max_global_steps = algo_cfg["max_global_steps"]
    
    # Evaluation and saving intervals
    eval_interval = config.params.get("training", {}).get("eval_interval", 10000)
    save_interval = config.params.get("training", {}).get("save_interval", 50000)
    
    global_step = 0
    update_count = 0
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Max global steps: {max_global_steps}")
    print(f"Rollout horizon: {rollout_horizon}")
    print(f"PPO epochs: {ppo_epoch}")
    print(f"Mini-batches: {num_mini_batch}")
    print(f"Eval interval: {eval_interval}")
    print(f"Save interval: {save_interval}")
    print("=" * 80 + "\n")
    
    try:
        while global_step < max_global_steps:
            # ============ Phase 1: Collect Rollout ============
            print(f"\n[Step {global_step}] Collecting rollout...")
            rollout_info = trainer.collect_rollout(rollout_horizon)
            
            global_step += rollout_horizon * args.num_envs
            
            # Log rollout metrics
            if logger:
                logger.log_rollout(global_step, rollout_info)
            
            print(f"[Step {global_step}] Rollout collected: "
                  f"return_task={rollout_info.get('return_task_mean', 0):.3f}, "
                  f"return_safe={rollout_info.get('return_safe_mean', 0):.3f}")
            
            # ============ Phase 2: Update Networks ============
            print(f"[Step {global_step}] Updating networks...")
            update_info = trainer.update(ppo_epoch, num_mini_batch)
            
            update_count += 1
            
            # Log update metrics
            if logger:
                logger.log_update(global_step, update_info)
            
            print(f"[Step {global_step}] Update {update_count}: "
                  f"loss_policy={update_info.get('loss_policy', 0):.4f}, "
                  f"loss_vl={update_info.get('loss_value_vl', 0):.4f}, "
                  f"loss_vh={update_info.get('loss_value_vh', 0):.4f}")
            
            # ============ Phase 3: Periodic Evaluation ============
            if global_step % eval_interval < rollout_horizon * args.num_envs:
                print(f"\n[Step {global_step}] Running evaluation with RootFinder...")
                eval_info = trainer.evaluate(num_episodes=10)
                
                if logger:
                    logger.log_eval(global_step, eval_info)
                
                print(f"[Step {global_step}] Eval: "
                      f"return={eval_info.get('return_mean', 0):.3f}, "
                      f"success_rate={eval_info.get('success_rate', 0):.2%}, "
                      f"z_global_mean={eval_info.get('z_global_mean', 0):.4f}")
            
            # ============ Phase 4: Save Checkpoint ============
            if global_step % save_interval < rollout_horizon * args.num_envs:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{global_step:08d}.pth")
                print(f"\n[Step {global_step}] Saving checkpoint: {ckpt_path}")
                
                trainer.save_checkpoint(
                    path=ckpt_path,
                    global_step=global_step,
                    update_count=update_count,
                )
                
                print(f"[Step {global_step}] Checkpoint saved successfully")
    
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Save final checkpoint
        final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pth")
        print(f"\n[FINAL] Saving final checkpoint: {final_ckpt_path}")
        
        trainer.save_checkpoint(
            path=final_ckpt_path,
            global_step=global_step,
            update_count=update_count,
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total steps: {global_step}")
        print(f"Total updates: {update_count}")
        print(f"Final checkpoint: {final_ckpt_path}")
        print("=" * 80 + "\n")


# ----------------------------- #
# Main Entry Point
# ----------------------------- #

def main():
    print("=" * 80)
    print("Epigraph Safe MARL Training")
    print("=" * 80)
    
    # Setup
    print(f"[SETUP] Loading config from: {args.config}")
    config = TrainingConfiguration.from_yaml(args.config)
    print(f"[SETUP] Config loaded successfully")
    
    setup_global_reproducibility(args.seed, strict_determinism=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Using device: {device}")
    
    # Create environment
    env = setup_environment(args, config)
    
    # Create trainer
    trainer = create_trainer(env, config, args, device)
    
    # Setup checkpoint directory
    ckpt_dir = setup_checkpoint_dir(config, args.run_name)
    
    # Setup WandB logger
    logger = None
    if config.params.get("logging", {}).get("use_wandb", True):
        try:
            logger = WandBLogger(
                project=args.project,
                run_name=args.run_name or f"epigraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config.params,
            )
            print("[SETUP] WandB logger initialized")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            logger = None
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"[SETUP] Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
        print("[SETUP] Checkpoint loaded successfully")
    
    # Start training
    train_loop(trainer, config, args, logger, ckpt_dir)
    
    # Cleanup
    env.close()
    if logger:
        logger.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Exception type   : {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done")