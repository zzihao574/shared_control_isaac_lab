"""
Training script for Epigraph safe MARL.
Main training loop with rollout collection, PPO updates, evaluation, and logging.

FIXED:
1. Correct function call signatures (no parameters passed to collect_rollout/update)
2. Proper checkpoint saving with global_step and update_count
3. Compatible with fixed trainer.py
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_tasks.manager_based.surgical_epigraph import agents as surgical_agents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Epigraph safe MARL policy")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: use agents/training_params_epigraph.yaml)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override number of parallel environments"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total training timesteps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Evaluate every N updates"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save checkpoint every N updates"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Log to tensorboard every N updates"
    )
    
    return parser.parse_args()


def load_config(config_path: str = None):
    """Load training configuration from YAML."""
    if config_path is None:
        # Default: use training_params_epigraph.yaml from agents directory
        agents_dir = os.path.join(
            project_root,
            "isaaclab_tasks",
            "manager_based",
            "surgical_epigraph",
            "agents"
        )
        config_path = os.path.join(agents_dir, "training_params_epigraph.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"[TRAIN] Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_env(config: dict, num_envs: int, device: str):
    """Create training environment."""
    from isaaclab_tasks.manager_based.surgical_epigraph.surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
    from isaaclab_tasks.manager_based.surgical_epigraph.surgical_epigraph_env import SurgicalEpigraphEnv
    
    # Create environment config
    env_cfg = SurgicalEpigraphEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Inject configuration into env_cfg
    env_cfg.params = config
    
    # Create environment
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode=None)
    
    # Also inject to unwrapped env
    if hasattr(env, 'unwrapped'):
        env.unwrapped.params = config
    
    print(f"[TRAIN] Environment created: {num_envs} parallel environments")
    return env


def create_trainer(env, config: dict, device: str):
    """Create Epigraph trainer."""
    from isaaclab_tasks.manager_based.surgical_epigraph.agents.trainer import EpigraphTrainer
    
    algo_cfg = config["algorithms"]["rmappo"]
    epi_cfg = config["epigraph"]
    
    trainer = EpigraphTrainer(
        env=env,
        device=torch.device(device),
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg
    )
    
    print(f"[TRAIN] Trainer created")
    return trainer


def setup_logging(log_dir: str, run_name: str, config: dict):
    """Setup logging directory and tensorboard writer."""
    # Create run directory
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=run_dir)
    
    print(f"[TRAIN] Logging to: {run_dir}")
    
    return run_dir, checkpoint_dir, writer


def train(
    trainer,
    env,
    total_timesteps: int,
    eval_interval: int,
    save_interval: int,
    log_interval: int,
    checkpoint_dir: str,
    writer: SummaryWriter,
    resume_checkpoint: str = None
):
    """
    Main training loop.
    
    FIXED: Use trainer.collect_rollout() and trainer.update() without parameters.
    """
    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f"[TRAIN] Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
        start_step = trainer.global_step
        start_update = start_step // (trainer.rollout_horizon * trainer.num_envs)
    else:
        start_step = 0
        start_update = 0
    
    # Calculate total updates needed
    steps_per_update = trainer.rollout_horizon * trainer.num_envs
    total_updates = total_timesteps // steps_per_update
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Total timesteps:       {total_timesteps:,}")
    print(f"Total updates:         {total_updates:,}")
    print(f"Steps per update:      {steps_per_update:,}")
    print(f"Rollout horizon:       {trainer.rollout_horizon}")
    print(f"Num envs:              {trainer.num_envs}")
    print(f"PPO epochs:            {trainer.ppo_epoch}")
    print(f"Mini batches:          {trainer.num_mini_batch}")
    print(f"Starting from update:  {start_update}")
    print("=" * 60 + "\n")
    
    # Training loop
    global_step = start_step
    
    for update in range(start_update, total_updates):
        update_start_time = datetime.now()
        
        # ========== FIXED: Call without parameters ==========
        # Collect rollout
        rollout_info = trainer.collect_rollout()
        
        # Update networks
        update_info = trainer.update()
        
        # Update global step counter
        global_step += steps_per_update
        trainer.global_step = global_step
        
        # Set global step in environment for logging
        if hasattr(env.unwrapped, 'set_trainer_global_step'):
            env.unwrapped.set_trainer_global_step(global_step)
        
        update_time = (datetime.now() - update_start_time).total_seconds()
        
        # ========== Logging ==========
        if update % log_interval == 0:
            # Log to tensorboard
            writer.add_scalar("train/return_task", rollout_info["return_task_mean"], global_step)
            writer.add_scalar("train/return_safe", rollout_info["return_safe_mean"], global_step)
            writer.add_scalar("train/episode_length", rollout_info["episode_length"], global_step)
            writer.add_scalar("train/episodes_done", rollout_info["episodes_done"], global_step)
            
            # Z statistics
            writer.add_scalar("train/z_mean", rollout_info["z_mean"], global_step)
            writer.add_scalar("train/z_std", rollout_info["z_std"], global_step)
            writer.add_scalar("train/z_min", rollout_info["z_min"], global_step)
            writer.add_scalar("train/z_max", rollout_info["z_max"], global_step)
            
            # Update losses
            writer.add_scalar("update/loss_policy", update_info["loss_policy"], global_step)
            writer.add_scalar("update/loss_value_vl", update_info["loss_value_vl"], global_step)
            writer.add_scalar("update/loss_value_vh", update_info["loss_value_vh"], global_step)
            writer.add_scalar("update/entropy", update_info["entropy"], global_step)
            writer.add_scalar("update/approx_kl", update_info["approx_kl"], global_step)
            writer.add_scalar("update/clipfrac", update_info["clipfrac"], global_step)
            
            # Timing
            writer.add_scalar("timing/update_time", update_time, global_step)
            writer.add_scalar("timing/fps", steps_per_update / update_time, global_step)
            
            # Console output
            print(f"[TRAIN] Update {update + 1}/{total_updates} | Step {global_step:,}/{total_timesteps:,}")
            print(f"  Return (task): {rollout_info['return_task_mean']:.2f} ± {rollout_info['return_task_std']:.2f}")
            print(f"  Return (safe): {rollout_info['return_safe_mean']:.2f} ± {rollout_info['return_safe_std']:.2f}")
            print(f"  Episode length: {rollout_info['episode_length']:.1f}")
            print(f"  Z: {rollout_info['z_mean']:.4f} ± {rollout_info['z_std']:.4f}")
            print(f"  Loss (π/Vl/Vh): {update_info['loss_policy']:.4f} / {update_info['loss_value_vl']:.4f} / {update_info['loss_value_vh']:.4f}")
            print(f"  Entropy: {update_info['entropy']:.4f} | KL: {update_info['approx_kl']:.4f}")
            print(f"  Update time: {update_time:.2f}s | FPS: {steps_per_update / update_time:.0f}")
            print()
        
        # ========== Evaluation ==========
        if update % eval_interval == 0 and update > 0:
            print(f"\n[TRAIN] Running evaluation at update {update}...")
            eval_info = trainer.evaluate(num_episodes=10)
            
            # Log evaluation results
            writer.add_scalar("eval/return_mean", eval_info["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_info["return_std"], global_step)
            writer.add_scalar("eval/episode_length", eval_info["episode_length"], global_step)
            writer.add_scalar("eval/success_rate", eval_info["success_rate"], global_step)
            writer.add_scalar("eval/z_global_mean", eval_info["z_global_mean"], global_step)
            writer.add_scalar("eval/z_global_std", eval_info["z_global_std"], global_step)
            
            print(f"[EVAL] Return: {eval_info['return_mean']:.2f} ± {eval_info['return_std']:.2f}")
            print(f"[EVAL] Success rate: {eval_info['success_rate']:.2%}")
            print(f"[EVAL] Z global: {eval_info['z_global_mean']:.4f} ± {eval_info['z_global_std']:.4f}\n")
        
        # ========== FIXED: Save checkpoint with parameters ==========
        if update % save_interval == 0 and update > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_update_{update}.pt")
            trainer.save_checkpoint(
                path=checkpoint_path,
                global_step=global_step,
                update_count=update
            )
            print(f"[TRAIN] Checkpoint saved: {checkpoint_path}\n")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    trainer.save_checkpoint(
        path=final_checkpoint,
        global_step=global_step,
        update_count=total_updates
    )
    print(f"\n[TRAIN] Final checkpoint saved: {final_checkpoint}")
    
    writer.close()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "=" * 60)
    print("EPIGRAPH SAFE MARL TRAINING")
    print("=" * 60)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_envs is not None:
        config["scene"]["num_envs"] = args.num_envs
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps
    
    # Extract training parameters
    num_envs = config.get("scene", {}).get("num_envs", 256)
    total_timesteps = config.get("training", {}).get("total_timesteps", 10_000_000)
    
    print(f"Num envs:       {num_envs}")
    print(f"Total steps:    {total_timesteps:,}")
    print(f"Device:         {args.device}")
    print(f"Seed:           {args.seed}")
    print("=" * 60 + "\n")
    
    # Create run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"epigraph_{timestamp}"
    else:
        run_name = args.run_name
    
    # Setup logging
    run_dir, checkpoint_dir, writer = setup_logging(args.log_dir, run_name, config)
    
    # Create environment
    env = create_env(config, num_envs, args.device)
    
    # Create trainer
    trainer = create_trainer(env, config, args.device)
    
    # Start training
    try:
        train(
            trainer=trainer,
            env=env,
            total_timesteps=total_timesteps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            checkpoint_dir=checkpoint_dir,
            writer=writer,
            resume_checkpoint=args.resume
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user")
        
        # Save checkpoint on interrupt
        interrupt_checkpoint = os.path.join(checkpoint_dir, "checkpoint_interrupt.pt")
        trainer.save_checkpoint(
            path=interrupt_checkpoint,
            global_step=trainer.global_step,
            update_count=trainer.global_step // (trainer.rollout_horizon * trainer.num_envs)
        )
        print(f"[TRAIN] Checkpoint saved: {interrupt_checkpoint}")
    
    finally:
        # Close environment
        env.close()
        print("\n[TRAIN] Training complete!")


if __name__ == "__main__":
    main()