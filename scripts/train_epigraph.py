#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epigraph training script (Pure Epigraph: self-contained, milestone-driven)

Key features:
- No dependency on RMAPPO control flow or interfaces
- Default config path (no need to specify --config)
- Manual env instantiation with unified config injection
- Trainer receives full_config, ckpt_dir, max_global_steps
- Training loop: rollout → update → print → maybe_milestone_eval_and_save()
- StepTracer controlled by YAML (env self-prints when enabled)
"""

import os
import sys
import time
import argparse

# -------------------------------
# Ensure src/ is on sys.path
# -------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (REPO_ROOT, SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# -------------------------------
# Default Config Path
# -------------------------------
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),        # scripts/
        "..", "src",
        "surgical_project",
        "envs",
        "multi_agent_epigraph",
        "agents",
        "training_params_epigraph.yaml",
    )
)

# -------------------------------
# 1) Parse args BEFORE AppLauncher
# -------------------------------
parser = argparse.ArgumentParser(description="Train Epigraph (multi-agent, milestone-driven).")
parser.add_argument(
    "--config",
    type=str,
    default=DEFAULT_CONFIG_PATH,
    help=f"Path to YAML config file. Defaults to {DEFAULT_CONFIG_PATH}",
)
parser.add_argument("--num_envs", type=int, default=48, 
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for reproducibility.")
parser.add_argument("--max_global_steps", type=int, default=150000,
                    help="Training stops when trainer.global_step >= this value.")
parser.add_argument("--ckpt_dir", type=str, default="logs/epigraph/checkpoints",
                    help="Directory to save checkpoints.")
parser.add_argument("--checkpoint", type=str, default="",
                    help="Path to checkpoint to resume training from.")
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Enable WandB logging (overrides YAML; disabled by default).",
)

# Isaac App launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -------------------------------
# 2) Launch Isaac App
# -------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# After Isaac is up, we can safely import the rest
import torch
import numpy as np
import random

# Import unified surgical_project modules (NOT isaaclab_tasks.*)
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env import SurgicalEpigraphEnv
from surgical_project.algorithms.marl.epigraph.trainer import EpigraphTrainer
from scripts.utils.training_helpers_epigraph import (
    TrainingConfiguration,
    print_training_progress,
)


# -------------------------------
# Helper Functions
# -------------------------------
def setup_global_reproducibility(seed: int, strict_determinism: bool = True):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[SEED] Set seed={seed}, strict_determinism={strict_determinism}")


def create_environment(config_params: dict, num_envs: int, seed: int):
    """
    Create SurgicalEpigraphEnv manually (no gym.make registry).
    Injects config params into env for reward computation.
    """
    # Create environment config
    env_cfg = SurgicalEpigraphEnvCfg()
    
    # Set parallel envs and seed
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    
    # ✅ CRITICAL: Inject YAML params (for reward_components, scaling, etc.)
    env_cfg.params = config_params
    
    # Instantiate environment
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode=None)
    
    # Also inject params to unwrapped env (ensures consistency)
    actual_env = getattr(env, "unwrapped", env)
    actual_env.params = config_params
    
    print(f"[ENV] Created SurgicalEpigraphEnv with {num_envs} parallel environments")
    return env


def check_step_tracer_status(config_params: dict):
    """
    Print StepTracer status based on YAML config.
    
    NOTE: Environment initializes StepTracer itself and controls printing.
    This function only provides visibility into the config setting.
    """
    logging_cfg = config_params.get("logging", {})
    enable = bool(logging_cfg.get("enable_console_logging", False))
    
    if enable:
        print_every = int(logging_cfg.get("print_every_steps", 10))
        max_print = int(logging_cfg.get("max_envs_to_print", 2))
        print(f"[STEPTRACER] Console debug is ENABLED by YAML (env will self-print).")
        print(f"[STEPTRACER] Settings: print_every_steps={print_every}, max_envs_to_print={max_print}")
    else:
        print("[STEPTRACER] Console debug is DISABLED (enable_console_logging=false in YAML).")


def get_device():
    """Get device (prefer CUDA if available)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Main Training Function
# -------------------------------
def main():
    print("\n" + "=" * 80)
    print("EPIGRAPH TRAINING (Pure Epigraph: Self-Contained)")
    print("=" * 80)
    print(f"Config:           {args.config}")
    print(f"Num Envs:         {args.num_envs}")
    print(f"Seed:             {args.seed}")
    print(f"Max Global Steps: {args.max_global_steps}")
    print(f"Checkpoint Dir:   {args.ckpt_dir}")
    if args.checkpoint:
        print(f"Resume from:      {args.checkpoint}")
    print("=" * 80 + "\n")
    
    # ===== 1. Load Configuration =====
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config_obj = TrainingConfiguration.from_yaml(args.config)
    config = config_obj.params  # Full YAML dict
    # Override WandB toggle via CLI (default: disabled unless --wandb provided)
    config.setdefault("logging", {})
    config["logging"]["use_wandb"] = bool(args.wandb)

    # ===== Scale milestone episodes by number of environments (align with rMAPPO) =====
    training_monitor_cfg = config.setdefault("training_monitor", {})
    milestone_episodes = list(training_monitor_cfg.get("milestone_episodes", []))
    if milestone_episodes:
        scaled_milestones = [int(m * args.num_envs) for m in milestone_episodes]
        training_monitor_cfg["milestone_episodes"] = scaled_milestones
        print(f"[MILESTONES] Using scaled milestones (per-env -> total episodes): {scaled_milestones}")
    else:
        print("[MILESTONES] No milestone_episodes defined in config.")

    print(f"[CONFIG] Loaded configuration from: {args.config}\n")
    
    # ===== 2. Set Reproducibility =====
    setup_global_reproducibility(args.seed, strict_determinism=True)
    
    # ===== 3. Create Environment =====
    env = create_environment(config, args.num_envs, args.seed)
    
    # ===== 4. Check StepTracer Status =====
    check_step_tracer_status(config)
    
    # ===== 5. Extract Algorithm Configs =====
    # NOTE: "algorithms.rmappo" is just a naming convention for hyperparameters
    # It does NOT mean we're using RMAPPO code/logic
    algo_cfg = config["algorithms"]["rmappo"]
    epi_cfg = config["epigraph"]
    
    device = get_device()
    print(f"[DEVICE] Using device: {device}\n")
    
    # ===== 6. Create Trainer =====
    trainer = EpigraphTrainer(
        env=env,
        device=device,
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg,
        full_config=config,              # For milestone, wandb, etc.
        ckpt_dir=args.ckpt_dir,          # Trainer manages checkpoints
        max_global_steps=args.max_global_steps,
    )
    
    print(f"[TRAINER] EpigraphTrainer initialized\n")
    
    # ===== 7. Resume from Checkpoint (optional) =====
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Loading checkpoint from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        print(f"[RESUME] Resumed at global_step={trainer.global_step}\n")
    
    # ===== 8. Training Loop =====
    print("[TRAIN] Starting training loop...\n")
    t0 = time.time()
    
    while trainer.global_step < args.max_global_steps:
        # Collect rollout
        rollout_stats = trainer.collect_rollout()
        
        # Update policy and value functions
        update_stats = trainer.update()
        
        # Print progress periodically (every 2000 steps)
        if trainer.global_step % 2000 == 0:
            elapsed = time.time() - t0
            print(f"[STEP {trainer.global_step}] Elapsed: {elapsed/60:.1f} min")
            print_training_progress(
                global_step=trainer.global_step,
                max_steps=args.max_global_steps,
                rollout_stats=rollout_stats,
                update_stats=update_stats,
                agent_labels=trainer.agent_ids,
            )
        
    # ===== 9. Final Checkpoint =====
    final_ckpt = os.path.join(args.ckpt_dir, f"epigraph_final_step{trainer.global_step}.pth")
    os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
    trainer.save_checkpoint(final_ckpt)
    print(f"\n[DONE] Training completed at {trainer.global_step} steps")
    print(f"[DONE] Final checkpoint saved: {final_ckpt}")
    
    # ===== 10. Cleanup =====
    env.close()
    print("[TRAIN] Environment closed\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")
