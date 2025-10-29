#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play/Evaluation script for trained Epigraph policies.

Key features:
- No dependency on RMAPPO evaluation logic
- Default config path with fallback mechanism
- Unified config injection (same as training)
- Delegates evaluation to trainer.run_single_eval_episode()
- Clean separation: trainer handles all eval logic internally
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

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
parser = argparse.ArgumentParser(description="Evaluate trained Epigraph policy")

parser.add_argument(
    "--config",
    type=str,
    default=DEFAULT_CONFIG_PATH,
    help=f"Path to YAML config file. Defaults to {DEFAULT_CONFIG_PATH}",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to checkpoint file (.pt or .pth)"
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=10,
    help="Number of episodes to evaluate"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments (default: 1 for visualization)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use deterministic policy (no exploration noise)"
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
import random

# Import unified surgical_project modules (NOT isaaclab_tasks.*)
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env import SurgicalEpigraphEnv
from surgical_project.algorithms.marl.epigraph.trainer import EpigraphTrainer
from scripts.utils.training_helpers_epigraph import summarize_eval_stats


# -------------------------------
# Helper Functions
# -------------------------------
def setup_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Set seed={seed}")


def load_config(checkpoint_path: str, fallback_path: str) -> dict:
    """
    Load configuration with fallback mechanism.
    
    Priority:
    1. Try to find config in checkpoint directory (training config)
    2. Fall back to provided fallback_path (typically DEFAULT_CONFIG_PATH)
    
    This ensures evaluation uses the same config as training when available.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Try to find config file next to checkpoint
    possible_config_names = [
        "training_params_epigraph.yaml",
        "config.yaml",
        "env_config.yaml",
    ]
    
    for config_name in possible_config_names:
        candidate = os.path.join(checkpoint_dir, config_name)
        if os.path.exists(candidate):
            print(f"[CONFIG] Using config found next to checkpoint: {candidate}")
            with open(candidate, 'r') as f:
                return yaml.safe_load(f)
    
    # Fallback to provided path
    if os.path.exists(fallback_path):
        print(f"[CONFIG] Config not found near checkpoint, falling back to: {fallback_path}")
        with open(fallback_path, 'r') as f:
            return yaml.safe_load(f)
    
    raise FileNotFoundError(
        f"No config found in checkpoint directory ({checkpoint_dir}), "
        f"and fallback config not found at: {fallback_path}"
    )


def create_env(config: dict, num_envs: int, seed: int):
    """
    Create evaluation environment (same way as train_epigraph.py).
    """
    # Create environment config
    env_cfg = SurgicalEpigraphEnvCfg()
    
    # Set parallel envs and seed
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    
    # ✅ CRITICAL: Inject YAML params (ensures consistency with training)
    env_cfg.params = config
    
    # Instantiate environment (with rendering for visualization)
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode="human")
    
    # Also inject params to unwrapped env
    actual_env = getattr(env, "unwrapped", env)
    actual_env.params = config
    
    print(f"[ENV] Created evaluation environment with {num_envs} parallel environments")
    return env


def create_trainer(env, config: dict, checkpoint_path: str, device: torch.device):
    """
    Create trainer (same way as train_epigraph.py).
    """
    # Extract algorithm configs
    # NOTE: "algorithms.rmappo" is just a naming convention for hyperparameters
    algo_cfg = config["algorithms"]["rmappo"]
    epi_cfg = config["epigraph"]
    
    # Get max_global_steps from config
    max_global_steps = algo_cfg.get("max_global_steps", 150000)
    
    # Create trainer with full context
    trainer = EpigraphTrainer(
        env=env,
        device=device,
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg,
        full_config=config,
        ckpt_dir=os.path.dirname(checkpoint_path),
        max_global_steps=max_global_steps,
    )
    
    print(f"[TRAINER] EpigraphTrainer created")
    return trainer


def print_eval_summary(stats: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Episodes:              {stats.get('eval_num_episodes', 0)}")
    print(f"Return (mean ± std):   {stats.get('eval_return_mean', 0):.2f} ± {stats.get('eval_return_std', 0):.2f}")
    print(f"Return (min/max):      {stats.get('eval_return_min', 0):.2f} / {stats.get('eval_return_max', 0):.2f}")
    print(f"Episode length:        {stats.get('eval_episode_length_mean', 0):.1f} ± {stats.get('eval_episode_length_std', 0):.1f}")
    print(f"Success rate:          {stats.get('eval_success_rate', 0):.2%}")
    print(f"Safe cost (mean):      {stats.get('eval_safe_cost_mean', 0):.2f}")
    print(f"Safe cost (total):     {stats.get('eval_safe_cost_sum', 0):.2f}")
    if 'eval_z_mean' in stats:
        print(f"Z values (mean ± std): {stats.get('eval_z_mean', 0):.4f} ± {stats.get('eval_z_std', 0):.4f}")
    print("=" * 80 + "\n")


# -------------------------------
# Main Evaluation Function
# -------------------------------
def main():
    print("\n" + "=" * 80)
    print("EPIGRAPH POLICY EVALUATION")
    print("=" * 80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Config:         {args.config}")
    print(f"Num episodes:   {args.num_episodes}")
    print(f"Num envs:       {args.num_envs}")
    print(f"Seed:           {args.seed}")
    print(f"Deterministic:  {args.deterministic}")
    print("=" * 80 + "\n")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # ===== 1. Set Reproducibility =====
    setup_reproducibility(args.seed)
    
    # ===== 2. Load Config =====
    config = load_config(args.checkpoint, args.config)
    
    # ===== 3. Create Environment =====
    env = create_env(config, args.num_envs, args.seed)
    
    # ===== 4. Get Device =====
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}\n")
    
    # ===== 5. Create Trainer =====
    trainer = create_trainer(env, config, args.checkpoint, device)
    
    # ===== 6. Load Checkpoint =====
    print(f"[LOAD] Loading checkpoint from: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # ===== 7. Set Evaluation Mode =====
    trainer.set_eval_mode()
    print("[EVAL] Trainer set to evaluation mode\n")
    
    # ===== 8. Run Evaluation Episodes =====
    print(f"[EVAL] Starting evaluation: {args.num_episodes} episodes\n")
    
    all_episode_returns = []
    all_episode_safe_costs = []
    all_episode_success = []
    all_episode_lengths = []
    all_z_values = []
    
    for ep in range(args.num_episodes):
        # Delegate to trainer's evaluation method
        # This method handles root_finder internally
        ep_stats = trainer.run_single_eval_episode(deterministic=args.deterministic)
        
        # Extract statistics
        all_episode_returns.append(ep_stats['task_return'])
        all_episode_safe_costs.append(ep_stats['safe_cost_sum'])
        all_episode_success.append(ep_stats['success'])
        all_episode_lengths.append(ep_stats['length'])
        all_z_values.append(ep_stats['z_mean'])
        
        # Print episode summary
        print(f"[EVAL] Episode {ep + 1}/{args.num_episodes}: "
              f"return={ep_stats['task_return']:.2f}, "
              f"length={ep_stats['length']}, "
              f"success={ep_stats['success']}, "
              f"safe_cost={ep_stats['safe_cost_sum']:.2f}, "
              f"z_mean={ep_stats['z_mean']:.4f}")
    
    # ===== 9. Aggregate Statistics =====
    eval_stats = summarize_eval_stats(
        episode_returns=all_episode_returns,
        episode_safe_costs=all_episode_safe_costs,
        episode_success=all_episode_success,
        episode_lengths=all_episode_lengths,
        z_values=all_z_values,
    )
    
    # ===== 10. Print Summary =====
    print_eval_summary(eval_stats)
    
    # ===== 11. Save Results =====
    results_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(results_dir, "eval_results.yaml")
    
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        yaml.dump(eval_stats, f, default_flow_style=False)
    print(f"[SAVE] Results saved to: {results_path}")
    
    # ===== 12. Cleanup =====
    env.close()
    print("\n[EVAL] Evaluation complete!")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")
