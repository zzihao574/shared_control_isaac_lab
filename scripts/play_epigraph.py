"""
Play/Evaluation script for trained Epigraph policies.
Loads checkpoint and runs evaluation episodes with RootFinder for safe z* computation.

FIXED: 
1. Corrected sys.path to use ".." instead of "."
2. Uses proper RootFinder logic (solve for minimum safe z per agent, then max)
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# FIXED: Correct sys.path (use ".." to go up one level from scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_tasks.manager_based.surgical_epigraph import agents as surgical_agents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained Epigraph policy")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
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
        "--render",
        action="store_true",
        help="Enable rendering/visualization"
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record video of evaluation"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./videos",
        help="Directory to save videos"
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
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration noise)"
    )
    parser.add_argument(
        "--use_root_finder",
        action="store_true",
        default=True,
        help="Use RootFinder to compute safe z* (default: True)"
    )
    parser.add_argument(
        "--z_fixed",
        type=float,
        default=None,
        help="If set, use fixed z value instead of RootFinder"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step information"
    )
    
    return parser.parse_args()


def load_config(checkpoint_path: str):
    """
    Load configuration from checkpoint directory.
    Assumes config YAML is in same directory as checkpoint.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Try to find config file
    possible_config_names = [
        "training_params_epigraph.yaml",
        "config.yaml",
        "env_config.yaml"
    ]
    
    config_path = None
    for config_name in possible_config_names:
        candidate = os.path.join(checkpoint_dir, config_name)
        if os.path.exists(candidate):
            config_path = candidate
            break
    
    # Fallback: look in agents directory
    if config_path is None:
        agents_dir = os.path.join(project_root, "isaaclab_tasks", "manager_based", "surgical_epigraph", "agents")
        config_path = os.path.join(agents_dir, "training_params_epigraph.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config YAML. Tried: {possible_config_names}")
    
    print(f"[PLAY] Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_env(config: dict, num_envs: int, render: bool, device: str):
    """Create evaluation environment."""
    from isaaclab_tasks.manager_based.surgical_epigraph.surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
    
    # Create environment config
    env_cfg = SurgicalEpigraphEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Inject configuration
    env_cfg.params = config
    
    # Create environment
    from isaaclab_tasks.manager_based.surgical_epigraph.surgical_epigraph_env import SurgicalEpigraphEnv
    
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode="human" if render else None)
    
    # Inject params to unwrapped env as well
    if hasattr(env, 'unwrapped'):
        env.unwrapped.params = config
    
    print(f"[PLAY] Environment created: {num_envs} envs")
    return env


def create_trainer(env, config: dict, device: str):
    """Create trainer (for loading networks only, no training)."""
    from isaaclab_tasks.manager_based.surgical_epigraph.agents.trainer import EpigraphTrainer
    
    algo_cfg = config["algorithms"]["rmappo"]
    epi_cfg = config["epigraph"]
    
    trainer = EpigraphTrainer(
        env=env,
        device=torch.device(device),
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg
    )
    
    print(f"[PLAY] Trainer created")
    return trainer


def evaluate_policy(
    trainer,
    env,
    num_episodes: int,
    deterministic: bool,
    use_root_finder: bool,
    z_fixed: float = None,
    verbose: bool = False
):
    """
    Evaluate trained policy.
    
    Args:
        trainer: EpigraphTrainer with loaded checkpoint
        env: Environment
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        use_root_finder: Use RootFinder to compute safe z*
        z_fixed: If set, use this fixed z value
        verbose: Print step information
    
    Returns:
        eval_stats: Dictionary of evaluation statistics
    """
    trainer.set_eval_mode()
    
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    z_values = []
    safety_violations = []
    
    print(f"\n[PLAY] Starting evaluation: {num_episodes} episodes")
    print(f"[PLAY] Deterministic: {deterministic}, RootFinder: {use_root_finder}, z_fixed: {z_fixed}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        trainer._init_rnn_states()
        
        episode_return = 0.0
        episode_length = 0
        episode_violations = 0
        done = False
        
        if verbose:
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done and episode_length < 2000:
            # ========== Compute z* ==========
            if z_fixed is not None:
                # Use fixed z
                z_global = torch.full((env.num_envs, 1), z_fixed, device=trainer.device)
            elif use_root_finder:
                # FIXED: Use RootFinder to solve for safe z* per agent, then max
                z_stars = []
                for agent in trainer.agent_ids:
                    z_i_star = trainer._solve_safe_z_for_agent(
                        critic_vh=trainer.critics_vh[agent],
                        obs=obs[agent],
                        rnn_state=trainer.rnn_states[agent]["vh"],
                        h_tgt=0.0  # Safety threshold
                    )
                    z_stars.append(z_i_star)
                
                # Take maximum across agents
                z_global = torch.max(torch.stack(z_stars, dim=0), dim=0)[0]
            else:
                # Random z
                z_global = torch.rand(env.num_envs, 1, device=trainer.device) * \
                           (trainer.z_max - trainer.z_min) + trainer.z_min
            
            z_values.append(z_global.mean().item())
            
            # ========== Encode z (shared by all agents) ==========
            z_enc_global = trainer.z_encoder(z_global)
            
            # ========== Actor forward ==========
            actions = {}
            for agent in trainer.agent_ids:
                obs_agent = obs[agent]
                masks = torch.ones(env.num_envs, 1, device=trainer.device)
                
                act, _, rnn_h = trainer.actors[agent](
                    obs_agent, z_enc_global,  # All agents use shared z_global
                    trainer.rnn_states[agent]["actor"],
                    masks,
                    deterministic=deterministic
                )
                
                actions[agent] = act
                trainer.rnn_states[agent]["actor"] = rnn_h
            
            # ========== Step environment ==========
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate statistics
            for agent in trainer.agent_ids:
                episode_return += rewards[agent].mean().item()
            episode_length += 1
            
            # Check safety violations
            if "is_violating" in info:
                episode_violations += info["is_violating"].sum().item()
            
            # Verbose logging
            if verbose and episode_length % 50 == 0:
                print(f"  Step {episode_length}: z={z_global.mean().item():.4f}, "
                      f"return={episode_return:.2f}")
            
            # Check done
            done_any = torch.zeros(env.num_envs, dtype=torch.bool, device=trainer.device)
            for agent in trainer.agent_ids:
                agent_done = terminated[agent] | truncated[agent]
                if agent_done.dim() > 1:
                    agent_done = agent_done.squeeze(-1)
                done_any |= agent_done
            
            if done_any.any():
                done = True
        
        # Episode statistics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        safety_violations.append(episode_violations)
        
        # Check success (if task completed)
        if "progress_ratio" in info:
            progress = info["progress_ratio"].mean().item()
            episode_successes.append(1.0 if progress >= 0.95 else 0.0)
        else:
            episode_successes.append(0.0)
        
        print(f"[PLAY] Episode {episode + 1}: return={episode_return:.2f}, "
              f"length={episode_length}, violations={episode_violations}, "
              f"z_mean={np.mean(z_values[-episode_length:]):.4f}")
    
    # Aggregate statistics
    eval_stats = {
        "return_mean": np.mean(episode_returns),
        "return_std": np.std(episode_returns),
        "return_min": np.min(episode_returns),
        "return_max": np.max(episode_returns),
        "episode_length_mean": np.mean(episode_lengths),
        "episode_length_std": np.std(episode_lengths),
        "success_rate": np.mean(episode_successes),
        "safety_violations_mean": np.mean(safety_violations),
        "safety_violations_total": np.sum(safety_violations),
        "z_mean": np.mean(z_values),
        "z_std": np.std(z_values),
        "z_min": np.min(z_values),
        "z_max": np.max(z_values),
    }
    
    return eval_stats


def print_eval_summary(stats: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:              {len(stats)}")
    print(f"Return (mean ± std):   {stats['return_mean']:.2f} ± {stats['return_std']:.2f}")
    print(f"Return (min/max):      {stats['return_min']:.2f} / {stats['return_max']:.2f}")
    print(f"Episode length:        {stats['episode_length_mean']:.1f} ± {stats['episode_length_std']:.1f}")
    print(f"Success rate:          {stats['success_rate']:.2%}")
    print(f"Safety violations:     {stats['safety_violations_mean']:.1f} (total: {stats['safety_violations_total']:.0f})")
    print(f"Z values:              {stats['z_mean']:.4f} ± {stats['z_std']:.4f}")
    print(f"Z range:               [{stats['z_min']:.4f}, {stats['z_max']:.4f}]")
    print("=" * 60 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "=" * 60)
    print("EPIGRAPH POLICY EVALUATION")
    print("=" * 60)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Num episodes:   {args.num_episodes}")
    print(f"Num envs:       {args.num_envs}")
    print(f"Device:         {args.device}")
    print(f"Seed:           {args.seed}")
    print("=" * 60 + "\n")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load config
    config = load_config(args.checkpoint)
    
    # Create environment
    env = create_env(config, args.num_envs, args.render, args.device)
    
    # Create trainer
    trainer = create_trainer(env, config, args.device)
    
    # Load checkpoint
    print(f"[PLAY] Loading checkpoint from: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # Evaluate
    eval_stats = evaluate_policy(
        trainer=trainer,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        use_root_finder=args.use_root_finder,
        z_fixed=args.z_fixed,
        verbose=args.verbose
    )
    
    # Print summary
    print_eval_summary(eval_stats)
    
    # Save results
    results_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(results_dir, "eval_results.yaml")
    
    with open(results_path, 'w') as f:
        yaml.dump(eval_stats, f, default_flow_style=False)
    print(f"[PLAY] Results saved to: {results_path}")
    
    # Close environment
    env.close()
    
    print("\n[PLAY] Evaluation complete!")


if __name__ == "__main__":
    main()