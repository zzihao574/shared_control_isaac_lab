#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for MADDPG shared-network checkpoints.

Features:
  * Unified CLI aligned with rMAPPO/Epigraph play scripts
  * Deterministic vs stochastic evaluation toggle
  * Shared evaluation recorder for progress/safety/force statistics
  * Support for flat dual checkpoints, final_shared_networks, and legacy top-k saves
  * Optional WandB logging
"""

import argparse
import os
import sys
from typing import Dict, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (REPO_ROOT, SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher

from scripts.utils.eval_logging import EvalRecorder, EvalWandBLogger, wandb_available
from scripts.utils.training_helpers_maddpg import TrainingConfiguration
from train_maddpg import (
    setup_environment,
    initialize_maddpg_algorithm,
    inject_step_tracer,
    setup_global_reproducibility,
)

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "surgical_project",
        "envs",
        "multi_agent",
        "agents",
        "training_params_maddpg.yaml",
    )
)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_config_path(explicit_path: Optional[str], checkpoint_path: str) -> str:
    """Locate configuration file: explicit path, neighbouring checkpoint, or default."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if ckpt_dir:
        candidates.extend(
            [
                os.path.join(ckpt_dir, name)
                for name in (
                    "training_params_maddpg.yaml",
                    "config.yaml",
                    "env_config.yaml",
                )
            ]
        )
    candidates.append(DEFAULT_CONFIG_PATH)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Unable to locate configuration file. Tried: {', '.join(candidates)}"
    )


def resolve_save_dir(save_dir: Optional[str], checkpoint_path: str) -> str:
    if save_dir:
        return save_dir
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return ckpt_dir if ckpt_dir else "."


def load_maddpg_checkpoint(maddpg, checkpoint_path: str):
    """
    Load shared-network MADDPG checkpoint.

    Supports:
      * Flat dual checkpoints (recommended)
      * final_shared_networks.pth
      * Legacy top-k checkpoints (uses rank_1 networks)
    """
    print(f"[LOAD] Loading MADDPG checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=maddpg.device)
    agent_ids = maddpg.agent_ids

    def has_flat_keys(prefix: str = "") -> bool:
        return all(f"{prefix}{aid}_actor" in ckpt and f"{prefix}{aid}_critic" in ckpt for aid in agent_ids)

    loaded_type = None

    if has_flat_keys():
        for aid in agent_ids:
            agent = maddpg.agents[aid]
            agent.actor.load_state_dict(ckpt[f"{aid}_actor"])
            agent.critic.load_state_dict(ckpt[f"{aid}_critic"])
            if f"{aid}_actor_target" in ckpt and f"{aid}_critic_target" in ckpt:
                agent.actor_target.load_state_dict(ckpt[f"{aid}_actor_target"])
                agent.critic_target.load_state_dict(ckpt[f"{aid}_critic_target"])
        loaded_type = "flat"
    elif has_flat_keys("rank_1_"):
        print("[LOAD] Detected legacy top-k checkpoint; using rank_1 weights.")
        for aid in agent_ids:
            agent = maddpg.agents[aid]
            agent.actor.load_state_dict(ckpt[f"rank_1_{aid}_actor"])
            agent.critic.load_state_dict(ckpt[f"rank_1_{aid}_critic"])
            if f"rank_1_{aid}_actor_target" in ckpt and f"rank_1_{aid}_critic_target" in ckpt:
                agent.actor_target.load_state_dict(ckpt[f"rank_1_{aid}_actor_target"])
                agent.critic_target.load_state_dict(ckpt[f"rank_1_{aid}_critic_target"])
        loaded_type = "topk"
    else:
        raise KeyError(
            "Checkpoint does not contain expected MADDPG keys. "
            "Expected flat dual checkpoint (human/robot actor/critic)."
        )

    if "params" in ckpt and isinstance(ckpt["params"], dict):
        maddpg.params = ckpt["params"]
        print("[LOAD] MADDPG params updated from checkpoint.")

    # Optional metadata
    for key in ("milestone", "score", "global_steps_total", "episodes_done_total"):
        if key in ckpt:
            print(f"[LOAD] {key}: {ckpt[key]}")

    print(f"[LOAD] Checkpoint type: {loaded_type}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG shared-network policy.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0", help="Environment task name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per episode (0 = unlimited).")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Use deterministic actions (default).")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Enable stochastic actions with exploration noise.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store evaluation outputs.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable WandB logging.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def evaluate(args):
    config_path = resolve_config_path(args.config, args.checkpoint)
    print(f"[SETUP] Using config: {config_path}")
    config = TrainingConfiguration.from_yaml(config_path)

    setup_global_reproducibility(args.seed, strict_determinism=True)

    # Align evaluation args with training helper expectations
    class EvalArgs:
        pass

    eval_args = EvalArgs()
    eval_args.num_envs = max(1, args.num_envs)
    eval_args.task = args.task
    eval_args.seed = args.seed

    env, _ = setup_environment(eval_args, config)
    actual_env = getattr(env, "unwrapped", env)
    if hasattr(actual_env, "params"):
        actual_env.params = config.params
        print("[SETUP] Injected params into environment.")

    inject_step_tracer(env, config, eval_args.num_envs)

    maddpg = initialize_maddpg_algorithm(env, config, eval_args)
    maddpg.set_eval_mode(True)

    load_maddpg_checkpoint(maddpg, args.checkpoint)

    completion_threshold = (
        config.params.get("reward_parameters", {}).get("completion_threshold", 0.01)
    )
    recorder = EvalRecorder(maddpg.agent_ids, mode="maddpg", completion_threshold=completion_threshold)

    expl_cfg = maddpg.params.get("exploration", {}) if isinstance(maddpg.params, dict) else {}
    eval_noise_scale = 0.0 if args.deterministic else float(expl_cfg.get("sigma_start", 0.7))

    wandb_logger = None
    if args.wandb:
        if wandb_available():
            run_name = f"maddpg_eval_{os.path.basename(args.checkpoint)}"
            wandb_logger = EvalWandBLogger(
                project="maddpg_evaluation",
                run_name=run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "config_path": config_path,
                    "task": args.task,
                    "num_envs": eval_args.num_envs,
                    "deterministic": args.deterministic,
                },
            )
            print("[WANDB] Logging enabled.")
        else:
            print("[WANDB] wandb not available; disabling logging.")

    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        recorder.start_episode(ep)
        obs, _ = env.reset()

        step = 0
        done_any = False
        while step < max_steps and not done_any:
            actions, detail = maddpg.select_actions(
                obs,
                add_noise=not args.deterministic,
                noise_scale=eval_noise_scale,
            )
            actual_env.set_detail_actor_info(detail)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            recorder.record_step(step, env, rewards, info, detail)

            done_mask = None
            for aid in maddpg.agent_ids:
                term = (terminated[aid] | truncated[aid]).to(torch.bool)
                done_mask = term if done_mask is None else (done_mask | term)
            done_any = bool(done_mask is not None and done_mask.any().item())

            obs = next_obs
            step += 1

        summary = recorder.end_episode()
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(
            f"[EPISODE] steps={summary.episode_length} "
            f"score={fmt(summary.score)} "
            f"progress={fmt(summary.progress_final)} "
            f"on_track={fmt(summary.on_track_ratio)} "
            f"c_zone={fmt(summary.c_zone_ratio)} "
            f"collision={fmt(summary.collision_ratio)}"
        )

        if wandb_logger:
            wandb_payload = {
                "eval/score": summary.score,
                "eval/episode_length": summary.episode_length,
                "eval/progress_final": summary.progress_final,
                "eval/on_track_ratio": summary.on_track_ratio,
                "eval/c_zone_ratio": summary.c_zone_ratio,
                "eval/collision_ratio": summary.collision_ratio,
                "eval/time_to_completion_steps": summary.time_to_completion_steps,
            }
            wandb_logger.log_episode(ep, {k: v for k, v in wandb_payload.items() if v is not None})

    save_dir = resolve_save_dir(args.save_dir, args.checkpoint)
    recorder.save(save_dir)
    print(f"[SAVE] Evaluation results written to: {save_dir}")

    if wandb_logger:
        # Upload per-step table
        table_data = [
            [
                rec.episode,
                rec.step,
                rec.reward_mean,
                rec.task_reward_mean,
                rec.safe_cost_mean,
                rec.progress_ratio,
                rec.deviation_m,
                rec.distance_to_final,
                rec.safety_distance,
                int(rec.in_c_zone) if rec.in_c_zone is not None else None,
                int(rec.is_colliding) if rec.is_colliding is not None else None,
            ]
            for rec in recorder.all_step_records
        ]
        wandb_logger.log_table(
            "eval_steps",
            [
                "episode",
                "step",
                "reward_mean",
                "task_reward_mean",
                "safe_cost_mean",
                "progress_ratio",
                "deviation_m",
                "distance_to_final",
                "safety_distance",
                "in_c_zone",
                "is_colliding",
            ],
            table_data,
        )
        wandb_logger.finish()


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        evaluate(args)
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    main()
