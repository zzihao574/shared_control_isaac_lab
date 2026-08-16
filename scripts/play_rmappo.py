#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for rMAPPO dual-network checkpoints.

Key features:
  * Unified CLI (aligned with Epigraph/MADDPG play scripts)
  * Deterministic/stochastic toggle
  * Shared evaluation recorder for progress/safety/force statistics
  * Optional WandB logging
"""

import argparse
import copy
import os
import sys
from typing import Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (REPO_ROOT, SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher

from scripts.utils.eval_logging import (
    EvalRecorder,
    EvalWandBLogger,
    print_paper_metrics,
    resolve_best_checkpoint,
    wandb_available,
)
from scripts.utils.training_helpers_rmappo import TrainingConfiguration, MetricsHub
from train_rmappo import (
    setup_global_reproducibility,
    setup_environment,
    initialize_rmappo_algorithm,
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
        "training_params_rmappo.yaml",
    )
)
DEFAULT_FIXED_CHECKPOINT_ROOT = os.path.join(
    REPO_ROOT, "logs", "rmappo_dual", "fixed_impedance"
)


def resolve_config_path(explicit_path: Optional[str], checkpoint_path: str) -> str:
    """Locate configuration file using explicit path, checkpoint directory, or default."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if ckpt_dir:
        candidates.extend(
            [
                os.path.join(ckpt_dir, name)
                for name in (
                    "training_params_rmappo.yaml",
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


def resolve_save_dir(requested: Optional[str], checkpoint_path: str) -> str:
    if requested:
        return requested
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return ckpt_dir if ckpt_dir else "."


def get_hidden_size(config: TrainingConfiguration, default: int = 256) -> int:
    try:
        return int(config.params["algorithms"]["rmappo"]["hidden_size"])
    except Exception:
        print(f"[WARN] 'algorithms.rmappo.hidden_size' missing. Using default={default}.")
        return default


def resolve_evaluation_config(args):
    """Restore the checkpoint experiment conditions before Isaac Sim starts."""
    config_path = resolve_config_path(args.config, args.checkpoint)
    config = TrainingConfiguration.from_yaml(config_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint.get("params"), dict):
        config.params = copy.deepcopy(checkpoint["params"])
        print("[SETUP] Restored resolved configuration embedded in checkpoint.")

    checkpoint_seed = int(
        config.params.get("seed", config.params.get("training", {}).get("seed", 42))
    )
    args.seed = checkpoint_seed if args.seed is None else int(args.seed)
    config.params["seed"] = args.seed
    config.params.setdefault("training", {})["seed"] = args.seed
    config.params.setdefault("human_model_type", "learnable")
    return config_path, config


def load_checkpoint(rmappo_wrapper, checkpoint_path: str):
    """Load flat dual-network checkpoint into wrapper."""
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=rmappo_wrapper.device, weights_only=False)
    flat_keys = {"human_actor", "human_critic", "robot_actor", "robot_critic"}
    if not flat_keys.issubset(ckpt.keys()):
        raise KeyError(
            f"Checkpoint missing required keys. Expected: {flat_keys}. "
            f"Found: {list(ckpt.keys())[:20]}..."
        )
    rmappo_wrapper.policies["human"].actor.load_state_dict(ckpt["human_actor"])
    rmappo_wrapper.policies["human"].critic.load_state_dict(ckpt["human_critic"])
    rmappo_wrapper.policies["robot"].actor.load_state_dict(ckpt["robot_actor"])
    rmappo_wrapper.policies["robot"].critic.load_state_dict(ckpt["robot_critic"])
    for key in ("milestone", "score", "global_steps_total", "episodes_done_total"):
        if key in ckpt:
            print(f"[LOAD] {key}: {ckpt[key]}")


def inject_step_tracer(env, config, num_envs):
    """Inject StepTracer for detailed console logging (controlled by YAML)."""
    actual_env = getattr(env, "unwrapped", env)
    from surgical_project.envs.multi_agent.utils import StepTracer

    enable_logging = config.params.get("logging", {}).get("enable_console_logging", False)
    actual_env.step_tracer = StepTracer(
        num_envs=num_envs,
        device=getattr(actual_env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        enable_console_logging=enable_logging,
        print_every_steps=int(
            config.params.get("logging", {}).get("print_every_steps", 10)
        ),
    )
    status = "Enabled" if enable_logging else "Disabled"
    print(f"[STEPTRACER] {status} (controlled by logging.enable_console_logging)")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained rMAPPO dual-network policy.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file. If omitted, the highest-score milestone "
            "below --checkpoint_root is selected."
        ),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=DEFAULT_FIXED_CHECKPOINT_ROOT,
        help="Root searched when --checkpoint is omitted.",
    )
    parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0", help="Environment task name.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per episode (0 = unlimited).")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Use deterministic actions (default).")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Enable stochastic action sampling.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store evaluation outputs.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable WandB logging.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def evaluate(args, simulation_app, config_path, config):
    print(f"[SETUP] Using config: {config_path}")

    eval_num_envs = max(1, args.num_envs)

    # prepare minimal namespace for environment helpers
    class EvalArgs:
        pass

    env_args = EvalArgs()
    env_args.num_envs = eval_num_envs
    env_args.task = args.task
    env_args.seed = args.seed
    env_args.checkpoint = args.checkpoint  # for logging compatibility

    print(f"[SETUP] Creating environment with num_envs={eval_num_envs} ...")
    env, _ = setup_environment(env_args, config)
    actual_env = getattr(env, "unwrapped", env)

    inject_step_tracer(env, config, eval_num_envs)

    metrics_hub = MetricsHub()
    rmappo = initialize_rmappo_algorithm(env, config, env_args, metrics_hub)
    rmappo.set_eval_mode(True)

    load_checkpoint(rmappo, args.checkpoint)

    hidden_size = get_hidden_size(config)
    completion_threshold = config.params.get("reward_parameters", {}).get("completion_threshold", 0.01)
    recorder = EvalRecorder(rmappo.agent_ids, mode="rmappo", completion_threshold=completion_threshold)

    wandb_logger = None
    if args.wandb:
        if wandb_available():
            run_name = f"rmappo_eval_{os.path.basename(args.checkpoint)}"
            wandb_logger = EvalWandBLogger(
                project="rmappo_evaluation",
                run_name=run_name,
                config={
                    **config.params,
                    "checkpoint": args.checkpoint,
                    "config_path": config_path,
                    "task": args.task,
                    "num_envs": eval_num_envs,
                    "deterministic": args.deterministic,
                },
            )
            print("[WANDB] Logging enabled.")
        else:
            print("[WANDB] wandb not available; logging disabled.")

    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    if not hasattr(actual_env, "set_evaluation_active_env"):
        raise RuntimeError("Environment does not support single-environment evaluation")

    actual_env.set_evaluation_active_env(0)
    try:
        for ep in range(args.num_episodes):
            print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
            recorder.start_episode(ep)
            obs, _ = env.reset()

            for aid in rmappo.agent_ids:
                rmappo.rnn_states[aid]["actor"] = torch.zeros(
                    eval_num_envs, hidden_size, device=rmappo.device
                )
                rmappo.rnn_states[aid]["critic"] = torch.zeros(
                    eval_num_envs, hidden_size, device=rmappo.device
                )

            step = 0
            done_any = False
            while simulation_app.is_running() and step < max_steps and not done_any:
                actions, detail = rmappo.select_actions(obs, deterministic=args.deterministic)
                actual_env.set_detail_actor_info(detail)

                actual_env.set_trainer_global_step(step)
                obs, rewards, terminated, truncated, info = env.step(actions)
                if hasattr(actual_env, "get_force_breakdown"):
                    breakdown = actual_env.get_force_breakdown()
                    detail["force_breakdown"] = breakdown
                    detail["applied_forces"] = {
                        "human": breakdown["human"],
                        "robot": breakdown["robot"],
                    }
                recorder.record_step(step, env, rewards, info, detail)

                done_mask = None
                for aid in rmappo.agent_ids:
                    term = (terminated[aid] | truncated[aid]).to(torch.bool)
                    done_mask = term if done_mask is None else (done_mask | term)
                done_any = bool(done_mask is not None and done_mask[0].item())
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

            if not simulation_app.is_running():
                print("[WARN] Simulation app stopped running; terminating evaluation early.")
                break
    finally:
        actual_env.clear_evaluation_active_env()

    save_dir = resolve_save_dir(args.save_dir, args.checkpoint)
    recorder.save(save_dir)
    print(f"[SAVE] Evaluation artifacts written to: {save_dir}")
    print_paper_metrics(recorder)

    if wandb_logger:
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
    args.checkpoint = resolve_best_checkpoint(args.checkpoint, args.checkpoint_root)

    config_path, config = resolve_evaluation_config(args)
    setup_global_reproducibility(args.seed, strict_determinism=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        evaluate(args, simulation_app, config_path, config)
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    main()
