#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for Epigraph policies with unified CLI/output.

Features:
  * Shared evaluation recorder for step/episode metrics
  * Deterministic/stochastic toggles
  * Optional WandB logging
  * Checkpoint/config discovery compatible with training script conventions
"""

import argparse
import os
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (REPO_ROOT, SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher

from scripts.utils.eval_logging import EvalRecorder, EvalWandBLogger, wandb_available

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "surgical_project",
        "envs",
        "multi_agent_epigraph",
        "agents",
        "training_params_epigraph.yaml",
    )
)


def resolve_config_path(explicit: Optional[str], checkpoint: str) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    if ckpt_dir:
        candidates.extend(
            [
                os.path.join(ckpt_dir, name)
                for name in (
                    "training_params_epigraph.yaml",
                    "config.yaml",
                    "env_config.yaml",
                )
            ]
        )
    candidates.append(DEFAULT_CONFIG_PATH)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to locate config file. Tried: {', '.join(candidates)}")


def resolve_save_dir(requested: Optional[str], checkpoint: str) -> str:
    if requested:
        return requested
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    return ckpt_dir if ckpt_dir else "."


def setup_reproducibility(seed: int, strict_determinism: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[SEED] Set seed={seed} strict={strict_determinism}")


def load_config(checkpoint_path: str, fallback_path: str) -> dict:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    possible_names = [
        "training_params_epigraph.yaml",
        "config.yaml",
        "env_config.yaml",
    ]
    for name in possible_names:
        candidate = os.path.join(checkpoint_dir, name)
        if os.path.exists(candidate):
            print(f"[CONFIG] Using checkpoint-adjacent config: {candidate}")
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
    print(f"[CONFIG] Falling back to: {fallback_path}")
    with open(fallback_path, "r") as f:
        return yaml.safe_load(f)


def create_env(config: dict, num_envs: int, seed: int):
    from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env_cfg import (
        SurgicalEpigraphEnvCfg,
    )
    from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env import (
        SurgicalEpigraphEnv,
    )

    env_cfg = SurgicalEpigraphEnvCfg()
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    env_cfg.params = config
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode="human")
    actual_env = getattr(env, "unwrapped", env)
    actual_env.params = config
    return env


def create_trainer(
    env, config: dict, checkpoint_path: str, device: torch.device
) -> Any:
    from surgical_project.algorithms.marl.epigraph.trainer import EpigraphTrainer

    algo_cfg = config["algorithms"]["rmappo"]
    epi_cfg = config["epigraph"]
    max_global_steps = algo_cfg.get("max_global_steps", 150000)
    trainer = EpigraphTrainer(
        env=env,
        device=device,
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg,
        full_config=config,
        ckpt_dir=os.path.join(os.path.dirname(checkpoint_path), "checkpoints"),
        max_global_steps=max_global_steps,
    )
    return trainer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained Epigraph policy.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt/.pth).")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per episode (0 = unlimited).")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Use deterministic policy (default).")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Enable stochastic evaluation.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store evaluation outputs.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable WandB logging.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def run_eval_episode(trainer: Any, deterministic: bool, max_steps: int, recorder: EvalRecorder) -> Dict[str, float]:
    trainer.set_eval_mode()
    obs_raw, _ = trainer.env.reset()
    obs = trainer._scale_obs(obs_raw)
    trainer._init_rnn_states()

    actual_env = getattr(trainer.env, "unwrapped", trainer.env)
    ones_mask = torch.ones(trainer.num_envs, 1, device=trainer.device)
    step_tracer = getattr(actual_env, "step_tracer", None)
    saved_enable = getattr(step_tracer, "enable_console_logging", None) if step_tracer else None
    saved_every = getattr(step_tracer, "print_every_steps", None) if step_tracer else None
    if step_tracer is not None:
        if not step_tracer.enable_console_logging:
            step_tracer.enable_console_logging = True
        if not saved_every or saved_every <= 0:
            step_tracer.print_every_steps = 10

    episode_task_return = 0.0
    episode_safe_cost = 0.0
    episode_violations = 0
    z_values = []
    done = False
    step = 0
    info = {}
    limit = max_steps if max_steps > 0 else 2000  # trainer default safeguard

    def _flag_to_bool(val):
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return False
            return bool(val.reshape(-1)[0].item())
        return bool(val)

    try:
        while not done and step < limit:
            z_candidates = []
            for agent in trainer.agent_ids:

                def vh_eval_fn(z_query: torch.Tensor, agent_id=agent):
                    z_enc_query = trainer.z_encoder_vh(z_query)
                    vh_pred, _ = trainer.critics_vh[agent_id].value_step(
                        obs[agent_id],
                        z_enc_query,
                        trainer.rnn_states_vh[agent_id],
                        ones_mask,
                    )
                    return vh_pred

                z_i_star = trainer.root_finder.solve(
                    vh_eval_fn=vh_eval_fn,
                    obs=obs[agent],
                )
                z_candidates.append(z_i_star)

            z_global = torch.max(torch.stack(z_candidates, dim=0), dim=0)[0]
            z_values.append(float(z_global[0].mean().item()))

            actions = {}
            z_enc_actor = trainer.z_encoder_actor(z_global)
            for agent in trainer.agent_ids:
                act_a, _, next_h_a, _ = trainer.actors[agent].act_step(
                    obs[agent],
                    z_enc_actor,
                    trainer.rnn_states_actor[agent],
                    ones_mask,
                    deterministic=deterministic,
                )
                actions[agent] = act_a
                trainer.rnn_states_actor[agent] = next_h_a

            env_actions = trainer._scale_actions(actions)
            detail_payload = {
                "applied_forces": {aid: env_actions[aid].clone() for aid in trainer.agent_ids},
                "mean_actions": {aid: env_actions[aid].clone() for aid in trainer.agent_ids},
                "noise_actions": {aid: torch.zeros_like(env_actions[aid]) for aid in trainer.agent_ids},
                "deterministic": deterministic,
            }
            if hasattr(actual_env, "set_detail_actor_info"):
                actual_env.set_detail_actor_info(detail_payload)
            if hasattr(actual_env, "_trainer_global_step"):
                actual_env._trainer_global_step = trainer.global_step
            actual_env._last_z_snapshot = z_global.clone().detach()

            obs_raw, rewards, terminated, truncated, info = trainer.env.step(env_actions)
            obs = trainer._scale_obs(obs_raw)

            recorder.record_step(step, trainer.env, rewards, info, detail_payload)

            # Aggregate env0 task/safe returns
            step_task_vals = []
            step_safe_vals = []
            for agent in trainer.agent_ids:
                if "r_task" in info and agent in info["r_task"]:
                    val = info["r_task"][agent]
                    if isinstance(val, torch.Tensor):
                        val = val.squeeze(-1) if val.dim() > 1 else val
                        step_task_vals.append(val[0])
                    else:
                        step_task_vals.append(torch.as_tensor(float(val), device=trainer.device))
                if "r_safe" in info and agent in info["r_safe"]:
                    val = info["r_safe"][agent]
                    if isinstance(val, torch.Tensor):
                        val = val.squeeze(-1) if val.dim() > 1 else val
                        step_safe_vals.append(val[0])
                    else:
                        step_safe_vals.append(torch.as_tensor(float(val), device=trainer.device))

            if step_task_vals:
                episode_task_return += float(torch.stack(step_task_vals).mean().item())
            if step_safe_vals:
                episode_safe_cost += float(torch.stack(step_safe_vals).mean().item())

            if "is_violating" in info and info["is_violating"] is not None:
                viol = info["is_violating"]
                if isinstance(viol, torch.Tensor):
                    while viol.dim() > 2:
                        viol = viol.squeeze(-1)
                    viol_env0 = viol[0]
                    if isinstance(viol_env0, torch.Tensor):
                        episode_violations += int(viol_env0.float().sum().item())
                    else:
                        episode_violations += int(bool(viol_env0))
                else:
                    episode_violations += int(bool(viol))

            step += 1

            done_env0 = False
            for agent in trainer.agent_ids:
                term_flag = _flag_to_bool(terminated[agent])
                trunc_flag = _flag_to_bool(truncated[agent])
                done_env0 = done_env0 or term_flag or trunc_flag
            done = done_env0

            share_obs_next = trainer._get_share_obs(obs)
            z_enc_vl = trainer.z_encoder_vl(z_global)
            _, trainer.rnn_states_vl = trainer.critic_vl.value_step(
                share_obs_next,
                z_enc_vl,
                trainer.rnn_states_vl,
                ones_mask,
            )
            z_enc_vh = trainer.z_encoder_vh(z_global)
            for agent in trainer.agent_ids:
                _, trainer.rnn_states_vh[agent] = trainer.critics_vh[agent].value_step(
                    obs[agent],
                    z_enc_vh,
                    trainer.rnn_states_vh[agent],
                    ones_mask,
                )
    finally:
        if step_tracer is not None:
            if saved_enable is not None:
                step_tracer.enable_console_logging = saved_enable
            if saved_every is not None:
                step_tracer.print_every_steps = saved_every

    progress_final = None
    if recorder._current_steps:
        progress_final = recorder._current_steps[-1].progress_ratio
    success = progress_final is not None and progress_final >= 0.95

    return {
        "task_return": episode_task_return,
        "safe_cost_sum": episode_safe_cost,
        "length": step,
        "violations": episode_violations,
        "z_mean": float(np.mean(z_values)) if z_values else 0.0,
        "success": success,
    }


def evaluate(args):
    config_path = resolve_config_path(args.config, args.checkpoint)
    print(f"[SETUP] Using config: {config_path}")
    config = load_config(args.checkpoint, config_path)

    setup_reproducibility(args.seed, strict_determinism=True)

    env = create_env(config, args.num_envs, args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = create_trainer(env, config, args.checkpoint, device=device)

    print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    trainer.set_eval_mode()

    completion_threshold = config.get("reward_parameters", {}).get("completion_threshold", 0.01)
    recorder = EvalRecorder(trainer.agent_ids, mode="epigraph", completion_threshold=completion_threshold)

    wandb_logger = None
    if args.wandb:
        if wandb_available():
            run_name = f"epigraph_eval_{os.path.basename(args.checkpoint)}"
            wandb_logger = EvalWandBLogger(
                project="epigraph_evaluation",
                run_name=run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "config_path": config_path,
                    "num_envs": args.num_envs,
                    "deterministic": args.deterministic,
                },
            )
            print("[WANDB] Logging enabled.")
        else:
            print("[WANDB] wandb not available; logging disabled.")

    max_steps = args.max_steps if args.max_steps > 0 else 2000

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        recorder.start_episode(ep)
        ep_stats = run_eval_episode(trainer, args.deterministic, max_steps, recorder)
        summary = recorder.end_episode()

        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(
            f"[EPISODE] steps={summary.episode_length} "
            f"score={fmt(summary.score)} "
            f"progress={fmt(summary.progress_final)} "
            f"on_track={fmt(summary.on_track_ratio)} "
            f"c_zone={fmt(summary.c_zone_ratio)} "
            f"collision={fmt(summary.collision_ratio)} "
            f"task_return={ep_stats['task_return']:.3f} "
            f"safe_cost={ep_stats['safe_cost_sum']:.3f}"
        )

        if wandb_logger:
            wandb_payload = {
                "eval/score": summary.score,
                "eval/episode_length": summary.episode_length,
                "eval/progress_final": summary.progress_final,
                "eval/on_track_ratio": summary.on_track_ratio,
                "eval/c_zone_ratio": summary.c_zone_ratio,
                "eval/collision_ratio": summary.collision_ratio,
                "eval/task_return": ep_stats["task_return"],
                "eval/safe_cost": ep_stats["safe_cost_sum"],
                "eval/time_to_completion_steps": summary.time_to_completion_steps,
                "eval/success": float(ep_stats["success"]),
            }
            wandb_logger.log_episode(ep, {k: v for k, v in wandb_payload.items() if v is not None})

    save_dir = resolve_save_dir(args.save_dir, args.checkpoint)
    recorder.save(save_dir)
    print(f"[SAVE] Evaluation artifacts written to: {save_dir}")

    aggregates = recorder._build_aggregates()
    if aggregates:
        print("\n[EVALUATION SUMMARY]")
        for key, stats in aggregates.items():
            print(
                f"  {key}: mean={stats['mean']:.3f} std={stats['std']:.3f} "
                f"min={stats['min']:.3f} max={stats['max']:.3f}"
            )

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

    trainer.env.close()


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    caught_error: Optional[BaseException] = None

    try:
        try:
            evaluate(args)
        except Exception as exc:
            print("[DEBUG] Evaluation raised an exception; printing traceback:")
            traceback.print_exc()
            caught_error = exc
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")

    if caught_error is not None:
        raise caught_error


if __name__ == "__main__":
    main()
