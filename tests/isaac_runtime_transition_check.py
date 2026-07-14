#!/usr/bin/env python3
"""Manual Isaac Lab integration check for MADDPG transition bookkeeping.

Run this script through the Isaac Lab Python environment.  It intentionally
launches the simulator with a viewer so it also exercises the supported 5.x
non-headless startup path.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    import yaml

    import surgical_project.envs.multi_agent  # noqa: F401
    from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import (
        SurgicalDirectMARLEnvCfg,
    )

    config_path = (
        REPO_ROOT
        / "src"
        / "surgical_project"
        / "envs"
        / "multi_agent"
        / "agents"
        / "training_params_maddpg.yaml"
    )
    with config_path.open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    params = copy.deepcopy(params)
    params["human_model_type"] = "fixed_impedance"

    env_cfg = SurgicalDirectMARLEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.seed = 42
    env_cfg.params = params

    env = gym.make("Isaac-Surgical-MARL-Direct-v0", cfg=env_cfg)
    try:
        actual_env = env.unwrapped
        observations, _ = env.reset(seed=42)
        zero_actions = {
            agent: torch.zeros(
                (2, actual_env.cfg.action_spaces[agent]), device=actual_env.device
            )
            for agent in actual_env.cfg.possible_agents
        }

        # On an ordinary transition, reward components and returned observation
        # must describe the same post-physics state.
        next_observations, _, terminated, truncated, _ = env.step(zero_actions)
        assert not any(bool(value.any().item()) for value in terminated.values())
        assert not any(bool(value.any().item()) for value in truncated.values())
        robot_observation = next_observations["robot"]
        torch.testing.assert_close(
            actual_env.reward_components["deviation"], robot_observation[:, 0]
        )
        torch.testing.assert_close(
            actual_env.reward_components["min_safety_distance"],
            robot_observation[:, 4],
        )
        torch.testing.assert_close(
            actual_env.reward_components["progress_ratio"], robot_observation[:, 5]
        )
        print("[PASS] Reward and observation use the same post-physics state.")

        # In-place milestone evaluation must physically advance env0 only. The
        # mask is applied after fixed-impedance composition, so every applied
        # force channel for env1 must remain zero.
        actual_env.set_evaluation_active_env(0)
        env.reset()
        inactive_start = actual_env._get_stylus_position()[1].clone()
        eval_robot_commands = torch.tensor(
            [[0.012, -0.007, 0.003], [0.02, 0.01, -0.01]],
            device=actual_env.device,
        )
        eval_actions = {
            "human": torch.zeros((2, 3), device=actual_env.device),
            "robot": eval_robot_commands,
        }
        for _ in range(10):
            env.step(eval_actions)

        evaluation_breakdown = actual_env.get_force_breakdown()
        torch.testing.assert_close(
            evaluation_breakdown["robot"][0], eval_robot_commands[0]
        )
        for channel in (
            "human_policy",
            "human_impedance",
            "human_residual",
            "human",
            "robot",
            "combined",
        ):
            torch.testing.assert_close(
                evaluation_breakdown[channel][1],
                torch.zeros(3, device=actual_env.device),
            )
        inactive_end = actual_env._get_stylus_position()[1]
        torch.testing.assert_close(inactive_end, inactive_start, atol=1e-5, rtol=0.0)

        actual_env.clear_evaluation_active_env()
        env.reset()
        env.step(zero_actions)
        restored_breakdown = actual_env.get_force_breakdown()
        assert torch.linalg.vector_norm(restored_breakdown["human_impedance"][1]).item() > 0.0
        print("[PASS] Evaluation advances/records env0 only and restores parallel forces.")

        # Force the next step to hit the configured 20-second time limit.  Isaac
        # Lab auto-resets before step() returns, but replay must still see the
        # action that generated the terminal transition.
        actual_env.episode_length_buf.fill_(actual_env.max_episode_length - 2)
        robot_command = torch.tensor(
            [[0.012, -0.007, 0.003], [0.006, 0.004, -0.002]],
            device=actual_env.device,
        )
        terminal_actions = {
            "human": torch.zeros((2, 3), device=actual_env.device),
            "robot": robot_command,
        }
        _, _, _, terminal_truncated, _ = env.step(terminal_actions)
        assert bool(terminal_truncated["robot"].all().item())

        force_breakdown = actual_env.get_force_breakdown()
        torch.testing.assert_close(force_breakdown["robot"], robot_command)
        torch.testing.assert_close(
            actual_env.get_applied_impedance_force(),
            force_breakdown["human_impedance"],
        )
        assert torch.linalg.vector_norm(force_breakdown["human_impedance"]).item() > 0.0
        print("[PASS] Auto-reset preserves terminal robot and impedance forces for replay.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
