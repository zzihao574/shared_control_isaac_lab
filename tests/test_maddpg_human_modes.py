from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_project.algorithms.marl.maddpg import MADDPG
from scripts.utils.training_helpers_maddpg import SeedPlan


class MockEnvironment:
    def __init__(self):
        self.cfg = SimpleNamespace(
            possible_agents=["human", "robot"],
            observation_spaces={"human": 6, "robot": 6},
            action_spaces={"human": 3, "robot": 3},
        )


def make_params(model_type: str) -> dict:
    return {
        "seed": 7,
        "human_model_type": model_type,
        "constraints": {"max_human_force": 0.04, "max_robot_force": 0.04},
        "obs_scaling": {"factors": [1.0] * 6},
        "networks": {
            "actor": {"hidden_layers": [16], "orthogonal_init": False},
            "critic": {"hidden_layers": [16], "orthogonal_init": False},
        },
        "maddpg_config": {
            "lr_actor": 1e-3,
            "lr_critic": 1e-3,
            "gamma": 0.99,
            "batch_size": 2,
            "update_interval": 1,
            "min_buffer_size": 1,
            "max_replay_buffer_len": 16,
            "tau": 0.01,
        },
    }


def fill_replay(maddpg: MADDPG) -> None:
    generator = np.random.default_rng(123)
    for _ in range(6):
        maddpg.replay.add(
            obs_all=generator.normal(size=12).astype(np.float32),
            act_all=generator.uniform(-0.04, 0.04, size=6).astype(np.float32),
            rewards_vec=generator.normal(size=2).astype(np.float32),
            next_obs_all=generator.normal(size=12).astype(np.float32),
            done_any=False,
            impedance=np.array([0.0, 0.034, 0.0], dtype=np.float32),
            next_impedance=np.array([0.0, 0.03, 0.0], dtype=np.float32),
        )


class MaddpgHumanModeTest(unittest.TestCase):
    def test_fixed_mode_updates_robot_but_not_human_actor(self):
        params = make_params("fixed_impedance")
        maddpg = MADDPG(
            1, MockEnvironment(), params, SeedPlan(params["seed"]), device="cpu"
        )
        fill_replay(maddpg)
        human_before = maddpg.human_actor_checksum()
        robot_before = sum(
            p.detach().double().sum().item()
            for p in maddpg.agents["robot"].actor.parameters()
        )

        maddpg.update()
        stats = maddpg.update()

        human_after = maddpg.human_actor_checksum()
        robot_after = sum(
            p.detach().double().sum().item()
            for p in maddpg.agents["robot"].actor.parameters()
        )
        self.assertEqual(human_before, human_after)
        self.assertNotEqual(robot_before, robot_after)
        self.assertNotIn("human", stats["loss/actor"])
        self.assertIn("robot", stats["loss/actor"])
        self.assertTrue(
            all(parameter.grad is None for parameter in maddpg.agents["robot"].critic.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in maddpg.agents["robot"].critic.parameters())
        )

    def test_residual_mode_updates_human_actor(self):
        params = make_params("residual_impedance")
        maddpg = MADDPG(
            1, MockEnvironment(), params, SeedPlan(params["seed"]), device="cpu"
        )
        fill_replay(maddpg)
        human_before = maddpg.human_actor_checksum()

        maddpg.update()
        stats = maddpg.update()

        self.assertNotEqual(human_before, maddpg.human_actor_checksum())
        self.assertIn("human", stats["loss/actor"])
        self.assertIn("robot", stats["loss/actor"])

    def test_fixed_target_uses_normalized_impedance(self):
        params = make_params("fixed_impedance")
        maddpg = MADDPG(
            1, MockEnvironment(), params, SeedPlan(params["seed"]), device="cpu"
        )
        output = maddpg._compose_human_action_norm(
            policy_action_norm=np_to_tensor([[0.9, -0.9, 0.9]]),
            impedance_force=np_to_tensor([[0.0, 0.02, -0.04]]),
        )
        expected = np_to_tensor([[0.0, 0.5, -1.0]])
        self.assertTrue(output.equal(expected))

    def test_residual_composition_bounds_impedance_before_addition(self):
        params = make_params("residual_impedance")
        maddpg = MADDPG(
            1, MockEnvironment(), params, SeedPlan(params["seed"]), device="cpu"
        )
        output = maddpg._compose_human_action_norm(
            policy_action_norm=np_to_tensor([[-1.0, 0.0, 0.0]]),
            impedance_force=np_to_tensor([[0.08, 0.0, 0.0]]),
        )
        self.assertTrue(output.equal(np_to_tensor([[0.0, 0.0, 0.0]])))

    def test_environment_action_selection_does_not_build_gradients(self):
        params = make_params("residual_impedance")
        maddpg = MADDPG(
            1, MockEnvironment(), params, SeedPlan(params["seed"]), device="cpu"
        )
        observations = {
            "human": np_to_tensor([[0.0] * 6]).requires_grad_(True),
            "robot": np_to_tensor([[0.0] * 6]).requires_grad_(True),
        }

        actions, detail = maddpg.select_actions(
            observations, add_noise=True, noise_scale=0.2
        )

        for agent_id in maddpg.agent_ids:
            self.assertFalse(actions[agent_id].requires_grad)
            self.assertFalse(detail["mean_actions"][agent_id].requires_grad)
            self.assertFalse(detail["noise_actions"][agent_id].requires_grad)


def np_to_tensor(values):
    import torch

    return torch.tensor(values, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
