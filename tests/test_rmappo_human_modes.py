from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


if "isaaclab.app" not in sys.modules:
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    isaaclab_app_module.AppLauncher = AppLauncher
    isaaclab_module.app = isaaclab_app_module
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module

from scripts.train_rmappo import DualRMAPPOWrapper
from scripts.utils.training_helpers_rmappo import RMAPPOSeedPlan


CONFIG_PATH = REPO_ROOT / (
    "src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml"
)


def make_params(model_type: str) -> dict:
    return {
        "seed": 7,
        "human_model_type": model_type,
        "constraints": {"max_human_force": 0.04, "max_robot_force": 0.04},
        "force_scaling": {"human_factor": 25.0, "robot_factor": 25.0},
        "obs_scaling": {"factors": [1.0] * 6 + [25.0] * 3},
        "training": {"seed": 7, "lr_decay": {"enabled": False}},
        "algorithms": {
            "rmappo": {
                "rollout_horizon": 2,
                "data_chunk_length": 1,
                "actor_lr": 1e-3,
                "critic_lr": 1e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_param": 0.2,
                "ppo_epoch": 1,
                "num_mini_batch": 1,
                "entropy_coef": 0.01,
                "max_grad_norm_actor": 5.0,
                "max_grad_norm_critic": 10.0,
                "opt_eps": 1e-5,
                "weight_decay": 0.0,
                "hidden_size": 16,
                "recurrent_N": 1,
                "use_orthogonal": False,
                "gain": 0.01,
                "use_popart": False,
                "use_clipped_value_loss": True,
                "huber_delta": 1.0,
                "max_global_steps": 10,
            }
        },
    }


def make_wrapper(model_type: str) -> DualRMAPPOWrapper:
    params = make_params(model_type)
    common = params["algorithms"]["rmappo"]
    RMAPPOSeedPlan(params["seed"]).apply_network_seed()
    wrapper = DualRMAPPOWrapper(
        {"human": dict(common), "robot": dict(common), "common": common},
        "cpu",
        2,
        9,
        18,
        3,
        params,
        None,
        SimpleNamespace(config=str(CONFIG_PATH)),
    )
    wrapper.train_generator = RMAPPOSeedPlan(params["seed"]).make_minibatch_generator()
    return wrapper


def collect_rollout(wrapper: DualRMAPPOWrapper) -> None:
    generator = torch.Generator().manual_seed(123)
    obs = {
        "human": torch.randn(2, 9, generator=generator),
        "robot": torch.randn(2, 9, generator=generator),
    }
    for step in range(wrapper.T):
        actions, _ = wrapper.select_actions(obs, deterministic=False)
        next_obs = {
            "human": torch.randn(2, 9, generator=generator),
            "robot": torch.randn(2, 9, generator=generator),
        }
        rewards = {
            "human": torch.tensor([0.2 + step, -0.3 + step]),
            "robot": torch.tensor([-0.4 + step, 0.7 + step]),
        }
        false = torch.zeros(2, dtype=torch.bool)
        dones = {"human": false.clone(), "robot": false.clone()}
        wrapper.add_experience_to_buffer(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            terminated=dones,
            truncated=dones,
        )
        obs = next_obs
    wrapper.store_next_obs(obs)


class RMAPPOHumanModeTest(unittest.TestCase):
    def test_fixed_mode_uses_zero_policy_placeholder(self):
        wrapper = make_wrapper("fixed_impedance")
        obs = {"human": torch.ones(2, 9), "robot": torch.ones(2, 9)}

        actions, _ = wrapper.select_actions(obs, deterministic=False)

        self.assertEqual(wrapper.trainable_agent_ids, ["robot"])
        self.assertTrue(torch.equal(actions["human"], torch.zeros(2, 3)))
        self.assertTrue(
            torch.equal(
                wrapper._current_step_data["human"]["actions"],
                torch.zeros(2, 3),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper._current_step_data["human"]["action_log_probs"],
                torch.zeros(2, 1),
            )
        )

    def test_residual_buffer_stores_policy_residual_not_composed_force(self):
        wrapper = make_wrapper("residual_impedance")
        obs = {"human": torch.ones(2, 9), "robot": torch.ones(2, 9)}

        actions, _ = wrapper.select_actions(obs, deterministic=False)
        residual_norm = wrapper._current_step_data["human"]["actions"]

        self.assertEqual(wrapper.trainable_agent_ids, ["human", "robot"])
        self.assertTrue(torch.allclose(actions["human"], residual_norm * 0.04))
        self.assertTrue(
            torch.isfinite(
                wrapper._current_step_data["human"]["action_log_probs"]
            ).all()
        )

    def test_previous_opponent_physical_force_is_scaled_in_observation(self):
        wrapper = make_wrapper("residual_impedance")
        obs = {
            "human": torch.zeros(2, 9),
            "robot": torch.zeros(2, 9),
        }
        obs["human"][:, 6:] = torch.tensor([0.04, -0.02, 0.0])
        obs["robot"][:, 6:] = torch.tensor([-0.04, 0.02, 0.0])

        wrapper.select_actions(obs, deterministic=False)

        expected_human = torch.tensor([1.0, -0.5, 0.0]).expand(2, -1)
        expected_robot = torch.tensor([-1.0, 0.5, 0.0]).expand(2, -1)
        self.assertTrue(
            torch.equal(wrapper._current_step_data["human"]["obs"][:, 6:], expected_human)
        )
        self.assertTrue(
            torch.equal(wrapper._current_step_data["robot"]["obs"][:, 6:], expected_robot)
        )

    def test_fixed_updates_robot_only_and_residual_updates_both(self):
        fixed = make_wrapper("fixed_impedance")
        fixed_human_before = fixed.human_actor_checksum()
        fixed_robot_before = [p.detach().clone() for p in fixed.policies["robot"].actor.parameters()]
        collect_rollout(fixed)
        fixed_stats = fixed.update()

        self.assertEqual(fixed_human_before, fixed.human_actor_checksum())
        self.assertNotIn("policy_loss/human", fixed_stats)
        self.assertIn("policy_loss/robot", fixed_stats)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    fixed_robot_before, fixed.policies["robot"].actor.parameters()
                )
            )
        )

        residual = make_wrapper("residual_impedance")
        residual_human_before = residual.human_actor_checksum()
        collect_rollout(residual)
        residual_stats = residual.update()

        self.assertNotEqual(residual_human_before, residual.human_actor_checksum())
        self.assertIn("policy_loss/human", residual_stats)
        self.assertIn("policy_loss/robot", residual_stats)


if __name__ == "__main__":
    unittest.main()
