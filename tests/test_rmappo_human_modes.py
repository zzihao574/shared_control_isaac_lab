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
from surgical_project.algorithms.marl.rmappo.mappo_utils import TanhDiagGaussian
from surgical_project.algorithms.marl.rmappo.rollout_buffer import SharedRolloutBuffer
from scripts.utils.training_helpers_rmappo import RMAPPOSeedPlan


CONFIG_PATH = REPO_ROOT / (
    "src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml"
)


def make_params(model_type: str) -> dict:
    return {
        "seed": 7,
        "human_model_type": model_type,
        "human_impedance": {
            "kp": [0.6, 0.6, 0.6],
            "kd": [0.075, 0.075, 0.075],
            "max_force": 0.03,
        },
        "human_residual": {"max_force": 0.015},
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
    def test_rollout_stores_pre_action_rnn_state_and_ratio_starts_at_one(self):
        wrapper = make_wrapper("residual_impedance")
        collect_rollout(wrapper)
        buffer = wrapper.buffers["human"]
        buffer.compute_returns_and_adv(
            torch.zeros(2, 1), gamma=0.99, gae_lambda=0.95
        )
        sample = next(buffer.recurrent_generator(1, 2))

        length, batch = sample["obs"].shape[:2]
        with torch.no_grad():
            _, new_log_probs, _ = wrapper.policies["human"].evaluate_actions(
                sample["share_obs"].view(length * batch, -1),
                sample["obs"].view(length * batch, -1),
                sample["rnn_states_actor"],
                sample["rnn_states_critic"],
                sample["actions"].view(length * batch, -1),
                sample["rnn_masks"].view(length * batch, -1),
            )

        old_log_probs = sample["action_log_probs"].view(length * batch, -1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6))
        self.assertTrue(torch.equal(buffer.rnn_states_actor[0], torch.zeros_like(buffer.rnn_states_actor[0])))

    def test_rnn_mask_is_pre_step_while_continuation_mask_is_post_transition(self):
        wrapper = make_wrapper("residual_impedance")
        obs = {"human": torch.ones(2, 9), "robot": torch.ones(2, 9)}
        actions, _ = wrapper.select_actions(obs, deterministic=False)
        rewards = {"human": torch.zeros(2), "robot": torch.zeros(2)}
        dones = {
            "human": torch.tensor([True, False]),
            "robot": torch.tensor([True, False]),
        }
        wrapper.add_experience_to_buffer(obs, actions, rewards, obs, dones)

        next_actions, _ = wrapper.select_actions(obs, deterministic=False)
        no_dones = {
            "human": torch.zeros(2, dtype=torch.bool),
            "robot": torch.zeros(2, dtype=torch.bool),
        }
        wrapper.add_experience_to_buffer(
            obs, next_actions, rewards, obs, no_dones
        )

        buffer = wrapper.buffers["human"]
        self.assertEqual(buffer.rnn_masks[0, 0].item(), 1.0)
        self.assertEqual(buffer.continuation_masks[0, 0].item(), 0.0)
        self.assertEqual(buffer.rnn_masks[1, 0].item(), 0.0)
        self.assertTrue(torch.equal(buffer.rnn_states_actor[1, 0], torch.zeros_like(buffer.rnn_states_actor[1, 0])))

    def test_natural_timeout_does_not_bootstrap(self):
        buffer = SharedRolloutBuffer(1, 1, 2, 2, 1, 4, "cpu")
        zeros = torch.zeros(1, 1)
        buffer.insert(
            0,
            obs=torch.zeros(1, 2),
            share_obs=torch.zeros(1, 2),
            actions=zeros,
            action_log_probs=zeros,
            value_preds=zeros,
            rewards=torch.tensor([[2.0]]),
            rnn_masks=torch.ones(1, 1),
            continuation_masks=torch.zeros(1, 1),
            term_masks=torch.zeros(1, 1),
            rnn_states_actor=torch.zeros(1, 4),
            rnn_states_critic=torch.zeros(1, 4),
        )

        buffer.compute_returns_and_adv(
            last_values=torch.tensor([[100.0]]), gamma=0.99, gae_lambda=0.95
        )
        self.assertEqual(buffer.returns[0, 0, 0].item(), 2.0)

    def test_advantages_normalize_over_terminal_and_nonterminal_samples(self):
        buffer = SharedRolloutBuffer(2, 2, 1, 1, 1, 2, "cpu")
        for step, rewards in enumerate((torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [5.0]]))):
            buffer.insert(
                step,
                obs=torch.zeros(2, 1),
                share_obs=torch.zeros(2, 1),
                actions=torch.zeros(2, 1),
                action_log_probs=torch.zeros(2, 1),
                value_preds=torch.zeros(2, 1),
                rewards=rewards,
                rnn_masks=torch.ones(2, 1),
                continuation_masks=torch.tensor([[0.0], [1.0]]),
                term_masks=torch.tensor([[0.0], [1.0]]),
                rnn_states_actor=torch.zeros(2, 2),
                rnn_states_critic=torch.zeros(2, 2),
            )

        buffer.compute_returns_and_adv(torch.zeros(2, 1), 0.99, 0.95)
        self.assertAlmostEqual(buffer.advantages.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(
            buffer.advantages.std(unbiased=False).item(), 1.0, places=6
        )

    def test_tanh_log_prob_is_finite_at_saturation(self):
        wrapper = make_wrapper("residual_impedance")
        action_layer = wrapper.policies["human"].actor.act
        action_layer._dist.fc_mean.weight.data.zero_()
        action_layer._dist.fc_mean.bias.data.fill_(20.0)
        features = torch.zeros(8, wrapper.policies["human"].actor.hidden_size)

        actions, sampled_log_probs = action_layer(features, deterministic=False)
        evaluated_log_probs, _ = action_layer.evaluate_actions(features, actions)

        self.assertTrue((actions.abs() < 1.0).all())
        self.assertTrue(torch.isfinite(sampled_log_probs).all())
        self.assertTrue(torch.isfinite(evaluated_log_probs).all())
        self.assertTrue(torch.allclose(sampled_log_probs, evaluated_log_probs, atol=1e-6))

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
        self.assertTrue(torch.allclose(actions["human"], residual_norm * 0.015))
        self.assertTrue(
            torch.isfinite(
                wrapper._current_step_data["human"]["action_log_probs"]
            ).all()
        )

    def test_learnable_human_keeps_full_force_scale(self):
        wrapper = make_wrapper("learnable")
        actions_norm = {
            "human": torch.tensor([[1.0, -0.5, 0.0]]),
            "robot": torch.tensor([[1.0, -0.5, 0.0]]),
        }

        actions = wrapper.actions_to_env_format(actions_norm)

        self.assertTrue(
            torch.equal(actions["human"], actions_norm["human"] * 0.04)
        )
        self.assertTrue(
            torch.equal(actions["robot"], actions_norm["robot"] * 0.04)
        )

    def test_logstd_effective_range_is_minus_one_to_minus_point_two(self):
        distribution = TanhDiagGaussian(4, 3, use_orthogonal=False)
        inputs = torch.zeros(2, 4)

        distribution.logstd._bias.data.fill_(-4.0)
        minimum_std = distribution.base_dist(inputs).scale
        distribution.logstd._bias.data.fill_(4.0)
        maximum_std = distribution.base_dist(inputs).scale

        self.assertTrue(
            torch.allclose(
                minimum_std,
                torch.full_like(minimum_std, torch.exp(torch.tensor(-1.0))),
            )
        )
        self.assertTrue(
            torch.allclose(
                maximum_std,
                torch.full_like(maximum_std, torch.exp(torch.tensor(-0.2))),
            )
        )

    def test_tanh_gaussian_entropy_is_batch_size_invariant(self):
        wrapper = make_wrapper("residual_impedance")
        action_layer = wrapper.policies["human"].actor.act
        feature_dim = wrapper.policies["human"].actor.hidden_size

        _, entropy_one = action_layer.evaluate_actions(
            torch.zeros(1, feature_dim),
            torch.zeros(1, 3),
        )
        _, entropy_eight = action_layer.evaluate_actions(
            torch.zeros(8, feature_dim),
            torch.zeros(8, 3),
        )

        self.assertTrue(torch.allclose(entropy_one, entropy_eight))

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
