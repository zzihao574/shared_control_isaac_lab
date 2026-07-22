from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_project.algorithms.marl.epigraph.epigraph_core import ActorRNN
from surgical_project.algorithms.marl.epigraph.rollout_buffer_z import RolloutBufferZ
from surgical_project.algorithms.marl.epigraph.trainer import (
    EpigraphTrainer,
    init_z_global,
    update_z_epigraph,
)
from surgical_project.algorithms.marl.epigraph.utils import (
    compute_dec_efocp_gae_dp,
    normalize_advantages,
)
from scripts.utils.training_helpers_epigraph import EpigraphSeedPlan, TrainingConfiguration


_ENV_UTILS_PATH = SRC_ROOT / "surgical_project/envs/multi_agent_epigraph/utils.py"
_ENV_UTILS_SPEC = importlib.util.spec_from_file_location(
    "epigraph_env_utils_for_test", _ENV_UTILS_PATH
)
_ENV_UTILS = importlib.util.module_from_spec(_ENV_UTILS_SPEC)
assert _ENV_UTILS_SPEC.loader is not None
_ENV_UTILS_SPEC.loader.exec_module(_ENV_UTILS)
compose_task_safe_from_rc = _ENV_UTILS.compose_task_safe_from_rc


class EpigraphAlignmentTest(unittest.TestCase):
    def test_reward_decomposition_respects_active_zone_masks(self):
        rc = {
            "zoneA_active_mask": torch.tensor([True, False, False, False]),
            "zoneB_active_mask": torch.tensor([False, True, False, False]),
            "zoneC_active_mask": torch.tensor([False, False, True, False]),
            "zoneD_active_mask": torch.tensor([False, False, False, True]),
            "zoneA_total_robot": torch.full((4,), 10.0),
            "zoneB_weight_robot": torch.tensor(2.0),
            "zoneB_gap_robot_contrib": torch.full((4,), 3.0),
            "zoneB_surftangent_robot_contrib": torch.full((4,), 4.0),
            "zoneB_inward_robot_contrib": torch.full((4,), -8.0),
            "zoneC_total_robot": torch.full((4,), -7.0),
            "zoneD_weight_robot": torch.tensor(4.0),
            "zoneD_progress_robot_contrib": torch.full((4,), 5.0),
            "zoneD_deviation_robot_contrib": torch.full((4,), 6.0),
            "zoneD_inward_robot_contrib": torch.full((4,), -9.0),
        }
        task, safety = compose_task_safe_from_rc(
            rc, agent="robot", device="cpu", num_envs=4
        )
        self.assertTrue(torch.equal(task, torch.tensor([10.0, 14.0, 0.0, 44.0])))
        self.assertTrue(torch.equal(safety, torch.tensor([0.0, -16.0, -7.0, -36.0])))

    def test_resolved_config_contract(self):
        config_path = REPO_ROOT / (
            "src/surgical_project/envs/multi_agent_epigraph/agents/"
            "training_params_epigraph.yaml"
        )
        with config_path.open("r") as config_file:
            params = yaml.safe_load(config_file)
        TrainingConfiguration(params)
        self.assertEqual(params["algorithms"]["epigraph"]["gamma"], 0.99)
        self.assertEqual(params["epigraph"]["z"]["min"], -450.0)
        self.assertEqual(params["epigraph"]["z"]["max"], 300.0)
        self.assertEqual(params["epigraph"]["z"]["encode"]["mean"], -75.0)
        self.assertEqual(params["epigraph"]["z"]["encode"]["scale"], 375.0)
        self.assertEqual(params["epigraph"]["z"]["init"]["p_min"], 0.30)
        self.assertEqual(params["epigraph"]["z"]["init"]["p_max"], 0.20)
        self.assertEqual(params["human_impedance"]["max_force"], 0.03)
        self.assertEqual(params["human_residual"]["max_force"], 0.015)
        self.assertEqual(len(params["obs_scaling"]["factors"]), 9)

    def test_seed_streams_are_reproducible_and_distinct(self):
        first = EpigraphSeedPlan(42)
        second = EpigraphSeedPlan(42)
        self.assertEqual(first.policy_seed(), second.policy_seed())
        self.assertNotEqual(first.policy_seed(), first.minibatch_seed())
        self.assertNotEqual(first.minibatch_seed(), first.z_seed())
        first_generator = first.make_generator(first.z_seed())
        second_generator = second.make_generator(second.z_seed())
        self.assertTrue(
            torch.equal(
                torch.rand(8, generator=first_generator),
                torch.rand(8, generator=second_generator),
            )
        )

    def test_z_transition_is_clamped_to_training_range(self):
        z = torch.tensor([[68.0]])
        z_next = update_z_epigraph(z, torch.tensor([[1.0]]), 0.99, -68.0, 68.0)
        self.assertEqual(z_next.item(), 68.0)
        z_next = update_z_epigraph(-z, torch.tensor([[-1.0]]), 0.99, -68.0, 68.0)
        self.assertEqual(z_next.item(), -68.0)

    def test_mixed_z_sampling_is_reproducible(self):
        first_generator = torch.Generator(device="cpu").manual_seed(123)
        second_generator = torch.Generator(device="cpu").manual_seed(123)
        first = init_z_global(
            128, -68.0, 68.0, "cpu", "mixed", 0.5, 0.5, first_generator
        )
        second = init_z_global(
            128, -68.0, 68.0, "cpu", "mixed", 0.5, 0.5, second_generator
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all((first == -68.0) | (first == 68.0)))
        self.assertGreater((first == -68.0).sum().item(), 40)
        self.assertGreater((first == 68.0).sum().item(), 40)

    def test_performance_q_branch_subtracts_z(self):
        q_perf, q_safe, advantages = compute_dec_efocp_gae_dp(
            rewards=torch.zeros(1, 1),
            costs=torch.zeros(1, 1, 1),
            z_traj=torch.tensor([[-2.0]]),
            vl_preds=torch.zeros(2, 1),
            vh_preds=torch.zeros(2, 1, 1),
            continuation_masks=torch.zeros(1, 1),
            bootstrap_masks=torch.zeros(1, 1),
            ov_mask_vl=torch.zeros(1, 1, dtype=torch.bool),
            ov_vl=torch.zeros(1, 1),
            ov_mask_vh=torch.zeros(1, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(1, 1, 1),
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertEqual(q_perf.item(), 0.0)
        self.assertEqual(q_safe.item(), 0.0)
        self.assertAlmostEqual(advantages.item(), 0.0, places=6)

    def test_dec_efocp_lambda_weights_each_horizon_after_max(self):
        q_perf, q_safe, advantages = compute_dec_efocp_gae_dp(
            rewards=torch.tensor([[0.0], [2.0]]),
            costs=torch.tensor([[[-1.0]], [[1.0]]]),
            z_traj=torch.tensor([[0.0], [0.0]]),
            vl_preds=torch.tensor([[0.0], [2.0], [0.0]]),
            vh_preds=torch.tensor([[[0.0]], [[-1.0]], [[0.0]]]),
            continuation_masks=torch.ones(2, 1),
            bootstrap_masks=torch.ones(2, 1),
            ov_mask_vl=torch.zeros(2, 1, dtype=torch.bool),
            ov_vl=torch.zeros(2, 1),
            ov_mask_vh=torch.zeros(2, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(2, 1, 1),
            gamma=0.5,
            gae_lambda=0.5,
        )
        # At t=0 the two horizons select different max branches. Weighting
        # each mixed horizon gives Q=0.5; max(weighted Qh, weighted Ql)=0.
        self.assertAlmostEqual(q_safe[0, 0, 0].item(), -0.5, places=6)
        self.assertAlmostEqual(q_perf[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(advantages[0, 0, 0].item(), 0.5, places=6)

    def test_dec_efocp_terminal_splits_the_dp_segment(self):
        _, q_safe, _ = compute_dec_efocp_gae_dp(
            rewards=torch.zeros(2, 1),
            costs=torch.tensor([[[-1.0]], [[1.0]]]),
            z_traj=torch.zeros(2, 1),
            vl_preds=torch.zeros(3, 1),
            vh_preds=torch.zeros(3, 1, 1),
            continuation_masks=torch.tensor([[0.0], [1.0]]),
            bootstrap_masks=torch.tensor([[0.0], [1.0]]),
            ov_mask_vl=torch.zeros(2, 1, dtype=torch.bool),
            ov_vl=torch.zeros(2, 1),
            ov_mask_vh=torch.zeros(2, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(2, 1, 1),
            gamma=0.5,
            gae_lambda=0.5,
        )
        self.assertAlmostEqual(q_safe[0, 0, 0].item(), -1.0, places=6)
        self.assertAlmostEqual(q_safe[1, 0, 0].item(), 1.0, places=6)

    def test_dec_efocp_safe_terminal_preserves_negative_constraint_value(self):
        _, q_safe, _ = compute_dec_efocp_gae_dp(
            rewards=torch.zeros(4, 1),
            costs=-torch.ones(4, 1, 1),
            z_traj=torch.zeros(4, 1),
            vl_preds=torch.zeros(5, 1),
            vh_preds=-torch.ones(5, 1, 1),
            continuation_masks=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
            bootstrap_masks=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
            ov_mask_vl=torch.zeros(4, 1, dtype=torch.bool),
            ov_vl=torch.zeros(4, 1),
            ov_mask_vh=torch.zeros(4, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(4, 1, 1),
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertTrue(torch.allclose(q_safe, -torch.ones_like(q_safe)))

    def test_dec_efocp_unsafe_terminal_preserves_positive_constraint_value(self):
        _, q_safe, _ = compute_dec_efocp_gae_dp(
            rewards=torch.zeros(1, 1),
            costs=torch.tensor([[[0.75]]]),
            z_traj=torch.zeros(1, 1),
            vl_preds=torch.zeros(2, 1),
            vh_preds=-torch.ones(2, 1, 1),
            continuation_masks=torch.zeros(1, 1),
            bootstrap_masks=torch.zeros(1, 1),
            ov_mask_vl=torch.zeros(1, 1, dtype=torch.bool),
            ov_vl=torch.zeros(1, 1),
            ov_mask_vh=torch.zeros(1, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(1, 1, 1),
            gamma=1.0,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(q_safe.item(), 0.75, places=6)

    def test_dec_efocp_rollout_boundary_still_bootstraps_vh(self):
        _, q_safe, _ = compute_dec_efocp_gae_dp(
            rewards=torch.zeros(1, 1),
            costs=-torch.ones(1, 1, 1),
            z_traj=torch.zeros(1, 1),
            vl_preds=torch.zeros(2, 1),
            vh_preds=torch.tensor([[[-1.0]], [[-0.25]]]),
            continuation_masks=torch.ones(1, 1),
            bootstrap_masks=torch.ones(1, 1),
            ov_mask_vl=torch.zeros(1, 1, dtype=torch.bool),
            ov_vl=torch.zeros(1, 1),
            ov_mask_vh=torch.zeros(1, 1, 1, dtype=torch.bool),
            ov_vh=torch.zeros(1, 1, 1),
            gamma=1.0,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(q_safe.item(), -0.25, places=6)

    def test_advantage_normalization_includes_terminal_transitions(self):
        advantages = torch.tensor([[[1.0]], [[3.0]]])
        normalized = normalize_advantages(advantages)
        self.assertTrue(torch.allclose(normalized.flatten(), torch.tensor([-1.0, 1.0])))

    def test_advantage_normalization_is_per_env_and_agent_over_time(self):
        advantages = torch.tensor(
            [
                [[1.0, 10.0], [100.0, 1000.0]],
                [[3.0, 14.0], [104.0, 1008.0]],
            ]
        )
        normalized = normalize_advantages(advantages)
        self.assertTrue(torch.allclose(normalized.mean(dim=0), torch.zeros(2, 2)))
        self.assertTrue(torch.allclose(normalized.std(dim=0, unbiased=False), torch.ones(2, 2)))

    def test_actor_logstd_contract(self):
        actor = ActorRNN(
            obs_dim=9,
            act_dim=3,
            hidden_size=8,
            nz=2,
            recurrent_N=1,
            use_orthogonal=False,
        )
        self.assertTrue(torch.all(actor.log_std == -0.5))
        actor.log_std.data.fill_(2.0)
        dist = actor._dist_from_latent(torch.zeros(4, 8))
        self.assertTrue(torch.all(dist.log_std == -0.2))

    def test_rollout_and_update_logprob_match_from_same_pre_state(self):
        actor = ActorRNN(
            obs_dim=9,
            act_dim=3,
            hidden_size=8,
            nz=2,
            recurrent_N=1,
            use_orthogonal=False,
        )
        obs = torch.randn(5, 9)
        z_enc = torch.randn(5, 2)
        pre_state = torch.zeros(5, 8)
        masks = torch.ones(5, 1)
        generator = torch.Generator(device="cpu").manual_seed(99)
        action, old_logp, _, _ = actor.act_step(
            obs, z_enc, pre_state, masks, generator=generator
        )
        new_logp, _, _ = actor.evaluate_actions_seq(
            obs, z_enc, pre_state, masks, action
        )
        ratio = torch.exp(new_logp - old_logp)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6))

    def test_recurrent_generator_is_time_major_and_covers_chunks_once(self):
        trainer = EpigraphTrainer.__new__(EpigraphTrainer)
        trainer.rollout_horizon = 4
        trainer.num_envs = 2
        trainer.num_agents = 2
        trainer.num_mini_batch = 3
        trainer.device = torch.device("cpu")
        trainer._minibatch_generator = torch.Generator(device="cpu").manual_seed(7)
        trainer.buffer = RolloutBufferZ(
            T=4,
            N=4,
            obs_dim=1,
            share_obs_dim=1,
            act_dim=1,
            rnn_hidden_dim=1,
            device="cpu",
        )
        for t in range(4):
            for slot in range(4):
                value = 100.0 * t + slot
                trainer.buffer.obs[t, slot, 0] = value
                trainer.buffer.share_obs[t, slot, 0] = value
                trainer.buffer.rnn_states_actor[t, slot, 0] = value

        initial_states = []
        for batch in trainer._recurrent_generator(chunk_length=2):
            batch_size = batch["h0_actor"].shape[0]
            obs = batch["obs_flat"].view(2, batch_size)
            self.assertTrue(torch.allclose(obs[1] - obs[0], torch.full((batch_size,), 100.0)))
            initial_states.extend(batch["h0_actor"].flatten().tolist())

        self.assertEqual(
            sorted(initial_states),
            sorted([0.0, 1.0, 2.0, 3.0, 200.0, 201.0, 202.0, 203.0]),
        )

    def test_fixed_impedance_excludes_human_actor_from_optimizer(self):
        trainer = EpigraphTrainer.__new__(EpigraphTrainer)
        trainer.algo_cfg = {
            "actor_lr": 1e-3,
            "vl_lr": 1e-3,
            "vh_lr": 1e-3,
            "opt_eps": 1e-5,
        }
        trainer.agent_ids = ["human", "robot"]
        trainer.trainable_agent_ids = ["robot"]
        trainer.z_encoder_actor = torch.nn.Linear(1, 1)
        trainer.z_encoder_vl = torch.nn.Linear(1, 1)
        trainer.z_encoder_vh = torch.nn.Linear(1, 1)
        trainer.actors = {
            "human": torch.nn.Linear(1, 1),
            "robot": torch.nn.Linear(1, 1),
        }
        trainer.critic_vl = torch.nn.Linear(1, 1)
        trainer.critics_vh = {
            "human": torch.nn.Linear(1, 1),
            "robot": torch.nn.Linear(1, 1),
        }
        trainer._build_optimizers()
        optimizer_ids = {
            id(parameter)
            for group in trainer.optimizer_actor.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            all(id(parameter) not in optimizer_ids for parameter in trainer.actors["human"].parameters())
        )
        self.assertTrue(
            all(id(parameter) in optimizer_ids for parameter in trainer.actors["robot"].parameters())
        )

    def test_wandb_payload_is_intentionally_compact(self):
        class CaptureLogger:
            enabled = True

            def log_rollout(self, step, stats):
                self.rollout = (step, stats)

            def log_update(self, step, stats):
                self.update = (step, stats)

        trainer = EpigraphTrainer.__new__(EpigraphTrainer)
        trainer.wandb_logger = CaptureLogger()
        trainer.trainable_agent_ids = ["human", "robot"]
        trainer._log_rollout_stats(
            10,
            {
                "avg_episode_task_return": 2.0,
                "avg_episode_safe_cost_sum": 0.5,
                "constraint_violation_ratio": 0.1,
                "constraint_h_mean": -0.4,
                "avg_progress_ratio": 0.7,
                "z_mean": -3.0,
                "forces/human_fx_mean": 99.0,
            },
        )
        self.assertEqual(
            set(trainer.wandb_logger.rollout[1]),
            {
                "task_return",
                "constraint_violation_ratio",
                "constraint_h_mean",
                "progress",
                "z_mean",
            },
        )

        trainer._log_update_stats(
            11,
            {
                "loss_policy": 1.0,
                "loss_value_vl": 2.0,
                "loss_value_vh": 3.0,
                "loss_value_vh/human": 3.1,
                "loss_value_vh/robot": 3.2,
                "vh_pred_mean": -0.3,
                "vh_target_mean": -0.4,
                "vh_target_negative_ratio": 0.8,
                "entropy": 4.0,
                "approx_kl": 5.0,
                "grad_actor": 6.0,
                "grad_vl": 7.0,
                "grad_vh": 8.0,
                "policy/human/grad_norm": 0.6,
                "policy/robot/grad_norm": 0.7,
                "policy/human/logstd_effective_mean": -0.4,
                "policy/robot/logstd_effective_mean": -0.3,
                "policy/human/loss": 99.0,
            },
        )
        self.assertEqual(
            set(trainer.wandb_logger.update[1]),
            {
                "policy_loss",
                "vl_loss",
                "vh_pred_mean",
                "vh_target_mean",
                "vh_target_negative_ratio",
                "entropy",
                "approx_kl",
                "vl_grad",
                "vh_grad",
                "vh_loss/human",
                "vh_loss/robot",
                "actor_grad/human",
                "actor_grad/robot",
                "logstd/human",
                "logstd/robot",
            },
        )



if __name__ == "__main__":
    unittest.main()
