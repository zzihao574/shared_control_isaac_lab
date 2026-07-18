from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from scripts.utils.training_helpers_maddpg import SeedPlan, TrainingRunner


class MaddpgReproducibilityTest(unittest.TestCase):
    def test_seed_derivation_is_reproducible_and_distinct(self):
        first = SeedPlan(42)
        second = SeedPlan(42)
        other = SeedPlan(43)

        self.assertEqual(first.network_seed(), second.network_seed())
        self.assertEqual(first.replay_seed(), second.replay_seed())
        self.assertEqual(
            first.exploration_seed(1, 7), second.exploration_seed(1, 7)
        )
        self.assertNotEqual(first.replay_seed(), other.replay_seed())
        self.assertNotEqual(
            first.exploration_seed(0, 0), first.exploration_seed(0, 1)
        )
        self.assertNotEqual(
            first.exploration_seed(0, 0), first.exploration_seed(1, 0)
        )

    def test_same_seed_reproduces_exploration_noise(self):
        plan = SeedPlan(123)
        first = plan.make_exploration_generator("cpu", 1, 5)
        second = plan.make_exploration_generator("cpu", 1, 5)
        different_env = plan.make_exploration_generator("cpu", 1, 6)

        first_noise = torch.randn(32, generator=first)
        second_noise = torch.randn(32, generator=second)
        different_noise = torch.randn(32, generator=different_env)

        self.assertTrue(torch.equal(first_noise, second_noise))
        self.assertFalse(torch.equal(first_noise, different_noise))

    def test_network_seed_controls_initialization(self):
        SeedPlan(9).apply_network_seed()
        first = torch.nn.Linear(6, 3).state_dict()
        SeedPlan(9).apply_network_seed()
        second = torch.nn.Linear(6, 3).state_dict()
        SeedPlan(10).apply_network_seed()
        different = torch.nn.Linear(6, 3).state_dict()

        self.assertTrue(
            all(torch.equal(first[name], second[name]) for name in first)
        )
        self.assertTrue(
            any(not torch.equal(first[name], different[name]) for name in first)
        )

    def test_noise_schedule_holds_during_warmup_and_reaches_exact_end(self):
        runner = TrainingRunner.__new__(TrainingRunner)
        runner.sigma_start = 0.28
        runner.sigma_end = 0.05
        runner.decay_k = 4.0
        runner.max_global_steps = 7200
        runner.maddpg = SimpleNamespace(min_buffer_size=18000, num_envs=3)

        runner.global_step = 6000
        self.assertEqual(runner._calculate_noise_scale(), 0.28)
        runner.global_step = 6600
        self.assertGreater(runner._calculate_noise_scale(), 0.05)
        self.assertLess(runner._calculate_noise_scale(), 0.28)
        runner.global_step = 7200
        self.assertAlmostEqual(runner._calculate_noise_scale(), 0.05)


if __name__ == "__main__":
    unittest.main()
