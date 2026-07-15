from __future__ import annotations

import unittest

import torch

from scripts.utils.training_helpers_rmappo import RMAPPOSeedPlan


class RMAPPOReproducibilityTest(unittest.TestCase):
    def test_seed_derivation_is_reproducible_and_distinct(self):
        first = RMAPPOSeedPlan(42)
        second = RMAPPOSeedPlan(42)
        other = RMAPPOSeedPlan(43)

        self.assertEqual(first.network_seed(), second.network_seed())
        self.assertEqual(first.minibatch_seed(), second.minibatch_seed())
        self.assertNotEqual(first.network_seed(), other.network_seed())
        self.assertNotEqual(first.minibatch_seed(), other.minibatch_seed())

    def test_same_seed_reproduces_minibatch_stream(self):
        plan = RMAPPOSeedPlan(123)
        first = torch.randperm(64, generator=plan.make_minibatch_generator())
        second = torch.randperm(64, generator=plan.make_minibatch_generator())
        other = torch.randperm(
            64, generator=RMAPPOSeedPlan(124).make_minibatch_generator()
        )

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_network_seed_controls_initialization(self):
        RMAPPOSeedPlan(9).apply_network_seed()
        first = torch.nn.Linear(6, 3).state_dict()
        RMAPPOSeedPlan(9).apply_network_seed()
        second = torch.nn.Linear(6, 3).state_dict()
        RMAPPOSeedPlan(10).apply_network_seed()
        other = torch.nn.Linear(6, 3).state_dict()

        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        self.assertTrue(any(not torch.equal(first[k], other[k]) for k in first))


if __name__ == "__main__":
    unittest.main()
