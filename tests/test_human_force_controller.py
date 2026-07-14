from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_project.envs.human_force_controller import HumanForceController


def make_params(model_type: str) -> dict:
    return {
        "human_model_type": model_type,
        "trajectory": {
            "start_point": [0.14, -0.2, 0.03],
            "end_point": [0.14, 0.2, 0.03],
        },
        "human_impedance": {
            "kp": [0.8, 0.8, 0.8],
            "kd": [0.1, 0.1, 0.1],
            "lookahead_distance": 0.04,
            "reference_speed": 0.02,
        },
        "constraints": {"max_human_force": 0.04},
    }


class HumanForceControllerTest(unittest.TestCase):
    def setUp(self):
        self.start = torch.tensor([[0.14, -0.2, 0.03]], dtype=torch.float32)
        self.zero_velocity = torch.zeros(1, 3)

    def test_start_reference_and_force_match_pilot_value(self):
        controller = HumanForceController(make_params("fixed_impedance"), "cpu")
        result = controller.compose(
            torch.zeros(1, 3), self.start, self.zero_velocity
        )

        self.assertTrue(
            torch.allclose(
                result.reference_position,
                torch.tensor([[0.14, -0.16, 0.03]]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.reference_velocity,
                torch.tensor([[0.0, 0.02, 0.0]]),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(result.impedance[0, 1]), 0.034, places=6)
        self.assertTrue(torch.allclose(result.total, result.impedance))

    def test_reference_velocity_tapers_near_endpoint(self):
        controller = HumanForceController(make_params("fixed_impedance"), "cpu")
        position = torch.tensor([[0.14, 0.19, 0.03]])
        reference_position, reference_velocity = controller.compute_reference(position)

        self.assertTrue(
            torch.allclose(reference_position, torch.tensor([[0.14, 0.2, 0.03]]), atol=1e-6)
        )
        self.assertAlmostEqual(float(reference_velocity[0, 1]), 0.005, places=6)

    def test_projection_uses_the_reference_line(self):
        controller = HumanForceController(make_params("fixed_impedance"), "cpu")
        off_path = torch.tensor([[0.17, -0.10, 0.01]])
        reference_position, _ = controller.compute_reference(off_path)

        self.assertTrue(
            torch.allclose(
                reference_position,
                torch.tensor([[0.14, -0.06, 0.03]]),
                atol=1e-6,
            )
        )

    def test_learnable_mode_preserves_policy_force(self):
        controller = HumanForceController(make_params("learnable"), "cpu")
        policy = torch.tensor([[0.01, -0.02, 0.03]])
        result = controller.compose(policy, self.start, self.zero_velocity)
        self.assertTrue(torch.equal(result.total, policy))
        self.assertTrue(torch.count_nonzero(result.residual) == 0)

    def test_all_modes_share_one_final_human_force_limit(self):
        excessive_policy = torch.tensor([[0.08, -0.09, 0.10]])
        far_off_path = torch.tensor([[0.50, -0.20, 0.03]])

        learnable = HumanForceController(make_params("learnable"), "cpu").compose(
            excessive_policy, self.start, self.zero_velocity
        )
        self.assertTrue(torch.equal(learnable.policy, excessive_policy))
        self.assertTrue(torch.equal(learnable.total, excessive_policy.clamp(-0.04, 0.04)))

        fixed = HumanForceController(make_params("fixed_impedance"), "cpu").compose(
            torch.zeros_like(excessive_policy), far_off_path, self.zero_velocity
        )
        self.assertLess(float(fixed.impedance[0, 0]), -0.04)
        self.assertAlmostEqual(float(fixed.total[0, 0]), -0.04, places=6)

        residual = HumanForceController(make_params("residual_impedance"), "cpu").compose(
            excessive_policy, far_off_path, self.zero_velocity
        )
        self.assertTrue(torch.equal(residual.residual, excessive_policy))
        expected = (residual.impedance + excessive_policy).clamp(-0.04, 0.04)
        self.assertTrue(torch.equal(residual.total, expected))

    def test_residual_adds_per_axis_and_clamps_total(self):
        controller = HumanForceController(make_params("residual_impedance"), "cpu")
        positive = controller.compose(
            torch.tensor([[0.0, 0.02, 0.0]]), self.start, self.zero_velocity
        )
        self.assertAlmostEqual(float(positive.total[0, 1]), 0.04, places=6)

        reversing = controller.compose(
            torch.tensor([[0.0, -0.04, 0.0]]), self.start, self.zero_velocity
        )
        self.assertAlmostEqual(float(reversing.total[0, 1]), -0.006, places=6)

    def test_unknown_model_type_is_rejected(self):
        with self.assertRaises(ValueError):
            HumanForceController(make_params("unknown"), "cpu")


if __name__ == "__main__":
    unittest.main()
