from __future__ import annotations

import unittest

from scripts.utils.training_helpers_rmappo import build_rmappo_wandb_config


class RMAPPOWandbConfigTest(unittest.TestCase):
    def test_builder_keeps_complete_resolved_configuration(self):
        resolved = {
            "seed": 17,
            "human_model_type": "residual_impedance",
            "human_impedance": {
                "kp": [0.8, 0.8, 0.8],
                "kd": [0.1, 0.1, 0.1],
                "lookahead_distance": 0.04,
                "reference_speed": 0.02,
            },
            "constraints": {
                "max_human_force": 0.04,
                "max_robot_force": 0.03,
            },
            "algorithms": {"rmappo": {"gamma": 0.99, "hidden_size": 256}},
            "obs_scaling": {"factors": [1.0] * 6},
        }
        metadata = {
            "human/kp_x": 0.8,
            "human/max_force_per_axis": 0.04,
            "human/residual_can_override_impedance": True,
        }

        config = build_rmappo_wandb_config(
            resolved,
            {"seed": 17, "num_envs": 48, "max_global_steps": 200000},
            metadata,
            "abc123",
        )

        self.assertEqual(config["algorithms"], resolved["algorithms"])
        self.assertEqual(config["obs_scaling"], resolved["obs_scaling"])
        self.assertEqual(config["experiment/algorithm"], "rmappo")
        self.assertEqual(config["experiment/human_model_type"], "residual_impedance")
        self.assertEqual(config["experiment/seed"], 17)
        self.assertEqual(config["robot/max_force_per_axis"], 0.03)
        self.assertEqual(config["human/kp_x"], 0.8)
        self.assertTrue(config["human/residual_can_override_impedance"])
        self.assertEqual(config["git_commit"], "abc123")


if __name__ == "__main__":
    unittest.main()
