from __future__ import annotations

import unittest

from scripts.utils.training_helpers_maddpg import build_maddpg_wandb_config


class MaddpgWandbConfigTest(unittest.TestCase):
    def test_builder_uses_resolved_metadata_without_false_hardcoded_fields(self):
        resolved = {
            "seed": 17,
            "human_model_type": "fixed_impedance",
            "constraints": {
                "max_human_force": 0.04,
                "max_robot_force": 0.03,
            },
            "force_scaling": {"human_factor": 25.0, "robot_factor": 100.0 / 3.0},
            "termination_conditions": {
                "z_below_zero": False,
                "edge_collision": False,
                "safety_distance_threshold": 0.002,
            },
            "maddpg_config": {"gamma": 0.99, "batch_size": 1024},
            "networks": {"actor": {"hidden_layers": [128, 128]}},
        }
        human_metadata = {
            "human/kp_x": 0.8,
            "human/max_force_per_axis": 0.04,
        }

        config = build_maddpg_wandb_config(
            resolved,
            {"seed": 17, "num_envs": 48, "max_global_steps": 200000},
            human_metadata,
            "abc123",
        )

        self.assertEqual(config["maddpg_config"], resolved["maddpg_config"])
        self.assertEqual(config["networks"], resolved["networks"])
        self.assertEqual(config["experiment/seed"], 17)
        self.assertEqual(config["robot/max_force_per_axis"], 0.03)
        self.assertFalse(config["termination/edge_collision"])
        self.assertEqual(config["termination/safety_distance_threshold"], 0.002)
        self.assertEqual(config["human/kp_x"], 0.8)
        self.assertEqual(config["human/force_input_factor"], 25.0)
        self.assertEqual(config["robot/force_input_factor"], 100.0 / 3.0)
        self.assertEqual(config["observation/total_dim"], 9)
        self.assertNotIn("reward_scale", config)
        self.assertNotIn("termination_mode", config)


if __name__ == "__main__":
    unittest.main()
