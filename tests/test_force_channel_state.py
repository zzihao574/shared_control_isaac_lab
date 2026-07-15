from __future__ import annotations

import unittest
import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_project.envs.human_force_controller import HumanForceResult


def load_force_channel_state():
    """Load the pure torch utility without requiring an Isaac Sim process."""
    omni = types.ModuleType("omni")
    omni_physx = types.ModuleType("omni.physx")
    omni_physx.get_physx_attachment_private_interface = lambda: None
    omni_physx.get_physx_scene_query_interface = lambda: None
    omni.physx = omni_physx

    carb = types.ModuleType("carb")
    carb_internal = types.ModuleType("carb._carb")
    carb_internal.Float3 = object
    carb._carb = carb_internal

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_math.quat_apply_inverse = lambda quaternion, vector: vector
    isaaclab.utils = isaaclab_utils
    isaaclab_utils.math = isaaclab_math

    stubs = {
        "omni": omni,
        "omni.physx": omni_physx,
        "carb": carb,
        "carb._carb": carb_internal,
        "isaaclab": isaaclab,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_math,
    }
    previous_modules = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        module_path = (
            SRC_ROOT / "surgical_project" / "envs" / "multi_agent" / "utils.py"
        )
        spec = importlib.util.spec_from_file_location(
            "force_channel_utils_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.ForceChannelState
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


ForceChannelState = load_force_channel_state()


def make_human_result(num_envs: int) -> HumanForceResult:
    policy = torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3) / 100
    impedance = policy + 0.01
    residual = policy - 0.01
    total = torch.clamp(impedance + residual, -0.04, 0.04)
    zeros = torch.zeros_like(policy)
    return HumanForceResult(policy, impedance, residual, total, zeros, zeros)


class ForceChannelStateTest(unittest.TestCase):
    def test_update_combines_forces_and_returns_defensive_clones(self):
        state = ForceChannelState(3, "cpu")
        human_result = make_human_result(3)
        robot = torch.full((3, 3), 0.005)

        state.update(human_result, robot)
        breakdown = state.get_breakdown()

        self.assertTrue(
            torch.equal(breakdown["combined"], breakdown["human"] + breakdown["robot"])
        )
        breakdown["human"].zero_()
        self.assertFalse(torch.equal(breakdown["human"], state.get_breakdown()["human"]))

    def test_evaluation_mask_is_applied_after_human_composition(self):
        state = ForceChannelState(3, "cpu")
        state.set_evaluation_active_env(1)
        state.update(make_human_result(3), torch.full((3, 3), 0.005))

        breakdown = state.get_breakdown()
        for channel in ForceChannelState.CHANNEL_NAMES:
            self.assertTrue(torch.equal(breakdown[channel][0], torch.zeros(3)))
            self.assertTrue(torch.equal(breakdown[channel][2], torch.zeros(3)))
        self.assertEqual(state.evaluation_active_env_id, 1)

        state.clear_evaluation_active_env()
        self.assertIsNone(state.evaluation_active_env_id)
        self.assertIsNone(state.evaluation_mask)

    def test_reset_live_preserves_latest_transition_snapshot(self):
        state = ForceChannelState(3, "cpu")
        state.update(make_human_result(3), torch.full((3, 3), 0.005))
        before = state.get_breakdown()

        state.reset_live(torch.tensor([0, 2]))

        self.assertTrue(torch.equal(state.human[0], torch.zeros(3)))
        self.assertTrue(torch.equal(state.robot[2], torch.zeros(3)))
        after = state.get_breakdown()
        for channel in ForceChannelState.CHANNEL_NAMES:
            self.assertTrue(torch.equal(before[channel], after[channel]))

    def test_observations_include_previous_opponent_actual_force(self):
        state = ForceChannelState(3, "cpu")
        human_result = make_human_result(3)
        robot = torch.tensor(
            [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006], [0.007, 0.008, 0.009]]
        )
        base_obs = torch.arange(18, dtype=torch.float32).reshape(3, 6)

        initial = state.augment_agent_observations(base_obs)
        self.assertTrue(torch.equal(initial["human"][:, 6:], torch.zeros(3, 3)))
        self.assertTrue(torch.equal(initial["robot"][:, 6:], torch.zeros(3, 3)))

        state.update(human_result, robot)
        observations = state.augment_agent_observations(base_obs)
        self.assertTrue(torch.equal(observations["human"][:, :6], base_obs))
        self.assertTrue(torch.equal(observations["robot"][:, :6], base_obs))
        self.assertTrue(torch.equal(observations["human"][:, 6:], state.robot))
        self.assertTrue(torch.equal(observations["robot"][:, 6:], state.human))

        state.reset_live(torch.tensor([1]))
        reset_observations = state.augment_agent_observations(base_obs)
        self.assertTrue(torch.equal(reset_observations["human"][1, 6:], torch.zeros(3)))
        self.assertTrue(torch.equal(reset_observations["robot"][1, 6:], torch.zeros(3)))
        self.assertTrue(torch.equal(reset_observations["human"][0, 6:], robot[0]))


if __name__ == "__main__":
    unittest.main()
