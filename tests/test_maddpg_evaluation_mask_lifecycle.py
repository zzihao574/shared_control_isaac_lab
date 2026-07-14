from __future__ import annotations

import unittest

from scripts.utils.training_helpers_maddpg import MilestoneEvaluator


class MaskTrackingEnvironment:
    def __init__(self):
        self.active_env = None
        self.reset_count = 0

    def set_evaluation_active_env(self, env_id: int) -> None:
        self.active_env = env_id

    def clear_evaluation_active_env(self) -> None:
        self.active_env = None

    def reset(self):
        self.reset_count += 1
        return {}, {}


class StubEvaluator(MilestoneEvaluator):
    def __init__(self, env, *, fail: bool = False):
        self.env = env
        self.fail = fail
        self.observed_active_env = None

    def _run_active_evaluation_episode(self, env, active_env: int):
        self.observed_active_env = env.active_env
        if self.fail:
            raise RuntimeError("evaluation failed")
        return 1.25, 1


class EvaluationMaskLifecycleTest(unittest.TestCase):
    def test_env0_mask_is_cleared_after_success(self):
        env = MaskTrackingEnvironment()
        evaluator = StubEvaluator(env)

        result = evaluator._run_single_evaluation_episode()

        self.assertEqual(result, (1.25, 1))
        self.assertEqual(evaluator.observed_active_env, 0)
        self.assertIsNone(env.active_env)
        self.assertEqual(env.reset_count, 1)

    def test_env0_mask_is_cleared_after_failure(self):
        env = MaskTrackingEnvironment()
        evaluator = StubEvaluator(env, fail=True)

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            evaluator._run_single_evaluation_episode()

        self.assertEqual(evaluator.observed_active_env, 0)
        self.assertIsNone(env.active_env)
        self.assertEqual(env.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
