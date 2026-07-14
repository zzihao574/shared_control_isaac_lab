from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_project.algorithms.marl.maddpg.replay_buffer import JointReplayBuffer


def make_batch(start: int, count: int):
    values = np.arange(start, start + count, dtype=np.float32)
    return {
        "obs_all": np.repeat(values[:, None], 4, axis=1),
        "act_all": np.repeat((values + 10)[:, None], 2, axis=1),
        "rewards_all": np.repeat((values + 20)[:, None], 2, axis=1),
        "next_obs_all": np.repeat((values + 30)[:, None], 4, axis=1),
        "done_all": (values.astype(np.int64) % 2 == 0),
        "impedance": np.repeat((values + 40)[:, None], 3, axis=1),
        "next_impedance": np.repeat((values + 50)[:, None], 3, axis=1),
    }


class ReplayBufferBatchTest(unittest.TestCase):
    def make_buffer(self, capacity: int = 5):
        return JointReplayBuffer(capacity, 4, 2, 2, torch.device("cpu"))

    def add_scalar_batch(self, buffer, batch):
        for index in range(batch["obs_all"].shape[0]):
            buffer.add(
                batch["obs_all"][index],
                batch["act_all"][index],
                batch["rewards_all"][index],
                batch["next_obs_all"][index],
                bool(batch["done_all"][index]),
                batch["impedance"][index],
                batch["next_impedance"][index],
            )

    def assert_buffers_equal(self, first, second):
        self.assertEqual(first.ptr, second.ptr)
        self.assertEqual(first.size, second.size)
        for name in (
            "obs",
            "act",
            "rew",
            "nobs",
            "done_any",
            "impedance",
            "next_impedance",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_batch_write_matches_scalar_write_across_wraparound(self):
        scalar = self.make_buffer()
        batched = self.make_buffer()

        for batch in (make_batch(0, 3), make_batch(3, 4)):
            self.add_scalar_batch(scalar, batch)
            batched.add_batch(**batch)

        self.assert_buffers_equal(scalar, batched)

    def test_batch_larger_than_capacity_matches_scalar_write(self):
        scalar = self.make_buffer()
        batched = self.make_buffer()
        prefix = make_batch(0, 2)
        oversized = make_batch(2, 8)

        self.add_scalar_batch(scalar, prefix)
        batched.add_batch(**prefix)
        self.add_scalar_batch(scalar, oversized)
        batched.add_batch(**oversized)

        self.assert_buffers_equal(scalar, batched)


if __name__ == "__main__":
    unittest.main()
