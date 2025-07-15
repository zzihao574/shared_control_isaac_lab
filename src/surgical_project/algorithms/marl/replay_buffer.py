# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch Replay Buffer for MADDPG"""

import numpy as np
import random
import torch
from typing import Tuple, List


class ReplayBuffer:
    """Experience replay buffer for MADDPG agents."""
    
    def __init__(self, size: int):
        """Create replay buffer.
        
        Args:
            size: Maximum number of transitions to store in the buffer.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
    
    def __len__(self):
        return len(self._storage)
    
    def clear(self):
        """Clear the buffer."""
        self._storage = []
        self._next_idx = 0
    
    def add(self, obs_t: np.ndarray, action: np.ndarray, reward: float, 
            obs_tp1: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        data = (obs_t, action, reward, obs_tp1, done)
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def _encode_sample(self, idxes: List[int]) -> Tuple[np.ndarray, ...]:
        """Encode a sample of transitions."""
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t), 
            np.array(actions), 
            np.array(rewards), 
            np.array(obses_tp1), 
            np.array(dones)
        )
    
    def make_index(self, batch_size: int) -> List[int]:
        """Create random indices for sampling."""
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    
    def sample_index(self, idxes: List[int]) -> Tuple[np.ndarray, ...]:
        """Sample transitions at given indices."""
        return self._encode_sample(idxes)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: How many transitions to sample.
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)