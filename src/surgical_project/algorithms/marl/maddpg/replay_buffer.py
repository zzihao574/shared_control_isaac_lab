"""
Joint replay buffer implementation for shared network MADDPG.
Stores concatenated multi-agent experiences for centralized training.

Features:
- Joint experience storage with concatenated observations/actions
- Efficient numpy-based storage with torch tensor sampling
- Circular buffer implementation for memory efficiency
- Per-agent reward tracking with logical OR done signals
- FIXED: Support for dedicated generator to avoid global RNG interference
"""
import numpy as np
import torch
from typing import Tuple, Optional


class JointReplayBuffer:
    """
    Joint replay buffer for shared network MADDPG.
    
    Storage format:
    - obs_all: [capacity, total_obs_dim] - concatenated observations from all agents
    - actions_all: [capacity, total_action_dim] - concatenated actions from all agents  
    - rewards_all: [capacity, num_agents] - per-agent rewards
    - next_obs_all: [capacity, total_obs_dim] - concatenated next observations
    - done_any: [capacity, 1] - logical OR of agent done signals
    - impedance / next_impedance: [capacity, 3] - analytic human prior
    """
    
    def __init__(self, capacity: int, total_obs_dim: int, total_action_dim: int, num_agents: int, device: torch.device):
        self.capacity = int(capacity)  # Buffer capacity
        self.total_obs_dim = int(total_obs_dim)  # Total observation dimension
        self.total_action_dim = int(total_action_dim)  # Total action dimension
        self.num_agents = int(num_agents)  # Number of agents
        self.device = device  # PyTorch device
        
        # Circular buffer state
        self.ptr = 0  # Current write pointer
        self.size = 0  # Current buffer size
        
        # Storage arrays (numpy for memory efficiency)
        self.obs = np.zeros((capacity, total_obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, total_action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, num_agents), dtype=np.float32)
        self.nobs = np.zeros((capacity, total_obs_dim), dtype=np.float32)
        self.done_any = np.zeros((capacity, 1), dtype=np.float32)
        self.impedance = np.zeros((capacity, 3), dtype=np.float32)
        self.next_impedance = np.zeros((capacity, 3), dtype=np.float32)
        
        print(f"[JOINT BUFFER] Initialized: capacity={capacity}, obs_dim={total_obs_dim}, act_dim={total_action_dim}")

    def add(self, obs_all: np.ndarray, act_all: np.ndarray, rewards_vec: np.ndarray, 
            next_obs_all: np.ndarray, done_any: bool,
            impedance: Optional[np.ndarray] = None,
            next_impedance: Optional[np.ndarray] = None) -> None:
        """Add one joint experience through the shared batch writer."""
        self.add_batch(
            obs_all=np.asarray(obs_all, dtype=np.float32)[None, ...],
            act_all=np.asarray(act_all, dtype=np.float32)[None, ...],
            rewards_all=np.asarray(rewards_vec, dtype=np.float32)[None, ...],
            next_obs_all=np.asarray(next_obs_all, dtype=np.float32)[None, ...],
            done_all=np.asarray([done_any], dtype=np.float32),
            impedance=(
                None
                if impedance is None
                else np.asarray(impedance, dtype=np.float32)[None, ...]
            ),
            next_impedance=(
                None
                if next_impedance is None
                else np.asarray(next_impedance, dtype=np.float32)[None, ...]
            ),
        )

    def add_batch(
        self,
        obs_all: np.ndarray,
        act_all: np.ndarray,
        rewards_all: np.ndarray,
        next_obs_all: np.ndarray,
        done_all: np.ndarray,
        impedance: Optional[np.ndarray] = None,
        next_impedance: Optional[np.ndarray] = None,
    ) -> None:
        """Write a vectorized environment batch into the circular buffer."""
        obs_all = np.asarray(obs_all, dtype=np.float32)
        act_all = np.asarray(act_all, dtype=np.float32)
        rewards_all = np.asarray(rewards_all, dtype=np.float32)
        next_obs_all = np.asarray(next_obs_all, dtype=np.float32)
        done_all = np.asarray(done_all, dtype=np.float32).reshape(-1, 1)

        batch_size = int(obs_all.shape[0])
        if batch_size == 0:
            return

        expected_shapes = {
            "obs_all": (batch_size, self.total_obs_dim),
            "act_all": (batch_size, self.total_action_dim),
            "rewards_all": (batch_size, self.num_agents),
            "next_obs_all": (batch_size, self.total_obs_dim),
            "done_all": (batch_size, 1),
        }
        arrays = {
            "obs_all": obs_all,
            "act_all": act_all,
            "rewards_all": rewards_all,
            "next_obs_all": next_obs_all,
            "done_all": done_all,
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"{name} shape {arrays[name].shape} does not match {expected}"
                )

        if impedance is not None:
            impedance = np.asarray(impedance, dtype=np.float32)
            if impedance.shape != (batch_size, 3):
                raise ValueError(
                    f"impedance shape {impedance.shape} does not match {(batch_size, 3)}"
                )
        if next_impedance is not None:
            next_impedance = np.asarray(next_impedance, dtype=np.float32)
            if next_impedance.shape != (batch_size, 3):
                raise ValueError(
                    "next_impedance shape "
                    f"{next_impedance.shape} does not match {(batch_size, 3)}"
                )

        # If one environment batch exceeds the full capacity, only its newest
        # transitions can survive the equivalent sequence of scalar writes.
        if batch_size >= self.capacity:
            keep = slice(batch_size - self.capacity, batch_size)
            retained = self.capacity
            start = (self.ptr + batch_size - self.capacity) % self.capacity
            obs_all = obs_all[keep]
            act_all = act_all[keep]
            rewards_all = rewards_all[keep]
            next_obs_all = next_obs_all[keep]
            done_all = done_all[keep]
            if impedance is not None:
                impedance = impedance[keep]
            if next_impedance is not None:
                next_impedance = next_impedance[keep]
        else:
            retained = batch_size
            start = self.ptr

        first = min(retained, self.capacity - start)
        second = retained - first

        def write(storage: np.ndarray, values: np.ndarray) -> None:
            storage[start : start + first] = values[:first]
            if second:
                storage[:second] = values[first:]

        write(self.obs, obs_all)
        write(self.act, act_all)
        write(self.rew, rewards_all)
        write(self.nobs, next_obs_all)
        write(self.done_any, done_all)
        if impedance is not None:
            write(self.impedance, impedance)
        if next_impedance is not None:
            write(self.next_impedance, next_impedance)

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample random batch from joint buffer using dedicated generator.
        
        Args:
            batch_size: Number of transitions to sample
            generator: Optional torch.Generator for reproducible sampling
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones,
            impedance, next_impedance) tensors or None if insufficient data
        """
        if self.size < batch_size:
            return None
        
        if generator is not None:
            # Use provided generator for reproducible sampling
            idx = torch.randint(high=self.size, size=(batch_size,), generator=generator).cpu().numpy()
        else:
            # Fallback to numpy random (for backward compatibility)
            idx = np.random.choice(self.size, batch_size, replace=True)
        
        obs = torch.from_numpy(self.obs[idx]).to(self.device)      # [B, total_obs_dim]
        act = torch.from_numpy(self.act[idx]).to(self.device)      # [B, total_action_dim]
        rew = torch.from_numpy(self.rew[idx]).to(self.device)      # [B, num_agents]
        nobs = torch.from_numpy(self.nobs[idx]).to(self.device)    # [B, total_obs_dim]
        done = torch.from_numpy(self.done_any[idx]).to(self.device) # [B, 1]
        impedance = torch.from_numpy(self.impedance[idx]).to(self.device)
        next_impedance = torch.from_numpy(self.next_impedance[idx]).to(self.device)
        
        return obs, act, rew, nobs, done, impedance, next_impedance

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def clear(self) -> None:
        """Clear buffer contents."""
        self.ptr = 0
        self.size = 0
        print("[JOINT BUFFER] Cleared")
