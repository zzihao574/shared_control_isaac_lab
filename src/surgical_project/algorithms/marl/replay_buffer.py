"""
Replay buffer implementation for shared network MADDPG.
FINAL VERSION: Joint replay buffer with concatenated observations/actions.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional

class MultiAgentReplayBuffer:
    """
    Legacy multi-agent replay buffer (kept for compatibility).
    NOT USED in shared network architecture.
    """
    
    def __init__(self, capacity: int, num_agents: int, obs_dims: List[int], action_dims: List[int], device: torch.device):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.device = device
        
        # Circular buffer pointers and size
        self.ptr = 0
        self.size = 0
        
        # Initialize storage buffers
        self.obs_buffers = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.next_obs_buffers = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.action_buffers = [np.zeros((self.capacity, d), dtype=np.float32) for d in action_dims]
        self.reward_buffers = [np.zeros((self.capacity, 1), dtype=np.float32) for _ in range(num_agents)]
        self.done_buffers = [np.zeros((self.capacity, 1), dtype=np.float32) for _ in range(num_agents)]
    
    def add(self, obs: Dict, actions: Dict, rewards: Dict, next_obs: Dict, dones: Dict) -> None:
        """Add experience tuple to replay buffer."""
        agent_ids = list(obs.keys())
        for i, agent_id in enumerate(agent_ids):
            self.obs_buffers[i][self.ptr] = obs[agent_id].detach().cpu().numpy().flatten()
            self.next_obs_buffers[i][self.ptr] = next_obs[agent_id].detach().cpu().numpy().flatten()
            self.action_buffers[i][self.ptr] = actions[agent_id].detach().cpu().numpy().flatten()
            self.reward_buffers[i][self.ptr] = float(rewards[agent_id].detach().cpu().numpy())
            self.done_buffers[i][self.ptr] = float(dones[agent_id].detach().cpu().numpy())
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        if self.size < batch_size: 
            return [], [], [], [], []
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        obs_list = [torch.from_numpy(self.obs_buffers[i][indices]).to(self.device) for i in range(self.num_agents)]
        action_list = [torch.from_numpy(self.action_buffers[i][indices]).to(self.device) for i in range(self.num_agents)]
        reward_list = [torch.from_numpy(self.reward_buffers[i][indices]).to(self.device) for i in range(self.num_agents)]
        next_obs_list = [torch.from_numpy(self.next_obs_buffers[i][indices]).to(self.device) for i in range(self.num_agents)]
        done_list = [torch.from_numpy(self.done_buffers[i][indices]).to(self.device) for i in range(self.num_agents)]
        
        return obs_list, action_list, reward_list, next_obs_list, done_list

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= int(min_size)
    
    def clear(self) -> None:
        """Clear buffer contents."""
        self.ptr = 0
        self.size = 0
        print(f"[BUFFER] Buffer cleared - size reset to 0")
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class JointReplayBuffer:
    """
    Joint replay buffer for shared network MADDPG.
    
    Storage format:
    - obs_all: [capacity, total_obs_dim] - concatenated observations from all agents
    - actions_all: [capacity, total_action_dim] - concatenated actions from all agents  
    - rewards_all: [capacity, num_agents] - per-agent rewards
    - next_obs_all: [capacity, total_obs_dim] - concatenated next observations
    - done_any: [capacity, 1] - logical OR of agent done signals
    """
    
    def __init__(self, capacity: int, total_obs_dim: int, total_action_dim: int, num_agents: int, device: torch.device):
        self.capacity = int(capacity)
        self.total_obs_dim = int(total_obs_dim)
        self.total_action_dim = int(total_action_dim)
        self.num_agents = int(num_agents)
        self.device = device
        
        # Circular buffer state
        self.ptr = 0
        self.size = 0
        
        # Storage arrays (numpy for memory efficiency)
        self.obs = np.zeros((capacity, total_obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, total_action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, num_agents), dtype=np.float32)
        self.nobs = np.zeros((capacity, total_obs_dim), dtype=np.float32)
        self.done_any = np.zeros((capacity, 1), dtype=np.float32)
        
        print(f"[JOINT BUFFER] Initialized: capacity={capacity}, obs_dim={total_obs_dim}, act_dim={total_action_dim}")

    def add(self, obs_all: np.ndarray, act_all: np.ndarray, rewards_vec: np.ndarray, 
            next_obs_all: np.ndarray, done_any: bool) -> None:
        """
        Add joint experience to buffer.
        
        Args:
            obs_all: Concatenated observations [total_obs_dim]
            act_all: Concatenated actions [total_action_dim]
            rewards_vec: Per-agent rewards [num_agents]
            next_obs_all: Concatenated next observations [total_obs_dim]
            done_any: Logical OR of agent done signals
        """
        i = self.ptr
        self.obs[i] = obs_all
        self.act[i] = act_all
        self.rew[i] = rewards_vec
        self.nobs[i] = next_obs_all
        self.done_any[i] = float(done_any)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample random batch from joint buffer.
        
        Returns:
            obs_all: [batch_size, total_obs_dim]
            act_all: [batch_size, total_action_dim]
            rew_all: [batch_size, num_agents]
            nobs_all: [batch_size, total_obs_dim]
            done_any: [batch_size, 1]
        """
        if self.size < batch_size:
            return None
            
        idx = np.random.choice(self.size, batch_size, replace=False)
        
        obs = torch.from_numpy(self.obs[idx]).to(self.device)      # [B, total_obs_dim]
        act = torch.from_numpy(self.act[idx]).to(self.device)      # [B, total_action_dim]
        rew = torch.from_numpy(self.rew[idx]).to(self.device)      # [B, num_agents]
        nobs = torch.from_numpy(self.nobs[idx]).to(self.device)    # [B, total_obs_dim]
        done = torch.from_numpy(self.done_any[idx]).to(self.device) # [B, 1]
        
        return obs, act, rew, nobs, done

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def clear(self) -> None:
        """Clear buffer contents."""
        self.ptr = 0
        self.size = 0
        print("[JOINT BUFFER] Cleared")