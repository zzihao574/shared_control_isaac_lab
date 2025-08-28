"""
Replay buffer for multi-agent multi-environment parallel training.
Simplified version with monitoring methods removed.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Union

class MultiAgentReplayBuffer:
    """
    Multi-agent replay buffer for experience storage and sampling.
    
    Features:
    - Separate storage for each agent's observations and actions
    - Efficient numpy-based storage with torch tensor sampling
    - Configurable capacity with circular buffer behavior
    - Ready state checking for minimum buffer size requirements
    """
    
    def __init__(self, capacity: int, num_agents: int, obs_dims: List[int], action_dims: List[int], device: torch.device):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.device = device
        self.ptr, self.size = 0, 0
        
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
        return self.size >= min_size
    
    def clear(self) -> None:
        """Clear buffer contents."""
        self.ptr = 0
        self.size = 0
        print(f"[BUFFER] Buffer cleared")
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size