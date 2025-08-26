"""
Replay buffer for multi-agent multi-environment parallel training.
Optimized implementation with improved data handling and monitoring capabilities.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any


class MultiAgentReplayBuffer:
    """
    Multi-agent replay buffer for MADDPG training.
    
    Optimized for multi-environment parallel training with efficient memory management
    and simplified data processing pipeline. Supports variable batch sizes and provides
    comprehensive statistics for monitoring.
    """
    
    def __init__(self, capacity: int, num_agents: int, obs_dims: List[int], 
                 action_dims: List[int], device: torch.device):
        """
        Initialize multi-agent replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            num_agents: Number of agents in the environment
            obs_dims: List of observation dimensions for each agent
            action_dims: List of action dimensions for each agent
            device: PyTorch device for tensor operations
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        
        # Buffer state tracking
        self.ptr = 0  # Current insertion pointer (circular buffer)
        self.size = 0  # Current number of stored transitions
        
        # Pre-allocate memory buffers for efficient storage
        # Using separate buffers for each agent to maintain data locality
        self._initialize_storage_buffers()
    
    def _initialize_storage_buffers(self) -> None:
        """Initialize pre-allocated storage buffers for all agents."""
        self.obs_buffers = []
        self.next_obs_buffers = []
        self.action_buffers = []
        self.reward_buffers = []
        self.done_buffers = []
        
        for i in range(self.num_agents):
            # Observations and next observations
            self.obs_buffers.append(
                np.zeros((self.capacity, self.obs_dims[i]), dtype=np.float32)
            )
            self.next_obs_buffers.append(
                np.zeros((self.capacity, self.obs_dims[i]), dtype=np.float32)
            )
            
            # Actions
            self.action_buffers.append(
                np.zeros((self.capacity, self.action_dims[i]), dtype=np.float32)
            )
            
            # Rewards and done flags (scalar per agent)
            self.reward_buffers.append(
                np.zeros((self.capacity, 1), dtype=np.float32)
            )
            self.done_buffers.append(
                np.zeros((self.capacity, 1), dtype=np.float32)
            )
    
    def add(self, obs_dict: Dict[str, torch.Tensor], action_dict: Dict[str, torch.Tensor], 
            reward_dict: Dict[str, torch.Tensor], next_obs_dict: Dict[str, torch.Tensor], 
            done_dict: Dict[str, torch.Tensor]) -> None:
        """
        Add a transition to the replay buffer.
        
        Processes and stores one transition from all agents. Handles various input
        formats and ensures consistent data storage format.
        
        Args:
            obs_dict: Current observations for each agent
            action_dict: Actions taken by each agent
            reward_dict: Rewards received by each agent
            next_obs_dict: Next observations for each agent
            done_dict: Terminal flags for each agent
        """
        agent_ids = list(obs_dict.keys())
        
        if len(agent_ids) != self.num_agents:
            print(f"[WARNING] Expected {self.num_agents} agents, got {len(agent_ids)}")
        
        for i, agent_id in enumerate(agent_ids):
            if i >= self.num_agents:
                break  # Skip excess agents
                
            try:
                # Convert and validate data for this agent
                obs = self._process_observation(obs_dict[agent_id], i)
                next_obs = self._process_observation(next_obs_dict[agent_id], i)
                action = self._process_action(action_dict[agent_id], i)
                reward = self._process_reward(reward_dict[agent_id])
                done = self._process_done(done_dict[agent_id])
                
                # Store in pre-allocated buffers
                self.obs_buffers[i][self.ptr] = obs
                self.next_obs_buffers[i][self.ptr] = next_obs
                self.action_buffers[i][self.ptr] = action
                self.reward_buffers[i][self.ptr] = reward
                self.done_buffers[i][self.ptr] = done
                
            except Exception as e:
                print(f"[ERROR] Failed to store transition for agent {agent_id}: {e}")
                # Continue with other agents even if one fails
                continue
        
        # Update buffer pointers (circular buffer behavior)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _process_observation(self, obs: Union[torch.Tensor, np.ndarray], agent_idx: int) -> np.ndarray:
        """
        Process observation data for storage.
        
        Args:
            obs: Observation tensor or array
            agent_idx: Agent index for dimension validation
            
        Returns:
            Processed observation as numpy array
        """
        # Convert to numpy if needed
        if torch.is_tensor(obs):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.array(obs, dtype=np.float32)
        
        # Flatten and validate dimensions
        obs_flat = obs_np.flatten()
        expected_dim = self.obs_dims[agent_idx]
        
        if len(obs_flat) != expected_dim:
            print(f"[WARNING] Agent {agent_idx} obs dim mismatch: got {len(obs_flat)}, expected {expected_dim}")
            # Pad or truncate as needed
            if len(obs_flat) < expected_dim:
                obs_flat = np.pad(obs_flat, (0, expected_dim - len(obs_flat)), 'constant')
            else:
                obs_flat = obs_flat[:expected_dim]
        
        return obs_flat
    
    def _process_action(self, action: Union[torch.Tensor, np.ndarray], agent_idx: int) -> np.ndarray:
        """
        Process action data for storage.
        
        Args:
            action: Action tensor or array
            agent_idx: Agent index for dimension validation
            
        Returns:
            Processed action as numpy array
        """
        # Convert to numpy if needed
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.array(action, dtype=np.float32)
        
        # Flatten and validate dimensions
        action_flat = action_np.flatten()
        expected_dim = self.action_dims[agent_idx]
        
        if len(action_flat) != expected_dim:
            print(f"[WARNING] Agent {agent_idx} action dim mismatch: got {len(action_flat)}, expected {expected_dim}")
            # Pad or truncate as needed
            if len(action_flat) < expected_dim:
                action_flat = np.pad(action_flat, (0, expected_dim - len(action_flat)), 'constant')
            else:
                action_flat = action_flat[:expected_dim]
        
        return action_flat
    
    def _process_reward(self, reward: Union[torch.Tensor, np.ndarray, float]) -> float:
        """
        Process reward data for storage.
        
        Args:
            reward: Reward value in various formats
            
        Returns:
            Processed reward as float
        """
        if torch.is_tensor(reward):
            reward_val = reward.detach().cpu().item()
        elif np.isscalar(reward):
            reward_val = float(reward)
        else:
            # Handle array-like rewards
            reward_array = np.array(reward).flatten()
            reward_val = float(reward_array[0]) if len(reward_array) > 0 else 0.0
        
        # Basic sanity check for reward values
        if not np.isfinite(reward_val):
            print(f"[WARNING] Invalid reward value: {reward_val}, setting to 0.0")
            reward_val = 0.0
        
        return reward_val
    
    def _process_done(self, done: Union[torch.Tensor, np.ndarray, bool]) -> float:
        """
        Process done flag for storage.
        
        Args:
            done: Done flag in various formats
            
        Returns:
            Processed done flag as float (0.0 or 1.0)
        """
        if torch.is_tensor(done):
            done_val = done.detach().cpu().item()
        elif np.isscalar(done):
            done_val = done
        else:
            # Handle array-like done flags
            done_array = np.array(done).flatten()
            done_val = done_array[0] if len(done_array) > 0 else False
        
        return 1.0 if bool(done_val) else 0.0
    
    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], 
                                             List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample a batch of transitions from the replay buffer.
        
        Uses uniform random sampling to select experiences. Returns data organized
        by agent for efficient multi-agent training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)
            Each element is a list of tensors, one per agent
        """
        if self.size == 0:
            return self._get_empty_batch()
        
        # Determine actual sampling size
        actual_batch_size = min(batch_size, self.size)
        
        # Sample random indices from available data
        indices = np.random.choice(self.size, actual_batch_size, replace=False)
        
        # Prepare output lists
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        
        # Sample data for each agent
        for i in range(self.num_agents):
            try:
                # Sample and convert to tensors
                obs_tensor = torch.from_numpy(
                    self.obs_buffers[i][indices].copy()
                ).to(self.device)
                
                action_tensor = torch.from_numpy(
                    self.action_buffers[i][indices].copy()
                ).to(self.device)
                
                reward_tensor = torch.from_numpy(
                    self.reward_buffers[i][indices].copy()
                ).to(self.device)
                
                next_obs_tensor = torch.from_numpy(
                    self.next_obs_buffers[i][indices].copy()
                ).to(self.device)
                
                done_tensor = torch.from_numpy(
                    self.done_buffers[i][indices].copy()
                ).to(self.device)
                
                # Add to output lists
                obs_list.append(obs_tensor)
                action_list.append(action_tensor)
                reward_list.append(reward_tensor)
                next_obs_list.append(next_obs_tensor)
                done_list.append(done_tensor)
                
            except Exception as e:
                print(f"[ERROR] Failed to sample data for agent {i}: {e}")
                # Return empty batch on error
                return self._get_empty_batch()
        
        return obs_list, action_list, reward_list, next_obs_list, done_list
    
    def _get_empty_batch(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], 
                                      List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate empty batch with correct dimensions.
        
        Returns:
            Tuple of empty tensor lists with proper shapes
        """
        obs_list = [torch.zeros(0, self.obs_dims[i], device=self.device) for i in range(self.num_agents)]
        action_list = [torch.zeros(0, self.action_dims[i], device=self.device) for i in range(self.num_agents)]
        reward_list = [torch.zeros(0, 1, device=self.device) for i in range(self.num_agents)]
        next_obs_list = [torch.zeros(0, self.obs_dims[i], device=self.device) for i in range(self.num_agents)]
        done_list = [torch.zeros(0, 1, device=self.device) for i in range(self.num_agents)]
        
        return obs_list, action_list, reward_list, next_obs_list, done_list
    
    def can_sample(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of samples required
            
        Returns:
            True if buffer can provide the required number of samples
        """
        return self.size >= min_size
    
    def is_ready(self, min_size: int) -> bool:
        """
        Legacy method name for backward compatibility.
        
        Args:
            min_size: Minimum number of samples required
            
        Returns:
            True if buffer can provide the required number of samples
        """
        return self.can_sample(min_size)
    
    def clear(self) -> None:
        """Clear the replay buffer and reset pointers."""
        self.ptr = 0
        self.size = 0
        print(f"[BUFFER] Buffer cleared")
    
    def __len__(self) -> int:
        """Get current number of stored transitions."""
        return self.size
    
    def get_fill_ratio(self) -> float:
        """Get buffer fill ratio (0.0 to 1.0)."""
        return self.size / self.capacity if self.capacity > 0 else 0.0
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get detailed buffer storage information.
        
        Returns:
            Dictionary containing buffer statistics and configuration
        """
        total_memory_mb = 0
        agent_memory_info = {}
        
        for i in range(self.num_agents):
            # Calculate memory usage per agent
            obs_memory = self.obs_buffers[i].nbytes
            next_obs_memory = self.next_obs_buffers[i].nbytes
            action_memory = self.action_buffers[i].nbytes
            reward_memory = self.reward_buffers[i].nbytes
            done_memory = self.done_buffers[i].nbytes
            
            agent_total = obs_memory + next_obs_memory + action_memory + reward_memory + done_memory
            total_memory_mb += agent_total
            
            agent_memory_info[f'agent_{i}'] = {
                'obs_memory_bytes': obs_memory,
                'action_memory_bytes': action_memory,
                'reward_memory_bytes': reward_memory,
                'total_memory_bytes': agent_total,
                'obs_dim': self.obs_dims[i],
                'action_dim': self.action_dims[i]
            }
        
        return {
            'capacity': self.capacity,
            'current_size': self.size,
            'fill_ratio': self.get_fill_ratio(),
            'num_agents': self.num_agents,
            'total_memory_mb': total_memory_mb / (1024 * 1024),
            'agent_memory_info': agent_memory_info,
            'device': str(self.device)
        }
    
    def get_recent_statistics(self, n_recent: int = 1000) -> Dict[str, Any]:
        """
        Analyze recent transitions for monitoring purposes.
        
        Args:
            n_recent: Number of recent transitions to analyze
            
        Returns:
            Dictionary containing recent transition statistics
        """
        if self.size == 0:
            return {'message': 'Buffer is empty'}
        
        n_samples = min(n_recent, self.size)
        recent_stats = {}
        
        for i in range(self.num_agents):
            # Get recent reward statistics
            if self.ptr >= n_samples:
                recent_rewards = self.reward_buffers[i][self.ptr - n_samples:self.ptr].flatten()
            else:
                # Handle wrap-around
                recent_rewards = np.concatenate([
                    self.reward_buffers[i][self.capacity - (n_samples - self.ptr):].flatten(),
                    self.reward_buffers[i][:self.ptr].flatten()
                ])
            
            # Get recent done statistics
            if self.ptr >= n_samples:
                recent_dones = self.done_buffers[i][self.ptr - n_samples:self.ptr].flatten()
            else:
                recent_dones = np.concatenate([
                    self.done_buffers[i][self.capacity - (n_samples - self.ptr):].flatten(),
                    self.done_buffers[i][:self.ptr].flatten()
                ])
            
            recent_stats[f'agent_{i}'] = {
                'mean_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'min_reward': np.min(recent_rewards),
                'max_reward': np.max(recent_rewards),
                'episode_end_rate': np.mean(recent_dones),
                'total_episodes_seen': np.sum(recent_dones)
            }
        
        return recent_stats