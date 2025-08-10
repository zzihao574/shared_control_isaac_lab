"""Replay buffer for multi-agent reinforcement learning - Improved for parallel environments"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent parallel environments"""
    
    def __init__(self, capacity, num_agents, obs_dims, action_dims, device):
        """
        Args:
            capacity: Maximum buffer size
            num_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            action_dims: List of action dimensions for each agent
            device: PyTorch device
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.obs_buffers = []
        self.next_obs_buffers = []
        self.action_buffers = []
        self.reward_buffers = []
        self.done_buffers = []
        
        for i in range(num_agents):
            self.obs_buffers.append(np.zeros((capacity, obs_dims[i]), dtype=np.float32))
            self.next_obs_buffers.append(np.zeros((capacity, obs_dims[i]), dtype=np.float32))
            self.action_buffers.append(np.zeros((capacity, action_dims[i]), dtype=np.float32))
            self.reward_buffers.append(np.zeros((capacity, 1), dtype=np.float32))
            self.done_buffers.append(np.zeros((capacity, 1), dtype=np.float32))
    
    def add(self, obs_dict, action_dict, reward_dict, next_obs_dict, done_dict):
        """Add transitions from parallel environments to buffer
        
        Args:
            obs_dict: Dict of observations for each agent (can be batched)
            action_dict: Dict of actions for each agent (can be batched)
            reward_dict: Dict of rewards for each agent (can be batched)
            next_obs_dict: Dict of next observations for each agent (can be batched)
            done_dict: Dict of done flags for each agent (can be batched)
        """
        # 获取第一个智能体的数据来确定批量大小
        first_agent = list(obs_dict.keys())[0]
        first_obs = obs_dict[first_agent]
        
        # 转换为numpy并检查维度
        if torch.is_tensor(first_obs):
            first_obs = first_obs.cpu().numpy()
        
        # 判断是否为批量数据
        is_batched = len(first_obs.shape) > 1
        batch_size = first_obs.shape[0] if is_batched else 1
        
        # 确保不会超过缓冲区容量
        if batch_size > self.capacity:
            batch_size = self.capacity
            print(f"[WARNING] Batch size {batch_size} exceeds buffer capacity, truncating")
        
        # 计算可用空间
        available_space = self.capacity - self.ptr
        actual_batch_size = min(batch_size, available_space)
        
        for i, agent_id in enumerate(obs_dict.keys()):
            obs = obs_dict[agent_id]
            next_obs = next_obs_dict[agent_id]
            action = action_dict[agent_id]
            reward = reward_dict[agent_id]
            done = done_dict[agent_id]
            
            # 转换为numpy
            if torch.is_tensor(obs):
                obs = obs.cpu().numpy()
            if torch.is_tensor(next_obs):
                next_obs = next_obs.cpu().numpy()
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            if torch.is_tensor(reward):
                reward = reward.cpu().numpy()
            if torch.is_tensor(done):
                done = done.cpu().numpy()
            
            # 处理批量数据
            if is_batched:
                # 存储前actual_batch_size个样本
                end_idx = self.ptr + actual_batch_size
                self.obs_buffers[i][self.ptr:end_idx] = obs[:actual_batch_size]
                self.next_obs_buffers[i][self.ptr:end_idx] = next_obs[:actual_batch_size]
                self.action_buffers[i][self.ptr:end_idx] = action[:actual_batch_size]
                
                # 处理标量奖励和done
                if np.isscalar(reward):
                    self.reward_buffers[i][self.ptr:end_idx] = reward
                else:
                    reward_array = reward[:actual_batch_size]
                    if len(reward_array.shape) == 1:
                        reward_array = reward_array.reshape(-1, 1)
                    self.reward_buffers[i][self.ptr:end_idx] = reward_array
                
                if np.isscalar(done):
                    self.done_buffers[i][self.ptr:end_idx] = done
                else:
                    done_array = done[:actual_batch_size]
                    if len(done_array.shape) == 1:
                        done_array = done_array.reshape(-1, 1)
                    self.done_buffers[i][self.ptr:end_idx] = done_array
                
                # 如果还有剩余数据需要循环存储
                if actual_batch_size < batch_size and self.capacity > 0:
                    remaining = batch_size - actual_batch_size
                    remaining = min(remaining, self.capacity)
                    
                    self.obs_buffers[i][0:remaining] = obs[actual_batch_size:actual_batch_size+remaining]
                    self.next_obs_buffers[i][0:remaining] = next_obs[actual_batch_size:actual_batch_size+remaining]
                    self.action_buffers[i][0:remaining] = action[actual_batch_size:actual_batch_size+remaining]
                    
                    if not np.isscalar(reward):
                        reward_array = reward[actual_batch_size:actual_batch_size+remaining]
                        if len(reward_array.shape) == 1:
                            reward_array = reward_array.reshape(-1, 1)
                        self.reward_buffers[i][0:remaining] = reward_array
                    else:
                        self.reward_buffers[i][0:remaining] = reward
                    
                    if not np.isscalar(done):
                        done_array = done[actual_batch_size:actual_batch_size+remaining]
                        if len(done_array.shape) == 1:
                            done_array = done_array.reshape(-1, 1)
                        self.done_buffers[i][0:remaining] = done_array
                    else:
                        self.done_buffers[i][0:remaining] = done
                        
            else:
                # 单个样本
                self.obs_buffers[i][self.ptr] = obs
                self.next_obs_buffers[i][self.ptr] = next_obs
                self.action_buffers[i][self.ptr] = action
                self.reward_buffers[i][self.ptr] = reward if np.isscalar(reward) else reward.item()
                self.done_buffers[i][self.ptr] = done if np.isscalar(done) else done.item()
        
        # 更新指针和大小
        if is_batched:
            total_stored = min(batch_size, self.capacity)
            self.ptr = (self.ptr + total_stored) % self.capacity
            self.size = min(self.size + total_stored, self.capacity)
        else:
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch of experiences
        
        Returns:
            Tuple of (obs_list, action_list, reward_list, next_obs_list, done_list)
            Each list contains tensors for all agents
        """
        # 确保采样大小不超过缓冲区大小
        actual_batch_size = min(batch_size, self.size)
        
        indices = np.random.choice(self.size, actual_batch_size, replace=False)
        
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        
        for i in range(self.num_agents):
            obs_list.append(torch.FloatTensor(self.obs_buffers[i][indices]).to(self.device))
            action_list.append(torch.FloatTensor(self.action_buffers[i][indices]).to(self.device))
            reward_list.append(torch.FloatTensor(self.reward_buffers[i][indices]).to(self.device))
            next_obs_list.append(torch.FloatTensor(self.next_obs_buffers[i][indices]).to(self.device))
            done_list.append(torch.FloatTensor(self.done_buffers[i][indices]).to(self.device))
        
        return obs_list, action_list, reward_list, next_obs_list, done_list
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
        
    def get_statistics(self):
        """Get buffer statistics"""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'ptr': self.ptr,
            'utilization': self.size / self.capacity if self.capacity > 0 else 0
        }