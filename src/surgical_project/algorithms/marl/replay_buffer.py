"""Replay buffer for multi-agent multi-environment parallel training - 官方对齐版"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class MultiAgentReplayBuffer:
    """多智能体回放缓冲区 - 与官方对齐，简化数据处理"""
    
    def __init__(self, capacity, num_agents, obs_dims, action_dims, device):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dims = obs_dims  # 用于_get_empty_batch
        self.action_dims = action_dims  # 用于_get_empty_batch
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory - 官方风格，简单numpy数组
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
        """添加经验 - 官方风格，简化处理"""
        agent_ids = list(obs_dict.keys())
        
        for i, agent_id in enumerate(agent_ids):
            # 简化数据处理 - 删除复杂的nan检查和clamp
            obs = self._to_numpy(obs_dict[agent_id])
            next_obs = self._to_numpy(next_obs_dict[agent_id])
            action = self._to_numpy(action_dict[agent_id])
            reward = self._process_reward(reward_dict[agent_id])
            done = self._process_done(done_dict[agent_id])
            
            # 存储
            self.obs_buffers[i][self.ptr] = obs
            self.next_obs_buffers[i][self.ptr] = next_obs
            self.action_buffers[i][self.ptr] = action
            self.reward_buffers[i][self.ptr] = reward
            self.done_buffers[i][self.ptr] = done
        
        # 更新指针
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _to_numpy(self, data):
        """转换为numpy - 官方风格，简单处理"""
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        
        data = np.array(data, dtype=np.float32).flatten()
        return data
    
    def _process_reward(self, reward):
        """处理奖励 - 官方风格，删除复杂处理"""
        if torch.is_tensor(reward):
            reward = reward.cpu().numpy()
        
        if np.isscalar(reward):
            reward_val = float(reward)
        else:
            reward = np.array(reward).flatten()
            reward_val = float(reward[0]) if len(reward) > 0 else 0.0
        
        # 删除clamp和nan检查 - 官方没有这些
        return reward_val
    
    def _process_done(self, done):
        """处理完成标志 - 官方风格"""
        if torch.is_tensor(done):
            done = done.cpu().numpy()
        
        if np.isscalar(done):
            return 1.0 if done else 0.0
        else:
            done = np.array(done).flatten()
            return 1.0 if (len(done) > 0 and np.any(done)) else 0.0
    
    def sample(self, batch_size):
        """采样经验 - 官方风格，简单uniform sampling"""
        if self.size == 0:
            return self._get_empty_batch()
        
        actual_batch_size = min(batch_size, self.size)
        
        # 官方使用简单的uniform sampling
        indices = np.random.choice(self.size, actual_batch_size, replace=False)
        
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        
        for i in range(self.num_agents):
            obs_list.append(torch.from_numpy(self.obs_buffers[i][indices].copy()).to(self.device))
            action_list.append(torch.from_numpy(self.action_buffers[i][indices].copy()).to(self.device))
            reward_list.append(torch.from_numpy(self.reward_buffers[i][indices].copy()).to(self.device))
            next_obs_list.append(torch.from_numpy(self.next_obs_buffers[i][indices].copy()).to(self.device))
            done_list.append(torch.from_numpy(self.done_buffers[i][indices].copy()).to(self.device))
        
        return obs_list, action_list, reward_list, next_obs_list, done_list
    
    def _get_empty_batch(self):
        """空批次 - 使用动态维度，消除硬编码"""
        obs_list = [torch.zeros(0, self.obs_dims[i], device=self.device) for i in range(self.num_agents)]
        action_list = [torch.zeros(0, self.action_dims[i], device=self.device) for i in range(self.num_agents)]
        reward_list = [torch.zeros(0, 1, device=self.device) for i in range(self.num_agents)]
        next_obs_list = [torch.zeros(0, self.obs_dims[i], device=self.device) for i in range(self.num_agents)]
        done_list = [torch.zeros(0, 1, device=self.device) for i in range(self.num_agents)]
        
        return obs_list, action_list, reward_list, next_obs_list, done_list
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0