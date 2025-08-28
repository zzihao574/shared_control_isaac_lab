"""
Multi-environment parallel MADDPG algorithm.
清理版本 - 从环境获取维度信息，移除对YAML中重复配置的依赖
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from .ddpg_agent import DDPGAgent
from .replay_buffer import MultiAgentReplayBuffer

class MADDPG:
    def __init__(self, num_envs: int, env, params: Dict[str, Any], device: str = 'cuda'):
        self.env = env
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # FIX: Use the explicitly passed num_envs, which is guaranteed to be correct.
        self.num_envs = num_envs
        
        # 获取真正的环境配置 - 处理Gymnasium包装器
        actual_env = self._get_actual_env(env)
        
        # 从环境cfg获取agent信息 (单一可信来源)
        self.agent_ids = list(actual_env.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        # 从环境cfg获取维度信息 (移除对YAML的依赖)
        self.obs_dims = [actual_env.cfg.observation_spaces[agent] for agent in self.agent_ids]
        self.action_dims = [actual_env.cfg.action_spaces[agent] for agent in self.agent_ids]
        self.total_obs_dim = sum(self.obs_dims)
        self.total_action_dim = sum(self.action_dims)

        print(f"[INFO] Initializing MADDPG: {self.num_envs} environments, {self.num_envs * self.num_agents * 4} networks")
        print(f"[INFO] Auto-detected from environment cfg:")
        print(f"  Agent IDs: {self.agent_ids}")
        print(f"  Observation dims: {self.obs_dims} (total: {self.total_obs_dim})")
        print(f"  Action dims: {self.action_dims} (total: {self.total_action_dim})")

        self._initialize_agents()
        self._initialize_replay_buffers()
        
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.batch_size = int(maddpg_cfg.get('batch_size', 1024))
        self.gamma = float(maddpg_cfg.get('gamma', 0.9))
        self.update_interval = int(maddpg_cfg.get('update_interval', 100))
        self.min_buffer_size = int(maddpg_cfg.get('min_buffer_size', 1024))
        
        self.env_active = [True] * self.num_envs
        self.training_steps = 0
    
    def _get_actual_env(self, env):
        """获取真正的环境对象，处理Gymnasium包装器"""
        # 尝试多种方式访问底层环境
        if hasattr(env, 'cfg'):
            return env
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'cfg'):
            return env.unwrapped
        elif hasattr(env, 'env') and hasattr(env.env, 'cfg'):
            return env.env
        else:
            # 遍历可能的包装器层级
            current = env
            max_depth = 10  # 防止无限循环
            for _ in range(max_depth):
                if hasattr(current, 'cfg'):
                    return current
                elif hasattr(current, 'unwrapped'):
                    current = current.unwrapped
                elif hasattr(current, 'env'):
                    current = current.env
                else:
                    break
            
            raise AttributeError(f"Cannot find cfg attribute in environment: {type(env)}")
        
    def _initialize_agents(self) -> None:
        self.env_agents = {}
        for env_id in range(self.num_envs):
            self.env_agents[env_id] = {}
            for i, agent_id in enumerate(self.agent_ids):
                agent_name = f"{agent_id}_env_{env_id}"
                self.env_agents[env_id][agent_id] = DDPGAgent(
                    agent_id=agent_name, 
                    state_dim=self.obs_dims[i], 
                    action_dim=self.action_dims[i],
                    total_state_dim=self.total_obs_dim, 
                    total_action_dim=self.total_action_dim,
                    params=self.params, 
                    device=self.device
                )
    
    def _initialize_replay_buffers(self) -> None:
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.buffer_size = int(maddpg_cfg.get('max_replay_buffer_len', 10000))
        self.env_replay_buffers = {}
        for env_id in range(self.num_envs):
            self.env_replay_buffers[env_id] = MultiAgentReplayBuffer(
                capacity=self.buffer_size, 
                num_agents=self.num_agents,
                obs_dims=self.obs_dims, 
                action_dims=self.action_dims, 
                device=self.device
            )

    def select_actions(self, observations: Dict[str, torch.Tensor], active_envs: List[int], add_noise: bool = True) -> Dict[str, torch.Tensor]:
        """
        Select actions only for active environments to prevent IndexError.
        """
        obs_len = observations[self.agent_ids[0]].shape[0]
        assert obs_len == self.num_envs, \
            f"Observation tensor dimension ({obs_len}) does not match configured num_envs ({self.num_envs})"

        actions = {
            agent_id: torch.zeros(self.num_envs, self.action_dims[i], device=self.device) 
            for i, agent_id in enumerate(self.agent_ids)
        }
        
        for env_id in active_envs:
            for i, agent_id in enumerate(self.agent_ids):
                obs = observations[agent_id][env_id].cpu().numpy()
                action = self.env_agents[env_id][agent_id].select_action(obs, add_noise)
                actions[agent_id][env_id] = torch.from_numpy(action)
                
        return actions
    
    def store_transitions_selective(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], 
                                   rewards: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor], 
                                   dones: Dict[str, torch.Tensor], active_envs: List[int]) -> None:
        for env_id in active_envs:
            is_done = dones[self.agent_ids[0]][env_id]
            
            env_obs = {aid: obs[aid][env_id] for aid in self.agent_ids}
            env_actions = {aid: actions[aid][env_id] for aid in self.agent_ids}
            env_rewards = {aid: rewards[aid][env_id] for aid in self.agent_ids}
            env_next_obs = {aid: next_obs[aid][env_id] for aid in self.agent_ids}
            env_dones = {aid: is_done for aid in self.agent_ids}
            
            self.env_replay_buffers[env_id].add(env_obs, env_actions, env_rewards, env_next_obs, env_dones)
    
    def disable_environment(self, env_id: int) -> None:
        if 0 <= env_id < self.num_envs and self.env_active[env_id]:
            self.env_active[env_id] = False
            if env_id in self.env_replay_buffers:
                self.env_replay_buffers[env_id].clear()
                self.env_replay_buffers[env_id].capacity = 0
            print(f"[INFO] Environment {env_id} disabled")
    
    def update(self) -> Dict[str, Any]:
        self.training_steps += 1
        if self.training_steps % self.update_interval != 0:
            return {}
        
        stats = {'actor_losses': [], 'critic_losses': [], 'buffer_sizes': []}
        
        active_and_ready_envs = [i for i in range(self.num_envs) if self.env_active[i] and self.env_replay_buffers[i].is_ready(self.min_buffer_size)]

        for env_id in active_and_ready_envs:
            stats['buffer_sizes'].append(len(self.env_replay_buffers[env_id]))
            
            try:
                obs, act, rew, next_obs, done = self.env_replay_buffers[env_id].sample(self.batch_size)
                if not obs: continue

                obs_cat = torch.cat(obs, dim=-1)
                next_obs_cat = torch.cat(next_obs, dim=-1)
                act_cat = torch.cat(act, dim=-1)
                
                with torch.no_grad():
                    next_actions_list = []
                    for j, agent_id_j in enumerate(self.agent_ids):
                         next_mean, _ = self.env_agents[env_id][agent_id_j].actor_target(next_obs[j])
                         next_actions_list.append(next_mean)
                    next_actions_cat = torch.cat(next_actions_list, dim=-1)

                for i, agent_id in enumerate(self.agent_ids):
                    agent = self.env_agents[env_id][agent_id]
                    
                    with torch.no_grad():
                        q_next = agent.critic_target(next_obs_cat, next_actions_cat)
                        q_target = rew[i] + self.gamma * q_next * (1 - done[i])
                    
                    critic_out = agent.update_critic(obs_cat, act_cat, q_target)
                    stats['critic_losses'].append(critic_out['critic_loss'])
                    
                    actions_pred_list = []
                    for j, agent_id_j in enumerate(self.agent_ids):
                        if i == j:
                            mean, _ = agent.actor(obs[j])
                            actions_pred_list.append(mean)
                        else:
                            actions_pred_list.append(act[j].detach())
                    actions_pred_cat = torch.cat(actions_pred_list, dim=-1)

                    actor_loss = -agent.critic(obs_cat, actions_pred_cat).mean()
                    
                    actor_out = agent.update_actor(actor_loss)
                    stats['actor_losses'].append(actor_out['actor_loss'])
                    
                for agent in self.env_agents[env_id].values():
                    agent.soft_update()

            except Exception as e:
                print(f"[WARNING] Environment {env_id} update failed: {e}")
        
        return stats