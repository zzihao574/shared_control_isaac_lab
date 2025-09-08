"""
Multi-environment parallel MADDPG algorithm.
Grouped replay buffers with dual protection mechanism.
Multi-agent update with single-agent control via select_actions.
MODIFIED: Unified buffer clearing at single entry point in update()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from .ddpg_agent import DDPGAgent
from .replay_buffer import MultiAgentReplayBuffer

class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient algorithm with grouped replay buffers.
    
    Features:
    - Multi-environment parallel training
    - Grouped replay buffers (4 envs share 1 buffer by default)
    - Centralized training, decentralized execution
    - Dual protection: environment disabling + training filtering
    - Multi-agent update logic with single-agent control via select_actions
    - MODIFIED: Unified buffer clearing at single entry point
    - Comprehensive debugging and monitoring
    """
    
    def __init__(self, num_envs: int, env, params: Dict[str, Any], device: str = 'cuda'):
        self.env = env
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        
        # Get actual environment configuration
        actual_env = self._get_actual_env(env)
        self.agent_ids = list(actual_env.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        # CRITICAL: Define robot/human indices explicitly
        self.robot_id = "robot"
        self.human_id = "human"
        self.robot_idx = self.agent_ids.index(self.robot_id)
        self.human_idx = self.agent_ids.index(self.human_id)
        
        # Get dimensions from environment cfg
        self.obs_dims = [actual_env.cfg.observation_spaces[agent] for agent in self.agent_ids]
        self.action_dims = [actual_env.cfg.action_spaces[agent] for agent in self.agent_ids]
        self.total_obs_dim = sum(self.obs_dims)
        self.total_action_dim = sum(self.action_dims)

        print(f"[INFO] Initializing MADDPG with grouped buffers: {self.num_envs} environments")
        print(f"[INFO] Agent IDs: {self.agent_ids}")
        print(f"[INFO] CRITICAL: robot_idx={self.robot_idx}, human_idx={self.human_idx}")
        print(f"[INFO] Multi-agent update + single-agent control via select_actions")
        print(f"[INFO] Observation dims: {self.obs_dims} (total: {self.total_obs_dim})")
        print(f"[INFO] Action dims: {self.action_dims} (total: {self.total_action_dim})")

        self._initialize_agents()
        self._initialize_replay_buffers()
        
        # Load training hyperparameters
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.batch_size = int(maddpg_cfg.get('batch_size', 1024))
        self.gamma = float(maddpg_cfg.get('gamma', 0.9))
        self.update_interval = int(maddpg_cfg.get('update_interval', 100))
        self.min_buffer_size = int(maddpg_cfg.get('min_buffer_size', 1024))
        
        self.training_steps = 0
        self.disabled_environments = set()  # Track disabled environments (don't clear shared buffers)
        
        # 累计计数器
        self.update_count = 0
        
        # 将在trainer中注入这些依赖
        self.progress = None  # UnifiedProgressManager
        self.max_episodes = None
    
    def _get_actual_env(self, env):
        """Get actual environment object through Gymnasium wrappers."""
        if hasattr(env, 'cfg'):
            return env
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'cfg'):
            return env.unwrapped
        elif hasattr(env, 'env') and hasattr(env.env, 'cfg'):
            return env.env
        else:
            # Traverse wrapper layers
            current = env
            for _ in range(10):  # Prevent infinite loops
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
        """Initialize DDPG agents for all environments and agents."""
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
        """Initialize grouped replay buffers with alias binding."""
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.buffer_size = int(maddpg_cfg.get('max_replay_buffer_len', 10000))
        self.group_size = int(maddpg_cfg.get('buffer_group_size', 4))
        
        # 构建分组映射
        self.env_to_group = {env_id: env_id // self.group_size for env_id in range(self.num_envs)}
        self.group_members = {}
        for env_id in range(self.num_envs):
            group_id = env_id // self.group_size
            if group_id not in self.group_members:
                self.group_members[group_id] = []
            self.group_members[group_id].append(env_id)
        
        # 组状态管理
        self.finished_groups = set()
        self.cleared_groups = set()
        
        # 创建实际的buffer实例（按组数量）
        num_groups = len(self.group_members)
        group_buffers = {}
        for group_id in range(num_groups):
            group_buffers[group_id] = MultiAgentReplayBuffer(
                capacity=self.buffer_size, 
                num_agents=self.num_agents,
                obs_dims=self.obs_dims, 
                action_dims=self.action_dims, 
                device=self.device
            )
        
        # 别名绑定：同组env指向同一个buffer实例
        self.env_replay_buffers = {}
        for env_id in range(self.num_envs):
            group_id = self.env_to_group[env_id]
            self.env_replay_buffers[env_id] = group_buffers[group_id]
        
        # 打印分组信息
        print(f"[BUFFER] Grouped {self.num_envs} environments into {num_groups} groups (group_size={self.group_size})")
        for group_id, members in self.group_members.items():
            print(f"[BUFFER] Group {group_id}: envs {members} (size: {len(members)})")
    
    def on_episode_end(self, env_id: int) -> None:
        """当某个env的episode结束时，检查其所属组是否完成"""
        if self.progress is None or self.max_episodes is None:
            return  # 依赖未注入，跳过
            
        group_id = self.env_to_group[env_id]
        if group_id in self.finished_groups:
            return  # 该组已标记完成
            
        # 检查组完成状态
        if self._check_group_completion(group_id):
            self.finished_groups.add(group_id)
            print(f"[BUFFER] Group {group_id} completed (all envs >= {self.max_episodes} episodes)")
    
    def _check_group_completion(self, group_id: int) -> bool:
        """检查组是否完成（组内所有活跃env的episode_count >= max_episodes）"""
        active_members = [e for e in self.group_members[group_id] 
                         if e not in self.disabled_environments]
        if not active_members:  # 整组都被禁用
            return True
        return all(self.progress.episode_counts[e] >= self.max_episodes 
                  for e in active_members)

    def select_actions(self, observations: Dict[str, torch.Tensor], active_envs: List[int], add_noise: bool = True) -> Dict[str, torch.Tensor]:
        """
        Select actions with single/multi-agent control.
        Single-agent mode: Skip human actions (they remain zero).
        Multi-agent mode: Generate actions for all agents.
        """
        obs_len = observations[self.agent_ids[0]].shape[0]
        assert obs_len == self.num_envs, f"Observation dimension mismatch: {obs_len} vs {self.num_envs}"

        actions = {
            agent_id: torch.zeros(self.num_envs, self.action_dims[i], device=self.device) 
            for i, agent_id in enumerate(self.agent_ids)
        }
        
        for env_id in active_envs:
            for i, agent_id in enumerate(self.agent_ids):
                # SINGLE AGENT CONTROL: Skip human agent (comment out this line to enable multi-agent)
                if agent_id.lower() == "human":
                    continue  # Human actions remain zero for single-agent training
                
                obs = observations[agent_id][env_id].cpu().numpy()
                action = self.env_agents[env_id][agent_id].select_action(obs, add_noise)
                actions[agent_id][env_id] = torch.from_numpy(action).to(self.device)
                
        return actions
    
    def store_transitions_selective(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], 
                                   rewards: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor], 
                                   dones: Dict[str, torch.Tensor], active_envs: List[int]) -> None:
        """Store transitions only for active environments with FIXED agent indexing."""
        for env_id in active_envs:
            if env_id in self.disabled_environments:
                continue  # 跳过已禁用的env

            # FIXED: Use robot's done signal (or logical OR of both agents) - keep as tensor
            is_done = dones[self.robot_id][env_id] | dones[self.human_id][env_id]
            
            env_obs = {aid: obs[aid][env_id] for aid in self.agent_ids}
            env_actions = {aid: actions[aid][env_id] for aid in self.agent_ids}
            env_rewards = {aid: rewards[aid][env_id] for aid in self.agent_ids}
            env_next_obs = {aid: next_obs[aid][env_id] for aid in self.agent_ids}
            env_dones = {aid: is_done for aid in self.agent_ids}
            
            # 由于别名绑定，同组env会自动写入同一个shared buffer
            self.env_replay_buffers[env_id].add(env_obs, env_actions, env_rewards, env_next_obs, env_dones)
    
    def disable_environment(self, env_id: int) -> None:
        """
        MODIFIED: 只标记env为禁用，不清空共享buffer（避免误伤同组其他env）
        """
        if 0 <= env_id < self.num_envs:
            self.disabled_environments.add(env_id)
            group_id = self.env_to_group[env_id]
            print(f"[DISABLE] Environment {env_id} disabled (group {group_id}), buffer preserved for group")
    
    def update(self, active_envs: List[int]) -> Dict[str, Any]:
        """
        MODIFIED: Multi-agent update logic with unified buffer clearing at entry point.
        Single-agent behavior is controlled by select_actions() skipping human actions.
        """
        self.training_steps += 1
        
        # UNIFIED ENTRY POINT: Handle group buffer clearing at the beginning of update
        self._handle_group_buffer_clearing()
        
        # Only train on active environments with sufficient data
        active_and_ready_envs = [
            i for i in active_envs
            if (i not in self.disabled_environments)
            and self.env_replay_buffers[i].is_ready(self.min_buffer_size)
        ]

        if not active_and_ready_envs:
            return {"updates": 0, "training_steps": self.training_steps}

        if self.training_steps % self.update_interval != 0:
            return {"updates": 0, "training_steps": self.training_steps}
        
        # Debug information for dual protection
        disabled_but_active = [env_id for env_id in active_envs 
                              if env_id in self.disabled_environments]
        
        if disabled_but_active:
            print(f"[DEBUG] Environments {disabled_but_active} are disabled but still in active list")

        # 收集Q值数据用于统计（robot作为基准）
        global_q_vals_robot = []
        global_q_targets_robot = []
        per_env_q_vals_robot = {0: [], 1: []}
        per_env_q_targets_robot = {0: [], 1: []}

        # 收集训练损失
        actor_losses = []
        critic_losses = []
        updates_count = 0

        for env_id in active_and_ready_envs:
            # Additional safety check: skip if environment was disabled
            if env_id in self.disabled_environments:
                continue
                
            try:
                obs, act, rew, next_obs, done = self.env_replay_buffers[env_id].sample(self.batch_size)
                if not obs: 
                    continue

                # Centralized training
                obs_cat = torch.cat(obs, dim=-1)
                next_obs_cat = torch.cat(next_obs, dim=-1)
                act_cat = torch.cat(act, dim=-1)
                
                # Target actions for centralized training
                with torch.no_grad():
                    next_actions_list = []
                    for j, agent_id_j in enumerate(self.agent_ids):
                        next_mean, _ = self.env_agents[env_id][agent_id_j].actor_target(next_obs[j])
                        next_actions_list.append(next_mean)
                    next_actions_cat = torch.cat(next_actions_list, dim=-1)

                # 计算robot的Q值和目标（用于统计）
                robot_agent = self.env_agents[env_id][self.robot_id]
                with torch.no_grad():
                    q_current = robot_agent.critic(obs_cat, act_cat)
                    q_target_next = robot_agent.critic_target(next_obs_cat, next_actions_cat)
                    q_target_robot = rew[self.robot_idx] + self.gamma * q_target_next * (1 - done[self.robot_idx])

                # 收集Q值数据
                global_q_vals_robot.append(q_current.detach())
                global_q_targets_robot.append(q_target_robot.detach())
                
                # 分环境收集（env0, env1）
                if env_id in (0, 1):
                    per_env_q_vals_robot[env_id].append(q_current.detach())
                    per_env_q_targets_robot[env_id].append(q_target_robot.detach())

                # Update each agent (multi-agent training)
                for i, agent_id in enumerate(self.agent_ids):
                    agent = self.env_agents[env_id][agent_id]
                    
                    # Each agent uses its own reward and done signal for TD target
                    agent_reward = rew[i]
                    agent_done = done[i]
                    
                    # Critic update - each agent learns its own value function
                    with torch.no_grad():
                        q_next = agent.critic_target(next_obs_cat, next_actions_cat)
                        q_target = agent_reward + self.gamma * q_next * (1 - agent_done)
                    
                    critic_out = agent.update_critic(obs_cat, act_cat, q_target)
                    critic_losses.append(critic_out['critic_loss'])
                    
                    # Actor update - CTDE (Centralized Training, Decentralized Execution)
                    actions_pred_list = []
                    for j, agent_id_j in enumerate(self.agent_ids):
                        if i == j:  # Current agent uses its own actor
                            mean, _ = agent.actor(obs[j])
                            actions_pred_list.append(mean)
                        else:  # Other agents use actual actions (detached)
                            actions_pred_list.append(act[j].detach())
                    actions_pred_cat = torch.cat(actions_pred_list, dim=-1)

                    actor_loss = -agent.critic(obs_cat, actions_pred_cat).mean()
                    actor_out = agent.update_actor(actor_loss)
                    actor_losses.append(actor_out['actor_loss'])
                    
                    # Soft update target networks for this agent
                    agent.soft_update()
                
                updates_count += 1

            except Exception as e:
                print(f"[WARNING] Environment {env_id} update failed: {e}")

        # ====== Minimal stats aggregation (ONLY the needed ones) ======
        stats = {}

        # 训练损失与缓冲区
        if actor_losses:
            stats["training/actor_loss_mean"] = float(np.mean(actor_losses))
            stats["training/actor_loss_std"] = float(np.std(actor_losses))
        if critic_losses:
            stats["training/critic_loss_mean"] = float(np.mean(critic_losses))
            stats["training/critic_loss_std"] = float(np.std(critic_losses))
        
        # 修改avg_buffer_size统计：按组buffer平均长度
        participating_groups = set(self.env_to_group[i] for i in active_and_ready_envs)
        group_buffer_sizes = []
        for group_id in participating_groups:
            # 取该组任一成员的buffer（都指向同一实例）
            representative_env = self.group_members[group_id][0]
            group_buffer_sizes.append(len(self.env_replay_buffers[representative_env]))
        avg_buffer_size = float(np.mean(group_buffer_sizes)) if group_buffer_sizes else 0.0
        stats["training/avg_buffer_size"] = avg_buffer_size

        # 全局 Q / target / TD（以 robot 为基准）
        if global_q_vals_robot:
            q_vals_all = torch.cat(global_q_vals_robot, dim=0)
            q_targets_all = torch.cat(global_q_targets_robot, dim=0)
            td_abs_all = torch.abs(q_vals_all - q_targets_all)

            stats["algo/q_mean"] = float(q_vals_all.mean().item())
            stats["algo/q_std"] = float(q_vals_all.std().item())
            stats["algo/q_target_mean"] = float(q_targets_all.mean().item())
            stats["algo/q_target_std"] = float(q_targets_all.std().item())
            stats["algo/td_error_mean"] = float(td_abs_all.mean().item())

        # 工具函数
        def _corrcoef_safe(a, b):
            if a.numel() < 2: return 0.0
            if torch.std(a) < 1e-8 or torch.std(b) < 1e-8: return 0.0
            return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item())

        def _rmse(a, b):
            return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

        # 单环境 env0 / env1
        for eid in (0, 1):
            if eid in per_env_q_vals_robot and len(per_env_q_vals_robot[eid]) > 0:
                q_e = torch.cat(per_env_q_vals_robot[eid], dim=0)
                qt_e = torch.cat(per_env_q_targets_robot[eid], dim=0)

                stats[f"env{eid}/algo/q_mean"] = float(q_e.mean().item())
                stats[f"env{eid}/algo/q_std"] = float(q_e.std().item())
                stats[f"env{eid}/algo/q_target_mean"] = float(qt_e.mean().item())
                stats[f"env{eid}/algo/q_std"] = float(q_e.std().item())
                stats[f"env{eid}/algo/q_target_mean"] = float(qt_e.mean().item())
                stats[f"env{eid}/algo/q_target_std"] = float(qt_e.std().item())
                stats[f"env{eid}/algo/td_error_mean"] = float(torch.abs(q_e - qt_e).mean().item())
                stats[f"env{eid}/algo/q_qt_corr"] = _corrcoef_safe(q_e, qt_e)
                stats[f"env{eid}/algo/td_rmse"] = _rmse(q_e, qt_e)

        # 累计 updates
        if updates_count > 0:
            self.update_count += 1
            stats["training/updates"] = int(self.update_count)

        return stats
    
    def _handle_group_buffer_clearing(self) -> None:
        """UNIFIED ENTRY POINT: 在update开头统一处理完成组的buffer清空"""
        groups_to_clear = self.finished_groups - self.cleared_groups
        
        for group_id in groups_to_clear:
            # 取该组任一成员env_id → 取到共享buffer → clear()一次
            representative_env = self.group_members[group_id][0]
            self.env_replay_buffers[representative_env].clear()
            self.cleared_groups.add(group_id)
            print(f"[BUFFER] Cleared group {group_id} buffer (envs: {self.group_members[group_id]})")
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get detailed buffer status for debugging grouped buffer mechanism."""
        # 计算每组的buffer大小（避免重复计算同一buffer）
        group_buffer_sizes = {}
        for group_id in self.group_members:
            representative_env = self.group_members[group_id][0]
            group_buffer_sizes[group_id] = len(self.env_replay_buffers[representative_env])
        
        status = {
            'total_envs': self.num_envs,
            'total_groups': len(self.group_members),
            'disabled_envs': len(self.disabled_environments),
            'finished_groups': len(self.finished_groups),
            'cleared_groups': len(self.cleared_groups),
            'group_buffer_sizes': group_buffer_sizes,
            'disabled_envs_list': list(self.disabled_environments),
            'finished_groups_list': list(self.finished_groups),
            'cleared_groups_list': list(self.cleared_groups),
        }
        return status