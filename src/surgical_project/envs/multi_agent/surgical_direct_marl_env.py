# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical Direct MARL Environment with Human-Robot Collaboration."""

from __future__ import annotations

import torch
import numpy as np
from typing import Any, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class TrajectoryManager:
    """轨迹管理器 - 距离驱动的目标点切换（与MBRL对齐）"""
    
    def __init__(self, device: torch.device, target_points: list, reach_threshold: float = 0.01):
        self.device = device
        self.target_points = [torch.tensor(point, device=device, dtype=torch.float32) for point in target_points]
        self.reach_threshold = reach_threshold
        self.current_target_index = 0
        
    def get_current_target(self) -> torch.Tensor:
        """获取当前目标点"""
        return self.target_points[self.current_target_index]
        
    def update_target(self, current_pos: torch.Tensor) -> bool:
        """基于当前位置更新目标点"""
        current_target = self.target_points[self.current_target_index]
        
        if current_pos.dim() > 1:
            distance = torch.norm(current_pos[0] - current_target)
        else:
            distance = torch.norm(current_pos - current_target)
        
        if distance < self.reach_threshold and self.current_target_index < len(self.target_points) - 1:
            self.current_target_index += 1
            return True
        return False
        
    def reset_trajectory(self):
        """重置轨迹到起始点"""
        self.current_target_index = 0
        
    def get_progress(self) -> float:
        """获取轨迹进度"""
        return self.current_target_index / max(1, len(self.target_points) - 1)
        
    def is_final_target_reached(self, current_pos: torch.Tensor) -> bool:
        """检查是否到达最终目标点"""
        if self.current_target_index < len(self.target_points) - 1:
            return False
            
        final_target = self.target_points[-1]
        if current_pos.dim() > 1:
            distance = torch.norm(current_pos[0] - final_target)
        else:
            distance = torch.norm(current_pos - final_target)
            
        return distance < self.reach_threshold


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """人机协作手术MARL环境 - 协作控制Omni设备"""
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境"""
        super().__init__(cfg, render_mode, **kwargs)
        
        # 时间步长
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # 轨迹管理器（与MBRL对齐）
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            target_points=self.cfg.target_points,
            reach_threshold=self.cfg.target_reach_threshold
        )
        
        # 智能体动作和状态跟踪
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, self.cfg.action_spaces[agent], device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # 物理参数跟踪
        self.combined_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 人类仿真状态
        self.human_intention = torch.zeros(self.num_envs, 3, device=self.device)
        self.human_reaction_timer = torch.zeros(self.num_envs, device=self.device)
        
        # 协作指标
        self.trust_levels = {
            agent: torch.ones(self.num_envs, device=self.device) * self.cfg.collaboration["trust_factor"]
            for agent in self.cfg.possible_agents
        }
        self.conflict_counter = torch.zeros(self.num_envs, device=self.device)
        
        # 任务跟踪
        self.task_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_in_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 约束相关变量（与MBRL对齐）
        self.safety_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_violating_constraint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        print(f"[INFO] MARL环境初始化完成，{self.num_envs}个环境")
        print(f"[INFO] 智能体: {self.cfg.possible_agents}")
        print(f"[INFO] 观测维度: human={self.cfg.observation_spaces['human']}, robot={self.cfg.observation_spaces['robot']}")
        
    def _setup_scene(self):
        """设置仿真场景"""
        # 创建Omni机器人（由两个智能体协作控制）
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        # 创建约束
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        # 创建地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)
        
        # 添加光照
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """物理步骤前应用动作"""
        # 处理智能体动作
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                action = torch.clamp(action, -1.0, 1.0)
                max_force = self.cfg.max_force[agent]
                forces = action * max_force * self.cfg.force_scale
                self.agent_actions[agent] = forces
        
        # 合成智能体力
        self.combined_forces = self._combine_agent_forces()
        
        # 应用力到末端执行器
        self._apply_forces_to_end_effector(self.combined_forces)
        
        # 更新人类仿真
        self._update_human_simulation()
        
        # 更新协作指标
        self._update_collaboration_metrics()
        
        # 更新约束信息
        self._update_safety_constraints()
        
    def _combine_agent_forces(self) -> torch.Tensor:
        """合成智能体力（MARL特有机制）"""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # 力分配比例
        human_ratio = self.cfg.force_sharing_ratio["human"]
        robot_ratio = self.cfg.force_sharing_ratio["robot"]
        
        # 信任调制
        human_trust = self.trust_levels["human"].unsqueeze(-1)
        robot_trust = self.trust_levels["robot"].unsqueeze(-1)
        
        total_trust = human_trust + robot_trust + 1e-6
        human_weight = (human_trust / total_trust) * human_ratio
        robot_weight = (robot_trust / total_trust) * robot_ratio
        
        # 合成力
        combined_forces = human_weight * human_forces + robot_weight * robot_forces
        
        # 交互耦合
        interaction_term = self.cfg.interaction_coupling * (human_forces + robot_forces) * 0.5
        combined_forces = combined_forces + interaction_term
        
        return combined_forces
        
    def _apply_forces_to_end_effector(self, forces: torch.Tensor) -> None:
        """应用力到末端执行器"""
        forces_reshaped = forces.unsqueeze(1)
        torques_reshaped = torch.zeros_like(forces_reshaped)
        
        body_ids = torch.tensor(
            [self.cfg.end_effector_body_id],
            dtype=torch.long,
            device=self.device
        )
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        self._omni_robot.set_external_force_and_torque(
            forces=forces_reshaped,
            torques=torques_reshaped,
            body_ids=body_ids,
            env_ids=env_ids
        )
        
    def _update_human_simulation(self) -> None:
        """更新人类行为仿真（与MBRL对齐）"""
        end_effector_pos = self._get_end_effector_position()
        
        self.human_reaction_timer += self.dt
        
        intention_update_period = 1.0 / self.cfg.human_dynamics["intention_update_rate"]
        update_mask = self.human_reaction_timer >= intention_update_period
        
        if update_mask.any():
            # 计算朝向目标的期望方向
            target_pos = self.trajectory_manager.get_current_target()
            target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
            
            target_direction = target_pos - end_effector_pos
            target_direction = target_direction / (torch.norm(target_direction, dim=-1, keepdim=True) + 1e-6)
            
            # 添加人类噪声
            noise_std = self.cfg.human_dynamics["noise_std"]
            noise = torch.randn_like(target_direction) * noise_std
            
            self.human_intention[update_mask] = (target_direction + noise)[update_mask]
            self.human_reaction_timer[update_mask] = 0.0
            
    def _update_collaboration_metrics(self) -> None:
        """更新协作指标（MARL特有）"""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # 计算力对齐
        force_diff = torch.norm(human_forces - robot_forces, dim=-1)
        conflict_threshold = self.cfg.collaboration["conflict_threshold"]
        
        conflicts = force_diff > conflict_threshold
        self.conflict_counter[conflicts] += 1
        self.conflict_counter[~conflicts] *= 0.9
        
        # 更新信任水平
        trust_decay = self.cfg.collaboration["trust_decay"]
        trust_recovery = self.cfg.collaboration["trust_recovery"]
        
        self.trust_levels["human"][conflicts] *= trust_decay
        self.trust_levels["robot"][conflicts] *= trust_decay
        
        self.trust_levels["human"][~conflicts] *= trust_recovery
        self.trust_levels["robot"][~conflicts] *= trust_recovery
        
        # 限制信任水平
        for agent in self.cfg.possible_agents:
            self.trust_levels[agent] = torch.clamp(self.trust_levels[agent], 0.1, 1.0)
            
    def _update_safety_constraints(self):
        """更新安全约束信息（与MBRL对齐）"""
        end_effector_pos = self._get_end_effector_position()
        self._compute_simplified_constraints(end_effector_pos)
        
    def _compute_simplified_constraints(self, current_positions: torch.Tensor):
        """简化约束模型（与MBRL对齐）"""
        constraint_center = torch.tensor([0.0, 0.15, 0.0], device=self.device)
        constraint_radius = 0.05
        
        for i in range(current_positions.shape[0]):
            pos = current_positions[i]
            diff = pos - constraint_center
            horizontal_dist = torch.norm(diff[:2])
            vertical_dist = torch.abs(diff[2])
            
            distance_to_constraint = torch.sqrt(horizontal_dist**2 + (vertical_dist * 2)**2)
            safety_distance = torch.clamp(distance_to_constraint - constraint_radius, min=0.0)
            
            if horizontal_dist > 1e-6:
                normal = diff / torch.norm(diff)
            else:
                normal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            
            self.safety_distances[i] = safety_distance
            self.constraint_normals[i] = normal
            self.is_violating_constraint[i] = safety_distance < self.cfg.collision_threshold
            
    def _apply_action(self) -> None:
        """应用处理过的动作"""
        pass
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """确定是否终止或截断（与MBRL对齐）"""
        end_effector_pos = self._get_end_effector_position()
        
        # 终止条件
        constraint_violated = self.is_violating_constraint.clone()
        fell_out = end_effector_pos[:, 2] < -0.01
        final_target_reached = torch.tensor([
            self.trajectory_manager.is_final_target_reached(end_effector_pos[i])
            for i in range(self.num_envs)
        ], device=self.device, dtype=torch.bool)
        
        terminated_condition = constraint_violated | fell_out | final_target_reached
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """计算每个智能体的奖励"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        # 更新目标点
        self.trajectory_manager.update_target(end_effector_pos)
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        rewards = {}
        
        # 基础奖励组件（与MBRL对齐）
        position_error = end_effector_pos - target_pos
        tracking_reward = -torch.sum(position_error**2, dim=-1)
        velocity_penalty = -torch.sum(end_effector_vel**2, dim=-1)
        safety_penalty = -self.safety_distances * 10.0
        collision_penalty = self.is_violating_constraint.float() * (-1.0)
        
        # 任务完成奖励
        distance_to_target = torch.norm(end_effector_pos - target_pos, dim=-1)
        target_reached = distance_to_target < self.cfg.target_reach_threshold
        completion_reward = target_reached.float()
        
        # 最终完成奖励
        final_completion = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            if self.trajectory_manager.is_final_target_reached(end_effector_pos[i]):
                final_completion[i] = 1.0
        
        for agent in self.cfg.possible_agents:
            reward_scales = self.cfg.reward_scales[agent]
            agent_reward = torch.zeros(self.num_envs, device=self.device)
            
            # 基础奖励
            agent_reward += tracking_reward * reward_scales["tracking_reward"]
            agent_reward += velocity_penalty * reward_scales["velocity_penalty"]
            agent_reward += safety_penalty * reward_scales["safety_penalty"]
            agent_reward += collision_penalty * reward_scales["collision_penalty"]
            agent_reward += completion_reward * reward_scales["completion_reward"]
            
            # 控制平滑性惩罚
            control_penalty = -torch.norm(self.agent_actions[agent], dim=-1) ** 2
            agent_reward += control_penalty * reward_scales["control_penalty"]
            
            # 协作奖励（MARL特有）
            collaboration_reward = self._calculate_collaboration_reward(agent)
            agent_reward += collaboration_reward * reward_scales["collaboration_reward"]
            
            # 智能体特有奖励
            if agent == "human":
                intention_reward = self._calculate_intention_alignment_reward()
                agent_reward += intention_reward * reward_scales["intention_alignment"]
            else:  # robot
                adaptation_reward = self._calculate_adaptation_reward()
                agent_reward += adaptation_reward * reward_scales["adaptation_reward"]
            
            # 最终完成奖励
            agent_reward += final_completion * 50.0
            
            rewards[agent] = torch.clamp(agent_reward, -100.0, 75.0)
            
        return rewards
        
    def _calculate_collaboration_reward(self, agent: str) -> torch.Tensor:
        """计算协作奖励"""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # 力对齐奖励
        force_alignment = -torch.norm(human_forces - robot_forces, dim=-1)
        alignment_reward = torch.exp(force_alignment * 0.5)
        
        # 信任奖励
        trust_reward = self.trust_levels[agent] * 2.0
        
        return alignment_reward + trust_reward
        
    def _calculate_intention_alignment_reward(self) -> torch.Tensor:
        """计算意图对齐奖励（人类特有）"""
        human_intent = self.human_intention
        robot_actions = self.agent_actions["robot"]
        
        intent_norm = torch.norm(human_intent, dim=-1, keepdim=True) + 1e-6
        action_norm = torch.norm(robot_actions, dim=-1, keepdim=True) + 1e-6
        
        intent_normalized = human_intent / intent_norm
        action_normalized = robot_actions / action_norm
        
        alignment = torch.sum(intent_normalized * action_normalized, dim=-1)
        return torch.clamp(alignment, 0.0, 1.0) * 2.0
        
    def _calculate_adaptation_reward(self) -> torch.Tensor:
        """计算适应性奖励（机器人特有）"""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        robot_adaptation = torch.norm(robot_forces - human_forces, dim=-1)
        return torch.exp(-robot_adaptation * 0.5)
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """获取智能体观测"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        # 更新目标点
        self.trajectory_manager.update_target(end_effector_pos)
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # 约束距离（与MBRL对齐）
        constraint_distances = self._calculate_constraint_distances(end_effector_pos)
        
        observations = {}
        
        for agent in self.cfg.possible_agents:
            # 基础观测（与MBRL对齐）：12D
            base_obs = torch.cat([
                end_effector_pos,      # 3D 位置
                end_effector_vel,      # 3D 速度  
                target_pos,            # 3D 目标位置
                constraint_distances,  # 3D 约束距离（简化）
            ], dim=-1)
            
            # 智能体特有观测：6D
            if agent == "human":
                # 人类观测：自身意图 + 信任信息
                trust_info = torch.stack([
                    self.trust_levels["human"],
                    self.trust_levels["robot"],
                    self.conflict_counter * 0.1,
                ], dim=-1)
                
                agent_specific_obs = torch.cat([
                    self.human_intention,  # 3D 人类意图
                    trust_info,           # 3D 信任信息
                ], dim=-1)
                
            else:  # robot
                # 机器人观测：人类动作 + 信任信息
                trust_info = torch.stack([
                    self.trust_levels["human"],
                    self.trust_levels["robot"],
                    self.conflict_counter * 0.1,
                ], dim=-1)
                
                agent_specific_obs = torch.cat([
                    self.agent_actions["human"],  # 3D 人类力
                    trust_info,                   # 3D 信任信息
                ], dim=-1)
            
            # 完整观测：基础12D + 特有6D = 18D
            obs = torch.cat([base_obs, agent_specific_obs], dim=-1)
            obs = torch.clamp(obs, -10.0, 10.0)
            observations[agent] = obs
            
        return observations
        
    def _calculate_constraint_distances(self, end_effector_pos: torch.Tensor) -> torch.Tensor:
        """计算约束距离（简化为3D）"""
        x, y, z = end_effector_pos[:, 0], end_effector_pos[:, 1], end_effector_pos[:, 2]
        
        # 与约束中心的距离
        constraint_center = torch.tensor([0.0, 0.15, 0.0], device=self.device)
        distance_to_center = torch.norm(end_effector_pos[:, :2] - constraint_center[:2].unsqueeze(0), dim=-1)
        
        # 到目标高度的距离
        target_pos = self.trajectory_manager.get_current_target()
        distance_to_target_z = torch.abs(end_effector_pos[:, 2] - target_pos[2])
        
        # 安全距离
        safety_distance = self.safety_distances
        
        return torch.stack([
            distance_to_center,
            distance_to_target_z,
            safety_distance
        ], dim=-1)
        
    def _get_states(self) -> torch.Tensor:
        """获取全局状态（用于集中式训练）"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # 全局状态：24D
        global_state = torch.cat([
            end_effector_pos,                     # 3D 位置
            end_effector_vel,                     # 3D 速度
            target_pos,                           # 3D 目标位置
            self.agent_actions["human"],          # 3D 人类力
            self.agent_actions["robot"],          # 3D 机器人力
            self.combined_forces,                 # 3D 合成力
            self.human_intention,                 # 3D 人类意图
            torch.stack([
                self.trust_levels["human"],
                self.trust_levels["robot"],
                self.conflict_counter * 0.1,
            ], dim=-1)  # 3D 协作状态
        ], dim=-1)
        
        return global_state
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # 设置初始关节位置以到达起始位置 [-0.2, 0.15, 0.03]（与MBRL对齐）
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = 0.0   # waist
        joint_pos[:, 1] = 0.0   # shoulder  
        joint_pos[:, 2] = 0.0   # elbow
        joint_pos[:, 3] = 4.0   # yaw
        joint_pos[:, 4] = 1.2   # pitch - stylus upright
        joint_pos[:, 5] = 0.0   # roll
        
        # 添加小噪声
        joint_noise = sample_uniform(-0.1, 0.1, (num_resets, 6), self.device)
        joint_pos += joint_noise
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置智能体跟踪变量
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
            self.trust_levels[agent][env_ids] = self.cfg.collaboration["trust_factor"]
        
        # 重置任务跟踪
        self.task_completed[env_ids] = False
        self.steps_in_target[env_ids] = 0
        self.conflict_counter[env_ids] = 0.0
        
        # 重置人类仿真
        self.human_intention[env_ids] = 0.0
        self.human_reaction_timer[env_ids] = 0.0
        self.combined_forces[env_ids] = 0.0
        
        # 重置约束相关
        self.safety_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.is_violating_constraint[env_ids] = False
        
        # 重置轨迹管理器
        self.trajectory_manager.reset_trajectory()
        
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        return self._omni_robot.data.body_link_state_w[..., self.cfg.end_effector_body_id, :3]
        
    def _get_end_effector_velocity(self):
        """获取末端执行器线速度"""
        return self._omni_robot.data.body_link_state_w[..., self.cfg.end_effector_body_id, 7:10]