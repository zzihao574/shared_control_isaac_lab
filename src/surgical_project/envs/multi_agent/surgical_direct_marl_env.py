# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical Direct MARL Environment - Optimized Version"""

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym
from typing import Any, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class CompleteConstraintChecker:
    """约束状态检测类"""
    
    def __init__(self, device, collision_threshold=0.001):
        self.device = device
        self.collision_threshold = collision_threshold
        
        from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
        self.physics_attachment_interface = acquire_physx_attachment_interface()
        self.physics_scene_query_interface = acquire_physx_scene_query_interface()
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """批量分析约束状态"""
        from carb._carb import Float3
        
        num_envs = stylus_positions.shape[0]
        stylus_world_positions = stylus_positions + env_base_positions
        
        batch_results = {
            'distances': torch.ones(num_envs, device=self.device),  # 默认安全距离
            'closest_points': torch.zeros(num_envs, 3, device=self.device),
            'normal_vectors': torch.ones(num_envs, 3, device=self.device),
            'is_overlapping': torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            'is_inside': torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        }
        
        for env_idx in range(num_envs):
            try:
                pos = stylus_world_positions[env_idx].cpu().numpy()
                query_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
                constraint_path = f"/World/envs/env_{env_idx:05d}/Constraint"
                
                result = self.physics_attachment_interface.get_closest_points([query_point], constraint_path)
                
                if result and 'closest_points' in result and result['closest_points']:
                    closest_pt = result['closest_points'][0]
                    closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
                    query_pos = np.array([query_point.x, query_point.y, query_point.z])
                    
                    distance = float(np.linalg.norm(query_pos - closest_pos))
                    batch_results['distances'][env_idx] = distance
                    batch_results['closest_points'][env_idx] = torch.tensor(closest_pos, device=self.device)
                    batch_results['is_overlapping'][env_idx] = distance < self.collision_threshold
                    
            except Exception:
                pass  # 使用默认值
        
        return batch_results


class TrajectoryManager:
    """轨迹管理器"""
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.setpoint_interval = traj['setpoint_interval']
        self.switch_threshold = traj['switch_threshold']
        
        self._generate_setpoints(params)
        self.current_setpoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.setpoints_tensor = torch.stack(self.setpoints_local)
        self.num_setpoints = len(self.setpoints_local)
        
    def _generate_setpoints(self, params):
        """生成轨迹设置点"""
        constraint = params['constraint_geometry']
        y_pos_range = constraint['y_range_positive']
        y_neg_range = constraint['y_range_negative']
        
        total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        num_setpoints = int(total_distance / self.setpoint_interval) + 1
        
        self.setpoints_local = []
        direction = (self.end_pos_local - self.start_pos_local) / torch.norm(self.end_pos_local - self.start_pos_local)
        
        for i in range(num_setpoints + 1):
            setpoint = self.end_pos_local.clone() if i == num_setpoints else self.start_pos_local + direction * (i * self.setpoint_interval)
            
            y_coord = setpoint[1].item()
            in_constraint = (y_pos_range[0] <= y_coord <= y_pos_range[1]) or (y_neg_range[0] <= y_coord <= y_neg_range[1])
            
            if not in_constraint:
                self.setpoints_local.append(setpoint)
        
    def get_current_setpoint_local(self) -> torch.Tensor:
        """获取当前设置点（局部坐标）"""
        indices = torch.clamp(self.current_setpoint_idx, 0, self.num_setpoints - 1)
        return self.setpoints_tensor[indices]
        
    def update_setpoint(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """更新设置点"""
        current_indices = self.current_setpoint_idx
        indices_clamped = torch.clamp(current_indices, 0, self.num_setpoints - 1)
        current_setpoints = self.setpoints_tensor[indices_clamped]
        
        distances = torch.norm(current_pos_local - current_setpoints, dim=-1)
        should_update = (distances < self.switch_threshold) & (current_indices < self.num_setpoints - 1)
        self.current_setpoint_idx[should_update] += 1
        
        return should_update
        
    def reset_trajectory(self, env_ids: torch.Tensor = None):
        """重置轨迹"""
        if env_ids is None:
            self.current_setpoint_idx.fill_(0)
        else:
            self.current_setpoint_idx[env_ids] = 0
        
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """检查是否到达终点"""
        at_final = (self.current_setpoint_idx >= self.num_setpoints - 1)
        final_setpoint = self.setpoints_local[-1]
        distances_to_final = torch.norm(current_pos_local - final_setpoint.unsqueeze(0), dim=-1)
        return at_final & (distances_to_final < self.switch_threshold)


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """人机协作手术MARL环境"""
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # 加载参数
        self.params = self._load_training_params()
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._load_yaml_parameters()
        
        # 环境基础位置
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 轨迹管理
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        # 智能体动作
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # 状态缓存（避免重复计算）
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.joint_pos_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        self.joint_vel_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        
        # 力和约束状态
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t = torch.zeros(self.num_envs, device=self.device)
        self.is_violating_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 固定关节配置
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
        
        self.stylus_body_idx = None
        self.constraint_checker = CompleteConstraintChecker(self.device, self.collision_threshold)
        
        # Gymnasium兼容性
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_spaces[agent],), dtype=np.float32)
            for agent in self.cfg.possible_agents
        })
        
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-10.0, high=10.0, shape=(self.cfg.observation_spaces[agent],), dtype=np.float32)
            for agent in self.cfg.possible_agents
        })
        
    def _load_training_params(self) -> dict:
        """加载训练参数"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_yaml_parameters(self):
        """加载YAML参数"""
        constraints = self.params['constraints']
        self.max_cartesian_vel = constraints['max_cartesian_velocity']
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        self.safety_margin = self.params['reward_parameters']['cbf_parameters']['safety_margin']
        self.constraint_center = torch.tensor(
            self.params['constraint_geometry']['center'], 
            device=self.device, dtype=torch.float32
        )
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
        self.cbf_gamma = self.params['reward_parameters']['cbf_parameters']['gamma']
        self.cbf_epsilon = self.params['reward_parameters']['cbf_parameters']['epsilon']
        
    def _setup_scene(self):
        """设置场景"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        self.scene.clone_environments(copy_from_source=False)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """场景创建后设置"""
        super()._setup_post_scene_creation()
        self._initialize_body_indices()
        
        if hasattr(self, '_omni_robot'):
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
        
    def _initialize_body_indices(self):
        """初始化body索引"""
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        search_patterns = ['stylus', 'tip', 'end_effector', 'link6', 'end', 'tool']
        for pattern in search_patterns:
            for i, name in enumerate(self._omni_robot.body_names):
                if pattern in name.lower():
                    self.stylus_body_idx = i
                    return
        
        # 默认使用最后一个body
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """物理步骤前处理"""
        # 处理动作
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                action = torch.clamp(action, -1.0, 1.0)
                force_scale = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = action * force_scale
        
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # 应用力
        if self.stylus_body_idx is not None:
            total_forces = self.robot_forces_t + self.human_forces_t
            stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]
            forces_local = quat_rotate_inverse(stylus_quat, total_forces)
            
            self._omni_robot.set_external_force_and_torque(
                forces_local.unsqueeze(1), 
                torch.zeros_like(forces_local.unsqueeze(1)),
                body_ids=[self.stylus_body_idx]
            )
        
        # 固定末端关节
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        joint_vel[:, :3] = torch.clamp(joint_vel[:, :3], -10.0, 10.0)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """应用动作并更新状态缓存"""
        self._omni_robot.write_data_to_sim()
        
        # 更新t+1状态缓存（避免重复计算）
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        joint_pos = self._omni_robot.data.joint_pos
        joint_vel = self._omni_robot.data.joint_vel
        
        # 确保6维
        if joint_pos.shape[-1] < 6:
            padding = torch.zeros(self.num_envs, 6 - joint_pos.shape[-1], device=self.device)
            self.joint_pos_t1 = torch.cat([joint_pos, padding], dim=-1)
            self.joint_vel_t1 = torch.cat([joint_vel, padding], dim=-1)
        else:
            self.joint_pos_t1 = joint_pos[..., :6]
            self.joint_vel_t1 = joint_vel[..., :6]
        
        # 更新约束状态
        constraint_results = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, self.env_base_positions
        )
        self.safety_distances_t = constraint_results['distances']
        self.is_violating_t = constraint_results['is_overlapping']
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """获取观测（使用缓存的状态）"""
        stylus_vel_constrained = torch.clamp(self.stylus_vel_t1, -self.max_cartesian_vel, self.max_cartesian_vel)
        
        constraint_distances = torch.stack([
            torch.norm(self.stylus_pos_t1 - self.constraint_center.unsqueeze(0).expand(self.num_envs, -1), dim=-1),
            self.safety_distances_t,
            torch.abs(self.stylus_pos_t1[:, 2] - self.constraint_center[2])
        ], dim=-1)
        
        obs = torch.cat([
            self.stylus_pos_t1,
            stylus_vel_constrained,
            self.joint_pos_t1,
            self.joint_vel_t1,
            constraint_distances,
        ], dim=-1)
        
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = torch.clamp(obs, -10.0, 10.0)
            
        return observations
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """计算奖励"""
        # 轨迹跟踪
        self.trajectory_manager.update_setpoint(self.stylus_pos_t1)
        current_setpoints = self.trajectory_manager.get_current_setpoint_local()
        
        pos_error = self.stylus_pos_t1 - current_setpoints
        position_tracking = -torch.sum(pos_error**2, dim=-1)
        velocity_regulation = -torch.sum(self.stylus_vel_t1**2, dim=-1)
        
        # CBF障碍函数
        s = torch.clamp(self.safety_distances_t, min=self.cbf_epsilon)
        gamma_s = self.cbf_gamma * s
        cbf_values = -torch.log(torch.clamp(gamma_s / (gamma_s + 1.0), min=self.cbf_epsilon, max=1.0 - self.cbf_epsilon))
        
        # 惩罚项
        z_violation = torch.where(
            self.stylus_pos_t1[:, 2] < self.min_z_pos,
            torch.full_like(self.stylus_pos_t1[:, 2], -500.0),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )
        
        collision_penalty = self.is_violating_t.float() * self.params['reward_parameters']['collision_penalty']
        
        # 完成奖励
        final_setpoint = self.trajectory_manager.setpoints_local[-1].unsqueeze(0).expand(self.num_envs, -1)
        distance_to_final = torch.norm(self.stylus_pos_t1 - final_setpoint, dim=-1)
        completion_reward = torch.where(
            distance_to_final < self.params['reward_parameters']['completion_threshold'],
            torch.full_like(distance_to_final, self.params['reward_parameters']['completion_reward']),
            torch.zeros_like(distance_to_final)
        )
        
        # 力冲突
        dot_product = torch.sum(self.human_forces_t * self.robot_forces_t, dim=-1)
        human_norm = torch.norm(self.human_forces_t, dim=-1)
        robot_norm = torch.norm(self.robot_forces_t, dim=-1)
        cos_angle = dot_product / (human_norm * robot_norm + 1e-6)
        
        force_magnitude = human_norm + robot_norm
        conflict_threshold = self.params['reward_parameters']['collaboration_parameters']['force_conflict_threshold']
        force_conflict = torch.where(
            (cos_angle < -0.5) & (force_magnitude > conflict_threshold),
            cos_angle * self.params['reward_parameters']['collaboration_parameters']['conflict_penalty_scale'],
            torch.zeros_like(cos_angle)
        )
        
        # 计算奖励
        rewards = {}
        
        robot_weights = self.params['reward_parameters']['robot_weights']
        robot_control_penalty = -torch.sum(self.robot_forces_t**2, dim=-1)
        rewards["robot"] = (
            position_tracking * robot_weights['position_tracking'] +
            velocity_regulation * robot_weights['velocity_regulation'] +
            cbf_values * robot_weights['obstacle_distance'] +
            robot_control_penalty * robot_weights['control_input'] +
            force_conflict + z_violation + collision_penalty + completion_reward
        )
        
        human_weights = self.params['reward_parameters']['human_weights']
        human_force_penalty = -torch.sum(self.human_forces_t**2, dim=-1)
        rewards["human"] = (
            position_tracking * human_weights['position_tracking'] +
            velocity_regulation * human_weights['velocity_regulation'] +
            cbf_values * human_weights['obstacle_distance'] +
            human_force_penalty * human_weights['force_input'] +
            force_conflict + z_violation + collision_penalty + completion_reward
        )
        
        self.extras["log"] = {
            "robot_reward": rewards["robot"].mean().item(),
            "human_reward": rewards["human"].mean().item(),
            "safety_distance": self.safety_distances_t.mean().item(),
        }
        
        return rewards
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """确定终止条件"""
        constraint_violated = self.is_violating_t
        fell_out = self.stylus_pos_t1[:, 2] < self.min_z_pos
        vel_exceeded = torch.any(torch.abs(self.stylus_vel_t1) > self.max_cartesian_vel, dim=1)
        
        joint_violated = torch.any(
            (self.joint_pos_t1[:, :3] < self.joint_lower_limits[:3] - 0.01) |
            (self.joint_pos_t1[:, :3] > self.joint_upper_limits[:3] + 0.01),
            dim=1
        )
        
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        terminated_condition = constraint_violated | fell_out | vel_exceeded | joint_violated | final_reached
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # 初始关节配置
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        for i, joint_name in enumerate(['waist', 'shoulder', 'elbow', 'yaw', 'pitch', 'roll']):
            joint_pos[:, i] = self.params['initial_conditions']['joint_positions'][joint_name]
        
        joint_pos += sample_uniform(-0.05, 0.05, (num_resets, 6), self.device)
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置状态
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t[env_ids] = 0.0
        self.is_violating_t[env_ids] = False
        
        self.trajectory_manager.reset_trajectory(env_ids)
        
    def _get_stylus_position(self):
        """获取stylus位置"""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        base_pos = self._omni_robot.data.root_link_pos_w
        ee_pos = self._omni_robot.data.body_link_pos_w[:, self.stylus_body_idx, :]
        return ee_pos - base_pos
    
    def _get_stylus_velocity(self):
        """获取stylus速度"""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        return self._omni_robot.data.body_link_lin_vel_w[:, self.stylus_body_idx, :]