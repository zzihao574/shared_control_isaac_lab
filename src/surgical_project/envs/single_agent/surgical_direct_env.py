# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术直接环境 - 严格按论文方程实现"""

from __future__ import annotations

import torch
import numpy as np
import math
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .surgical_direct_env_cfg import SurgicalDirectEnvCfg


class TrajectoryManager:
    """简单轨迹管理器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.current_index = 0
        self.trajectory_points = None
        self.trajectory_velocities = None
        self._generate_default_trajectory()
        
    def _generate_default_trajectory(self):
        """生成简单螺旋轨迹"""
        num_points = 200
        turns = 2.0
        radius_range = (0.01, 0.04)  # 1-4cm
        height_range = (0.002, 0.08)  # 2-80mm
        
        t = np.linspace(0, turns * 2 * np.pi, num_points)
        
        r_min, r_max = radius_range
        h_min, h_max = height_range
        
        radius = np.linspace(r_min, r_max, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.linspace(h_min, h_max, num_points)
        
        self.trajectory_points = torch.tensor(np.column_stack([x, y, z]), 
                                            dtype=torch.float32, device=self.device)
        
        velocities = torch.diff(self.trajectory_points, dim=0, prepend=self.trajectory_points[:1])
        self.trajectory_velocities = velocities
        
        print(f"[INFO] Generated spiral trajectory with {num_points} points")
        
    def get_current_target(self):
        """获取当前轨迹目标位置和速度"""
        if self.trajectory_points is None:
            return torch.zeros(3, device=self.device), torch.zeros(3, device=self.device)
            
        target_pos = self.trajectory_points[self.current_index]
        target_vel = self.trajectory_velocities[self.current_index]
        
        return target_pos, target_vel
        
    def advance_trajectory(self, step_size: int = 1):
        """推进到下一轨迹点"""
        if self.trajectory_points is not None:
            self.current_index = min(
                self.current_index + step_size,
                len(self.trajectory_points) - 1
            )
            
    def reset_trajectory(self):
        """重置轨迹到起始点"""
        self.current_index = 0


class SurgicalDirectEnv(DirectRLEnv):
    """论文对齐的手术直接环境"""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化手术环境"""
        super().__init__(cfg, render_mode, **kwargs)
        
        # 时间步长
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # 初始化跟踪变量
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 约束几何参数
        self.constraint_inner_radius_min = self.cfg.constraint_inner_radius_min
        self.constraint_inner_radius_max = self.cfg.constraint_inner_radius_max
        self.constraint_height = self.cfg.constraint_height
        
        # 约束计算状态
        self.constraint_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_overlapping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 任务完成跟踪
        self.task_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 轨迹管理器
        self.trajectory_manager = TrajectoryManager(self.device)
        
        # 人体工作空间参数
        self.human_workspace_center = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.human_workspace_radius = 0.2  # 20cm半径
        
        # 论文成本函数权重
        self.Q1_weight = self.cfg.Q1_weight
        self.Q2_weight = self.cfg.Q2_weight  
        self.Q3_weight = self.cfg.Q3_weight
        self.R_weight = self.cfg.R_weight
        
        print(f"[INFO] Surgical environment initialized (Paper-aligned):")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation space: {self.cfg.observation_space}D") 
        print(f"  - Action space: {self.cfg.action_space}D")
        print(f"  - Paper state space: {self.cfg.state_space}D (z = [x, ẋ, f]^T)")
        print(f"  - Cost weights: Q1={self.Q1_weight}, Q2={self.Q2_weight}, Q3={self.Q3_weight}, R={self.R_weight}")
        
    def _setup_scene(self):
        """设置仿真场景"""
        # 创建手术刀
        self._scalpel = RigidObject(self.cfg.scalpel)
        self.scene.rigid_objects["scalpel"] = self._scalpel
        
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
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步进前应用动作"""
        # 限制动作范围
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 缩放动作到力值
        forces = actions * self.cfg.max_force * self.cfg.force_scale
        
        # 存储动作用于平滑性计算
        self.previous_actions = actions.clone()
        
        # 应用力到手术刀
        self._apply_forces_to_scalpel(forces)
        
        # 更新约束计算
        self._update_constraints()
        
        # 每10步推进轨迹
        if self.episode_length_buf[0] % 10 == 0:
            self.trajectory_manager.advance_trajectory()
        
    def _apply_forces_to_scalpel(self, forces: torch.Tensor) -> None:
        """对手术刀球心施加力"""
        forces_reshaped = forces.unsqueeze(1)  # [num_envs, 1, 3]
        torques_reshaped = torch.zeros_like(forces_reshaped)  # 无扭矩
        
        self._scalpel.set_external_force_and_torque(
            forces_reshaped,
            torques_reshaped,
            body_ids=None
        )
        
    def _update_constraints(self):
        """使用简化几何更新约束信息"""
        scalpel_pos = self._scalpel.data.root_pos_w
        
        x, y, z = scalpel_pos[..., 0], scalpel_pos[..., 1], scalpel_pos[..., 2]
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # 锥形线性插值
        height_ratio = torch.clamp(z / self.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.constraint_inner_radius_min + 
            height_ratio * (self.constraint_inner_radius_max - self.constraint_inner_radius_min)
        )
        
        # 到内壁距离
        self.constraint_distances = torch.abs(radial_dist - inner_radius_at_z)
        
        # 法向量（指向约束外）
        normal_x = torch.where(radial_dist > 1e-6, x / radial_dist, torch.ones_like(x))
        normal_y = torch.where(radial_dist > 1e-6, y / radial_dist, torch.zeros_like(y))
        normal_z = torch.zeros_like(z)
        
        self.constraint_normals = torch.stack([normal_x, normal_y, normal_z], dim=-1)
        
        # 简单重叠检查
        sphere_radius = 0.002  # 2mm
        self.is_overlapping = self.constraint_distances < sphere_radius
        
    def _apply_action(self) -> None:
        """应用处理后的动作到环境"""
        pass
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """确定剧集是否终止或截断"""
        # 碰撞终止
        terminated = self.is_overlapping.clone()
        
        # 检查是否掉落过低
        scalpel_pos = self._scalpel.data.root_pos_w
        fell_out = scalpel_pos[..., 2] < -0.01  # 低于-10mm
        terminated = terminated | fell_out
        
        # 截断条件
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _get_rewards(self) -> torch.Tensor:
        """基于论文方程(13)的成本函数计算奖励"""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # 获取当前轨迹目标
        target_pos, _ = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # 论文方程(13): r = (x-x_d)^T Q_1(x-x_d) + ẋ^T Q_2 ẋ + f^T Q_3 f + u^T R u
        
        # 1. 位置跟踪成本: (x - x_d)^T Q_1 (x - x_d)
        position_error = scalpel_pos - target_pos
        tracking_cost = torch.sum(position_error**2, dim=-1) * self.Q1_weight
        
        # 2. 速度调节成本: ẋ^T Q_2 ẋ
        velocity_cost = torch.sum(scalpel_vel**2, dim=-1) * self.Q2_weight
        
        # 3. 力调节成本: f^T Q_3 f
        force_cost = torch.sum(self.interaction_forces**2, dim=-1) * self.Q3_weight
        
        # 4. 控制输入成本: u^T R u
        control_cost = torch.sum(self.previous_actions**2, dim=-1) * self.R_weight
        
        # 总成本
        total_cost = tracking_cost + velocity_cost + force_cost + control_cost
        
        # 转换为奖励（强化学习最大化奖励 = 最小化成本）
        paper_reward = -total_cost
        
        # 添加安全约束奖励（避免碰撞）
        collision_penalty = self.is_overlapping.float() * (-10.0)
        
        # 工作空间约束奖励
        workspace_distance = torch.norm(scalpel_pos[..., :2] - self.human_workspace_center[:2], dim=-1)
        workspace_penalty = torch.clamp(workspace_distance - self.human_workspace_radius, 0, 0.1) * (-5.0)
        
        # 最终奖励
        total_reward = paper_reward + collision_penalty + workspace_penalty
        
        # 限制奖励范围
        total_reward = torch.clamp(total_reward, -20.0, 5.0)
        
        # 存储奖励组件用于记录
        self.extras["log"] = {
            "tracking_cost": tracking_cost.mean().item(),
            "velocity_cost": velocity_cost.mean().item(),
            "force_cost": force_cost.mean().item(),
            "control_cost": control_cost.mean().item(),
            "total_paper_cost": total_cost.mean().item(),
            "paper_reward": paper_reward.mean().item(),
            "collision_penalty": collision_penalty.mean().item(),
            "workspace_penalty": workspace_penalty.mean().item(),
            "total_reward": total_reward.mean().item(),
            "constraint_distance": self.constraint_distances.mean().item(),
            "trajectory_index": self.trajectory_manager.current_index,
            "overlap_rate": self.is_overlapping.float().mean().item(),
            "scalpel_height": scalpel_pos[..., 2].mean().item(),
            "velocity_norm": torch.norm(scalpel_vel, dim=-1).mean().item(),
        }
        
        return total_reward
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        # 重置手术刀到人体工作空间内的随机起始位置
        num_resets = len(env_ids)
        
        # 工作空间内随机位置（约束上方）
        angles = sample_uniform(0, 2*math.pi, (num_resets,), self.device)
        radii = sample_uniform(0.02, 0.15, (num_resets,), self.device)  # 2-15cm从中心
        
        scalpel_pos = torch.zeros((num_resets, 3), device=self.device)
        scalpel_pos[:, 0] = radii * torch.cos(angles)
        scalpel_pos[:, 1] = radii * torch.sin(angles)
        scalpel_pos[:, 2] = sample_uniform(0.015, 0.025, (num_resets,), self.device)  # 15-25mm高度
        
        # 重置速度
        scalpel_vel = torch.zeros((num_resets, 3), device=self.device)
        
        # 创建姿态（位置+四元数）
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        pose = torch.cat([scalpel_pos, quaternion], dim=-1)
        
        # 创建速度（线性+角速度）
        velocity = torch.cat([scalpel_vel, torch.zeros_like(scalpel_vel)], dim=-1)
        
        # 应用重置
        self._scalpel.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._scalpel.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        
        # 重置跟踪变量
        self.task_completed[env_ids] = False
        self.previous_actions[env_ids] = 0.0
        self.interaction_forces[env_ids] = 0.0
        self.constraint_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.is_overlapping[env_ids] = False
        
        # 重置轨迹
        self.trajectory_manager.reset_trajectory()
        
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取论文对齐的环境观测 - 修改为包含论文标准状态"""
        scalpel_pos = self._scalpel.data.root_pos_w      # x ∈ R^3
        scalpel_vel = self._scalpel.data.root_lin_vel_w  # ẋ ∈ R^3
        
        # 获取当前轨迹目标
        target_pos, _ = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)  # x_d ∈ R^3
        
        # 估计交互力（简化方法）
        # 在真实应用中，这应该来自力/扭矩传感器
        if hasattr(self, 'previous_scalpel_vel'):
            # 基于加速度估计力（简化）
            acceleration = (scalpel_vel - self.previous_scalpel_vel) / self.dt
            estimated_force = acceleration * 0.1  # 简化质量假设
            self.interaction_forces = torch.clamp(estimated_force, -5.0, 5.0)
        
        self.previous_scalpel_vel = scalpel_vel.clone()
        
        # 论文标准观测：包含论文状态 z = [x, ẋ, f]^T 和期望轨迹 x_d
        # 观测维度: pos(3) + vel(3) + force(3) + desired_pos(3) = 12D
        obs = torch.cat([
            scalpel_pos,           # [num_envs, 3] 当前位置 x
            scalpel_vel,           # [num_envs, 3] 当前速度 ẋ
            self.interaction_forces, # [num_envs, 3] 交互力 f
            target_pos,            # [num_envs, 3] 期望位置 x_d
        ], dim=-1)  # [num_envs, 12]
        
        # 限制观测数值稳定性
        obs = torch.clamp(obs, -10.0, 10.0)
        
        # 验证观测维度
        assert obs.shape[-1] == self.cfg.observation_space, \
            f"Observation dimension mismatch: {obs.shape[-1]} != {self.cfg.observation_space}"
        
        return {"policy": obs}