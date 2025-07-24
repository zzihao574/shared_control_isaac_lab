# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术直接环境 - 使用Omni haptic device的人机共享控制"""

from __future__ import annotations

import torch
import numpy as np
import math
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
from carb._carb import Float3
import omni

from .surgical_direct_env_cfg import SurgicalDirectEnvCfg


class TrajectoryManager:
    """轨迹管理器 - 直线轨迹"""
    
    def __init__(self, device: torch.device, start_point: tuple, end_point: tuple, num_points: int = 200):
        self.device = device
        self.current_index = 0
        self.start_point = start_point
        self.end_point = end_point
        self.num_points = num_points
        self.trajectory_points = None
        self._generate_trajectory()
        
    def _generate_trajectory(self):
        """生成直线轨迹从start_point到end_point"""
        start = np.array(self.start_point)
        end = np.array(self.end_point)
        
        t = np.linspace(0, 1, self.num_points)
        trajectory_points = np.array([start + t_val * (end - start) for t_val in t])
        
        self.trajectory_points = torch.tensor(trajectory_points, 
                                            dtype=torch.float32, device=self.device)
        
        print(f"[INFO] Linear trajectory generated: {self.num_points} points")
        print(f"  Start: {self.start_point}")
        print(f"  End: {self.end_point}")
        
    def get_current_target(self):
        """获取当前轨迹目标位置"""
        if self.trajectory_points is None:
            return torch.zeros(3, device=self.device)
            
        return self.trajectory_points[self.current_index]
        
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
        
    def get_progress(self):
        """获取轨迹进度 (0-1)"""
        if self.trajectory_points is None:
            return 0.0
        return self.current_index / (len(self.trajectory_points) - 1)


class SurgicalDirectEnv(DirectRLEnv):
    """论文对齐的手术直接环境 - 人机共享控制"""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化手术环境"""
        self._is_closed = False
        super().__init__(cfg, render_mode, **kwargs)
        
        # 获取物理查询接口
        self.pai = acquire_physx_attachment_interface()
        self.psqi = acquire_physx_scene_query_interface()
        
        # 时间步长
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # 初始化跟踪变量
        self.previous_robot_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.human_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 约束计算状态
        self.constraint_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_closest_points = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_overlapping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 轨迹管理器
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            start_point=self.cfg.trajectory_start_point,
            end_point=self.cfg.trajectory_end_point,
            num_points=self.cfg.trajectory_points_count
        )
        
        # 人体工作空间参数
        self.human_workspace_radius = 1.0
        
        # 论文成本函数权重
        self.Q1_weight = self.cfg.Q1_weight
        self.Q2_weight = self.cfg.Q2_weight  
        self.Q3_weight = self.cfg.Q3_weight
        self.R_weight = self.cfg.R_weight
        
        # 末端执行器参数
        self.end_effector_body_id = self.cfg.end_effector_body_id
        
        print(f"[INFO] Surgical environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation space: {self.cfg.observation_space}D") 
        print(f"  - Action space: {self.cfg.action_space}D")
        print(f"  - End effector body ID: {self.end_effector_body_id}")
        print(f"  - Linear trajectory: {self.cfg.trajectory_start_point} -> {self.cfg.trajectory_end_point}")
        
    def _setup_scene(self):
        """设置仿真场景"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        self.scene.clone_environments(copy_from_source=False)
        
        print(f"[INFO] Scene setup complete with {self.num_envs} environments")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步进前应用动作"""
        robot_actions = torch.clamp(actions, -1.0, 1.0)
        self.robot_forces = robot_actions * self.cfg.max_robot_force * self.cfg.force_scale
        self.previous_robot_actions = robot_actions.clone()
        
        self._simulate_human_input()
        self._compute_shared_control_forces()
        self._apply_forces_to_end_effector()
        self._update_constraints()
            
    def _simulate_human_input(self):
        """模拟人类输入力"""
        self.human_forces = torch.zeros_like(self.human_forces)
        
    def _compute_shared_control_forces(self):
        """计算人机共享控制的总力"""
        alpha = self.cfg.robot_action_weight
        self.total_interaction_forces = (alpha * self.robot_forces + 
                                       (1 - alpha) * self.human_forces)
        
    def _apply_forces_to_end_effector(self):
        """对末端执行器施加总的交互力"""
        body_ids = torch.full((self.num_envs,), self.end_effector_body_id, 
                             dtype=torch.long, device=self.device)
        
        forces_reshaped = self.total_interaction_forces.unsqueeze(1)
        torques_reshaped = torch.zeros_like(forces_reshaped)
        
        self._omni_robot.set_external_force_and_torque(
            forces_reshaped,
            torques_reshaped,
            body_ids=body_ids.unsqueeze(1)
        )
        
    def _compute_constraint_distance_and_normal(self, current_positions):
        """使用物理查询API计算到约束表面的距离和法向量"""
        batch_size = current_positions.shape[0]
        
        # 初始化
        self.constraint_distances = torch.zeros(batch_size, device=self.device)
        self.constraint_normals = torch.zeros(batch_size, 3, device=self.device)
        self.constraint_closest_points = torch.zeros(batch_size, 3, device=self.device)
        self.is_overlapping = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        try:
            for i in range(batch_size):
                pos = current_positions[i].cpu().numpy()
                query_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
                
                # 获取最近点
                dic = self.pai.get_closest_points([query_point], "/World/envs/env_.*/Constraint/geometry/mesh")
                
                if dic and 'closest_points' in dic and dic['closest_points']:
                    closest_pt = dic['closest_points'][0]
                    closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
                    
                    # 计算距离
                    distance = np.linalg.norm(pos - closest_pos)
                    
                    # 射线投射获取法向量
                    direction = Float3(closest_pt.x - pos[0], closest_pt.y - pos[1], closest_pt.z - pos[2])
                    dic2 = self.psqi.raycast_closest(query_point, direction, 10000)
                    
                    # 重叠检测
                    overlapping = distance < self.cfg.collision_tolerance
                    
                    # 计算法向量
                    if dic2 and 'normal' in dic2:
                        normal_carb = dic2['normal']
                        normal_array = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        # 默认法向量（从当前点指向最近点）
                        diff = closest_pos - pos
                        normal_array = diff / (np.linalg.norm(diff) + 1e-8)
                    
                    # 存储结果
                    self.constraint_distances[i] = distance
                    self.constraint_normals[i] = torch.tensor(normal_array, device=self.device)
                    self.constraint_closest_points[i] = torch.tensor(closest_pos, device=self.device)
                    self.is_overlapping[i] = overlapping
                    
                else:
                    # 无法获取约束信息，使用默认值
                    self.constraint_distances[i] = 0.1
                    self.constraint_normals[i] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                    self.constraint_closest_points[i] = current_positions[i]
                    self.is_overlapping[i] = False
                
        except Exception as e:
            print(f"[ERROR] Physics query constraint computation failed: {e}")
            # 填充默认值
            self.constraint_distances.fill_(0.1)
            self.constraint_normals[:, 0] = 1.0
        
    def _apply_action(self) -> None:
        """应用处理后的动作到环境"""
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取包含机器人完整状态的观测"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_quat = self._get_end_effector_quaternion()
        end_effector_lin_vel = self._get_end_effector_velocity()
        end_effector_ang_vel = self._get_end_effector_angular_velocity()
        
        joint_pos = self._omni_robot.data.joint_pos
        joint_vel = self._omni_robot.data.joint_vel
        
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        constraint_info = torch.cat([
            self.constraint_distances.unsqueeze(-1),
            self.constraint_normals,
            self.constraint_closest_points,
        ], dim=-1)
        
        try:
            obs = torch.cat([
                end_effector_pos,
                end_effector_quat,
                end_effector_lin_vel,
                end_effector_ang_vel,
                joint_pos,
                joint_vel,
                constraint_info,
                target_pos,
            ], dim=-1)
            
        except Exception as e:
            print(f"[ERROR] Observation concatenation failed: {e}")
            obs = torch.cat([
                end_effector_pos,
                end_effector_lin_vel,
                joint_pos,
                joint_vel,
                target_pos,
            ], dim=-1)
        
        obs = torch.clamp(obs, -10.0, 10.0)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """基于论文方程(13)的成本函数计算奖励"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # 1. 位置跟踪成本
        position_error = end_effector_pos - target_pos
        tracking_cost = torch.sum(position_error**2, dim=-1) * self.Q1_weight
        
        # 2. 速度调节成本
        velocity_cost = torch.sum(end_effector_vel**2, dim=-1) * self.Q2_weight
        
        # 3. 力调节成本
        force_cost = torch.sum(self.total_interaction_forces**2, dim=-1) * self.Q3_weight
        
        # 4. 控制输入成本
        control_cost = torch.sum(self.previous_robot_actions**2, dim=-1) * self.R_weight
        
        total_cost = tracking_cost + velocity_cost + force_cost + control_cost
        paper_reward = -total_cost
        
        # 安全约束奖励
        collision_penalty = self.is_overlapping.float() * (-10.0)
        
        
        # 到达终点的大奖励
        reached_target = distance_to_target < 0.01
        completion_reward = reached_target.float() * 20.0

        # total loss
        total_reward = paper_reward + collision_penalty + completion_reward
        total_reward = torch.clamp(total_reward, -20.0, 25.0)
        
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
            "progress_reward": progress_reward.mean().item(),
            "completion_reward": completion_reward.mean().item(),
            "total_reward": total_reward.mean().item(),
            "constraint_distance": self.constraint_distances.mean().item(),
            "trajectory_index": self.trajectory_manager.current_index,
            "trajectory_progress": self.trajectory_manager.get_progress(),
            "distance_to_target": distance_to_target.mean().item(),
            "overlap_rate": self.is_overlapping.float().mean().item(),
            "robot_force_norm": torch.norm(self.robot_forces, dim=-1).mean().item(),
            "human_force_norm": torch.norm(self.human_forces, dim=-1).mean().item(),
            "total_force_norm": torch.norm(self.total_interaction_forces, dim=-1).mean().item(),
        }
        
        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """确定剧集是否终止或截断"""
        terminated = self.is_overlapping.clone()
        
        end_effector_pos = self._get_end_effector_position()
        fell_out = end_effector_pos[..., 2] < -0.01
        terminated = terminated | fell_out
        
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        distance_to_target = torch.norm(end_effector_pos - target_pos, dim=-1)
        reached_target = distance_to_target < 0.001
        terminated = terminated | reached_target
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # 设置初始关节位置
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = 0.0  # waist
        joint_pos[:, 1] = 0.0  # shoulder  
        joint_pos[:, 2] = 0.0  # elbow
        joint_pos[:, 3] = 4.0  # yaw
        joint_pos[:, 4] = 1.2  # pitch - 笔身立起
        joint_pos[:, 5] = 0.0  # roll
        
        joint_noise = sample_uniform(-0.1, 0.1, (num_resets, 6), self.device)
        joint_pos += joint_noise
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置跟踪变量
        self.previous_robot_actions[env_ids] = 0.0
        self.human_forces[env_ids] = 0.0
        self.robot_forces[env_ids] = 0.0
        self.total_interaction_forces[env_ids] = 0.0
        self.constraint_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.constraint_closest_points[env_ids] = 0.0
        self.is_overlapping[env_ids] = False
        
        self.trajectory_manager.reset_trajectory()

    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, :3]
        
    def _get_end_effector_velocity(self):
        """获取末端执行器线速度"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, 7:10]
        
    def _get_end_effector_quaternion(self):
        """获取末端执行器姿态（四元数）"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, 3:7]
        
    def _get_end_effector_angular_velocity(self):
        """获取末端执行器角速度"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, 10:13]
        
    def _update_constraints(self):
        """更新约束信息"""
        end_effector_pos = self._get_end_effector_position()
        self._compute_constraint_distance_and_normal(end_effector_pos)