# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术直接环境 - 使用Omni haptic device的人机共享控制，集成CBF约束"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .surgical_direct_env_cfg import SurgicalDirectEnvCfg


class TrajectoryManager:
    """轨迹管理器 - 基于距离的目标点切换（非时间驱动）"""
    
    def __init__(self, device: torch.device, target_points: list, reach_threshold: float = 0.01):
        self.device = device
        self.target_points = [torch.tensor(point, device=device, dtype=torch.float32) for point in target_points]
        self.reach_threshold = reach_threshold
        self.current_target_index = 0
        
        print(f"[INFO] Trajectory manager initialized with {len(self.target_points)} target points")
        print(f"  Target points: {target_points}")
        print(f"  Reach threshold: {reach_threshold}")
        
    def get_current_target(self) -> torch.Tensor:
        """获取当前目标点"""
        return self.target_points[self.current_target_index]
        
    def update_target(self, current_pos: torch.Tensor) -> bool:
        """根据当前位置更新目标点，返回是否切换了目标"""
        current_target = self.target_points[self.current_target_index]
        
        # 计算到当前目标的距离
        if current_pos.dim() > 1:
            # 批处理情况，取第一个环境的位置
            distance = torch.norm(current_pos[0] - current_target)
        else:
            distance = torch.norm(current_pos - current_target)
        
        # 如果到达当前目标点且还有下一个目标
        if distance < self.reach_threshold and self.current_target_index < len(self.target_points) - 1:
            self.current_target_index += 1
            print(f"[INFO] Switched to target {self.current_target_index}: {self.target_points[self.current_target_index]}")
            return True
        
        return False
        
    def reset_trajectory(self):
        """重置轨迹到起始点"""
        self.current_target_index = 0
        
    def get_progress(self) -> float:
        """获取轨迹进度 (0-1)"""
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


class SurgicalDirectEnv(DirectRLEnv):
    """论文对齐的手术直接环境 - 人机共享控制"""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化手术环境"""
        super().__init__(cfg, render_mode, **kwargs)
        
        # 获取物理查询接口（用于约束计算）
        try:
            from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
            self.physics_attachment_interface = acquire_physx_attachment_interface()
            self.physics_scene_query_interface = acquire_physx_scene_query_interface()
        except ImportError:
            print("[WARNING] Physics query interfaces not available, using simplified constraint model")
            self.physics_attachment_interface = None
            self.physics_scene_query_interface = None
        
        # 时间步长
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # 初始化跟踪变量
        self.previous_robot_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.human_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # CBF约束相关变量（重命名更易理解）
        self.safety_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.closest_constraint_points = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_violating_constraint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # CBF参数
        self.cbf_gamma = self.cfg.cbf_gamma  # γ参数，控制约束强度
        self.safety_margin = self.cfg.safety_margin  # 安全边界
        
        # 轨迹管理器 - 使用论文要求的两个平衡点
        target_points = [
            (0.0, 0.15, 0.03),    # 第一个平衡点
            (0.2, 0.15, 0.03)     # 第二个平衡点
        ]
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            target_points=target_points,
            reach_threshold=self.cfg.target_reach_threshold
        )
        
        # 论文成本函数权重
        self.Q1_weight = self.cfg.Q1_weight
        self.Q2_weight = self.cfg.Q2_weight  
        self.Q3_weight = self.cfg.Q3_weight
        self.R_weight = self.cfg.R_weight
        
        # CBF约束权重
        self.cbf_weight = self.cfg.cbf_weight  # CBF在成本函数中的权重
        
        # 末端执行器参数
        self.end_effector_body_id = self.cfg.end_effector_body_id
        
        print(f"[INFO] Surgical environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation space: {self.cfg.observation_space}D") 
        print(f"  - Action space: {self.cfg.action_space}D")
        print(f"  - End effector body ID: {self.end_effector_body_id}")
        print(f"  - Target-based trajectory with {len(target_points)} points")
        print(f"  - CBF gamma: {self.cbf_gamma}, safety margin: {self.safety_margin}")
        
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
        self._update_safety_constraints()
            
    def _simulate_human_input(self):
        """模拟人类输入力"""
        # 简化人类输入模拟
        self.human_forces = torch.zeros_like(self.human_forces)
        
    def _compute_shared_control_forces(self):
        """计算人机共享控制的总力"""
        alpha = self.cfg.robot_action_weight
        self.total_interaction_forces = (alpha * self.robot_forces + 
                                       (1 - alpha) * self.human_forces)
        
    def _apply_forces_to_end_effector(self):
        """对末端执行器施加总的交互力"""
        # （1）准备 forces 和 torques，shape → [num_envs, 1, 3]
        forces_reshaped = self.total_interaction_forces.unsqueeze(1)
        torques_reshaped = torch.zeros_like(forces_reshaped)

        # （2）只指定一次末端执行器的 body_id，shape → [1]
        body_ids = torch.tensor(
            [self.end_effector_body_id],
            dtype=torch.long,
            device=self.device
        )

        # （3）明确指定 env_ids，shape → [num_envs]
        env_ids = torch.arange(self.num_envs, device=self.device)

        # （4）调用核心 API → 内部会自动 broadcast
        self._omni_robot.set_external_force_and_torque(
            forces=forces_reshaped,    # [num_envs,1,3]
            torques=torques_reshaped,  # [num_envs,1,3]
            body_ids=body_ids,         # [1]
            env_ids=env_ids            # [num_envs]
        )

    def _compute_safety_barrier_function(self, end_effector_positions: torch.Tensor) -> torch.Tensor:
        """
        计算控制屏障函数 Br(x) = -log(γs(x)/(γs(x)+1))
        其中 s(x) 是到约束边界的距离函数
        """
        batch_size = end_effector_positions.shape[0]
        
        # 初始化安全距离
        self.safety_distances = torch.zeros(batch_size, device=self.device)
        self.constraint_normals = torch.zeros(batch_size, 3, device=self.device)
        self.closest_constraint_points = torch.zeros(batch_size, 3, device=self.device)
        self.is_violating_constraint = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        if self.physics_attachment_interface is not None and self.physics_scene_query_interface is not None:
            # 使用物理查询接口计算真实距离
            self._compute_physics_based_constraints(end_effector_positions)
        else:
            # 使用简化约束模型
            self._compute_simplified_constraints(end_effector_positions)
        
        # 计算CBF值：Br(x) = -log(γs(x)/(γs(x)+1))
        # s(x) 是安全距离减去安全边界
        s_x = self.safety_distances - self.safety_margin
        
        # 确保s(x)有一个最小值，避免数值问题
        s_x = torch.clamp(s_x, min=1e-6)
        
        # 计算CBF
        gamma_s = self.cbf_gamma * s_x
        cbf_values = -torch.log(gamma_s / (gamma_s + 1))
        
        return cbf_values
    
    def _compute_physics_based_constraints(self, current_positions: torch.Tensor):
        """使用物理查询API计算到约束表面的距离和法向量"""
        try:
            from carb._carb import Float3
            
            for i in range(current_positions.shape[0]):
                pos = current_positions[i].cpu().numpy()
                query_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
                
                # 获取最近点
                constraint_path = f"/World/envs/env_{i}/Constraint/geometry/mesh"
                closest_point_result = self.physics_attachment_interface.get_closest_points(
                    [query_point],
                    constraint_path
                )
                
                if closest_point_result and 'closest_points' in closest_point_result and closest_point_result['closest_points']:
                    closest_pt = closest_point_result['closest_points'][0]
                    closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
                    
                    # 计算安全距离
                    distance = np.linalg.norm(pos - closest_pos)
                    
                    # 射线投射获取法向量
                    direction = Float3(closest_pt.x - pos[0], closest_pt.y - pos[1], closest_pt.z - pos[2])
                    raycast_result = self.physics_scene_query_interface.raycast_closest(query_point, direction, 10000)
                    
                    # 约束违反检测
                    is_violating = bool(distance < self.safety_margin)
                    
                    # 计算约束表面法向量
                    if raycast_result and 'normal' in raycast_result:
                        normal_carb = raycast_result['normal']
                        normal_array = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        # 默认法向量（从约束表面指向当前点）
                        diff = pos - closest_pos
                        normal_array = diff / (np.linalg.norm(diff) + 1e-8)
                    
                    # 存储结果
                    self.safety_distances[i] = float(distance)
                    self.constraint_normals[i] = torch.tensor(normal_array, device=self.device)
                    self.closest_constraint_points[i] = torch.tensor(closest_pos, device=self.device)
                    self.is_violating_constraint[i] = is_violating
                    
                else:
                    # 无法获取约束信息，使用安全默认值
                    self.safety_distances[i] = self.safety_margin * 2  # 假设安全
                    self.constraint_normals[i] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                    self.closest_constraint_points[i] = current_positions[i]
                    self.is_violating_constraint[i] = False
                    
        except Exception as e:
            print(f"[ERROR] Physics-based constraint computation failed: {e}")
            # 回退到简化模型
            self._compute_simplified_constraints(current_positions)
    
    def _compute_simplified_constraints(self, current_positions: torch.Tensor):
        """简化的约束模型 - 基于几何距离"""
        # 假设约束位于 (0, 0.15, 0) 附近的圆锥形区域
        constraint_center = torch.tensor([0.0, 0.15, 0.0], device=self.device)
        constraint_radius = 0.05  # 约束半径
        
        for i in range(current_positions.shape[0]):
            pos = current_positions[i]
            
            # 计算到约束中心的距离
            diff = pos - constraint_center
            horizontal_dist = torch.norm(diff[:2])  # x-y平面距离
            vertical_dist = torch.abs(diff[2])      # z方向距离
            
            # 简化的圆锥约束：水平距离 + 垂直距离权重
            distance_to_constraint = torch.sqrt(horizontal_dist**2 + (vertical_dist * 2)**2)
            safety_distance = torch.clamp(distance_to_constraint - constraint_radius, min=0.0)
            
            # 计算约束法向量
            if horizontal_dist > 1e-6:
                normal = diff / torch.norm(diff)
            else:
                normal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            
            # 存储结果
            self.safety_distances[i] = safety_distance
            self.constraint_normals[i] = normal
            self.closest_constraint_points[i] = constraint_center + normal * constraint_radius
            self.is_violating_constraint[i] = safety_distance < self.safety_margin
    
    def _update_safety_constraints(self):
        """更新安全约束信息"""
        end_effector_pos = self._get_end_effector_position()
        self._compute_safety_barrier_function(end_effector_pos)
        
    def _apply_action(self) -> None:
        """应用处理后的动作到环境"""
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取简化的观测"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        # 更新目标点（基于距离）
        self.trajectory_manager.update_target(end_effector_pos)
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # 简化观测：位置(3) + 速度(3) + 目标位置(3) = 9D，扩展到12D用于兼容
        obs = torch.cat([
            end_effector_pos,      # 当前位置 [0:3]
            end_effector_vel,      # 当前速度 [3:6]
            target_pos,            # 目标位置 [6:9]
            torch.zeros(self.num_envs, 3, device=self.device)  # 填充到12D [9:12]
        ], dim=-1)
        
        obs = torch.clamp(obs, -10.0, 10.0)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """基于论文方程(13)的成本函数计算奖励，集成CBF约束"""
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
        
        # 5. CBF约束成本 - 集成控制屏障函数
        cbf_values = self._compute_safety_barrier_function(end_effector_pos)
        cbf_cost = cbf_values * self.cbf_weight
        
        # 总成本函数（扩展的论文方程13）
        total_cost = tracking_cost + velocity_cost + force_cost + control_cost + cbf_cost
        paper_reward = -total_cost
        
        # 安全约束惩罚（额外的安全层）
        constraint_violation_penalty = self.is_violating_constraint.float() * (-50.0)
        
        # 目标点到达奖励
        distance_to_target = torch.norm(end_effector_pos - target_pos, dim=-1)
        target_reached = distance_to_target < self.cfg.target_reach_threshold
        completion_reward = target_reached.float() * 20.0
        
        # 最终目标完成奖励
        final_completion = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            if self.trajectory_manager.is_final_target_reached(end_effector_pos[i]):
                final_completion[i] = 50.0  # 大奖励完成整个轨迹

        total_reward = paper_reward + constraint_violation_penalty + completion_reward + final_completion
        total_reward = torch.clamp(total_reward, -100.0, 75.0)
        
        # 存储奖励组件用于记录
        self.extras["log"] = {
            "tracking_cost": tracking_cost.mean().item(),
            "velocity_cost": velocity_cost.mean().item(),
            "force_cost": force_cost.mean().item(),
            "control_cost": control_cost.mean().item(),
            "cbf_cost": cbf_cost.mean().item(),
            "total_paper_cost": total_cost.mean().item(),
            "paper_reward": paper_reward.mean().item(),
            "constraint_violation_penalty": constraint_violation_penalty.mean().item(),
            "completion_reward": completion_reward.mean().item(),
            "final_completion_reward": final_completion.mean().item(),
            "total_reward": total_reward.mean().item(),
            "trajectory_progress": self.trajectory_manager.get_progress(),
            "current_target_index": self.trajectory_manager.current_target_index,
            "distance_to_target": distance_to_target.mean().item(),
            "safety_distance": self.safety_distances.mean().item(),
            "cbf_value": cbf_values.mean().item(),
            "constraint_violation_rate": self.is_violating_constraint.float().mean().item(),
            "robot_force_norm": torch.norm(self.robot_forces, dim=-1).mean().item(),
            "human_force_norm": torch.norm(self.human_forces, dim=-1).mean().item(),
            "total_force_norm": torch.norm(self.total_interaction_forces, dim=-1).mean().item(),
        }
        
        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """确定剧集是否终止或截断"""
        end_effector_pos = self._get_end_effector_position()
        
        # 终止条件：约束违反、掉落或完成最终目标
        constraint_violated = self.is_violating_constraint.clone()
        fell_out = end_effector_pos[..., 2] < -0.01
        final_target_reached = torch.tensor([
            self.trajectory_manager.is_final_target_reached(end_effector_pos[i])
            for i in range(self.num_envs)
        ], device=self.device, dtype=torch.bool)
        
        terminated = constraint_violated | fell_out | final_target_reached
        
        # 截断条件：超时
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
        
        # 重置约束相关变量
        self.safety_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.closest_constraint_points[env_ids] = 0.0
        self.is_violating_constraint[env_ids] = False
        
        # 重置轨迹管理器
        self.trajectory_manager.reset_trajectory()

    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, :3]
        
    def _get_end_effector_velocity(self):
        """获取末端执行器线速度"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, 7:10]