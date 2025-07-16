# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术直接环境配置 - 基于论文方程实现"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SurgicalDirectEnvCfg(DirectRLEnvCfg):
    """论文对齐的手术直接环境配置"""
    
    # 环境设置
    episode_length_s = 8.0  
    decimation = 2  
    action_space = 3  # xyz forces only
    
    # 论文方程(18)的状态空间: z = [x, ẋ, f]^T ∈ R^9
    observation_space = 12  # 修改为: pos(3) + vel(3) + force(3) + desired_pos(3) = 12D
    state_space = 9  # 论文标准状态: z = [x, ẋ, f]^T ∈ R^9
    
    # 仿真设置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120 Hz simulation
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
        ),
    )
    
    # 场景设置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, 
        env_spacing=2.0, 
        replicate_physics=True
    )
    
    # 手术刀配置
    scalpel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Scalpel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/scalpel_simple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.015),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    
    # 约束配置
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(0.017, 0.017, 0.0125),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # 地面
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0, 
            restitution=0.0,
        ),
    )
    
    # 力和控制参数
    max_force = 1.0  
    force_scale = 1.0  
    
    # 论文方程(13)的成本函数权重 - 严格按论文设置
    Q1_weight = 100.0   # 位置跟踪权重 (x-x_d)^T Q_1 (x-x_d) 
    Q2_weight = 0.01    # 速度调节权重 ẋ^T Q_2 ẋ
    Q3_weight = 0.001   # 力调节权重 f^T Q_3 f
    R_weight = 0.001    # 控制输入权重 u^T R u
    
    # 论文方程(6)的人体阻抗参数
    human_damping_CH = [21.0, 21.0, 21.0]        # C_H 矩阵对角元素
    human_stiffness_KH = [201.0, 201.0, 201.0]   # K_H 矩阵对角元素
    interaction_force_threshold = 0.1             # 交互力阈值
    
    # 任务参数
    target_height = 0.005  
    collision_threshold = 0.0001  
    
    # 约束几何参数
    constraint_inner_radius_min = 0.01   
    constraint_inner_radius_max = 0.05   
    constraint_outer_radius_min = 0.015  
    constraint_outer_radius_max = 0.055  
    constraint_height = 0.1              
    
    # 人体工作空间参数
    human_workspace_radius = 0.2         
    human_max_velocity = 0.3             
    
    # 论文共享控制参数
    robot_action_weight = 0.7            # α in paper
    human_action_weight = 0.3            # 1-α in paper  
    collaboration_adaptation_rate = 0.05 # 自适应率
    
    # 轨迹参数
    trajectory_update_frequency = 10     
    trajectory_completion_threshold = 0.005  
    spiral_trajectory_turns = 2.0        
    spiral_radius_range = (0.01, 0.04)   
    spiral_height_range = (0.002, 0.08)  
    trajectory_points_count = 200        
    
    # 安全参数
    max_constraint_distance = 0.02       
    emergency_stop_distance = 0.001      
    collision_recovery_force = 1.0       
    sphere_radius = 0.002                
    
    # 动作缩放参数
    action_velocity_scale = 0.1           
    action_force_scale = 1.0             
    max_action_norm = 0.1                
    
    # 观测归一化参数
    position_normalization_range = 0.3   
    velocity_normalization_range = 0.5   
    force_normalization_range = 5.0      
    observation_clamp_range = 10.0       
    
    # 环境重置参数
    reset_position_noise = 0.005         
    reset_velocity_noise = 0.002         
    reset_within_workspace = True        
    reset_workspace_radius_range = (0.02, 0.15)  
    
    # 物理仿真参数
    physics_dt = 1/120                   
    render_dt = 1/60                     
    solver_iterations = 4                
    solver_velocity_iterations = 1       
    use_gpu_physics = True