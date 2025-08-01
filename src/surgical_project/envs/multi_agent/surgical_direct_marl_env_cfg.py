# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Surgical Direct MARL Environment with Human-Robot Collaboration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SurgicalDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Direct MARL Environment."""
    
    # Environment settings (与MBRL对齐)
    episode_length_s = 8.0  # 与MBRL统一
    decimation = 2
    
    # Multi-agent settings
    possible_agents = ["human", "robot"]
    
    # Agent-specific action and observation spaces (与MBRL基础观测对齐)
    action_spaces = {
        "human": 3,   # xyz forces (与MBRL统一)
        "robot": 3,   # xyz forces (与MBRL统一)
    }
    
    observation_spaces = {
        "human": 18,   # 基础12D + 人类特有6D (意图3D + 信任3D)
        "robot": 18,   # 基础12D + 机器人特有6D (人类动作3D + 信任3D)
    }
    
    # Global state space
    state_space = 24  # 全局状态信息
    
    # Simulation settings (与MBRL完全对齐)
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
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,  # 与MBRL对齐
        replicate_physics=True
    )
    
    # Phantom Omni (由两个智能体协作控制)
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=100,
                solver_velocity_iteration_count=100,
                sleep_threshold=0.01,
                stabilization_threshold=0.01,
                fix_root_link=True
            ),
            scale=(1, 1, 1),
            asset_path='/home/zzh/workspace/surgical_robot_project/assets/models/urdf/omni.urdf',
            usd_dir='/home/zzh/workspace/surgical_robot_project/assets/models/usd',
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={
                        "waist": 1,    
                        "shoulder": 1,
                        "elbow": 1,
                        "yaw": 10000,      
                        "pitch": 10000,    
                        "roll": 10000,     
                    }, 
                    damping={
                        "waist": 0.1,
                        "shoulder": 0.1,
                        "elbow": 0.1,
                        "yaw": 1000,       
                        "pitch": 1000,     
                        "roll": 1000,      
                    },
                ),
                drive_type="acceleration",
                target_type="position"
            ),
            fix_base=True,
            root_link_name='base',
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": 0.0,
                "shoulder": 0.0,
                "elbow": 0.0,
                "yaw": 4.0,
                "pitch": 1.2,
                "roll": 0.0,
            },
            joint_vel={
                "waist": 0.0,
                "shoulder": 0.0,
                "elbow": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
            },
        ),
        actuators={
            "arm_joints": IdealPDActuatorCfg(
                joint_names_expr=[
                    "waist", 
                    "shoulder", 
                    "elbow", 
                    "yaw", 
                    "pitch", 
                    "roll"
                ],
                stiffness={
                    "waist": 1,    
                    "shoulder": 1,
                    "elbow": 1,
                    "yaw": 10000,
                    "pitch": 10000,
                    "roll": 10000,
                },
                damping={
                    "waist": 0.1,
                    "shoulder": 0.1,
                    "elbow": 0.1,
                    "yaw": 1000,
                    "pitch": 1000,
                    "roll": 1000,
                },
            ),
        },
    )
    
    # Constraint (与MBRL完全一致)
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(0.01, 0.01, 0.015),  # 与MBRL一致
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.15, 0.0),  # 与MBRL一致
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Ground plane
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
    
    # Force and control parameters (与MBRL对齐)
    max_force = {
        "human": 3.3,   # 与MBRL统一
        "robot": 3.3,   # 与MBRL统一
    }
    force_scale = 1.0
    
    # End effector parameters (与MBRL对齐)
    end_effector_body_name = "stylus"
    end_effector_body_id = 6
    interaction_coupling = 0.3
    force_sharing_ratio = {
        "human": 0.6,
        "robot": 0.4,
    }
    
    # Trajectory parameters (与MBRL完全对齐)
    target_reach_threshold = 0.01
    target_points = [
        (-0.2, 0.15, 0.03),  # 第一个平衡点
        (0.2, 0.15, 0.03)    # 第二个平衡点
    ]
    
    # Constraint geometry parameters (与MBRL一致)
    constraint_inner_radius_min = 0.1
    constraint_inner_radius_max = 0.25
    constraint_height = 0.2598
    collision_threshold = 0.001
    
    # Human dynamics parameters (与MBRL对齐)
    human_dynamics = {
        "stiffness": 201.0,
        "damping": 21.0,
        "noise_std": 0.1,
        "reaction_delay": 0.05,
        "intention_update_rate": 10.0,
    }
    
    # Collaboration parameters (MARL特有)
    collaboration = {
        "trust_factor": 0.8,
        "trust_decay": 0.95,
        "trust_recovery": 1.02,
        "conflict_threshold": 2.0,
    }
    
    # Task completion criteria (与MBRL对齐)
    task_completion = {
        "target_radius": 0.01,
        "completion_time": 50,
        "max_completion_bonus": 100.0,
        "partial_completion_steps": [10, 25, 40],
        "partial_completion_rewards": [10.0, 25.0, 40.0],
    }
    
    # Reward scales (保持MARL差异化奖励)
    reward_scales = {
        "human": {
            "tracking_reward": 100.0,      # 与MBRL对齐
            "velocity_penalty": -0.01,      # 与MBRL对齐
            "control_penalty": -0.001,      # 与MBRL对齐
            "safety_penalty": -10.0,        # 与MBRL对齐
            "collision_penalty": -50.0,     # 与MBRL对齐
            "completion_reward": 20.0,      # 与MBRL对齐
            "collaboration_reward": 2.0,    # MARL特有
            "intention_alignment": 1.5,     # MARL特有
        },
        "robot": {
            "tracking_reward": 100.0,
            "velocity_penalty": -0.01,
            "control_penalty": -0.001,
            "safety_penalty": -10.0,
            "collision_penalty": -50.0,
            "completion_reward": 20.0,
            "collaboration_reward": 2.0,    # MARL特有
            "adaptation_reward": 1.5,       # MARL特有
        }
    }