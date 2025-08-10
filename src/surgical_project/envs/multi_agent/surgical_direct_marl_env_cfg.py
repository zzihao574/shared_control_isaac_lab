# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Surgical Direct MARL Environment with Human-Robot Collaboration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration containing only basic elements"""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class SurgicalDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Direct MARL Environment - 与Single Agent完全对齐"""
    
    # Environment settings (与Single Agent对齐)
    episode_length_s = 15.0
    decimation = 2
    
    # Multi-agent settings
    possible_agents = ["human", "robot"]
    
    # Agent-specific action and observation spaces (相同观测)
    action_spaces = {
        "human": 3,   # xyz forces in Cartesian space
        "robot": 3,   # xyz forces in Cartesian space
    }
    
    observation_spaces = {
        "human": 21,   # [x, ẋ, q, q̇, f, constraint_distances] (相同观测)
        "robot": 21,   # [x, ẋ, q, q̇, f, constraint_distances] (相同观测)
    }
    
    # Global state space
    state_space = 24  # 全局状态信息
    
    # Simulation settings (与Single Agent完全对齐)
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,  # 120 Hz simulation
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),  # Disable gravity
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
        ),
    )
    
    # Scene settings (与Single Agent对齐)
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)
    
    # Phantom Omni (与Single Agent完全对齐)
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/omni.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=50.0,
                max_angular_velocity=50.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": -0.96,
                "shoulder": 0.0,
                "elbow": 1.0,
                "yaw": 0.0,
                "pitch": 2.0944,  # 120 degrees in radians
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
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist", 
                    "shoulder", 
                    "elbow", 
                    "yaw", 
                    "pitch", 
                    "roll"
                ],
                effort_limit_sim=5.0,
                velocity_limit_sim=50.0,
                stiffness={
                    "waist": 0.25,
                    "shoulder": 0.25,
                    "elbow": 0.25,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                },
                damping={
                    "waist": 0.07,
                    "shoulder": 0.07,
                    "elbow": 0.07,
                    "yaw": 0.4,
                    "pitch": 0.4,
                    "roll": 0.4,
                },
            ),
        },
    )
    
    # Constraint geometry (与Single Agent完全对齐)
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(0.01, 0.01, 0.015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.14, 0.0, 0.0),  # 约束位置修正到 0.14 0.0 0.0
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Physics simulation settings (与Single Agent对齐)
    physics_dt = 1/120
    render_dt = 1/60
    solver_iterations = 16
    solver_velocity_iterations = 8
    use_gpu_physics = True