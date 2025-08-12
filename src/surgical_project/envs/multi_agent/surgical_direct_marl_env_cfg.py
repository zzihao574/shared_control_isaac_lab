# surgical_direct_marl_env_cfg.py

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
    """Scene configuration"""
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class SurgicalDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Direct MARL Environment"""
    
    episode_length_s = 15.0
    decimation = 2
    
    possible_agents = ["human", "robot"]
    
    action_spaces = {
        "human": 3,
        "robot": 3,
    }
    
    observation_spaces = {
        "human": 21,
        "robot": 21,
    }
    
    state_space = 24
    
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
        ),
    )
    
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)
    
    # 每个关节单独配置执行器
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/omni.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=0.04,
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
                "pitch": 2.0944,
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
            "waist_actuator": ImplicitActuatorCfg(
                joint_names_expr=["waist"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "shoulder_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "elbow_actuator": ImplicitActuatorCfg(
                joint_names_expr=["elbow"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "yaw_actuator": ImplicitActuatorCfg(
                joint_names_expr=["yaw"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "pitch_actuator": ImplicitActuatorCfg(
                joint_names_expr=["pitch"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "roll_actuator": ImplicitActuatorCfg(
                joint_names_expr=["roll"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    
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
            pos=(0.14, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    physics_dt = 1/120
    render_dt = 1/60
    solver_iterations = 16
    solver_velocity_iterations = 8
    use_gpu_physics = True