"""Paper-aligned surgical direct environment - using Omni haptic device human-robot shared control with error-based trajectory"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.math import sample_uniform


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Test scene configuration"""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )


@configclass
class DirectEnvTestCfg(DirectRLEnvCfg):
    """Test environment configuration for Omni haptic device"""
    
    # Environment settings
    episode_length_s = 10.0  
    decimation = 2  
    action_space = 3  
    observation_space = 12  
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
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
    scene: InteractiveSceneCfg = TestSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    
    # Omni haptic device
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/omni.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.01,
                stabilization_threshold=0.01,
                fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": 0.0,
                "shoulder": 0.0,
                "elbow": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,       # 120度转弧度
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
                    "waist": 50,    
                    "shoulder": 50,
                    "elbow": 50,
                    "yaw": 200,      # 适中刚度
                    "pitch": 200,    # 适中刚度  
                    "roll": 200,     # 适中刚度
                },
                damping={
                    "waist": 10,
                    "shoulder": 10,
                    "elbow": 10,
                    "yaw": 50,       # 适中阻尼
                    "pitch": 50,     # 适中阻尼
                    "roll": 50,      # 适中阻尼
                },
            ),
        },
        # Override joint limits to match USD (degrees)
        soft_joint_pos_limit_factor=0.95,
    )
    
    # Force and control parameters
    max_robot_force = 3.3
    force_scale = 1.0  
    
    # End effector parameters
    end_effector_body_name = "stylus"
    end_effector_body_id = 6
    
    # Joint limit enforcement
    enforce_joint_limits = True
    # Fixed joints during training (last 3 joints)
    fixed_joints_during_training = [3, 4, 5]  # yaw, pitch, roll indices


class DirectEnvTest(DirectRLEnv):
    """Simplified test environment for Omni configuration validation"""
    
    cfg: DirectEnvTestCfg
    
    def __init__(self, cfg: DirectEnvTestCfg, render_mode: str | None = None, **kwargs):
        """Initialize test environment"""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Time step
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # End effector parameters
        self.end_effector_body_id = self.cfg.end_effector_body_id
        
        print(f"[INFO] Test environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - End effector body ID: {self.end_effector_body_id}")
        print(f"  - Target position: [-0.1, 0.1, 0.05]")
        
    def _setup_scene(self):
        """Setup simulation scene with Omni robot only"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self.scene.clone_environments(copy_from_source=False)
        
        print(f"[INFO] Scene setup complete with {self.num_envs} environments")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions with fixed joints constraint"""
        # Ensure actions only affect first 3 joints
        joint_targets = torch.zeros(self.num_envs, 6, device=self.device)
        joint_targets[:, :3] = actions[:, :3]  # waist, shoulder, elbow
        joint_targets[:, 3] = 0.0              # yaw fixed
        joint_targets[:, 4] = 120.0            # pitch fixed at 120°
        joint_targets[:, 5] = 0.0              # roll fixed
        
        self._omni_robot.set_joint_position_target(joint_targets)

    def _apply_action(self) -> None:
        """Apply processed actions to environment"""
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get simple observations"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        joint_pos = self.get_joint_positions()
        
        # Target position
        target_pos = torch.tensor([-0.1, 0.1, 0.05], device=self.device)
        target_expanded = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # Simple observation: position + velocity + target
        obs = torch.cat([
            end_effector_pos,     # [0:3]
            end_effector_vel,     # [3:6] 
            target_expanded,      # [6:9]
            torch.zeros(self.num_envs, 3, device=self.device)  # [9:12] padding
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Simple distance-based reward"""
        end_effector_pos = self._get_end_effector_position()
        target_pos = torch.tensor([-0.1, 0.1, 0.05], device=self.device)
        target_expanded = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # Distance to target
        distance = torch.norm(end_effector_pos - target_expanded, dim=-1)
        reward = -distance * 100.0  # Negative distance as reward
        
        # Bonus for reaching target
        reached_target = distance < 0.02
        reward[reached_target] += 50.0
        
        # Store logging info
        self.extras["log"] = {
            "distance_to_target": distance.mean().item(),
            "end_effector_pos_x": end_effector_pos[:, 0].mean().item(),
            "end_effector_pos_y": end_effector_pos[:, 1].mean().item(), 
            "end_effector_pos_z": end_effector_pos[:, 2].mean().item(),
            "reward": reward.mean().item(),
        }
        
        return reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple termination conditions"""
        end_effector_pos = self._get_end_effector_position()
        target_pos = torch.tensor([-0.1, 0.1, 0.05], device=self.device)

        # Check if reached target
        distance = torch.norm(end_effector_pos - target_pos.unsqueeze(0), dim=-1)
        target_reached = distance < 0.02
        
        # Check if fell out of workspace
        fell_out = (end_effector_pos[..., 2] < -0.1) | (torch.norm(end_effector_pos, dim=-1) > 1.0)
        
        terminated = target_reached | fell_out
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset to target position with fixed last 3 joints"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # Set joint positions to reach target with fixed last 3 joints
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -135.0  # waist
        joint_pos[:, 1] = 1.8   # shoulder
        joint_pos[:, 2] = 71.5   # elbow
        joint_pos[:, 3] = 0.0    # yaw - FIXED
        joint_pos[:, 4] = 120.0  # pitch - FIXED
        joint_pos[:, 5] = 0.0    # roll - FIXED
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_end_effector_position(self):
        """Get end effector position"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, :3]
        
    def _get_end_effector_velocity(self):
        """Get end effector linear velocity"""
        return self._omni_robot.data.body_link_state_w[..., self.end_effector_body_id, 7:10]
    
    def get_joint_positions(self):
        """Get joint positions q"""
        return self._omni_robot.data.joint_pos
    
    def get_joint_velocities(self):
        """Get joint velocities q̇"""
        return self._omni_robot.data.joint_vel
    
    # Add unwrapped property for gym compatibility
    @property 
    def unwrapped(self):
        """Return self for unwrapped access"""
        return self