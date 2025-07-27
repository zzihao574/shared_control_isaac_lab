# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Paper-aligned surgical direct environment - using Omni haptic device human-robot shared control, integrating CBF constraints"""

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
    """Trajectory manager - distance-based target point switching (non-time driven)"""
    
    def __init__(self, device: torch.device, target_points: list, reach_threshold: float = 0.01):
        self.device = device
        self.target_points = [torch.tensor(point, device=device, dtype=torch.float32) for point in target_points]
        self.reach_threshold = reach_threshold
        self.current_target_index = 0
        
        print(f"[INFO] Trajectory manager initialized with {len(self.target_points)} target points")
        print(f"  Target points: {target_points}")
        print(f"  Reach threshold: {reach_threshold}")
        
    def get_current_target(self) -> torch.Tensor:
        """Get current target point"""
        return self.target_points[self.current_target_index]
        
    def update_target(self, current_pos: torch.Tensor) -> bool:
        """Update target point based on current position, return whether target was switched"""
        current_target = self.target_points[self.current_target_index]
        
        # Calculate distance to current target
        if current_pos.dim() > 1:
            # Batch processing case, take position of first environment
            distance = torch.norm(current_pos[0] - current_target)
        else:
            distance = torch.norm(current_pos - current_target)
        
        # If reached current target and there's a next target
        if distance < self.reach_threshold and self.current_target_index < len(self.target_points) - 1:
            self.current_target_index += 1
            print(f"[INFO] Switched to target {self.current_target_index}: {self.target_points[self.current_target_index]}")
            return True
        
        return False
        
    def reset_trajectory(self):
        """Reset trajectory to starting point"""
        self.current_target_index = 0
        
    def get_progress(self) -> float:
        """Get trajectory progress (0-1)"""
        return self.current_target_index / max(1, len(self.target_points) - 1)
        
    def is_final_target_reached(self, current_pos: torch.Tensor) -> bool:
        """Check if final target point is reached"""
        if self.current_target_index < len(self.target_points) - 1:
            return False
            
        final_target = self.target_points[-1]
        if current_pos.dim() > 1:
            distance = torch.norm(current_pos[0] - final_target)
        else:
            distance = torch.norm(current_pos - final_target)
            
        return distance < self.reach_threshold


class SurgicalDirectEnv(DirectRLEnv):
    """Paper-aligned surgical direct environment - human-robot shared control"""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize surgical environment"""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get physics query interface (for constraint computation)
        try:
            from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
            self.physics_attachment_interface = acquire_physx_attachment_interface()
            self.physics_scene_query_interface = acquire_physx_scene_query_interface()
        except ImportError:
            print("[WARNING] Physics query interfaces not available, using simplified constraint model")
            self.physics_attachment_interface = None
            self.physics_scene_query_interface = None
        
        # Time step
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Initialize tracking variables
        self.previous_robot_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.human_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # CBF constraint related variables (renamed for better understanding)
        self.safety_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.closest_constraint_points = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_violating_constraint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Trajectory manager - using paper required two equilibrium points
        target_points = [
            (-0.2, 0.15, 0.03),    # First equilibrium point (start)
            (0.2, 0.15, 0.03)      # Second equilibrium point (end)
        ]
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            target_points=target_points,
            reach_threshold=self.cfg.target_reach_threshold
        )
        
        # End effector parameters
        self.end_effector_body_id = self.cfg.end_effector_body_id
        
        print(f"[INFO] Surgical environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation space: {self.cfg.observation_space}D") 
        print(f"  - Action space: {self.cfg.action_space}D")
        print(f"  - End effector body ID: {self.end_effector_body_id}")
        print(f"  - Target-based trajectory with {len(target_points)} points")
        
    def _setup_scene(self):
        """Setup simulation scene"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        self.scene.clone_environments(copy_from_source=False)
        
        print(f"[INFO] Scene setup complete with {self.num_envs} environments")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step - implementing paper control framework"""
        # Robot control input u (from actor network output, processed according to paper)
        robot_actions = torch.clamp(actions, -1.0, 1.0)
        self.robot_forces = robot_actions * self.cfg.max_robot_force * self.cfg.force_scale
        self.previous_robot_actions = robot_actions.clone()
        
        # Simulate human input force f (from human impedance model)
        self._simulate_human_input()
        
        # Apply total force to end effector: robot control u + human force f
        self._apply_forces_to_end_effector()
        self._update_safety_constraints()
            
    def _simulate_human_input(self):
        """Simulate human input force according to paper equation (6)"""
        # Get current end effector state
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        # Get current human equilibrium point based on trajectory
        current_target_index = self.trajectory_manager.current_target_index
        if current_target_index < len(self.trajectory_manager.target_points):
            equilibrium_pos = self.trajectory_manager.target_points[current_target_index]
            equilibrium_pos = equilibrium_pos.unsqueeze(0).expand(self.num_envs, -1)
        else:
            equilibrium_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Simplified human force computation (to be enhanced with proper impedance model)
        position_error = end_effector_pos - equilibrium_pos
        
        # Simple spring-damper model for human force
        kh_simple = 50.0  # Human stiffness
        ch_simple = 10.0  # Human damping
        
        self.human_forces = -(kh_simple * position_error + ch_simple * end_effector_vel)
        self.human_forces = torch.clamp(self.human_forces, -self.cfg.max_human_force, self.cfg.max_human_force)
        
    def _apply_forces_to_end_effector(self):
        """Apply total forces to end effector: u + f"""
        # Paper control framework: total force = robot control u + human force f
        total_forces = self.robot_forces + self.human_forces
        self.total_interaction_forces = total_forces.clone()
        
        # Apply forces to end effector
        forces_reshaped = total_forces.unsqueeze(1)
        torques_reshaped = torch.zeros_like(forces_reshaped)

        body_ids = torch.tensor(
            [self.end_effector_body_id],
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

    def _compute_safety_barrier_function(self, end_effector_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute control barrier function Br(x) = -log(γs(x)/(γs(x)+1))
        where s(x) is distance function to constraint boundary
        """
        batch_size = end_effector_positions.shape[0]
        
        # Initialize safety distance
        self.safety_distances = torch.zeros(batch_size, device=self.device)
        self.constraint_normals = torch.zeros(batch_size, 3, device=self.device)
        self.closest_constraint_points = torch.zeros(batch_size, 3, device=self.device)
        self.is_violating_constraint = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        if self.physics_attachment_interface is not None and self.physics_scene_query_interface is not None:
            # Use physics query interface to compute real distance
            self._compute_physics_based_constraints(end_effector_positions)
        else:
            # Use simplified constraint model
            self._compute_simplified_constraints(end_effector_positions)
        
        # Compute CBF value: Br(x) = -log(γs(x)/(γs(x)+1))
        # s(x) is safety distance minus safety margin
        s_x = self.safety_distances - 0.002  # safety_margin from config
        
        # Ensure s(x) has a minimum value to avoid numerical issues
        s_x = torch.clamp(s_x, min=1e-6)
        
        # Compute CBF
        gamma_s = 1.0 * s_x  # cbf_gamma from config
        cbf_values = -torch.log(gamma_s / (gamma_s + 1))
        
        return cbf_values
    
    def _compute_physics_based_constraints(self, current_positions: torch.Tensor):
        """Use physics query API to compute distance and normal to constraint surface"""
        try:
            from carb._carb import Float3
            
            for i in range(current_positions.shape[0]):
                pos = current_positions[i].cpu().numpy()
                query_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
                
                # Get closest point
                constraint_path = f"/World/envs/env_{i}/Constraint/geometry/mesh"
                closest_point_result = self.physics_attachment_interface.get_closest_points(
                    [query_point],
                    constraint_path
                )
                
                if closest_point_result and 'closest_points' in closest_point_result and closest_point_result['closest_points']:
                    closest_pt = closest_point_result['closest_points'][0]
                    closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
                    
                    # Compute safety distance
                    distance = np.linalg.norm(pos - closest_pos)
                    
                    # Raycast to get normal vector
                    direction = Float3(closest_pt.x - pos[0], closest_pt.y - pos[1], closest_pt.z - pos[2])
                    raycast_result = self.physics_scene_query_interface.raycast_closest(query_point, direction, 10000)
                    
                    # Constraint violation detection
                    is_violating = bool(distance < 0.002)  # safety_margin
                    
                    # Compute constraint surface normal
                    if raycast_result and 'normal' in raycast_result:
                        normal_carb = raycast_result['normal']
                        normal_array = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        # Default normal (from constraint surface to current point)
                        diff = pos - closest_pos
                        normal_array = diff / (np.linalg.norm(diff) + 1e-8)
                    
                    # Store results
                    self.safety_distances[i] = float(distance)
                    self.constraint_normals[i] = torch.tensor(normal_array, device=self.device)
                    self.closest_constraint_points[i] = torch.tensor(closest_pos, device=self.device)
                    self.is_violating_constraint[i] = is_violating
                    
                else:
                    # Unable to get constraint information, use safe default values
                    self.safety_distances[i] = 0.004  # Assume safe (2x safety margin)
                    self.constraint_normals[i] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                    self.closest_constraint_points[i] = current_positions[i]
                    self.is_violating_constraint[i] = False
                    
        except Exception as e:
            print(f"[ERROR] Physics-based constraint computation failed: {e}")
            # Fallback to simplified model
            self._compute_simplified_constraints(current_positions)
    
    def _compute_simplified_constraints(self, current_positions: torch.Tensor):
        """Simplified constraint model - based on geometric distance"""
        # Assume constraint located near (0, 0.15, 0) in cone-shaped region
        constraint_center = torch.tensor([0.0, 0.15, 0.0], device=self.device)
        constraint_radius = 0.05  # Constraint radius
        
        for i in range(current_positions.shape[0]):
            pos = current_positions[i]
            
            # Compute distance to constraint center
            diff = pos - constraint_center
            horizontal_dist = torch.norm(diff[:2])  # x-y plane distance
            vertical_dist = torch.abs(diff[2])      # z direction distance
            
            # Simplified cone constraint: horizontal distance + weighted vertical distance
            distance_to_constraint = torch.sqrt(horizontal_dist**2 + (vertical_dist * 2)**2)
            safety_distance = torch.clamp(distance_to_constraint - constraint_radius, min=0.0)
            
            # Compute constraint normal
            if horizontal_dist > 1e-6:
                normal = diff / torch.norm(diff)
            else:
                normal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            
            # Store results
            self.safety_distances[i] = safety_distance
            self.constraint_normals[i] = normal
            self.closest_constraint_points[i] = constraint_center + normal * constraint_radius
            self.is_violating_constraint[i] = safety_distance < 0.002  # safety_margin
    
    def _update_safety_constraints(self):
        """Update safety constraint information"""
        end_effector_pos = self._get_end_effector_position()
        self._compute_safety_barrier_function(end_effector_pos)
        
    def _apply_action(self) -> None:
        """Apply processed actions to environment"""
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get simplified observations"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        # Update target point (distance-based)
        self.trajectory_manager.update_target(end_effector_pos)
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # Simplified observation: position(3) + velocity(3) + target position(3) = 9D, extended to 12D for compatibility
        obs = torch.cat([
            end_effector_pos,      # Current position [0:3]
            end_effector_vel,      # Current velocity [3:6]
            target_pos,            # Target position [6:9]
            torch.zeros(self.num_envs, 3, device=self.device)  # Padding to 12D [9:12]
        ], dim=-1)
        
        obs = torch.clamp(obs, -10.0, 10.0)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards - paper-aligned (simplified for now, will be enhanced by trainer)"""
        end_effector_pos = self._get_end_effector_position()
        end_effector_vel = self._get_end_effector_velocity()
        
        target_pos = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # Basic reward components (detailed cost function will be handled by trainer)
        
        # 1. Position tracking reward
        position_error = end_effector_pos - target_pos
        tracking_reward = -torch.sum(position_error**2, dim=-1) * 100.0
        
        # 2. Velocity penalty
        velocity_penalty = -torch.sum(end_effector_vel**2, dim=-1) * 0.01
        
        # 3. Control effort penalty
        control_penalty = -torch.sum(self.previous_robot_actions**2, dim=-1) * 0.001
        
        # 4. Safety constraint penalty
        cbf_values = self._compute_safety_barrier_function(end_effector_pos)
        safety_penalty = -cbf_values * 10.0
        
        # 5. Hard constraint violation penalty
        constraint_violation_penalty = self.is_violating_constraint.float() * (-50.0)
        
        # 6. Target reaching reward
        distance_to_target = torch.norm(end_effector_pos - target_pos, dim=-1)
        target_reached = distance_to_target < self.cfg.target_reach_threshold
        completion_reward = target_reached.float() * 20.0
        
        # 7. Final target completion reward
        final_completion = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            if self.trajectory_manager.is_final_target_reached(end_effector_pos[i]):
                final_completion[i] = 50.0  # Large reward for completing entire trajectory

        total_reward = (tracking_reward + velocity_penalty + control_penalty + 
                       safety_penalty + constraint_violation_penalty + 
                       completion_reward + final_completion)
        total_reward = torch.clamp(total_reward, -100.0, 75.0)
        
        # Store reward components for logging
        self.extras["log"] = {
            "tracking_reward": tracking_reward.mean().item(),
            "velocity_penalty": velocity_penalty.mean().item(),
            "control_penalty": control_penalty.mean().item(),
            "safety_penalty": safety_penalty.mean().item(),
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
        """Determine if episodes should terminate or truncate"""
        end_effector_pos = self._get_end_effector_position()
        
        # Termination conditions: constraint violation, falling out or completing final target
        constraint_violated = self.is_violating_constraint.clone()
        fell_out = end_effector_pos[..., 2] < -0.01
        final_target_reached = torch.tensor([
            self.trajectory_manager.is_final_target_reached(end_effector_pos[i])
            for i in range(self.num_envs)
        ], device=self.device, dtype=torch.bool)
        
        terminated = constraint_violated | fell_out | final_target_reached
        
        # Truncation condition: timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # Set initial joint positions to reach stylus position [-0.2, 0.15, 0.03]
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = 0.0   # waist
        joint_pos[:, 1] = 0.0   # shoulder  
        joint_pos[:, 2] = 0.0   # elbow
        joint_pos[:, 3] = 4.0   # yaw
        joint_pos[:, 4] = 1.2   # pitch - stylus upright
        joint_pos[:, 5] = 0.0   # roll
        
        joint_noise = sample_uniform(-0.1, 0.1, (num_resets, 6), self.device)
        joint_pos += joint_noise
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset tracking variables
        self.previous_robot_actions[env_ids] = 0.0
        self.human_forces[env_ids] = 0.0
        self.robot_forces[env_ids] = 0.0
        self.total_interaction_forces[env_ids] = 0.0
        
        # Reset constraint related variables
        self.safety_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.closest_constraint_points[env_ids] = 0.0
        self.is_violating_constraint[env_ids] = False
        
        # Reset trajectory manager
        self.trajectory_manager.reset_trajectory()

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