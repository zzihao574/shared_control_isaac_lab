# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical Direct Environment with Human-Robot Shared Control."""

from __future__ import annotations

import torch
import numpy as np
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .surgical_direct_env_cfg import SurgicalDirectEnvCfg


class SurgicalDirectEnv(DirectRLEnv):
    """Surgical Direct Environment for Human-Robot Shared Control Training.
    
    This environment simulates a surgical task where a human and robot collaborate
    to control a surgical tool (scalpel) within geometric constraints.
    """
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the surgical environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Time step
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Simulation scaling factor (simulation is 10x larger than real world)
        self.sim_scale = self.cfg.simulation_scale
        self.inv_sim_scale = 1.0 / self.sim_scale
        
        # Initialize tracking variables
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.human_intentions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Constraint geometry (scaled by 0.05)
        self.constraint_inner_radius_min = self.cfg.constraint_inner_radius_min
        self.constraint_inner_radius_max = self.cfg.constraint_inner_radius_max
        self.constraint_height = self.cfg.constraint_height
        
        # Task completion tracking
        self.task_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_in_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        print(f"[INFO] Surgical environment initialized with {self.num_envs} environments")
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create scalpel (sphere)
        self._scalpel = RigidObject(self.cfg.scalpel)
        self.scene.rigid_objects["scalpel"] = self._scalpel
        
        # Create constraint (cone)
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        # Create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step.
        
        Note: Actions come from network in real-world scale, need to be scaled up for simulation.
        """
        # Clamp actions to valid range (real world scale)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Scale actions from real world to simulation scale
        actions_sim_scale = actions * self.sim_scale
        
        # Scale actions to force values (now in simulation scale)
        forces = actions_sim_scale * self.cfg.max_force * self.cfg.force_scale
        
        # Store original actions (real world scale) for smoothness calculation
        self.previous_actions = actions.clone()
        
        # Apply forces to scalpel sphere center (simulation scale)
        self._apply_forces_to_scalpel(forces)
        
    def _apply_forces_to_scalpel(self, forces: torch.Tensor) -> None:
        """Apply forces to the scalpel sphere center.
        
        Args:
            forces: Force tensor of shape [num_envs, 3]
            
        Note: Forces are applied to the root body. The USD model should have 
        the sphere positioned 20mm above the scalpel_simple origin.
        """
        # Reshape forces to match expected shape [num_envs, num_bodies, 3]
        # For RigidObject, num_bodies = 1
        forces_reshaped = forces.unsqueeze(1)  # [num_envs, 1, 3]
        torques_reshaped = torch.zeros_like(forces_reshaped)  # [num_envs, 1, 3] - No torques
        
        # Apply external forces to the scalpel root body
        self._scalpel.set_external_force_and_torque(
            forces_reshaped,  # Linear forces (xyz) with correct shape
            torques_reshaped,  # No torques (prevent rotation) with correct shape
            body_ids=None  # Apply to all bodies (in this case, just the root body)
        )
        
    def _apply_action(self) -> None:
        """Apply processed actions to the environment."""
        # Actions are already applied in _pre_physics_step
        pass
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if episodes are terminated or truncated."""
        # Get scalpel position
        scalpel_pos = self._scalpel.data.root_pos_w
        
        # Check for collision with constraint walls
        collisions = self._check_collision_with_constraint(scalpel_pos)
        
        # Check if scalpel fell too low
        fell_out = scalpel_pos[:, 2] < -0.1  # Below -100mm
        
        # Termination conditions
        terminated = collisions | fell_out
        
        # Truncation condition (episode length)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _get_rewards(self) -> torch.Tensor:
        """Calculate reward based on task performance."""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # Initialize total reward
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Collision penalty
        collisions = self._check_collision_with_constraint(scalpel_pos)
        collision_penalty = collisions.float() * self.cfg.collision_penalty_scale
        total_reward += collision_penalty
        
        # 2. Distance-based reward (encourage moving toward target)
        distance_reward = self._calculate_distance_reward(scalpel_pos)
        total_reward += distance_reward * self.cfg.distance_reward_scale
        
        # 3. Task completion reward
        task_reward = self._calculate_task_completion_reward(scalpel_pos)
        total_reward += task_reward * self.cfg.task_completion_scale
        
        # 4. Smoothness reward (penalize large action changes)
        smoothness_penalty = torch.norm(self.previous_actions, dim=-1) ** 2
        total_reward += smoothness_penalty * self.cfg.smoothness_reward_scale
        
        # 5. Human collaboration reward
        collaboration_reward = self._calculate_collaboration_reward()
        total_reward += collaboration_reward * self.cfg.human_collaboration_scale
        
        # Store reward components for logging
        self.extras["log"] = {
            "collision_penalty": collision_penalty.mean(),
            "distance_reward": distance_reward.mean(), 
            "task_reward": task_reward.mean(),
            "smoothness_penalty": smoothness_penalty.mean(),
            "collaboration_reward": collaboration_reward.mean(),
            "total_reward": total_reward.mean(),
        }
        
        return total_reward
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        # Reset scalpel to random starting position above constraint
        scalpel_pos = torch.zeros((len(env_ids), 3), device=self.device)
        scalpel_pos[:, 0] = sample_uniform(-0.05, 0.05, (len(env_ids),), self.device)  # ±50mm in X
        scalpel_pos[:, 1] = sample_uniform(-0.05, 0.05, (len(env_ids),), self.device)  # ±50mm in Y  
        scalpel_pos[:, 2] = sample_uniform(0.15, 0.25, (len(env_ids),), self.device)   # 150-250mm in Z
        
        # Reset velocity
        scalpel_vel = torch.zeros((len(env_ids), 3), device=self.device)
        
        # Apply reset
        self._scalpel.write_root_pose_to_sim(
            root_pose=torch.cat([scalpel_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)], dim=-1),
            env_ids=env_ids
        )
        self._scalpel.write_root_velocity_to_sim(
            root_velocity=torch.cat([scalpel_vel, torch.zeros_like(scalpel_vel)], dim=-1),
            env_ids=env_ids
        )
        
        # Reset tracking variables
        self.task_completed[env_ids] = False
        self.steps_in_target[env_ids] = 0
        self.previous_actions[env_ids] = 0.0
        self.interaction_forces[env_ids] = 0.0
        self.human_intentions[env_ids] = 0.0
        
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get environment observations.
        
        Note: All observations are scaled down from simulation to real-world scale for the network.
        """
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # Scale positions and velocities from simulation to real world scale
        scalpel_pos_real = scalpel_pos * self.inv_sim_scale
        scalpel_vel_real = scalpel_vel * self.inv_sim_scale
        
        # Calculate distance to constraint boundaries (in real world scale)
        constraint_distances = self._calculate_constraint_distances(scalpel_pos) * self.inv_sim_scale
        
        # Estimate human intention (in real world scale)
        human_intent = self._estimate_human_intention(scalpel_pos_real, scalpel_vel_real)
        
        # Concatenate all observations (all in real world scale)
        obs = torch.cat([
            scalpel_pos_real,      # 3D position (real world scale)
            scalpel_vel_real,      # 3D velocity (real world scale)
            constraint_distances,  # 6D constraint distances (real world scale)
            human_intent,          # 3D human intention (real world scale)
        ], dim=-1)
        
        # Clamp observations for numerical stability
        obs = torch.clamp(obs, -10.0, 10.0)
        
        return {"policy": obs}
        
    def _check_collision_with_constraint(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Check if scalpel collides with constraint walls."""
        # Extract x, y, z coordinates
        x, y, z = scalpel_pos[:, 0], scalpel_pos[:, 1], scalpel_pos[:, 2]
        
        # Calculate radial distance from center
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # Calculate inner radius at current height (linear interpolation)
        # At z=0: inner_radius = constraint_inner_radius_min
        # At z=constraint_height: inner_radius = constraint_inner_radius_max
        height_ratio = torch.clamp(z / self.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.constraint_inner_radius_min + 
            height_ratio * (self.constraint_inner_radius_max - self.constraint_inner_radius_min)
        )
        
        # Sphere radius is 20mm = 0.02m
        sphere_radius = 0.02
        
        # Check collision conditions
        # 1. Inside inner wall
        inside_collision = radial_dist < (inner_radius_at_z - sphere_radius)
        # 2. Below constraint base
        below_collision = z < -sphere_radius
        # 3. Above constraint top  
        above_collision = z > (self.constraint_height + sphere_radius)
        
        return inside_collision | below_collision | above_collision
        
    def _calculate_distance_reward(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate reward based on distance to target region.
        
        Note: This uses simulation scale positions for internal reward calculation.
        """
        target_z = self.cfg.target_height  # Already in simulation scale
        z_distance = torch.abs(scalpel_pos[:, 2] - target_z)
        
        # Exponential reward that decreases with distance
        distance_reward = torch.exp(-z_distance * 10.0)
        
        return distance_reward
        
    def _calculate_task_completion_reward(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate reward for task completion.
        
        Note: This uses simulation scale positions for internal reward calculation.
        """
        # Check if scalpel is in target region (simulation scale)
        target_z = self.cfg.target_height  # Already in simulation scale
        in_target = torch.abs(scalpel_pos[:, 2] - target_z) < 0.01  # Within 10mm in simulation scale
        
        # Update step counter
        self.steps_in_target[in_target] += 1
        self.steps_in_target[~in_target] = 0
        
        # Task completed if stayed in target for 50 steps (about 1 second)
        newly_completed = (self.steps_in_target >= 50) & (~self.task_completed)
        self.task_completed |= newly_completed
        
        # Large reward for task completion
        task_reward = newly_completed.float() * 100.0
        
        # Small continuous reward for being in target
        task_reward += in_target.float() * 1.0
        
        return task_reward
        
    def _calculate_collaboration_reward(self) -> torch.Tensor:
        """Calculate reward for effective human-robot collaboration."""
        # Simplified collaboration metric
        # In practice, this would consider human input and robot response alignment
        
        # For now, provide small reward for smooth operation
        collaboration_reward = torch.ones(self.num_envs, device=self.device) * 0.1
        
        return collaboration_reward
        
    def _calculate_constraint_distances(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate distances to constraint boundaries."""
        x, y, z = scalpel_pos[:, 0], scalpel_pos[:, 1], scalpel_pos[:, 2]
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # Calculate inner radius at current height
        height_ratio = torch.clamp(z / self.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.constraint_inner_radius_min + 
            height_ratio * (self.constraint_inner_radius_max - self.constraint_inner_radius_min)
        )
        
        # Distance measurements
        dist_to_inner_wall = radial_dist - inner_radius_at_z
        dist_to_bottom = z
        dist_to_top = self.constraint_height - z
        dist_to_center_xy = radial_dist
        dist_to_target_z = torch.abs(z - self.cfg.target_height)
        dist_to_axis = radial_dist  # Distance to central axis
        
        return torch.stack([
            dist_to_inner_wall,
            dist_to_bottom, 
            dist_to_top,
            dist_to_center_xy,
            dist_to_target_z,
            dist_to_axis
        ], dim=-1)
        
    def _estimate_human_intention(self, scalpel_pos_real: torch.Tensor, scalpel_vel_real: torch.Tensor) -> torch.Tensor:
        """Estimate human intention based on current state.
        
        Args:
            scalpel_pos_real: Scalpel position in real world scale
            scalpel_vel_real: Scalpel velocity in real world scale
            
        Returns:
            Human intention in real world scale
        """
        # Simplified human intention model
        # In practice, this would use the human dynamics model from shared_control.py
        
        # For now, assume human wants to move toward target height (in real world scale)
        target_height_real = self.cfg.target_height * self.inv_sim_scale  # Convert to real world scale
        target_direction = torch.zeros_like(scalpel_pos_real)
        target_direction[:, 2] = torch.sign(target_height_real - scalpel_pos_real[:, 2])
        
        # Scale by distance to target (real world scale)
        distance_to_target = torch.abs(scalpel_pos_real[:, 2] - target_height_real)
        intention_strength = torch.clamp(distance_to_target * 5.0, 0.0, 1.0)
        
        human_intention = target_direction * intention_strength.unsqueeze(-1)
        
        return human_intention