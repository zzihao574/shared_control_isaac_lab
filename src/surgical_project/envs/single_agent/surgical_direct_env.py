# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified Surgical Direct Environment with Human-Robot Shared Control."""

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
    """Simple trajectory manager for surgical task guidance."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.current_index = 0
        self.trajectory_points = None
        self.trajectory_velocities = None
        self._generate_default_trajectory()
        
    def _generate_default_trajectory(self):
        """Generate a simple spiral trajectory."""
        num_points = 200
        turns = 2.0
        radius_range = (0.01, 0.04)  # 1-4cm
        height_range = (0.002, 0.08)  # 2-80mm
        
        t = np.linspace(0, turns * 2 * np.pi, num_points)
        
        # Spiral parameters
        r_min, r_max = radius_range
        h_min, h_max = height_range
        
        # Create spiral trajectory
        radius = np.linspace(r_min, r_max, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.linspace(h_min, h_max, num_points)
        
        self.trajectory_points = torch.tensor(np.column_stack([x, y, z]), 
                                            dtype=torch.float32, device=self.device)
        
        # Calculate velocities (finite differences)
        velocities = torch.diff(self.trajectory_points, dim=0, prepend=self.trajectory_points[:1])
        self.trajectory_velocities = velocities
        
        print(f"[INFO] Generated spiral trajectory with {num_points} points")
        
    def get_current_target(self):
        """Get current trajectory target position and velocity."""
        if self.trajectory_points is None:
            return torch.zeros(3, device=self.device), torch.zeros(3, device=self.device)
            
        target_pos = self.trajectory_points[self.current_index]
        target_vel = self.trajectory_velocities[self.current_index]
        
        return target_pos, target_vel
        
    def advance_trajectory(self, step_size: int = 1):
        """Advance to next trajectory point."""
        if self.trajectory_points is not None:
            self.current_index = min(
                self.current_index + step_size,
                len(self.trajectory_points) - 1
            )
            
    def reset_trajectory(self):
        """Reset trajectory to beginning."""
        self.current_index = 0


class SurgicalDirectEnv(DirectRLEnv):
    """Simplified Surgical Direct Environment for Human-Robot Shared Control Training."""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the surgical environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Time step
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Initialize tracking variables
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Constraint geometry (real-world scale)
        self.constraint_inner_radius_min = self.cfg.constraint_inner_radius_min
        self.constraint_inner_radius_max = self.cfg.constraint_inner_radius_max
        self.constraint_height = self.cfg.constraint_height
        
        # Simplified constraint computation state
        self.constraint_distances = torch.zeros(self.num_envs, device=self.device)
        self.constraint_normals = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_overlapping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Task completion tracking
        self.task_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Trajectory manager
        self.trajectory_manager = TrajectoryManager(self.device)
        
        # Human workspace parameters (20cm radius around origin)
        self.human_workspace_center = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.human_workspace_radius = 0.2  # 20cm radius
        
        print(f"[INFO] Surgical environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation space: {self.cfg.observation_space}D") 
        print(f"  - Action space: {self.cfg.action_space}D")
        print(f"  - Human workspace: {self.human_workspace_radius*1000:.0f}mm radius")
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create scalpel (sphere)
        self._scalpel = RigidObject(self.cfg.scalpel)
        self.scene.rigid_objects["scalpel"] = self._scalpel
        
        # Create constraint
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
        """Apply actions before physics step."""
        # Clamp actions to valid range
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Scale actions to force values
        forces = actions * self.cfg.max_force * self.cfg.force_scale
        
        # Store actions for smoothness calculation
        self.previous_actions = actions.clone()
        
        # Apply forces to scalpel
        self._apply_forces_to_scalpel(forces)
        
        # Update constraint computation
        self._update_constraints()
        
        # Advance trajectory every 10 steps
        if self.episode_length_buf[0] % 10 == 0:
            self.trajectory_manager.advance_trajectory()
        
    def _apply_forces_to_scalpel(self, forces: torch.Tensor) -> None:
        """Apply forces to the scalpel sphere center."""
        forces_reshaped = forces.unsqueeze(1)  # [num_envs, 1, 3]
        torques_reshaped = torch.zeros_like(forces_reshaped)  # No torques
        
        self._scalpel.set_external_force_and_torque(
            forces_reshaped,
            torques_reshaped,
            body_ids=None
        )
        
    def _update_constraints(self):
        """Update constraint information using simplified geometry."""
        scalpel_pos = self._scalpel.data.root_pos_w
        
        x, y, z = scalpel_pos[..., 0], scalpel_pos[..., 1], scalpel_pos[..., 2]
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # Linear interpolation for cone shape
        height_ratio = torch.clamp(z / self.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.constraint_inner_radius_min + 
            height_ratio * (self.constraint_inner_radius_max - self.constraint_inner_radius_min)
        )
        
        # Distance to inner wall
        self.constraint_distances = torch.abs(radial_dist - inner_radius_at_z)
        
        # Normal vector (pointing outward from constraint)
        normal_x = torch.where(radial_dist > 1e-6, x / radial_dist, torch.ones_like(x))
        normal_y = torch.where(radial_dist > 1e-6, y / radial_dist, torch.zeros_like(y))
        normal_z = torch.zeros_like(z)
        
        self.constraint_normals = torch.stack([normal_x, normal_y, normal_z], dim=-1)
        
        # Simple overlap check
        sphere_radius = 0.002  # 2mm
        self.is_overlapping = self.constraint_distances < sphere_radius
        
    def _apply_action(self) -> None:
        """Apply processed actions to the environment."""
        pass
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if episodes are terminated or truncated."""
        # Collision termination
        terminated = self.is_overlapping.clone()
        
        # Check if fell too low
        scalpel_pos = self._scalpel.data.root_pos_w
        fell_out = scalpel_pos[..., 2] < -0.01  # Below -10mm
        terminated = terminated | fell_out
        
        # Truncation condition
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
        
    def _get_rewards(self) -> torch.Tensor:
        """Calculate simplified reward based on task performance."""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # Initialize total reward
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Collision penalty (moderate penalty)
        collision_penalty = self.is_overlapping.float() * (-5.0)  # Reduced from -10.0
        total_reward += collision_penalty
        
        # 2. Distance to constraint reward (encourage staying at safe distance)
        safe_distance = 0.005  # 5mm safe distance
        # Reward being at safe distance, penalize being too close or too far
        distance_error = torch.abs(self.constraint_distances - safe_distance)
        distance_reward = torch.exp(-distance_error * 100.0)  # Exponential reward for being at safe distance
        total_reward += distance_reward * 2.0
        
        # 3. Trajectory tracking reward
        trajectory_reward = self._calculate_trajectory_reward(scalpel_pos, scalpel_vel)
        total_reward += trajectory_reward * 3.0  # Reduced from 5.0
        
        # 4. Human workspace compliance reward
        workspace_reward = self._calculate_workspace_reward(scalpel_pos)
        total_reward += workspace_reward * 1.0  # Reduced from 2.0
        
        # 5. Smoothness reward (encourage smooth actions)
        action_magnitude = torch.norm(self.previous_actions, dim=-1)
        smoothness_reward = torch.exp(-action_magnitude * 2.0)  # Reward small actions
        total_reward += smoothness_reward * 0.5
        
        # 6. Velocity penalty (encourage controlled motion)
        velocity_magnitude = torch.norm(scalpel_vel, dim=-1)
        velocity_reward = torch.exp(-velocity_magnitude * 5.0)  # Reward slow motion
        total_reward += velocity_reward * 0.5
        
        # 7. Height reward (encourage staying above ground)
        height_reward = torch.clamp(scalpel_pos[..., 2] * 20.0, 0, 2.0)  # Reward for positive height
        total_reward += height_reward
        
        # Ensure reward is always reasonable
        total_reward = torch.clamp(total_reward, -10.0, 10.0)
        
        # Store reward components for logging
        self.extras["log"] = {
            "collision_penalty": collision_penalty.mean().item(),
            "distance_reward": distance_reward.mean().item(),
            "trajectory_reward": trajectory_reward.mean().item(),
            "workspace_reward": workspace_reward.mean().item(),
            "smoothness_reward": smoothness_reward.mean().item(),
            "velocity_reward": velocity_reward.mean().item(),
            "height_reward": height_reward.mean().item(),
            "total_reward": total_reward.mean().item(),
            "constraint_distance": self.constraint_distances.mean().item(),
            "trajectory_index": self.trajectory_manager.current_index,
            "overlap_rate": self.is_overlapping.float().mean().item(),
            "scalpel_height": scalpel_pos[..., 2].mean().item(),
            "velocity_norm": velocity_magnitude.mean().item(),
        }
        
        return total_reward
        
    def _calculate_trajectory_reward(self, scalpel_pos: torch.Tensor, scalpel_vel: torch.Tensor) -> torch.Tensor:
        """Calculate reward for trajectory tracking."""
        target_pos, target_vel = self.trajectory_manager.get_current_target()
        
        # Expand target for all environments
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
        target_vel = target_vel.unsqueeze(0).expand(self.num_envs, -1)
        
        # Position tracking reward (use smaller scaling factor)
        pos_error = torch.norm(scalpel_pos - target_pos, dim=-1)
        pos_reward = torch.exp(-pos_error * 10.0)  # Reduced from 30.0
        
        # Velocity tracking reward (use smaller scaling factor)
        vel_error = torch.norm(scalpel_vel - target_vel, dim=-1)
        vel_reward = torch.exp(-vel_error * 5.0)  # Reduced from 10.0
        
        # Combined trajectory reward (weighted combination)
        trajectory_reward = 0.7 * pos_reward + 0.3 * vel_reward
        
        return trajectory_reward
        
    def _calculate_workspace_reward(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate reward for staying within human workspace."""
        # Distance from workspace center (only x-y, ignore z)
        workspace_distance = torch.norm(scalpel_pos[..., :2] - self.human_workspace_center[:2], dim=-1)
        
        # Reward for staying within workspace (exponential decay)
        workspace_reward = torch.exp(-workspace_distance * 5.0)  # Exponential reward
        
        # Additional penalty for being far outside workspace
        outside_distance = torch.clamp(workspace_distance - self.human_workspace_radius, 0, 0.1)
        outside_penalty = outside_distance * (-5.0)
        
        return workspace_reward + outside_penalty
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        # Reset scalpel to random starting position within human workspace
        num_resets = len(env_ids)
        
        # Random position within workspace (but above constraint)
        angles = sample_uniform(0, 2*math.pi, (num_resets,), self.device)
        radii = sample_uniform(0.02, 0.15, (num_resets,), self.device)  # 2-15cm from center
        
        scalpel_pos = torch.zeros((num_resets, 3), device=self.device)
        scalpel_pos[:, 0] = radii * torch.cos(angles)
        scalpel_pos[:, 1] = radii * torch.sin(angles)
        scalpel_pos[:, 2] = sample_uniform(0.015, 0.025, (num_resets,), self.device)  # 15-25mm height
        
        # Reset velocity
        scalpel_vel = torch.zeros((num_resets, 3), device=self.device)
        
        # Create pose (position + quaternion)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        pose = torch.cat([scalpel_pos, quaternion], dim=-1)
        
        # Create velocity (linear + angular)
        velocity = torch.cat([scalpel_vel, torch.zeros_like(scalpel_vel)], dim=-1)
        
        # Apply reset
        self._scalpel.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._scalpel.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        
        # Reset tracking variables
        self.task_completed[env_ids] = False
        self.previous_actions[env_ids] = 0.0
        self.interaction_forces[env_ids] = 0.0
        self.constraint_distances[env_ids] = 0.0
        self.constraint_normals[env_ids] = 0.0
        self.is_overlapping[env_ids] = False
        
        # Reset trajectory
        self.trajectory_manager.reset_trajectory()
        
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get simplified environment observations (19D total)."""
        scalpel_pos = self._scalpel.data.root_pos_w      # [num_envs, 3]
        scalpel_vel = self._scalpel.data.root_lin_vel_w  # [num_envs, 3]
        
        # Get current trajectory target
        target_pos, target_vel = self.trajectory_manager.get_current_target()
        target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        target_vel = target_vel.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        
        # Simplified constraint information (5D total)
        constraint_info = torch.cat([
            self.constraint_distances.unsqueeze(-1),      # [num_envs, 1] Distance to constraint
            self.constraint_normals,                      # [num_envs, 3] Normal vectors  
            self.is_overlapping.float().unsqueeze(-1),    # [num_envs, 1] Overlap flag
        ], dim=-1)  # [num_envs, 5]
        
        # Human workspace information (2D)
        workspace_distance = torch.norm(scalpel_pos[..., :2] - self.human_workspace_center[:2], dim=-1)
        workspace_info = torch.cat([
            workspace_distance.unsqueeze(-1),  # [num_envs, 1] Distance from workspace center
            (workspace_distance <= self.human_workspace_radius).float().unsqueeze(-1),  # [num_envs, 1] Within workspace flag
        ], dim=-1)  # [num_envs, 2]
        
        # Concatenate all observations (total: 3+3+3+3+5+2 = 19D)
        obs = torch.cat([
            scalpel_pos,           # [num_envs, 3] Current position
            scalpel_vel,           # [num_envs, 3] Current velocity
            target_pos,            # [num_envs, 3] Trajectory target position
            target_vel,            # [num_envs, 3] Trajectory target velocity
            constraint_info,       # [num_envs, 5] Simplified constraint information
            workspace_info,        # [num_envs, 2] Human workspace information
        ], dim=-1)  # [num_envs, 19]
        
        # Clamp observations for numerical stability
        obs = torch.clamp(obs, -10.0, 10.0)
        
        # Verify observation dimension
        assert obs.shape[-1] == self.cfg.observation_space, f"Observation dimension mismatch: {obs.shape[-1]} != {self.cfg.observation_space}"
        
        return {"policy": obs}