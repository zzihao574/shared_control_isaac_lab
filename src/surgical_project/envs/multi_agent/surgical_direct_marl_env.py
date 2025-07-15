# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surgical Direct MARL Environment with Human-Robot Collaboration."""

from __future__ import annotations

import torch
import numpy as np
from typing import Any, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """Surgical Direct MARL Environment for Human-Robot Collaborative Control.
    
    This environment simulates a surgical task where a human and robot collaborate
    to control a surgical tool (scalpel) within geometric constraints. Each agent
    (human and robot) provides force inputs that are combined to control the scalpel.
    """
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the surgical MARL environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Time step
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Simulation scaling factor
        self.sim_scale = self.cfg.simulation_scale
        self.inv_sim_scale = 1.0 / self.sim_scale
        
        # Initialize tracking variables
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_spaces["human"], device=self.device)
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.human_intentions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Agent actions (forces)
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, self.cfg.action_spaces[agent], device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Agent interaction tracking
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_history = {
            agent: torch.zeros(self.num_envs, 10, 3, device=self.device)  # Last 10 steps
            for agent in self.cfg.possible_agents
        }
        self.history_idx = 0
        
        # Human simulation state
        self.human_intention = torch.zeros(self.num_envs, 3, device=self.device)
        self.human_reaction_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Collaboration metrics
        self.trust_levels = {
            agent: torch.ones(self.num_envs, device=self.device) * self.cfg.collaboration["trust_factor"]
            for agent in self.cfg.possible_agents
        }
        self.conflict_counter = torch.zeros(self.num_envs, device=self.device)
        
        # Task tracking
        self.task_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_in_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.partial_completion_achieved = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
        
        # Debug flag for observation dimensions
        self._obs_dim_printed = False
        
        print(f"[INFO] Surgical MARL environment initialized with {self.num_envs} environments")
        print(f"[INFO] Agents: {self.cfg.possible_agents}")
        print(f"[INFO] Expected observation dimensions:")
        for agent in self.cfg.possible_agents:
            print(f"  {agent}: {self.cfg.observation_spaces[agent]}")
        print(f"[INFO] Action dimensions:")
        for agent in self.cfg.possible_agents:
            print(f"  {agent}: {self.cfg.action_spaces[agent]}")
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Create scalpel (controlled by both agents)
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
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Apply agent actions before physics step.
        
        Args:
            actions: Dictionary of actions for each agent
        """
        # Store and process agent actions
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                # Clamp actions to valid range (real world scale)
                action = torch.clamp(action, -1.0, 1.0)
                
                # Scale actions from real world to simulation scale
                action_sim_scale = action * self.sim_scale
                
                # Scale to force values
                max_force = self.cfg.max_force[agent]
                forces = action_sim_scale * max_force * self.cfg.force_scale
                
                self.agent_actions[agent] = forces
                
                # Update force history
                self.force_history[agent][:, self.history_idx] = forces
        
        # Update history index
        self.history_idx = (self.history_idx + 1) % 10
        
        # Combine agent forces
        combined_forces = self._combine_agent_forces()
        
        # Apply combined forces to scalpel
        self._apply_forces_to_scalpel(combined_forces)
        
        # Update human simulation
        self._update_human_simulation()
        
        # Update collaboration metrics
        self._update_collaboration_metrics()
        
    def _combine_agent_forces(self) -> torch.Tensor:
        """Combine forces from both agents based on collaboration model."""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # Force sharing based on configuration
        human_ratio = self.cfg.force_sharing_ratio["human"]
        robot_ratio = self.cfg.force_sharing_ratio["robot"]
        
        # Trust-based modulation
        human_trust = self.trust_levels["human"].unsqueeze(-1)
        robot_trust = self.trust_levels["robot"].unsqueeze(-1)
        
        # Normalize trust levels
        total_trust = human_trust + robot_trust
        human_weight = (human_trust / (total_trust + 1e-6)) * human_ratio
        robot_weight = (robot_trust / (total_trust + 1e-6)) * robot_ratio
        
        # Combine forces
        combined_forces = human_weight * human_forces + robot_weight * robot_forces
        
        # Add interaction coupling
        interaction_term = self.cfg.interaction_coupling * (human_forces + robot_forces) * 0.5
        combined_forces = combined_forces + interaction_term
        
        # Store interaction forces for observation
        self.interaction_forces = combined_forces.clone()
        
        return combined_forces
        
    def _apply_forces_to_scalpel(self, forces: torch.Tensor) -> None:
        """Apply combined forces to the scalpel."""
        # Reshape forces for RigidObject interface
        forces_reshaped = forces.unsqueeze(1)  # [num_envs, 1, 3]
        torques_reshaped = torch.zeros_like(forces_reshaped)
        
        # Apply external forces
        self._scalpel.set_external_force_and_torque(
            forces_reshaped,
            torques_reshaped,
            body_ids=None
        )
        
    def _update_human_simulation(self) -> None:
        """Update simulated human behavior."""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # Update reaction timer
        self.human_reaction_timer += self.dt
        
        # Update human intention at specified rate
        intention_update_period = 1.0 / self.cfg.human_dynamics["intention_update_rate"]
        update_mask = self.human_reaction_timer >= intention_update_period
        
        if update_mask.any():
            # Calculate desired direction towards target
            target_height = self.cfg.target_height
            target_direction = torch.zeros_like(scalpel_pos)
            target_direction[:, 2] = torch.sign(target_height - scalpel_pos[:, 2])
            
            # Add human noise and preferences
            noise_std = self.cfg.human_dynamics["noise_std"]
            noise = torch.randn_like(target_direction) * noise_std
            
            # Update intention with noise
            self.human_intention[update_mask] = (target_direction + noise)[update_mask]
            self.human_reaction_timer[update_mask] = 0.0
            
    def _update_collaboration_metrics(self) -> None:
        """Update collaboration metrics between agents."""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # Calculate force alignment
        force_diff = torch.norm(human_forces - robot_forces, dim=-1)
        conflict_threshold = self.cfg.collaboration["conflict_threshold"]
        
        # Update conflict counter
        conflicts = force_diff > conflict_threshold
        self.conflict_counter[conflicts] += 1
        self.conflict_counter[~conflicts] *= 0.9  # Decay conflicts
        
        # Update trust levels
        trust_decay = self.cfg.collaboration["trust_decay"]
        trust_recovery = self.cfg.collaboration["trust_recovery"]
        
        # Decay trust on conflicts
        self.trust_levels["human"][conflicts] *= trust_decay
        self.trust_levels["robot"][conflicts] *= trust_decay
        
        # Recover trust on cooperation
        self.trust_levels["human"][~conflicts] *= trust_recovery
        self.trust_levels["robot"][~conflicts] *= trust_recovery
        
        # Clamp trust levels
        for agent in self.cfg.possible_agents:
            self.trust_levels[agent] = torch.clamp(self.trust_levels[agent], 0.1, 1.0)
            
    def _apply_action(self) -> None:
        """Apply processed actions to the environment."""
        # Actions are already applied in _pre_physics_step
        pass
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Determine if episodes are terminated or truncated."""
        scalpel_pos = self._scalpel.data.root_pos_w
        
        # Check for collision with constraint walls
        collisions = self._check_collision_with_constraint(scalpel_pos)
        
        # Check if scalpel fell too low
        fell_out = scalpel_pos[:, 2] < -0.1
        
        # Termination conditions (same for both agents)
        terminated_condition = collisions | fell_out
        
        # Truncation condition (episode length)
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        # Return per-agent termination and truncation
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Calculate rewards for each agent."""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        rewards = {}
        
        for agent in self.cfg.possible_agents:
            reward_scales = self.cfg.reward_scales[agent]
            agent_reward = torch.zeros(self.num_envs, device=self.device)
            
            # 1. Collision penalty
            collisions = self._check_collision_with_constraint(scalpel_pos)
            collision_penalty = collisions.float() * reward_scales["collision_penalty"]
            agent_reward += collision_penalty
            
            # 2. Distance-based reward
            distance_reward = self._calculate_distance_reward(scalpel_pos)
            agent_reward += distance_reward * reward_scales["distance_reward"]
            
            # 3. Task completion reward
            task_reward = self._calculate_task_completion_reward(scalpel_pos)
            agent_reward += task_reward * reward_scales["task_completion"]
            
            # 4. Smoothness reward (agent-specific)
            smoothness_penalty = torch.norm(self.agent_actions[agent], dim=-1) ** 2
            agent_reward += smoothness_penalty * reward_scales["smoothness_penalty"]
            
            # 5. Collaboration reward
            collaboration_reward = self._calculate_collaboration_reward(agent)
            agent_reward += collaboration_reward * reward_scales["collaboration_reward"]
            
            # 6. Agent-specific rewards
            if agent == "human":
                # Human gets intention alignment reward
                intention_reward = self._calculate_intention_alignment_reward()
                agent_reward += intention_reward * reward_scales["intention_alignment"]
            else:  # robot
                # Robot gets adaptation reward
                adaptation_reward = self._calculate_adaptation_reward()
                agent_reward += adaptation_reward * reward_scales["adaptation_reward"]
            
            rewards[agent] = agent_reward
            
        # Store reward components for logging
        self.extras = {
            agent: {
                "collision_penalty": (collisions.float() * self.cfg.reward_scales[agent]["collision_penalty"]).mean(),
                "distance_reward": (distance_reward * self.cfg.reward_scales[agent]["distance_reward"]).mean(),
                "task_reward": (task_reward * self.cfg.reward_scales[agent]["task_completion"]).mean(),
                "collaboration_reward": self._calculate_collaboration_reward(agent).mean(),
                "total_reward": rewards[agent].mean(),
                "trust_level": self.trust_levels[agent].mean(),
            }
            for agent in self.cfg.possible_agents
        }
        
        return rewards
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for each agent."""
        scalpel_pos = self._scalpel.data.root_pos_w
        scalpel_vel = self._scalpel.data.root_lin_vel_w
        
        # Scale to real world for observations
        scalpel_pos_real = scalpel_pos * self.inv_sim_scale
        scalpel_vel_real = scalpel_vel * self.inv_sim_scale
        
        # Common observations
        constraint_distances = self._calculate_constraint_distances(scalpel_pos) * self.inv_sim_scale
        task_progress = self._calculate_task_progress(scalpel_pos_real)
        
        observations = {}
        
        for agent in self.cfg.possible_agents:
            # Base observations (shared)
            obs_components = [
                scalpel_pos_real,      # 3D position
                scalpel_vel_real,      # 3D velocity  
                constraint_distances,  # 6D constraint distances
                task_progress,         # 3D task progress info
            ]
            
            # Agent-specific observations
            if agent == "human":
                # Human observes: own intention, robot actions, trust level
                obs_components.extend([
                    self.human_intention * self.inv_sim_scale,  # 3D human intention
                ])
            else:  # robot
                # Robot observes: human actions, collaboration state, adaptation info
                obs_components.extend([
                    self.agent_actions["human"] * self.inv_sim_scale,  # 3D human forces
                ])
            
            # Add trust and collaboration info for both agents
            trust_info = torch.stack([
                self.trust_levels["human"],
                self.trust_levels["robot"],
                self.conflict_counter * 0.1,  # Normalized conflict level
            ], dim=-1)
            obs_components.append(trust_info)
            
            # Concatenate all observations
            obs = torch.cat(obs_components, dim=-1)
            
            # Print observation dimension for debugging
            if not hasattr(self, '_obs_dim_printed'):
                print(f"[DEBUG] {agent} observation components:")
                for i, comp in enumerate(obs_components):
                    print(f"  Component {i}: shape {comp.shape}")
                print(f"  Total observation dim: {obs.shape[-1]}")
                self._obs_dim_printed = True
            
            # Clamp observations for numerical stability
            obs = torch.clamp(obs, -10.0, 10.0)
            
            observations[agent] = obs
            
        return observations
        
    def _get_states(self) -> torch.Tensor:
        """Get global state for centralized training."""
        scalpel_pos = self._scalpel.data.root_pos_w * self.inv_sim_scale
        scalpel_vel = self._scalpel.data.root_lin_vel_w * self.inv_sim_scale
        
        # Global state includes all agent actions and system state
        global_state = torch.cat([
            scalpel_pos,                                    # 3D
            scalpel_vel,                                    # 3D
            self.agent_actions["human"] * self.inv_sim_scale,  # 3D
            self.agent_actions["robot"] * self.inv_sim_scale,  # 3D
            self.interaction_forces * self.inv_sim_scale,      # 3D
            self.human_intention * self.inv_sim_scale,         # 3D
            torch.stack([
                self.trust_levels["human"],
                self.trust_levels["robot"],
                self.conflict_counter * 0.1,
                self.steps_in_target.float() * 0.01,
                self.task_completed.float(),
                torch.zeros(self.num_envs, device=self.device),  # Padding to reach 24D
            ], dim=-1)  # 6D
        ], dim=-1)
        
        return global_state
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        super()._reset_idx(env_ids)
        
        # Reset scalpel to random starting position
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
        
        # Reset agent tracking variables
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
            self.force_history[agent][env_ids] = 0.0
            self.trust_levels[agent][env_ids] = self.cfg.collaboration["trust_factor"]
        
        # Reset task tracking
        self.task_completed[env_ids] = False
        self.steps_in_target[env_ids] = 0
        self.partial_completion_achieved[env_ids] = False
        self.conflict_counter[env_ids] = 0.0
        
        # Reset human simulation
        self.human_intention[env_ids] = 0.0
        self.human_reaction_timer[env_ids] = 0.0
        self.interaction_forces[env_ids] = 0.0
        
    # Helper methods (same as single agent but adapted for MARL)
    def _check_collision_with_constraint(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Check if scalpel collides with constraint walls."""
        x, y, z = scalpel_pos[:, 0], scalpel_pos[:, 1], scalpel_pos[:, 2]
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # Calculate inner radius at current height
        height_ratio = torch.clamp(z / self.cfg.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.cfg.constraint_inner_radius_min + 
            height_ratio * (self.cfg.constraint_inner_radius_max - self.cfg.constraint_inner_radius_min)
        )
        
        sphere_radius = 0.02  # 20mm sphere radius
        
        # Collision conditions
        inside_collision = radial_dist < (inner_radius_at_z - sphere_radius)
        below_collision = z < -sphere_radius
        above_collision = z > (self.cfg.constraint_height + sphere_radius)
        
        return inside_collision | below_collision | above_collision
        
    def _calculate_distance_reward(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate reward based on distance to target region."""
        target_z = self.cfg.target_height
        z_distance = torch.abs(scalpel_pos[:, 2] - target_z)
        distance_reward = torch.exp(-z_distance * 10.0)
        return distance_reward
        
    def _calculate_task_completion_reward(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate reward for task completion with progressive rewards."""
        target_z = self.cfg.target_height
        target_radius = self.cfg.task_completion["target_radius"]
        
        # Check if in target region
        in_target = torch.abs(scalpel_pos[:, 2] - target_z) < target_radius
        
        # Update step counter
        self.steps_in_target[in_target] += 1
        self.steps_in_target[~in_target] = 0
        
        # Progressive completion rewards
        task_reward = torch.zeros(self.num_envs, device=self.device)
        
        for i, (steps, reward) in enumerate(zip(
            self.cfg.task_completion["partial_completion_steps"],
            self.cfg.task_completion["partial_completion_rewards"]
        )):
            newly_achieved = (self.steps_in_target >= steps) & (~self.partial_completion_achieved[:, i])
            task_reward[newly_achieved] += reward
            self.partial_completion_achieved[newly_achieved, i] = True
        
        # Final completion
        completion_steps = self.cfg.task_completion["completion_time"]
        newly_completed = (self.steps_in_target >= completion_steps) & (~self.task_completed)
        self.task_completed |= newly_completed
        task_reward[newly_completed] += self.cfg.task_completion["max_completion_bonus"]
        
        # Small continuous reward for being in target
        task_reward[in_target] += 1.0
        
        return task_reward
        
    def _calculate_collaboration_reward(self, agent: str) -> torch.Tensor:
        """Calculate collaboration reward for specific agent."""
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # Force alignment reward
        force_alignment = -torch.norm(human_forces - robot_forces, dim=-1)
        alignment_reward = torch.exp(force_alignment * 0.5)
        
        # Trust-based reward
        trust_reward = self.trust_levels[agent] * 2.0
        
        # Conflict penalty
        conflict_penalty = -self.conflict_counter * 0.1
        
        collaboration_reward = alignment_reward + trust_reward + conflict_penalty
        
        return collaboration_reward
        
    def _calculate_intention_alignment_reward(self) -> torch.Tensor:
        """Calculate reward for human intention alignment (human agent specific)."""
        # Measure how well robot actions align with human intentions
        human_intent = self.human_intention
        robot_actions = self.agent_actions["robot"]
        
        # Normalize vectors
        intent_norm = torch.norm(human_intent, dim=-1, keepdim=True) + 1e-6
        action_norm = torch.norm(robot_actions, dim=-1, keepdim=True) + 1e-6
        
        intent_normalized = human_intent / intent_norm
        action_normalized = robot_actions / action_norm
        
        # Calculate alignment (dot product)
        alignment = torch.sum(intent_normalized * action_normalized, dim=-1)
        alignment_reward = torch.clamp(alignment, 0.0, 1.0) * 2.0
        
        return alignment_reward
        
    def _calculate_adaptation_reward(self) -> torch.Tensor:
        """Calculate adaptation reward (robot agent specific)."""
        # Reward robot for adapting to human behavior
        human_forces = self.agent_actions["human"]
        robot_forces = self.agent_actions["robot"]
        
        # Calculate adaptation metric based on force history
        human_history = self.force_history["human"]
        robot_history = self.force_history["robot"]
        
        # Measure how robot adjusts to human patterns
        human_variance = torch.var(human_history, dim=1).mean(dim=-1)
        robot_adaptation = torch.norm(robot_forces - human_forces, dim=-1)
        
        # Lower adaptation distance when human is consistent = better
        adaptation_reward = torch.exp(-robot_adaptation * 0.5) * (1.0 + human_variance)
        
        return adaptation_reward
        
    def _calculate_constraint_distances(self, scalpel_pos: torch.Tensor) -> torch.Tensor:
        """Calculate distances to constraint boundaries."""
        x, y, z = scalpel_pos[:, 0], scalpel_pos[:, 1], scalpel_pos[:, 2]
        radial_dist = torch.sqrt(x**2 + y**2)
        
        # Calculate inner radius at current height
        height_ratio = torch.clamp(z / self.cfg.constraint_height, 0.0, 1.0)
        inner_radius_at_z = (
            self.cfg.constraint_inner_radius_min + 
            height_ratio * (self.cfg.constraint_inner_radius_max - self.cfg.constraint_inner_radius_min)
        )
        
        # Distance measurements
        dist_to_inner_wall = radial_dist - inner_radius_at_z
        dist_to_bottom = z
        dist_to_top = self.cfg.constraint_height - z
        dist_to_center_xy = radial_dist
        dist_to_target_z = torch.abs(z - self.cfg.target_height)
        dist_to_axis = radial_dist
        
        return torch.stack([
            dist_to_inner_wall,
            dist_to_bottom, 
            dist_to_top,
            dist_to_center_xy,
            dist_to_target_z,
            dist_to_axis
        ], dim=-1)
        
    def _calculate_task_progress(self, scalpel_pos_real: torch.Tensor) -> torch.Tensor:
        """Calculate task progress information."""
        target_height_real = self.cfg.target_height * self.inv_sim_scale
        
        # Progress metrics
        height_progress = 1.0 - torch.abs(scalpel_pos_real[:, 2] - target_height_real) / 0.2  # Normalized
        height_progress = torch.clamp(height_progress, 0.0, 1.0)
        
        # Completion progress
        completion_progress = self.steps_in_target.float() / self.cfg.task_completion["completion_time"]
        completion_progress = torch.clamp(completion_progress, 0.0, 1.0)
        
        # Overall task status
        task_status = self.task_completed.float()
        
        return torch.stack([height_progress, completion_progress, task_status], dim=-1)