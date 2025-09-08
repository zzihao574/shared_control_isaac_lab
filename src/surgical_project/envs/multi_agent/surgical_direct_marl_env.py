# surgical_direct_marl_env.py - Modified for unified progress management and MetricsHub

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym  
from typing import Any, Dict, List, Optional
from collections import defaultdict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
from .utils import CompleteConstraintChecker, TrajectoryManager, RewardLogger


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """
    Human-robot collaborative surgical MARL environment.
    MODIFIED: Removed extras['episode'] and extras['log'] dependencies for MetricsHub integration
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Trajectory following with progress tracking
    - Comprehensive reward system with safety considerations
    - Performance evaluation and milestone tracking
    - Console logging via utils (configured by YAML)
    """
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize surgical MARL environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (optional)
            **kwargs: Additional arguments
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Progress manager will be injected by trainer
        self.progress_manager = None
        
        # Load configuration and initialize core parameters
        self._setup_core_configuration()
        
        # Initialize state variables and caches
        self._initialize_state_variables()
        
        # Initialize utility managers and components
        self._initialize_components()
        
        # Restore gymnasium spaces setup - both Isaac Lab and Gymnasium need this
        self._setup_gymnasium_spaces()
        
    def set_progress_manager(self, pm):
        """Inject progress manager reference."""
        self.progress_manager = pm
        
    def _setup_core_configuration(self) -> None:
        """Load and setup core configuration parameters."""
        # If trainer already injected self.params, use it directly; otherwise fallback to local YAML
        if not hasattr(self, "params") or not isinstance(getattr(self, "params", None), dict):
            self.params = self._load_training_params()
            
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Load and cache constraint parameters
        self._load_constraint_parameters()
        
        # Load safety and CBF parameters  
        self._load_safety_parameters()
        
        # Load termination conditions
        self._load_termination_parameters()
        
        print(f"[INFO] Episode length controlled by cfg.episode_length_s: {self.cfg.episode_length_s}s")
    
    def _load_training_params(self) -> dict:
        """Load training parameters from YAML configuration."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_constraint_parameters(self) -> None:
        """Load and cache constraint-related parameters."""
        constraints = self.params['constraints']
        
        # Position and force limits
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        # Joint limits as tensors for efficient operations
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        # Fixed end joints configuration
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
    
    def _load_safety_parameters(self) -> None:
        """Load safety-related parameters."""        
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
    
    def _load_termination_parameters(self) -> None:
        """Load episode termination condition parameters."""
        term_config = self.params.get('termination_conditions', {})
        self.enable_z_termination = term_config.get('z_below_zero', True)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.001)
    
    def _initialize_state_variables(self) -> None:
        """Initialize all state variables and caches."""
        # Environment base positions
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Agent actions storage
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Physics interaction state (forces applied at time t)
        self._initialize_physics_state()
        
        # State cache (will be updated in _get_observations)
        self._initialize_observation_cache()
        
        # Reward components cache
        self.reward_components = {}
        
        # Body index for stylus/end-effector (will be set during scene setup)
        self.stylus_body_idx = None
    
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables (forces at time t)."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
    
    def _initialize_observation_cache(self) -> None:
        """Initialize state caching variables (updated in _get_observations)."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.normal_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_results_t1 = None
    
    def _initialize_components(self) -> None:
        """Initialize utility managers and components."""
        # Trajectory manager for path following
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        # Reward logger with milestone tracking
        self.reward_logger = RewardLogger(self.num_envs, self.device)
        
        # Constraint checker for collision detection
        self.constraint_checker = CompleteConstraintChecker(
            device=self.device, 
            collision_threshold=self.collision_threshold
        )
    
    def _setup_gymnasium_spaces(self) -> None:
        """Setup Gymnasium compatibility spaces with unlimited bounds."""
        # Use unlimited bounds, actual limits handled by environment logic
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.cfg.action_spaces[agent],), 
                dtype=np.float32
            ) for agent in self.cfg.possible_agents
        })
        
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.cfg.observation_spaces[agent],), 
                dtype=np.float32
            ) for agent in self.cfg.possible_agents
        })
        
    def _setup_scene(self):
        """Setup the simulation scene with robot and constraints."""
        # Initialize robot articulation
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        # Initialize constraint object
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        # Clone environments for parallel simulation
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """Post-scene creation setup including physics configuration."""
        super()._setup_post_scene_creation()
        
        # Initialize robot-specific configurations
        self._initialize_body_indices()

        # Update managers with robot data
        self._update_managers_with_robot_data()

    def _update_managers_with_robot_data(self) -> None:
        """Update managers with robot position data."""
        if hasattr(self, '_omni_robot'):
            # Update environment base positions from robot data
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
    def _initialize_body_indices(self):
        """Initialize body indices for end-effector/stylus identification."""
        if self.stylus_body_idx is not None:
            return  # Already cached
            
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        # Hard-coded stylus body name for performance
        target_name = "stylus"
        
        for i, name in enumerate(self._omni_robot.body_names):
            if target_name in name.lower():
                self.stylus_body_idx = i
                print(f"[INFO] Found stylus body: {name} (index {i})")
                return
        
        # Fallback to last body if stylus not found
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
            print(f"[INFO] Using fallback stylus body index: {self.stylus_body_idx}")
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Pre-physics step processing including action validation and force application."""
        # Process and validate actions - expect (num_envs, 3) from trainer
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                # Assert correct shape - no dimension handling here
                assert action.shape == (self.num_envs, 3), f"Action shape mismatch for {agent}: expected ({self.num_envs}, 3), got {action.shape}"
                
                # Apply force constraints
                max_force = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = torch.clamp(action, -max_force, max_force)
        
        # Cache forces for reward calculation (these are at time t)
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # Apply external forces to end-effector
        self._apply_external_forces()
        
        # Enforce joint constraints and fix end joints
        self._enforce_joint_constraints()
        
    def _apply_external_forces(self) -> None:
        """Apply external forces to the robot end-effector."""
        total_forces = self.robot_forces_t + self.human_forces_t
        stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]
        
        # Transform forces to local frame
        forces_local = quat_rotate_inverse(stylus_quat, total_forces)
        
        # Reshape for Isaac Lab API
        forces_with_body_dim = forces_local.unsqueeze(1)
        torques_with_body_dim = torch.zeros_like(forces_with_body_dim)
        
        self._omni_robot.set_external_force_and_torque(
            forces_with_body_dim, 
            torques_with_body_dim,
            body_ids=[self.stylus_body_idx]
        )
        
    def _enforce_joint_constraints(self) -> None:
        """Enforce joint limits and fix end joints."""
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        # Apply joint limits
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        
        # Fix end joints (wrist orientation)
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """Apply actions to simulation."""
        self._omni_robot.write_data_to_sim()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for all agents and update state cache."""
        # Update all state cache (t+1 state after physics)
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # Update constraint state (t+1 state after physics)
        current_base_positions = self._omni_robot.data.root_link_pos_w.clone()
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']
        self.normal_t1 = self.constraint_results_t1['normal_vectors']

        # Ensure constraint distances have correct shape for concatenation
        constraint_distances = self.safety_distances_t1.unsqueeze(-1)  # [num_envs] -> [num_envs, 1]

        # Concatenate observation components (7 dimensions total)
        obs = torch.cat([
            self.stylus_pos_t1,       # End-effector position (3)
            self.stylus_vel_t1,       # End-effector velocity (3) 
            constraint_distances,     # Distance measurements (1)
        ], dim=-1)                    # Total: 7 dimensions
        
        # Create observation dictionary for each agent
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = obs
            
        return observations
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """MODIFIED: Reward system without extras dependencies."""
        # Calculate individual reward components
        trajectory_reward = self._calculate_adaptive_trajectory_reward()
        progress_reward = self._calculate_progress_reward()
        potential_field_reward = self._calculate_potential_field_reward()
        force_penalties = self._calculate_force_penalties()
        z_penalty = self._calculate_z_penalty()
        completion_reward = self._calculate_completion_reward()
        time_efficiency_reward = self._calculate_time_efficiency_reward()
        
        # Get reward weights from configuration
        robot_weights = self.params['reward_parameters']['robot_weights']
        human_weights = self.params['reward_parameters']['human_weights']
        
        # Apply weights to ALL components including global ones
        rewards = {}
        rewards["robot"] = (
            trajectory_reward * robot_weights['trajectory_tracking'] +
            progress_reward * robot_weights['progress'] +
            potential_field_reward * robot_weights['potential_field'] +
            force_penalties['robot'] * robot_weights['force_efficiency'] +
            force_penalties['human'] * robot_weights['human_awareness'] +
            z_penalty * robot_weights.get('z_penalty', 0.0) +
            completion_reward * robot_weights.get('completion_reward', 0.0) +
            time_efficiency_reward * robot_weights.get('time_efficiency', 0.0)
        )
        
        rewards["human"] = (
            trajectory_reward * human_weights['trajectory_tracking'] +
            progress_reward * human_weights['progress'] +
            potential_field_reward * human_weights['potential_field'] +
            force_penalties['human'] * human_weights['force_efficiency'] +
            force_penalties['robot'] * human_weights['robot_awareness'] +
            z_penalty * human_weights.get('z_penalty', 0.0) +
            completion_reward * human_weights.get('completion_reward', 0.0) +
            time_efficiency_reward * human_weights.get('time_efficiency', 0.0)
        )
        
        # Cache reward components for logging
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), dim=-1
        )
        
        self.reward_components = {
            'trajectory_reward': trajectory_reward,
            'progress_reward': progress_reward,
            'potential_field_reward': potential_field_reward,
            'robot_force_penalty': force_penalties['robot'],
            'human_force_penalty': force_penalties['human'],
            'z_penalty': z_penalty,
            'completion_reward': completion_reward,
            'time_efficiency_reward': time_efficiency_reward,
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_final': distance_to_final
        }

        # Console logging via utils (controlled by YAML)
        self.reward_logger.log_console_if_enabled(self, rewards, robot_weights, human_weights)
        
        # Vectorized batch update for all environments  
        self.reward_logger.update_step_metrics_batch(self.reward_components, self.safety_distances_t1, rewards)
        
        return rewards

    def _calculate_adaptive_trajectory_reward(self) -> torch.Tensor:
        """
        Calculate adaptive trajectory reward based on distance to obstacle.
        - Outside 1.5cm: Strict deviation requirements
        - Within 1.5cm: Relaxed deviation requirements (4cm tolerance)
        """
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        safety_distances = self.safety_distances_t1
        
        # Define phase boundary
        OBSTACLE_BOUNDARY = 0.015  # 1.5cm
        
        # Phase A: Outside obstacle boundary (>1.5cm) - Strict trajectory following
        outside_mask = safety_distances > OBSTACLE_BOUNDARY
        trajectory_reward = torch.zeros_like(deviations)
        
        if torch.any(outside_mask):
            outside_deviations = deviations[outside_mask]
            trajectory_reward[outside_mask] = torch.where(
                outside_deviations < 0.01,  # <1cm: full reward
                torch.full_like(outside_deviations, 5.0),
                torch.where(
                    outside_deviations < 0.025,  # 1-2.5cm: linear decrease
                    5.0 * (1.0 - (outside_deviations - 0.01) / 0.015),
                    -10.0 * (outside_deviations - 0.025)  # >2.5cm: penalty
                )
            )
        
        # Phase B: Inside obstacle boundary (≤1.5cm) - Relaxed trajectory following
        inside_mask = safety_distances <= OBSTACLE_BOUNDARY
        trajectory_reward[inside_mask] = 2.0  # Fixed small reward, not dependent on deviation
        
        return trajectory_reward

    def _calculate_progress_reward(self) -> torch.Tensor:
        """Calculate progress reward - shared across all phases."""
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        return progress_ratio - 1.0 # Scale progress reward

    def _calculate_potential_field_reward(self) -> torch.Tensor:
        """
        Calculate potential field reward using bounded squared hinge function.
        Combines repulsion from obstacle and attraction to goal.
        Only active when within obstacle boundary (≤1.5cm).
        """
        safety_distances = self.safety_distances_t1
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        
        # Define parameters for bounded potential field
        d_safe = 0.01       # 1 cm safe distance
        lam = 800.0         # repulsion strength
        beta = 20.0         # attraction strength
        OBSTACLE_BOUNDARY = 0.015  # 1.5cm boundary
        
        # Initialize reward tensor
        reward = torch.zeros_like(safety_distances)
        
        # Only apply potential field within obstacle region
        inside_mask = safety_distances <= OBSTACLE_BOUNDARY
        
        if torch.any(inside_mask):
            d_in = safety_distances[inside_mask]
            prog_in = progress_ratio[inside_mask]
            
            # Squared hinge function: -λ * max(0, d_safe - d)²
            # This is bounded and has smooth gradients
            repulsion = -lam * torch.relu(d_safe - d_in).pow(2)
            
            # Attraction force: encourage progress toward goal
            attraction = (prog_in - 1.0) * beta
            
            # Combined potential field
            reward[inside_mask] = repulsion + attraction
        
        return reward

    def _calculate_force_penalties(self) -> Dict[str, torch.Tensor]:
        """Calculate force efficiency penalties - shared across all phases."""
        return {
            'robot': -50.0 * torch.norm(self.robot_forces_t, dim=-1),
            'human': -50.0 * torch.norm(self.human_forces_t, dim=-1)
        }

    def _calculate_z_penalty(self) -> torch.Tensor:
        """Calculate Z-axis constraint penalty - shared across all phases."""
        return torch.where(
            self.stylus_pos_t1[:, 2] < 0.0,
            -500.0 * torch.abs(self.stylus_pos_t1[:, 2]),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )

    def _calculate_completion_reward(self) -> torch.Tensor:
        """Calculate task completion reward - shared across all phases."""
        is_final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        return torch.where(
            is_final_reached,
            torch.full_like(self.safety_distances_t1, 100.0),
            torch.zeros_like(self.safety_distances_t1)
        )

    def _calculate_time_efficiency_reward(self) -> torch.Tensor:
        """
        Calculate time efficiency reward using progress manager's step tracking.
        Encourages faster completion.
        """
        max_steps = 1200  # 20s * 60fps = 1200 steps per episode
        
        # Get step count from progress manager (float tensor)
        current_steps = self.progress_manager.step_counts.to(self.device).float()
        
        # Time efficiency: higher reward for completing tasks faster
        time_efficiency = (max_steps - current_steps) / max_steps
        time_efficiency = torch.clamp(time_efficiency, min=0.0, max=1.0)

        return time_efficiency * 3.0  # Scale time efficiency reward

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Determine termination and truncation conditions."""
        
        # Z-axis termination (safety)
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # Edge collision termination (safety) - MODIFIED for direct termination
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t1 <= self.safety_distance_threshold
        
        # Task completion
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # Combine termination conditions
        terminated_condition = z_below_zero | edge_collision | final_reached

        # Time truncation - Isaac Lab manages independent episode_length_buf per environment
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """MODIFIED: Reset specified environments without extras['episode'] handling."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Unified filtering: only episodes that actually ran can be settled
        valid_env_ids = self.progress_manager.filter_valid_for_episode_end(env_ids, min_steps=1)

        if valid_env_ids:
            # Fixed order: PM settlement first, then RewardLogger aggregation
            self.progress_manager.on_episode_end(valid_env_ids, reason="env_reset")
            self.reward_logger.on_episode_end(valid_env_ids)

        # Parent class reset + other initialization
        super()._reset_idx(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # Use stable initial configuration
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96    # waist
        joint_pos[:, 1] = 0.0      # shoulder
        joint_pos[:, 2] = 1.0      # elbow
        joint_pos[:, 3] = 0.0      # yaw
        joint_pos[:, 4] = 2.0944   # pitch (~120 degrees)
        joint_pos[:, 5] = 0.0      # roll
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset state variables
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t1[env_ids] = 0.01
        self.is_violating_t1[env_ids] = False
        self.normal_t1[env_ids] = torch.zeros((num_resets, 3), device=self.device)

    def _get_stylus_position(self) -> torch.Tensor:
        """Get stylus position relative to robot base."""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        base_pos = self._omni_robot.data.root_link_pos_w
        ee_pos = self._omni_robot.data.body_link_pos_w[:, self.stylus_body_idx, :]
        return ee_pos - base_pos
    
    def _get_stylus_velocity(self) -> torch.Tensor:
        """Get stylus velocity in world frame."""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        return self._omni_robot.data.body_link_lin_vel_w[:, self.stylus_body_idx, :]
    
    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close_all_files()
    
    def get_constraint_state(self, env_ids: Optional[List[int]] = None) -> Dict:
        """Get constraint state information."""
        if self.constraint_results_t1 is None:
            return {}
        
        if env_ids is None:
            return self.constraint_results_t1
        
        return {
            'distances_constraint': self.constraint_results_t1['distances_constraint'][env_ids],
            'closest_points': self.constraint_results_t1['closest_points'][env_ids],
            'normal_vectors': self.constraint_results_t1['normal_vectors'][env_ids],
            'is_overlapping': self.constraint_results_t1['is_overlapping'][env_ids],
            'is_inside': self.constraint_results_t1['is_inside'][env_ids]
        }
    
    def get_reward_details(self, env_ids: Optional[List[int]] = None) -> Dict:
        """Get detailed reward component information."""
        if not self.reward_components:
            return {}
        
        if env_ids is None:
            return self.reward_components
        
        return {
            key: value[env_ids] if torch.is_tensor(value) else value
            for key, value in self.reward_components.items()
        }