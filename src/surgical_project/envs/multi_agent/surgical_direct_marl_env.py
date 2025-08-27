# surgical_direct_marl_env.py - Complete version with console logging

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
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Trajectory following with progress tracking
    - Comprehensive reward system with safety considerations
    - Performance evaluation and milestone tracking
    - Detailed console logging for verification
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
        
        # Load training parameters and environment configuration
        self.params = self._load_training_params()
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._load_configuration_parameters()
        
        # Initialize environment base positions
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize utility managers
        self._initialize_managers()
        
        # Initialize agent actions storage
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Physics interaction state (forces applied at time t)
        self._initialize_physics_state()
        
        # Fixed end joints configuration
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
        
        # Body index for stylus/end-effector
        self.stylus_body_idx = None
        
        # Constraint checker for collision detection
        self.constraint_checker = CompleteConstraintChecker(self.device, self.collision_threshold)
        
        # State cache (will be updated in _get_observations)
        self._initialize_state_cache()
        
        # Reward components cache
        self.reward_components = {}
        
        # Console logging configuration
        self.enable_console_logging = kwargs.get('enable_console_logging', True)
        
        # Gymnasium compatibility
        self._setup_gymnasium_spaces()
        
    def _load_training_params(self) -> dict:
        """Load training parameters from YAML configuration."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_configuration_parameters(self):
        """Load and cache configuration parameters for efficient access."""
        constraints = self.params['constraints']
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        # Joint limits as tensors for efficient clamping
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        # CBF and safety parameters
        self.safety_margin = self.params['reward_parameters']['cbf_parameters']['safety_margin']
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
        self.cbf_gamma = self.params['reward_parameters']['cbf_parameters']['gamma']
        self.cbf_epsilon = self.params['reward_parameters']['cbf_parameters']['epsilon']
        
        # Termination conditions
        term_config = self.params.get('termination_conditions', {})
        self.enable_z_termination = term_config.get('z_below_zero', True)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.001)
        
        print(f"[INFO] Episode length controlled by cfg.episode_length_s: {self.cfg.episode_length_s}s")
        
    def _initialize_managers(self) -> None:
        """Initialize utility managers for trajectory and reward tracking."""
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        self.reward_logger = RewardLogger(self.num_envs, self.device)
        
    def _initialize_state_cache(self) -> None:
        """Initialize state caching variables (updated in _get_observations)."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.joint_pos_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        self.joint_vel_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.constraint_results_t1 = None
        
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables (forces at time t)."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        
    def _setup_gymnasium_spaces(self) -> None:
        """Setup Gymnasium compatibility spaces."""
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.cfg.action_spaces[agent],), 
                dtype=np.float32
            ) for agent in self.cfg.possible_agents
        })
        
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-10.0, high=10.0, 
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
        
        # Setup scene lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """Post-scene creation setup including physics configuration."""
        super()._setup_post_scene_creation()
        self._initialize_body_indices()
        
        if hasattr(self, '_omni_robot'):
            # Update environment base positions
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
            # Configure robot for force control (zero stiffness and damping)
            num_joints = self._omni_robot.num_joints
            zero_stiffness = torch.zeros(self.num_envs, num_joints, device=self.device)
            zero_damping = torch.zeros(self.num_envs, num_joints, device=self.device)
            
            self._omni_robot.write_joint_stiffness_to_sim(zero_stiffness)
            self._omni_robot.write_joint_damping_to_sim(zero_damping)
            
    def _initialize_body_indices(self):
        """Initialize body indices for end-effector/stylus identification."""
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        # Search for end-effector using common naming patterns
        search_patterns = ['stylus', 'tip', 'end_effector', 'link6', 'end', 'tool']
        for pattern in search_patterns:
            for i, name in enumerate(self._omni_robot.body_names):
                if pattern in name.lower():
                    self.stylus_body_idx = i
                    print(f"[INFO] Found stylus body: {name} (index {i})")
                    return
        
        # Fallback to last body if no pattern match
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
            print(f"[INFO] Using fallback stylus body index: {self.stylus_body_idx}")
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """
        Pre-physics step processing including action validation and force application.
        
        Args:
            actions: Dictionary of actions for each agent
        """
        # Process and validate actions
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                # Ensure proper action dimensions
                if action.dim() == 1:
                    if action.shape[0] == 3:
                        action = action.unsqueeze(0).expand(self.num_envs, -1)
                    else:
                        action = action.unsqueeze(-1).expand(-1, 3)
                
                # Apply force constraints
                max_force = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = torch.clamp(action, -max_force, max_force)
        
        # Cache forces for reward calculation (these are at time t)
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # Apply external forces to end-effector
        if self.stylus_body_idx is not None:
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
        """
        Compute observations for all agents and update state cache.
        
        Returns:
            Dictionary of observations for each agent
        """
        # Update all state cache (t+1 state after physics)
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # Update joint state cache
        joint_pos = self._omni_robot.data.joint_pos
        joint_vel = self._omni_robot.data.joint_vel
        
        if joint_pos.shape[-1] < 6:
            # Pad if necessary
            padding = torch.zeros(self.num_envs, 6 - joint_pos.shape[-1], device=self.device)
            self.joint_pos_t1 = torch.cat([joint_pos, padding], dim=-1)
            self.joint_vel_t1 = torch.cat([joint_vel, padding], dim=-1)
        else:
            self.joint_pos_t1 = joint_pos[..., :6]
            self.joint_vel_t1 = joint_vel[..., :6]
        
        # Update constraint state (t+1 state after physics)
        current_base_positions = self._omni_robot.data.root_link_pos_w
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']

        # Constrain velocity for stability
        stylus_vel_constrained = self.stylus_vel_t1
        
        # Ensure constraint distances have correct shape for concatenation
        constraint_distances = self.safety_distances_t1.unsqueeze(-1)  # [num_envs] -> [num_envs, 1]

        # Concatenate all observation components
        obs = torch.cat([
            self.stylus_pos_t1,      # End-effector position (3)
            stylus_vel_constrained,   # End-effector velocity (3)
            self.joint_pos_t1,       # Joint positions (6)
            self.joint_vel_t1,       # Joint velocities (6)
            constraint_distances,     # Distance measurements (1)
        ], dim=-1)                   # Total: 19 dimensions
        
        # Create observation dictionary for each agent
        observations = {}
        for agent in self.cfg.possible_agents:
            # Apply observation bounds for stability
            observations[agent] = torch.clamp(obs, -20.0, 20.0)
            
        return observations
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """
        Compute rewards using modular reward components.
        
        Returns:
            Dictionary of rewards for each agent
        """
        self.reward_logger.on_step()
        
        # Calculate individual reward components
        trajectory_reward = self._calculate_trajectory_reward()
        progress_reward = self._calculate_progress_reward()
        velocity_reward = self._calculate_velocity_reward()
        cbf_reward = self._calculate_cbf_reward()
        force_penalties = self._calculate_force_penalties()
        z_penalty = self._calculate_z_penalty()
        completion_reward = self._calculate_completion_reward()
        
        # Get reward weights from configuration
        robot_weights = self.params['reward_parameters']['robot_weights']
        human_weights = self.params['reward_parameters']['human_weights']
        
        # Compute final rewards
        rewards = {}
        rewards["robot"] = (
            trajectory_reward * robot_weights['trajectory_tracking'] +
            progress_reward * robot_weights['progress'] +
            velocity_reward * robot_weights['velocity'] +
            cbf_reward * robot_weights['obstacle_cbf'] +
            force_penalties['robot'] * robot_weights['force_efficiency'] +
            force_penalties['human'] * robot_weights['human_awareness'] +
            z_penalty +
            completion_reward
        )
        
        rewards["human"] = (
            trajectory_reward * human_weights['trajectory_tracking'] +
            progress_reward * human_weights['progress'] +
            velocity_reward * human_weights['velocity'] +
            cbf_reward * human_weights['obstacle_cbf'] +
            force_penalties['human'] * human_weights['force_efficiency'] +
            force_penalties['robot'] * human_weights['robot_awareness'] +
            z_penalty +
            completion_reward
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
            'velocity_reward': velocity_reward,
            'cbf_reward': cbf_reward,
            'robot_force_penalty': force_penalties['robot'],
            'human_force_penalty': force_penalties['human'],
            'z_penalty': z_penalty,
            'completion_reward': completion_reward,
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_final': distance_to_final
        }
        
        # Add console logging if enabled
        if self.enable_console_logging:
            self._log_console_step_info(rewards, robot_weights, human_weights)
        
        # Update environment-specific metrics
        for env_id in range(self.num_envs):
            self.reward_logger.update_step_metrics(
                env_id, self.reward_components, self.safety_distances_t1, rewards
            )
        
        # Update extras for logging
        self.extras["log"] = {
            "robot_reward": rewards["robot"].mean().item(),
            "human_reward": rewards["human"].mean().item(),
            "deviation": deviations.mean().item(),
            "progress": progress_ratio.mean().item(),
            "safety_distance": self.safety_distances_t1.mean().item(),
            "z_penalty": z_penalty.mean().item(),
        }
        
        return rewards
    
    def _calculate_trajectory_reward(self) -> torch.Tensor:
        """Calculate trajectory tracking reward."""
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        
        return torch.where(
            deviations < 0.01,
            torch.ones_like(deviations),
            torch.where(
                deviations < 0.025,
                1.0 - (deviations - 0.01) / 0.015,
                -10.0 * (deviations - 0.025)
            )
        )
    
    def _calculate_progress_reward(self) -> torch.Tensor:
        """Calculate progress reward."""
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        return progress_ratio * 5.0
    
    def _calculate_velocity_reward(self) -> torch.Tensor:
        """Calculate velocity control reward."""
        velocity_along_line = torch.abs(
            torch.sum(self.stylus_vel_t1 * self.trajectory_manager.line_direction.unsqueeze(0), dim=-1)
        )
        return torch.exp(-velocity_along_line * 20.0)
    
    def _calculate_cbf_reward(self) -> torch.Tensor:
        """Calculate Control Barrier Function reward for obstacle avoidance."""
        return torch.where(
            self.safety_distances_t1 < 0.001,
            torch.full_like(self.safety_distances_t1, -500.0),
            torch.where(
                self.safety_distances_t1 < 0.008,
                torch.full_like(self.safety_distances_t1, -200.0),
                0.5 + 10.0 * torch.clamp(self.safety_distances_t1, max=0.05)
            )
        )
    
    def _calculate_force_penalties(self) -> Dict[str, torch.Tensor]:
        """Calculate force efficiency penalties."""
        return {
            'robot': -50.0 * torch.sum(self.robot_forces_t**2, dim=-1),
            'human': -50.0 * torch.sum(self.human_forces_t**2, dim=-1)
        }
    
    def _calculate_z_penalty(self) -> torch.Tensor:
        """Calculate Z-axis constraint penalty."""
        return torch.where(
            self.stylus_pos_t1[:, 2] < 0.0,
            -500.0 * torch.abs(self.stylus_pos_t1[:, 2]),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )
    
    def _calculate_completion_reward(self) -> torch.Tensor:
        """Calculate task completion reward."""
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), 
            dim=-1
        )
        return torch.where(
            distance_to_final < 0.01,
            torch.full_like(distance_to_final, 50.0),
            torch.zeros_like(distance_to_final)
        )
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Determine termination and truncation conditions.
        
        Returns:
            Tuple of (terminated, truncated) dictionaries
        """
        
        # Z-axis termination (safety)
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # Edge collision termination (safety)
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t1 < self.safety_distance_threshold
        
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
        """
        Reset specified environments.
        
        Args:
            env_ids: Environment IDs to reset
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Only log episode end for environments that actually ran
        if torch.is_tensor(env_ids):
            env_ids_list = env_ids.cpu().numpy().tolist()
        else:
            env_ids_list = [env_ids] if isinstance(env_ids, int) else env_ids
        
        # Filter out environments that actually ran episodes
        valid_env_ids = []
        for env_id in env_ids_list:
            if self.reward_logger.current_episode_basic[env_id]['steps'] > 0:
                valid_env_ids.append(env_id)
        
        # Only log episodes that actually ran
        if valid_env_ids:
            self.reward_logger.on_episode_end(torch.tensor(valid_env_ids, device=self.device))
        
        # Call parent reset
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
    
    def _log_console_step_info(self, rewards: Dict[str, torch.Tensor], robot_weights: dict, human_weights: dict):
        """
        Log detailed step information to console for verification.
        """
        current_step = self.common_step_counter
        
        # Only log every few steps to avoid overwhelming output
        if current_step % 10 != 0:  # Log every 10 steps
            return
        
        print(f"\n{'='*80}")
        print(f"STEP {current_step} - Detailed Environment State")
        print(f"{'='*80}")
        
        # Only log first 2 environments to avoid spam
        for env_id in range(min(2, self.num_envs)):
            print(f"\n--- Environment {env_id} ---")
            
            # Stylus position (relative to base)
            stylus_pos = self.stylus_pos_t1[env_id]
            print(f"Stylus Position (local): [{stylus_pos[0]:.4f}, {stylus_pos[1]:.4f}, {stylus_pos[2]:.4f}]")
            
            # Trajectory information
            deviation = self.reward_components['deviation'][env_id].item()
            progress = self.reward_components['progress_ratio'][env_id].item()
            distance_to_final = self.reward_components['distance_to_final'][env_id].item()
            print(f"Trajectory - Deviation: {deviation:.4f}m, Progress: {progress:.1%}, Distance to Final: {distance_to_final:.4f}m")
            
            # Constraint information
            safety_distance = self.safety_distances_t1[env_id].item()
            is_overlapping = self.is_violating_t1[env_id].item()
            print(f"Constraint - Safety Distance: {safety_distance:.4f}m, Overlapping: {is_overlapping}")
            
            # Agent forces
            robot_force = self.robot_forces_t[env_id]
            human_force = self.human_forces_t[env_id]
            robot_force_mag = torch.norm(robot_force).item()
            human_force_mag = torch.norm(human_force).item()
            print(f"Forces - Robot: {robot_force_mag:.3f}N, Human: {human_force_mag:.3f}N")
            
            # Reward breakdown with weights
            print(f"\nReward Breakdown:")
            print(f"Robot Agent:")
            traj_r = self.reward_components['trajectory_reward'][env_id].item()
            prog_r = self.reward_components['progress_reward'][env_id].item()
            vel_r = self.reward_components['velocity_reward'][env_id].item()
            cbf_r = self.reward_components['cbf_reward'][env_id].item()
            robot_force_pen = self.reward_components['robot_force_penalty'][env_id].item()
            human_force_pen = self.reward_components['human_force_penalty'][env_id].item()
            z_pen = self.reward_components['z_penalty'][env_id].item()
            comp_r = self.reward_components['completion_reward'][env_id].item()
            
            print(f"  Trajectory: {traj_r:.3f} * {robot_weights['trajectory_tracking']:.2f} = {traj_r * robot_weights['trajectory_tracking']:.3f}")
            print(f"  Progress: {prog_r:.3f} * {robot_weights['progress']:.2f} = {prog_r * robot_weights['progress']:.3f}")
            print(f"  Velocity: {vel_r:.3f} * {robot_weights['velocity']:.2f} = {vel_r * robot_weights['velocity']:.3f}")
            print(f"  CBF: {cbf_r:.3f} * {robot_weights['obstacle_cbf']:.2f} = {cbf_r * robot_weights['obstacle_cbf']:.3f}")
            print(f"  Robot Force: {robot_force_pen:.3f} * {robot_weights['force_efficiency']:.2f} = {robot_force_pen * robot_weights['force_efficiency']:.3f}")
            print(f"  Human Awareness: {human_force_pen:.3f} * {robot_weights['human_awareness']:.2f} = {human_force_pen * robot_weights['human_awareness']:.3f}")
            print(f"  Z Penalty: {z_pen:.3f}")
            print(f"  Completion: {comp_r:.3f}")
            robot_total = rewards["robot"][env_id].item()
            print(f"  ROBOT TOTAL: {robot_total:.3f}")
            
            print(f"Human Agent:")
            print(f"  Trajectory: {traj_r:.3f} * {human_weights['trajectory_tracking']:.2f} = {traj_r * human_weights['trajectory_tracking']:.3f}")
            print(f"  Progress: {prog_r:.3f} * {human_weights['progress']:.2f} = {prog_r * human_weights['progress']:.3f}")
            print(f"  Velocity: {vel_r:.3f} * {human_weights['velocity']:.2f} = {vel_r * human_weights['velocity']:.3f}")
            print(f"  CBF: {cbf_r:.3f} * {human_weights['obstacle_cbf']:.2f} = {cbf_r * human_weights['obstacle_cbf']:.3f}")
            print(f"  Human Force: {human_force_pen:.3f} * {human_weights['force_efficiency']:.2f} = {human_force_pen * human_weights['force_efficiency']:.3f}")
            print(f"  Robot Awareness: {robot_force_pen:.3f} * {human_weights['robot_awareness']:.2f} = {robot_force_pen * human_weights['robot_awareness']:.3f}")
            print(f"  Z Penalty: {z_pen:.3f}")
            print(f"  Completion: {comp_r:.3f}")
            human_total = rewards["human"][env_id].item()
            print(f"  HUMAN TOTAL: {human_total:.3f}")
            
            print(f"Combined Total Reward: {robot_total + human_total:.3f}")

    def _log_constraint_state(self):
        """Log constraint state information to console."""
        current_step = self.common_step_counter
        
        if current_step % 20 != 0:
            return
        
        print(f"\n--- CONSTRAINT STATE (Step {current_step}) ---")
        for env_id in range(min(2, self.num_envs)):
            if self.constraint_results_t1:
                distance = self.safety_distances_t1[env_id].item()
                overlapping = self.is_violating_t1[env_id].item()
                closest_point = self.constraint_results_t1['closest_points'][env_id]
                normal_vector = self.constraint_results_t1['normal_vectors'][env_id]
                
                print(f"Env {env_id}: Distance={distance:.4f}m, Overlapping={overlapping}")
                print(f"  Closest Point: [{closest_point[0]:.4f}, {closest_point[1]:.4f}, {closest_point[2]:.4f}]")
                print(f"  Normal Vector: [{normal_vector[0]:.4f}, {normal_vector[1]:.4f}, {normal_vector[2]:.4f}]")
    
    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close_all_files()
    
    # Public interface methods
    def get_trajectory_info(self) -> Dict:
        """Get trajectory configuration information."""
        return self.trajectory_manager.get_trajectory_info()
    
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
        """
        Get detailed reward component information.
        
        Args:
            env_ids: Optional list of environment IDs to query
            
        Returns:
            Dictionary containing reward component details
        """
        if not self.reward_components:
            return {}
        
        if env_ids is None:
            return self.reward_components
        
        return {
            key: value[env_ids] if torch.is_tensor(value) else value
            for key, value in self.reward_components.items()
        }
    
    def enable_logging(self, enable: bool = True):
        """Enable or disable console logging."""
        self.enable_console_logging = enable
        print(f"[INFO] Console logging {'enabled' if enable else 'disabled'}")
        
    def set_logging_frequency(self, step_interval: int = 10):
        """Set console logging frequency."""
        # This would require modifying the _log_console_step_info method
        # For now, it's hardcoded to every 10 steps
        print(f"[INFO] Logging frequency setting not implemented yet. Currently logs every 10 steps.")