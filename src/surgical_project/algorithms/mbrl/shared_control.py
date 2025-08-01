# Paper-aligned surgical robot human-robot shared control algorithm - fixed version with wandb integration
import torch
import yaml
import os
import wandb
from typing import Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .actor_critic import SurgicalActorCritic
from ..utils import (
    ReplayBuffer, PaperCostFunction, HumanImpedanceModel, 
    OffPolicyTrainer, ControlBarrierFunction,
    extract_paper_state, create_augmented_state, extract_actor_input, compute_robot_control
)


class SharedControlTrainer:
    """Paper-aligned shared control trainer with wandb integration and fixed environment access"""
    def __init__(self, env, agent_cfg: Dict[str, Any], log_dir: str = None):
        self.env = env
        self.device = torch.device(agent_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_envs = getattr(env, 'num_envs', 1)
        self.log_dir = log_dir or "logs"
        
        # Use agent_cfg directly without loading additional config files
        self.config = agent_cfg
        
        # Initialize wandb if enabled
        self.use_wandb = agent_cfg.get('wandb_logging', False)
        if self.use_wandb:
            wandb.init(
                project="surgical-shared-control",
                config=agent_cfg,
                name=f"surgical_training_{agent_cfg.get('seed', 42)}",
                tags=["surgical", "shared-control", "rbf", "cbf"]
            )
            print("[INFO] Wandb logging initialized")
        
        # Extract network architecture parameters from configuration
        network_cfg = self.config.get('rbf_network', {})
        
        # Paper standard dimensions
        state_dim = self.config.get('state_space_dim', 9)
        action_dim = self.config.get('action_space_dim', 3)
        augmented_state_dim = self.config.get('augmented_state_dim', 12)
        actor_input_dim = self.config.get('actor_input_dim', 18)  # [q, q̇, ẋr, ẍr]
        
        # Initialize networks with RBF configuration
        self.policy = SurgicalActorCritic(
            state_dim=state_dim,
            action_dim=action_dim, 
            augmented_state_dim=augmented_state_dim,
            actor_input_dim=actor_input_dim,
            network_cfg=self.config  # Pass complete config including RBF settings
        ).to(self.device)
        
        # Initialize components - get all parameters from configuration
        buffer_size = self.config.get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        
        # Cost function - use configuration parameters
        self.cost_function = PaperCostFunction(
            Q1_weight=self.config.get('Q1_weight', 100.0),
            Q2_weight=self.config.get('Q2_weight', 0.01),
            Q3_weight=self.config.get('Q3_weight', 0.001),
            R_weight=self.config.get('R_weight', 0.001),
            cbf_weight=self.config.get('cbf_weight', 10.0),
            device=self.device
        )
        
        # Human impedance model - use configuration parameters
        self.human_dynamics = HumanImpedanceModel(
            device=self.device,
            damping_diag=self.config.get('human_damping_CH', [21.0, 21.0, 21.0]),
            stiffness_diag=self.config.get('human_stiffness_KH', [201.0, 201.0, 201.0]),
            damping_variation=self.config.get('human_damping_variation', [20.0, 20.0, 20.0]),
            stiffness_variation=self.config.get('human_stiffness_variation', [200.0, 200.0, 200.0])
        )
        
        # Off-Policy trainer with paper update laws
        self.trainer = OffPolicyTrainer(self.policy, self.config, self.device)
        
        # # CBF constraint manager - use configuration parameters
        # self.cbf = ControlBarrierFunction(
        #     gamma=self.config.get('cbf_gamma', 1.0),
        #     safety_margin=self.config.get('safety_margin', 0.002),
        #     device=self.device
        # )
        
        # Initialize interaction force (will be computed from paper equations)
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Human equilibrium points - from configuration
        equilibrium_points_cfg = self.config.get('equilibrium_points', [
            [-0.2, 0.15, 0.03],  # Default first equilibrium point (start)
            [0.2, 0.15, 0.03]    # Default second equilibrium point (end)
        ])
        self.equilibrium_points = torch.tensor(
            equilibrium_points_cfg, device=self.device, dtype=torch.float32
        )
        
        # Control parameters from configuration
        self.K1_gain = self.config.get('K1_gain', 1.0)
        self.K2_gain = self.config.get('K2_gain', 600.0)
        
        # Training parameters
        self.save_frequency = self.config.get('save_frequency', 100)
        self.eval_frequency = self.config.get('eval_frequency', 50)
        self.log_frequency = self.config.get('log_frequency', 50)
        
        print(f"[INFO] SharedControlTrainer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  - RBF networks: Critic({network_cfg.get('critic', {}).get('nodes', 10)} nodes), "
              f"Actor({network_cfg.get('actor', {}).get('nodes', 10)} nodes), "
              f"Identifier({network_cfg.get('identifier', {}).get('nodes', 10)} nodes)")
        print(f"  - CBF gamma: {self.config.get('cbf_gamma', 1.0)}")
        print(f"  - Human equilibrium points: {self.equilibrium_points.cpu().numpy().tolist()}")
        print(f"  - Control gains: K1={self.K1_gain}, K2={self.K2_gain}")
        
    def get_current_equilibrium_point(self, target_index: int) -> torch.Tensor:
        """Get corresponding human equilibrium point based on current target index"""
        if target_index >= len(self.equilibrium_points):
            target_index = len(self.equilibrium_points) - 1
        return self.equilibrium_points[target_index]
    
    def get_unwrapped_env(self):
        """Get unwrapped environment to access joint data"""
        # Handle various wrapper types
        env = self.env
        while hasattr(env, 'env') or hasattr(env, 'unwrapped'):
            if hasattr(env, 'unwrapped'):
                return env.unwrapped
            elif hasattr(env, 'env'):
                env = env.env
            else:
                break
        return env
    
    def train_off_policy(self, total_episodes: int):
        """
        Train using off-policy RL method similar to the provided template
        """
        min_buffer_size = self.config.get('min_buffer_size', 1000)
        batch_size = self.config.get('batch_size', 128)
        episodes_per_iter = max(1, total_episodes // 10)
        
        return_list = []
        
        print(f"[INFO] Starting off-policy training for {total_episodes} episodes")
        print(f"[INFO] Using paper control law: u = Ŵᵀₐ Sa(Za) - e - K2*ev")
        
        for iter_idx in range(10):
            with tqdm(total=episodes_per_iter, desc=f'Iteration {iter_idx}') as pbar:
                for episode_idx in range(episodes_per_iter):
                    episode_return = 0
                    
                    # Reset environment
                    obs_dict, _ = self.env.reset()
                    obs = obs_dict["policy"]
                    done = False
                    step_count = 0
                    max_steps = self.config.get('max_eval_steps', 500)
                    
                    while not done and step_count < max_steps:
                        # Get action using current policy
                        action = self._take_action(obs)
                        
                        # Environment step
                        next_obs_dict, reward, terminated, truncated, info = self.env.step(action)
                        next_obs = next_obs_dict["policy"]
                        done = (terminated | truncated).any()

                        # 优化 Update interaction forces
                        
                        # Add to replay buffer
                        self._add_to_buffer(obs, action, reward, next_obs, done)

                        obs = next_obs
                        episode_return += reward.mean().item()
                        step_count += 1
                        
                        # Update networks if buffer has enough samples
                        if len(self.replay_buffer) > min_buffer_size:
                            self._update_networks()
                    
                    return_list.append(episode_return)
                    
                    # Log progress
                    if (episode_idx + 1) % 10 == 0:
                        avg_return = np.mean(return_list[-10:])
                        pbar.set_postfix({
                            'episode': f'{episodes_per_iter * iter_idx + episode_idx + 1}',
                            'return': f'{avg_return:.3f}'
                        })
                        
                        # Wandb logging
                        if self.use_wandb:
                            wandb.log({
                                'episode': episodes_per_iter * iter_idx + episode_idx + 1,
                                'episode_return': episode_return,
                                'avg_return_10': avg_return,
                                'buffer_size': len(self.replay_buffer)
                            })
                    
                    pbar.update(1)
        
        print(f"[INFO] Training completed. Final average return: {np.mean(return_list[-10:]):.3f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return return_list
    
    def _take_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Take action using current policy following paper framework"""
        try:
            # Extract paper state z = [x, ẋ, f]ᵀ for critic and identifier
            paper_state, desired_pos = extract_paper_state(obs, self.interaction_forces)
            
            # Get joint states for actor input Za = [q, q̇, ẋr, ẍr]
            unwrapped_env = self.get_unwrapped_env()
            joint_pos = unwrapped_env.get_joint_positions()
            joint_vel = unwrapped_env.get_joint_velocities()
            
            # Compute desired velocity and acceleration for reference trajectory
            current_pos = paper_state[..., :3]
            current_vel = paper_state[..., 3:6]
            
            # Simple desired trajectory (straight line motion)
            desired_vel = torch.zeros_like(current_pos)
            desired_vel[..., 0] = 0.08  # ！！！！！！，改为期望速度和离目标点距离以及离障碍物距离相关
            
            # Extract actor input Za = [q, q̇, ẋr, ẍr]
            actor_input = extract_actor_input(
                joint_pos, joint_vel, desired_pos, desired_vel, 
                current_pos, self.K1_gain, self.config.get('dt', 0.01)
            )
            
            # Get robot action from actor network
            with torch.no_grad():
                actor_output = self.policy.get_action(
                    actor_input, 
                    deterministic=False,
                    exploration_noise=self.config.get('exploration_noise', 0.01)
                )
            
            # Compute robot control according to paper: u = Ŵᵀₐ Sa(Za) - e - K2*ev #优化！！！！-f
            tracking_error = current_pos - desired_pos
            sliding_error = current_vel - desired_vel + self.K1_gain * tracking_error
            
            robot_control = compute_robot_control(
                actor_output, tracking_error, sliding_error, self.K2_gain
            )
            robot_control = torch.clamp(robot_control, -1.0, 1.0) #优化！！！！，不要限制在+-1.0

            # Compute human force for this iteration
            # Get current target index for human equilibrium point
            try:
                tm = getattr(unwrapped_env, 'trajectory_manager', None)
                current_target_index = tm.current_target_index if tm else 0
            except:
                current_target_index = 0
                
            current_equilibrium = self.get_current_equilibrium_point(current_target_index) #优化！！！！，不是一个target index
            
            # Compute human force using impedance model
            human_force = self.human_dynamics.compute_human_force(
                current_pos, 
                current_equilibrium.unsqueeze(0).expand(self.num_envs, -1),
                current_vel
            )
            
            # Update interaction forces for this iteration
            self.interaction_forces = human_force.clone()
            
            return robot_control
            
        except Exception as e:
            print(f"[WARNING] Action computation failed: {e}, using zero action")
            return torch.zeros(self.num_envs, 3, device=self.device)
    
    def _add_to_buffer(self, obs: torch.Tensor, action: torch.Tensor, 
                       reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor):
        """Add experience to replay buffer with both state representations"""
        try:
            # Extract augmented states for critic/identifier
            paper_state, desired_pos = extract_paper_state(obs, self.interaction_forces)
            augmented_state = create_augmented_state(paper_state, desired_pos)
            
            next_paper_state, next_desired_pos = extract_paper_state(next_obs, self.interaction_forces)
            next_augmented_state = create_augmented_state(next_paper_state, next_desired_pos)
            
            # Extract actor inputs Za = [q, q̇, ẋr, ẍr]
            unwrapped_env = self.get_unwrapped_env()
            joint_pos = unwrapped_env.get_joint_positions()
            joint_vel = unwrapped_env.get_joint_velocities()
            
            current_pos = paper_state[..., :3]
            current_vel = paper_state[..., 3:6]
            desired_vel = torch.zeros_like(current_pos)
            desired_vel[..., 0] = 0.08  # 优化！！！！， xd.不应该是一个常数
            
            actor_input = extract_actor_input(
                joint_pos, joint_vel, desired_pos, desired_vel, 
                current_pos, self.K1_gain, self.config.get('dt', 0.01)
            )
            
            # For next actor input, approximate using current method
            next_current_pos = next_paper_state[..., :3]
            next_actor_input = extract_actor_input(
                joint_pos, joint_vel, next_desired_pos, desired_vel,
                next_current_pos, self.K1_gain, self.config.get('dt', 0.01)
            )
            
            # Add to buffer for each environment with both state representations
            for i in range(self.num_envs):
                self.replay_buffer.add(
                    augmented_state[i],
                    actor_input[i],
                    action[i],               #优化！！！！！少了人力记录
                    reward[i],
                    next_augmented_state[i],
                    next_actor_input[i],
                    done if done.dim() == 0 else done[i] #优化！！！！，缺了一个状态
                )
        except Exception as e:
            print(f"[WARNING] Failed to add to buffer: {e}")
    
    def _update_networks(self):
        """Update networks using paper update laws"""
        try:
            self.trainer.update_networks(self.replay_buffer)
        except Exception as e:
            print(f"[WARNING] Network update failed: {e}")
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
        }, path)
        print(f"[INFO] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[INFO] Model loaded from {path}")