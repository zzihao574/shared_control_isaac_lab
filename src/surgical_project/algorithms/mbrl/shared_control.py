# surgical_project/algorithms/shared_control.py - Simplified version aligned with paper
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
from .actor_critic import SurgicalActorCritic

class ReplayBuffer:
    """Simple Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.stack([b[2] for b in batch])
        next_obs = torch.stack([b[3] for b in batch])
        dones = torch.stack([b[4] for b in batch])
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)


class HumanDynamicsModel:
    """Simplified Human Dynamics Model - Aligned with paper Equation (6)"""
    def __init__(self, device, config=None):
        self.device = device
        
        # Human impedance parameters from paper Equation (6): CHx˙ + KH(x - xH) = -f
        if config is not None:
            self.CH = config.get('human_damping', 21.0)     # Damping coefficient
            self.KH = config.get('human_stiffness', 201.0)   # Stiffness coefficient
        else:
            self.CH = 21.0   # Default damping
            self.KH = 201.0  # Default stiffness
            
        # Human workspace parameters
        self.workspace_radius = 0.2  # 20cm radius
        self.max_human_velocity = 0.3  # 30cm/s maximum velocity
        
        print(f"[INFO] Human dynamics model initialized:")
        print(f"  - Damping (CH): {self.CH}")
        print(f"  - Stiffness (KH): {self.KH}")
        print(f"  - Workspace radius: {self.workspace_radius*100:.0f}cm")
        
    def get_human_intention(self, obs, interaction_force):
        """Estimate human intention based on paper's impedance model"""
        try:
            # Ensure proper tensor dimensions
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if interaction_force.dim() == 1:
                interaction_force = interaction_force.unsqueeze(0)
            
            batch_size = obs.shape[0]
            
            # Extract position and velocity (first 6 dimensions of observation)
            current_pos = obs[..., :3]   # Current position
            current_vel = obs[..., 3:6]  # Current velocity
            
            # If trajectory information is available (obs_dim >= 12)
            if obs.shape[-1] >= 12:
                target_pos = obs[..., 6:9]   # Trajectory target position
                target_vel = obs[..., 9:12]  # Trajectory target velocity
            else:
                # Default to workspace center if no trajectory info
                target_pos = torch.zeros_like(current_pos)
                target_vel = torch.zeros_like(current_vel)
            
            # Human intention based on impedance control (paper Eq. 6)
            # From CHx˙ + KH(x - xH) = -f, we can estimate xH (human intention)
            # Rearranging: xH = x + (f + CHx˙)/KH
            
            force_2d = interaction_force[..., :2]  # Only x-y components
            pos_2d = current_pos[..., :2]
            vel_2d = current_vel[..., :2]
            
            # Estimate human intended position using impedance model
            impedance_term = (force_2d + self.CH * vel_2d) / self.KH
            human_intended_pos_2d = pos_2d + impedance_term
            
            # Combine with trajectory following
            if obs.shape[-1] >= 12:
                # Weight between trajectory following and human intention
                trajectory_weight = 0.6
                human_weight = 0.4
                
                intended_pos_2d = (trajectory_weight * target_pos[..., :2] + 
                                 human_weight * human_intended_pos_2d)
            else:
                intended_pos_2d = human_intended_pos_2d
            
            # Create 3D intention (keep z component from trajectory or current)
            intention_3d = current_pos.clone()
            intention_3d[..., :2] = intended_pos_2d
            
            if obs.shape[-1] >= 12:
                intention_3d[..., 2] = target_pos[..., 2]  # Use trajectory z
            
            # Ensure intention is within workspace
            workspace_center = torch.zeros_like(intention_3d)
            to_center = intention_3d - workspace_center
            distance_2d = torch.norm(to_center[..., :2], dim=-1, keepdim=True)
            
            # Clamp to workspace radius
            scale = torch.clamp(distance_2d / self.workspace_radius, max=1.0)
            intention_3d[..., :2] = workspace_center[..., :2] + to_center[..., :2] / scale
            
            return intention_3d
            
        except Exception as e:
            print(f"[WARNING] Human intention estimation failed: {e}")
            # Return safe default (current position)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            return obs[..., :3].clone()
    
    def get_human_action(self, current_pos, intention_pos, dt=0.01):
        """Calculate human action based on intention - simplified impedance control"""
        try:
            # Ensure proper dimensions
            if current_pos.dim() == 1:
                current_pos = current_pos.unsqueeze(0)
            if intention_pos.dim() == 1:
                intention_pos = intention_pos.unsqueeze(0)
            
            # Calculate desired velocity (position error)
            position_error = intention_pos - current_pos
            desired_velocity = position_error / dt
            
            # Limit velocity to human capabilities
            velocity_norm = torch.norm(desired_velocity, dim=-1, keepdim=True)
            velocity_scale = torch.clamp(velocity_norm / self.max_human_velocity, max=1.0)
            desired_velocity = desired_velocity / (velocity_scale + 1e-6)
            
            # Convert to action (simplified force control)
            human_action = desired_velocity * 0.1  # Scale to reasonable force range
            
            return torch.clamp(human_action, -1.0, 1.0)
            
        except Exception as e:
            print(f"[WARNING] Human action calculation failed: {e}")
            # Return zero action as safe default
            if current_pos.dim() == 1:
                current_pos = current_pos.unsqueeze(0)
            return torch.zeros_like(current_pos)


class SharedControlTrainer:
    """Simplified Shared Control Trainer - Aligned with paper framework"""
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get environment properties
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # Detect observation dimensions from environment
        print("[INFO] Detecting observation dimensions...")
        obs_dict, _ = env.reset()
        actual_obs_dim = obs_dict["policy"].shape[-1]
        action_dim = 3  # xyz forces
        
        print(f"[INFO] Observation dimension: {actual_obs_dim}")
        print(f"[INFO] Action dimension: {action_dim}")
        
        # Initialize networks
        self.policy = SurgicalActorCritic(actual_obs_dim, action_dim).to(self.device)
        
        # Safely extract numeric values from config
        def safe_float(key, default):
            value = config.get(key, default)
            try:
                return float(value)
            except (ValueError, TypeError):
                print(f"[WARNING] Invalid {key}: {value}, using default {default}")
                return float(default)
        
        def safe_int(key, default):
            value = config.get(key, default)
            try:
                return int(value)
            except (ValueError, TypeError):
                print(f"[WARNING] Invalid {key}: {value}, using default {default}")
                return int(default)
        
        # Extract configuration with safe conversion
        lr = safe_float('learning_rate', 3e-4)
        id_lr = safe_float('identifier_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr)
        self.identifier_optimizer = optim.Adam(self.policy.identifier.parameters(), lr=id_lr)
        
        # Experience replay buffer
        buffer_size = safe_int('buffer_size', 50000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Human dynamics model with safe config
        human_config = {
            'human_damping': safe_float('human_damping', 21.0),
            'human_stiffness': safe_float('human_stiffness', 201.0),
        }
        self.human_dynamics = HumanDynamicsModel(self.device, human_config)
        
        # Collaboration parameters (from paper's shared control framework)
        self.robot_weight = safe_float('robot_action_weight', 0.7)  # α in paper
        self.human_weight = 1.0 - self.robot_weight                # 1-α in paper
        
        # Training parameters
        self.batch_size = safe_int('batch_size', 256)
        self.min_buffer_size = safe_int('min_buffer_size', 1000)
        self.gamma = safe_float('gamma', 0.99)
        self.tau = safe_float('tau', 0.005)
        self.max_grad_norm = safe_float('max_grad_norm', 0.5)
        
        # Initialize interaction forces
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        print(f"[INFO] SharedControlTrainer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Collaboration weights: {self.robot_weight:.1f} robot, {self.human_weight:.1f} human")
        print(f"  - Buffer size: {buffer_size}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Human damping: {human_config['human_damping']}")
        print(f"  - Human stiffness: {human_config['human_stiffness']}")
        
    def train(self, total_steps: int):
        """Main training loop - simplified and stable"""
        print(f"[INFO] Starting training for {total_steps} steps...")
        
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        step_rewards = []
        update_count = 0
        
        for step in range(total_steps):
            try:
                # Get robot action
                with torch.no_grad():
                    robot_action = self.policy.get_action(obs)
                    robot_action = torch.clamp(robot_action, -1.0, 1.0)
                
                # Get human action (based on paper's human dynamics model)
                current_pos = obs[..., :3]
                human_intention = self.human_dynamics.get_human_intention(obs, self.interaction_forces)
                human_action = self.human_dynamics.get_human_action(current_pos, human_intention)
                
                # Collaborative action fusion (paper's shared control framework)
                final_action = self.robot_weight * robot_action + self.human_weight * human_action
                final_action = torch.clamp(final_action, -1.0, 1.0)
                
                # Environment step
                next_obs_dict, reward, terminated, truncated, info = self.env.step(final_action)
                next_obs = next_obs_dict["policy"]
                done = terminated | truncated
                
                # Update interaction forces (simplified from observation)
                if obs.shape[-1] >= 15:  # Has constraint information
                    self.interaction_forces = obs[..., 12:15] * 0.1  # Use constraint normals as proxy
                
                # Store experience
                for i in range(self.num_envs):
                    self.replay_buffer.add(
                        obs[i].cpu(), final_action[i].cpu(), reward[i].cpu(),
                        next_obs[i].cpu(), done[i].cpu()
                    )
                
                step_rewards.append(reward.mean().item())
                
                # Network update
                if len(self.replay_buffer) > self.min_buffer_size and step % 20 == 0:
                    self.update_networks()
                    update_count += 1
                
                obs = next_obs
                
                # Progress logging
                if step % 50 == 0:
                    avg_reward = np.mean(step_rewards[-50:]) if step_rewards else 0
                    print(f"Step {step:5d} | Reward: {avg_reward:.3f} | "
                          f"Buffer: {len(self.replay_buffer)} | Updates: {update_count}")
                
            except Exception as e:
                print(f"[WARNING] Step {step} failed: {e}")
                # Reset environment on error
                obs_dict, _ = self.env.reset()
                obs = obs_dict["policy"]
                continue
        
        print(f"[INFO] Training completed. Total updates: {update_count}")
    
    def update_networks(self):
        """Update actor, critic, and identifier networks"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = [x.to(self.device) for x in [obs, actions, rewards, next_obs, dones]]
        
        # Update Critic (Q and V networks)
        with torch.no_grad():
            next_actions = self.policy.get_action(next_obs, deterministic=True)
            next_q = self.policy.critic.forward_q(next_obs, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones.float())
            
        current_q = self.policy.critic.forward_q(obs, actions).squeeze()
        current_v = self.policy.critic.forward_v(obs).squeeze()
        
        # Critic losses
        q_loss = F.mse_loss(current_q, target_q)
        v_loss = F.mse_loss(current_v, target_q)  # V should match target Q
        critic_loss = q_loss + v_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions = self.policy.get_action(obs, deterministic=True)
        actor_q = self.policy.critic.forward_q(obs, new_actions)
        actor_loss = -actor_q.mean()  # Maximize Q value
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update Dynamics Identifier (simple supervised learning)
        with torch.no_grad():
            # Use finite differences to estimate state derivatives
            state_current = obs[..., :6]  # position and velocity
            state_next = next_obs[..., :6]
            state_dot_target = (state_next - state_current) / 0.01  # Assume 100Hz
        
        state_dot_pred = self.policy.identifier(obs, actions)
        identifier_loss = F.mse_loss(state_dot_pred[..., :6], state_dot_target)
        
        self.identifier_optimizer.zero_grad()
        identifier_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.identifier.parameters(), self.max_grad_norm)
        self.identifier_optimizer.step()