# surgical_project/algorithms/shared_control.py - Enhanced CUDA-synchronized version
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import time
from .actor_critic import SurgicalActorCritic

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done, state=None, next_state=None):
        self.buffer.append((obs, action, reward, next_obs, done, state, next_state))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.stack([b[2] for b in batch])
        next_obs = torch.stack([b[3] for b in batch])
        dones = torch.stack([b[4] for b in batch])
        
        states = torch.stack([b[5] for b in batch]) if batch[0][5] is not None else obs
        next_states = torch.stack([b[6] for b in batch]) if batch[0][6] is not None else next_obs
        
        return obs, actions, rewards, next_obs, dones, states, next_states
    
    def __len__(self):
        return len(self.buffer)

class HumanDynamicsModel:
    """Human Dynamics Modeling - Fix matrix operations"""
    def __init__(self, device):
        self.device = device
        self.CH = torch.tensor([[21.0, 0.0], [0.0, 21.0]], device=device)
        self.KH = torch.tensor([[201.0, 0.0], [0.0, 201.0]], device=device)
        self.eps = 1e-6
        self.max_force = 5.0
        self.KH_inv = torch.inverse(self.KH + torch.eye(2, device=device) * self.eps)
        
    def get_human_intention(self, obs, interaction_force):
        """Estimate human intention based on impedance control model"""
        try:
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
            interaction_force = torch.nan_to_num(interaction_force, nan=0.0, posinf=self.max_force, neginf=-self.max_force)
            
            pos = obs[..., :3]
            vel = obs[..., 3:6] if obs.shape[-1] > 3 else torch.zeros_like(pos)
            
            pos_2d = pos[..., :2]
            vel_2d = vel[..., :2]
            force_2d = interaction_force[..., :2]
            
            force_norm = torch.norm(force_2d, dim=-1, keepdim=True)
            force_scale = torch.clamp(force_norm / (self.max_force + self.eps), max=1.0)
            force_2d = force_2d / (force_scale + self.eps)
            
            damping_term = vel_2d @ self.CH
            combined_force = damping_term + force_2d
            intended_offset = combined_force @ self.KH_inv
            
            intended_offset = torch.clamp(intended_offset, -0.2, 0.2)
            human_intention = pos_2d + intended_offset
            
            intention_3d = pos.clone()
            intention_3d[..., :2] = human_intention
            intention_3d = torch.nan_to_num(intention_3d, nan=0.0)
            
            return intention_3d
            
        except Exception as e:
            print(f"Human intention estimation failed: {e}")
            return obs[..., :3] if obs.shape[-1] >= 3 else torch.zeros((obs.shape[0], 3), device=self.device)

class SharedControlTrainer:
    """Shared Control Trainer with Enhanced CUDA Synchronization"""
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = env.device
        
        # Numerical stability parameters
        self.eps = 1e-6
        self.max_action_norm = 0.3
        self.max_reward = 50.0
        
        # Network initialization
        obs_dim = env.cfg.observation_space
        action_dim = env.cfg.action_space
        state_dim = obs_dim
        
        self.policy = SurgicalActorCritic(obs_dim, action_dim, state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=config.learning_rate)
        self.identifier_optimizer = optim.Adam(self.policy.identifier.parameters(), lr=config.identifier_lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Human dynamics model
        self.human_dynamics = HumanDynamicsModel(self.device)
        
        # Weight parameters
        self.Q1_diag = torch.ones(3, device=self.device) * config.q1_weight
        self.Q2_diag = torch.ones(3, device=self.device) * config.q2_weight
        self.Q3_diag = torch.ones(3, device=self.device) * config.q3_weight
        self.R_diag = torch.ones(action_dim, device=self.device) * config.r_weight
        
        # State estimation
        self.estimated_state = torch.zeros(self.env.num_envs, state_dim, device=self.device)
        
        print("SharedControlTrainer initialized (Enhanced CUDA version)")
        
    def safe_action_processing(self, action):
        """Safe action post-processing"""
        action = torch.nan_to_num(action, nan=0.0, posinf=self.max_action_norm, neginf=-self.max_action_norm)
        action = torch.clamp(action, -self.max_action_norm, self.max_action_norm)
        return action
        
    def train(self, total_steps: int):
        """Main training loop with CUDA synchronization"""
        try:
            # Initial CUDA sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            obs_dict, _ = self.env.reset()
            obs = obs_dict["policy"]
            
            # Sync after reset
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.estimated_state = obs.clone()
            successful_steps = 0
            step_rewards = []
            
            for step in range(total_steps):
                try:
                    # Sync before each step
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Safety check
                    obs = torch.nan_to_num(obs, nan=0.0)
                    
                    # Get robot action
                    with torch.no_grad():
                        robot_action = self.policy.get_action(obs)
                        robot_action = self.safe_action_processing(robot_action)
                    
                    # Sync after action
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Simplified human action
                    human_action = torch.zeros_like(robot_action)
                    if robot_action.shape[-1] >= 3:
                        if obs.shape[-1] >= 10:
                            current_pos = obs[..., :3]
                            target_pos = obs[..., 7:10]
                            direction = target_pos - current_pos
                            direction = torch.nan_to_num(direction, nan=0.0)
                            direction = torch.clamp(direction, -0.5, 0.5)
                            human_action[:, :3] = direction * 0.05
                    
                    human_action = self.safe_action_processing(human_action)
                    
                    # Simplified action fusion
                    final_action = 0.9 * robot_action + 0.1 * human_action
                    final_action = self.safe_action_processing(final_action)
                    
                    # Sync after action fusion
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    current_state = self.estimated_state.clone()
                    
                    # Environment interaction
                    next_obs_dict, reward, terminated, truncated, info = self.env.step(final_action)
                    
                    # Sync after env step
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    next_obs = next_obs_dict["policy"]
                    done = terminated | truncated
                    
                    # Safety check reward
                    reward = torch.nan_to_num(reward, nan=0.0)
                    reward = torch.clamp(reward, -self.max_reward, self.max_reward)
                    
                    # Simplified state estimation
                    self.estimated_state = next_obs.clone()
                    next_state = self.estimated_state.clone()
                    
                    # Store experience
                    for i in range(self.env.num_envs):
                        try:
                            self.replay_buffer.add(
                                obs[i].cpu(), final_action[i].cpu(), reward[i].cpu(),
                                next_obs[i].cpu(), done[i].cpu(),
                                current_state[i].cpu(), next_state[i].cpu()
                            )
                        except:
                            pass
                    
                    step_rewards.extend(reward.cpu().numpy())
                    successful_steps += 1
                    
                    # Network update - less frequent and with sync
                    if len(self.replay_buffer) > self.config.min_buffer_size and step % 25 == 0:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            self.update_networks_safe()
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                
                        except Exception as e:
                            print(f"Network update failed: {str(e)[:50]}...")
                    
                    obs = next_obs
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Progress logging
                    if step % 25 == 0:
                        avg_reward = np.mean(step_rewards[-25:]) if step_rewards else 0
                        success_rate = successful_steps / (step + 1)
                        print(f"Step {step:5d} | Avg. Reward: {avg_reward:.3f} | Buffer: {len(self.replay_buffer)} | Success Rate: {success_rate:.1%}")
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                
                except Exception as e:
                    print(f"Training step {step} failed: {str(e)[:60]}...")
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    try:
                        obs_dict, _ = self.env.reset()
                        obs = obs_dict["policy"]
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                    except:
                        print(f"Environment reset failed, skipping step {step}")
                        continue
            
            print(f"Training finished: {successful_steps}/{total_steps} successful steps")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Critical error during training: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            raise e
    
    def update_networks_safe(self):
        """Safe network update"""
        try:
            if len(self.replay_buffer) < self.config.batch_size:
                return
                
            batch = self.replay_buffer.sample(min(self.config.batch_size, len(self.replay_buffer)))
            obs, actions, rewards, next_obs, dones, states, next_states = [b.to(self.device) for b in batch]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            obs = torch.nan_to_num(obs, nan=0.0)
            actions = torch.nan_to_num(actions, nan=0.0)
            rewards = torch.nan_to_num(rewards, nan=0.0)
            
            # Simplified critic update
            try:
                current_q = self.policy.critic.forward_q(obs, actions).squeeze()
                current_q = torch.nan_to_num(current_q, nan=0.0)
                
                target_q = rewards
                target_q = torch.clamp(target_q, -self.max_reward, self.max_reward)
                
                critic_loss = F.mse_loss(current_q, target_q)
                critic_loss = torch.clamp(critic_loss, 0, 100.0)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
            except Exception as e:
                print(f"Critic update failed: {str(e)[:40]}")
            
            # Less frequent actor update
            if torch.rand(1).item() < 0.2:
                try:
                    predicted_actions = self.policy.get_action(obs, deterministic=True)
                    predicted_actions = self.safe_action_processing(predicted_actions)
                    
                    q_values = self.policy.critic.forward_q(obs, predicted_actions)
                    q_values = torch.nan_to_num(q_values, nan=0.0)
                    
                    actor_loss = -q_values.mean()
                    actor_loss = torch.clamp(actor_loss, -100.0, 100.0)
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                except Exception as e:
                    print(f"Actor update failed: {str(e)[:40]}")
                    
        except Exception as e:
            print(f"Network update process failed: {str(e)[:40]}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
