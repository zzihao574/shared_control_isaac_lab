"""Surgical robot shared control trainer with optimized state management"""

import torch
import yaml
import os
import wandb
import numpy as np
from typing import Dict, Any, Tuple

from .actor_critic import SurgicalActorCritic


class SharedControlTrainer:
    """Surgical shared control trainer with optimized z_true_t management"""
    
    def __init__(self, env, params: Dict[str, Any], log_dir: str = None):
        self.env = env
        self.unwrapped_env = env.unwrapped  # Get the actual environment
        self.params = params
        self.device = torch.device(params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_envs = getattr(env, 'num_envs', 1)
        self.log_dir = log_dir or "logs"
        self.dt = params.get('dt', 0.01)
        
        # Initialize wandb
        self.use_wandb = params.get('logging', {}).get('wandb_logging', False)
        if self.use_wandb:
            wandb_cfg = params.get('wandb_config', {})
            wandb.init(
                project=wandb_cfg.get('project', 'surgical-shared-control'),
                config=params,
                name=f"surgical_y_axis_{params.get('seed', 42)}",
                tags=wandb_cfg.get('tags', ['surgical', 'y-axis'])
            )
        
        # Initialize networks
        self.policy = SurgicalActorCritic(params).to(self.device)
        
        # Paper parameters
        update_rates = params.get('update_rates', {})
        self.sigma_critic = update_rates.get('sigma_critic', 10.0)
        self.sigma_actor = update_rates.get('sigma_actor', 10.0)
        self.sigma_identifier = update_rates.get('sigma_identifier', 10.0)
        
        ctrl_params = params.get('control_parameters', {})
        self.K1_gain = ctrl_params.get('K1_gain', 1.0)
        self.K2_gain = ctrl_params.get('K2_gain', 8.0)
        self.Kid_gain = ctrl_params.get('Kid_gain', 4.0)
        self.kΓ_gain = ctrl_params.get('kΓ_gain', 3.5)
        self.psi = ctrl_params.get('psi', 1.0)
        
        self.max_force = params.get('constraints', {}).get('max_robot_force', 3.3)
        
        # State tracking with single source of truth
        self.z_true_t = None   # z(t) = [x(t), ẋ(t), f(t)] - maintained centrally
        self.z_hat_t = None    # ẑ(t) = [x̂(t), x̂̇(t), f̂(t)] - estimated state
        
        # DEBUG: Print control gains
        print(f"[DEBUG] Control gains: K1={self.K1_gain}, K2={self.K2_gain}, Kid={self.Kid_gain}")
        print(f"[DEBUG] Update rates: σc={self.sigma_critic}, σa={self.sigma_actor}, σid={self.sigma_identifier}")
        
    def train_on_policy(self, total_episodes: int):
        """Main training loop with optimized state management"""
        episode_returns = []
        
        for episode in range(total_episodes):
            episode_return = 0
            
            # Reset at t=0
            obs_dict, _ = self.env.reset()
            obs_t = obs_dict["policy"]  # [x(0), ẋ(0), q(0), q̇(0), f(0)] - 21D
            self.unwrapped_env.reset_trajectory()
            self.policy.reset_pe_time()
            
            # Initialize z_true_t from initial observation (only place we extract from obs)
            self._initialize_z_true_from_obs(obs_t)
            
            step_count = 0
            max_steps = self.params.get('max_eval_steps', 500)
            
            while step_count < max_steps:
                # Step from time t to t+1
                
                # 1. Compute control using maintained z_true_t (no extraction from obs)
                u_t, Za_t, z_bar_t = self._compute_robot_control(obs_t)
                
                # DEBUG: Print control output (first few steps)
                if step_count < 3 and episode % 100 == 0:
                    print(f"[DEBUG] Episode {episode}, Step {step_count} - Control u_t: min={u_t.min():.4f}, max={u_t.max():.4f}, mean={u_t.mean():.4f}")
                
                # 2. Environment step: t → t+1
                obs_t1_dict, reward_t, terminated, truncated, info = self.env.step(u_t)
                obs_t1 = obs_t1_dict["policy"]  # [x(t+1), ẋ(t+1), q(t+1), q̇(t+1), f(t+1)]
                done = (terminated | truncated).any()
                
                # 3. Update z_true_t from new observation
                self._update_z_true_from_obs(obs_t1)
                
                # 4. Update networks using time t parameters
                self._update_networks_paper(
                    u_t=u_t,
                    reward_t=reward_t,
                    Za_t=Za_t,
                    z_bar_t=z_bar_t
                )
                
                # 5. Step trajectory: t → t+1
                self.unwrapped_env.step_trajectory()
                
                # 6. Advance to next time step
                obs_t = obs_t1  # obs(t) ← obs(t+1)
                episode_return += reward_t.mean().item()
                step_count += 1
                
                if done:
                    break
            
            episode_returns.append(episode_return)
            
            # Logging
            if episode % self.params.get('log_frequency', 10) == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"Episode {episode}, Return: {episode_return:.3f}, Avg: {avg_return:.3f}")
                
                if self.use_wandb:
                    log_data = {
                        'episode': episode,
                        'episode_return': episode_return,
                        'avg_return_10': avg_return
                    }
                    if 'log' in info:
                        log_data.update({f"env_{k}": v for k, v in info['log'].items()})
                    wandb.log(log_data)
        
        if self.use_wandb:
            wandb.finish()
        
        return episode_returns
    
    def _initialize_z_true_from_obs(self, obs_t: torch.Tensor):
        """Initialize z_true_t from initial observation (episode start only)"""
        x_t = obs_t[..., :3]       # x(t)
        x_dot_t = obs_t[..., 3:6]  # ẋ(t)
        f_t = obs_t[..., 18:21]    # f(t)
        
        self.z_true_t = torch.cat([x_t, x_dot_t, f_t], dim=-1)  # z(t) = [x, ẋ, f]
        self.z_hat_t = self.z_true_t.clone()  # Initialize ẑ(0) = z(0)
    
    def _update_z_true_from_obs(self, obs_t1: torch.Tensor):
        """Update z_true_t from new observation"""
        x_t1 = obs_t1[..., :3]      # x(t+1)
        x_dot_t1 = obs_t1[..., 3:6] # ẋ(t+1)
        f_t1 = obs_t1[..., 18:21]   # f(t+1)
        
        self.z_true_t = torch.cat([x_t1, x_dot_t1, f_t1], dim=-1)  # Update z(t) ← z(t+1)
    
    def _compute_robot_control(self, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute robot control using maintained z_true_t"""
        # Extract states from maintained z_true_t (no redundant construction)
        x_t = self.z_true_t[..., :3]      # x(t)
        x_dot_t = self.z_true_t[..., 3:6] # ẋ(t)
        f_t = self.z_true_t[..., 6:9]     # f(t)
        
        # Joint states from observation (not in z_true_t)
        q_t = obs_t[..., 6:12]      # q(t)
        q_dot_t = obs_t[..., 12:18] # q̇(t)
        
        # Get trajectory state at time t (real-time calculation)
        traj_state = self.unwrapped_env.get_complete_trajectory_state(x_t)
        xd_t = traj_state['xd']         # xd(t)
        xd_dot_t = traj_state['xd_dot'] # ẋd(t)  
        xd_ddot_t = traj_state['xd_ddot'] # ẍd(t)
        
        # Tracking errors at time t
        e_t = x_t - xd_t                    # e(t) = x(t) - xd(t)
        e_dot_t = x_dot_t - xd_dot_t        # ė(t) = ẋ(t) - ẋd(t)
        
        # Reference trajectory at time t
        xr_dot_t = xd_dot_t - self.K1_gain * e_t   # ẋr(t)
        xr_ddot_t = xd_ddot_t - self.K1_gain * e_dot_t # ẍr(t)
        
        # Sliding error at time t
        ev_t = x_dot_t - xr_dot_t   # ev(t) = ẋ(t) - ẋr(t)
        
        # Actor input at time t: Za(t) = [q(t), q̇(t), ẋr(t), ẍr(t)]
        Za_t = torch.cat([q_t, q_dot_t, xr_dot_t, xr_ddot_t], dim=-1)
        
        # Actor output with exploration noise at time t (training uses exploration)
        actor_output_t = self.policy.get_action_now(Za_t, self.dt, add_exploration=True)
        
        # Robot control at time t: u(t) = Ŵa^T Sa(Za(t)) - f(t) - e(t) - K2*ev(t)
        K2_matrix = torch.eye(3, device=self.device) * self.K2_gain
        u_t = (actor_output_t - f_t - e_t - (K2_matrix @ ev_t.T).T)
        # Note: Force limiting will be done in environment's _pre_physics_step()
        
        # Augmented state at time t: z̄(t) = [z(t), xd(t)] - use maintained z_true_t
        z_bar_t = torch.cat([self.z_true_t, xd_t], dim=-1)
        
        return u_t, Za_t, z_bar_t
    
    def _update_networks_paper(self, u_t: torch.Tensor, reward_t: torch.Tensor,
                              Za_t: torch.Tensor, z_bar_t: torch.Tensor):
        """Update networks using exact paper formulas"""
        # All computations at time t using maintained states
        z_tilde_t = self.z_hat_t - self.z_true_t  # z̃(t) = ẑ(t) - z(t)
        
        # 1. Identifier forward at time t: ż̂(t) = Ŵid^T Sid(ẑ(t), u(t)) - Kid*z̃(t)
        z_hat_dot_t = self.policy.predict_dynamics_now(self.z_hat_t, u_t) - self.Kid_gain * z_tilde_t
        
        # 2. Critic forward at time t: Γ̂(t) = Ŵc^T Sc(z̄(t))
        Gamma_hat_t = self.policy.evaluate_value_now(z_bar_t)
        
        # 3. Compute Λ(t) = -(1/ψ)Sc(z̄(t)) + ∇Sc*z̄̇ĥat(t)
        rbf_features_critic = self.policy.critic.rbf_layer(z_bar_t)
        
        # Get ẋd(t) at time t from environment - already [num_envs, 3]
        xd_dot_t = self.unwrapped_env.xd_dot_t  # No need to unsqueeze/expand
        
        # z̄̇ĥat(t) = [ż̂(t), ẋd(t)]
        z_bar_hat_dot_t = torch.cat([z_hat_dot_t, xd_dot_t], dim=-1)
        
        grad_Sc = self._compute_rbf_gradient(z_bar_t, rbf_features_critic)
        Lambda_t = (-(1.0 / self.psi) * rbf_features_critic + 
                     torch.sum(grad_Sc * z_bar_hat_dot_t.unsqueeze(1), dim=-1))
        
        # 4. Update networks (130 individual weight updates using time t parameters)
        self._update_critic_weights_individual(rbf_features_critic, Lambda_t, reward_t)
        self._update_identifier_weights_individual(self.z_hat_t, u_t, z_tilde_t)
        self._update_actor_weights_individual(Za_t, Gamma_hat_t)
        
        # 5. Update state estimate: ẑ(t+1) = ẑ(t) + ż̂(t) * dt
        self.z_hat_t = self.z_hat_t + z_hat_dot_t * self.dt
    
    def _update_identifier_weights_individual(self, z_hat_t: torch.Tensor, u_t: torch.Tensor, z_tilde_t: torch.Tensor):
        """Update identifier: 90 individual weight updates (9 states × 10 RBF nodes)"""
        combined_input = torch.cat([z_hat_t, u_t], dim=-1)
        rbf_features = self.policy.identifier.rbf_layer(combined_input)  # (num_envs, 10)
        
        with torch.no_grad():
            # Vectorized computation for all 90 weights
            rbf_expanded = rbf_features.unsqueeze(-1)  # (num_envs, 10, 1)
            z_tilde_expanded = z_tilde_t.unsqueeze(1)    # (num_envs, 1, 9)
            
            # Compute gradient terms: Sid_j * z̃i for all i,j pairs
            grad_matrix = rbf_expanded * z_tilde_expanded  # (num_envs, 10, 9)
            grad_terms = torch.sum(grad_matrix, dim=0).T   # (9, 10) - sum over envs
            
            # Current weights and regularization
            current_weights = self.policy.identifier.output_layer.weight  # (9, 10)
            reg_terms = self.sigma_identifier * current_weights
            
            # Individual weight updates: Ŵ̇id,ij = -Sid_j * z̃i - σid * Ŵid,ij
            weight_updates = -self.dt * (grad_terms + reg_terms)
            
            # Apply updates
            self.policy.identifier.output_layer.weight += weight_updates
    
    def _update_critic_weights_individual(self, rbf_features: torch.Tensor, Lambda_t: torch.Tensor, reward_t: torch.Tensor):
        """Update critic: 10 individual weight updates (1 output × 10 RBF nodes)"""
        with torch.no_grad():
            current_weights = self.policy.critic.output_layer.weight[0, :]  # (10,)
            
            # Compute Ŵc^T * Λ for each environment
            weighted_lambda = torch.mm(Lambda_t, current_weights.unsqueeze(-1)).squeeze(-1)  # (num_envs,)
            
            # Error term: r(t) + Ŵc^T * Λ
            error_term = reward_t + weighted_lambda  # (num_envs,)
            
            # Gradient for each RBF node: (r + Ŵc^T*Λ) * Λj
            weight_gradients = torch.sum(error_term.unsqueeze(-1) * Lambda_t, dim=0)  # (10,)
            
            # Individual weight updates: Ŵ̇c,j = -σc * (r + Ŵc^T*Λ) * Λj
            weight_updates = -self.sigma_critic * self.dt * weight_gradients
            
            # Apply updates
            self.policy.critic.output_layer.weight[0, :] += weight_updates
    
    def _update_actor_weights_individual(self, Za_t: torch.Tensor, Gamma_hat_t: torch.Tensor):
        """Update actor: 30 individual weight updates (3 actions × 10 RBF nodes)"""
        rbf_features = self.policy.actor.rbf_layer(Za_t)  # (num_envs, 10)
        
        with torch.no_grad():
            current_weights = self.policy.actor.output_layer.weight  # (3, 10)
            
            # Compute Ŵa,i^T * Sa for each action dimension
            weighted_features = torch.mm(rbf_features, current_weights.T)  # (num_envs, 3)
            
            # Error term: Ŵa,i^T*Sa + kΓ*Γ̂ for each action dimension
            Gamma_expanded = Gamma_hat_t.expand(-1, 3)  # (num_envs, 3)
            error_terms = weighted_features + self.kΓ_gain * Gamma_expanded  # (num_envs, 3)
            
            # For each action dimension i and RBF node j
            for i in range(3):  # 3 action dimensions
                error_i = error_terms[:, i]  # (num_envs,)
                
                # Gradient: (Ŵa,i^T*Sa + kΓ*Γ̂) * Sa_j for each RBF node j
                weight_gradients_i = torch.sum(error_i.unsqueeze(-1) * rbf_features, dim=0)  # (10,)
                
                # Individual weight updates: Ŵ̇a,ij = -σa * (Ŵa,i^T*Sa + kΓ*Γ̂) * Sa_j
                weight_updates_i = -self.sigma_actor * self.dt * weight_gradients_i
                
                # Apply updates for action dimension i
                self.policy.actor.output_layer.weight[i, :] += weight_updates_i
    
    def _compute_rbf_gradient(self, inputs: torch.Tensor, rbf_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF gradient ∇Sc with respect to z_bar_t (augmented state)
        
        For RBF: si(z̄) = exp(-(z̄-μi)^T(z̄-μi)/ηi^2)
        Gradient: ∂si/∂z̄ = -2(z̄-μi)/ηi^2 * si(z̄)
        
        Args:
            inputs: z_bar_t augmented state [num_envs, 12] where z̄ = [z, xd]
            rbf_outputs: RBF activations [num_envs, 10]
        
        Returns:
            grad: Gradient ∇Sc [num_envs, num_nodes, augmented_dim]
        """
        # Get RBF centers and width from critic network
        centers = self.policy.critic.rbf_layer.centers  # [num_nodes, augmented_dim=12]
        width = self.policy.critic.rbf_layer.width      # scalar
        
        # Expand dimensions for broadcasting
        inputs_expanded = inputs.unsqueeze(1)           # [num_envs, 1, 12]
        centers_expanded = centers.unsqueeze(0)         # [1, num_nodes, 12]
        
        # Compute difference: (z̄ - μi)
        diff = inputs_expanded - centers_expanded       # [num_envs, num_nodes, 12]
        
        # Compute gradient: ∂si/∂z̄ = -2(z̄-μi)/η^2 * si(z̄)
        # Note: rbf_outputs already contains si(z̄)
        grad = -2 * diff / (width**2) * rbf_outputs.unsqueeze(-1)  # [num_envs, num_nodes, 12]
        
        return grad
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'params': self.params,
        }, path)
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
    
    def evaluate_policy(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate policy with deterministic actions"""
        total_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs_dict, _ = self.env.reset()
            obs_t = obs_dict["policy"]
            self.unwrapped_env.reset_trajectory()
            self.policy.reset_pe_time()
            
            # Initialize z_true_t for evaluation
            self._initialize_z_true_from_obs(obs_t)
            
            episode_reward = 0
            step_count = 0
            max_steps = self.params.get('max_eval_steps', 500)
            
            while step_count < max_steps:
                # Use maintained z_true_t for evaluation
                x_t = self.z_true_t[..., :3]
                x_dot_t = self.z_true_t[..., 3:6]
                f_t = self.z_true_t[..., 6:9]
                
                # Joint states from observation
                q_t = obs_t[..., 6:12]
                q_dot_t = obs_t[..., 12:18]
                
                traj_state = self.unwrapped_env.get_complete_trajectory_state(x_t)
                xd = traj_state['xd']
                xd_dot = traj_state['xd_dot']
                xd_ddot = traj_state['xd_ddot']
                
                e = x_t - xd
                e_dot = x_dot_t - xd_dot
                xr_dot = xd_dot - self.K1_gain * e
                xr_ddot = xd_ddot - self.K1_gain * e_dot
                
                Za_t = torch.cat([q_t, q_dot_t, xr_dot, xr_ddot], dim=-1)
                
                with torch.no_grad():
                    actor_output = self.policy.get_action_now(Za_t, self.dt, add_exploration=False)
                
                ev = x_dot_t - xr_dot
                K2_matrix = torch.eye(3, device=self.device) * self.K2_gain
                action = actor_output - f_t - e - (K2_matrix @ ev.T).T
                # Force limiting will be done in environment
                
                obs_dict, reward, terminated, truncated, info = self.env.step(action)
                obs_t = obs_dict["policy"]
                
                # Update z_true_t for next iteration
                self._update_z_true_from_obs(obs_t)
                
                episode_reward += reward.mean().item()
                step_count += 1
                
                if (terminated | truncated).any():
                    if 'log' in info and info['log'].get('distance_to_target', float('inf')) < 0.02:
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': success_count / num_episodes,
            'episodes_evaluated': num_episodes
        }