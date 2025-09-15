"""
Multi-environment parallel MADDPG algorithm with shared network architecture.
Enhanced with per-agent metrics and target Q-value statistics.

Features:
- Single shared network per agent (not per environment)
- Joint replay buffer with concatenated observations/actions
- CTDE: Centralized training with decentralized execution
- Gradient norm monitoring for training stability
- Enhanced per-agent metrics with target Q-value tracking
- Global noise scaling for exploration schedule
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from .ddpg_agent import DDPGAgent
from .replay_buffer import JointReplayBuffer

class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient with shared network architecture.
    
    Architecture:
    - self.agents[agent_id] = One shared network per agent type
    - 512 parallel environments share these networks
    - Joint replay buffer stores (obs_all, act_all, rewards_vec, next_obs_all, done_any)
    - Training progress driven by global episode count
    """
    
    def __init__(self, num_envs: int, env, params: Dict[str, Any], device: str = 'cuda'):
        self.env = env
        self.actual_env = self._unwrap_environment(env)
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        
        # Get environment configuration
        self.agent_ids = list(self.actual_env.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        # Get dimensions from environment cfg
        self.obs_dims = [self.actual_env.cfg.observation_spaces[agent] for agent in self.agent_ids]
        self.action_dims = [self.actual_env.cfg.action_spaces[agent] for agent in self.agent_ids]
        self.total_obs_dim = sum(self.obs_dims)  # Total observation dimension
        self.total_action_dim = sum(self.action_dims)  # Total action dimension

        print(f"[MADDPG] Shared Network Architecture:")
        print(f"  Environments: {self.num_envs}")
        print(f"  Agent IDs: {self.agent_ids}")
        print(f"  Obs dims: {self.obs_dims} (total: {self.total_obs_dim})")
        print(f"  Action dims: {self.action_dims} (total: {self.total_action_dim})")
        print(f"  Networks: ONE per agent type (shared across ALL environments)")

        # Initialize shared agents and joint buffer
        self._initialize_agents()
        self._initialize_replay_buffers()
        self._build_slices()
        
        # Load training hyperparameters
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.batch_size = int(maddpg_cfg.get('batch_size', 512))
        self.gamma = float(maddpg_cfg.get('gamma', 0.95))
        self.update_interval = int(maddpg_cfg.get('update_interval', 100))
        self.min_buffer_size = int(maddpg_cfg.get('min_buffer_size', 4096))
        
        self.training_steps = 0  # Total training steps
        self.update_count = 0  # Update count
        
        print(f"[MADDPG] Shared network initialization complete")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Update interval: {self.update_interval}")
        print(f"  Min buffer size: {self.min_buffer_size}")
    
    def _unwrap_environment(self, env):
        """Get actual environment object."""
        return getattr(env, 'unwrapped', env)
        
    def _initialize_agents(self) -> None:
        """Initialize shared DDPG agents (one per agent type)."""
        self.agents = {}
        for i, agent_id in enumerate(self.agent_ids):
            self.agents[agent_id] = DDPGAgent(
                agent_id=agent_id,
                state_dim=self.obs_dims[i],
                action_dim=self.action_dims[i],
                total_state_dim=self.total_obs_dim,
                total_action_dim=self.total_action_dim,
                params=self.params,
                device=self.device,
            )
            print(f"[SHARED] Created shared network for agent: {agent_id}")
    
    def _initialize_replay_buffers(self) -> None:
        """Initialize joint replay buffer for shared architecture."""
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.buffer_size = int(maddpg_cfg.get('max_replay_buffer_len', 100000))
        
        self.replay = JointReplayBuffer(
            capacity=self.buffer_size,
            total_obs_dim=self.total_obs_dim,
            total_action_dim=self.total_action_dim,
            num_agents=self.num_agents,
            device=self.device,
        )
        print(f"[BUFFER] Joint replay buffer initialized: capacity={self.buffer_size}")
    
    def _build_slices(self) -> None:
        """Build slicing indices for concatenated observations and actions."""
        self.obs_slices = []  # Observation slices for each agent
        self.act_slices = []  # Action slices for each agent
        
        obs_offset = 0
        act_offset = 0
        
        for obs_dim, act_dim in zip(self.obs_dims, self.action_dims):
            self.obs_slices.append(slice(obs_offset, obs_offset + obs_dim))
            self.act_slices.append(slice(act_offset, act_offset + act_dim))
            obs_offset += obs_dim
            act_offset += act_dim
            
        print(f"[SLICES] Observation slices: {self.obs_slices}")
        print(f"[SLICES] Action slices: {self.act_slices}")

    def select_actions(self, observations: Dict[str, torch.Tensor], add_noise: bool, noise_scale: float = 1.0) -> tuple[Dict[str, torch.Tensor], Dict]:
        """
        Select actions using shared networks with global noise scaling.
        
        Args:
            observations: {agent_id: Tensor[num_envs, obs_dim]}
            add_noise: Whether to add exploration noise
            noise_scale: Global noise scaling factor for exploration schedule
            
        Returns:
            actions: {agent_id: Tensor[num_envs, action_dim]}
            detail: Detail information for console display
        """
        actions = {}
        detail = {"mean_actions": {}, "noise_actions": {}}  # Debug information
        
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]
            obs_i = observations[agent_id]  # [num_envs, obs_dim]
            
            with torch.no_grad():
                mean, std = agent.actor(obs_i)
                noise = (noise_scale * std * torch.randn_like(mean)) if add_noise else torch.zeros_like(mean)
                action = (mean + noise).clamp_(-agent.max_action, agent.max_action)
            
            # Single agent training: zero out human actions
            if agent_id == "human":
                mean = torch.zeros_like(mean)
                noise = torch.zeros_like(noise)
                action = torch.zeros_like(action)

            actions[agent_id] = action
            detail["mean_actions"][agent_id] = mean
            detail["noise_actions"][agent_id] = noise
        
        return actions, detail

    def add_experience_to_buffer(self, obs, actions, rewards, next_obs, dones):
        """
        Store transitions in joint replay buffer.
        
        Args:
            obs: {agent_id: Tensor[num_envs, obs_dim]}
            actions: {agent_id: Tensor[num_envs, action_dim]}
            rewards: {agent_id: Tensor[num_envs]}
            next_obs: {agent_id: Tensor[num_envs, obs_dim]}
            dones: {agent_id: Tensor[num_envs]}
        """
        for env_id in range(self.num_envs):
            # Concatenate observations and actions following agent_ids order
            obs_all = torch.cat([obs[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            act_all = torch.cat([actions[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            rew_vec = torch.stack([rewards[aid][env_id].reshape(()).float() for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            nobs_all = torch.cat([next_obs[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            
            # Compute done_any (logical OR over agents)
            done_any = False
            for aid in self.agent_ids:
                if bool(dones[aid][env_id]):
                    done_any = True
                    break
            
            self.replay.add(obs_all, act_all, rew_vec, nobs_all, done_any)

    def _module_grad_norm(self, module) -> float:
        """Calculate L2 gradient norm for a module."""
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def update(self) -> Dict[str, Any]:
        """
        CTDE update: Centralized Training, Decentralized Execution.
        Enhanced with per-agent metrics and target Q-value statistics.
        
        - Critics see global state-action space (∑obs, ∑act)
        - Actors update with respect to their own actions only
        - Other agents' actions are detached from gradient flow
        """
        if len(self.replay) < self.min_buffer_size:
            return {}

        self.training_steps += 1
        
        # Only update every update_interval steps
        if self.training_steps % self.update_interval != 0:
            return {}

        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return {}

        obs_all, act_all, rew_all, nobs_all, done_any = batch
        
        gamma = float(self.params.get('maddpg_config', {}).get('gamma', 0.95))

        # Enhanced statistics structure with per-agent metrics
        stats = {
            "loss/actor": {}, "loss/critic": {}, "q_mean": {}, "q_std": {},
            "q_target_mean": {}, "q_target_std": {},  # New: Target Q statistics
            "grad_norm/actor": {}, "grad_norm/critic": {}
        }

        # Compute target actions using target actors
        next_action_parts = []
        for i, agent_id in enumerate(self.agent_ids):
            slice_i = self.obs_slices[i]
            with torch.no_grad():
                next_action_i, _ = self.agents[agent_id].actor_target(nobs_all[:, slice_i])
            next_action_parts.append(next_action_i)
        next_act_all = torch.cat(next_action_parts, dim=-1)

        # Update each agent using CTDE
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]

            # Critic Update with enhanced statistics
            with torch.no_grad():
                q_next = agent.critic_target(nobs_all, next_act_all).squeeze(-1)  # [B]
                y = rew_all[:, i] + (1.0 - done_any.squeeze(-1)) * gamma * q_next  # [B]

            q = agent.critic(obs_all, act_all).squeeze(-1)  # [B]
            critic_loss = torch.nn.functional.smooth_l1_loss(q, y)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            c_grad_norm = self._module_grad_norm(agent.critic)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
            agent.critic_optimizer.step()

            # Store enhanced statistics including target Q values
            stats["loss/critic"][agent_id] = float(critic_loss.detach().cpu().item())
            stats["q_mean"][agent_id] = float(q.detach().cpu().mean().item())
            stats["q_std"][agent_id] = float(q.detach().cpu().std().item())
            stats["q_target_mean"][agent_id] = float(y.detach().cpu().mean().item())
            stats["q_target_std"][agent_id] = float(y.detach().cpu().std().item())
            stats["grad_norm/critic"][agent_id] = float(c_grad_norm)

            # Actor Update (CTDE: only own action has gradient)
            action_parts = []
            for j, agent_j in enumerate(self.agent_ids):
                slice_j = self.obs_slices[j]
                if j == i:  # Current agent: needs gradient
                    action_j, _ = self.agents[agent_j].actor(obs_all[:, slice_j])
                else:  # Other agents: detach gradients
                    with torch.no_grad():
                        action_j, _ = self.agents[agent_j].actor(obs_all[:, slice_j])
                    action_j = action_j.detach()
                action_parts.append(action_j)
            
            action_pred_all = torch.cat(action_parts, dim=-1)
            actor_loss = -agent.critic(obs_all, action_pred_all).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            a_grad_norm = self._module_grad_norm(agent.actor)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
            agent.actor_optimizer.step()

            # Soft target network updates
            agent.soft_update()

            # Store actor statistics
            stats["loss/actor"][agent_id] = float(actor_loss.detach().cpu().item())
            stats["grad_norm/actor"][agent_id] = float(a_grad_norm)

        # Aggregate statistics with averages
        for k in ["loss/critic", "loss/actor", "q_mean", "q_std", "q_target_mean", "q_target_std", "grad_norm/actor", "grad_norm/critic"]:
            if stats[k]:
                stats[f"{k}/avg"] = float(np.mean(list(stats[k].values())))

        self.update_count += 1
        stats["training/updates"] = int(self.update_count)

        return stats