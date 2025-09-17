"""
Multi-environment parallel MADDPG algorithm with shared network architecture.
Enhanced with per-agent metrics and target Q-value statistics.
Modified: Asynchronous update frequencies - Critic updates 2x more than Actor.

Features:
- Single shared network per agent (not per environment)
- Joint replay buffer with concatenated observations/actions
- CTDE: Centralized training with decentralized execution
- Gradient norm monitoring for training stability
- Enhanced per-agent metrics with target Q-value tracking
- Global noise scaling for exploration schedule
- IDENTICAL NETWORK INITIALIZATION with proper optimizer separation
- Force constraints applied only in select_actions method
- ASYNC UPDATES: Critic updates every interval, Actor updates every 2*interval
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
    - IDENTICAL INITIALIZATION: Both agents start with identical weights
    - ASYNC UPDATES: Critic:Actor = 2:1 update ratio
    """
    
    def __init__(self, num_envs: int, env, params: Dict[str, Any], device: str = 'cuda'):
        self.env = env
        self.actual_env = self._unwrap_environment(env)
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        
        # Set global seeds FIRST for identical initialization
        self._set_global_seeds()
        
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

        # Initialize shared agents with IDENTICAL weights
        self._initialize_agents_with_identical_init()
        self._initialize_replay_buffers()
        self._build_slices()
        
        # Load training hyperparameters
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.batch_size = int(maddpg_cfg.get('batch_size', 512))
        self.gamma = float(maddpg_cfg.get('gamma', 0.95))
        self.update_interval = int(maddpg_cfg.get('update_interval', 100))
        self.min_buffer_size = int(maddpg_cfg.get('min_buffer_size', 4096))
        
        self.training_steps = 0  # Total training steps
        self.critic_update_count = 0  # Critic update count
        self.actor_update_count = 0   # Actor update count
        
        print(f"[MADDPG] Shared network initialization complete")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Critic update interval: {self.update_interval}")
        print(f"  Actor update interval: {self.update_interval * 2}")
        print(f"  Min buffer size: {self.min_buffer_size}")
    
    def _set_global_seeds(self) -> None:
        """Set all random seeds for reproducible initialization."""
        seed = int(self.params.get("seed", 42))
        import random
        random.seed(seed)
        np.random.seed(seed)  
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[SEED] Global seeds set to {seed}")
    
    def _unwrap_environment(self, env):
        """Get actual environment object."""
        return getattr(env, 'unwrapped', env)
        
    def _initialize_agents_with_identical_init(self) -> None:
        """Initialize agents with identical network weights but separate optimizers."""
        self.agents = {}
        
        # Create first agent (human) with seeded initialization
        first_agent_id = self.agent_ids[0]  # Usually "human"
        self.agents[first_agent_id] = self._build_single_agent(0, first_agent_id)
        print(f"[SHARED] Created shared network for agent: {first_agent_id}")
        
        # Create second agent (robot) with separate optimizers
        second_agent_id = self.agent_ids[1]  # Usually "robot" 
        second_agent = self._build_single_agent(1, second_agent_id)
        
        # Copy only the network weights, not the entire object (avoids optimizer sharing)
        second_agent.actor.load_state_dict(self.agents[first_agent_id].actor.state_dict())
        second_agent.actor_target.load_state_dict(self.agents[first_agent_id].actor_target.state_dict())
        second_agent.critic.load_state_dict(self.agents[first_agent_id].critic.state_dict())
        second_agent.critic_target.load_state_dict(self.agents[first_agent_id].critic_target.state_dict())
        
        self.agents[second_agent_id] = second_agent
        print(f"[SHARED] Created shared network for agent: {second_agent_id} with identical weights")
        
        print(f"[INIT] {first_agent_id}/{second_agent_id} initialized with IDENTICAL weights (separate optimizers)")

    def _build_single_agent(self, agent_idx: int, agent_id: str) -> DDPGAgent:
        """Build a single DDPG agent."""
        return DDPGAgent(
            agent_id=agent_id,
            state_dim=self.obs_dims[agent_idx],
            action_dim=self.action_dims[agent_idx],
            total_state_dim=self.total_obs_dim,
            total_action_dim=self.total_action_dim,
            params=self.params,
            device=self.device,
        )
    
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
        MODIFIED: Single point force constraint applied here only.
        
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
        
        # Get force constraints from configuration
        constraints = self.params.get('constraints', {})
        max_robot_force = constraints.get('max_robot_force', 0.04)
        max_human_force = constraints.get('max_human_force', 0.04)
        
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]
            obs_i = observations[agent_id]  # [num_envs, obs_dim]
            
            # Determine force limit based on agent type
            if 'robot' in agent_id.lower():
                max_force = max_robot_force
            else:
                max_force = max_human_force
            
            with torch.no_grad():
                mean, std = agent.actor(obs_i)
                noise = (noise_scale * std * torch.randn_like(mean)) if add_noise else torch.zeros_like(mean)
                # SINGLE POINT FORCE CONSTRAINT: Only here
                action = (mean + noise).clamp_(-max_force, max_force)
            
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
        CTDE update with asynchronous update frequencies.
        Critic updates every interval, Actor updates every 2*interval.
        """
        if len(self.replay) < self.min_buffer_size:
            return {}

        self.training_steps += 1
        
        # Determine what to update based on step count
        should_update_critic = (self.training_steps % self.update_interval == 0)
        should_update_actor = (self.training_steps % (self.update_interval * 2) == 0)
        
        if not (should_update_critic or should_update_actor):
            return {}

        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return {}

        obs_all, act_all, rew_all, nobs_all, done_any = batch
        gamma = float(self.params.get('maddpg_config', {}).get('gamma', 0.95))

        # Enhanced statistics structure with per-agent metrics
        stats = {
            "loss/actor": {}, "loss/critic": {}, "q_mean": {}, "q_std": {},
            "q_target_mean": {}, "q_target_std": {},
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

        # Update each agent with async frequencies
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]

            # CRITIC UPDATE (every interval steps)
            if should_update_critic:
                with torch.no_grad():
                    q_next = agent.critic_target(nobs_all, next_act_all).squeeze(-1)
                    y = rew_all[:, i] + (1.0 - done_any.squeeze(-1)) * gamma * q_next

                q = agent.critic(obs_all, act_all).squeeze(-1)
                critic_loss = torch.nn.functional.smooth_l1_loss(q, y)

                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                c_grad_norm = self._module_grad_norm(agent.critic)
                agent.critic_optimizer.step()

                # Store critic statistics
                stats["loss/critic"][agent_id] = float(critic_loss.detach().cpu().item())
                stats["q_mean"][agent_id] = float(q.detach().cpu().mean().item())
                stats["q_std"][agent_id] = float(q.detach().cpu().std().item())
                stats["q_target_mean"][agent_id] = float(y.detach().cpu().mean().item())
                stats["q_target_std"][agent_id] = float(y.detach().cpu().std().item())
                stats["grad_norm/critic"][agent_id] = float(c_grad_norm)

            # ACTOR UPDATE (every 2*interval steps)
            if should_update_actor:
                action_parts = []
                for j, agent_j in enumerate(self.agent_ids):
                    slice_j = self.obs_slices[j]
                    if j == i:
                        action_j, _ = self.agents[agent_j].actor(obs_all[:, slice_j])
                    else:
                        with torch.no_grad():
                            action_j, _ = self.agents[agent_j].actor(obs_all[:, slice_j])
                        action_j = action_j.detach()
                    action_parts.append(action_j)
                
                action_pred_all = torch.cat(action_parts, dim=-1)
                actor_loss = -agent.critic(obs_all, action_pred_all).mean()

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                a_grad_norm = self._module_grad_norm(agent.actor)
                agent.actor_optimizer.step()

                # Store actor statistics
                stats["loss/actor"][agent_id] = float(actor_loss.detach().cpu().item())
                stats["grad_norm/actor"][agent_id] = float(a_grad_norm)

            # Soft target network updates
            if should_update_critic or should_update_actor:
                agent.soft_update()

        # Update counters
        if should_update_critic:
            self.critic_update_count += 1
        if should_update_actor:
            self.actor_update_count += 1

        # Aggregate statistics
        for k in ["loss/critic", "loss/actor", "q_mean", "q_std", "q_target_mean", "q_target_std", "grad_norm/actor", "grad_norm/critic"]:
            if stats[k]:
                stats[f"{k}/avg"] = float(np.mean(list(stats[k].values())))

        # Include update statistics
        if should_update_critic or should_update_actor:
            stats["training/critic_updates"] = int(self.critic_update_count)
            stats["training/actor_updates"] = int(self.actor_update_count)
        
        return stats
    
