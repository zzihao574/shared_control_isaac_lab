"""
Multi-agent DDPG with shared networks, persistent generators, and eval mode support.
Features normalized action training, external noise scheduling, and reproducible noise generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from .ddpg_agent import DDPGAgent
from .replay_buffer import JointReplayBuffer

class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient with shared network architecture."""
    
    def __init__(self, num_envs: int, env, params: Dict[str, Any], device: str = 'cuda'):
        self.env = env
        self.actual_env = self._unwrap_environment(env)
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        
        # Eval mode control
        self._is_eval_mode = False
        
        # Get base seed for reproducible generator initialization
        self.base_seed = int(self.params.get("seed", 42))
        
        # Phase 1: Set deterministic seeds for network initialization
        self._set_network_seeds()
        
        # Get environment configuration
        self.agent_ids = list(self.actual_env.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        # Get dimensions from environment cfg
        self.obs_dims = [self.actual_env.cfg.observation_spaces[agent] for agent in self.agent_ids]
        self.action_dims = [self.actual_env.cfg.action_spaces[agent] for agent in self.agent_ids]
        self.total_obs_dim = sum(self.obs_dims)
        self.total_action_dim = sum(self.action_dims)

        print(f"[MADDPG] Shared Network Architecture with Persistent Generator Strategy:")
        print(f"  Environments: {self.num_envs}")
        print(f"  Agent IDs: {self.agent_ids}")
        print(f"  Obs dims: {self.obs_dims} (total: {self.total_obs_dim})")
        print(f"  Action dims: {self.action_dims} (total: {self.total_action_dim})")
        print(f"  Networks: ONE per agent type (shared across ALL environments)")

        # Initialize shared agents with IDENTICAL weights
        self._initialize_agents_with_identical_init()
        self._initialize_replay_buffers()
        self._build_slices()
        
        # Phase 2: Initialize persistent noise generators for reproducible yet diverse noise
        self._init_noise_generators()
        
        # Load training hyperparameters
        maddpg_cfg = self.params.get('maddpg_config', {})
        self.batch_size = int(maddpg_cfg.get('batch_size', 512))
        self.gamma = float(maddpg_cfg.get('gamma', 0.95))
        self.update_interval = int(maddpg_cfg.get('update_interval', 100))
        self.min_buffer_size = int(maddpg_cfg.get('min_buffer_size', 4096))
        
        self.training_steps = 0
        self.critic_update_count = 0
        self.actor_update_count = 0
        
        # Flag to track if training has started
        self._training_started = False
        
        print(f"[MADDPG] Shared network initialization complete")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Critic update interval: {self.update_interval}")
        print(f"  Actor update interval: {self.update_interval * 2}")
        print(f"  Min buffer size: {self.min_buffer_size}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode - disables buffer/network updates and noise during evaluation."""
        self._is_eval_mode = bool(is_eval)
        print(f"[MADDPG] Eval mode: {'ENABLED' if self._is_eval_mode else 'DISABLED'}")
    
    def _set_network_seeds(self) -> None:
        """Set deterministic seeds for network initialization."""
        import random
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)  
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)
    
    def _init_noise_generators(self) -> None:
        """Initialize persistent noise generators for each (agent, env) pair."""
        self.noise_generators = {}  # {agent_id: [gen_env0, gen_env1, ...]}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i, agent_id in enumerate(self.agent_ids):
            gens = []
            for env_id in range(self.num_envs):
                gen = torch.Generator(device=device)
                # Reproducible but unique seed for each (agent, env) pair
                seed_ae = (self.base_seed + 1234567 + i * 10007 + env_id * 97) % (2**32)
                gen.manual_seed(seed_ae)
                gens.append(gen)
            self.noise_generators[agent_id] = gens
        
        # Dedicated generator for replay buffer sampling
        self.replay_generator = torch.Generator(device='cpu')
        self.replay_generator.manual_seed((self.base_seed + 424242) % (2**32))
        
        print(f"[GENERATORS] Initialized {self.num_agents * self.num_envs} noise generators + 1 replay generator")
    
    def _unwrap_environment(self, env):
        """Get actual environment object."""
        return getattr(env, 'unwrapped', env)
        
    def _initialize_agents_with_identical_init(self) -> None:
        """Initialize agents with identical network weights but separate optimizers."""
        self.agents = {}
        
        # Create first agent with seeded initialization
        first_agent_id = self.agent_ids[0]
        self.agents[first_agent_id] = self._build_single_agent(0, first_agent_id)
        print(f"[SHARED] Created shared network for agent: {first_agent_id}")
        
        # Create second agent with separate optimizers
        second_agent_id = self.agent_ids[1]
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
        self.obs_slices = []
        self.act_slices = []
        
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
        """Generate diverse noise using persistent per-(agent,env) generators."""
        actions = {}
        detail = {"mean_actions": {}, "noise_actions": {}}
        
        # Force disable noise during evaluation
        effective_add_noise = add_noise and (not self._is_eval_mode)
        
        # Get force constraint configuration
        constraints = self.params.get('constraints', {})
        max_robot_force = float(constraints.get('max_robot_force', 0.04))
        max_human_force = float(constraints.get('max_human_force', 0.04))
        
        for i, agent_id in enumerate(self.agent_ids):
            obs_i = observations[agent_id]
            
            # Actor outputs normalized actions [-1,1]
            a_norm = self.agents[agent_id].actor(obs_i)
            
            # Generate noise using persistent generators (no global seed changes)
            if effective_add_noise:
                noise_norm = torch.zeros_like(a_norm)  # [num_envs, action_dim]
                
                # Each environment uses its dedicated generator
                for env_id in range(self.num_envs):
                    gen = self.noise_generators[agent_id][env_id]
                    noise_norm[env_id] = noise_scale * torch.randn(a_norm.shape[1], device=a_norm.device, generator=gen)
                    
            else:
                noise_norm = torch.zeros_like(a_norm)
                
            # Normalized domain clamp
            a_norm_with_noise = (a_norm + noise_norm).clamp_(-1.0, 1.0)
            
            # Determine force limit based on agent type
            max_force = max_robot_force if 'robot' in agent_id.lower() else max_human_force
            
            # Map to physical units for environment
            action = (a_norm_with_noise * max_force).clamp_(-max_force, max_force)
            actions[agent_id] = action
            
            # Store debug information
            detail["mean_actions"][agent_id] = (a_norm * max_force).clamp_(-max_force, max_force)
            detail["noise_actions"][agent_id] = action - detail["mean_actions"][agent_id]

        for agent_id in actions.keys():
            actions[agent_id] = actions[agent_id].detach()
            detail["mean_actions"][agent_id] = detail["mean_actions"][agent_id].detach()
            detail["noise_actions"][agent_id] = detail["noise_actions"][agent_id].detach()
                
        return actions, detail

    def add_experience_to_buffer(self, obs, actions, rewards, next_obs, dones):
        """Store transitions in joint replay buffer. Skip during evaluation mode."""
        # Skip buffer updates during evaluation
        if self._is_eval_mode:
            return
            
        for env_id in range(self.num_envs):
            # Concatenate observations and actions (by agent_ids order)
            obs_all = torch.cat([obs[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            act_all = torch.cat([actions[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            rew_vec = torch.stack([rewards[aid][env_id].reshape(()).float() for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            nobs_all = torch.cat([next_obs[aid][env_id].reshape(-1) for aid in self.agent_ids], dim=0).detach().cpu().numpy()
            
            # Calculate done_any (logical OR across agents)
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
        CTDE update with async frequency. Critic updates every interval steps, 
        Actor updates every 2*interval steps. Skip all updates during evaluation mode.
        """
        # Skip all updates during evaluation
        if self._is_eval_mode:
            return {"actor_updates": 0, "critic_updates": 0}
            
        if len(self.replay) < self.min_buffer_size:
            return {}

        # Check if this is the first time we're starting actual training
        if not self._training_started:
            print("=" * 80)
            print("🚀 NEURAL NETWORK TRAINING STARTED 🚀")
            print(f"Buffer size reached: {len(self.replay)} >= {self.min_buffer_size}")
            print("=" * 80)
            self._training_started = True

        self.training_steps += 1
        
        # Determine what to update based on step count
        should_update_critic = (self.training_steps % self.update_interval == 0)
        should_update_actor = (self.training_steps % (self.update_interval * 2) == 0)
        
        if not (should_update_critic or should_update_actor):
            return {}

        # Use dedicated generator for reproducible replay sampling
        batch = self.replay.sample(self.batch_size, generator=self.replay_generator)
        if batch is None:
            return {}

        obs_all, act_all, rew_all, nobs_all, done_any = batch
        gamma = float(self.params.get('maddpg_config', {}).get('gamma', 0.95))

        # Dynamically build per-dim action limits
        constraints = self.params.get('constraints', {})
        a_max_list = []
        for agent_id in self.agent_ids:
            if 'human' in agent_id.lower():
                max_force = float(constraints.get('max_human_force', 0.04))
            else:
                max_force = float(constraints.get('max_robot_force', 0.04))
            a_max_list.extend([max_force] * 3)  # 3D force
        a_max = torch.tensor(a_max_list, device=self.device).view(1, -1)

        # Enhanced statistics structure with per-agent metrics
        stats = {
            "loss/actor": {}, "loss/critic": {}, "q_mean": {}, "q_std": {},
            "q_target_mean": {}, "q_target_std": {},
            "grad_norm/actor": {}, "grad_norm/critic": {}
        }

        # Calculate target actions (use target Actor, outputs normalized actions)
        next_action_parts = []
        for i, agent_id in enumerate(self.agent_ids):
            slice_i = self.obs_slices[i]
            with torch.no_grad():
                a2_norm_i = self.agents[agent_id].actor_target(nobs_all[:, slice_i])
            next_action_parts.append(a2_norm_i)
        next_act_all_norm = torch.cat(next_action_parts, dim=-1)

        # Update each agent with async frequency
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]

            # CRITIC UPDATE (every interval steps)
            if should_update_critic:
                with torch.no_grad():
                    q_next = agent.critic_target(nobs_all, next_act_all_norm).squeeze(-1)
                    y = rew_all[:, i] + (1.0 - done_any.squeeze(-1)) * gamma * q_next

                # Current Q: Convert Buffer's physical actions to normalized
                act_all_norm = act_all / a_max
                q = agent.critic(obs_all, act_all_norm).squeeze(-1)
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
                        a_norm_j = self.agents[agent_j].actor(obs_all[:, slice_j])
                    else:
                        with torch.no_grad():
                            a_norm_j = self.agents[agent_j].actor(obs_all[:, slice_j])
                    action_parts.append(a_norm_j)
                
                action_pred_all_norm = torch.cat(action_parts, dim=-1)
                actor_loss = -agent.critic(obs_all, action_pred_all_norm).mean()

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                a_grad_norm = self._module_grad_norm(agent.actor)
                agent.actor_optimizer.step()

                # Store actor statistics
                stats["loss/actor"][agent_id] = float(actor_loss.detach().cpu().item())
                stats["grad_norm/actor"][agent_id] = float(a_grad_norm)

            # Soft target network update
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