"""
Rollout buffer for Epigraph algorithm with z-encoding support.
Stores complete trajectories including RNN states for recurrent policies.
"""

import torch
from typing import Optional


class RolloutBufferZ:
    """
    Rollout buffer for Epigraph MARL with complete RNN support.
    
    Stores:
    - Observations, actions, log_probs
    - Task rewards, safety costs
    - Value predictions (Vl, Vh)
    - Z values
    - RNN states (actor, critic_vl, critic_vh)
    - Masks, term_masks, and override_bootstrap fields
    - Computed returns and advantages
    """
    
    def __init__(
        self,
        T: int,
        N: int,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        rnn_hidden_dim: int,
        device: torch.device
    ):
        """
        Initialize rollout buffer.
        
        Args:
            T: Rollout horizon
            N: Total batch size (num_envs * num_agents)
            obs_dim: Observation dimension per agent
            share_obs_dim: Shared observation dimension
            act_dim: Action dimension per agent
            rnn_hidden_dim: RNN hidden state dimension
            device: PyTorch device
        """
        self.T = T
        self.N = N
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device
        
        # Observations and actions
        self.obs = torch.zeros(T + 1, N, obs_dim, device=device)
        self.share_obs = torch.zeros(T + 1, N, share_obs_dim, device=device)
        self.actions = torch.zeros(T, N, act_dim, device=device)
        self.action_log_probs = torch.zeros(T, N, 1, device=device)
        
        # Rewards
        self.rewards = torch.zeros(T, N, 1, device=device)
        self.rewards_task = torch.zeros(T, N, 1, device=device)
        self.costs_safe = torch.zeros(T, N, 1, device=device)
        
        # Values
        self.values_vl = torch.zeros(T + 1, N, 1, device=device)
        self.values_vh = torch.zeros(T + 1, N, 1, device=device)
        
        # Z values
        self.z = torch.zeros(T + 1, N, 1, device=device)
        
        # Masks
        self.masks = torch.zeros(T, N, 1, device=device)
        self.term_masks = torch.zeros(T, N, 1, device=device)
        
        # Override bootstrap values
        self.override_bootstrap_mask_vl = torch.zeros(T, N, 1, dtype=torch.bool, device=device)
        self.override_bootstrap_vl = torch.zeros(T, N, 1, device=device)
        self.override_bootstrap_mask_vh = torch.zeros(T, N, 1, dtype=torch.bool, device=device)
        self.override_bootstrap_vh = torch.zeros(T, N, 1, device=device)
        
        # RNN states
        self.rnn_states_actor = torch.zeros(T, N, rnn_hidden_dim, device=device)
        self.rnn_states_critic = torch.zeros(T, N, rnn_hidden_dim, device=device)
        self.rnn_states_vh = torch.zeros(T, N, rnn_hidden_dim, device=device)
        
        # Computed by GAE
        self.returns_vl = torch.zeros(T, N, 1, device=device)
        self.returns_vh = torch.zeros(T, N, 1, device=device)
        self.advantages = torch.zeros(T, N, 1, device=device)
        
        # Current timestep pointer
        self.step = 0
    
    def insert(
        self,
        obs: torch.Tensor,
        share_obs: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        rewards_task: torch.Tensor,
        costs_safe: torch.Tensor,
        values_vl: torch.Tensor,
        values_vh: torch.Tensor,
        z: torch.Tensor,
        masks: torch.Tensor,
        term_masks: torch.Tensor,
        override_bootstrap_mask_vl: torch.Tensor,
        override_bootstrap_vl: torch.Tensor,
        override_bootstrap_mask_vh: torch.Tensor,
        override_bootstrap_vh: torch.Tensor,
        rnn_states_actor: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        rnn_states_vh: torch.Tensor,
    ):
        """
        Insert one timestep of data into buffer.
        
        All tensors should have shape [N, ...] where N = num_envs * num_agents.
        """
        t = self.step
        
        if t >= self.T:
            raise RuntimeError(f"Buffer overflow: trying to insert at step {t}, but T={self.T}")
        
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.actions[t].copy_(actions)
        self.action_log_probs[t].copy_(action_log_probs)
        
        self.rewards[t].copy_(rewards)
        self.rewards_task[t].copy_(rewards_task)
        self.costs_safe[t].copy_(costs_safe)
        
        self.values_vl[t].copy_(values_vl)
        self.values_vh[t].copy_(values_vh)
        
        self.z[t].copy_(z)
        
        self.masks[t].copy_(masks)
        self.term_masks[t].copy_(term_masks)
        
        self.override_bootstrap_mask_vl[t].copy_(override_bootstrap_mask_vl)
        self.override_bootstrap_vl[t].copy_(override_bootstrap_vl)
        self.override_bootstrap_mask_vh[t].copy_(override_bootstrap_mask_vh)
        self.override_bootstrap_vh[t].copy_(override_bootstrap_vh)
        
        self.rnn_states_actor[t].copy_(rnn_states_actor)
        self.rnn_states_critic[t].copy_(rnn_states_critic)
        self.rnn_states_vh[t].copy_(rnn_states_vh)
        
        self.step += 1
    
    def insert_final_step(
        self,
        obs: torch.Tensor,
        share_obs: torch.Tensor,
        values_vl: torch.Tensor,
        values_vh: torch.Tensor,
        z: torch.Tensor,
    ):
        """
        Insert final (T+1-th) step data for bootstrap.
        
        Only need obs, share_obs, values, and z for computing returns.
        """
        t = self.T
        
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.values_vl[t].copy_(values_vl)
        self.values_vh[t].copy_(values_vh)
        self.z[t].copy_(z)
    
    def compute_returns_and_advantages(
        self,
        Q_perf: torch.Tensor,
        Q_safe: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """
        Store computed returns and advantages from Epigraph GAE.
        
        Args:
            Q_perf: [T, N, 1] - Performance Q values
            Q_safe: [T, N, 1] - Safety Q values
            advantages: [T, N, 1] - Epigraph advantages
        """
        self.returns_vl = Q_perf
        self.returns_vh = Q_safe
        self.advantages = advantages
    
    def reset(self):
        """Reset buffer for next rollout."""
        self.step = 0
        
        self.obs.zero_()
        self.share_obs.zero_()
        self.actions.zero_()
        self.action_log_probs.zero_()
        
        self.rewards.zero_()
        self.rewards_task.zero_()
        self.costs_safe.zero_()
        
        self.values_vl.zero_()
        self.values_vh.zero_()
        
        self.z.zero_()
        
        self.masks.zero_()
        self.term_masks.zero_()
        
        self.override_bootstrap_mask_vl.zero_()
        self.override_bootstrap_vl.zero_()
        self.override_bootstrap_mask_vh.zero_()
        self.override_bootstrap_vh.zero_()
        
        self.rnn_states_actor.zero_()
        self.rnn_states_critic.zero_()
        self.rnn_states_vh.zero_()
        
        self.returns_vl.zero_()
        self.returns_vh.zero_()
        self.advantages.zero_()
    
    def get_all_data(self):
        """Get all stored data (for debugging)."""
        return {
            "obs": self.obs[:self.T],
            "share_obs": self.share_obs[:self.T],
            "actions": self.actions,
            "action_log_probs": self.action_log_probs,
            "rewards": self.rewards,
            "rewards_task": self.rewards_task,
            "costs_safe": self.costs_safe,
            "values_vl": self.values_vl[:self.T],
            "values_vh": self.values_vh[:self.T],
            "z": self.z[:self.T],
            "masks": self.masks,
            "term_masks": self.term_masks,
            "override_bootstrap_mask_vl": self.override_bootstrap_mask_vl,
            "override_bootstrap_vl": self.override_bootstrap_vl,
            "override_bootstrap_mask_vh": self.override_bootstrap_mask_vh,
            "override_bootstrap_vh": self.override_bootstrap_vh,
            "rnn_states_actor": self.rnn_states_actor,
            "rnn_states_critic": self.rnn_states_critic,
            "rnn_states_vh": self.rnn_states_vh,
            "returns_vl": self.returns_vl,
            "returns_vh": self.returns_vh,
            "advantages": self.advantages,
        }