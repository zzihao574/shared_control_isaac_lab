"""
Rollout buffer for Epigraph algorithm with z-encoding support.
Stores complete trajectories including RNN states for recurrent policies.
Enhanced with milestone truncation support.
"""

import torch
from typing import Optional

from .utils import compute_dec_efocp_gae_dp, normalize_advantages


class RolloutBufferZ:
    """
    Rollout buffer for Epigraph MARL with complete RNN support.
    
    Stores:
    - Observations, actions, log_probs
    - Task rewards, safety costs
    - Value predictions (Vl, Vh)
    - Z values
    - RNN states (actor, critic_vl, critic_vh)
    - Separate RNN, GAE-continuation, and bootstrap masks
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
        
        # State-reset, GAE-recursion, and bootstrap semantics are distinct.
        self.rnn_masks = torch.ones(T, N, 1, device=device)
        self.continuation_masks = torch.ones(T, N, 1, device=device)
        self.bootstrap_masks = torch.ones(T, N, 1, device=device)
        
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
        rnn_masks: torch.Tensor,
        continuation_masks: torch.Tensor,
        bootstrap_masks: torch.Tensor,
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
        
        self.rnn_masks[t].copy_(rnn_masks)
        self.continuation_masks[t].copy_(continuation_masks)
        self.bootstrap_masks[t].copy_(bootstrap_masks)
        
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
    
    def mark_milestone_truncation(
        self,
        t_last: Optional[int] = None,
        vl_bootstrap: Optional[torch.Tensor] = None,
        vh_bootstrap: Optional[torch.Tensor] = None,
    ):
        """
        Mark milestone truncation at the last valid step.
        
        This is the CRITICAL method for milestone-driven evaluation:
        - When a milestone is reached, we want to truncate the current rollout
          to run evaluation, but NOT treat it as a true episode termination.
        - This method sets override bootstrap masks and values so that GAE
          treats it as a "time truncation" with proper bootstrap values.
        
        Args:
            t_last: Last valid timestep index (default: self.step - 1)
            vl_bootstrap: [N, 1] Bootstrap value for Vl (default: use current values_vl[t_last+1])
            vh_bootstrap: [N, 1] Bootstrap value for Vh (default: use current values_vh[t_last+1])
        
        The rMAPPO Runner milestone trick:
        - Set override_bootstrap_mask = True at t_last
        - Set override_bootstrap_value = V(s_{t_last+1}, z_{t_last+1})
        - Keep bootstrap_masks[t_last] = 1 for the supplied next value
        - Set continuation_masks[t_last] = 0 to stop GAE recursion
        
        This allows clean evaluation without corrupting the next rollout's RNN states.
        """
        if t_last is None:
            t_last = self.step - 1
        
        if t_last < 0 or t_last >= self.T:
            raise ValueError(f"Invalid t_last={t_last}, must be in [0, {self.T-1}]")
        
        # Use provided bootstrap values or default to next-step predictions
        if vl_bootstrap is None:
            vl_bootstrap = self.values_vl[t_last + 1]  # [N, 1]
        
        if vh_bootstrap is None:
            vh_bootstrap = self.values_vh[t_last + 1]  # [N, 1]
        
        # Set override masks to True at t_last
        self.override_bootstrap_mask_vl[t_last, :, :] = True
        self.override_bootstrap_mask_vh[t_last, :, :] = True
        
        # Set override values
        self.override_bootstrap_vl[t_last].copy_(vl_bootstrap)
        self.override_bootstrap_vh[t_last].copy_(vh_bootstrap)
        
        self.continuation_masks[t_last, :, :] = 0.0
        self.bootstrap_masks[t_last, :, :] = 1.0
        
        print(f"[BUFFER] Marked milestone truncation at t={t_last}")
        print(f"[BUFFER] Override Vl: mean={vl_bootstrap.mean().item():.4f}")
        print(f"[BUFFER] Override Vh: mean={vh_bootstrap.mean().item():.4f}")
    
    def compute_epigraph_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        num_envs: int,
        num_agents: int,
    ):
        """
        Compute Epigraph returns and advantages using GAE.
        
        This method:
        1. Reshapes agent-major buffer data to [T, E, A] format
        2. Calls compute_dec_efocp_gae_dp from utils.py
        3. Normalizes advantages
        4. Stores results back to buffer
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            num_envs: Number of parallel environments (E)
            num_agents: Number of agents (A)
        """
        
        T = self.T
        E = num_envs
        A = num_agents
        N = E * A
        
        # ========== Reshape data from agent-major [T, N, 1] to [T, E, A] ==========
        # Buffer stores in agent-major order: [agent0_env0...agent0_envE-1, agent1_env0...]
        # We need [T, E, A] for compute_dec_efocp_gae_dp
        
        # Task rewards: average over agents to get team reward [T, E]
        # rewards_task shape: [T, N, 1] -> [T, A, E, 1] -> [T, E, A, 1]
        rewards_task_reshaped = self.rewards_task.view(T, A, E, 1).permute(0, 2, 1, 3)  # [T,E,A,1]
        team_task_reward = rewards_task_reshaped.mean(dim=2).squeeze(-1)  # [T,E]
        
        # Signed constraints h: [T, N, 1] -> [T, E, A]
        costs_safe_reshaped = self.costs_safe.view(T, A, E, 1).permute(0, 2, 1, 3).squeeze(-1)  # [T,E,A]
        
        # Z trajectory: [T, N, 1] -> [T, E] (take first agent's z since it's shared)
        z_traj = self.z[:T, :E, 0]  # [T, E]
        
        # Vl predictions: [T+1, N, 1] -> [T+1, E] (take first agent since Vl is shared)
        vl_preds = self.values_vl[:T+1, :E, 0]  # [T+1, E]
        
        # Vh predictions: [T+1, N, 1] -> [T+1, E, A]
        vh_preds = self.values_vh.view(T+1, A, E, 1).permute(0, 2, 1, 3).squeeze(-1)  # [T+1,E,A]
        
        continuation_masks = self.continuation_masks[:, :E, 0]
        bootstrap_masks = self.bootstrap_masks[:, :E, 0]
        
        # Override bootstrap: [T, N, 1] -> [T, E] and [T, E, A]
        ov_mask_vl = self.override_bootstrap_mask_vl[:, :E, 0]  # [T, E]
        ov_vl = self.override_bootstrap_vl[:, :E, 0]  # [T, E]
        ov_mask_vh = self.override_bootstrap_mask_vh.view(T, A, E, 1).permute(0, 2, 1, 3).squeeze(-1)  # [T,E,A]
        ov_vh = self.override_bootstrap_vh.view(T, A, E, 1).permute(0, 2, 1, 3).squeeze(-1)  # [T,E,A]
        
        # ========== Call Dec-EFOCP GAE ==========
        Q_perf, Q_safe, advantages = compute_dec_efocp_gae_dp(
            rewards=team_task_reward,     # [T, E]
            costs=costs_safe_reshaped,    # [T, E, A]
            z_traj=z_traj,                # [T, E]
            vl_preds=vl_preds,            # [T+1, E]
            vh_preds=vh_preds,            # [T+1, E, A]
            continuation_masks=continuation_masks,
            bootstrap_masks=bootstrap_masks,
            ov_mask_vl=ov_mask_vl,        # [T, E]
            ov_vl=ov_vl,                  # [T, E]
            ov_mask_vh=ov_mask_vh,        # [T, E, A]
            ov_vh=ov_vh,                  # [T, E, A]
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        # ========== Normalize Advantages ==========
        advantages_normalized = normalize_advantages(advantages)
        
        # ========== Reshape back to agent-major [T, N, 1] and store ==========
        # Q_perf: [T, E] -> [T, E, A] (broadcast to all agents) -> [T, A, E] -> [T, N, 1]
        Q_perf_broadcast = Q_perf.unsqueeze(-1).expand(T, E, A)  # [T, E, A]
        Q_perf_agentmajor = Q_perf_broadcast.permute(0, 2, 1).contiguous().view(T, N, 1)
        
        # Q_safe: [T, E, A] -> [T, A, E] -> [T, N, 1]
        Q_safe_agentmajor = Q_safe.permute(0, 2, 1).contiguous().view(T, N, 1)
        
        # advantages: [T, E, A] -> [T, A, E] -> [T, N, 1]
        advantages_agentmajor = advantages_normalized.permute(0, 2, 1).contiguous().view(T, N, 1)
        
        # Store to buffer
        self.returns_vl = Q_perf_agentmajor
        self.returns_vh = Q_safe_agentmajor
        self.advantages = advantages_agentmajor
        
        print(f"[BUFFER] Computed returns and advantages:")
        print(f"         Q_perf: mean={Q_perf.mean().item():.4f}, std={Q_perf.std().item():.4f}")
        print(f"         Q_safe: mean={Q_safe.mean().item():.4f}, std={Q_safe.std().item():.4f}")
        print(f"         Advantages: mean={advantages_normalized.mean().item():.4f}, std={advantages_normalized.std().item():.4f}")
    
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
        
        self.rnn_masks.fill_(1.0)
        self.continuation_masks.fill_(1.0)
        self.bootstrap_masks.fill_(1.0)
        
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
            "rnn_masks": self.rnn_masks,
            "continuation_masks": self.continuation_masks,
            "bootstrap_masks": self.bootstrap_masks,
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
