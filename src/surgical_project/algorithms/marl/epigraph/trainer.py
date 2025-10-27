"""
Epigraph Trainer for Safe Multi-Agent Reinforcement Learning.
With sequence-based RNN training (rMAPPO-aligned).
Final corrected version with:
1. No gamma in z dynamics
2. Agent-major reshape logic
3. z_encoder gradient isolation
4. z_encoder config from YAML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from .epigraph_core import ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder
from .rollout_buffer_z import RolloutBufferZ
from .utils import compute_epigraph_gae, normalize_advantages


def init_z_global(num_envs: int, z_min: float, z_max: float, device, mode: str = "mixed", p_extreme: float = 0.3):
    """Initialize z values for training."""
    if mode == "uniform":
        z = torch.rand(num_envs, 1, device=device) * (z_max - z_min) + z_min
    elif mode == "extreme":
        z = torch.empty(num_envs, 1, device=device)
        z[: num_envs // 2] = z_min
        z[num_envs // 2 :] = z_max
    elif mode == "mixed":
        z = torch.rand(num_envs, 1, device=device) * (z_max - z_min) + z_min
        extreme_mask = torch.rand(num_envs, device=device) < p_extreme
        z[extreme_mask & (torch.rand(num_envs, device=device) < 0.5), 0] = z_min
        z[extreme_mask & (torch.rand(num_envs, device=device) >= 0.5), 0] = z_max
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    return z


def update_z_epigraph(z: torch.Tensor, reward: torch.Tensor, z_min: float, z_max: float):
    """
    Epigraph z dynamics (paper-style).
    z_{t+1} = z_t + r_team(t)
    No gamma here.
    """
    z_next = z + reward
    return torch.clamp(z_next, z_min, z_max)


class EpigraphTrainer:
    """
    Epigraph MARL trainer with sequence-based RNN training.
    """
    
    def __init__(self, env, device: torch.device, algo_cfg: Dict[str, Any], epi_cfg: Dict[str, Any]):
        """Initialize Epigraph trainer."""
        self.env = env
        self.device = device
        self.algo_cfg = algo_cfg
        self.epi_cfg = epi_cfg
        
        if hasattr(env.unwrapped, "params"):
            self.env_cfg = env.unwrapped.params.get("epigraph_env", {})
        else:
            self.env_cfg = {}
        
        self.num_envs = env.num_envs
        self.agent_ids = list(env.unwrapped.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        self.obs_dim = env.unwrapped.cfg.observation_spaces[self.agent_ids[0]]
        self.share_obs_dim = self.obs_dim * self.num_agents
        self.act_dim = env.unwrapped.cfg.action_spaces[self.agent_ids[0]]
        self.hidden_size = algo_cfg["hidden_size"]
        self.recurrent_N = algo_cfg.get("recurrent_N", 1)
        
        self.rollout_horizon = algo_cfg["rollout_horizon"]
        self.gamma = algo_cfg["gamma"]
        self.gae_lambda = algo_cfg["gae_lambda"]
        self.ppo_epoch = algo_cfg["ppo_epoch"]
        self.num_mini_batch = algo_cfg["num_mini_batch"]
        self.data_chunk_length = algo_cfg["data_chunk_length"]
        self.clip_param = algo_cfg["clip_param"]
        self.value_clip_param = algo_cfg.get("value_clip_param", self.clip_param)
        self.entropy_coef = algo_cfg["entropy_coef"]
        self.max_grad_norm_actor = algo_cfg["max_grad_norm_actor"]
        self.max_grad_norm_critic = algo_cfg["max_grad_norm_critic"]
        
        self.z_min = epi_cfg["z"]["min"]
        self.z_max = epi_cfg["z"]["max"]
        self.z_nz = epi_cfg["z"]["encode"]["nz"]
        self.z_init_mode = epi_cfg["z"]["init"]["mode"]
        self.z_init_p_extreme = epi_cfg["z"]["init"]["p_extreme"]
        self.lambda_safe = epi_cfg["losses"]["lambda_safe"]
        
        self._build_networks()
        self._build_buffer()
        self._build_optimizers()
        self._init_rnn_states()
        
        self.global_step = 0
        self.episodes_done = 0
        
    def _build_networks(self):
        """Build all networks."""
        # FIX 4: Read z_encoder params from config
        self.z_encoder = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
        ).to(self.device)
        
        self.actors = {}
        self.critics_vh = {}
        for agent in self.agent_ids:
            self.actors[agent] = ActorRNN(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                nz=self.z_nz,
                recurrent_N=self.recurrent_N,
            ).to(self.device)
            
            self.critics_vh[agent] = CriticVhRNN(
                obs_dim=self.obs_dim,
                hidden_size=self.hidden_size,
                nz=self.z_nz,
                recurrent_N=self.recurrent_N,
            ).to(self.device)
        
        self.critic_vl = CriticVlRNN(
            share_obs_dim=self.share_obs_dim,
            hidden_size=self.hidden_size,
            nz=self.z_nz,
            recurrent_N=self.recurrent_N,
        ).to(self.device)
        
        self.root_finder = RootFinder(
            max_iter=self.epi_cfg["root_finder"]["max_iter"],
            tol=self.epi_cfg["root_finder"]["tol"],
            z_min=self.z_min,
            z_max=self.z_max,
        )
        
    def _build_buffer(self):
        """Build rollout buffer."""
        self.buffer = RolloutBufferZ(
            T=self.rollout_horizon,
            N=self.num_envs * self.num_agents,
            obs_dim=self.obs_dim,
            share_obs_dim=self.share_obs_dim,
            act_dim=self.act_dim,
            rnn_hidden_dim=self.hidden_size,
            device=self.device,
        )
        
    def _build_optimizers(self):
        """Build optimizers."""
        actor_params = list(self.z_encoder.parameters())
        for agent in self.agent_ids:
            actor_params.extend(list(self.actors[agent].parameters()))
        
        self.optimizer_actor = optim.Adam(
            actor_params, lr=self.algo_cfg["actor_lr"], eps=self.algo_cfg["opt_eps"]
        )
        
        self.optimizer_vl = optim.Adam(
            self.critic_vl.parameters(), lr=self.algo_cfg["critic_lr"], eps=self.algo_cfg["opt_eps"]
        )
        
        self.optimizers_vh = {}
        for agent in self.agent_ids:
            self.optimizers_vh[agent] = optim.Adam(
                self.critics_vh[agent].parameters(),
                lr=self.algo_cfg["critic_lr"],
                eps=self.algo_cfg["opt_eps"]
            )
        
    def _init_rnn_states(self):
        """Initialize RNN hidden states."""
        self.rnn_states = {}
        for agent in self.agent_ids:
            self.rnn_states[agent] = {
                "actor": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
                "vh": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
            }
        self.rnn_states_vl = torch.zeros(self.num_envs, self.hidden_size, device=self.device)
        
    def _flatten_per_agent(self, per_agent_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten per-agent dict to [num_envs*num_agents, ...] in agent-major order."""
        return torch.cat([per_agent_dict[agent] for agent in self.agent_ids], dim=0)
    
    def _replicate_global_for_agents(self, global_tensor: torch.Tensor) -> torch.Tensor:
        """Replicate global tensor for all agents in agent-major order."""
        return global_tensor.repeat(self.num_agents, 1)
    
    def _init_z_training(self) -> torch.Tensor:
        """Initialize z for training rollout."""
        return init_z_global(
            self.num_envs, self.z_min, self.z_max, self.device,
            mode=self.z_init_mode, p_extreme=self.z_init_p_extreme
        )
    
    def _get_share_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate all agent observations."""
        return torch.cat([obs_dict[agent] for agent in self.agent_ids], dim=-1)
    
    def _reshape_agentmajor_to_env_agent_T(self, x, has_trailing_dim1: bool):
        """
        Reshape from agent-major buffer format to [T,E,A] or [T,E,A,1].
        
        x shape:
          if has_trailing_dim1=True : [T, N, 1]
          else                      : [T, N]
        We stored N = A * E in agent-major order: [agent0_env0...agent0_envE-1, agent1_env0...]
        We want to get [T, E, A] (or [T,E,A,1]) for team/value computations.
        """
        T = self.rollout_horizon
        E = self.num_envs
        A = self.num_agents
        if has_trailing_dim1:
            # [T, N, 1] -> [T, A, E, 1] -> [T, E, A, 1]
            x = x.view(T, A, E, 1)
            x = x.permute(0, 2, 1, 3)  # T,E,A,1
        else:
            # [T, N] -> [T, A, E] -> [T, E, A]
            x = x.view(T, A, E)
            x = x.permute(0, 2, 1)     # T,E,A
        return x

    def _reshape_agentmajor_to_env_agent_Tplus1(self, x, has_trailing_dim1: bool):
        """
        Same as above but x shape is [T+1, N, ...] -> [T+1, E, A, ...]
        """
        T = self.rollout_horizon
        E = self.num_envs
        A = self.num_agents
        if has_trailing_dim1:
            x = x.view(T+1, A, E, 1)
            x = x.permute(0, 2, 1, 3)  # T+1,E,A,1
        else:
            x = x.view(T+1, A, E)
            x = x.permute(0, 2, 1)     # T+1,E,A
        return x
    
    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, Any]:
        """
        Collect one rollout with dynamic masks and proper bootstrap logic.
        """
        self.set_eval_mode()
        
        obs, _ = self.env.reset()
        z_global = self._init_z_training()
        self._init_rnn_states()
        
        masks = torch.ones(self.num_envs, 1, device=self.device)
        
        episode_returns_task = []
        episode_returns_safe = []
        episode_lengths = []
        current_returns_task = torch.zeros(self.num_envs, device=self.device)
        current_returns_safe = torch.zeros(self.num_envs, device=self.device)
        current_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        for t in range(self.rollout_horizon):
            z_enc = self.z_encoder(z_global)
            
            actions = {}
            action_log_probs = {}
            for agent in self.agent_ids:
                act, logp, next_h_actor, _ = self.actors[agent].act_step(
                    obs[agent], z_enc, self.rnn_states[agent]["actor"], masks, deterministic=False
                )
                actions[agent] = act
                action_log_probs[agent] = logp
                self.rnn_states[agent]["actor"] = next_h_actor
            
            share_obs = self._get_share_obs(obs)
            vl, next_h_vl = self.critic_vl.value_step(share_obs, z_enc, self.rnn_states_vl, masks)
            self.rnn_states_vl = next_h_vl
            
            vh = {}
            for agent in self.agent_ids:
                vh_val, next_h_vh = self.critics_vh[agent].value_step(
                    obs[agent], z_enc, self.rnn_states[agent]["vh"], masks
                )
                vh[agent] = vh_val
                self.rnn_states[agent]["vh"] = next_h_vh
            
            obs_next, rewards, terminated, truncated, info = self.env.step(actions)
            
            r_task = info["r_task"]
            r_safe = info["r_safe"]
            r_team = sum(r_task[a] for a in self.agent_ids) / self.num_agents
            
            term_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            trunc_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            done_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            
            for agent in self.agent_ids:
                term_a = terminated[agent] if terminated[agent].dim() == 2 else terminated[agent].unsqueeze(-1)
                trunc_a = truncated[agent] if truncated[agent].dim() == 2 else truncated[agent].unsqueeze(-1)
                term_any_env |= term_a
                trunc_any_env |= trunc_a
                done_any_env |= (term_a | trunc_a)
            
            masks_flat_t = masks.repeat(self.num_agents, 1)
            term_mask_env = (~term_any_env).float()
            term_masks_flat_t = term_mask_env.repeat(self.num_agents, 1)
            
            (
                override_mask_vl_flat_t,
                override_vl_flat_t,
                override_mask_vh_flat_t,
                override_vh_flat_t,
            ) = self._maybe_override_bootstrap(
                obs_next=obs_next,
                z_next=z_global,
                rnn_states_snapshot={
                    "vl": self.rnn_states_vl.clone(),
                    "vh": {aid: self.rnn_states[aid]["vh"].clone() for aid in self.agent_ids},
                },
                allow_env_mask=masks.clone(),
                term_any_env=term_any_env,
                trunc_any_env=trunc_any_env,
            )
            
            self.buffer.insert(
                obs=self._flatten_per_agent(obs),
                share_obs=self._replicate_global_for_agents(share_obs),
                actions=self._flatten_per_agent(actions),
                action_log_probs=self._flatten_per_agent(action_log_probs),
                rewards=self._flatten_per_agent(rewards),
                rewards_task=self._flatten_per_agent(r_task),
                costs_safe=self._flatten_per_agent(r_safe),
                values_vl=self._replicate_global_for_agents(vl),
                values_vh=self._flatten_per_agent(vh),
                z=self._replicate_global_for_agents(z_global),
                masks=masks_flat_t,
                term_masks=term_masks_flat_t,
                override_bootstrap_mask_vl=override_mask_vl_flat_t,
                override_bootstrap_vl=override_vl_flat_t,
                override_bootstrap_mask_vh=override_mask_vh_flat_t,
                override_bootstrap_vh=override_vh_flat_t,
                rnn_states_actor=torch.cat([self.rnn_states[a]["actor"] for a in self.agent_ids], dim=0),
                rnn_states_critic=self.rnn_states_vl.repeat(self.num_agents, 1),
                rnn_states_vh=torch.cat([self.rnn_states[a]["vh"] for a in self.agent_ids], dim=0),
            )
            
            # FIX 1: No gamma in z dynamics
            z_global = update_z_epigraph(z_global, r_team, self.z_min, self.z_max)
            
            r_task_mean = sum(r_task[a].squeeze(-1) for a in self.agent_ids) / self.num_agents
            r_safe_mean = sum(r_safe[a].squeeze(-1) for a in self.agent_ids) / self.num_agents
            current_returns_task += r_task_mean
            current_returns_safe += r_safe_mean
            current_lengths += 1
            
            if done_any_env.any():
                done_indices = done_any_env.squeeze(-1).nonzero(as_tuple=True)[0]
                episode_returns_task.extend(current_returns_task[done_indices].cpu().tolist())
                episode_returns_safe.extend(current_returns_safe[done_indices].cpu().tolist())
                episode_lengths.extend(current_lengths[done_indices].cpu().tolist())
                current_returns_task[done_indices] = 0.0
                current_returns_safe[done_indices] = 0.0
                current_lengths[done_indices] = 0
                self.episodes_done += len(done_indices)
            
            obs = obs_next
            masks = (~done_any_env).float()
            self.global_step += self.num_envs
        
        z_enc_last = self.z_encoder(z_global)
        share_obs_last = self._get_share_obs(obs)
        masks_last = torch.ones(self.num_envs, 1, device=self.device)
        
        vl_last, _ = self.critic_vl.value_step(share_obs_last, z_enc_last, self.rnn_states_vl, masks_last)
        vh_last = {}
        for agent in self.agent_ids:
            vh_last[agent], _ = self.critics_vh[agent].value_step(
                obs[agent], z_enc_last, self.rnn_states[agent]["vh"], masks_last
            )
        
        self.buffer.insert_final_step(
            obs=self._flatten_per_agent(obs),
            share_obs=self._replicate_global_for_agents(share_obs_last),
            values_vl=self._replicate_global_for_agents(vl_last),
            values_vh=self._flatten_per_agent(vh_last),
            z=self._replicate_global_for_agents(z_global)
        )
        
        rollout_info = {
            "return_task_mean": np.mean(episode_returns_task) if episode_returns_task else 0.0,
            "return_task_std": np.std(episode_returns_task) if episode_returns_task else 0.0,
            "return_safe_mean": np.mean(episode_returns_safe) if episode_returns_safe else 0.0,
            "return_safe_std": np.std(episode_returns_safe) if episode_returns_safe else 0.0,
            "episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "episodes_done": len(episode_returns_task),
            "z_mean": float(self.buffer.z[:self.rollout_horizon].mean().item()),
            "z_std": float(self.buffer.z[:self.rollout_horizon].std().item()),
            "z_min": float(self.buffer.z[:self.rollout_horizon].min().item()),
            "z_max": float(self.buffer.z[:self.rollout_horizon].max().item()),
        }
        
        return rollout_info
    
    def _maybe_override_bootstrap(
        self,
        obs_next,
        z_next,
        rnn_states_snapshot,
        allow_env_mask,
        term_any_env,
        trunc_any_env,
    ):
        """
        Placeholder for override bootstrap mechanism.
        """
        E = self.num_envs
        A = self.num_agents
        device = self.device
        
        override_mask_vl_env = torch.zeros(E, 1, dtype=torch.bool, device=device)
        override_vl_env = torch.zeros(E, 1, device=device)
        
        override_mask_vh_env = torch.zeros(E, A, dtype=torch.bool, device=device)
        override_vh_env = torch.zeros(E, A, device=device)
        
        override_mask_vl_flat = override_mask_vl_env.repeat(A, 1)
        override_vl_flat = override_vl_env.repeat(A, 1)
        
        override_mask_vh_flat = override_mask_vh_env.reshape(E * A, 1)
        override_vh_flat = override_vh_env.reshape(E * A, 1)
        
        return (
            override_mask_vl_flat,
            override_vl_flat,
            override_mask_vh_flat,
            override_vh_flat,
        )
    
    def update(self):
        """
        Sequence-based PPO update with RNN training.
        With proper agent-major reshape and z_encoder gradient isolation.
        """
        self.set_train_mode()
        
        T = self.rollout_horizon
        E = self.num_envs
        A = self.num_agents
        N = E * A
        
        # FIX 2: Use proper agent-major reshape
        rewards_task_buf = self.buffer.rewards_task[:T]
        costs_safe_buf = self.buffer.costs_safe[:T]
        z_buf = self.buffer.z[:T]
        vl_buf = self.buffer.values_vl[:T+1]
        vh_buf = self.buffer.values_vh[:T+1]
        masks_buf = self.buffer.masks[:T]
        term_masks_buf = self.buffer.term_masks[:T]
        ov_mask_vl_buf = self.buffer.override_bootstrap_mask_vl[:T]
        ov_vl_buf = self.buffer.override_bootstrap_vl[:T]
        ov_mask_vh_buf = self.buffer.override_bootstrap_mask_vh[:T]
        ov_vh_buf = self.buffer.override_bootstrap_vh[:T]
        
        # Reshape from agent-major [T,N,1] to [T,E,A]
        rewards_team = self._reshape_agentmajor_to_env_agent_T(rewards_task_buf, has_trailing_dim1=True).mean(dim=2).squeeze(-1)  # [T,E]
        costs_safe = self._reshape_agentmajor_to_env_agent_T(costs_safe_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E,A]
        z_traj = self._reshape_agentmajor_to_env_agent_T(z_buf, has_trailing_dim1=True).mean(dim=2).squeeze(-1)  # [T,E]
        vl_preds = self._reshape_agentmajor_to_env_agent_Tplus1(vl_buf, has_trailing_dim1=True).mean(dim=2).squeeze(-1)  # [T+1,E]
        vh_preds = self._reshape_agentmajor_to_env_agent_Tplus1(vh_buf, has_trailing_dim1=True).squeeze(-1)  # [T+1,E,A]
        masks = self._reshape_agentmajor_to_env_agent_T(masks_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E]
        term_masks = self._reshape_agentmajor_to_env_agent_T(term_masks_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E]
        
        # Vl overrides are team-level, need to average agent dim
        ov_mask_vl = self._reshape_agentmajor_to_env_agent_T(ov_mask_vl_buf, has_trailing_dim1=True).squeeze(-1).float()  # [T,E,A]
        ov_mask_vl = ov_mask_vl.mean(dim=2)  # [T,E]
        ov_vl = self._reshape_agentmajor_to_env_agent_T(ov_vl_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E,A]
        ov_vl = ov_vl.mean(dim=2)  # [T,E]
        
        # Vh overrides are per-agent
        ov_mask_vh = self._reshape_agentmajor_to_env_agent_T(ov_mask_vh_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E,A]
        ov_vh = self._reshape_agentmajor_to_env_agent_T(ov_vh_buf, has_trailing_dim1=True).squeeze(-1)  # [T,E,A]
        
        Q_perf, Q_safe, advantages = compute_epigraph_gae(
            rewards=rewards_team,
            costs=costs_safe,
            z_traj=z_traj,
            vl_preds=vl_preds,
            vh_preds=vh_preds,
            masks=masks,
            term_masks=term_masks,
            ov_mask_vl=ov_mask_vl,
            ov_vl=ov_vl,
            ov_mask_vh=ov_mask_vh,
            ov_vh=ov_vh,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        # Convert back to agent-major [T,N,1]
        # Q_perf [T,E] -> [T,E,A] -> agent-major -> [T,N,1]
        Q_perf_expanded = Q_perf.unsqueeze(-1).expand(T, E, A)  # [T,E,A]
        Q_perf_agentmajor = Q_perf_expanded.permute(0, 2, 1).reshape(T, N, 1)
        
        # Q_safe [T,E,A] -> agent-major
        Q_safe_agentmajor = Q_safe.permute(0, 2, 1).reshape(T, N, 1)
        
        # advantages [T,E,A] -> agent-major
        adv_agentmajor = advantages.permute(0, 2, 1).reshape(T, N, 1)
        
        # Normalize advantages
        adv_agentmajor = normalize_advantages(adv_agentmajor, masks_buf, eps=1e-8)
        
        # Store back
        self.buffer.returns_vl = Q_perf_agentmajor
        self.buffer.returns_vh = Q_safe_agentmajor
        self.buffer.advantages = adv_agentmajor
        
        # Prepare data for sequence training
        update_info = {
            "loss_policy": 0.0,
            "loss_value_vl": 0.0,
            "loss_value_vh": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0
        }
        
        num_updates = 0
        clip_eps = self.clip_param
        value_clip_eps = self.value_clip_param
        ent_coef = self.entropy_coef
        L = self.data_chunk_length
        
        for epoch in range(self.ppo_epoch):
            generator = self._recurrent_generator(L)
            
            for mb in generator:
                # Extract mini-batch data
                obs_flat = mb["obs_flat"]
                share_obs_flat = mb["share_obs_flat"]
                act_flat = mb["act_flat"]
                old_logp_flat = mb["old_logp_flat"]
                adv_flat = mb["adv_flat"]
                ret_vl_flat = mb["ret_vl_flat"]
                ret_vh_flat = mb["ret_vh_flat"]
                old_vl_flat = mb["old_vl_flat"]
                old_vh_flat = mb["old_vh_flat"]
                z_flat = mb["z_flat"]
                masks_flat = mb["masks_flat"]
                
                h0_actor = mb["h0_actor"]
                h0_vl = mb["h0_vl"]
                h0_vh = mb["h0_vh"]
                agent_idx = mb["agent_idx"]
                
                # FIX 3: z_encoder gradient isolation
                z_enc_flat = self.z_encoder(z_flat)
                z_enc_flat_detached = z_enc_flat.detach()  # For critics only
                
                # Critic Vl (centralized) - uses detached z_enc
                vl_pred_flat, _ = self.critic_vl.value_seq(
                    share_obs_seq=share_obs_flat,
                    z_enc_seq=z_enc_flat_detached,
                    hxs_init=h0_vl,
                    masks_seq=masks_flat,
                )
                
                vl_clipped = old_vl_flat + torch.clamp(
                    vl_pred_flat - old_vl_flat, -value_clip_eps, value_clip_eps
                )
                vl_loss_unclipped = (vl_pred_flat - ret_vl_flat).pow(2)
                vl_loss_clipped = (vl_clipped - ret_vl_flat).pow(2)
                vl_loss = 0.5 * torch.max(vl_loss_unclipped, vl_loss_clipped).mean()
                
                # Per-agent critics and policies
                policy_loss_total = 0.0
                entropy_total = 0.0
                vh_loss_total = 0.0
                approx_kl_total = 0.0
                clipfrac_total = 0.0
                count_total = 0.0
                
                B = h0_actor.size(0)
                
                for a_i, agent in enumerate(self.agent_ids):
                    mask_agent_seq = (agent_idx == a_i)
                    if not mask_agent_seq.any():
                        continue
                    
                    choose_b = mask_agent_seq.to(self.device)
                    choose_lb = choose_b.unsqueeze(0).expand(L, B)
                    choose_lb = choose_lb.reshape(L * B)
                    
                    obs_a = obs_flat[choose_lb]
                    act_a = act_flat[choose_lb]
                    old_logp_a = old_logp_flat[choose_lb]
                    adv_a = adv_flat[choose_lb]
                    masks_a = masks_flat[choose_lb]
                    ret_vh_a = ret_vh_flat[choose_lb]
                    old_vh_a = old_vh_flat[choose_lb]
                    
                    h0_actor_a = h0_actor[mask_agent_seq]
                    h0_vh_a = h0_vh[mask_agent_seq]
                    
                    # z_enc for this agent: detached for critic, non-detached for actor
                    z_enc_a = z_enc_flat[choose_lb]
                    z_enc_a_det = z_enc_a.detach()
                    
                    # Vh critic - uses detached z_enc
                    vh_pred_a, _ = self.critics_vh[agent].value_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_a_det,
                        hxs_init=h0_vh_a,
                        masks_seq=masks_a,
                    )
                    
                    vh_clipped_a = old_vh_a + torch.clamp(
                        vh_pred_a - old_vh_a, -value_clip_eps, value_clip_eps
                    )
                    vh_loss_unclipped_a = (vh_pred_a - ret_vh_a).pow(2)
                    vh_loss_clipped_a = (vh_clipped_a - ret_vh_a).pow(2)
                    vh_loss_a = 0.5 * torch.max(vh_loss_unclipped_a, vh_loss_clipped_a).mean()
                    
                    # Policy - uses non-detached z_enc
                    logp_new_a, entropy_a, _ = self.actors[agent].evaluate_actions_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_a,  # NOT detached
                        hxs_init=h0_actor_a,
                        masks_seq=masks_a,
                        act_seq=act_a,
                    )
                    
                    ratio = torch.exp(logp_new_a - old_logp_a)
                    surr1 = ratio * adv_a
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_a
                    policy_loss_a = -torch.min(surr1, surr2).mean() - ent_coef * entropy_a.mean()
                    
                    with torch.no_grad():
                        approx_kl_a = ((ratio - 1) - torch.log(ratio)).mean()
                        clipfrac_a = (torch.abs(ratio - 1) > clip_eps).float().mean()
                    
                    policy_loss_total += policy_loss_a
                    entropy_total += entropy_a.mean()
                    vh_loss_total += vh_loss_a
                    approx_kl_total += approx_kl_a
                    clipfrac_total += clipfrac_a
                    count_total += 1.0
                
                if count_total < 1e-8:
                    continue
                    
                policy_loss = policy_loss_total / count_total
                vh_loss = vh_loss_total / count_total
                entropy = entropy_total / count_total
                approx_kl = approx_kl_total / count_total
                clipfrac = clipfrac_total / count_total
                
                # Backward pass
                self.optimizer_actor.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.z_encoder.parameters()) +
                    [p for ag in self.agent_ids for p in self.actors[ag].parameters()],
                    self.max_grad_norm_actor
                )
                self.optimizer_actor.step()
                
                self.optimizer_vl.zero_grad()
                vl_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_vl.parameters(),
                    self.max_grad_norm_critic
                )
                self.optimizer_vl.step()
                
                for ag in self.agent_ids:
                    self.optimizers_vh[ag].zero_grad()
                vh_loss.backward()
                for ag in self.agent_ids:
                    torch.nn.utils.clip_grad_norm_(
                        self.critics_vh[ag].parameters(),
                        self.max_grad_norm_critic
                    )
                    self.optimizers_vh[ag].step()
                
                update_info["loss_policy"] += policy_loss.item()
                update_info["loss_value_vl"] += vl_loss.item()
                update_info["loss_value_vh"] += vh_loss.item()
                update_info["entropy"] += entropy.item()
                update_info["approx_kl"] += approx_kl.item()
                update_info["clipfrac"] += clipfrac.item()
                
                num_updates += 1
        
        for k in update_info:
            update_info[k] /= max(1, num_updates)
        
        self.buffer.reset()
        
        return update_info
    
    def _recurrent_generator(self, chunk_length: int):
        """
        Generate mini-batches for sequence-based RNN training.
        Fixed with proper agent-major indexing.
        """
        T = self.rollout_horizon
        E = self.num_envs
        A = self.num_agents
        N = E * A
        
        assert T % chunk_length == 0, f"T {T} not divisible by chunk_length {chunk_length}"
        num_chunks = T // chunk_length
        
        # Flatten all data
        obs_all = self.buffer.obs[:T].reshape(T * N, -1)
        share_obs_all = self.buffer.share_obs[:T].reshape(T * N, -1)
        actions_all = self.buffer.actions[:T].reshape(T * N, -1)
        z_all = self.buffer.z[:T].reshape(T * N, -1)
        old_logp_all = self.buffer.action_log_probs[:T].reshape(T * N, 1)
        
        returns_vl_all = self.buffer.returns_vl.reshape(T * N, 1)
        returns_vh_all = self.buffer.returns_vh.reshape(T * N, 1)
        advantages_all = self.buffer.advantages.reshape(T * N, 1)
        
        values_vl_all = self.buffer.values_vl[:T].reshape(T * N, 1)
        values_vh_all = self.buffer.values_vh[:T].reshape(T * N, 1)
        
        masks_all = self.buffer.masks[:T].reshape(T * N, 1)
        
        rnn_states_actor_all = self.buffer.rnn_states_actor[:T].reshape(T, N, -1)
        rnn_states_critic_all = self.buffer.rnn_states_critic[:T].reshape(T, N, -1)
        rnn_states_vh_all = self.buffer.rnn_states_vh[:T].reshape(T, N, -1)
        
        # Create (chunk_id, env_id, agent_id) indices
        indices = []
        for chunk_id in range(num_chunks):
            for env_id in range(E):
                for agent_id in range(A):
                    indices.append((chunk_id, env_id, agent_id))
        
        total_samples = len(indices)
        batch_size = total_samples // self.num_mini_batch
        
        for _ in range(self.num_mini_batch):
            # Sample mini-batch
            perm = torch.randperm(total_samples)
            mb_indices = perm[:batch_size].tolist()
            
            mb_data = []
            for idx in mb_indices:
                chunk_id, env_id, agent_id = indices[idx]
                t0 = chunk_id * chunk_length
                
                # FIX 2: Agent-major indexing
                n_idx = agent_id * E + env_id
                
                # Extract sequence [t0:t0+L] for this (env, agent)
                seq_indices = [t * N + n_idx for t in range(t0, t0 + chunk_length)]
                seq_indices = torch.tensor(seq_indices, device=self.device, dtype=torch.long)
                
                mb_data.append({
                    "obs": obs_all[seq_indices],
                    "share_obs": share_obs_all[seq_indices],
                    "actions": actions_all[seq_indices],
                    "z": z_all[seq_indices],
                    "old_logp": old_logp_all[seq_indices],
                    "ret_vl": returns_vl_all[seq_indices],
                    "ret_vh": returns_vh_all[seq_indices],
                    "advantages": advantages_all[seq_indices],
                    "old_vl": values_vl_all[seq_indices],
                    "old_vh": values_vh_all[seq_indices],
                    "masks": masks_all[seq_indices],
                    "h0_actor": rnn_states_actor_all[t0, n_idx],
                    "h0_vl": rnn_states_critic_all[t0, n_idx],
                    "h0_vh": rnn_states_vh_all[t0, n_idx],
                    "agent_id": agent_id,
                })
            
            # Stack mini-batch
            B = len(mb_data)
            L = chunk_length
            
            yield {
                "obs_flat": torch.cat([d["obs"] for d in mb_data], dim=0),
                "share_obs_flat": torch.cat([d["share_obs"] for d in mb_data], dim=0),
                "act_flat": torch.cat([d["actions"] for d in mb_data], dim=0),
                "old_logp_flat": torch.cat([d["old_logp"] for d in mb_data], dim=0),
                "adv_flat": torch.cat([d["advantages"] for d in mb_data], dim=0),
                "ret_vl_flat": torch.cat([d["ret_vl"] for d in mb_data], dim=0),
                "ret_vh_flat": torch.cat([d["ret_vh"] for d in mb_data], dim=0),
                "old_vl_flat": torch.cat([d["old_vl"] for d in mb_data], dim=0),
                "old_vh_flat": torch.cat([d["old_vh"] for d in mb_data], dim=0),
                "z_flat": torch.cat([d["z"] for d in mb_data], dim=0),
                "masks_flat": torch.cat([d["masks"] for d in mb_data], dim=0),
                "h0_actor": torch.stack([d["h0_actor"] for d in mb_data], dim=0),
                "h0_vl": torch.stack([d["h0_vl"] for d in mb_data], dim=0),
                "h0_vh": torch.stack([d["h0_vh"] for d in mb_data], dim=0),
                "agent_idx": torch.tensor([d["agent_id"] for d in mb_data], device=self.device, dtype=torch.long),
            }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained policy."""
        self.set_eval_mode()
        
        episode_returns = []
        episode_lengths = []
        episode_successes = []
        z_values = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            self._init_rnn_states()
            
            z_global = torch.zeros(self.num_envs, 1, device=self.device)
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 2000:
                z_enc = self.z_encoder(z_global)
                z_values.append(z_global.mean().item())
                
                actions = {}
                masks = torch.ones(self.num_envs, 1, device=self.device)
                
                for agent in self.agent_ids:
                    act, _, rnn_h, _ = self.actors[agent].act_step(
                        obs[agent], z_enc, self.rnn_states[agent]["actor"], masks, deterministic=True
                    )
                    actions[agent] = act
                    self.rnn_states[agent]["actor"] = rnn_h
                
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                for agent in self.agent_ids:
                    episode_return += rewards[agent].mean().item()
                episode_length += 1
                
                done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for agent in self.agent_ids:
                    agent_done = terminated[agent] | truncated[agent]
                    if agent_done.dim() > 1:
                        agent_done = agent_done.squeeze(-1)
                    done_any |= agent_done
                
                if done_any.any():
                    done = True
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if "progress_ratio" in info:
                progress = info["progress_ratio"].mean().item()
                episode_successes.append(1.0 if progress >= 0.95 else 0.0)
            else:
                episode_successes.append(0.0)
        
        return {
            "return_mean": np.mean(episode_returns),
            "return_std": np.std(episode_returns),
            "episode_length": np.mean(episode_lengths),
            "success_rate": np.mean(episode_successes),
            "z_global_mean": np.mean(z_values),
            "z_global_std": np.std(z_values),
        }
    
    def save_checkpoint(self, path: str, global_step: Optional[int] = None, update_count: Optional[int] = None):
        """Save checkpoint."""
        checkpoint = {
            "actors": {agent: self.actors[agent].state_dict() for agent in self.agent_ids},
            "critics_vh": {agent: self.critics_vh[agent].state_dict() for agent in self.agent_ids},
            "critic_vl": self.critic_vl.state_dict(),
            "z_encoder": self.z_encoder.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_vl": self.optimizer_vl.state_dict(),
            "optimizers_vh": {agent: self.optimizers_vh[agent].state_dict() for agent in self.agent_ids},
            "global_step": global_step if global_step is not None else self.global_step,
            "episodes_done": self.episodes_done,
        }
        
        if update_count is not None:
            checkpoint["update_count"] = update_count
        
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for agent in self.agent_ids:
            self.actors[agent].load_state_dict(checkpoint["actors"][agent])
            self.critics_vh[agent].load_state_dict(checkpoint["critics_vh"][agent])
        
        self.critic_vl.load_state_dict(checkpoint["critic_vl"])
        self.z_encoder.load_state_dict(checkpoint["z_encoder"])
        
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_vl.load_state_dict(checkpoint["optimizer_vl"])
        for agent in self.agent_ids:
            self.optimizers_vh[agent].load_state_dict(checkpoint["optimizers_vh"][agent])
        
        self.global_step = checkpoint["global_step"]
        self.episodes_done = checkpoint["episodes_done"]
        
        print(f"[CHECKPOINT] Loaded from {path}")
    
    def set_train_mode(self):
        """Set all networks to train mode."""
        self.z_encoder.train()
        for agent in self.agent_ids:
            self.actors[agent].train()
            self.critics_vh[agent].train()
        self.critic_vl.train()
    
    def set_eval_mode(self):
        """Set all networks to eval mode."""
        self.z_encoder.eval()
        for agent in self.agent_ids:
            self.actors[agent].eval()
            self.critics_vh[agent].eval()
        self.critic_vl.eval()