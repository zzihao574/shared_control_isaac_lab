"""
Epigraph Trainer for Safe Multi-Agent Reinforcement Learning.
Handles complete training loop: rollout collection, dual PPO updates, evaluation with root finding.
Maintains compatibility with rMAPPO training structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import numpy as np

from .epigraph_core import ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder
from .rollout_buffer_z import RolloutBufferZ


class EpigraphTrainer:
    """
    Complete Epigraph training system for surgical MARL environment.
    
    Architecture:
    - ZEncoder: Shared z encoding (z -> z_enc)
    - ActorRNN: Per-agent policy (obs_i ⊕ z_enc -> action)
    - CriticVlRNN: Centralized task value (share_obs ⊕ z_enc -> Vl)
    - CriticVhRNN: Per-agent safety value (obs_i ⊕ z_enc -> Vh)
    - RootFinder: Solve Vh(o,z) - z = 0 for evaluation
    """
    
    def __init__(self, env, device: torch.device, algo_cfg: Dict[str, Any], epi_cfg: Dict[str, Any]):
        """
        Initialize Epigraph trainer.
        
        Args:
            env: Gymnasium MARL environment
            device: Training device
            algo_cfg: PPO hyperparameters (from algorithms.rmappo in YAML)
            epi_cfg: Epigraph-specific config (from epigraph in YAML)
        """
        self.env = env
        self.device = device
        self.algo_cfg = algo_cfg
        self.epi_cfg = epi_cfg
        
        # Get environment config flags from injected params
        if hasattr(env.unwrapped, "params"):
            self.env_cfg = env.unwrapped.params.get("epigraph_env", {})
        else:
            self.env_cfg = {}
        
        # Environment info
        self.num_envs = env.num_envs
        self.agent_ids = list(env.unwrapped.cfg.possible_agents)  # ['human', 'robot']
        self.num_agents = len(self.agent_ids)
        
        # Dimensions
        self.obs_dim = env.unwrapped.cfg.observation_spaces[self.agent_ids[0]]
        # TODO: share_obs_dim should be env.unwrapped.cfg.state_space (38)
        # Currently using concatenated obs (12) as temporary solution
        # If environment provides get_state() method, use that instead
        self.share_obs_dim = self.obs_dim * self.num_agents  # 6 * 2 = 12 (concat obs)
        self.act_dim = env.unwrapped.cfg.action_spaces[self.agent_ids[0]]
        self.hidden_size = algo_cfg["hidden_size"]
        
        # Training hyperparameters
        self.rollout_horizon = algo_cfg["rollout_horizon"]
        self.gamma = algo_cfg["gamma"]
        self.gae_lambda = algo_cfg["gae_lambda"]
        self.ppo_epoch = algo_cfg["ppo_epoch"]
        self.num_mini_batch = algo_cfg["num_mini_batch"]
        self.data_chunk_length = algo_cfg["data_chunk_length"]
        self.clip_param = algo_cfg["clip_param"]
        self.entropy_coef = algo_cfg["entropy_coef"]
        self.max_grad_norm_actor = algo_cfg["max_grad_norm_actor"]
        self.max_grad_norm_critic = algo_cfg["max_grad_norm_critic"]
        
        # Epigraph-specific
        self.z_min = epi_cfg["z"]["min"]
        self.z_max = epi_cfg["z"]["max"]
        self.z_nz = epi_cfg["z"]["encode"]["nz"]
        self.lambda_safe = epi_cfg["losses"]["lambda_safe"]
        
        # Initialize networks and buffer
        self._build_networks()
        self._build_buffer()
        self._build_optimizers()
        
        # RNN state tracking
        self._init_rnn_states()
        
        # Statistics
        self.global_step = 0
        self.episodes_done = 0
        
        print(f"[EPIGRAPH] Trainer initialized")
        print(f"[EPIGRAPH] Agents: {self.agent_ids}")
        print(f"[EPIGRAPH] Dims: obs={self.obs_dim}, act={self.act_dim}, share_obs={self.share_obs_dim}")
        print(f"[EPIGRAPH] Z range: [{self.z_min}, {self.z_max}], encoding dim={self.z_nz}")
    
    def _build_networks(self):
        """Build all Epigraph networks."""
        use_orthogonal = self.algo_cfg["use_orthogonal"]
        gain = self.algo_cfg["gain"]
        
        # Shared Z encoder
        self.z_encoder = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
            use_orthogonal=use_orthogonal
        ).to(self.device)
        
        # Per-agent actors
        self.actors = nn.ModuleDict({
            agent: ActorRNN(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                z_nz=self.z_nz,
                hidden_size=self.hidden_size,
                recurrent_N=self.algo_cfg["recurrent_N"],
                use_orthogonal=use_orthogonal,
                gain=gain
            ).to(self.device) for agent in self.agent_ids
        })
        
        # Centralized task critic (Vl)
        self.critic_vl = CriticVlRNN(
            share_obs_dim=self.share_obs_dim,
            z_nz=self.z_nz,
            hidden_size=self.hidden_size,
            recurrent_N=self.algo_cfg["recurrent_N"],
            use_orthogonal=use_orthogonal,
            gain=gain
        ).to(self.device)
        
        # Per-agent safety critics (Vh)
        self.critics_vh = nn.ModuleDict({
            agent: CriticVhRNN(
                obs_dim=self.obs_dim,
                z_nz=self.z_nz,
                hidden_size=self.hidden_size,
                recurrent_N=self.algo_cfg["recurrent_N"],
                use_orthogonal=use_orthogonal,
                gain=gain
            ).to(self.device) for agent in self.agent_ids
        })
        
        # Root finder for evaluation
        self.root_finder = RootFinder(
            z_min=self.z_min,
            z_max=self.z_max,
            max_iter=32,
            tol=1e-4
        )
        
        print(f"[EPIGRAPH] Networks built: 1 ZEncoder + {self.num_agents} Actors + 1 Vl + {self.num_agents} Vh")
    
    def _build_buffer(self):
        """Build rollout buffer."""
        # N = num_envs * num_agents
        N = self.num_envs * self.num_agents
        
        self.buffer = RolloutBufferZ(
            T=self.rollout_horizon,
            N=N,
            obs_dim=self.obs_dim,
            share_obs_dim=self.share_obs_dim,
            act_dim=self.act_dim,
            rnn_hidden_dim=self.hidden_size,
            device=self.device
        )
        
        print(f"[EPIGRAPH] Buffer created: T={self.rollout_horizon}, N={N}")
    
    def _build_optimizers(self):
        """Build optimizers for all networks."""
        actor_lr = self.algo_cfg["actor_lr"]
        critic_lr = self.algo_cfg["critic_lr"]
        opt_eps = self.algo_cfg["opt_eps"]
        weight_decay = self.algo_cfg["weight_decay"]
        
        # Collect all actor parameters
        actor_params = []
        for actor in self.actors.values():
            actor_params.extend(list(actor.parameters()))
        actor_params.extend(list(self.z_encoder.parameters()))  # Include z_encoder in actor optimization
        
        self.optimizer_actor = optim.Adam(actor_params, lr=actor_lr, eps=opt_eps, weight_decay=weight_decay)
        
        # Vl optimizer
        self.optimizer_vl = optim.Adam(self.critic_vl.parameters(), lr=critic_lr, eps=opt_eps, weight_decay=weight_decay)
        
        # Vh optimizers (per-agent)
        self.optimizers_vh = {
            agent: optim.Adam(self.critics_vh[agent].parameters(), lr=critic_lr, eps=opt_eps, weight_decay=weight_decay)
            for agent in self.agent_ids
        }
        
        print(f"[EPIGRAPH] Optimizers built: actor_lr={actor_lr}, critic_lr={critic_lr}")
    
    def _init_rnn_states(self):
        """Initialize RNN hidden states."""
        # Per-agent RNN states
        self.rnn_states = {
            agent: {
                "actor": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
                "vh": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
            } for agent in self.agent_ids
        }
        
        # Centralized Vl RNN state
        self.rnn_states_vl = torch.zeros(self.num_envs, self.hidden_size, device=self.device)
    
    def set_train_mode(self):
        """Set all networks to training mode."""
        self.z_encoder.train()
        for actor in self.actors.values():
            actor.train()
        self.critic_vl.train()
        for critic in self.critics_vh.values():
            critic.train()
    
    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        self.z_encoder.eval()
        for actor in self.actors.values():
            actor.eval()
        self.critic_vl.eval()
        for critic in self.critics_vh.values():
            critic.eval()
    
    def _init_z_training(self):
        """
        Initialize z values for training rollout.
        
        Returns:
            z: [num_envs, num_agents, 1] - per-agent z values
        """
        mode = self.epi_cfg["z"]["init"]["mode"]
        p_extreme = self.epi_cfg["z"]["init"].get("p_extreme", 0.3)
        
        z = torch.zeros(self.num_envs, self.num_agents, 1, device=self.device)
        
        if mode == "uniform":
            z = torch.rand_like(z) * (self.z_max - self.z_min) + self.z_min
        elif mode == "extreme":
            rand = torch.rand(self.num_envs, self.num_agents, 1, device=self.device)
            z = torch.where(rand > 0.5, 
                          torch.full_like(z, self.z_max),
                          torch.full_like(z, self.z_min))
        elif mode == "mixed":
            rand1 = torch.rand(self.num_envs, self.num_agents, 1, device=self.device)
            rand2 = torch.rand(self.num_envs, self.num_agents, 1, device=self.device)
            
            use_extreme = rand1 < p_extreme
            z_extreme = torch.where(rand2 > 0.5,
                                   torch.full_like(z, self.z_max),
                                   torch.full_like(z, self.z_min))
            z_uniform = torch.rand_like(z) * (self.z_max - self.z_min) + self.z_min
            
            z = torch.where(use_extreme, z_extreme, z_uniform)
        else:
            raise ValueError(f"Unknown z init mode: {mode}")
        
        return z
    
    def _update_z_training(self, z_current, r_safe):
        """
        Update z values during training rollout.
        
        Args:
            z_current: [num_envs, num_agents, 1]
            r_safe: Dict[agent, [num_envs, 1]]
        
        Returns:
            z_next: [num_envs, num_agents, 1]
        """
        z_next = torch.zeros_like(z_current)
        
        for i, agent in enumerate(self.agent_ids):
            r_h = r_safe[agent]  # [num_envs, 1]
            z_next[:, i:i+1] = (z_current[:, i:i+1] + r_h.unsqueeze(1)) / self.gamma
        
        z_next = torch.clamp(z_next, self.z_min, self.z_max)
        
        return z_next
    @torch.no_grad()
    def collect_rollout(self):
        """
        Collect one rollout of experience with per-agent z recursion.
        
        Returns:
            rollout_info: Dictionary with rollout statistics
        """
        self.set_eval_mode()
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Initialize z for all agents
        z = self._init_z_training()  # [num_envs, num_agents, 1]
        
        # Reset RNN states
        self._init_rnn_states()
        
        # Statistics tracking
        episode_returns_task = []
        episode_returns_safe = []
        episode_returns_total = []
        episode_lengths = []
        current_returns_task = torch.zeros(self.num_envs, device=self.device)
        current_returns_safe = torch.zeros(self.num_envs, device=self.device)
        current_returns_total = torch.zeros(self.num_envs, device=self.device)
        current_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Rollout loop
        for t in range(self.rollout_horizon):
            # ========== Encode z for each agent ==========
            z_enc = {}
            for i, agent in enumerate(self.agent_ids):
                z_enc[agent] = self.z_encoder(z[:, i])  # [num_envs, z_nz]
            
            # ========== Actor forward ==========
            actions = {}
            action_log_probs = {}
            
            for agent in self.agent_ids:
                obs_agent = obs[agent]  # [num_envs, obs_dim]
                masks = torch.ones(self.num_envs, 1, device=self.device)
                
                act, logp, rnn_h = self.actors[agent](
                    obs_agent, z_enc[agent],
                    self.rnn_states[agent]["actor"], masks,
                    deterministic=False
                )
                
                actions[agent] = act
                action_log_probs[agent] = logp
                self.rnn_states[agent]["actor"] = rnn_h
            
            # ========== Critic Vl forward (centralized) ==========
            share_obs = self._get_share_obs(obs)  # [num_envs, share_obs_dim]
            z_enc_global = self._get_global_z_enc(z_enc)  # [num_envs, z_nz]
            masks_vl = torch.ones(self.num_envs, 1, device=self.device)
            
            vl, rnn_vl_h = self.critic_vl(share_obs, z_enc_global, self.rnn_states_vl, masks_vl)
            self.rnn_states_vl = rnn_vl_h
            
            # ========== Critic Vh forward (per-agent) ==========
            vh = {}
            for agent in self.agent_ids:
                obs_agent = obs[agent]
                masks = torch.ones(self.num_envs, 1, device=self.device)
                
                vh_val, rnn_vh_h = self.critics_vh[agent](
                    obs_agent, z_enc[agent],
                    self.rnn_states[agent]["vh"], masks
                )
                
                vh[agent] = vh_val
                self.rnn_states[agent]["vh"] = rnn_vh_h
            
            # ========== Step environment ==========
            obs_next, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Extract decomposed rewards
            r_task = info["r_task"]  # Dict[agent, [num_envs, 1]]
            r_safe = info["r_safe"]  # Dict[agent, [num_envs, 1]]
            
            # ========== Insert into buffer ==========
            self._insert_buffer_step(
                t, obs, share_obs, actions, action_log_probs,
                rewards, r_task, r_safe, vl, vh, z, terminated, truncated
            )
            
            # ========== Update z for next step ==========
            z = self._update_z_training(z, r_safe)
            
            # ========== Track statistics ==========
            for agent in self.agent_ids:
                current_returns_task += r_task[agent].squeeze(-1)
                current_returns_safe += r_safe[agent].squeeze(-1)
                current_returns_total += rewards[agent].squeeze(-1)
            current_lengths += 1
            
            # Check for episode completion
            done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent in self.agent_ids:
                done_any |= (terminated[agent] | truncated[agent]).squeeze(-1)
            
            if done_any.any():
                episode_returns_task.append(current_returns_task[done_any].mean().item())
                episode_returns_safe.append(current_returns_safe[done_any].mean().item())
                episode_returns_total.append(current_returns_total[done_any].mean().item())
                episode_lengths.append(current_lengths[done_any].float().mean().item())
                current_returns_task[done_any] = 0
                current_returns_safe[done_any] = 0
                current_returns_total[done_any] = 0
                current_lengths[done_any] = 0
                self.episodes_done += done_any.sum().item()
            
            # Move to next observation
            obs = obs_next
            self.global_step += self.num_envs
        
        # ========== Compute bootstrap values ==========
        with torch.no_grad():
            z_enc_last = {}
            for i, agent in enumerate(self.agent_ids):
                z_enc_last[agent] = self.z_encoder(z[:, i])
            
            share_obs_last = self._get_share_obs(obs)
            z_enc_global_last = self._get_global_z_enc(z_enc_last)
            masks_last = torch.ones(self.num_envs, 1, device=self.device)
            
            vl_last, _ = self.critic_vl(share_obs_last, z_enc_global_last, self.rnn_states_vl, masks_last)
            
            vh_last = {}
            for agent in self.agent_ids:
                vh_val, _ = self.critics_vh[agent](
                    obs[agent], z_enc_last[agent],
                    self.rnn_states[agent]["vh"], masks_last
                )
                vh_last[agent] = vh_val
        
        # Flatten bootstrap values for buffer (N = num_envs * num_agents)
        vl_last_flat = vl_last.repeat(1, self.num_agents).view(-1, 1)
        vh_last_flat = torch.cat([vh_last[agent] for agent in self.agent_ids], dim=0)
        
        # ========== Compute GAE ==========
        self.buffer.compute_gae_dual(
            vl_last_flat, vh_last_flat,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            lambda_safe=self.lambda_safe
        )
        
        # ========== Rollout statistics ==========
        rollout_info = {
            "return_task_mean": np.mean(episode_returns_task) if episode_returns_task else 0.0,
            "return_safe_mean": np.mean(episode_returns_safe) if episode_returns_safe else 0.0,
            "return_total_mean": np.mean(episode_returns_total) if episode_returns_total else 0.0,
            "episode_length_mean": np.mean(episode_lengths) if episode_lengths else 0.0,
            "episodes_done": self.episodes_done,
        }
        
        return rollout_info
    
    def _get_share_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Construct centralized observation from per-agent observations.
        
        TODO: If environment provides get_state() method that returns 38-dim state,
        use that instead of concatenating observations. For now, we use simple concat.
        
        Args:
            obs: Dict[agent, [num_envs, obs_dim]]
        
        Returns:
            share_obs: [num_envs, share_obs_dim = obs_dim * num_agents]
        """
        obs_list = [obs[agent] for agent in self.agent_ids]
        share_obs = torch.cat(obs_list, dim=-1)
        return share_obs
    
    def _get_global_z_enc(self, z_enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute global z encoding from per-agent encodings.
        
        Strategy: Max pooling across agents
        
        Args:
            z_enc: Dict[agent, [num_envs, z_nz]]
        
        Returns:
            z_enc_global: [num_envs, z_nz]
        """
        z_enc_stacked = torch.stack([z_enc[agent] for agent in self.agent_ids], dim=1)
        z_enc_global, _ = z_enc_stacked.max(dim=1)
        return z_enc_global
    
    def _insert_buffer_step(self, t, obs, share_obs, actions, action_log_probs,
                           rewards, r_task, r_safe, vl, vh, z, terminated, truncated):
        """
        Insert one timestep of experience into buffer with agent flattening.
        
        Agent ordering: [robot_env0, ..., robot_envN, human_env0, ..., human_envN]
        """
        # ========== Flatten observations ==========
        # [num_envs * num_agents, obs_dim]
        obs_flat = torch.cat([obs[agent] for agent in self.agent_ids], dim=0)
        
        # Share obs: same for both agents, so duplicate
        share_obs_flat = share_obs.repeat(self.num_agents, 1)
        
        # ========== Flatten actions and log probs ==========
        actions_flat = torch.cat([actions[agent] for agent in self.agent_ids], dim=0)
        action_log_probs_flat = torch.cat([action_log_probs[agent] for agent in self.agent_ids], dim=0)
        
        # ========== Flatten rewards ==========
        rewards_flat = torch.cat([rewards[agent] for agent in self.agent_ids], dim=0)
        r_task_flat = torch.cat([r_task[agent] for agent in self.agent_ids], dim=0)
        r_safe_flat = torch.cat([r_safe[agent] for agent in self.agent_ids], dim=0)
        
        # ========== Flatten values ==========
        vl_flat = vl.repeat(self.num_agents, 1)
        vh_flat = torch.cat([vh[agent] for agent in self.agent_ids], dim=0)
        
        # ========== Flatten z ==========
        # [num_envs, num_agents, 1] -> [num_envs * num_agents, 1]
        z_flat = z.view(-1, 1)
        
        # ========== Flatten masks ==========
        # Episode ends if ANY agent terminates/truncates
        done_any = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        trunc_any = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        
        for agent in self.agent_ids:
            done_any |= (terminated[agent] | truncated[agent])
            trunc_any |= truncated[agent]
        
        # masks: 0 at any episode boundary (terminated OR truncated)
        masks_flat = (~done_any).float().repeat(self.num_agents, 1)
        
        # term_masks: 1 ONLY for truncated episodes (time limit), 0 for terminated
        # This allows bootstrap for time-limited episodes while preventing it for naturally terminated ones
        term_masks_flat = trunc_any.float().repeat(self.num_agents, 1)
        
        # ========== Flatten RNN states ==========
        rnn_states_actor_flat = torch.cat([self.rnn_states[agent]["actor"] for agent in self.agent_ids], dim=0)
        rnn_states_vl_flat = self.rnn_states_vl.repeat(self.num_agents, 1)
        rnn_states_vh_flat = torch.cat([self.rnn_states[agent]["vh"] for agent in self.agent_ids], dim=0)
        
        # ========== Insert into buffer ==========
        self.buffer.insert(
            t,
            obs=obs_flat,
            share_obs=share_obs_flat,
            actions=actions_flat,
            action_log_probs=action_log_probs_flat,
            rewards=rewards_flat,
            masks=masks_flat,
            rnn_states_actor=rnn_states_actor_flat,
            rnn_states_critic=rnn_states_vl_flat,
            term_masks=term_masks_flat,
            zs=z_flat,
            rewards_task=r_task_flat,
            rewards_safe=r_safe_flat,
            value_preds_l=vl_flat,
            value_preds_h=vh_flat,
            rnn_states_vh=rnn_states_vh_flat,
        )
    def update(self):
        """
        Perform PPO update on collected rollout.
        
        Returns:
            update_info: Dictionary with training statistics
        """
        self.set_train_mode()
        
        train_info = {
            "loss_policy": [],
            "loss_value_vl": [],
            "loss_value_vh": [],
            "loss_entropy": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "z_mean": [],
            "z_std": [],
        }
        
        # PPO epochs
        for epoch in range(self.ppo_epoch):
            # Generate mini-batches
            for batch in self.buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length):
                # ========== Extract batch data ==========
                obs = batch["obs"]  # [L, B, obs_dim]
                share_obs = batch["share_obs"]  # [L, B, share_obs_dim]
                actions = batch["actions"]  # [L, B, act_dim]
                old_action_log_probs = batch["action_log_probs"]  # [L, B, 1]
                advantages_task = batch["advantages_task"]  # [L, B, 1]
                advantages_safe = batch["advantages_safe"]  # [L, B, 1]
                returns_task = batch["returns_task"]  # [L, B, 1]
                returns_safe = batch["returns_safe"]  # [L, B, 1]
                value_preds_l_old = batch["value_preds_l"]  # [L, B, 1]
                value_preds_h_old = batch["value_preds_h"]  # [L, B, 1]
                masks = batch["masks"]  # [L, B, 1]
                zs = batch["zs"]  # [L, B, 1]
                rnn_states_actor = batch["rnn_states_actor"]  # [B, H]
                rnn_states_critic = batch["rnn_states_critic"]  # [B, H]
                rnn_states_vh = batch["rnn_states_vh"]  # [B, H]
                
                L, B = obs.size(0), obs.size(1)
                
                # ========== Encode z ==========
                # Flatten to [L*B, 1] and encode
                z_flat = zs.reshape(L * B, 1)
                z_enc = self.z_encoder(z_flat)  # [L*B, z_nz]
                
                # ========== Flatten for network input ==========
                obs_flat = obs.reshape(L * B, -1)
                share_obs_flat = share_obs.reshape(L * B, -1)
                actions_flat = actions.reshape(L * B, -1)
                masks_flat = masks.reshape(L * B, 1)
                old_action_log_probs_flat = old_action_log_probs.reshape(L * B, 1)
                advantages_task_flat = advantages_task.reshape(L * B, 1)
                advantages_safe_flat = advantages_safe.reshape(L * B, 1)
                returns_task_flat = returns_task.reshape(L * B, 1)
                returns_safe_flat = returns_safe.reshape(L * B, 1)
                value_preds_l_old_flat = value_preds_l_old.reshape(L * B, 1)
                value_preds_h_old_flat = value_preds_h_old.reshape(L * B, 1)
                
                # ========== Combined advantage ==========
                # A_combined = A_task - lambda_safe * A_safe
                advantages_combined = advantages_task_flat - self.lambda_safe * advantages_safe_flat
                
                # ========== Per-agent processing ==========
                # We need to process each agent separately for actor evaluation
                agents_per_batch = B // self.num_agents
                
                action_log_probs_new_list = []
                dist_entropy_list = []
                
                for i, agent in enumerate(self.agent_ids):
                    start_idx = i * agents_per_batch
                    end_idx = (i + 1) * agents_per_batch
                    
                    obs_agent = obs_flat[start_idx:end_idx]
                    z_enc_agent = z_enc[start_idx:end_idx]
                    actions_agent = actions_flat[start_idx:end_idx]
                    masks_agent = masks_flat[start_idx:end_idx]
                    rnn_states_actor_agent = rnn_states_actor[i::self.num_agents]  # Slice by stride
                    
                    # Evaluate actions
                    action_log_probs_new, dist_entropy = self.actors[agent].evaluate_actions(
                        obs_agent, z_enc_agent, rnn_states_actor_agent, actions_agent, masks_agent
                    )
                    
                    action_log_probs_new_list.append(action_log_probs_new)
                    dist_entropy_list.append(dist_entropy)
                
                # Concatenate across agents
                action_log_probs_new = torch.cat(action_log_probs_new_list, dim=0)
                dist_entropy = torch.cat(dist_entropy_list, dim=0)
                
                # ========== Policy loss ==========
                ratio = torch.exp(action_log_probs_new - old_action_log_probs_flat)
                
                surr1 = ratio * advantages_combined
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_combined
                
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -dist_entropy.mean()
                
                policy_loss = actor_loss + self.entropy_coef * entropy_loss
                
                # ========== Update actor ==========
                self.optimizer_actor.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for actor in self.actors.values() for p in actor.parameters()] + 
                    list(self.z_encoder.parameters()),
                    self.max_grad_norm_actor
                )
                self.optimizer_actor.step()
                
                # ========== Value loss Vl (centralized) ==========
                # Encode global z
                z_enc_stacked = z_enc.view(B, -1, self.z_nz)  # [B, num_agents, z_nz]
                z_enc_global, _ = z_enc_stacked.max(dim=1)  # [B, z_nz]
                z_enc_global = z_enc_global.repeat_interleave(L, dim=0)  # [L*B, z_nz]
                
                value_preds_l_new = self.critic_vl.evaluate_values(
                    share_obs_flat, z_enc_global, rnn_states_critic, masks_flat
                )
                
                # Clipped value loss
                value_pred_clipped = value_preds_l_old_flat + torch.clamp(
                    value_preds_l_new - value_preds_l_old_flat, -self.clip_param, self.clip_param
                )
                value_loss_unclipped = (value_preds_l_new - returns_task_flat) ** 2
                value_loss_clipped = (value_pred_clipped - returns_task_flat) ** 2
                vl_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Update Vl
                self.optimizer_vl.zero_grad()
                vl_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_vl.parameters(), self.max_grad_norm_critic)
                self.optimizer_vl.step()
                
                # ========== Value loss Vh (per-agent) ==========
                vh_loss_total = 0.0
                
                for i, agent in enumerate(self.agent_ids):
                    start_idx = i * agents_per_batch
                    end_idx = (i + 1) * agents_per_batch
                    
                    obs_agent = obs_flat[start_idx:end_idx]
                    z_enc_agent = z_enc[start_idx:end_idx]
                    masks_agent = masks_flat[start_idx:end_idx]
                    rnn_states_vh_agent = rnn_states_vh[i::self.num_agents]
                    
                    value_preds_h_new = self.critics_vh[agent].evaluate_values(
                        obs_agent, z_enc_agent, rnn_states_vh_agent, masks_agent
                    )
                    
                    returns_safe_agent = returns_safe_flat[start_idx:end_idx]
                    value_preds_h_old_agent = value_preds_h_old_flat[start_idx:end_idx]
                    
                    # Clipped value loss
                    value_pred_clipped = value_preds_h_old_agent + torch.clamp(
                        value_preds_h_new - value_preds_h_old_agent, -self.clip_param, self.clip_param
                    )
                    value_loss_unclipped = (value_preds_h_new - returns_safe_agent) ** 2
                    value_loss_clipped = (value_pred_clipped - returns_safe_agent) ** 2
                    vh_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    # Update Vh
                    self.optimizers_vh[agent].zero_grad()
                    vh_loss.backward()
                    nn.utils.clip_grad_norm_(self.critics_vh[agent].parameters(), self.max_grad_norm_critic)
                    self.optimizers_vh[agent].step()
                    
                    vh_loss_total += vh_loss.item()
                
                # ========== Record statistics ==========
                train_info["loss_policy"].append(actor_loss.item())
                train_info["loss_value_vl"].append(vl_loss.item())
                train_info["loss_value_vh"].append(vh_loss_total / self.num_agents)
                train_info["loss_entropy"].append(entropy_loss.item())
                train_info["entropy"].append(dist_entropy.mean().item())
                
                # Approximate KL and clip fraction
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()
                    train_info["approx_kl"].append(approx_kl)
                    train_info["clip_fraction"].append(clip_frac)
                    
                    # Z statistics
                    train_info["z_mean"].append(z_flat.mean().item())
                    train_info["z_std"].append(z_flat.std().item())
        
        # Reset buffer
        self.buffer.after_update()
        
        # Average statistics
        update_info = {k: np.mean(v) for k, v in train_info.items()}
        
        # Add additional z statistics
        update_info["z_min"] = train_info["z_mean"][0] - 2 * train_info["z_std"][0] if train_info["z_std"] else 0
        update_info["z_max"] = train_info["z_mean"][0] + 2 * train_info["z_std"][0] if train_info["z_std"] else 0
        
        return update_info
    @torch.no_grad()
    def evaluate(self, num_episodes=10):
        """
        Evaluate policy with root finding for z*.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            eval_info: Dictionary with evaluation statistics
        """
        self.set_eval_mode()
        
        episode_returns = []
        episode_lengths = []
        z_global_values = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            self._init_rnn_states()
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # ========== Root finding for each agent ==========
                z_stars = {}
                for agent in self.agent_ids:
                    obs_agent = obs[agent]
                    rnn_vh_agent = self.rnn_states[agent]["vh"]
                    masks = torch.ones(self.num_envs, 1, device=self.device)
                    
                    # Define Vh function for root finding
                    def vh_fn(z, obs_a, rnn_a, mask_a):
                        z_enc = self.z_encoder(z)
                        vh, _ = self.critics_vh[agent](obs_a, z_enc, rnn_a, mask_a)
                        return vh
                    
                    z_star = self.root_finder.solve(vh_fn, obs_agent, rnn_vh_agent, masks)
                    z_stars[agent] = z_star
                
                # ========== Global z = max(z_i) ==========
                z_global = torch.max(torch.stack([z_stars[agent] for agent in self.agent_ids]), dim=0)[0]
                z_global_values.append(z_global.mean().item())
                
                # ========== Encode global z ==========
                z_enc_global = self.z_encoder(z_global)
                
                # ========== Actor forward (deterministic) ==========
                actions = {}
                for agent in self.agent_ids:
                    obs_agent = obs[agent]
                    masks = torch.ones(self.num_envs, 1, device=self.device)
                    
                    act, _, rnn_h = self.actors[agent](
                        obs_agent, z_enc_global,
                        self.rnn_states[agent]["actor"], masks,
                        deterministic=True
                    )
                    
                    actions[agent] = act
                    self.rnn_states[agent]["actor"] = rnn_h
                
                # Step environment
                obs, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # Accumulate statistics
                for agent in self.agent_ids:
                    episode_return += rewards[agent].mean().item()
                episode_length += 1
                
                # Check done
                done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for agent in self.agent_ids:
                    done_any |= (terminated[agent] | truncated[agent]).squeeze(-1)
                
                if done_any.any():
                    done = True
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        eval_info = {
            "return_mean": np.mean(episode_returns),
            "return_std": np.std(episode_returns),
            "episode_length": np.mean(episode_lengths),
            "success_rate": 0.0,  # TODO: implement based on task completion
            "z_global_mean": np.mean(z_global_values),
            "z_global_std": np.std(z_global_values),
        }
        
        return eval_info
    
    def save_checkpoint(self, path: str):
        """
        Save all networks and optimizer states.
        
        Checkpoint format (unified):
        {
            "z_encoder": state_dict,
            "actor_human": state_dict,
            "actor_robot": state_dict,
            "critic_vl": state_dict,  # Shared!
            "critic_vh_human": state_dict,
            "critic_vh_robot": state_dict,
            "optimizer_actor": state_dict,
            "optimizer_vl": state_dict,
            "optimizer_vh_human": state_dict,
            "optimizer_vh_robot": state_dict,
            "global_step": int,
            "episodes_done": int,
        }
        """
        checkpoint = {
            "z_encoder": self.z_encoder.state_dict(),
            "critic_vl": self.critic_vl.state_dict(),  # Centralized
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_vl": self.optimizer_vl.state_dict(),
            "global_step": self.global_step,
            "episodes_done": self.episodes_done,
        }
        
        # Add per-agent networks
        for agent in self.agent_ids:
            checkpoint[f"actor_{agent}"] = self.actors[agent].state_dict()
            checkpoint[f"critic_vh_{agent}"] = self.critics_vh[agent].state_dict()
            checkpoint[f"optimizer_vh_{agent}"] = self.optimizers_vh[agent].state_dict()
        
        torch.save(checkpoint, path)
        print(f"[EPIGRAPH] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load all networks and optimizer states."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load shared networks
        self.z_encoder.load_state_dict(checkpoint["z_encoder"])
        self.critic_vl.load_state_dict(checkpoint["critic_vl"])
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_vl.load_state_dict(checkpoint["optimizer_vl"])
        
        # Load per-agent networks
        for agent in self.agent_ids:
            self.actors[agent].load_state_dict(checkpoint[f"actor_{agent}"])
            self.critics_vh[agent].load_state_dict(checkpoint[f"critic_vh_{agent}"])
            self.optimizers_vh[agent].load_state_dict(checkpoint[f"optimizer_vh_{agent}"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.episodes_done = checkpoint["episodes_done"]
        
        print(f"[EPIGRAPH] Checkpoint loaded from {path}")
        print(f"[EPIGRAPH] Resumed at step {self.global_step}, episodes {self.episodes_done}")