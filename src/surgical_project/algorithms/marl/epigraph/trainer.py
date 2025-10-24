# SPDX-License-Identifier: BSD-3-Clause
"""
Epigraph Trainer - Complete training and evaluation pipeline
Features:
- Training: per-agent z_i recursion without root-finding
- Evaluation: root-finding → max → broadcast z_global
- Dual-path GAE: task and safe rewards
- PPO update with composite advantage: A = A_task - λ·A_safe
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np

from surgical_project.algorithms.marl.epigraph.epigraph_core import (
    ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder
)
from surgical_project.algorithms.marl.epigraph.rollout_buffer_z import RolloutBufferZ
from surgical_project.algorithms.marl.epigraph.utils import init_z, clip_z, clip_value_loss


class EpigraphTrainer:
    """
    Epigraph trainer aligned with rMAPPO interface.
    
    Key differences from rMAPPO:
    - Dual critics: Vl (centralized task) and Vh (decentralized safe)
    - Training: per-agent z_i dynamics without root-finding
    - Evaluation: root-finding per agent → take max → broadcast z_global
    """
    
    def __init__(self, env, device, algo_cfg: dict, epi_cfg: dict):
        """
        Args:
            env: Multi-agent environment
            device: torch.device
            algo_cfg: algorithms.rmappo config (lr, clip, gamma, etc.)
            epi_cfg: epigraph config (z range, encoding, lambda_safe)
        """
        self.env = env
        self.device = device
        self.algo_cfg = algo_cfg
        self.epi_cfg = epi_cfg
        
        # Environment info
        self.num_envs = env.num_envs
        self.agent_ids = env.cfg.possible_agents
        self.num_agents = len(self.agent_ids)
        
        # Observation/action dimensions
        self.obs_dim = 6
        self.share_obs_dim = 12  # concat([obs_robot, obs_human])
        self.action_dim = 3
        
        # Parse hyperparameters
        self._parse_config()
        
        # Build networks
        self._build_networks()
        self._build_optimizers()
        self._build_buffer()
        
        # Initialize RNN states
        self._init_rnn_states()
        
        # Initialize z
        self._init_z()
        
        # Root finder for evaluation
        self.root_finder = RootFinder(
            z_min=self.z_min,
            z_max=self.z_max,
            n_iters=20,
            device=self.device
        )
        
        self.global_step = 0
        
        print(f"[TRAINER] EpigraphTrainer initialized")
        print(f"  Agents: {self.agent_ids}")
        print(f"  Z range: [{self.z_min}, {self.z_max}], nz: {self.nz}")
        print(f"  Lambda safe: {self.lambda_safe}")
    
    def _parse_config(self):
        """Parse hyperparameters from config dicts."""
        # rMAPPO hyperparameters
        self.gamma = self.algo_cfg.get("gamma", 0.99)
        self.gae_lambda = self.algo_cfg.get("gae_lambda", 0.95)
        self.clip_param = self.algo_cfg.get("clip_param", 0.1)
        self.entropy_coef = self.algo_cfg.get("entropy_coef", 0.01)
        self.actor_lr = self.algo_cfg.get("actor_lr", 3e-4)
        self.critic_lr = self.algo_cfg.get("critic_lr", 1e-3)
        self.max_grad_norm_actor = self.algo_cfg.get("max_grad_norm_actor", 5.0)
        self.max_grad_norm_critic = self.algo_cfg.get("max_grad_norm_critic", 10.0)
        self.hidden_size = self.algo_cfg.get("hidden_size", 256)
        self.recurrent_N = self.algo_cfg.get("recurrent_N", 1)
        self.use_orthogonal = self.algo_cfg.get("use_orthogonal", True)
        self.gain = self.algo_cfg.get("gain", 0.01)
        self.huber_delta = self.algo_cfg.get("huber_delta", 1.0)
        
        # Epigraph-specific parameters
        z_cfg = self.epi_cfg["z"]
        self.z_min = z_cfg["min"]
        self.z_max = z_cfg["max"]
        self.z_init_mode = z_cfg["init"]["mode"]
        self.z_init_p_extreme = z_cfg["init"]["p_extreme"]
        self.nz = z_cfg["encode"]["nz"]
        self.z_mean = z_cfg["encode"]["mean"]
        self.z_scale = z_cfg["encode"]["scale"]
        
        losses_cfg = self.epi_cfg["losses"]
        self.lambda_safe = losses_cfg["lambda_safe"]
    
    def _build_networks(self):
        """Build networks: ZEncoder, Actor (per-agent), Vl (shared), Vh (per-agent)."""
        # Z Encoder (shared across all agents)
        self.z_encoder = ZEncoder(
            nz=self.nz,
            z_mean=self.z_mean,
            z_scale=self.z_scale
        ).to(self.device)
        
        # Per-agent Actor & Critic Vh
        self.policies = {}
        for agent in self.agent_ids:
            from gym.spaces import Box
            action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            
            actor = ActorRNN(
                obs_dim=self.obs_dim,
                nz=self.nz,
                action_dim=self.action_dim,
                hidden_size=self.hidden_size,
                recurrent_N=self.recurrent_N,
                use_orthogonal=self.use_orthogonal,
                gain=self.gain,
                action_space=action_space
            ).to(self.device)
            
            critic_h = CriticVhRNN(
                obs_dim=self.obs_dim,
                nz=self.nz,
                hidden_size=self.hidden_size,
                recurrent_N=self.recurrent_N,
                use_orthogonal=self.use_orthogonal,
                gain=self.gain
            ).to(self.device)
            
            self.policies[agent] = {
                "actor": actor,
                "critic_h": critic_h
            }
        
        # Critic Vl (centralized, shared)
        self.critic_l = CriticVlRNN(
            share_obs_dim=self.share_obs_dim,
            nz=self.nz,
            hidden_size=self.hidden_size,
            recurrent_N=self.recurrent_N,
            use_orthogonal=self.use_orthogonal,
            gain=self.gain
        ).to(self.device)
    
    def _build_optimizers(self):
        """Build optimizers for actor and critic networks."""
        # Actor optimizer (all actors + z_encoder)
        actor_params = []
        for agent in self.agent_ids:
            actor_params += list(self.policies[agent]["actor"].parameters())
        actor_params += list(self.z_encoder.parameters())
        
        self.actor_optimizer = optim.Adam(
            actor_params,
            lr=self.actor_lr,
            eps=self.algo_cfg.get("opt_eps", 1e-5)
        )
        
        # Critic optimizer (Vl + all Vh)
        critic_params = list(self.critic_l.parameters())
        for agent in self.agent_ids:
            critic_params += list(self.policies[agent]["critic_h"].parameters())
        
        self.critic_optimizer = optim.Adam(
            critic_params,
            lr=self.critic_lr,
            eps=self.algo_cfg.get("opt_eps", 1e-5)
        )
    
    def _build_buffer(self):
        """Build rollout buffer."""
        T = self.algo_cfg.get("rollout_horizon", 256)
        N = self.num_envs * self.num_agents
        
        self.buffer = RolloutBufferZ(
            T=T,
            N=N,
            obs_dim=self.obs_dim,
            share_obs_dim=self.share_obs_dim,
            act_dim=self.action_dim,
            rnn_hidden_dim=self.hidden_size,
            device=self.device
        )
    
    def _init_rnn_states(self):
        """Initialize RNN states for all networks."""
        self.rnn_states = {}
        for agent in self.agent_ids:
            self.rnn_states[agent] = {
                "actor": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
                "critic_h": torch.zeros(self.num_envs, self.hidden_size, device=self.device),
            }
        
        # Vl's RNN state (shared but per-env)
        self.rnn_state_critic_l = torch.zeros(self.num_envs, self.hidden_size, device=self.device)
    
    def _init_z(self):
        """Initialize per-agent z values for training."""
        self.z = {}
        for agent in self.agent_ids:
            self.z[agent] = init_z(
                mode=self.z_init_mode,
                p_extreme=self.z_init_p_extreme,
                z_min=self.z_min,
                z_max=self.z_max,
                shape=(self.num_envs, 1),
                device=self.device
            )
    
    # ========================================================================
    # collect_rollout() - Training data collection
    # ========================================================================
    
    def collect_rollout(self, rollout_horizon: int) -> Dict:
        """
        Collect T steps of experience (training mode: per-agent z_i dynamics).
        
        Args:
            rollout_horizon: Number of steps T
        
        Returns:
            rollout_info: Dict with episode statistics
        """
        T = rollout_horizon
        
        # Episode statistics
        episode_returns = {agent: [] for agent in self.agent_ids}
        episode_returns_task = {agent: [] for agent in self.agent_ids}
        episode_returns_safe = {agent: [] for agent in self.agent_ids}
        episode_lengths = []
        
        current_episode_returns = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.agent_ids}
        current_episode_returns_task = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.agent_ids}
        current_episode_returns_safe = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.agent_ids}
        current_episode_lengths = torch.zeros(self.num_envs, device=self.device)
        
        # Collect T steps
        for t in range(T):
            with torch.no_grad():
                # === 1. Get current observations ===
                obs_dict = self.env.unwrapped.obs_buf  # {"human": [N, 6], "robot": [N, 6]}
                
                # Construct share_obs (fixed order: robot, human)
                share_obs = torch.cat([obs_dict["robot"], obs_dict["human"]], dim=-1)  # [N, 12]
                
                # === 2. Encode z ===
                z_enc = {}
                for agent in self.agent_ids:
                    z_enc[agent] = self.z_encoder(self.z[agent])  # [N, nz]
                
                # For training Vl: use max(z_i) as global approximation
                z_train_global = torch.max(
                    torch.cat([self.z[agent] for agent in self.agent_ids], dim=-1),
                    dim=-1, keepdim=True
                )[0]  # [N, 1]
                z_enc_train_global = self.z_encoder(z_train_global)  # [N, nz]
                
                # === 3. Actor forward ===
                actions_dict = {}
                log_probs_dict = {}
                
                for agent in self.agent_ids:
                    obs_i = obs_dict[agent]
                    z_enc_i = z_enc[agent]
                    rnn_state_actor = self.rnn_states[agent]["actor"]
                    masks = torch.ones(self.num_envs, 1, device=self.device)
                    
                    actions, log_probs, rnn_state_actor_new = self.policies[agent]["actor"](
                        obs=obs_i,
                        z_enc=z_enc_i,
                        rnn_state=rnn_state_actor,
                        masks=masks,
                        deterministic=False
                    )
                    
                    actions_dict[agent] = actions
                    log_probs_dict[agent] = log_probs
                    self.rnn_states[agent]["actor"] = rnn_state_actor_new
                
                # === 4. Critic forward ===
                # Vl (centralized)
                masks_critic_l = torch.ones(self.num_envs, 1, device=self.device)
                value_l, rnn_state_critic_l_new = self.critic_l(
                    share_obs=share_obs,
                    z_enc=z_enc_train_global,
                    rnn_state=self.rnn_state_critic_l,
                    masks=masks_critic_l
                )
                self.rnn_state_critic_l = rnn_state_critic_l_new
                
                # Vh (decentralized)
                values_h_dict = {}
                for agent in self.agent_ids:
                    obs_i = obs_dict[agent]
                    z_enc_i = z_enc[agent]
                    rnn_state_critic_h = self.rnn_states[agent]["critic_h"]
                    masks_critic_h = torch.ones(self.num_envs, 1, device=self.device)
                    
                    value_h, rnn_state_critic_h_new = self.policies[agent]["critic_h"](
                        obs=obs_i,
                        z_enc=z_enc_i,
                        rnn_state=rnn_state_critic_h,
                        masks=masks_critic_h
                    )
                    
                    values_h_dict[agent] = value_h
                    self.rnn_states[agent]["critic_h"] = rnn_state_critic_h_new
            
            # === 5. Environment step ===
            obs_dict_new, rewards_dict, terminated_dict, truncated_dict, info = self.env.step(actions_dict)
            
            # Extract dual-path rewards
            r_task_dict = info["r_task"]
            r_safe_dict = info["r_safe"]
            
            # Ensure [N, 1] shape
            for agent in self.agent_ids:
                if r_task_dict[agent].dim() == 1:
                    r_task_dict[agent] = r_task_dict[agent].unsqueeze(-1)
                if r_safe_dict[agent].dim() == 1:
                    r_safe_dict[agent] = r_safe_dict[agent].unsqueeze(-1)
                if rewards_dict[agent].dim() == 1:
                    rewards_dict[agent] = rewards_dict[agent].unsqueeze(-1)
            
            # === 6. Update z (training mode: per-agent dynamics) ===
            for agent in self.agent_ids:
                self.z[agent] = clip_z(
                    (self.z[agent] + r_safe_dict[agent]) / self.gamma,
                    self.z_min,
                    self.z_max
                )
            
            # === 7. Construct masks ===
            dones_dict = {agent: (terminated_dict[agent] | truncated_dict[agent]) for agent in self.agent_ids}
            masks_dict = {agent: (~dones_dict[agent]).float().unsqueeze(-1) for agent in self.agent_ids}
            term_masks_dict = {agent: truncated_dict[agent].float().unsqueeze(-1) for agent in self.agent_ids}
            
            # === 8. Write to buffer (flattened by agent) ===
            # Stack order: [robot, human]
            obs_flat = torch.cat([obs_dict[agent] for agent in self.agent_ids], dim=0)
            share_obs_flat = share_obs.repeat(self.num_agents, 1)
            actions_flat = torch.cat([actions_dict[agent] for agent in self.agent_ids], dim=0)
            log_probs_flat = torch.cat([log_probs_dict[agent] for agent in self.agent_ids], dim=0)
            rewards_flat = torch.cat([rewards_dict[agent] for agent in self.agent_ids], dim=0)
            r_task_flat = torch.cat([r_task_dict[agent] for agent in self.agent_ids], dim=0)
            r_safe_flat = torch.cat([r_safe_dict[agent] for agent in self.agent_ids], dim=0)
            masks_flat = torch.cat([masks_dict[agent] for agent in self.agent_ids], dim=0)
            term_masks_flat = torch.cat([term_masks_dict[agent] for agent in self.agent_ids], dim=0)
            zs_flat = torch.cat([self.z[agent] for agent in self.agent_ids], dim=0)
            values_l_flat = value_l.repeat(self.num_agents, 1)
            values_h_flat = torch.cat([values_h_dict[agent] for agent in self.agent_ids], dim=0)
            rnn_states_actor_flat = torch.cat([self.rnn_states[agent]["actor"] for agent in self.agent_ids], dim=0)
            rnn_states_critic_l_flat = self.rnn_state_critic_l.repeat(self.num_agents, 1)
            rnn_states_critic_h_flat = torch.cat([self.rnn_states[agent]["critic_h"] for agent in self.agent_ids], dim=0)
            
            self.buffer.insert(
                t=t,
                obs=obs_flat,
                share_obs=share_obs_flat,
                actions=actions_flat,
                action_log_probs=log_probs_flat,
                rewards=rewards_flat,
                masks=masks_flat,
                rnn_states_actor=rnn_states_actor_flat,
                term_masks=term_masks_flat,
                zs=zs_flat,
                r_task=r_task_flat,
                r_safe=r_safe_flat,
                values_l=values_l_flat,
                values_h=values_h_flat,
                rnn_states_critic_l=rnn_states_critic_l_flat,
                rnn_states_critic_h=rnn_states_critic_h_flat,
            )
            
            # === 9. Reset done envs ===
            for agent in self.agent_ids:
                done_mask = dones_dict[agent]
                
                self.rnn_states[agent]["actor"][done_mask] = 0.0
                self.rnn_states[agent]["critic_h"][done_mask] = 0.0
                
                if done_mask.any():
                    n_reset = done_mask.sum().item()
                    self.z[agent][done_mask] = init_z(
                        mode=self.z_init_mode,
                        p_extreme=self.z_init_p_extreme,
                        z_min=self.z_min,
                        z_max=self.z_max,
                        shape=(n_reset, 1),
                        device=self.device
                    )
            
            any_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent in self.agent_ids:
                any_done = any_done | dones_dict[agent]
            self.rnn_state_critic_l[any_done] = 0.0
            
            # === 10. Episode statistics ===
            for agent in self.agent_ids:
                current_episode_returns[agent] += rewards_dict[agent].squeeze(-1)
                current_episode_returns_task[agent] += r_task_dict[agent].squeeze(-1)
                current_episode_returns_safe[agent] += r_safe_dict[agent].squeeze(-1)
            
            current_episode_lengths += 1
            
            for agent in self.agent_ids:
                done_mask = dones_dict[agent]
                if done_mask.any():
                    episode_returns[agent].extend(current_episode_returns[agent][done_mask].cpu().tolist())
                    episode_returns_task[agent].extend(current_episode_returns_task[agent][done_mask].cpu().tolist())
                    episode_returns_safe[agent].extend(current_episode_returns_safe[agent][done_mask].cpu().tolist())
                    
                    current_episode_returns[agent][done_mask] = 0.0
                    current_episode_returns_task[agent][done_mask] = 0.0
                    current_episode_returns_safe[agent][done_mask] = 0.0
            
            any_done_cpu = any_done.cpu()
            if any_done_cpu.any():
                episode_lengths.extend(current_episode_lengths[any_done_cpu].tolist())
                current_episode_lengths[any_done] = 0
        
        # === Aggregate statistics ===
        rollout_info = {}
        
        for agent in self.agent_ids:
            if len(episode_returns[agent]) > 0:
                rollout_info[f"return_mean_{agent}"] = np.mean(episode_returns[agent])
                rollout_info[f"return_task_mean_{agent}"] = np.mean(episode_returns_task[agent])
                rollout_info[f"return_safe_mean_{agent}"] = np.mean(episode_returns_safe[agent])
        
        if len(episode_lengths) > 0:
            rollout_info["episode_length_mean"] = np.mean(episode_lengths)
        
        # Z statistics (training: per-agent z_i)
        z_all = torch.cat([self.z[agent] for agent in self.agent_ids], dim=0)
        rollout_info["z_mean"] = float(z_all.mean().item())
        rollout_info["z_std"] = float(z_all.std().item())
        
        return rollout_info
    
    # ========================================================================
    # update() - PPO update with dual-path GAE
    # ========================================================================
    
    def update(
        self,
        ppo_epoch: int,
        num_mini_batch: int,
        clip_param: float,
        entropy_coef: float
    ) -> Dict:
        """
        PPO update with composite advantage: A = A_task - λ·A_safe.
        
        Args:
            ppo_epoch: Number of PPO epochs
            num_mini_batch: Number of mini-batches
            clip_param: PPO clipping parameter
            entropy_coef: Entropy coefficient
        
        Returns:
            update_info: Dict with loss metrics
        """
        # === 1. Bootstrap and compute GAE ===
        with torch.no_grad():
            obs_dict = self.env.unwrapped.obs_buf
            share_obs = torch.cat([obs_dict["robot"], obs_dict["human"]], dim=-1)
            
            # Encode z
            z_enc = {}
            for agent in self.agent_ids:
                z_enc[agent] = self.z_encoder(self.z[agent])
            
            z_train_global = torch.max(
                torch.cat([self.z[agent] for agent in self.agent_ids], dim=-1),
                dim=-1, keepdim=True
            )[0]
            z_enc_train_global = self.z_encoder(z_train_global)
            
            # Vl
            masks = torch.ones(self.num_envs, 1, device=self.device)
            last_value_l, _ = self.critic_l(
                share_obs=share_obs,
                z_enc=z_enc_train_global,
                rnn_state=self.rnn_state_critic_l,
                masks=masks
            )
            
            # Vh
            last_values_h_dict = {}
            for agent in self.agent_ids:
                last_value_h, _ = self.policies[agent]["critic_h"](
                    obs=obs_dict[agent],
                    z_enc=z_enc[agent],
                    rnn_state=self.rnn_states[agent]["critic_h"],
                    masks=masks
                )
                last_values_h_dict[agent] = last_value_h
            
            # Flatten
            last_value_l_flat = last_value_l.repeat(self.num_agents, 1)
            last_value_h_flat = torch.cat([last_values_h_dict[agent] for agent in self.agent_ids], dim=0)
            
            # Compute GAE
            self.buffer.compute_returns_and_adv(
                last_values_l=last_value_l_flat,
                last_values_h=last_value_h_flat,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
        
        # === 2. PPO iterations ===
        update_info = {
            "loss_policy": 0.0,
            "loss_value_vl": 0.0,
            "loss_value_vh": 0.0,
            "entropy_mean": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }
        
        n_updates = 0
        
        for epoch in range(ppo_epoch):
            data_generator = self.buffer.recurrent_generator(
                num_mini_batch=num_mini_batch,
                data_chunk_length=self.algo_cfg.get("data_chunk_length", 16),
                generator=None
            )
            
            for batch in data_generator:
                n_updates += 1
                
                # Unpack batch
                obs = batch["obs"]  # [L, B, obs_dim]
                share_obs = batch["share_obs"]  # [L, B, share_obs_dim]
                actions = batch["actions"]
                old_log_probs = batch["action_log_probs"]
                zs = batch["zs"]
                values_l_old = batch["values_l"]
                values_h_old = batch["values_h"]
                returns_task = batch["returns_task"]
                returns_safe = batch["returns_safe"]
                advantages_task = batch["advantages_task"]  # Already normalized
                advantages_safe = batch["advantages_safe"]  # Already normalized
                masks = batch["masks"]
                rnn_states_actor = batch["rnn_states_actor"]
                rnn_states_critic_l = batch["rnn_states_critic_l"]
                rnn_states_critic_h = batch["rnn_states_critic_h"]
                
                L, B = obs.shape[0], obs.shape[1]
                
                # === 3. Composite advantage ===
                advantages = advantages_task - self.lambda_safe * advantages_safe  # [L, B, 1]
                
                # === 4. Encode z ===
                z_enc = self.z_encoder(zs.view(L * B, 1))  # [L*B, nz]
                
                # For Vl: approximate global z as max within batch
                z_global_batch = zs.view(L, B, 1).max(dim=1, keepdim=True)[0]  # [L, 1, 1]
                z_global_batch = z_global_batch.expand(L, B, 1).reshape(L * B, 1)
                z_enc_global_batch = self.z_encoder(z_global_batch)  # [L*B, nz]
                
                # === 5. Flatten ===
                obs_flat = obs.view(L * B, -1)
                share_obs_flat = share_obs.view(L * B, -1)
                actions_flat = actions.view(L * B, -1)
                masks_flat = masks.view(L * B, 1)
                old_log_probs_flat = old_log_probs.view(L * B, 1)
                advantages_flat = advantages.view(L * B, 1)
                
                # === 6. Actor forward (use first agent temporarily) ===
                # TODO: Properly handle multi-agent by splitting samples
                agent_name = self.agent_ids[0]  # Temporary: use first agent
                new_log_probs, entropy, _ = self.policies[agent_name]["actor"].evaluate_actions(
                    obs=obs_flat,
                    z_enc=z_enc,
                    rnn_state=rnn_states_actor,
                    masks=masks_flat,
                    actions=actions_flat
                )
                
                # === 7. Policy loss ===
                ratio = torch.exp(new_log_probs - old_log_probs_flat)
                surr1 = ratio * advantages_flat
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages_flat
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # === 8. Value loss ===
                # Vl
                new_value_l, _ = self.critic_l(
                    share_obs=share_obs_flat,
                    z_enc=z_enc_global_batch,
                    rnn_state=rnn_states_critic_l,
                    masks=masks_flat
                )
                
                value_loss_l = clip_value_loss(
                    value_pred=new_value_l,
                    value_target=returns_task.view(L * B, 1),
                    old_value_pred=values_l_old.view(L * B, 1),
                    clip_param=clip_param,
                    huber_delta=self.huber_delta,
                    masks=masks_flat
                )
                
                # Vh (use first agent temporarily)
                new_value_h, _ = self.policies[agent_name]["critic_h"](
                    obs=obs_flat,
                    z_enc=z_enc,
                    rnn_state=rnn_states_critic_h,
                    masks=masks_flat
                )
                
                value_loss_h = clip_value_loss(
                    value_pred=new_value_h,
                    value_target=returns_safe.view(L * B, 1),
                    old_value_pred=values_h_old.view(L * B, 1),
                    clip_param=clip_param,
                    huber_delta=self.huber_delta,
                    masks=masks_flat
                )
                
                # === 9. Total loss ===
                total_loss = policy_loss - entropy_coef * entropy + value_loss_l + value_loss_h
                
                # === 10. Backward ===
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                actor_params = []
                for agent in self.agent_ids:
                    actor_params += list(self.policies[agent]["actor"].parameters())
                actor_params += list(self.z_encoder.parameters())
                torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm_actor)
                
                critic_params = list(self.critic_l.parameters())
                for agent in self.agent_ids:
                    critic_params += list(self.policies[agent]["critic_h"].parameters())
                torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm_critic)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # === 11. Log statistics ===
                with torch.no_grad():
                    update_info["loss_policy"] += float(policy_loss.item())
                    update_info["loss_value_vl"] += float(value_loss_l.item())
                    update_info["loss_value_vh"] += float(value_loss_h.item())
                    update_info["entropy_mean"] += float(entropy.item())
                    
                    clip_fraction = float(((ratio - 1.0).abs() > clip_param).float().mean().item())
                    update_info["clip_fraction"] += clip_fraction
                    
                    approx_kl = float(((ratio - 1.0) - (ratio.log())).mean().item())
                    update_info["approx_kl"] += approx_kl
        
        # Average
        for key in update_info:
            update_info[key] /= max(n_updates, 1)
        
        # Add learning rates
        update_info["lr_actor"] = self.actor_optimizer.param_groups[0]["lr"]
        update_info["lr_critic"] = self.critic_optimizer.param_groups[0]["lr"]
        
        # Reset buffer
        self.buffer.after_update()
        
        return update_info
    
    # ========================================================================
    # evaluate() - Evaluation with root-finding
    # ========================================================================
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate with root-finding: z* per agent → max → broadcast z_global.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            eval_info: Dict with evaluation metrics
        """
        self.set_eval_mode(True)
        
        episode_returns = []
        episode_returns_task = []
        episode_returns_safe = []
        z_global_history = []
        
        for ep in range(num_episodes):
            obs_dict, _ = self.env.reset()
            
            # Reset RNN states
            for agent in self.agent_ids:
                self.rnn_states[agent]["actor"].zero_()
                self.rnn_states[agent]["critic_h"].zero_()
            self.rnn_state_critic_l.zero_()
            
            done = False
            episode_return = 0.0
            episode_return_task = 0.0
            episode_return_safe = 0.0
            
            while not done:
                with torch.no_grad():
                    # === 1. Root-finding (evaluation mode) ===
                    z_stars = {}
                    for agent in self.agent_ids:
                        obs_i = obs_dict[agent]
                        rnn_state_critic_h = self.rnn_states[agent]["critic_h"]
                        masks = torch.ones(self.num_envs, 1, device=self.device)
                        
                        z_star = self.root_finder.solve_with_encoder(
                            obs=obs_i,
                            z_encoder=self.z_encoder,
                            critic_vh=self.policies[agent]["critic_h"],
                            rnn_state=rnn_state_critic_h,
                            mask=masks
                        )
                        
                        z_stars[agent] = z_star
                    
                    # === 2. Take max → z_global ===
                    z_global = torch.max(
                        torch.cat([z_stars[agent] for agent in self.agent_ids], dim=-1),
                        dim=-1, keepdim=True
                    )[0]
                    
                    z_global_history.append(float(z_global.mean().item()))
                    
                    # === 3. Broadcast encoding ===
                    z_enc_global = self.z_encoder(z_global)
                    
                    # === 4. Actor forward (all use z_enc_global) ===
                    actions_dict = {}
                    for agent in self.agent_ids:
                        obs_i = obs_dict[agent]
                        rnn_state_actor = self.rnn_states[agent]["actor"]
                        masks = torch.ones(self.num_envs, 1, device=self.device)
                        
                        actions, _, rnn_state_actor_new = self.policies[agent]["actor"](
                            obs=obs_i,
                            z_enc=z_enc_global,  # Broadcast
                            rnn_state=rnn_state_actor,
                            masks=masks,
                            deterministic=True
                        )
                        
                        actions_dict[agent] = actions
                        self.rnn_states[agent]["actor"] = rnn_state_actor_new
                
                # === 5. Step ===
                obs_dict, rewards_dict, terminated_dict, truncated_dict, info = self.env.step(actions_dict)
                
                r_task_dict = info["r_task"]
                r_safe_dict = info["r_safe"]
                
                for agent in self.agent_ids:
                    episode_return += float(rewards_dict[agent].mean().item())
                    episode_return_task += float(r_task_dict[agent].mean().item())
                    episode_return_safe += float(r_safe_dict[agent].mean().item())
                
                done = any((terminated_dict[agent] | truncated_dict[agent]).any() for agent in self.agent_ids)
            
            episode_returns.append(episode_return)
            episode_returns_task.append(episode_return_task)
            episode_returns_safe.append(episode_return_safe)
        
        self.set_eval_mode(False)
        
        eval_info = {
            "return_mean": np.mean(episode_returns),
            "return_task_mean": np.mean(episode_returns_task),
            "return_safe_mean": np.mean(episode_returns_safe),
            "z_global_mean": np.mean(z_global_history),
            "z_global_std": np.std(z_global_history),
        }
        
        return eval_info
    
    # ========================================================================
    # Checkpoint management
    # ========================================================================
    
    def save_checkpoint(self, path: str, global_step: int):
        """Save checkpoint."""
        ckpt = {
            "global_step": global_step,
            "z_encoder": self.z_encoder.state_dict(),
            "critic_l": self.critic_l.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        
        for agent in self.agent_ids:
            ckpt[f"{agent}_actor"] = self.policies[agent]["actor"].state_dict()
            ckpt[f"{agent}_critic_h"] = self.policies[agent]["critic_h"].state_dict()
        
        torch.save(ckpt, path)
        print(f"[TRAINER] Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.z_encoder.load_state_dict(ckpt["z_encoder"])
        self.critic_l.load_state_dict(ckpt["critic_l"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        
        for agent in self.agent_ids:
            self.policies[agent]["actor"].load_state_dict(ckpt[f"{agent}_actor"])
            self.policies[agent]["critic_h"].load_state_dict(ckpt[f"{agent}_critic_h"])
        
        self.global_step = ckpt["global_step"]
        
        print(f"[TRAINER] Checkpoint loaded: {path}")
        print(f"  Global step: {self.global_step}")
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def set_eval_mode(self, eval_mode: bool):
        """Set evaluation or training mode."""
        if eval_mode:
            self.z_encoder.eval()
            self.critic_l.eval()
            for agent in self.agent_ids:
                self.policies[agent]["actor"].eval()
                self.policies[agent]["critic_h"].eval()
        else:
            self.z_encoder.train()
            self.critic_l.train()
            for agent in self.agent_ids:
                self.policies[agent]["actor"].train()
                self.policies[agent]["critic_h"].train()