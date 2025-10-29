"""
Epigraph Trainer for Safe Multi-Agent Reinforcement Learning.
With sequence-based RNN training (rMAPPO-aligned) and milestone-driven evaluation.

ROUTE B MODIFICATIONS:
1. Added full_config, ckpt_dir, max_global_steps to constructor
2. Added milestone tracking and maybe_milestone_eval_and_save()
3. Added run_single_eval_episode() for root_finder-based evaluation
4. Improved evaluate() to use run_single_eval_episode()
5. Added WandB logger integration
6. Added global_episodes tracking
7. collect_rollout() now returns proper statistics using summarize_rollout_stats()
8. update() uses buffer.compute_returns_and_advantages()
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import numpy as np
import traceback

from .epigraph_core import ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder
from .rollout_buffer_z import RolloutBufferZ


# Import training helpers
try:
    from scripts.utils.training_helpers_epigraph import (
        summarize_rollout_stats,
        summarize_eval_stats,
        WandBLogger,
    )
except ImportError:
    try:
        from surgical_project.scripts.utils.training_helpers_epigraph import (
            summarize_rollout_stats,
            summarize_eval_stats,
            WandBLogger,
        )
    except ImportError:
        print("[WARNING] Could not import training_helpers_epigraph, some features may be limited")
        summarize_rollout_stats = None
        summarize_eval_stats = None
        WandBLogger = None


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


def update_z_epigraph(
    z: torch.Tensor,
    reward: torch.Tensor,      # team reward per env, shape [E, 1]
    gamma: float,
    z_min: float,
    z_max: float,
):
    """
    Epigraph z dynamics aligned with the source rollout:
        z_{t+1} = (z_t + r_team_t) / gamma
    Then clamp to [z_min, z_max].
    """
    z_next = (z + reward) / gamma
    z_next = torch.clamp(z_next, z_min, z_max)
    return z_next


class EpigraphTrainer:
    """
    Epigraph MARL trainer with sequence-based RNN training and milestone-driven evaluation.
    
    This trainer acts as a unified "Runner + Updater + Evaluator" following Route B design.
    """
    
    def __init__(
        self,
        env,
        device: torch.device,
        algo_cfg: Dict[str, Any],
        epi_cfg: Dict[str, Any],
        full_config: Dict[str, Any],
        ckpt_dir: str,
        max_global_steps: int,
    ):
        """
        Initialize Epigraph trainer.
        
        Args:
            env: Multi-agent environment
            device: PyTorch device
            algo_cfg: Algorithm configuration (from config["algorithms"]["rmappo"])
            epi_cfg: Epigraph-specific configuration (from config["epigraph"])
            full_config: Complete YAML configuration dictionary
            ckpt_dir: Directory to save checkpoints
            max_global_steps: Maximum training steps
        """
        self.env = env
        self.device = device
        self.algo_cfg = algo_cfg
        self.epi_cfg = epi_cfg
        self.full_config = full_config
        self.ckpt_dir = ckpt_dir
        self.max_global_steps = max_global_steps
        
        # Extract environment params
        if hasattr(env.unwrapped, "params"):
            self.env_cfg = env.unwrapped.params.get("epigraph_env", {})
        else:
            self.env_cfg = {}
        
        # Environment dimensions
        self.num_envs = env.num_envs
        self.agent_ids = list(env.unwrapped.cfg.possible_agents)
        self.num_agents = len(self.agent_ids)
        
        self.obs_dim = env.unwrapped.cfg.observation_spaces[self.agent_ids[0]]
        self.share_obs_dim = self.obs_dim * self.num_agents
        self.act_dim = env.unwrapped.cfg.action_spaces[self.agent_ids[0]]
        self.hidden_size = algo_cfg["hidden_size"]
        self.recurrent_N = algo_cfg.get("recurrent_N", 1)
        
        # Training hyperparameters
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
        
        # Epigraph-specific hyperparameters
        self.z_min = epi_cfg["z"]["min"]
        self.z_max = epi_cfg["z"]["max"]
        self.z_nz = epi_cfg["z"]["encode"]["nz"]
        self.z_init_mode = epi_cfg["z"]["init"]["mode"]
        self.z_init_p_extreme = epi_cfg["z"]["init"]["p_extreme"]
        self.lambda_safe = epi_cfg["losses"]["lambda_safe"]
        
        # Build networks, buffer, optimizers
        self._build_networks()
        self._build_buffer()
        self._build_optimizers()
        self._init_rnn_states()
        
        # Training state
        self.global_step = 0
        self.global_episodes = 0
        self.episodes_done = 0  # Legacy name for compatibility
        
        # Milestone tracking (Route B)
        milestone_episodes = full_config.get("training_monitor", {}).get("milestone_episodes", [])
        self.milestones = deque(sorted(milestone_episodes))
        print(f"[TRAINER] Milestones configured: {list(self.milestones)}")
        
        # WandB logger (if enabled)
        self.wandb_logger = None
        if full_config.get("logging", {}).get("use_wandb", False) and WandBLogger is not None:
            wandb_cfg = full_config.get("logging", {}).get("wandb", {})
            self.wandb_logger = WandBLogger(
                project=wandb_cfg.get("project", "epigraph_training"),
                run_name=wandb_cfg.get("run_name", f"epigraph_{os.path.basename(ckpt_dir)}"),
                config=full_config,
                entity=wandb_cfg.get("entity", None),
            )
            if self.wandb_logger.enabled:
                print("[TRAINER] WandB logger initialized")
            else:
                print("[TRAINER] WandB logger disabled")

        self._cached_obs = None
        self._cached_z = None
        self._cached_masks = None
        self._needs_env_reset = True


    def _build_networks(self):
        """Build all networks."""
        # Z encoder
        self.z_encoder = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
        ).to(self.device)
        
        # Per-agent actors and safety critics
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
        
        # Centralized performance critic
        self.critic_vl = CriticVlRNN(
            share_obs_dim=self.share_obs_dim,
            hidden_size=self.hidden_size,
            nz=self.z_nz,
            recurrent_N=self.recurrent_N,
        ).to(self.device)
        
        # Root finder for safe z* computation (used in evaluation)
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
        # Actor optimizer includes z_encoder parameters
        actor_params = list(self.z_encoder.parameters())
        for agent in self.agent_ids:
            actor_params.extend(list(self.actors[agent].parameters()))
        
        self.optimizer_actor = optim.Adam(
            actor_params, lr=self.algo_cfg["actor_lr"], eps=self.algo_cfg["opt_eps"]
        )
        
        # Centralized Vl critic optimizer
        self.optimizer_vl = optim.Adam(
            self.critic_vl.parameters(), lr=self.algo_cfg["critic_lr"], eps=self.algo_cfg["opt_eps"]
        )
        
        # Per-agent Vh critic optimizers
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

    def _mark_env_reset_needed(self):
        """Flag that the environment should be fully reset before the next rollout."""
        self._needs_env_reset = True
        self._cached_obs = None
        self._cached_z = None
        self._cached_masks = None

        
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
        Collect one rollout of length self.rollout_horizon from all parallel envs.

        Returns:
            rollout_stats: Dictionary with statistics for logging
        """
        self.set_eval_mode()

        if self._needs_env_reset or self._cached_obs is None:
            obs, _ = self.env.reset()
            z_global = self._init_z_training()
            self._init_rnn_states()
            masks = torch.ones(self.num_envs, 1, device=self.device)
            self._needs_env_reset = False
        else:
            obs = {agent: self._cached_obs[agent].clone().detach() for agent in self.agent_ids}
            cached_z = self._cached_z
            if cached_z is None or cached_z.shape[0] != self.num_envs:
                z_global = self._init_z_training()
            else:
                z_global = cached_z.clone().detach()
            if self._cached_masks is not None:
                masks = self._cached_masks.clone().detach()
            else:
                masks = torch.ones(self.num_envs, 1, device=self.device)

        # Episode-level bookkeeping for logging
        episode_returns_task = []
        episode_returns_safe = []
        episode_lengths = []
        current_returns_task = torch.zeros(self.num_envs, device=self.device)
        current_returns_safe = torch.zeros(self.num_envs, device=self.device)
        current_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Track z values and violations for rollout stats
        z_values_rollout = []
        violations_rollout = []

        for t in range(self.rollout_horizon):
            # Encode z for all envs (shared z is broadcast to all agents)
            z_enc = self.z_encoder(z_global)  # [E, nz]
            z_values_rollout.append(z_global.clone())

            # Act for each agent
            actions = {}
            action_log_probs = {}
            for agent in self.agent_ids:
                act, logp, next_h_actor, _ = self.actors[agent].act_step(
                    obs[agent],                        # [E, obs_dim]
                    z_enc,                             # [E, nz]
                    self.rnn_states[agent]["actor"],   # [E, H]
                    masks,                             # [E, 1]
                    deterministic=False
                )
                actions[agent] = act
                action_log_probs[agent] = logp
                self.rnn_states[agent]["actor"] = next_h_actor

            # Centralized Vl critic uses concatenated obs from all agents
            share_obs = self._get_share_obs(obs)  # [E, A*obs_dim]
            vl, next_h_vl = self.critic_vl.value_step(
                share_obs,            # [E, share_obs_dim]
                z_enc,                # [E, nz]
                self.rnn_states_vl,   # [E, H]
                masks                 # [E, 1]
            )
            self.rnn_states_vl = next_h_vl

            # Per-agent Vh critics
            vh = {}
            for agent in self.agent_ids:
                vh_val, next_h_vh = self.critics_vh[agent].value_step(
                    obs[agent],                      # [E, obs_dim]
                    z_enc,                           # [E, nz]
                    self.rnn_states[agent]["vh"],    # [E, H]
                    masks                            # [E, 1]
                )
                vh[agent] = vh_val
                self.rnn_states[agent]["vh"] = next_h_vh

            # Env step
            obs_next, rewards, terminated, truncated, info = self.env.step(actions)

            # Pull training signals (dicts keyed by agent)
            r_task = info["r_task"]   # {agent: [E,1]} task part for Vl
            r_safe = info["r_safe"]   # {agent: [E,1]} safety cost part for Vh

            # Team reward = average over agents (shape [E,1])
            r_team = sum(r_task[a] for a in self.agent_ids) / self.num_agents

            # Track violations if available
            if "is_violating" in info and info["is_violating"] is not None:
                violations_rollout.append(info["is_violating"])

            # Figure out which envs ended this step
            term_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            trunc_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            done_any_env = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

            for agent in self.agent_ids:
                term_a = terminated[agent] if terminated[agent].dim() == 2 else terminated[agent].unsqueeze(-1)
                trunc_a = truncated[agent] if truncated[agent].dim() == 2 else truncated[agent].unsqueeze(-1)
                term_any_env |= term_a
                trunc_any_env |= trunc_a
                done_any_env |= (term_a | trunc_a)

            # Masks used for PPO/RNN:
            masks_flat_t = masks.repeat(self.num_agents, 1)  # [N,1] (N=E*A)
            term_mask_env = (~term_any_env).float()          # [E,1]
            term_masks_flat_t = term_mask_env.repeat(self.num_agents, 1)  # [N,1]

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

            # Ensure rewards have shape [E, 1]
            rewards_formatted = {}
            for agent in self.agent_ids:
                rew_agent = rewards[agent]
                if rew_agent.dim() == 1:
                    rew_agent = rew_agent.unsqueeze(-1)
                rewards_formatted[agent] = rew_agent

            # Write step t into rollout buffer (agent-major order)
            self.buffer.insert(
                obs=self._flatten_per_agent(obs),                           # [N, obs_dim]
                share_obs=self._replicate_global_for_agents(share_obs),     # [N, share_obs_dim]
                actions=self._flatten_per_agent(actions),                   # [N, act_dim]
                action_log_probs=self._flatten_per_agent(action_log_probs), # [N, 1]
                rewards=self._flatten_per_agent(rewards_formatted),         # [N, 1] raw env reward
                rewards_task=self._flatten_per_agent(r_task),               # [N, 1] task part
                costs_safe=self._flatten_per_agent(r_safe),                 # [N, 1] safety cost
                values_vl=self._replicate_global_for_agents(vl),            # [N, 1]
                values_vh=self._flatten_per_agent(vh),                      # [N, 1]
                z=self._replicate_global_for_agents(z_global),              # [N, 1]
                masks=masks_flat_t,                                         # [N, 1] alive going into t
                term_masks=term_masks_flat_t,                               # [N, 1] alive after t
                override_bootstrap_mask_vl=override_mask_vl_flat_t,         # [N, 1]
                override_bootstrap_vl=override_vl_flat_t,                   # [N, 1]
                override_bootstrap_mask_vh=override_mask_vh_flat_t,         # [N, 1]
                override_bootstrap_vh=override_vh_flat_t,                   # [N, 1]
                rnn_states_actor=torch.cat(
                    [self.rnn_states[a]["actor"] for a in self.agent_ids],
                    dim=0
                ),                                                          # [N, H]
                rnn_states_critic=self.rnn_states_vl.repeat(self.num_agents, 1),  # [N, H]
                rnn_states_vh=torch.cat(
                    [self.rnn_states[a]["vh"] for a in self.agent_ids],
                    dim=0
                ),                                                          # [N, H]
            )

            # Z update: source-style z_{t+1} = (z_t + r_team) / gamma
            z_global = update_z_epigraph(
                z_global,
                r_team,
                self.gamma,
                self.z_min,
                self.z_max,
            )

            # Episode return bookkeeping (for logging only)
            r_task_mean = sum(r_task[a].squeeze(-1) for a in self.agent_ids) / self.num_agents  # [E]
            r_safe_mean = sum(r_safe[a].squeeze(-1) for a in self.agent_ids) / self.num_agents  # [E]
            current_returns_task += r_task_mean
            current_returns_safe += r_safe_mean
            current_lengths += 1

            # Finalize stats for envs that ended
            if done_any_env.any():
                done_indices = done_any_env.squeeze(-1).nonzero(as_tuple=True)[0]
                episode_returns_task.extend(current_returns_task[done_indices].cpu().tolist())
                episode_returns_safe.extend(current_returns_safe[done_indices].cpu().tolist())
                episode_lengths.extend(current_lengths[done_indices].cpu().tolist())
                current_returns_task[done_indices] = 0.0
                current_returns_safe[done_indices] = 0.0
                current_lengths[done_indices] = 0
                self.episodes_done += len(done_indices)
                self.global_episodes += len(done_indices)

            # Advance loop state
            obs = obs_next
            masks = (~done_any_env).float()  # [E,1] 1 for still-running envs
            self.global_step += self.num_envs  # Count only *training* env steps

        # Bootstrap final values at t = T
        z_enc_last = self.z_encoder(z_global)  # [E, nz]
        share_obs_last = self._get_share_obs(obs)
        masks_last = torch.ones(self.num_envs, 1, device=self.device)

        vl_last, _ = self.critic_vl.value_step(
            share_obs_last,
            z_enc_last,
            self.rnn_states_vl,
            masks_last,
        )
        vh_last = {}
        for agent in self.agent_ids:
            vh_last[agent], _ = self.critics_vh[agent].value_step(
                obs[agent],
                z_enc_last,
                self.rnn_states[agent]["vh"],
                masks_last,
            )

        # Store bootstrap slice at T (for GAE computation)
        self.buffer.insert_final_step(
            obs=self._flatten_per_agent(obs),
            share_obs=self._replicate_global_for_agents(share_obs_last),
            values_vl=self._replicate_global_for_agents(vl_last),
            values_vh=self._flatten_per_agent(vh_last),
            z=self._replicate_global_for_agents(z_global),
        )

        # Compute rollout statistics using helper function (if available)
        if summarize_rollout_stats is not None:
            T = self.rollout_horizon
            E = self.num_envs
            A = self.num_agents

            r_task_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.rewards_task[:T], has_trailing_dim1=True
            ).squeeze(-1)  # [T, E, A]

            r_safe_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.costs_safe[:T], has_trailing_dim1=True
            ).squeeze(-1)  # [T, E, A]

            z_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.z[:T], has_trailing_dim1=True
            ).squeeze(-1)  # [T, E, A]

            term_masks_bool = (self.buffer.term_masks[:T] > 0.5)
            dones_tensor = self._reshape_agentmajor_to_env_agent_T(
                ~term_masks_bool, has_trailing_dim1=True
            ).squeeze(-1).any(dim=2).float()  # [T, E]

            info_for_stats = {}
            if len(violations_rollout) > 0:
                info_for_stats["is_violating"] = torch.stack(violations_rollout, dim=0)
            if "progress_ratio" in info:
                info_for_stats["progress_ratio"] = info["progress_ratio"]

            rollout_stats = summarize_rollout_stats(
                r_task=r_task_tensor,
                r_safe_cost=r_safe_tensor,
                z=z_tensor,
                dones=dones_tensor,
                info=info_for_stats,
                agent_labels=self.agent_ids,
            )
        else:
            rollout_stats = {
                "return_task_mean": np.mean(episode_returns_task) if episode_returns_task else 0.0,
                "return_task_std": np.std(episode_returns_task) if episode_returns_task else 0.0,
                "return_safe_mean": np.mean(episode_returns_safe) if episode_returns_safe else 0.0,
                "return_safe_std": np.std(episode_returns_safe) if episode_returns_safe else 0.0,
                "episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
                "episodes_finished": len(episode_returns_task),
                "z_mean": float(torch.stack(z_values_rollout).mean().item()) if z_values_rollout else 0.0,
                "z_std": float(torch.stack(z_values_rollout).std().item()) if z_values_rollout else 0.0,
            }

        # Cache state for next rollout
        self._cached_obs = {agent: obs[agent].clone().detach() for agent in self.agent_ids}
        self._cached_z = z_global.clone().detach()
        self._cached_masks = masks.clone().detach()

        return rollout_stats
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
        In milestone truncation, this will be used to set custom bootstrap values.
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
    
    def update(self) -> Dict[str, Any]:
        """
        Sequence-based PPO update with RNN training.
        
        Returns:
            update_stats: Dictionary with loss and training statistics
        """
        self.set_train_mode()
        
        # Let the buffer compute returns/advantages (handles normalization internally)
        self.buffer.compute_epigraph_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            lambda_safe=self.lambda_safe,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
        )
        
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
                # Unpack minibatch
                obs_flat        = mb["obs_flat"]          # [L*B, obs_dim]
                share_obs_flat  = mb["share_obs_flat"]    # [L*B, share_obs_dim]
                act_flat        = mb["act_flat"]          # [L*B, act_dim]
                old_logp_flat   = mb["old_logp_flat"]     # [L*B, 1]
                adv_flat        = mb["adv_flat"]          # [L*B, 1] (advantage after normalize)
                ret_vl_flat     = mb["ret_vl_flat"]       # [L*B, 1] target for Vl
                ret_vh_flat     = mb["ret_vh_flat"]       # [L*B, 1] target for Vh
                old_vl_flat     = mb["old_vl_flat"]       # [L*B, 1] old Vl predictions
                old_vh_flat     = mb["old_vh_flat"]       # [L*B, 1] old Vh predictions
                z_flat          = mb["z_flat"]            # [L*B, 1] z traj
                masks_flat      = mb["masks_flat"]        # [L*B, 1] RNN masks

                h0_actor        = mb["h0_actor"]          # [B, H] init RNN state for actor
                h0_vl           = mb["h0_vl"]             # [B, H] init RNN state for Vl critic
                h0_vh           = mb["h0_vh"]             # [B, H] init RNN state for Vh critic
                agent_idx       = mb["agent_idx"]         # [B] which agent each seq belongs to

                # Z encoding
                # z_enc goes into policy with grad,
                # critics see a detached copy (no grad to z_encoder from value losses)
                z_enc_flat = self.z_encoder(z_flat)
                z_enc_flat_detached = z_enc_flat.detach()

                # Centralized performance critic Vl (team value)
                vl_pred_flat, _ = self.critic_vl.value_seq(
                    share_obs_seq=share_obs_flat,
                    z_enc_seq=z_enc_flat_detached,
                    hxs_init=h0_vl,
                    masks_seq=masks_flat,
                )

                vl_clipped = old_vl_flat + torch.clamp(
                    vl_pred_flat - old_vl_flat,
                    -value_clip_eps,
                    value_clip_eps
                )
                vl_loss_unclipped = (vl_pred_flat - ret_vl_flat).pow(2)
                vl_loss_clipped   = (vl_clipped   - ret_vl_flat).pow(2)
                vl_loss = 0.5 * torch.max(vl_loss_unclipped, vl_loss_clipped).mean()

                # Per-agent policy and safety critic
                policy_loss_total = 0.0
                vh_loss_total     = 0.0
                entropy_total     = 0.0
                approx_kl_total   = 0.0
                clipfrac_total    = 0.0
                count_total       = 0.0

                B = h0_actor.size(0)  # number of sequences in this minibatch

                for a_i, agent in enumerate(self.agent_ids):
                    # Pick only the sequences that belong to this agent
                    seq_mask_b = (agent_idx == a_i)         # [B]
                    if not seq_mask_b.any():
                        continue

                    # Expand that mask across time steps
                    seq_mask_lb = seq_mask_b.unsqueeze(0).expand(L, B).reshape(L * B)

                    # Slice per-agent data
                    obs_a        = obs_flat[seq_mask_lb]
                    act_a        = act_flat[seq_mask_lb]
                    old_logp_a   = old_logp_flat[seq_mask_lb]
                    adv_a        = adv_flat[seq_mask_lb]
                    masks_a      = masks_flat[seq_mask_lb]
                    ret_vh_a     = ret_vh_flat[seq_mask_lb]
                    old_vh_a     = old_vh_flat[seq_mask_lb]

                    h0_actor_a   = h0_actor[seq_mask_b]
                    h0_vh_a      = h0_vh[seq_mask_b]

                    # Per-agent z features:
                    # actor sees non-detached z_enc; critic sees detached
                    z_enc_a      = z_enc_flat[seq_mask_lb]
                    z_enc_a_det  = z_enc_a.detach()

                    # Safety critic Vh (per agent)
                    vh_pred_a, _ = self.critics_vh[agent].value_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_a_det,
                        hxs_init=h0_vh_a,
                        masks_seq=masks_a,
                    )

                    vh_clipped_a = old_vh_a + torch.clamp(
                        vh_pred_a - old_vh_a,
                        -value_clip_eps,
                        value_clip_eps
                    )
                    vh_loss_unclipped_a = (vh_pred_a - ret_vh_a).pow(2)
                    vh_loss_clipped_a   = (vh_clipped_a - ret_vh_a).pow(2)
                    vh_loss_a = 0.5 * torch.max(vh_loss_unclipped_a, vh_loss_clipped_a).mean()

                    # Policy loss (per agent)
                    # evaluate_actions_seq returns new log prob and entropy under current policy
                    logp_new_a, entropy_a, _ = self.actors[agent].evaluate_actions_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_a,     # NOT detached -> actor + z_encoder get grad
                        hxs_init=h0_actor_a,
                        masks_seq=masks_a,
                        act_seq=act_a,
                    )

                    ratio = torch.exp(logp_new_a - old_logp_a)  # PPO ratio
                    surr1 = ratio * adv_a
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_a
                    policy_loss_a = -torch.min(surr1, surr2).mean() - ent_coef * entropy_a.mean()

                    with torch.no_grad():
                        # Approx KL and clip fraction for logging/debug
                        approx_kl_a  = ((ratio - 1) - torch.log(ratio)).mean()
                        clipfrac_a   = (torch.abs(ratio - 1) > clip_eps).float().mean()

                    # Accumulate stats across agents in this minibatch
                    policy_loss_total += policy_loss_a
                    vh_loss_total     += vh_loss_a
                    entropy_total     += entropy_a.mean()
                    approx_kl_total   += approx_kl_a
                    clipfrac_total    += clipfrac_a
                    count_total       += 1.0

                if count_total < 1e-8:
                    continue

                # Average across all agents that contributed data
                policy_loss = policy_loss_total / count_total
                vh_loss     = vh_loss_total     / count_total
                entropy     = entropy_total     / count_total
                approx_kl   = approx_kl_total   / count_total
                clipfrac    = clipfrac_total    / count_total

                # Backprop / update
                # Actor + shared z_encoder
                self.optimizer_actor.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.z_encoder.parameters()) +
                    [p for ag in self.agent_ids for p in self.actors[ag].parameters()],
                    self.max_grad_norm_actor
                )
                self.optimizer_actor.step()

                # Centralized Vl critic
                self.optimizer_vl.zero_grad()
                vl_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_vl.parameters(),
                    self.max_grad_norm_critic
                )
                self.optimizer_vl.step()

                # Per-agent Vh critics
                for ag in self.agent_ids:
                    self.optimizers_vh[ag].zero_grad()
                vh_loss.backward()
                for ag in self.agent_ids:
                    torch.nn.utils.clip_grad_norm_(
                        self.critics_vh[ag].parameters(),
                        self.max_grad_norm_critic
                    )
                    self.optimizers_vh[ag].step()

                # Logging accumulators
                update_info["loss_policy"]     += policy_loss.item()
                update_info["loss_value_vl"]   += vl_loss.item()
                update_info["loss_value_vh"]   += vh_loss.item()
                update_info["entropy"]         += entropy.item()
                update_info["approx_kl"]       += approx_kl.item()
                update_info["clipfrac"]        += clipfrac.item()

                num_updates += 1

        for k in update_info:
            update_info[k] /= max(1, num_updates)
        
        self.buffer.reset()
        
        return update_info
    
    def _recurrent_generator(self, chunk_length: int):
        """
        Build RNN training minibatches.
        We:
        1. Split each (env, agent) trajectory into fixed-length chunks of length L=chunk_length.
        2. Sample a set of these chunks to form a minibatch.
        3. Pack them so the RNN can be unrolled on [L*B] time steps with initial hidden states per sequence.
        """
        T = self.rollout_horizon
        E = self.num_envs
        A = self.num_agents
        N = E * A

        assert T % chunk_length == 0, f"T {T} not divisible by chunk_length {chunk_length}"
        num_chunks = T // chunk_length
        L = chunk_length

        # Flatten time/env/agent into [T*N, ...]
        obs_all           = self.buffer.obs[:T].reshape(T * N, -1)
        share_obs_all     = self.buffer.share_obs[:T].reshape(T * N, -1)
        actions_all       = self.buffer.actions[:T].reshape(T * N, -1)
        z_all             = self.buffer.z[:T].reshape(T * N, -1)
        old_logp_all      = self.buffer.action_log_probs[:T].reshape(T * N, 1)

        returns_vl_all    = self.buffer.returns_vl.reshape(T * N, 1)
        returns_vh_all    = self.buffer.returns_vh.reshape(T * N, 1)
        advantages_all    = self.buffer.advantages.reshape(T * N, 1)

        values_vl_all     = self.buffer.values_vl[:T].reshape(T * N, 1)
        values_vh_all     = self.buffer.values_vh[:T].reshape(T * N, 1)

        masks_all         = self.buffer.masks[:T].reshape(T * N, 1)

        # RNN hidden states at the start of each time step
        rnn_states_actor_all  = self.buffer.rnn_states_actor[:T].reshape(T, N, -1)
        rnn_states_critic_all = self.buffer.rnn_states_critic[:T].reshape(T, N, -1)
        rnn_states_vh_all     = self.buffer.rnn_states_vh[:T].reshape(T, N, -1)

        # Build list of (chunk_id, env_id, agent_id) for all segments
        indices = []
        for chunk_id in range(num_chunks):
            for env_id in range(E):
                for agent_id in range(A):
                    indices.append((chunk_id, env_id, agent_id))

        total_samples = len(indices)
        batch_size = total_samples // self.num_mini_batch

        for _ in range(self.num_mini_batch):
            # Sample which (chunk, env, agent) tuples go in this minibatch
            perm = torch.randperm(total_samples)
            mb_indices = perm[:batch_size].tolist()

            seq_list = []
            for idx in mb_indices:
                chunk_id, env_id, agent_id = indices[idx]
                t0 = chunk_id * L

                # Agent-major index in [0..N)
                n_idx = agent_id * E + env_id

                # Collect time indices for this L-step window
                seq_indices = [t * N + n_idx for t in range(t0, t0 + L)]
                seq_indices = torch.tensor(seq_indices, device=self.device, dtype=torch.long)

                seq_list.append({
                    "obs":        obs_all[seq_indices],
                    "share_obs":  share_obs_all[seq_indices],
                    "actions":    actions_all[seq_indices],
                    "z":          z_all[seq_indices],
                    "old_logp":   old_logp_all[seq_indices],
                    "ret_vl":     returns_vl_all[seq_indices],
                    "ret_vh":     returns_vh_all[seq_indices],
                    "advantages": advantages_all[seq_indices],
                    "old_vl":     values_vl_all[seq_indices],
                    "old_vh":     values_vh_all[seq_indices],
                    "masks":      masks_all[seq_indices],

                    # RNN initial hidden state is taken from the first timestep t0
                    "h0_actor":   rnn_states_actor_all[t0, n_idx],
                    "h0_vl":      rnn_states_critic_all[t0, n_idx],
                    "h0_vh":      rnn_states_vh_all[t0, n_idx],

                    "agent_id":   agent_id,
                })

            B = len(seq_list)

            yield {
                # Concat over sequences -> [L*B, ...]
                "obs_flat":        torch.cat([d["obs"]        for d in seq_list], dim=0),
                "share_obs_flat":  torch.cat([d["share_obs"]  for d in seq_list], dim=0),
                "act_flat":        torch.cat([d["actions"]    for d in seq_list], dim=0),
                "old_logp_flat":   torch.cat([d["old_logp"]   for d in seq_list], dim=0),
                "adv_flat":        torch.cat([d["advantages"] for d in seq_list], dim=0),
                "ret_vl_flat":     torch.cat([d["ret_vl"]     for d in seq_list], dim=0),
                "ret_vh_flat":     torch.cat([d["ret_vh"]     for d in seq_list], dim=0),
                "old_vl_flat":     torch.cat([d["old_vl"]     for d in seq_list], dim=0),
                "old_vh_flat":     torch.cat([d["old_vh"]     for d in seq_list], dim=0),
                "z_flat":          torch.cat([d["z"]          for d in seq_list], dim=0),
                "masks_flat":      torch.cat([d["masks"]      for d in seq_list], dim=0),

                # Stack per-sequence init states -> [B, H]
                "h0_actor":        torch.stack([d["h0_actor"] for d in seq_list], dim=0),
                "h0_vl":           torch.stack([d["h0_vl"]    for d in seq_list], dim=0),
                "h0_vh":           torch.stack([d["h0_vh"]    for d in seq_list], dim=0),

                # Which agent each of the B sequences belongs to -> [B]
                "agent_idx":       torch.tensor(
                                    [d["agent_id"] for d in seq_list],
                                    device=self.device,
                                    dtype=torch.long
                                ),
            }
    
    @torch.no_grad()
    def run_single_eval_episode(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Run a single evaluation episode using root_finder for safe z*.
        
        Args:
            deterministic: If True, use deterministic policy (no exploration noise)
            
        Returns:
            Dictionary with episode statistics:
                - task_return: Total task reward
                - safe_cost_sum: Total safety cost
                - length: Episode length
                - success: Task success flag
                - violations: Number of safety violations
                - z_mean: Average z value used
        """
        self.set_eval_mode()
        
        obs, _ = self.env.reset()
        self._init_rnn_states()
        
        episode_task_return = 0.0
        episode_safe_cost = 0.0
        episode_length = 0
        episode_violations = 0
        z_values = []
        done = False
        
        while not done and episode_length < 2000:
            # Use root_finder to compute safe z*
            z_candidates = []
            
            for agent in self.agent_ids:
                # Define vh evaluation function for this agent
                def vh_eval_fn(z_query: torch.Tensor):
                    # z_query: [E, 1]
                    z_enc_query = self.z_encoder(z_query)
                    vh_pred, _ = self.critics_vh[agent].value_step(
                        obs[agent],
                        z_enc_query,
                        self.rnn_states[agent]["vh"],
                        torch.ones(self.num_envs, 1, device=self.device),
                    )
                    return vh_pred  # [E, 1]
                
                # Solve for safe z_i* for this agent
                z_i_star = self.root_finder.solve(
                    vh_eval_fn=vh_eval_fn,
                    obs=obs[agent],
                    h_tgt=0.0,  # Safety threshold
                )  # -> [E, 1]
                
                z_candidates.append(z_i_star)
            
            # Take maximum over all agents (most conservative)
            z_global = torch.max(torch.stack(z_candidates, dim=0), dim=0)[0]
            z_values.append(z_global.mean().item())
            
            # Encode z and get actions
            z_enc = self.z_encoder(z_global)
            
            actions = {}
            masks = torch.ones(self.num_envs, 1, device=self.device)
            
            for agent in self.agent_ids:
                act, _, next_h, _ = self.actors[agent].act_step(
                    obs[agent],
                    z_enc,
                    self.rnn_states[agent]["actor"],
                    masks,
                    deterministic=deterministic,
                )
                actions[agent] = act
                self.rnn_states[agent]["actor"] = next_h
            
            # Step environment
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Accumulate statistics
            if "r_task" in info:
                for agent in self.agent_ids:
                    episode_task_return += info["r_task"][agent].mean().item()
            
            if "r_safe" in info:
                for agent in self.agent_ids:
                    episode_safe_cost += info["r_safe"][agent].mean().item()
            
            if "is_violating" in info and info["is_violating"] is not None:
                episode_violations += info["is_violating"].sum().item()
            
            episode_length += 1
            
            # Check done
            done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for agent in self.agent_ids:
                agent_done = terminated[agent] | truncated[agent]
                if agent_done.dim() > 1:
                    agent_done = agent_done.squeeze(-1)
                done_any |= agent_done
            
            if done_any.any():
                done = True
        
        # Check success
        success = False
        if "progress_ratio" in info:
            progress = info["progress_ratio"].mean().item()
            success = (progress >= 0.95)
        
        return {
            "task_return": episode_task_return,
            "safe_cost_sum": episode_safe_cost,
            "length": episode_length,
            "success": success,
            "violations": episode_violations,
            "z_mean": np.mean(z_values) if z_values else 0.0,
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run multiple evaluation episodes and aggregate statistics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Aggregated evaluation statistics
        """
        self.set_eval_mode()
        
        all_episode_returns = []
        all_episode_safe_costs = []
        all_episode_success = []
        all_episode_lengths = []
        all_z_values = []
        
        print(f"[EVAL] Starting evaluation: {num_episodes} episodes")
        
        for ep in range(num_episodes):
            ep_stats = self.run_single_eval_episode(deterministic=True)
            
            all_episode_returns.append(ep_stats["task_return"])
            all_episode_safe_costs.append(ep_stats["safe_cost_sum"])
            all_episode_success.append(ep_stats["success"])
            all_episode_lengths.append(ep_stats["length"])
            all_z_values.append(ep_stats["z_mean"])
            
            print(f"[EVAL] Episode {ep + 1}/{num_episodes}: "
                  f"return={ep_stats['task_return']:.2f}, "
                  f"length={ep_stats['length']}, "
                  f"success={ep_stats['success']}")
        
        # Aggregate statistics using helper function if available
        if summarize_eval_stats is not None:
            eval_stats = summarize_eval_stats(
                episode_returns=all_episode_returns,
                episode_safe_costs=all_episode_safe_costs,
                episode_success=all_episode_success,
                episode_lengths=all_episode_lengths,
                z_values=all_z_values,
            )
        else:
            # Fallback to basic stats
            eval_stats = {
                "return_mean": np.mean(all_episode_returns),
                "return_std": np.std(all_episode_returns),
                "episode_length": np.mean(all_episode_lengths),
                "success_rate": np.mean(all_episode_success),
                "z_global_mean": np.mean(all_z_values),
                "z_global_std": np.std(all_z_values),
            }
        
        self._mark_env_reset_needed()
        return eval_stats
    
    def maybe_milestone_eval_and_save(self):
        """
        Check if milestone reached. If yes:
        1. Mark milestone truncation in buffer (if mid-rollout)
        2. Set eval mode
        3. Run evaluation
        4. Log to wandb
        5. Save checkpoint
        6. Reset env and RNN, back to train mode
        
        This is the ROUTE B core method.
        """
        if len(self.milestones) == 0:
            return
        
        # Check if we've reached the next milestone
        if self.global_episodes >= self.milestones[0]:
            milestone = self.milestones.popleft()
            
            print(f"\n{'='*80}")
            print(f"MILESTONE REACHED: {milestone} episodes")
            print(f"{'='*80}\n")
            
            # Run evaluation
            num_eval_episodes = self.full_config.get("training_monitor", {}).get("eval_episodes", 10)
            eval_stats = self.evaluate(num_episodes=num_eval_episodes)
            
            # Log to wandb if enabled
            if self.wandb_logger is not None:
                self.wandb_logger.log_eval(self.global_step, eval_stats)
            
            # Save checkpoint
            os.makedirs(self.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(self.ckpt_dir, f"epigraph_milestone_{milestone}.pth")
            self.save_checkpoint(ckpt_path)
            print(f"[MILESTONE] Checkpoint saved: {ckpt_path}")
            
            # Print evaluation summary
            print(f"\n[MILESTONE EVAL] Milestone {milestone}:")
            print(f"  Return:       {eval_stats.get('return_mean', 0):.2f} ± {eval_stats.get('return_std', 0):.2f}")
            print(f"  Success Rate: {eval_stats.get('success_rate', 0):.2%}")
            print(f"  Ep Length:    {eval_stats.get('episode_length', 0):.1f}")
            print(f"  Z Mean:       {eval_stats.get('z_global_mean', 0):.4f}\n")
            
            # Back to training mode
            self.set_train_mode()
            self._mark_env_reset_needed()
            print(f"[MILESTONE] Resuming training...\n")
    
    def save_checkpoint(self, path: str, global_step: Optional[int] = None, update_count: Optional[int] = None):
        """Save checkpoint with all networks, optimizers, and training state."""
        checkpoint = {
            "actors": {agent: self.actors[agent].state_dict() for agent in self.agent_ids},
            "critics_vh": {agent: self.critics_vh[agent].state_dict() for agent in self.agent_ids},
            "critic_vl": self.critic_vl.state_dict(),
            "z_encoder": self.z_encoder.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_vl": self.optimizer_vl.state_dict(),
            "optimizers_vh": {agent: self.optimizers_vh[agent].state_dict() for agent in self.agent_ids},
            "global_step": global_step if global_step is not None else self.global_step,
            "global_episodes": self.global_episodes,
            "episodes_done": self.episodes_done,  # Legacy compatibility
            "milestones": list(self.milestones),  # Save remaining milestones
        }
        
        if update_count is not None:
            checkpoint["update_count"] = update_count
        
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint with all networks, optimizers, and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load networks
        for agent in self.agent_ids:
            self.actors[agent].load_state_dict(checkpoint["actors"][agent])
            self.critics_vh[agent].load_state_dict(checkpoint["critics_vh"][agent])
        
        self.critic_vl.load_state_dict(checkpoint["critic_vl"])
        self.z_encoder.load_state_dict(checkpoint["z_encoder"])
        
        # Load optimizers
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_vl.load_state_dict(checkpoint["optimizer_vl"])
        for agent in self.agent_ids:
            self.optimizers_vh[agent].load_state_dict(checkpoint["optimizers_vh"][agent])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.global_episodes = checkpoint.get("global_episodes", checkpoint.get("episodes_done", 0))
        self.episodes_done = self.global_episodes  # Legacy compatibility
        
        # Load remaining milestones
        if "milestones" in checkpoint:
            self.milestones = deque(checkpoint["milestones"])
        
        print(f"[CHECKPOINT] Loaded from {path}")
        print(f"[CHECKPOINT] Resumed at global_step={self.global_step}, global_episodes={self.global_episodes}")
        self._mark_env_reset_needed()
    
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
