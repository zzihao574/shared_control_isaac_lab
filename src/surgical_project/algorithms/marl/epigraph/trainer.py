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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import numpy as np
import traceback
from datetime import datetime

from .epigraph_core import ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder
from .rollout_buffer_z import RolloutBufferZ

from scripts.utils.training_helpers_epigraph import (
    summarize_rollout_stats,
    summarize_eval_stats,
    WandBLogger,
)


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
        self.max_global_steps = max_global_steps
        self.ckpt_dir = ckpt_dir

        # ------------------------------------------------------------------
        # Checkpoint directory layout (aligned with rMAPPO):
        #   <root>/<YYYYMMDD_HHMMSS>/checkpoints/ckpt_milestone_...
        # If user passes logs/.../checkpoints -> use parent as root.
        # If path already contains a timestamp run dir, reuse it.
        # ------------------------------------------------------------------
        def _is_timestamp_dir(name: str) -> bool:
            return (
                len(name) == 15
                and name[8] == "_"
                and name[:8].isdigit()
                and name[9:].isdigit()
            )

        ckpt_dir_norm = os.path.normpath(ckpt_dir)
        base, last = os.path.split(ckpt_dir_norm)

        run_root: Optional[str] = None
        run_dir: Optional[str] = None

        if last == "checkpoints":
            if base:
                base_parent, base_last = os.path.split(base)
                if _is_timestamp_dir(base_last):
                    # logs/.../<timestamp>/checkpoints -> reuse timestamp directory
                    run_root = base_parent if base_parent else "."
                    run_dir = base
                else:
                    run_root = base if base else "."
                    run_dir = os.path.join(run_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
            else:
                run_root = "."
                run_dir = os.path.join(run_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        elif _is_timestamp_dir(last):
            run_root = base if base else "."
            run_dir = ckpt_dir_norm
        else:
            run_root = ckpt_dir_norm
            run_dir = os.path.join(run_root, datetime.now().strftime("%Y%m%d_%H%M%S"))

        final_ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(final_ckpt_dir, exist_ok=True)

        self.ckpt_root = run_root
        self.ckpt_run_dir = run_dir
        self.ckpt_dir = final_ckpt_dir
        self.milestone_index_path = os.path.join(self.ckpt_dir, "milestones_index.txt")

        print(f"[TRAINER] Checkpoints will be stored in: {self.ckpt_dir}")

        # Extract environment params
        if hasattr(env.unwrapped, "params"):
            self.env_cfg = env.unwrapped.params.get("epigraph_env", {})
        else:
            self.env_cfg = {}
        
        # Environment dimensions
        self.num_envs = env.num_envs
        self.agent_ids = list(env.unwrapped.cfg.possible_agents)
        desired_order = ["human", "robot"]
        self.agent_ids = [aid for aid in desired_order if aid in self.agent_ids] + \
                         [aid for aid in self.agent_ids if aid not in desired_order]
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
        self.base_entropy_coef = float(algo_cfg.get("entropy_coef", 0.01))
        self.entropy_coef = self.base_entropy_coef
        self.use_orthogonal = bool(algo_cfg.get("use_orthogonal", True))
        self.init_gain = float(algo_cfg.get("gain", 0.01))
        self.use_clipped_value_loss = bool(algo_cfg.get("use_clipped_value_loss", True))
        self.huber_delta = float(algo_cfg.get("huber_delta", 1.0))
        self.max_grad_norm_actor = algo_cfg["max_grad_norm_actor"]
        self.max_grad_norm_critic = algo_cfg["max_grad_norm_critic"]

        per_actor_cfg = algo_cfg.get("per_actor", {})
        entropy_mult_cfg = per_actor_cfg.get("entropy_multiplier", {})
        self.entropy_multiplier = {
            agent: float(entropy_mult_cfg.get(agent, 1.0))
            for agent in self.agent_ids
        }
        
        # Epigraph-specific hyperparameters
        self.z_min = epi_cfg["z"]["min"]
        self.z_max = epi_cfg["z"]["max"]
        self.z_nz = epi_cfg["z"]["encode"]["nz"]
        self.z_init_mode = epi_cfg["z"]["init"]["mode"]
        self.z_init_p_extreme = epi_cfg["z"]["init"]["p_extreme"]

        constraints_cfg = full_config.get("constraints", {})
        self.max_forces = {
            "robot": float(constraints_cfg.get("max_robot_force", 1.0)),
            "human": float(constraints_cfg.get("max_human_force", 1.0)),
        }

        entropy_sched_cfg = full_config.get("training", {}).get("entropy_schedule", {})
        self.entropy_schedule_enabled = bool(entropy_sched_cfg.get("enabled", False))
        self.entropy_schedule_start = float(entropy_sched_cfg.get("start", self.base_entropy_coef))
        self.entropy_schedule_end = float(entropy_sched_cfg.get("end", self.base_entropy_coef))
        
        # Build networks, buffer, optimizers
        self._build_networks()
        self._build_buffer()
        self._build_optimizers()
        self._init_rnn_states()
        
        # Training state
        self.global_step = 0
        self.global_episodes = 0
        self.episodes_done = 0  # Legacy name for compatibility

        train_seed = int(full_config.get("training", {}).get("seed", 42))
        self._train_generator = torch.Generator(device="cpu")
        self._train_generator.manual_seed(train_seed + 424242)
        
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

        lr_decay_cfg = full_config.get("training", {}).get("lr_decay", {})
        self.lr_decay_enabled = bool(lr_decay_cfg.get("enabled", False))
        self.lr_final_factor = float(lr_decay_cfg.get("final_factor", 0.1))
        self.global_update_step = 0

        self._cached_obs = None
        self._cached_z = None
        self._cached_masks = None
        self._needs_env_reset = True


    def _build_networks(self):
        """Build all networks."""
        # Z encoders (shared per network)
        self.z_encoder_actor = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
            use_orthogonal=self.use_orthogonal,
        ).to(self.device)
        self.z_encoder_vl = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
            use_orthogonal=self.use_orthogonal,
        ).to(self.device)
        self.z_encoder_vh = ZEncoder(
            nz=self.z_nz,
            z_mean=self.epi_cfg["z"]["encode"]["mean"],
            z_scale=self.epi_cfg["z"]["encode"]["scale"],
            use_orthogonal=self.use_orthogonal,
        ).to(self.device)

        # Per-agent actors (shared Vl, per-agent Vh)
        self.actors = {
            agent: ActorRNN(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                nz=self.z_nz,
                recurrent_N=self.recurrent_N,
                use_orthogonal=self.use_orthogonal,
                gain=self.init_gain,
            ).to(self.device)
            for agent in self.agent_ids
        }

        self.critic_vl = CriticVlRNN(
            share_obs_dim=self.share_obs_dim,
            hidden_size=self.hidden_size,
            nz=self.z_nz,
            recurrent_N=self.recurrent_N,
            use_orthogonal=self.use_orthogonal,
        ).to(self.device)

        self.critics_vh = {
            agent: CriticVhRNN(
                obs_dim=self.obs_dim,
                hidden_size=self.hidden_size,
                nz=self.z_nz,
                recurrent_N=self.recurrent_N,
                use_orthogonal=self.use_orthogonal,
            ).to(self.device)
            for agent in self.agent_ids
        }

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
        actor_params = list(self.z_encoder_actor.parameters())
        for agent in self.agent_ids:
            actor_params.extend(self.actors[agent].parameters())
        self.actor_parameters = actor_params
        self.optimizer_actor = optim.Adam(
            self.actor_parameters,
            lr=self.algo_cfg["actor_lr"],
            eps=self.algo_cfg["opt_eps"],
        )

        self.vl_parameters = list(self.critic_vl.parameters()) + list(self.z_encoder_vl.parameters())
        self.optimizer_vl = optim.Adam(
            self.vl_parameters,
            lr=self.algo_cfg["critic_lr"],
            eps=self.algo_cfg["opt_eps"],
        )

        vh_params = list(self.z_encoder_vh.parameters())
        for agent in self.agent_ids:
            vh_params.extend(self.critics_vh[agent].parameters())
        self.vh_parameters = vh_params
        self.optimizer_vh = optim.Adam(
            self.vh_parameters,
            lr=self.algo_cfg["critic_lr"],
            eps=self.algo_cfg["opt_eps"],
        )

        self.actor_lr_init = self.optimizer_actor.param_groups[0]["lr"]
        self.vl_lr_init = self.optimizer_vl.param_groups[0]["lr"]
        self.vh_lr_init = self.optimizer_vh.param_groups[0]["lr"]
        
    def _init_rnn_states(self):
        """Initialize RNN hidden states."""
        self.rnn_states_actor = {
            agent: torch.zeros(self.num_envs, self.hidden_size, device=self.device)
            for agent in self.agent_ids
        }
        self.rnn_states_vl = torch.zeros(self.num_envs, self.hidden_size, device=self.device)
        self.rnn_states_vh = {
            agent: torch.zeros(self.num_envs, self.hidden_size, device=self.device)
            for agent in self.agent_ids
        }

    def _mark_env_reset_needed(self):
        """Flag that the environment should be fully reset before the next rollout."""
        self._needs_env_reset = True
        self._cached_obs = None
        self._cached_z = None
        self._cached_masks = None

    def _scale_actions(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scale normalized actions to physical force limits."""
        scaled = {}
        for agent in self.agent_ids:
            limit = float(self.max_forces.get(agent, 1.0))
            scaled[agent] = actions[agent] * limit
        return scaled

    def _check_milestone_crossed(self, future_episodes: int):
        """Return the next milestone if crossed by future episode count."""
        if not self.milestones:
            return None
        if future_episodes >= self.milestones[0]:
            return self.milestones[0]
        return None

    def _maybe_decay_lr(self, update_idx: int):
        """Apply cosine LR decay to all optimizers."""
        if not self.lr_decay_enabled:
            return

        progress = min(1.0, update_idx / 1000.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        actor_min = self.actor_lr_init * self.lr_final_factor
        actor_lr = actor_min + (self.actor_lr_init - actor_min) * cosine
        for pg in self.optimizer_actor.param_groups:
            pg["lr"] = actor_lr

        vl_min = self.vl_lr_init * self.lr_final_factor
        vl_lr = vl_min + (self.vl_lr_init - vl_min) * cosine
        for pg in self.optimizer_vl.param_groups:
            pg["lr"] = vl_lr

        vh_min = self.vh_lr_init * self.lr_final_factor
        vh_lr = vh_min + (self.vh_lr_init - vh_min) * cosine
        for pg in self.optimizer_vh.param_groups:
            pg["lr"] = vh_lr

    def _get_current_lrs(self) -> Dict[str, float]:
        """Return current learning rates for all optimizers."""
        return {
            "lr/actor": self.optimizer_actor.param_groups[0]["lr"],
            "lr/vl": self.optimizer_vl.param_groups[0]["lr"],
            "lr/vh": self.optimizer_vh.param_groups[0]["lr"],
        }

    def _get_entropy_coef(self) -> float:
        if not self.entropy_schedule_enabled or (self.max_global_steps is None) or self.max_global_steps <= 0:
            return self.base_entropy_coef
        progress = min(1.0, self.global_step / self.max_global_steps)
        start = self.entropy_schedule_start
        end = self.entropy_schedule_end
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
        abs_err = error.abs()
        return torch.where(
            abs_err <= delta,
            0.5 * error.pow(2),
            delta * (abs_err - 0.5 * delta)
        )

    def _log_rollout_stats(self, step: int, stats: Dict[str, float]):
        """Log rollout statistics to WandB with epigraph-specific keys."""
        if self.wandb_logger is None or not getattr(self.wandb_logger, "enabled", False):
            return

        payload = {
            "avg_task_return": stats.get("avg_episode_task_return"),
            "avg_safe_cost": stats.get("avg_episode_safe_cost_sum"),
            "avg_combined_return": stats.get("avg_episode_combined_return"),
            "episodes_finished": stats.get("episodes_finished"),
            "epigraph/unsafe_step_ratio": stats.get("unsafe_step_ratio"),
            "epigraph/progress_ratio": stats.get("avg_progress_ratio"),
            "epigraph/z_mean_rollout": stats.get("z_mean"),
            "epigraph/z_std_rollout": stats.get("z_std"),
            "collection_steps": step,
        }

        for agent in self.agent_ids:
            key_suffix = agent.replace(" ", "_")
            payload[f"task_mean_{key_suffix}"] = stats.get(f"r_task_mean_{key_suffix}")
            payload[f"safe_cost_mean_{key_suffix}"] = stats.get(f"r_safe_cost_mean_{key_suffix}")
            payload[f"epigraph/z_mean_{key_suffix}"] = stats.get(f"avg_z_{key_suffix}")
            payload[f"forces/{key_suffix}_fx_mean"] = stats.get(f"forces/{agent}_fx_mean")
            payload[f"forces/{key_suffix}_fy_mean"] = stats.get(f"forces/{agent}_fy_mean")
            payload[f"forces/{key_suffix}_fz_mean"] = stats.get(f"forces/{agent}_fz_mean")

        cleaned = {k: v for k, v in payload.items() if v is not None}
        if cleaned:
            self.wandb_logger.log_rollout(step, cleaned)

    def _log_update_stats(self, step: int, stats: Dict[str, float]):
        """Log update statistics to WandB."""
        if self.wandb_logger is None or not getattr(self.wandb_logger, "enabled", False):
            return

        payload = {
            "loss_policy": stats.get("loss_policy"),
            "loss_value_vl": stats.get("loss_value_vl"),
            "loss_value_vh": stats.get("loss_value_vh"),
            "entropy": stats.get("entropy"),
            "approx_kl": stats.get("approx_kl"),
            "clipfrac": stats.get("clipfrac"),
            "entropy_coef": stats.get("entropy_coef"),
            "global_step": step,
            "grad/actor": stats.get("grad_actor"),
            "grad/vl": stats.get("grad_vl"),
            "grad/vh": stats.get("grad_vh"),
            "lr/actor": stats.get("lr/actor"),
            "lr/vl": stats.get("lr/vl"),
            "lr/vh": stats.get("lr/vh"),
        }
        for agent in self.agent_ids:
            key_suffix = agent.replace(" ", "_")
            payload[f"entropy_coef_{key_suffix}"] = stats.get(f"entropy_coef_{key_suffix}")
        cleaned = {k: v for k, v in payload.items() if v is not None}
        if cleaned:
            self.wandb_logger.log_update(step, cleaned)

        
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
        actual_env = getattr(self.env, "unwrapped", self.env)

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

        episode_returns_task = []
        episode_returns_safe = []
        episode_lengths = []
        current_returns_task = torch.zeros(self.num_envs, device=self.device)
        current_returns_safe = torch.zeros(self.num_envs, device=self.device)
        current_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        z_values_rollout = []
        violations_rollout = []

        episodes_window = 0
        evaluation_done_this_rollout = False
        force_sums = {agent: torch.zeros(3, device=self.device) for agent in self.agent_ids}
        steps_collected = 0
        progress_rollout = []

        for _ in range(self.rollout_horizon):
            share_obs = self._get_share_obs(obs)
            z_values_rollout.append(z_global.clone())

            actions = {}
            action_log_probs = {}
            z_enc_actor_global = self.z_encoder_actor(z_global)
            for agent in self.agent_ids:
                act_a, logp_a, next_h_a, _ = self.actors[agent].act_step(
                    obs[agent],
                    z_enc_actor_global,
                    self.rnn_states_actor[agent],
                    masks,
                    deterministic=False,
                )
                actions[agent] = act_a
                action_log_probs[agent] = logp_a
                self.rnn_states_actor[agent] = next_h_a

            env_actions = self._scale_actions(actions)
            detail_payload = {
                "applied_forces": {aid: env_actions[aid].clone() for aid in self.agent_ids},
                "mean_actions": {aid: env_actions[aid].clone() for aid in self.agent_ids},
                "noise_actions": {aid: torch.zeros_like(env_actions[aid]) for aid in self.agent_ids},
                "deterministic": False,
            }
            if hasattr(actual_env, "set_detail_actor_info"):
                actual_env.set_detail_actor_info(detail_payload)

            for agent in self.agent_ids:
                force_sums[agent] += env_actions[agent].mean(dim=0)
            steps_collected += 1
            actual_env._last_z_snapshot = z_global.clone().detach()
            # Update environment-side step tracer counters (trainer drives the step count)
            if hasattr(actual_env, "_trainer_global_step"):
                actual_env._trainer_global_step = self.global_step + 1

            z_enc_vl = self.z_encoder_vl(z_global)
            vl, next_h_vl = self.critic_vl.value_step(
                share_obs,
                z_enc_vl,
                self.rnn_states_vl,
                masks,
            )
            self.rnn_states_vl = next_h_vl

            z_enc_vh_global = self.z_encoder_vh(z_global)
            vh = {}
            for agent in self.agent_ids:
                vh_val, next_h_vh = self.critics_vh[agent].value_step(
                    obs[agent],
                    z_enc_vh_global,
                    self.rnn_states_vh[agent],
                    masks,
                )
                vh[agent] = vh_val
                self.rnn_states_vh[agent] = next_h_vh

            rnn_states_snapshot = {
                "vl": self.rnn_states_vl.clone(),
                "vh": {agent: self.rnn_states_vh[agent].clone() for agent in self.agent_ids},
            }

            obs_next, rewards, terminated, truncated, info = self.env.step(env_actions)
            progress_rollout.append(
                actual_env.reward_components.get(
                    "progress_ratio",
                    torch.zeros(self.num_envs, device=self.device),
                ).clone().detach()
            )

            r_task = info["r_task"]
            r_safe = info["r_safe"]
            r_team = sum(r_task[a] for a in self.agent_ids) / self.num_agents

            if "is_violating" in info and info["is_violating"] is not None:
                violations_rollout.append(info["is_violating"])

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

            episode_increment = int(done_any_env.sum().item())
            z_next = update_z_epigraph(
                z_global,
                r_team,
                self.gamma,
                self.z_min,
                self.z_max,
            )

            override_mask_vl_flat = torch.zeros(self.num_envs * self.num_agents, 1, dtype=torch.bool, device=self.device)
            override_vl_flat = torch.zeros(self.num_envs * self.num_agents, 1, device=self.device)
            override_mask_vh_flat = torch.zeros(self.num_envs * self.num_agents, 1, dtype=torch.bool, device=self.device)
            override_vh_flat = torch.zeros(self.num_envs * self.num_agents, 1, device=self.device)
            ongoing_env_mask = None
            milestone_to_trigger = None

            if not evaluation_done_this_rollout:
                future_episodes = self.global_episodes + episodes_window + episode_increment
                milestone_to_trigger = self._check_milestone_crossed(future_episodes)
                if milestone_to_trigger is not None:
                    (
                        override_mask_vl_flat,
                        override_vl_flat,
                        override_mask_vh_flat,
                        override_vh_flat,
                        ongoing_env_mask,
                    ) = self._maybe_override_bootstrap(
                        obs_next=obs_next,
                        z_next=z_next,
                        rnn_states_snapshot=rnn_states_snapshot,
                        allow_env_mask=masks,
                        term_any_env=term_any_env,
                        trunc_any_env=trunc_any_env,
                    )

            rewards_formatted = {}
            for agent in self.agent_ids:
                rew_agent = rewards[agent]
                if rew_agent.dim() == 1:
                    rew_agent = rew_agent.unsqueeze(-1)
                rewards_formatted[agent] = rew_agent

            self.buffer.insert(
                obs=self._flatten_per_agent(obs),
                share_obs=self._replicate_global_for_agents(share_obs),
                actions=self._flatten_per_agent(actions),
                action_log_probs=self._flatten_per_agent(action_log_probs),
                rewards=self._flatten_per_agent(rewards_formatted),
                rewards_task=self._flatten_per_agent(r_task),
                costs_safe=self._flatten_per_agent(r_safe),
                values_vl=self._replicate_global_for_agents(vl),
                values_vh=self._flatten_per_agent(vh),
                z=self._replicate_global_for_agents(z_global),
                masks=masks_flat_t,
                term_masks=term_masks_flat_t,
                override_bootstrap_mask_vl=override_mask_vl_flat,
                override_bootstrap_vl=override_vl_flat,
                override_bootstrap_mask_vh=override_mask_vh_flat,
                override_bootstrap_vh=override_vh_flat,
                rnn_states_actor=torch.cat([self.rnn_states_actor[a] for a in self.agent_ids], dim=0),
                rnn_states_critic=self.rnn_states_vl.repeat(self.num_agents, 1),
                rnn_states_vh=torch.cat(
                    [self.rnn_states_vh[a] for a in self.agent_ids],
                    dim=0,
                ),
            )
            # Count one environment step (do not count eval)
            self.global_step += 1
            assert override_mask_vl_flat.shape == (self.num_agents * self.num_envs, 1)
            assert override_vl_flat.shape == (self.num_agents * self.num_envs, 1)
            assert override_mask_vh_flat.shape == (self.num_agents * self.num_envs, 1)
            assert override_vh_flat.shape == (self.num_agents * self.num_envs, 1)

            buffer_step_idx = self.buffer.step - 1

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
                self.global_episodes += len(done_indices)

            if milestone_to_trigger is not None:
                self._apply_truncation_masks(buffer_step_idx, ongoing_env_mask)
                episodes_window += episode_increment
                self._run_milestone_evaluation(milestone_to_trigger)
                self.milestones.popleft()
                evaluation_done_this_rollout = True

                obs, _ = self.env.reset()
                if hasattr(actual_env, "_last_z_snapshot"):
                    actual_env._last_z_snapshot = None
                z_global = self._init_z_training()
                self._init_rnn_states()
                masks = torch.ones(self.num_envs, 1, device=self.device)
                current_returns_task.zero_()
                current_returns_safe.zero_()
                current_lengths.zero_()
                self._needs_env_reset = False
                continue
            else:
                z_global = z_next
                episodes_window += episode_increment

            obs = obs_next
            masks = (~done_any_env).float()

        z_enc_vl_last = self.z_encoder_vl(z_global)
        share_obs_last = self._get_share_obs(obs)
        masks_last = torch.ones(self.num_envs, 1, device=self.device)

        vl_last, _ = self.critic_vl.value_step(
            share_obs_last,
            z_enc_vl_last,
            self.rnn_states_vl,
            masks_last,
        )
        z_enc_vh_last = self.z_encoder_vh(z_global)
        vh_last = {}
        for agent in self.agent_ids:
            vh_last[agent], _ = self.critics_vh[agent].value_step(
                obs[agent],
                z_enc_vh_last,
                self.rnn_states_vh[agent],
                masks_last,
            )

        self.buffer.insert_final_step(
            obs=self._flatten_per_agent(obs),
            share_obs=self._replicate_global_for_agents(share_obs_last),
            values_vl=self._replicate_global_for_agents(vl_last),
            values_vh=self._flatten_per_agent(vh_last),
            z=self._replicate_global_for_agents(z_global),
        )

        if summarize_rollout_stats is not None:
            T = self.rollout_horizon
            E = self.num_envs
            A = self.num_agents

            r_task_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.rewards_task[:T], has_trailing_dim1=True
            ).squeeze(-1)

            r_safe_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.costs_safe[:T], has_trailing_dim1=True
            ).squeeze(-1)

            z_tensor = self._reshape_agentmajor_to_env_agent_T(
                self.buffer.z[:T], has_trailing_dim1=True
            ).squeeze(-1)

            term_masks_bool = (self.buffer.term_masks[:T] > 0.5)
            dones_tensor = self._reshape_agentmajor_to_env_agent_T(
                ~term_masks_bool, has_trailing_dim1=True
            ).squeeze(-1).any(dim=2).float()

            info_for_stats = {}
            if violations_rollout:
                info_for_stats["is_violating"] = torch.stack(violations_rollout, dim=0)
            if progress_rollout:
                info_for_stats["progress_ratio"] = torch.stack(progress_rollout, dim=0)

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
                "avg_progress_ratio": float(torch.stack(progress_rollout).mean().item()) if progress_rollout else 0.0,
            }

        if steps_collected > 0:
            for agent in self.agent_ids:
                mean_force = force_sums[agent] / steps_collected
                rollout_stats[f"forces/{agent}_fx_mean"] = float(mean_force[0].item())
                rollout_stats[f"forces/{agent}_fy_mean"] = float(mean_force[1].item())
                rollout_stats[f"forces/{agent}_fz_mean"] = float(mean_force[2].item())

        self._cached_obs = {agent: obs[agent].clone().detach() for agent in self.agent_ids}
        self._cached_z = z_global.clone().detach()
        self._cached_masks = masks.clone().detach()

        self._log_rollout_stats(self.global_step, rollout_stats)

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
        Compute override bootstrap values for mid-rollout evaluation if needed.
        
        Returns:
            override_mask_vl_flat, override_vl_flat,
            override_mask_vh_flat, override_vh_flat,
            ongoing_env_mask (bool tensor [E])
        """
        E = self.num_envs
        A = self.num_agents
        device = self.device

        with torch.no_grad():
            alive_mask = allow_env_mask.squeeze(-1) > 0.5
            term_mask = term_any_env.squeeze(-1).to(torch.bool)
            trunc_mask = trunc_any_env.squeeze(-1).to(torch.bool)
            ongoing_env = alive_mask & (~term_mask) & (~trunc_mask)

            if not ongoing_env.any():
                override_mask_vl_env = torch.zeros(E, 1, dtype=torch.bool, device=device)
                override_vl_env = torch.zeros(E, 1, device=device)
                override_mask_vh_env = torch.zeros(E, A, dtype=torch.bool, device=device)
                override_vh_env = torch.zeros(E, A, device=device)
            else:
                share_obs_next = self._get_share_obs(obs_next)
                masks_next = torch.ones(E, 1, device=device)
                z_enc_vl_next = self.z_encoder_vl(z_next)

                vl_next, _ = self.critic_vl.value_step(
                    share_obs_next,
                    z_enc_vl_next,
                    rnn_states_snapshot["vl"],
                    masks_next,
                )

                z_enc_vh_next = self.z_encoder_vh(z_next)
                vh_next = {}
                for agent in self.agent_ids:
                    vh_next[agent], _ = self.critics_vh[agent].value_step(
                        obs_next[agent],
                        z_enc_vh_next,
                        rnn_states_snapshot["vh"][agent],
                        masks_next,
                    )

                override_mask_vl_env = ongoing_env.view(E, 1)
                override_vl_env = vl_next

                override_mask_vh_env = ongoing_env.view(E, 1).repeat(1, A)
                override_vh_env = torch.stack([vh_next[aid].squeeze(-1) for aid in self.agent_ids], dim=1)
        # Repeat per agent to produce agent-major order [A*E, 1]
        override_mask_vl_flat = override_mask_vl_env.repeat(A, 1)
        override_vl_flat = override_vl_env.repeat(A, 1)

        # Convert [E, A] -> agent-major [A*E, 1] i.e., human(env0..E-1), robot(env0..E-1)
        override_mask_vh_flat = override_mask_vh_env.permute(1, 0).reshape(A * E, 1)
        override_vh_flat = override_vh_env.permute(1, 0).reshape(A * E, 1)

        return (
            override_mask_vl_flat,
            override_vl_flat,
            override_mask_vh_flat,
            override_vh_flat,
            ongoing_env,
        )

    def _apply_truncation_masks(self, buffer_step_idx: int, ongoing_env_mask: torch.Tensor):
        """Zero masks for ongoing environments after mid-rollout evaluation."""
        if ongoing_env_mask is None or not ongoing_env_mask.any():
            return

        env_ids = torch.nonzero(ongoing_env_mask, as_tuple=False).view(-1)
        if env_ids.numel() == 0:
            return

        for agent_idx, _ in enumerate(self.agent_ids):
            slots = env_ids + agent_idx * self.num_envs
            self.buffer.masks[buffer_step_idx, slots, 0] = 0.0
            self.buffer.term_masks[buffer_step_idx, slots, 0] = 1.0

    def _run_milestone_evaluation(self, milestone: int):
        """Run evaluation, log metrics, and save checkpoint for the milestone."""
        eval_episodes_cfg = int(
            self.full_config.get("training_monitor", {}).get("eval_episodes", 1)
        )
        if eval_episodes_cfg != 1:
            print(f"[MILESTONE] Overriding configured eval_episodes={eval_episodes_cfg} to 1 (environment 0 only).")
        eval_episodes = 1
        print(f"\n{'='*80}")
        print(f"MILESTONE REACHED: {milestone} episodes")
        print(f"Running evaluation for {eval_episodes} episode...")
        print(f"{'='*80}\n")

        try:
            eval_stats = self.evaluate(num_episodes=eval_episodes)
        except Exception:
            print("[MILESTONE][ERROR] Evaluation failed. Traceback:")
            traceback.print_exc()
            raise

        if self.wandb_logger is not None and getattr(self.wandb_logger, "enabled", False):
            payload = {
                "return_mean": eval_stats.get("eval_return_mean"),
                "return_std": eval_stats.get("eval_return_std"),
                "num_episodes": eval_stats.get("eval_num_episodes"),
                "epigraph/safe_cost_mean": eval_stats.get("eval_safe_cost_mean"),
                "epigraph/safe_cost_std": eval_stats.get("eval_safe_cost_std"),
                "epigraph/z_mean": eval_stats.get("eval_z_mean"),
                "epigraph/z_std": eval_stats.get("eval_z_std"),
                "epigraph/success_rate": eval_stats.get("eval_success_rate"),
            }
            cleaned = {k: v for k, v in payload.items() if v is not None}
            if cleaned:
                self.wandb_logger.log_eval(self.global_step, cleaned)

        # Derive score for checkpoint naming (fallback to 0.0 if unavailable)
        score = None
        for key in ("eval_return_mean", "return_mean"):
            val = eval_stats.get(key)
            if val is not None:
                score = float(val)
                break
        if score is None:
            score = 0.0

        os.makedirs(self.ckpt_dir, exist_ok=True)
        padded_milestone = f"{milestone:06d}"
        ckpt_name = f"ckpt_milestone_{padded_milestone}_score_{score:.6f}.pth"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        self.save_checkpoint(ckpt_path)
        print(f"[MILESTONE] Checkpoint saved: {ckpt_path}")

        self._append_and_sort_milestones_index(
            index_path=self.milestone_index_path,
            milestone=milestone,
            score=score,
            rel_path=ckpt_name,
        )

        print(f"[MILESTONE] Evaluation stats summary: {eval_stats}")

        self.set_train_mode()
        self._mark_env_reset_needed()
        actual_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(actual_env, "_last_z_snapshot"):
            actual_env._last_z_snapshot = None

        return eval_stats
    
    def update(self) -> Dict[str, Any]:
        """
        Sequence-based PPO update with RNN training.
        """
        self.set_train_mode()

        self.buffer.compute_epigraph_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
        )

        entropy_coef = self._get_entropy_coef()
        self.entropy_coef = entropy_coef

        update_info = {
            "loss_policy": 0.0,
            "loss_value_vl": 0.0,
            "loss_value_vh": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "entropy_coef": entropy_coef,
        }

        clip_eps = self.clip_param
        value_clip_eps = self.value_clip_param
        ent_coef = entropy_coef
        L = self.data_chunk_length

        grad_actor_total = 0.0
        grad_vl_total = 0.0
        grad_vh_total = 0.0
        entropy_coef_per_actor = {
            agent: ent_coef * self.entropy_multiplier.get(agent, 1.0)
            for agent in self.agent_ids
        }

        num_actor_updates = 0
        num_value_updates = 0

        for _ in range(self.ppo_epoch):
            for mb in self._recurrent_generator(L):
                obs_flat = mb["obs_flat"]
                act_flat = mb["act_flat"]
                old_logp_flat = mb["old_logp_flat"]
                adv_flat = mb["adv_flat"]
                z_flat = mb["z_flat"]
                masks_flat = mb["masks_flat"]
                h0_actor = mb["h0_actor"]
                agent_idx = mb["agent_idx"]

                B = h0_actor.size(0)
                for a_i, agent in enumerate(self.agent_ids):
                    seq_mask_b = (agent_idx == a_i)
                    if not seq_mask_b.any():
                        continue

                    seq_mask_lb = seq_mask_b.unsqueeze(0).expand(L, B).reshape(L * B)

                    obs_a = obs_flat[seq_mask_lb]
                    act_a = act_flat[seq_mask_lb]
                    old_logp_a = old_logp_flat[seq_mask_lb]
                    adv_a = adv_flat[seq_mask_lb]
                    masks_a = masks_flat[seq_mask_lb]
                    h0_actor_a = h0_actor[seq_mask_b]
                    z_a = z_flat[seq_mask_lb]

                    z_enc_a = self.z_encoder_actor(z_a)
                    logp_new_a, entropy_a, _ = self.actors[agent].evaluate_actions_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_a,
                        hxs_init=h0_actor_a,
                        masks_seq=masks_a,
                        act_seq=act_a,
                    )

                    ratio = torch.exp(logp_new_a - old_logp_a)
                    surr1 = ratio * adv_a
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_a
                    loss_surrogate = torch.max(surr1, surr2).mean()
                    ent_coef_a = entropy_coef_per_actor[agent]
                    entropy_mean = entropy_a.mean()
                    policy_loss = loss_surrogate - ent_coef_a * entropy_mean

                    self.optimizer_actor.zero_grad()
                    policy_loss.backward()
                    gn_actor = torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters,
                        self.max_grad_norm_actor,
                    )
                    self.optimizer_actor.step()

                    grad_actor_total += float(gn_actor)
                    update_info["loss_policy"] += policy_loss.item()
                    update_info["entropy"] += entropy_mean.item()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                        clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
                    update_info["approx_kl"] += float(approx_kl.item())
                    update_info["clipfrac"] += float(clipfrac.item())
                    num_actor_updates += 1

            for mb in self._recurrent_generator(L):
                obs_flat = mb["obs_flat"]
                share_obs_flat = mb["share_obs_flat"]
                ret_vl_flat = mb["ret_vl_flat"]
                ret_vh_flat = mb["ret_vh_flat"]
                old_vl_flat = mb["old_vl_flat"]
                old_vh_flat = mb["old_vh_flat"]
                z_flat = mb["z_flat"]
                masks_flat = mb["masks_flat"]
                h0_vl = mb["h0_vl"]
                h0_vh = mb["h0_vh"]
                agent_idx = mb["agent_idx"]

                z_enc_vl_flat = self.z_encoder_vl(z_flat)
                vl_pred_flat, _ = self.critic_vl.value_seq(
                    share_obs_seq=share_obs_flat,
                    z_enc_seq=z_enc_vl_flat,
                    hxs_init=h0_vl,
                    masks_seq=masks_flat,
                )

                vl_clipped = old_vl_flat + torch.clamp(
                    vl_pred_flat - old_vl_flat,
                    -value_clip_eps,
                    value_clip_eps,
                )
                error_vl = vl_pred_flat - ret_vl_flat
                error_vl_clipped = vl_clipped - ret_vl_flat
                vl_loss_unclipped = self._huber_loss(error_vl, self.huber_delta)
                vl_loss_clipped = self._huber_loss(error_vl_clipped, self.huber_delta)
                if self.use_clipped_value_loss:
                    vl_loss = torch.max(vl_loss_unclipped, vl_loss_clipped).mean()
                else:
                    vl_loss = vl_loss_unclipped.mean()

                vh_loss_total = 0.0
                count_total = 0.0

                B = h0_vh.size(0)
                for a_i, agent in enumerate(self.agent_ids):
                    seq_mask_b = (agent_idx == a_i)
                    if not seq_mask_b.any():
                        continue

                    seq_mask_lb = seq_mask_b.unsqueeze(0).expand(L, B).reshape(L * B)

                    obs_a = obs_flat[seq_mask_lb]
                    masks_a = masks_flat[seq_mask_lb]
                    ret_vh_a = ret_vh_flat[seq_mask_lb]
                    old_vh_a = old_vh_flat[seq_mask_lb]
                    h0_vh_a = h0_vh[seq_mask_b]
                    z_flat_a = z_flat[seq_mask_lb]
                    z_enc_vh_a = self.z_encoder_vh(z_flat_a)

                    vh_pred_a, _ = self.critics_vh[agent].value_seq(
                        obs_seq=obs_a,
                        z_enc_seq=z_enc_vh_a,
                        hxs_init=h0_vh_a,
                        masks_seq=masks_a,
                    )

                    vh_clipped_a = old_vh_a + torch.clamp(
                        vh_pred_a - old_vh_a,
                        -value_clip_eps,
                        value_clip_eps,
                    )
                    error_vh = vh_pred_a - ret_vh_a
                    error_vh_clipped = vh_clipped_a - ret_vh_a
                    vh_loss_unclipped_a = self._huber_loss(error_vh, self.huber_delta)
                    vh_loss_clipped_a = self._huber_loss(error_vh_clipped, self.huber_delta)
                    if self.use_clipped_value_loss:
                        vh_loss_a = torch.max(vh_loss_unclipped_a, vh_loss_clipped_a).mean()
                    else:
                        vh_loss_a = vh_loss_unclipped_a.mean()
                    vh_loss_total += vh_loss_a
                    count_total += 1.0

                if count_total < 1e-8:
                    continue

                vh_loss = vh_loss_total / count_total

                self.optimizer_vl.zero_grad()
                self.optimizer_vh.zero_grad()

                (vl_loss + vh_loss).backward()

                gn_vl = torch.nn.utils.clip_grad_norm_(
                    self.vl_parameters,
                    self.max_grad_norm_critic,
                )
                self.optimizer_vl.step()
                grad_vl_total += float(gn_vl)

                gn_vh = torch.nn.utils.clip_grad_norm_(
                    self.vh_parameters,
                    self.max_grad_norm_critic,
                )
                self.optimizer_vh.step()
                grad_vh_total += float(gn_vh)

                update_info["loss_value_vl"] += vl_loss.item()
                update_info["loss_value_vh"] += vh_loss.item()
                num_value_updates += 1

        actor_denom = max(1, num_actor_updates)
        value_denom = max(1, num_value_updates)

        update_info["loss_policy"] /= actor_denom
        update_info["entropy"] /= actor_denom
        update_info["approx_kl"] /= actor_denom
        update_info["clipfrac"] /= actor_denom

        update_info["loss_value_vl"] /= value_denom
        update_info["loss_value_vh"] /= value_denom

        update_info["grad_actor"] = grad_actor_total / actor_denom if actor_denom > 0 else 0.0
        update_info["grad_vl"] = grad_vl_total / value_denom if value_denom > 0 else 0.0
        update_info["grad_vh"] = grad_vh_total / value_denom if value_denom > 0 else 0.0
        for agent in self.agent_ids:
            key_suffix = agent.replace(" ", "_")
            update_info[f"entropy_coef_{key_suffix}"] = entropy_coef_per_actor[agent]

        self.buffer.reset()

        self._maybe_decay_lr(self.global_update_step)
        update_info.update(self._get_current_lrs())
        # global_step counts environment steps only; do not increment here
        self._log_update_stats(self.global_step, update_info)
        self.global_update_step += 1

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
            perm = torch.randperm(
                total_samples,
                device=torch.device("cpu"),
                generator=self._train_generator,
            )
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
        
        actual_env = getattr(self.env, "unwrapped", self.env)

        ones_mask = torch.ones(self.num_envs, 1, device=self.device)

        step_tracer = getattr(actual_env, "step_tracer", None)
        saved_enable = None
        saved_every = None
        if step_tracer is not None:
            saved_enable = getattr(step_tracer, "enable_console_logging", False)
            saved_every = getattr(step_tracer, "print_every_steps", 10)
            if not saved_enable:
                step_tracer.enable_console_logging = True
            if saved_every is None or saved_every <= 0:
                saved_every = 10
            step_tracer.print_every_steps = saved_every

        try:
            while not done and episode_length < 2000:
                z_candidates = []

                for agent in self.agent_ids:
                    def vh_eval_fn(z_query: torch.Tensor):
                        z_enc_query = self.z_encoder_vh(z_query)
                        vh_pred, _ = self.critics_vh[agent].value_step(
                            obs[agent],
                            z_enc_query,
                            self.rnn_states_vh[agent],
                            ones_mask,
                        )
                        return vh_pred

                    z_i_star = self.root_finder.solve(
                        vh_eval_fn=vh_eval_fn,
                        obs=obs[agent],
                        h_tgt=-0.05,
                    )
                    z_candidates.append(z_i_star)

                z_global = torch.max(torch.stack(z_candidates, dim=0), dim=0)[0]
                z_values.append(float(z_global[0].mean().item()))

                actions = {}
                z_enc_actor_global = self.z_encoder_actor(z_global)
                for agent in self.agent_ids:
                    act_a, _, next_h_a, _ = self.actors[agent].act_step(
                        obs[agent],
                        z_enc_actor_global,
                        self.rnn_states_actor[agent],
                        ones_mask,
                        deterministic=deterministic,
                    )
                    masked_action = torch.zeros_like(act_a)
                    masked_action[0] = act_a[0]
                    actions[agent] = masked_action
                    self.rnn_states_actor[agent] = next_h_a

                env_actions = self._scale_actions(actions)
                detail_payload = {
                    "applied_forces": {aid: env_actions[aid].clone() for aid in self.agent_ids},
                    "mean_actions": {aid: env_actions[aid].clone() for aid in self.agent_ids},
                    "noise_actions": {aid: torch.zeros_like(env_actions[aid]) for aid in self.agent_ids},
                    "deterministic": deterministic,
                }
                if hasattr(actual_env, "set_detail_actor_info"):
                    actual_env.set_detail_actor_info(detail_payload)

                if hasattr(actual_env, "_trainer_global_step"):
                    actual_env._trainer_global_step = self.global_step
                actual_env._last_z_snapshot = z_global.clone().detach()

                obs, rewards, terminated, truncated, info = self.env.step(env_actions)

                # Aggregate statistics (env0 only, averaged over agents)
                step_task_vals = []
                step_safe_vals = []
                for agent in self.agent_ids:
                    if "r_task" in info:
                        val = info["r_task"][agent]
                        if isinstance(val, torch.Tensor):
                            val = val.squeeze(-1) if val.dim() > 1 else val
                            step_task_vals.append(val[0])
                        else:
                            step_task_vals.append(torch.as_tensor(float(val), device=self.device))
                    if "r_safe" in info:
                        val = info["r_safe"][agent]
                        if isinstance(val, torch.Tensor):
                            val = val.squeeze(-1) if val.dim() > 1 else val
                            step_safe_vals.append(val[0])
                        else:
                            step_safe_vals.append(torch.as_tensor(float(val), device=self.device))

                if step_task_vals:
                    task_tensor = torch.stack(step_task_vals)
                    episode_task_return += float(task_tensor.mean().item())
                if step_safe_vals:
                    safe_tensor = torch.stack(step_safe_vals)
                    episode_safe_cost += float(safe_tensor.mean().item())

                if "is_violating" in info and info["is_violating"] is not None:
                    viol = info["is_violating"]
                    if isinstance(viol, torch.Tensor):
                        while viol.dim() > 2:
                            viol = viol.squeeze(-1)
                        viol_env0 = viol[0]
                        if isinstance(viol_env0, torch.Tensor):
                            episode_violations += int(viol_env0.float().sum().item())
                        else:
                            episode_violations += int(bool(viol_env0))
                    else:
                        episode_violations += int(bool(viol))

                episode_length += 1

                done_env0 = False
                for agent in self.agent_ids:
                    agent_term = terminated[agent]
                    agent_trunc = truncated[agent]
                    if isinstance(agent_term, torch.Tensor) and agent_term.dim() > 1:
                        agent_term = agent_term.squeeze(-1)
                    if isinstance(agent_trunc, torch.Tensor) and agent_trunc.dim() > 1:
                        agent_trunc = agent_trunc.squeeze(-1)
                    term_flag = bool(agent_term[0].item()) if isinstance(agent_term, torch.Tensor) else bool(agent_term)
                    trunc_flag = bool(agent_trunc[0].item()) if isinstance(agent_trunc, torch.Tensor) else bool(agent_trunc)
                    done_env0 = done_env0 or term_flag or trunc_flag
                done = done_env0

                share_obs_next = self._get_share_obs(obs)
                z_enc_vl = self.z_encoder_vl(z_global)
                _, self.rnn_states_vl = self.critic_vl.value_step(
                    share_obs_next,
                    z_enc_vl,
                    self.rnn_states_vl,
                    ones_mask,
                )
                z_enc_vh = self.z_encoder_vh(z_global)
                for agent in self.agent_ids:
                    _, self.rnn_states_vh[agent] = self.critics_vh[agent].value_step(
                        obs[agent],
                        z_enc_vh,
                        self.rnn_states_vh[agent],
                        ones_mask,
                    )

                if done:
                    break
        finally:
            if step_tracer is not None:
                if saved_enable is not None:
                    step_tracer.enable_console_logging = saved_enable
                if saved_every is not None:
                    step_tracer.print_every_steps = saved_every

        # Check success (environment 0)
        success = False
        if "progress_ratio" in info:
            progress_val = info["progress_ratio"]
            if isinstance(progress_val, torch.Tensor):
                prog = progress_val
                if prog.dim() > 1:
                    prog = prog.squeeze(-1)
                if prog.dim() == 0:
                    progress = float(prog.item())
                else:
                    progress = float(prog[0].item())
            else:
                progress = float(progress_val)
            success = (progress >= 0.95)
        
        return {
            "task_return": episode_task_return,
            "safe_cost_sum": episode_safe_cost,
            "length": episode_length,
            "success": success,
            "violations": episode_violations,
            "z_mean": np.mean(z_values) if z_values else 0.0,
        }
    
    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Run multiple evaluation episodes and aggregate statistics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Aggregated evaluation statistics
        """
        self.set_eval_mode()
        num_episodes = max(1, int(num_episodes))
        
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
        """Legacy interface retained for compatibility; milestones handled during rollouts."""
        return

    def _append_and_sort_milestones_index(self, index_path: str, milestone: int, score: float, rel_path: str) -> None:
        """Append milestone metadata and keep index sorted by milestone."""

        timestamp = datetime.now().isoformat()
        new_record = f"{timestamp}\tmilestone={milestone}\tscore={score:.6f}\tpath={rel_path}"

        records: List[str] = []
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if line:
                        records.append(line)

        records.append(new_record)

        def _extract_milestone(entry: str) -> int:
            for field in entry.split("\t"):
                if field.startswith("milestone="):
                    try:
                        return int(field.split("=", 1)[1])
                    except ValueError:
                        return 0
            return 0

        records_sorted = sorted(records, key=_extract_milestone)

        with open(index_path, "w") as f:
            for rec in records_sorted:
                f.write(rec + "\n")

    def save_checkpoint(self, path: str, global_step: Optional[int] = None, update_count: Optional[int] = None):
        """Save checkpoint with all networks, optimizers, and training state."""
        checkpoint = {
            "actors": {agent: self.actors[agent].state_dict() for agent in self.agent_ids},
            "critics_vh": {agent: self.critics_vh[agent].state_dict() for agent in self.agent_ids},
            "critic_vl": self.critic_vl.state_dict(),
            "z_encoder_actor": self.z_encoder_actor.state_dict(),
            "z_encoder_vl": self.z_encoder_vl.state_dict(),
            "z_encoder_vh": self.z_encoder_vh.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_vl": self.optimizer_vl.state_dict(),
            "optimizer_vh": self.optimizer_vh.state_dict(),
            "global_step": global_step if global_step is not None else self.global_step,
            "global_episodes": self.global_episodes,
            "episodes_done": self.episodes_done,
            "milestones": list(self.milestones),
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
        self.z_encoder_actor.load_state_dict(checkpoint["z_encoder_actor"])
        self.z_encoder_vl.load_state_dict(checkpoint["z_encoder_vl"])
        self.z_encoder_vh.load_state_dict(checkpoint["z_encoder_vh"])

        # Load optimizers
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_vl.load_state_dict(checkpoint["optimizer_vl"])
        self.optimizer_vh.load_state_dict(checkpoint["optimizer_vh"])
        
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
        for agent in self.agent_ids:
            self.actors[agent].train()
        self.z_encoder_vl.train()
        self.z_encoder_actor.train()
        self.z_encoder_vh.train()
        for agent in self.agent_ids:
            self.critics_vh[agent].train()
        self.critic_vl.train()

    def set_eval_mode(self):
        """Set all networks to eval mode."""
        for agent in self.agent_ids:
            self.actors[agent].eval()
        self.z_encoder_vl.eval()
        self.z_encoder_actor.eval()
        self.z_encoder_vh.eval()
        for agent in self.agent_ids:
            self.critics_vh[agent].eval()
        self.critic_vl.eval()
