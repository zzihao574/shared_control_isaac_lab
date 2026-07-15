#!/usr/bin/env python3

"""
rMAPPO training script with dual independent networks.
Features global reproducibility, unified WandB through helpers, mid-rollout evaluation support.
MODIFIED: Evaluation now handled within runner for mid-rollout trigger support.
FIXED: Simplified action selection interface, removed TopKManager, optimized config loading.
"""

import sys
import os
import json
import subprocess
import torch
import yaml
from datetime import datetime
from typing import Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from isaaclab.app import AppLauncher
from utils.training_helpers_rmappo import (
    WandBLogger, create_argument_parser,
    MetricsHub, RMAPPOTrainingRunner, RMAPPOMilestoneEvaluator, 
    resume_from_checkpoint, save_final_checkpoint_rmappo,
    RMAPPOSeedPlan, build_rmappo_wandb_config, resolve_rmappo_config,
    setup_global_reproducibility,
)


def setup_environment(args, config):
    """Create and configure the surgical robot environment."""
    from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
    import gymnasium as gym
    import surgical_project.envs.multi_agent
    
    env_cfg = SurgicalDirectMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.params = config.params
    
    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg


def _validate_mappo_args(args: dict, agent_id: str):
    """Validate required mappo arguments."""
    required = ["hidden_size", "recurrent_N", "actor_lr", "critic_lr",
                "clip_param", "ppo_epoch", "num_mini_batch", "data_chunk_length",
                "entropy_coef", "max_grad_norm_actor", "max_grad_norm_critic",
                "opt_eps", "gamma", "gae_lambda", "max_global_steps"]
    
    missing = [k for k in required if k not in args]
    if missing:
        raise ValueError(f"[CONFIG ERROR] Missing required keys for {agent_id}: {missing}")
    
    if args.get("data_chunk_length", 0) <= 0:
        raise ValueError(f"[CONFIG ERROR] data_chunk_length must be > 0")
    
    if args.get("hidden_size", 0) <= 0:
        raise ValueError(f"[CONFIG ERROR] hidden_size must be > 0")


def build_dual_network_config_from_params(params: dict):
    """
    Build dual network configuration from already-loaded params dictionary.
    Avoids redundant YAML file reading.
    
    Args:
        params: Already-loaded configuration dictionary from TrainingConfiguration
        
    Returns:
        Dictionary with human/robot/common configurations
    """
    try:
        common = params["algorithms"]["rmappo"]
    except Exception:
        raise ValueError("[CONFIG ERROR] Missing 'algorithms.rmappo' in loaded params")
    
    _validate_mappo_args(common, "mappo")
    
    import copy
    human_config = copy.deepcopy(common)
    robot_config = copy.deepcopy(common)
    lr_decay = copy.deepcopy(params.get("training", {}).get("lr_decay", {}))
    human_config["lr_decay"] = lr_decay
    robot_config["lr_decay"] = copy.deepcopy(lr_decay)
    
    print(f"[CONFIG] Successfully built dual network configuration from params")
    
    return {
        "human": human_config,
        "robot": robot_config,
        "common": common
    }


def initialize_rmappo_algorithm(env, config, args, metrics_hub):
    """Create and initialize dual rMAPPO algorithm wrapper."""
    device = config.get_compute_device()
    actual_env = getattr(env, "unwrapped", env)
    try:
        num_envs = int(actual_env.num_envs)
        obs_dim = int(actual_env.cfg.observation_spaces["human"])
        act_dim = int(actual_env.cfg.action_spaces["human"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "[ERROR] Cannot infer rMAPPO dimensions from environment cfg"
        ) from exc
    share_obs_dim = obs_dim * 2

    print(f"[RMAPPO] Dual Network Architecture:")
    print(f"  Environments: {num_envs}")
    print(f"  Obs dim: {obs_dim}, Share obs dim: {share_obs_dim}, Action dim: {act_dim}")

    dual_config = build_dual_network_config_from_params(config.params)
    
    rollout_horizon = dual_config['common'].get('rollout_horizon', 256)
    data_chunk_length = dual_config['common'].get('data_chunk_length', 16)
    if rollout_horizon % data_chunk_length != 0:
        raise ValueError(f"[CONFIG ERROR] rollout_horizon must be divisible by data_chunk_length")
    
    RMAPPOSeedPlan(config.params["seed"]).apply_network_seed()
    return DualRMAPPOWrapper(
        dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, config.params, metrics_hub, args
    )


def inject_step_tracer(env, config, num_envs):
    """Inject StepTracer into environment."""
    actual_env = getattr(env, "unwrapped", env)
    
    from surgical_project.envs.multi_agent.utils import StepTracer
    
    actual_env.step_tracer = StepTracer(
        num_envs=num_envs,
        device=getattr(actual_env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        enable_console_logging=config.params.get("logging", {}).get("enable_console_logging", False),
        print_every_steps=int(
            config.params.get("logging", {}).get("print_every_steps", 10)
        ),
    )


class DualRMAPPOWrapper:
    """Dual independent network rMAPPO wrapper."""
    
    def __init__(self, dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, params, metrics_hub, args):
        from surgical_project.algorithms.marl.rmappo.r_mappo_core import RMAPPOPolicy, RMAPPOAlgorithm
        from surgical_project.algorithms.marl.rmappo.rollout_buffer import SharedRolloutBuffer
        
        self.device = device
        self.num_envs = num_envs
        self.params = params
        self.agent_ids = ["human", "robot"]
        self.human_model_type = str(params.get("human_model_type", "learnable"))
        self.trainable_agent_ids = (
            ["robot"]
            if self.human_model_type == "fixed_impedance"
            else list(self.agent_ids)
        )
        self.metrics_hub = metrics_hub
        self.args = args

        self.rmappo_cfg = params["algorithms"]["rmappo"]
        self.T = int(self.rmappo_cfg.get('rollout_horizon', 256))
        self.rollout_step = 0
        self.train_updates = 0
        self._is_eval_mode = False
        
        constraints = params.get('constraints', {})
        self.max_robot_force = float(constraints.get('max_robot_force', 0.04))
        self.max_human_force = float(constraints.get('max_human_force', 0.04))
        
        obs_scaling_config = params.get('obs_scaling', {})
        self.obs_scale_factors = {}
        if 'factors' in obs_scaling_config:
            factors_list = obs_scaling_config['factors']
            if len(factors_list) != obs_dim:
                raise ValueError(f"obs_scaling.factors length mismatch")
            
            scale_tensor = torch.tensor(factors_list, device=device, dtype=torch.float32)
            self.obs_scale_factors['human'] = scale_tensor
            self.obs_scale_factors['robot'] = scale_tensor
        else:
            self.obs_scale_factors['human'] = torch.ones(obs_dim, device=device, dtype=torch.float32)
            self.obs_scale_factors['robot'] = torch.ones(obs_dim, device=device, dtype=torch.float32)

        print(f"[WRAPPER] obs_scaling factors length={len(self.obs_scale_factors['human'])} (obs_dim={obs_dim})")

        obs_space_desc = {'shape': (obs_dim,)}
        cent_obs_space_desc = {'shape': (share_obs_dim,)}
        act_space_desc = {'shape': (act_dim,)}
        
        self.policies = {}
        self.trainers = {}
        self.buffers = {}
        self.rnn_states = {}
        
        # Initialize human network
        human_config = dual_config["human"]
        human_config['metrics_hub'] = metrics_hub
        human_config['args'] = args
        pol_h = RMAPPOPolicy(obs_space_desc, cent_obs_space_desc, act_space_desc, device, human_config)
        trn_h = RMAPPOAlgorithm(args=human_config, policy=pol_h, device=device)
        buf_h = SharedRolloutBuffer(
            T=self.T, N=num_envs, obs_dim=obs_dim, share_obs_dim=share_obs_dim,
            act_dim=act_dim, rnn_hidden_dim=human_config.get('hidden_size', 256), device=device
        )
        
        self.policies["human"] = pol_h
        self.trainers["human"] = trn_h 
        self.buffers["human"] = buf_h
        self.rnn_states["human"] = {
            "actor": torch.zeros(num_envs, human_config.get('hidden_size', 256), device=device),
            "critic": torch.zeros(num_envs, human_config.get('hidden_size', 256), device=device),
        }
        
        # Initialize robot network and copy weights
        robot_config = dual_config["robot"]
        robot_config['metrics_hub'] = metrics_hub
        robot_config['args'] = args
        pol_r = RMAPPOPolicy(obs_space_desc, cent_obs_space_desc, act_space_desc, device, robot_config)
        trn_r = RMAPPOAlgorithm(args=robot_config, policy=pol_r, device=device)
        buf_r = SharedRolloutBuffer(
            T=self.T, N=num_envs, obs_dim=obs_dim, share_obs_dim=share_obs_dim,
            act_dim=act_dim, rnn_hidden_dim=robot_config.get('hidden_size', 256), device=device
        )
        
        pol_r.actor.load_state_dict(pol_h.actor.state_dict())
        pol_r.critic.load_state_dict(pol_h.critic.state_dict())
        
        self.policies["robot"] = pol_r
        self.trainers["robot"] = trn_r
        self.buffers["robot"] = buf_r
        self.rnn_states["robot"] = {
            "actor": torch.zeros(num_envs, robot_config.get('hidden_size', 256), device=device),
            "critic": torch.zeros(num_envs, robot_config.get('hidden_size', 256), device=device),
        }
        
        print(f"[DUAL RMAPPO] Initialized:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  Networks: independent human & robot")
        print(f"  Initial weights: robot copied from human")
        print(f"  Human model: {self.human_model_type}")
        print(f"  Trainable agents: {self.trainable_agent_ids}")
        
        self.train_generator = None

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self._is_eval_mode = is_eval
        for aid in self.agent_ids:
            if is_eval:
                self.trainers[aid].prep_rollout()
            else:
                self.trainers[aid].prep_training()

    def build_obs_scaled(self, obs_raw):
        """Build scaled observations."""
        obs_scaled = {}
        for aid in self.agent_ids:
            if aid in obs_raw:
                obs_scaled[aid] = obs_raw[aid] * self.obs_scale_factors[aid].unsqueeze(0)
            else:
                raise ValueError(f"Agent {aid} not found in obs_raw")
        return obs_scaled

    def build_obs_tensors(self, obs_dict, agent_id: str):
        """Convert agent-specific obs dict to tensors."""
        obs = torch.as_tensor(obs_dict[agent_id], device=self.device, dtype=torch.float32)
        
        if not hasattr(self, '_cached_share_obs') or self._cached_share_obs is None:
            human_obs = torch.as_tensor(obs_dict["human"], device=self.device, dtype=torch.float32)
            robot_obs = torch.as_tensor(obs_dict["robot"], device=self.device, dtype=torch.float32)
            self._cached_share_obs = torch.cat([human_obs, robot_obs], dim=1)
        
        return obs, self._cached_share_obs
    
    def _clear_obs_cache(self):
        """Clear shared observation cache."""
        self._cached_share_obs = None

    def actions_to_env_format(self, actions_dict):
        """Convert normalized actions to environment format."""
        env_actions = {}
        force_limits = {"human": self.max_human_force, "robot": self.max_robot_force}
        
        for aid, actions_norm in actions_dict.items():
            env_actions[aid] = actions_norm * force_limits[aid]
            
        return env_actions

    def select_actions(self, observations: Dict[str, torch.Tensor], deterministic: bool):
        """
        Generate actions from both networks.
        
        FIXED: Simplified interface - only deterministic parameter.
        - deterministic=False: Sample from policy (training/exploration)
        - deterministic=True: Use mean action (evaluation/greedy)
        
        Args:
            observations: Dictionary of observations per agent
            deterministic: Whether to use deterministic (mean) actions
            
        Returns:
            env_actions: Dictionary of actions scaled to environment force limits
            detail: Dictionary with action details for logging
                - applied_forces: Actual forces applied
                - mean_actions: Policy mean actions
                - noise_actions: Always zero (kept for environment compatibility)
                - deterministic: Whether deterministic mode was used
        """
        obs_scaled = self.build_obs_scaled(observations)
        self._clear_obs_cache()
        
        actions_norm = {}
        action_log_probs = {}
        values = {}
        
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(obs_scaled, aid)
            masks = torch.ones(obs.shape[0], 1, device=self.device)

            if aid == "human" and self.human_model_type == "fixed_impedance":
                a = torch.zeros(obs.shape[0], self.policies[aid].act_dim, device=self.device)
                lp = torch.zeros(obs.shape[0], 1, device=self.device)
                v = torch.zeros(obs.shape[0], 1, device=self.device)
                rnn_a_new = self.rnn_states[aid]["actor"]
                rnn_c_new = self.rnn_states[aid]["critic"]
            else:
                with torch.no_grad():
                    v, a, lp, rnn_a_new, rnn_c_new = self.policies[aid].get_actions(
                        share_obs, obs,
                        self.rnn_states[aid]["actor"], self.rnn_states[aid]["critic"],
                        masks, deterministic=deterministic
                    )
            
            actions_norm[aid] = a
            action_log_probs[aid] = lp
            values[aid] = v
            
            self.rnn_states[aid]["actor"] = rnn_a_new
            self.rnn_states[aid]["critic"] = rnn_c_new
        
        env_actions = self.actions_to_env_format(actions_norm)
        
        if not self._is_eval_mode:
            self._store_rollout_data(obs_scaled, actions_norm, action_log_probs, values)
        
        # Build detail dictionary for environment
        # Note: noise_actions kept for environment compatibility (always zero now)
        detail = {
            "applied_forces": {k: v.detach().clone() for k, v in env_actions.items()},
            "mean_actions": {k: v.detach().clone() for k, v in env_actions.items()},
            "noise_actions": {aid: torch.zeros_like(env_actions[aid]) for aid in self.agent_ids},
            "deterministic": bool(deterministic)
        }
        
        return env_actions, detail

    def _store_rollout_data(self, observations_scaled, actions_norm, action_log_probs, values):
        """Store data for current rollout step."""
        self._current_step_data = {}
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(observations_scaled, aid)
            masks = torch.ones(obs.shape[0], 1, device=self.device)
            
            self._current_step_data[aid] = {
                'obs': obs.detach(),
                'share_obs': share_obs.detach(),
                'actions': actions_norm[aid].detach(), 
                'action_log_probs': action_log_probs[aid].detach(),
                'value_preds': values[aid].detach(),
                'masks': masks,
                'rnn_states_actor': self.rnn_states[aid]["actor"].detach().clone(),
                'rnn_states_critic': self.rnn_states[aid]["critic"].detach().clone()
            }

    def add_experience_to_buffer(self, obs, actions, rewards, next_obs, dones, terminated=None, truncated=None, infos=None):
        """Add experience to per-agent rollout buffers."""
        if self._is_eval_mode:
            return
            
        for aid in self.agent_ids:
            reward_tensor = rewards[aid].unsqueeze(-1) if len(rewards[aid].shape) == 1 else rewards[aid]
            mask_t = (1.0 - dones[aid].float()).view(-1, 1)
            
            if terminated is not None and truncated is not None:
                is_terminal = terminated[aid]
                term_mask_t = (~is_terminal).float().view(-1, 1)
            else:
                term_mask_t = torch.ones_like(mask_t)
            
            self.buffers[aid].insert(
                t=self.rollout_step,
                obs=self._current_step_data[aid]['obs'],
                share_obs=self._current_step_data[aid]['share_obs'],
                actions=self._current_step_data[aid]['actions'],
                action_log_probs=self._current_step_data[aid]['action_log_probs'],
                value_preds=self._current_step_data[aid]['value_preds'],
                rewards=reward_tensor,
                masks=mask_t,
                rnn_states_actor=self._current_step_data[aid]['rnn_states_actor'],
                rnn_states_critic=self._current_step_data[aid]['rnn_states_critic'],
                term_masks=term_mask_t
            )
            
            done_indices = dones[aid].nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                self.rnn_states[aid]["actor"][done_indices].zero_()
                self.rnn_states[aid]["critic"][done_indices].zero_()
        
        self.rollout_step += 1

    def update(self):
        """Perform dual rMAPPO update."""
        if self._is_eval_mode:
            return {}
            
        if self.rollout_step < self.T:
            return {}
        
        stats = {}
        
        for aid in self.agent_ids:
            if aid not in self.trainable_agent_ids:
                self.buffers[aid].after_update()
                continue

            next_obs_dict = getattr(self, '_next_obs', None)
            if next_obs_dict is not None:
                next_obs_scaled = self.build_obs_scaled(next_obs_dict)
                _, share_obs = self.build_obs_tensors(next_obs_scaled, aid)
                masks = torch.ones(share_obs.shape[0], 1, device=self.device)
                
                with torch.no_grad():
                    last_values = self.policies[aid].get_values(
                        share_obs, self.rnn_states[aid]["critic"], masks
                    )
            else:
                last_values = torch.zeros(self.num_envs, 1, device=self.device)
            
            gamma = self.rmappo_cfg.get('gamma', 0.99)
            gae_lambda = self.rmappo_cfg.get('gae_lambda', 0.95)
            
            self.buffers[aid].compute_returns_and_adv(last_values, gamma, gae_lambda)
            
            train_info = self.trainers[aid].train(
                self.buffers[aid],
                generator=self.train_generator
            )
            
            self.buffers[aid].after_update()
            
            for k, v in train_info.items():
                stats[f"{k}/{aid}"] = v
        
        self.train_updates += 1
        self.rollout_step = 0
        
        return stats

    def human_actor_checksum(self) -> float:
        """Return a compact checksum for fixed-human immutability checks."""
        actor = self.policies["human"].actor
        return float(sum(p.detach().double().sum().item() for p in actor.parameters()))

    def store_next_obs(self, next_obs):
        """Store next observations for bootstrapping."""
        self._next_obs = next_obs


class TrainingOrchestrator:
    """Main rMAPPO trainer with dual network infrastructure and mid-rollout evaluation support."""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[ORCHESTRATOR] Initializing Dual rMAPPO Training Orchestrator...")
        
        self._setup_configuration()
        self._setup_environment()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        
        print(f"[ORCHESTRATOR] Initialization complete")

    def _setup_configuration(self):
        """Finalize bookkeeping derived from the already-resolved config."""
        if not self.args.checkpoint:
            self._apply_per_env_scaling()

    def _apply_per_env_scaling(self):
        """Scale parameters by number of environments."""
        num_envs = int(self.args.num_envs)
        monitor_cfg = self.config.params.get("training_monitor", {})
        base_milestones = list(monitor_cfg.get("milestone_episodes", []))
        scaled_milestones = [int(m * num_envs) for m in base_milestones]
        monitor_cfg["milestone_episodes"] = scaled_milestones
        self.config.params["training_monitor"] = monitor_cfg

    def _setup_environment(self):
        """Create and configure the environment."""
        self.env, self.env_cfg = setup_environment(self.args, self.config)
        inject_step_tracer(self.env, self.config, self.args.num_envs)

    def _setup_training_components(self):
        """Initialize training components."""
        self.metrics_hub = MetricsHub()
        self.rmappo = initialize_rmappo_algorithm(self.env, self.config, self.args, self.metrics_hub)

        rmappo_cfg = self.config.params["algorithms"]["rmappo"]
        self.max_global_steps = int(rmappo_cfg["max_global_steps"])
        human_model_type = self.config.params.get("human_model_type", "learnable")
        self.log_dir = self.args.run_dir or os.path.join(
            "logs", "rmappo_dual", human_model_type, self.timestamp
        )
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                text=True,
            ).strip()
        except Exception:
            git_commit = "unknown"
        self.config.params["git_commit"] = git_commit

        resolved_config_path = os.path.join(self.log_dir, "resolved_config.yaml")
        with open(resolved_config_path, "w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                self.config.params,
                config_file,
                sort_keys=False,
                allow_unicode=True,
            )
        manifest = {
            "algorithm": "rmappo",
            "human_model_type": human_model_type,
            "seed": self.args.seed,
            "num_envs": self.args.num_envs,
            "max_global_steps": self.max_global_steps,
            "git_commit": git_commit,
            "resolved_config": resolved_config_path,
        }
        with open(
            os.path.join(self.log_dir, "run_manifest.json"), "w", encoding="utf-8"
        ) as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            actual_env = getattr(self.env, "unwrapped", self.env)
            human_metadata = actual_env.human_force_controller.wandb_metadata()
            run_config = build_rmappo_wandb_config(
                resolved_config=self.config.params,
                runtime={
                    "seed": self.args.seed,
                    "num_envs": self.args.num_envs,
                    "max_global_steps": self.max_global_steps,
                },
                human_metadata=human_metadata,
                git_commit=git_commit,
            )
            run_name = (
                f"rmappo_{human_model_type}_seed{self.args.seed}_{self.timestamp}"
            )
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")
        self.wandb_logger.attach_metrics_hub(self.metrics_hub)

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and evaluator with cross-references."""
        self.runner = RMAPPOTrainingRunner(
            env=self.env,
            rmappo_wrapper=self.rmappo,
            metrics_hub=self.metrics_hub,
            agent_ids=self.rmappo.agent_ids,
            max_global_steps=self.max_global_steps,
            evaluator=None
        )

        self.evaluator = RMAPPOMilestoneEvaluator(
            env=self.env,
            rmappo_wrapper=self.rmappo,
            metrics_hub=self.metrics_hub,
            log_dir=self.log_dir,
            agent_ids=self.rmappo.agent_ids,
            runner=self.runner
        )

        self.runner.evaluator = self.evaluator

        if self.args.checkpoint:
            resume_from_checkpoint(
                self.args.checkpoint, 
                self.rmappo, 
                self.runner, 
                device=self.config.get_compute_device()
            )
            print(f"[SETUP] Resumed from checkpoint: {self.args.checkpoint}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.runner.set_eval_mode(is_eval)
        self.rmappo.set_eval_mode(is_eval)

    def train(self):
        """Main training loop - simplified as evaluation is now handled in runner."""
        print(f"[TRAIN] Starting dual rMAPPO training with mid-rollout evaluation support")

        obs_dict, _ = self.env.reset()
        self.runner._current_obs = obs_dict
        
        while self.runner.global_step < self.max_global_steps:
            # ============ SIMPLIFIED: Runner handles evaluation internally ============
            self.runner.execute_training_step()
            # ==========================================================================
            
            if self.runner.global_step > 0 and self.runner.global_step % 2000 == 0:
                print(f"[Step {self.runner.global_step}] Episodes: {self.runner.global_episodes}")
            
            if self.runner.global_step >= self.max_global_steps:
                break
        
        print(f"\n[TRAINING COMPLETE]")
        print(f"  Final steps: {self.runner.global_step}")
        print(f"  Final episodes: {self.runner.global_episodes}")
        print(f"  Training updates: {self.rmappo.train_updates}")
        print(f"  Last milestone: {self.runner.max_milestone_triggered}")
        final_checkpoint = save_final_checkpoint_rmappo(
            self.ckpt_dir, self.rmappo, self.runner
        )
        print(f"\n[INFO] All checkpoints saved to: {self.ckpt_dir}")
        print(f"[INFO] Final checkpoint: {final_checkpoint}")
        print(f"[INFO] Check milestones_index.txt for saved checkpoints list")

    def close(self):
        """Close training resources once, including after an exception."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "env"):
            self.env.close()
        if hasattr(self, "wandb_logger"):
            self.wandb_logger.finalize_run()


def main():
    """Main entry point."""
    print("="*80)
    print("Dual rMAPPO Training with Mid-Rollout Evaluation")
    print("="*80)
    
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    resolved_config = resolve_rmappo_config(args_cli)
    setup_global_reproducibility(args_cli.seed, strict_determinism=True)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    trainer = None
    try:
        trainer = TrainingOrchestrator(args_cli, resolved_config)
        trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
