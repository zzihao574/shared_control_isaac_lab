#!/usr/bin/env python3

"""
rMAPPO training script with dual independent networks.
Features global reproducibility, unified WandB through helpers, direct reset after eval.
"""

import sys
import os
import torch
import numpy as np
import random
import copy
import yaml
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from isaaclab.app import AppLauncher
from utils.training_helpers_rmappo import (
    WandBLogger, TrainingConfiguration, create_argument_parser, 
    MetricsHub, TopKModelManager, RMAPPOTrainingRunner, RMAPPOMilestoneEvaluator, 
    save_final_rmappo_networks
)


def setup_global_reproducibility(seed: int, strict_determinism: bool = True):
    """Setup global reproducibility."""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[SEED] Reproducibility set. seed={seed}, strict={strict_determinism}")


def setup_environment(args, config):
    """Create and configure the surgical robot environment."""
    from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
    import gymnasium as gym
    import surgical_project.envs.multi_agent
    
    env_cfg = SurgicalDirectMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    
    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg


def _select_mappo_block(cfg: dict) -> dict:
    """Select and validate mappo configuration block."""
    has_mappo = "mappo" in cfg and isinstance(cfg["mappo"], dict)
    has_ppo = "ppo" in cfg and isinstance(cfg["ppo"], dict)
    has_algo = "algo" in cfg and isinstance(cfg["algo"], dict)
    has_agents = "agents" in cfg and isinstance(cfg["agents"], dict)
    has_networks = "networks" in cfg and isinstance(cfg["networks"], dict)

    conflicting_blocks = []
    if has_ppo:
        conflicting_blocks.append("ppo")
    if has_algo:
        conflicting_blocks.append("algo")
    if has_agents:
        conflicting_blocks.append("agents")
    if has_networks:
        conflicting_blocks.append("networks")
    
    if has_mappo and conflicting_blocks:
        raise ValueError(f"[CONFIG ERROR] Found 'mappo' together with {conflicting_blocks}. Keep ONLY 'mappo:'")
    
    if not has_mappo:
        available_blocks = [k for k in ["ppo", "algo", "agents", "networks"] if k in cfg]
        if available_blocks:
            raise ValueError(f"[CONFIG ERROR] Found deprecated blocks {available_blocks} but missing 'mappo:'")
        else:
            raise ValueError("[CONFIG ERROR] Missing 'mappo:' block")

    return cfg["mappo"]


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


def load_dual_network_config(config_path: str):
    """Load YAML config with unified mappo block validation."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Support both old and new structures
    if "algorithms" in cfg and "rmappo" in cfg["algorithms"]:
        common = cfg["algorithms"]["rmappo"]
    elif "mappo" in cfg:
        common = cfg["mappo"]
    else:
        common = _select_mappo_block(cfg)
    
    _validate_mappo_args(common, "mappo")
    
    import copy
    human_config = copy.deepcopy(common)
    robot_config = copy.deepcopy(common)
    
    print(f"[CONFIG] Successfully loaded unified mappo configuration")
    
    return {
        "human": human_config,
        "robot": robot_config,
        "common": common,
        "raw_config": cfg
    }


def initialize_rmappo_algorithm(env, config, args, metrics_hub):
    """Create and initialize dual rMAPPO algorithm wrapper."""
    device = config.get_compute_device()
    
    num_envs = args.num_envs
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_envs'):
        num_envs = env.unwrapped.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs

    obs_dict, _ = env.reset()
    obs_dim = int(obs_dict["human"].shape[1])
    share_obs_dim = obs_dim * 2
    
    try:
        act_dim = int(env.unwrapped.action_space['human'].shape[0])
    except Exception:
        raise RuntimeError("[ERROR] Cannot infer action dimension from environment")

    print(f"[RMAPPO] Dual Network Architecture:")
    print(f"  Environments: {num_envs}")
    print(f"  Obs dim: {obs_dim}, Share obs dim: {share_obs_dim}, Action dim: {act_dim}")

    dual_config = load_dual_network_config(args.config)
    config.params['mappo_args'] = dual_config['common']
    
    rollout_horizon = config.params.get('rollout_horizon', 256)
    data_chunk_length = dual_config['common'].get('data_chunk_length', 16)
    if rollout_horizon % data_chunk_length != 0:
        raise ValueError(f"[CONFIG ERROR] rollout_horizon must be divisible by data_chunk_length")
    
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
        enable_console_logging=config.params.get("logging", {}).get("enable_console_logging", False)
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
        self.metrics_hub = metrics_hub
        self.args = args
        
        self.T = int(params.get('rollout_horizon', 256))
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

    def select_actions(self, observations: Dict[str, torch.Tensor], add_noise: bool, deterministic: bool = None, noise_scale: float = 1.0):
        """Generate actions from both networks."""
        obs_scaled = self.build_obs_scaled(observations)
        self._clear_obs_cache()
        
        actions_norm = {}
        action_log_probs = {}
        values = {}
        
        if deterministic is None:
            deterministic = self._is_eval_mode or not add_noise
        
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(obs_scaled, aid)
            masks = torch.ones(obs.shape[0], 1, device=self.device)
            
            with torch.no_grad():
                v, a, lp, rnn_a_new, rnn_c_new = self.policies[aid].get_actions(
                    share_obs, obs, 
                    self.rnn_states[aid]["actor"], self.rnn_states[aid]["critic"],
                    masks, deterministic=deterministic
                )
            
            actions_norm[aid] = a
            action_log_probs[aid] = lp
            values[aid] = v
            
            if not self._is_eval_mode:
                self.rnn_states[aid]["actor"] = rnn_a_new
                self.rnn_states[aid]["critic"] = rnn_c_new
        
        env_actions = self.actions_to_env_format(actions_norm)
        
        if not self._is_eval_mode:
            self._store_rollout_data(obs_scaled, actions_norm, action_log_probs, values)
        
        # Ensure both applied_forces and mean_actions always exist
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
            
            gamma = self.params.get('mappo_args', {}).get('gamma', 0.99)
            gae_lambda = self.params.get('mappo_args', {}).get('gae_lambda', 0.95)
            
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

    def store_next_obs(self, next_obs):
        """Store next observations for bootstrapping."""
        self._next_obs = next_obs


class TrainingOrchestrator:
    """Main rMAPPO trainer with dual network infrastructure."""
    
    def __init__(self, args):
        self.args = args
        print(f"[ORCHESTRATOR] Initializing Dual rMAPPO Training Orchestrator...")
        
        self._setup_configuration()
        self._setup_environment()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[ORCHESTRATOR] Initialization complete")

    def _setup_configuration(self):
        """Load and setup training configuration."""
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        setup_global_reproducibility(self.args.seed, strict_determinism=True)
        self.config.params['seed'] = self.args.seed
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
        actual_env = getattr(self.env, 'unwrapped', self.env)
        actual_env.params = self.config.params
        inject_step_tracer(self.env, self.config, self.args.num_envs)

    def _setup_training_components(self):
        """Initialize training components."""
        self.metrics_hub = MetricsHub()
        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        run_config = {**vars(self.args), **self.config.params}
        run_name = f"rmappo_dual_{self.args.num_envs}envs_{time.strftime('%Y%m%d_%H%M%S')}"
        self.wandb_logger.initialize_run(run_config, run_name)
        self.wandb_logger.attach_metrics_hub(self.metrics_hub)
        
        self.rmappo = initialize_rmappo_algorithm(self.env, self.config, self.args, self.metrics_hub)
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        mappo_max_steps = int(self.config.params.get('mappo_args', {}).get('max_global_steps', 200000))
        if self.args.max_global_steps > 0:
            self.max_global_steps = self.args.max_global_steps
        else:
            self.max_global_steps = mappo_max_steps
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and evaluator."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/rmappo_dual/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.runner = RMAPPOTrainingRunner(
            env=self.env,
            rmappo_wrapper=self.rmappo,
            metrics_hub=self.metrics_hub,
            agent_ids=self.rmappo.agent_ids,
            max_global_steps=self.max_global_steps
        )
        
        self.evaluator = RMAPPOMilestoneEvaluator(
            env=self.env,
            rmappo_wrapper=self.rmappo,
            topk_mgr=self.top_k_manager,
            metrics_hub=self.metrics_hub,
            log_dir=self.log_dir,
            agent_ids=self.rmappo.agent_ids
        )

    def _setup_milestone_management(self):
        """Initialize milestone tracking."""
        self.max_milestone_triggered = 0

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.runner.set_eval_mode(is_eval)
        self.rmappo.set_eval_mode(is_eval)

    def evaluate_milestone_if_due(self):
        """Check and trigger milestone evaluation."""
        if not self.milestone_episodes:
            return
            
        candidate = 0
        for milestone in sorted(self.milestone_episodes):
            if self.runner.global_episodes >= milestone:
                candidate = milestone
            else:
                break
                
        if candidate > self.max_milestone_triggered:
            print(f"[MILESTONE] Crossed threshold: episodes {self.runner.global_episodes} >= milestone {candidate}")
            
            self.set_eval_mode(True)
            result = self.evaluator.run_evaluation(candidate, self.runner.global_step)
            if result.get("skip_episode_once", False):
                self.runner.mark_skip_episode_once()
            
            self.set_eval_mode(False)
            self.metrics_hub.push_scalars({"lifecycle/eval_to_train": 1}, step=self.runner.global_step)
            
            obs_dict, _ = self.env.reset()
            self.runner._current_obs = obs_dict
            
            self.max_milestone_triggered = candidate

    def train(self):
        """Main training loop."""
        print(f"[TRAIN] Starting dual rMAPPO training")
        
        obs_dict, _ = self.env.reset()
        self.runner._current_obs = obs_dict
        
        while self.runner.global_step < self.max_global_steps:
            self.runner.execute_training_step()
            self.evaluate_milestone_if_due()
            
            if self.runner.global_step > 0 and self.runner.global_step % 2000 == 0:
                print(f"[Step {self.runner.global_step}] Episodes: {self.runner.global_episodes}")
            
            if self.runner.global_step >= self.max_global_steps:
                break
        
        print(f"\n[TRAINING COMPLETE]")
        
        save_final_rmappo_networks(
            log_directory=self.log_dir,
            rmappo_wrapper=self.rmappo,
            global_step=self.runner.global_step,
            global_episodes=self.runner.global_episodes,
            max_milestone_triggered=self.max_milestone_triggered
        )
        
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.finalize_run()


def main():
    """Main entry point."""
    print("="*80)
    print("Dual rMAPPO Training")
    print("="*80)
    
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    trainer = TrainingOrchestrator(args_cli)
    trainer.train()
    
    simulation_app.close()


if __name__ == "__main__":
    main()