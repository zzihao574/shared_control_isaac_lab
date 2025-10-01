#!/usr/bin/env python3

"""
rMAPPO training script with dual independent networks integration.
Features unified configuration source and enhanced error reporting.
MODIFIED: Removed all finite_check functions, relying on PyTorch natural failure + wandb monitoring.
STABLE: Gradient clipping with separate thresholds, unified WandB initialization.
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

# WandB support with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("[WARNING] WandB not available. Install with: pip install wandb")


def setup_global_reproducibility(seed: int, strict_determinism: bool = False):
    """Setup global reproducibility for consistent training results."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] Strict determinism enabled (may slow down training)")
    
    print(f"[SEED] Global reproducibility set: seed={seed}")


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
    has_ppo   = "ppo"   in cfg and isinstance(cfg["ppo"], dict)
    has_algo  = "algo"  in cfg and isinstance(cfg["algo"], dict)
    has_agents = "agents" in cfg and isinstance(cfg["agents"], dict)
    has_networks = "networks" in cfg and isinstance(cfg["networks"], dict)

    # Only allow mappo to exist
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
        raise ValueError(f"[CONFIG ERROR] Found 'mappo' together with {conflicting_blocks}. "
                        f"Keep ONLY 'mappo:' as the single source of hyperparameters.")
    
    if not has_mappo:
        available_blocks = [k for k in ["ppo", "algo", "agents", "networks"] if k in cfg]
        if available_blocks:
            raise ValueError(f"[CONFIG ERROR] Found deprecated blocks {available_blocks} but missing 'mappo:'. "
                           f"Please rename your algorithm block to 'mappo:' and remove others.")
        else:
            raise ValueError("[CONFIG ERROR] Missing 'mappo:' block. Please define it as the single source of hyperparameters.")

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
    
    # Additional validation
    if args.get("data_chunk_length", 0) <= 0:
        raise ValueError(f"[CONFIG ERROR] data_chunk_length must be > 0, got {args.get('data_chunk_length')}")
    
    if args.get("hidden_size", 0) <= 0:
        raise ValueError(f"[CONFIG ERROR] hidden_size must be > 0, got {args.get('hidden_size')}")


def load_dual_network_config(config_path: str):
    """Load YAML config with unified mappo block validation."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Select and validate mappo block
    common = _select_mappo_block(cfg)
    
    # Strong validation
    _validate_mappo_args(common, "mappo")
    
    # Human/robot completely consistent: return dual copy
    import copy
    human_config = copy.deepcopy(common)
    robot_config = copy.deepcopy(common)
    
    print(f"[CONFIG] Successfully loaded unified mappo configuration:")
    print(f"  Actor LR: {common.get('actor_lr')}")
    print(f"  Hidden size: {common.get('hidden_size')}")
    print(f"  Max steps: {common.get('max_global_steps')}")
    print(f"  Gradient clipping: actor={common.get('max_grad_norm_actor')}, critic={common.get('max_grad_norm_critic')}")
    print(f"  Networks: independent human & robot (identical initialization)")
    
    return {
        "human": human_config,
        "robot": robot_config,
        "common": common,
        "raw_config": cfg
    }


def make_wandb_logger():
    """Create unified WandB logger function for gradient monitoring."""
    def _log_fn(d: dict, step: int = None):
        if step is not None:
            d = {"global_step": step, **d}
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(d)
    return _log_fn


def initialize_rmappo_algorithm(env, config, args, log_fn):
    """Create and initialize dual rMAPPO algorithm wrapper with logging support."""
    device = config.get_compute_device()
    
    # Get environment count and dimensions
    num_envs = args.num_envs
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_envs'):
        num_envs = env.unwrapped.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs

    # Get dimensions from environment
    obs_dict, _ = env.reset()
    obs_dim = int(obs_dict["human"].shape[1])
    share_obs_dim = obs_dim * 2  # Centralized: human||robot
    
    # Get action dimension from environment
    try:
        act_dim = int(env.unwrapped.action_space['human'].shape[0])
    except Exception:
        act_dim = 3  # Fallback to default
        print(f"[WARNING] Could not determine action dimension from env, using default: {act_dim}")

    print(f"[RMAPPO] Dual Network Architecture:")
    print(f"  Environments: {num_envs}")
    print(f"  Agents: human, robot (independent networks)")
    print(f"  Obs dim: {obs_dim}, Share obs dim: {share_obs_dim}")
    print(f"  Action dim: {act_dim}")

    # Load dual config with unified mappo source
    dual_config = load_dual_network_config(args.config)
    
    # Add mappo_args to params for unified access
    config.params['mappo_args'] = dual_config['common']
    
    # Validate rollout horizon compatibility
    rollout_horizon = config.params.get('rollout_horizon', 256)
    data_chunk_length = dual_config['common'].get('data_chunk_length', 16)
    if rollout_horizon % data_chunk_length != 0:
        raise ValueError(f"[CONFIG ERROR] rollout_horizon ({rollout_horizon}) must be divisible by "
                        f"data_chunk_length ({data_chunk_length}) for RNN training")
    
    # Create wrapper with dual configs and logging support
    return DualRMAPPOWrapper(
        dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, config.params, log_fn, args
    )


def inject_step_tracer(env, config, num_envs):
    """Inject StepTracer into environment for console logging."""
    actual_env = getattr(env, "unwrapped", env)
    
    from surgical_project.envs.multi_agent.utils import StepTracer
    
    actual_env.step_tracer = StepTracer(
        num_envs=num_envs,
        device=getattr(actual_env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        enable_console_logging=config.params.get("logging", {}).get("enable_console_logging", False)
    )
    
    print(f"[INFO] StepTracer injected:")
    print(f"  Console logging: {'enabled' if actual_env.step_tracer.enable_console_logging else 'disabled'}")


class DualRMAPPOWrapper:
    """Dual independent network rMAPPO wrapper with gradient clipping."""
    
    def __init__(self, dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, params, log_fn, args):
        # Import algorithm layer
        from surgical_project.algorithms.marl.rmappo.r_mappo_core import RMAPPOPolicy, RMAPPOAlgorithm
        from surgical_project.algorithms.marl.rmappo.rollout_buffer import SharedRolloutBuffer
        
        self.device = device
        self.num_envs = num_envs
        self.params = params
        self.agent_ids = ["human", "robot"]
        self.log_fn = log_fn
        self.args = args
        
        # Rollout parameters
        self.T = int(params.get('rollout_horizon', 256))
        self.rollout_step = 0
        self.train_updates = 0
        
        # Evaluation mode control
        self._is_eval_mode = False
        
        # Force constraints for physical scaling
        constraints = params.get('constraints', {})
        self.max_robot_force = float(constraints.get('max_robot_force', 0.04))
        self.max_human_force = float(constraints.get('max_human_force', 0.04))
        
        # Load obs scaling factors
        obs_scaling_config = params.get('obs_scaling', {})
        self.obs_scale_factors = {}
        if 'factors' in obs_scaling_config:
            factors_list = obs_scaling_config['factors']
            if len(factors_list) != obs_dim:
                raise ValueError(f"obs_scaling.factors length ({len(factors_list)}) != obs_dim ({obs_dim})")
            
            scale_tensor = torch.tensor(factors_list, device=device, dtype=torch.float32)
            self.obs_scale_factors['human'] = scale_tensor
            self.obs_scale_factors['robot'] = scale_tensor
            
            print(f"[OBS SCALING] Loaded scaling factors: {factors_list}")
        else:
            # No scaling - use identity
            self.obs_scale_factors['human'] = torch.ones(obs_dim, device=device, dtype=torch.float32)
            self.obs_scale_factors['robot'] = torch.ones(obs_dim, device=device, dtype=torch.float32)
            print(f"[OBS SCALING] No scaling factors found, using identity scaling")
        
        # Create space descriptors
        obs_space_desc = {'shape': (obs_dim,)}
        cent_obs_space_desc = {'shape': (share_obs_dim,)}
        act_space_desc = {'shape': (act_dim,)}
        
        # Initialize dual networks
        self.policies = {}
        self.trainers = {}
        self.buffers = {}
        self.rnn_states = {}
        
        # Initialize human network first
        human_config = dual_config["human"]
        human_config['log_fn'] = log_fn
        human_config['args'] = args
        pol_h = RMAPPOPolicy(
            obs_space_desc=obs_space_desc,
            cent_obs_space_desc=cent_obs_space_desc, 
            act_space_desc=act_space_desc,
            device=device,
            args=human_config
        )
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
        
        # Initialize robot network and copy weights from human
        robot_config = dual_config["robot"]
        robot_config['log_fn'] = log_fn
        robot_config['args'] = args
        pol_r = RMAPPOPolicy(
            obs_space_desc=obs_space_desc,
            cent_obs_space_desc=cent_obs_space_desc,
            act_space_desc=act_space_desc, 
            device=device,
            args=robot_config
        )
        trn_r = RMAPPOAlgorithm(args=robot_config, policy=pol_r, device=device)
        buf_r = SharedRolloutBuffer(
            T=self.T, N=num_envs, obs_dim=obs_dim, share_obs_dim=share_obs_dim,
            act_dim=act_dim, rnn_hidden_dim=robot_config.get('hidden_size', 256), device=device
        )
        
        # Copy human weights to robot for identical initialization
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
        print(f"  Force limits: robot={self.max_robot_force}, human={self.max_human_force}")
        print(f"  Action domain: Tanh-Gaussian [-1, 1] -> physical forces")
        print(f"  Gradient clipping: actor={human_config.get('max_grad_norm_actor')}, critic={human_config.get('max_grad_norm_critic')}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode for both networks."""
        self._is_eval_mode = is_eval
        for aid in self.agent_ids:
            if is_eval:
                self.trainers[aid].prep_rollout()
            else:
                self.trainers[aid].prep_training()

    def build_obs_scaled(self, obs_raw):
        """Build scaled observations from raw observations."""
        obs_scaled = {}
        for aid in self.agent_ids:
            if aid in obs_raw:
                # Apply per-element scaling
                obs_scaled[aid] = obs_raw[aid] * self.obs_scale_factors[aid].unsqueeze(0)
            else:
                raise ValueError(f"Agent {aid} not found in obs_raw")
        return obs_scaled

    def build_obs_tensors(self, obs_dict, agent_id: str):
        """Convert agent-specific obs dict to tensors."""
        obs = torch.as_tensor(obs_dict[agent_id], device=self.device, dtype=torch.float32)
        
        # Build centralized observation (concatenate human and robot obs)
        if not hasattr(self, '_cached_share_obs') or self._cached_share_obs is None:
            human_obs = torch.as_tensor(obs_dict["human"], device=self.device, dtype=torch.float32)
            robot_obs = torch.as_tensor(obs_dict["robot"], device=self.device, dtype=torch.float32)
            
            self._cached_share_obs = torch.cat([human_obs, robot_obs], dim=1)  # [E, 2*obs_dim]
        
        return obs, self._cached_share_obs
    
    def _clear_obs_cache(self):
        """Clear shared observation cache for next step."""
        self._cached_share_obs = None

    def actions_to_env_format(self, actions_dict):
        """Convert normalized actions [-1, 1] to environment format."""
        env_actions = {}
        force_limits = {"human": self.max_human_force, "robot": self.max_robot_force}
        
        for aid, actions_norm in actions_dict.items():
            # Linear scaling from [-1, 1] to physical force range
            env_actions[aid] = actions_norm * force_limits[aid]
            
        return env_actions

    def select_actions(self, observations: Dict[str, torch.Tensor], add_noise: bool, deterministic: bool = None, noise_scale: float = 1.0):
        """Generate actions from both networks independently."""
        # Scale observations before processing
        obs_scaled = self.build_obs_scaled(observations)
        
        # Clear cache at start of action selection
        self._clear_obs_cache()
        
        actions_norm = {}
        action_log_probs = {}
        values = {}
        
        # Use explicit deterministic parameter if provided, otherwise use eval mode or noise settings
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
            
            actions_norm[aid] = a  # Already in [-1, 1] from Tanh-Gaussian
            action_log_probs[aid] = lp
            values[aid] = v
            
            # Update RNN states (if not in eval mode)
            if not self._is_eval_mode:
                self.rnn_states[aid]["actor"] = rnn_a_new
                self.rnn_states[aid]["critic"] = rnn_c_new
        
        # Convert to environment format
        env_actions = self.actions_to_env_format(actions_norm)
        
        # Store rollout data (if not in eval mode)
        if not self._is_eval_mode:
            self._store_rollout_data(obs_scaled, actions_norm, action_log_probs, values)
        
        # Create detail info for StepTracer
        detail = {
            "mean_actions": {k: v.detach().clone() for k, v in env_actions.items()},
            "noise_actions": {
                aid: torch.zeros_like(env_actions[aid]) for aid in self.agent_ids
            }
        }
        
        return env_actions, detail

    def _store_rollout_data(self, observations_scaled, actions_norm, action_log_probs, values):
        """Store data for current rollout step (per-agent) with obs scaling verification."""
        self._current_step_data = {}
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(observations_scaled, aid)
            masks = torch.ones(obs.shape[0], 1, device=self.device)
            
            # Data chain verification: ensure buffer obs matches scaled obs
            scaled = observations_scaled[aid]
            if not torch.allclose(obs, scaled, atol=0, rtol=0):
                raise RuntimeError(f"[SCALING MISMATCH] buffer obs != scaled obs for {aid}")
            
            # Use detach() to avoid gradient accumulation
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
        # Skip buffer writes during evaluation
        if self._is_eval_mode:
            return
            
        for aid in self.agent_ids:
            # Prepare rewards
            reward_tensor = rewards[aid].unsqueeze(-1) if len(rewards[aid].shape) == 1 else rewards[aid]
            
            # Generate proper term_masks based on termination reason
            mask_t = (1.0 - dones[aid].float()).view(-1, 1)
            
            if terminated is not None and truncated is not None:
                # Use provided terminated/truncated info
                is_terminal = terminated[aid]
                is_truncated = truncated[aid]
                
                # term_mask = 1 for truncated (allow bootstrap), 0 for terminated (no bootstrap)
                term_mask_t = (~is_terminal).float().view(-1, 1)
            else:
                # Fallback: analyze info to infer termination type
                if infos is not None and aid in infos:
                    info = infos[aid]
                    if isinstance(info, dict):
                        success = info.get("success", False)
                        hit_obstacle = info.get("hit_obstacle", False)
                        hit_ground = info.get("hit_ground", False)
                        time_limit = info.get("time_limit", False)
                        
                        # True terminal states: success, collisions
                        is_terminal = success or hit_obstacle or hit_ground
                        term_mask_t = torch.zeros_like(mask_t) if is_terminal else torch.ones_like(mask_t)
                    else:
                        # Conservative fallback: assume all dones are time-limit
                        term_mask_t = torch.ones_like(mask_t)
                else:
                    # Conservative fallback: assume all dones are time-limit
                    term_mask_t = torch.ones_like(mask_t)
            
            # Insert into agent's buffer
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
            
            # Reset RNN states for done environments
            done_indices = dones[aid].nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                self.rnn_states[aid]["actor"][done_indices].zero_()
                self.rnn_states[aid]["critic"][done_indices].zero_()
        
        self.rollout_step += 1

    def update(self):
        """Perform dual rMAPPO update when rollout is complete."""
        # Skip updates during evaluation
        if self._is_eval_mode:
            return {}
            
        if self.rollout_step < self.T:
            return {}  # Not ready to update yet
        
        stats = {}
        
        # Bootstrap and train each agent independently
        for aid in self.agent_ids:
            # Bootstrap with final values
            next_obs_dict = getattr(self, '_next_obs', None)
            if next_obs_dict is not None:
                # Scale next_obs before computing final values
                next_obs_scaled = self.build_obs_scaled(next_obs_dict)
                _, share_obs = self.build_obs_tensors(next_obs_scaled, aid)
                masks = torch.ones(share_obs.shape[0], 1, device=self.device)
                
                with torch.no_grad():
                    last_values = self.policies[aid].get_values(
                        share_obs, self.rnn_states[aid]["critic"], masks
                    )
            else:
                last_values = torch.zeros(self.num_envs, 1, device=self.device)
            
            # Compute returns and advantages for this agent
            gamma = self.params.get('mappo_args', {}).get('gamma', 0.99)
            gae_lambda = self.params.get('mappo_args', {}).get('gae_lambda', 0.95)
            
            self.buffers[aid].compute_returns_and_adv(last_values, gamma, gae_lambda)
            
            # Train this agent's networks
            train_info = self.trainers[aid].train(self.buffers[aid])
            
            # Reset buffer
            self.buffers[aid].after_update()
            
            # Store per-agent statistics  
            for k, v in train_info.items():
                stats[f"{k}/{aid}"] = v
        
        # Increment training round counter
        self.train_updates += 1
        
        # Reset rollout step
        self.rollout_step = 0
        
        return stats

    def store_next_obs(self, next_obs):
        """Store next observations for bootstrapping."""
        self._next_obs = next_obs


class TrainingOrchestrator:
    """Main rMAPPO trainer with dual network infrastructure and gradient monitoring."""
    
    def __init__(self, args):
        self.args = args
        print(f"[ORCHESTRATOR] Initializing Dual rMAPPO Training Orchestrator with gradient clipping...")
        
        self._setup_configuration()
        self._setup_wandb_logging()
        self._setup_environment()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[ORCHESTRATOR] Dual rMAPPO Training Orchestrator initialized successfully")

    def _setup_configuration(self):
        """Load and setup training configuration."""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        
        setup_global_reproducibility(self.args.seed, strict_determinism=False)
        self.config.params['seed'] = self.args.seed
        
        self._apply_per_env_scaling()

    def _setup_wandb_logging(self):
        """Setup WandB logging with comprehensive metrics and unified initialization."""
        if self.args.wandb and WANDB_AVAILABLE:
            try:
                # Create wandb directory with proper permissions
                wandb_dir = os.path.expanduser("~/wandb_logs")
                os.makedirs(wandb_dir, exist_ok=True)
                
                run_name = f"rmappo_{time.strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project="surgical_robot_rmappo",
                    name=run_name,
                    tags=["rmappo","rnn","surgical","gradient-clipping","stable"],
                    config=vars(self.args),
                    dir=wandb_dir,
                    reinit=True,
                    settings=wandb.Settings(
                        start_method="thread",
                        _disable_service=True,
                    )
                )
                
                # Define global step as base metric
                wandb.define_metric("global_step")
                
                # Define comprehensive monitoring metrics
                for k in [
                    # PPO metrics
                    "ppo/ratio_mean","ppo/ratio_max","ppo/kl_mean","ppo/clip_fraction",
                    "ppo/adv_mean_norm","ppo/adv_std_norm",
                    # Value metrics (with means)
                    "value/ret_abs_mean","value/v_abs_mean","value/ret_absmax","value/v_absmax",
                    # Policy metrics
                    "policy/saturation","policy/logstd_mean","policy/entropy",
                    # RNN metrics
                    "rnn/actor_h_norm","rnn/critic_h_norm",
                    # Gradient metrics
                    "grad/actor","grad/critic",
                ]:
                    wandb.define_metric(k, step_metric="global_step")
                
                # Define lifecycle markers
                wandb.define_metric("lifecycle/eval_to_train", step_metric="global_step")
                
                self.log_fn = make_wandb_logger()
                print(f"[SETUP] WandB initialized with comprehensive gradient monitoring")
                
            except Exception as e:
                print(f"[ERROR] WandB initialization failed: {e}")
                print("[INFO] Python environment may be corrupted. Falling back to dummy logger.")
                self.log_fn = lambda d, step=None: None
                self.args.wandb = False
        else:
            self.log_fn = lambda d, step=None: None
            if self.args.wandb and not WANDB_AVAILABLE:
                print(f"[SETUP] WandB requested but not available, using dummy logger")
            else:
                print(f"[SETUP] WandB disabled, using dummy logger")

    def _apply_per_env_scaling(self):
        """Scale parameters by number of environments."""
        num_envs = int(self.args.num_envs)
        
        # Scale milestone episodes
        monitor_cfg = self.config.params.get("training_monitor", {})
        base_milestones = list(monitor_cfg.get("milestone_episodes", []))
        scaled_milestones = [int(m * num_envs) for m in base_milestones]
        monitor_cfg["milestone_episodes"] = scaled_milestones
        self.config.params["training_monitor"] = monitor_cfg
        
        print(f"[SETUP][PER-ENV SCALING] num_envs = {num_envs}")
        print(f"  milestone_episodes: {base_milestones} → {scaled_milestones}")

    def _setup_environment(self):
        """Create and configure the environment."""
        print(f"[SETUP] Creating environment: {self.args.task}")
        self.env, self.env_cfg = setup_environment(self.args, self.config)
        
        # Inject configuration to environment
        actual_env = getattr(self.env, 'unwrapped', self.env)
        actual_env.params = self.config.params
        print(f"[SETUP] Injected configuration parameters to environment")
        
        inject_step_tracer(self.env, self.config, self.args.num_envs)

    def _setup_training_components(self):
        """Initialize training components with logging support."""
        print(f"[SETUP] Setting up dual rMAPPO training components with gradient clipping...")
        
        self.metrics_hub = MetricsHub()
        
        # Only setup WandB logger if wandb was successfully initialized
        if self.args.wandb and hasattr(self, 'log_fn') and self.log_fn != (lambda d, step=None: None):
            try:
                # Create and setup WandB logger for metrics hub
                self.wandb_logger = WandBLogger(enabled=True)
                run_config = {**vars(self.args), **self.config.params}
                run_name = f"rmappo_dual_{self.args.num_envs}envs_{time.strftime('%Y%m%d_%H%M%S')}"
                self.wandb_logger.initialize_run(run_config, run_name)
                self.wandb_logger.attach_metrics_hub(self.metrics_hub)
                print(f"[SETUP] WandB attached to MetricsHub")
            except Exception as e:
                print(f"[WARNING] WandB logger setup failed: {e}")
                print(f"[INFO] Continuing with dummy logger")
                self.args.wandb = False
        
        self.rmappo = initialize_rmappo_algorithm(self.env, self.config, self.args, self.log_fn)
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        # Unified max_global_steps logic using mappo_args
        mappo_max_steps = int(self.config.params.get('mappo_args', {}).get('max_global_steps', 200000))
        
        if self.args.max_global_steps > 0:
            self.max_global_steps = self.args.max_global_steps
            print(f"[SETUP] Using CLI max_global_steps: {self.max_global_steps}")
        else:
            self.max_global_steps = mappo_max_steps
            print(f"[SETUP] Using MAPPO max_global_steps: {self.max_global_steps}")
        
        if self.max_global_steps <= 0:
            self.max_global_steps = 200000
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and evaluator with logging support."""
        print(f"[SETUP] Creating dual rMAPPO TrainingRunner and MilestoneEvaluator...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/rmappo_dual/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[SETUP] Log directory created: {self.log_dir}")
        
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
        """Set evaluation mode for both runner and wrapper."""
        self.runner.set_eval_mode(is_eval)
        self.rmappo.set_eval_mode(is_eval)

    def evaluate_milestone_if_due(self):
        """Check and trigger milestone evaluation with lifecycle marker."""
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
            
            # Coordinated eval mode setting
            self.set_eval_mode(True)
            
            result = self.evaluator.run_evaluation(candidate, self.runner.global_step)
            if result.get("skip_episode_once", False):
                self.runner.mark_skip_episode_once()
            
            # Always restore training mode with lifecycle marker
            self.set_eval_mode(False)
            if self.log_fn:
                self.log_fn({"lifecycle/eval_to_train": 1}, step=self.runner.global_step)
            
            # Refresh observations
            obs_dict, _ = self.env.reset()
            self.runner._current_obs = obs_dict
            
            self.max_milestone_triggered = candidate

    def train(self):
        """Main training loop with gradient clipping and comprehensive monitoring."""
        print(f"[TRAIN] Starting dual rMAPPO training with gradient clipping:")
        print(f"  Max collection steps: {self.max_global_steps}")
        print(f"  Rollout horizon: {self.rmappo.T}")
        print(f"  Milestone episodes: {self.milestone_episodes}")
        print(f"  Gradient clipping: enabled with separate thresholds")
        
        # Initialize metrics
        if self.args.wandb and WANDB_AVAILABLE and wandb.run is not None:
            initial_stats = {
                "train/global_episodes": 0,
                "train/collection_steps": 0,
                "train/training_rounds": 0,
            }
            self.metrics_hub.push_update(0, initial_stats)
        
        # Reset environment
        obs_dict, _ = self.env.reset()
        self.runner._current_obs = obs_dict
        
        # Main training loop
        while self.runner.global_step < self.max_global_steps:
            self.runner.execute_training_step()
            self.evaluate_milestone_if_due()
            
            # Enhanced progress reporting with separated step types
            if self.runner.global_step > 0 and self.runner.global_step % 2000 == 0:
                print(f"[Step {self.runner.global_step}] Episodes: {self.runner.global_episodes} | "
                      f"Training rounds: {self.runner.train_updates}")
            
            if self.runner.global_step >= self.max_global_steps:
                break
        
        print(f"\n[TRAINING COMPLETE]")
        print(f"  Total collection steps: {self.runner.global_step}")
        print(f"  Total training rounds: {self.runner.train_updates}")
        print(f"  Total episodes: {self.runner.global_episodes}")
        
        # Save final model with enhanced step information
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
        if self.args.wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        print("[TRAIN] Cleanup completed")


def main():
    """Main entry point for dual rMAPPO training with gradient clipping."""
    print("="*80)
    print("Dual rMAPPO Training with Gradient Clipping and Comprehensive Monitoring")
    print("="*80)
    
    # Parse arguments
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max collection steps: {args_cli.max_global_steps if args_cli.max_global_steps > 0 else 'from YAML'}")
    print(f"  WandB: {args_cli.wandb}")
    print(f"  Gradient clipping: enabled with separate thresholds")
    
    # Launch Isaac Sim
    print(f"[MAIN] Launching Isaac Sim...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    print(f"[MAIN] Creating Dual rMAPPO Training Orchestrator...")
    trainer = TrainingOrchestrator(args_cli)
    
    print(f"[MAIN] Starting training...")
    trainer.train()
    
    print(f"[MAIN] Training completed successfully")
    
    print(f"[MAIN] Closing Isaac Sim...")
    simulation_app.close()


if __name__ == "__main__":
    main()