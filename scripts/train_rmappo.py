#!/usr/bin/env python3

"""
rMAPPO training script with dual independent networks integration.
Features same CLI interface as MADDPG, complete training infrastructure reuse.
"""

import sys
import os
import torch
import numpy as np
import random
import copy
import yaml
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


def load_dual_network_config(config_path: str):
    """Load YAML config and merge agent-specific parameters."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    common = cfg.get("algo", cfg)  # Fallback to root if no 'algo' section
    per_agent = cfg.get("agents", {})
    
    def merge_config_for_agent(agent_id: str):
        final = copy.deepcopy(common)
        agent_overrides = per_agent.get(agent_id, {})
        final.update(agent_overrides)
        return final
    
    return {
        "human": merge_config_for_agent("human"),
        "robot": merge_config_for_agent("robot"),
        "common": common,
        "raw_config": cfg
    }


def initialize_rmappo_algorithm(env, config, args):
    """Create and initialize dual rMAPPO algorithm wrapper."""
    # ✅ 修正后的导入路径
    from surgical_project.algorithms.marl.rmappo.r_mappo_core import RMAPPOPolicy, RMAPPOTrainer
    from surgical_project.algorithms.marl.rmappo.rollout_buffer import SharedRolloutBuffer
    
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
    act_dim = 3

    print(f"[RMAPPO] Dual Network Architecture:")
    print(f"  Environments: {num_envs}")
    print(f"  Agents: human, robot (independent networks)")
    print(f"  Obs dim: {obs_dim}, Share obs dim: {share_obs_dim}")
    print(f"  Action dim: {act_dim}")

    # Load dual config
    dual_config = load_dual_network_config(args.config)
    
    # Create wrapper with dual configs
    return DualRMAPPOWrapper(
        dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, config.params
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
    """
    Dual independent network rMAPPO wrapper.
    Manages two completely separate networks with synchronized training.
    """
    
    def __init__(self, dual_config, device, num_envs, obs_dim, share_obs_dim, act_dim, params):
        # ✅ 修正后的导入路径 
        from surgical_project.algorithms.marl.rmappo.r_mappo_core import RMAPPOPolicy, RMAPPOTrainer
        from surgical_project.algorithms.marl.rmappo.rollout_buffer import SharedRolloutBuffer
        
        self.device = device
        self.num_envs = num_envs
        self.params = params
        self.agent_ids = ["human", "robot"]
        
        # Rollout parameters
        self.T = int(params.get('rollout_horizon', 256))
        self.rollout_step = 0
        
        # Force constraints
        constraints = params.get('constraints', {})
        self.max_robot_force = float(constraints.get('max_robot_force', 0.04))
        self.max_human_force = float(constraints.get('max_human_force', 0.04))
        
        # Create space descriptors
        obs_space_desc = {'shape': (obs_dim,)}
        cent_obs_space_desc = {'shape': (share_obs_dim,)}
        act_space_desc = {'shape': (act_dim,)}
        
        # Initialize dual networks
        self.policies = {}
        self.trainers = {}
        self.buffers = {}
        self.rnn_states = {}
        
        # 1. Initialize human network first
        human_config = dual_config["human"]
        pol_h = RMAPPOPolicy(
            obs_space_desc=obs_space_desc,
            cent_obs_space_desc=cent_obs_space_desc, 
            act_space_desc=act_space_desc,
            device=device,
            args=human_config
        )
        trn_h = RMAPPOTrainer(args=human_config, policy=pol_h, device=device)
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
        
        # 2. Initialize robot network and copy weights from human
        robot_config = dual_config["robot"]
        pol_r = RMAPPOPolicy(
            obs_space_desc=obs_space_desc,
            cent_obs_space_desc=cent_obs_space_desc,
            act_space_desc=act_space_desc, 
            device=device,
            args=robot_config
        )
        trn_r = RMAPPOTrainer(args=robot_config, policy=pol_r, device=device)
        buf_r = SharedRolloutBuffer(
            T=self.T, N=num_envs, obs_dim=obs_dim, share_obs_dim=share_obs_dim,
            act_dim=act_dim, rnn_hidden_dim=robot_config.get('hidden_size', 256), device=device
        )
        
        # ✅ Copy human weights to robot for identical initialization
        pol_r.actor.load_state_dict(pol_h.actor.state_dict())
        pol_r.critic.load_state_dict(pol_h.critic.state_dict())
        
        self.policies["robot"] = pol_r
        self.trainers["robot"] = trn_r
        self.buffers["robot"] = buf_r
        self.rnn_states["robot"] = {
            "actor": torch.zeros(num_envs, robot_config.get('hidden_size', 256), device=device),
            "critic": torch.zeros(num_envs, robot_config.get('hidden_size', 256), device=device),
        }
        
        # Eval mode flag
        self._is_eval_mode = False
        
        print(f"[DUAL RMAPPO] Initialized:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  Networks: independent human & robot")
        print(f"  Initial weights: robot copied from human")
        print(f"  Force limits: robot={self.max_robot_force}, human={self.max_human_force}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode for both networks."""
        self._is_eval_mode = is_eval
        for aid in self.agent_ids:
            if is_eval:
                self.trainers[aid].prep_rollout()
            else:
                self.trainers[aid].prep_training()

    def build_obs_tensors(self, obs_dict, agent_id: str):
        """Convert agent-specific obs dict to tensors."""
        obs = torch.as_tensor(obs_dict[agent_id], device=self.device, dtype=torch.float32)
        
        # Build centralized observation (human||robot) for each environment
        human_obs = torch.as_tensor(obs_dict["human"], device=self.device, dtype=torch.float32)
        robot_obs = torch.as_tensor(obs_dict["robot"], device=self.device, dtype=torch.float32)
        share_obs = torch.cat([human_obs, robot_obs], dim=1)  # [E, 2*obs_dim]
        
        return obs, share_obs

    def actions_to_env_format(self, actions_dict):
        """Convert normalized actions to environment format with per-agent clamping."""
        env_actions = {}
        force_limits = {"human": self.max_human_force, "robot": self.max_robot_force}
        
        for aid, actions_norm in actions_dict.items():
            # Clamp to [-1, 1] then scale by agent-specific limits
            actions_norm = actions_norm.clamp(-1.0, 1.0)
            env_actions[aid] = actions_norm * force_limits[aid]
            
        return env_actions

    def select_actions(self, observations: Dict[str, torch.Tensor], add_noise: bool, noise_scale: float = 1.0):
        """Generate actions from both networks independently."""
        actions_norm = {}
        action_log_probs = {}
        values = {}
        
        # Determine if deterministic (eval mode or no noise)
        deterministic = self._is_eval_mode or not add_noise
        
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(observations, aid)
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
            
            # Update RNN states (if not in eval mode)
            if not self._is_eval_mode:
                self.rnn_states[aid]["actor"] = rnn_a_new
                self.rnn_states[aid]["critic"] = rnn_c_new
        
        # Convert to environment format
        env_actions = self.actions_to_env_format(actions_norm)
        
        # Store rollout data (if not in eval mode)
        if not self._is_eval_mode:
            self._store_rollout_data(observations, actions_norm, action_log_probs, values)
        
        # Create detail info for StepTracer
        detail = {
            "mean_actions": {k: v.clone() for k, v in env_actions.items()},
            "noise_actions": {
                aid: torch.zeros_like(env_actions[aid]) for aid in self.agent_ids
            }
        }
        
        return env_actions, detail

    def _store_rollout_data(self, observations, actions_norm, action_log_probs, values):
        """Store data for current rollout step (per-agent)."""
        self._current_step_data = {}
        for aid in self.agent_ids:
            obs, share_obs = self.build_obs_tensors(observations, aid)
            masks = torch.ones(obs.shape[0], 1, device=self.device)
            
            self._current_step_data[aid] = {
                'obs': obs,
                'share_obs': share_obs,
                'actions': actions_norm[aid], 
                'action_log_probs': action_log_probs[aid],
                'value_preds': values[aid],
                'masks': masks,
                'rnn_states_actor': self.rnn_states[aid]["actor"].clone(),
                'rnn_states_critic': self.rnn_states[aid]["critic"].clone()
            }

    def add_experience_to_buffer(self, obs, actions, rewards, next_obs, dones):
        """Add experience to per-agent rollout buffers."""
        if self._is_eval_mode:
            return
            
        for aid in self.agent_ids:
            # Prepare rewards (use agent-specific rewards)
            reward_tensor = rewards[aid].unsqueeze(-1) if len(rewards[aid].shape) == 1 else rewards[aid]
            
            # Prepare masks (0.0 if done, 1.0 otherwise)
            done_mask = torch.where(dones[aid], 0.0, 1.0).unsqueeze(-1) if len(dones[aid].shape) == 1 else torch.where(dones[aid], 0.0, 1.0)
            
            # Insert into agent's buffer
            self.buffers[aid].insert(
                t=self.rollout_step,
                obs=self._current_step_data[aid]['obs'],
                share_obs=self._current_step_data[aid]['share_obs'],
                actions=self._current_step_data[aid]['actions'],
                action_log_probs=self._current_step_data[aid]['action_log_probs'],
                value_preds=self._current_step_data[aid]['value_preds'],
                rewards=reward_tensor,
                masks=done_mask,
                rnn_states_actor=self._current_step_data[aid]['rnn_states_actor'],
                rnn_states_critic=self._current_step_data[aid]['rnn_states_critic']
            )
            
            # Reset RNN states for done environments
            done_indices = dones[aid].nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                self.rnn_states[aid]["actor"][done_indices].zero_()
                self.rnn_states[aid]["critic"][done_indices].zero_()
        
        self.rollout_step += 1

    def update(self):
        """Perform dual rMAPPO update when rollout is complete."""
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
                _, share_obs = self.build_obs_tensors(next_obs_dict, aid)
                masks = torch.ones(share_obs.shape[0], 1, device=self.device)
                
                with torch.no_grad():
                    last_values = self.policies[aid].get_values(
                        share_obs, self.rnn_states[aid]["critic"], masks
                    )
            else:
                last_values = torch.zeros(self.num_envs, 1, device=self.device)
            
            # Compute returns and advantages for this agent
            gamma = self.params.get('ppo', {}).get('gamma', 0.99)
            gae_lambda = self.params.get('ppo', {}).get('gae_lambda', 0.95)
            self.buffers[aid].compute_returns_and_adv(last_values, gamma, gae_lambda)
            
            # Train this agent's networks
            train_info = self.trainers[aid].train(self.buffers[aid])
            
            # Reset buffer
            self.buffers[aid].after_update()
            
            # Store per-agent statistics  
            for k, v in train_info.items():
                stats[f"{k}/{aid}"] = v
        
        # Reset rollout step
        self.rollout_step = 0
        
        # Add unified training counters
        stats["training/policy_updates"] = 2  # Both agents updated
        stats["training/value_updates"] = 2
        
        return stats

    def store_next_obs(self, next_obs):
        """Store next observations for bootstrapping."""
        self._next_obs = next_obs


class RMAPPOTrainer:
    """Main rMAPPO trainer with dual network infrastructure."""
    
    def __init__(self, args):
        self.args = args
        print(f"[TRAINER] Initializing Dual rMAPPO Trainer...")
        
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[TRAINER] Dual rMAPPO Trainer initialized successfully")

    def _setup_configuration(self):
        """Load and setup training configuration."""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        
        setup_global_reproducibility(self.args.seed, strict_determinism=False)
        self.config.params['seed'] = self.args.seed
        
        self._apply_per_env_scaling()

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

    def _setup_logging_and_wandb(self):
        """Setup logging directory and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/rmappo_dual/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[SETUP] Log directory created: {self.log_dir}")

        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"rmappo_dual_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")

    def _setup_training_components(self):
        """Initialize training components."""
        print(f"[SETUP] Setting up dual rMAPPO training components...")
        
        self.metrics_hub = MetricsHub()
        
        if self.wandb_logger.enabled:
            self.wandb_logger.attach_metrics_hub(self.metrics_hub)
            print(f"[SETUP] WandB attached to MetricsHub")
        
        self.rmappo = initialize_rmappo_algorithm(self.env, self.config, self.args)
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        # Unified max_global_steps logic
        yaml_max_steps = int(self.config.params.get('ppo', {}).get('max_global_steps', 200000))
        
        if self.args.max_global_steps > 0:
            self.max_global_steps = self.args.max_global_steps
            print(f"[SETUP] Using CLI max_global_steps: {self.max_global_steps}")
        else:
            self.max_global_steps = yaml_max_steps
            print(f"[SETUP] Using YAML max_global_steps: {self.max_global_steps}")
        
        if self.max_global_steps <= 0:
            self.max_global_steps = 200000
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and evaluator."""
        print(f"[SETUP] Creating dual rMAPPO TrainingRunner and MilestoneEvaluator...")
        
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
            
            try:
                result = self.evaluator.run_evaluation(candidate, self.runner.global_step)
                if result.get("skip_episode_once", False):
                    self.runner.mark_skip_episode_once()
            finally:
                self.set_eval_mode(False)
            
            # Refresh observations
            obs_dict, _ = self.env.reset()
            self.runner._current_obs = obs_dict
            
            self.max_milestone_triggered = candidate

    def train(self):
        """Main training loop."""
        print(f"[TRAIN] Starting dual rMAPPO training:")
        print(f"  Max steps: {self.max_global_steps}")
        print(f"  Rollout horizon: {self.rmappo.T}")
        print(f"  Milestone episodes: {self.milestone_episodes}")
        
        try:
            # Initialize metrics
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                }
                self.metrics_hub.push_update(0, initial_stats)
            
            # Reset environment
            obs_dict, _ = self.env.reset()
            self.runner._current_obs = obs_dict
            
            # Main training loop
            while self.runner.global_step < self.max_global_steps:
                self.runner.execute_training_step()
                self.evaluate_milestone_if_due()
                
                if self.runner.global_step % 2000 == 0:
                    print(f"[Step {self.runner.global_step}] Episodes: {self.runner.global_episodes}")
                
                if self.runner.global_step >= self.max_global_steps:
                    break
            
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.runner.global_step}")
            print(f"  Total episodes: {self.runner.global_episodes}")
            
            # Save final model
            save_final_rmappo_networks(
                log_directory=self.log_dir,
                rmappo_wrapper=self.rmappo,
                global_step=self.runner.global_step,
                global_episodes=self.runner.global_episodes,
                max_milestone_triggered=self.max_milestone_triggered
            )
            
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user")
            raise
        finally:
            self.env.close()
            self.wandb_logger.finalize_run()
            print("[TRAIN] Cleanup completed")


def main():
    """Main entry point for dual rMAPPO training."""
    print("="*80)
    print("Dual rMAPPO Training with Independent Networks")
    print("="*80)
    
    # Parse arguments (reuse existing parser)
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max steps: {args_cli.max_global_steps if args_cli.max_global_steps > 0 else 'from YAML'}")
    print(f"  WandB: {args_cli.wandb}")
    
    # Launch Isaac Sim
    print(f"[MAIN] Launching Isaac Sim...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    try:
        print(f"[MAIN] Creating Dual rMAPPO Trainer...")
        trainer = RMAPPOTrainer(args_cli)
        
        print(f"[MAIN] Starting training...")
        trainer.train()
        
        print(f"[MAIN] Training completed successfully")
        
    finally:
        print(f"[MAIN] Closing Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()