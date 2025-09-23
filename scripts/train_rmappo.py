#!/usr/bin/env python3

"""
rMAPPO training script with dedicated rMAPPO training helpers integration.
Features same CLI interface as MADDPG, complete training infrastructure reuse.
"""

import sys
import os
import torch
import numpy as np
import random
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


def initialize_rmappo_algorithm(env, config, args):
    """Create and initialize rMAPPO algorithm wrapper."""
    from surgical_project.algorithms.ppo import R_MAPPO, RMAPPO, SharedRolloutBuffer
    
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
    num_agents = 2
    N = num_envs * num_agents

    print(f"[RMAPPO] Architecture:")
    print(f"  Environments: {num_envs}")
    print(f"  Agents: {num_agents} (human, robot)")
    print(f"  Obs dim: {obs_dim}, Share obs dim: {share_obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Total slots: {N}")

    # Create policy
    ppo_config = config.params.get('ppo', {})
    policy = R_MAPPO(
        obs_space_desc={'shape': (obs_dim,)},
        cent_obs_space_desc={'shape': (share_obs_dim,)},
        act_space_desc={'shape': (act_dim,)},
        device=device,
        args=ppo_config,
    )

    # Create trainer
    trainer = RMAPPO(
        args=ppo_config,
        policy=policy,
        device=device
    )

    # Create buffer
    T = int(config.params.get('rollout_horizon', 256))
    buffer = SharedRolloutBuffer(
        T=T, N=N, obs_dim=obs_dim, share_obs_dim=share_obs_dim,
        act_dim=act_dim, rnn_hidden_dim=ppo_config.get('hidden_size', 256), device=device
    )

    return RMAPPOWrapper(policy, trainer, buffer, config.params, device, num_envs)


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


class RMAPPOWrapper:
    """Wrapper to make rMAPPO compatible with existing training infrastructure."""
    
    def __init__(self, policy, trainer, buffer, params, device, num_envs):
        self.policy = policy
        self.trainer = trainer
        self.buffer = buffer
        self.params = params
        self.device = device
        self.num_envs = num_envs
        
        # Get agent configuration
        self.agent_ids = ["human", "robot"]
        self.num_agents = len(self.agent_ids)
        
        # RNN states
        H = params.get('ppo', {}).get('hidden_size', 256)
        N = num_envs * self.num_agents
        self.rnn_states_actor = torch.zeros(N, H, device=device)
        self.rnn_states_critic = torch.zeros(N, H, device=device)
        
        # Rollout collection state
        self.rollout_step = 0
        self.T = int(params.get('rollout_horizon', 256))
        
        # Force limits
        constraints = params.get('constraints', {})
        self.max_robot_force = float(constraints.get('max_robot_force', 0.04))
        self.max_human_force = float(constraints.get('max_human_force', 0.04))
        
        # Eval mode flag
        self._is_eval_mode = False
        
        print(f"[RMAPPO WRAPPER] Initialized:")
        print(f"  Rollout horizon: {self.T}")
        print(f"  RNN hidden size: {H}")
        print(f"  Force limits: robot={self.max_robot_force}, human={self.max_human_force}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self._is_eval_mode = is_eval
        if is_eval:
            self.trainer.prep_rollout()
        else:
            self.trainer.prep_training()

    def build_obs_tensors(self, obs_dict):
        """Convert obs dict to tensors for rMAPPO."""
        human = torch.as_tensor(obs_dict["human"], device=self.device, dtype=torch.float32)
        robot = torch.as_tensor(obs_dict["robot"], device=self.device, dtype=torch.float32)
        E = human.shape[0]
        
        # Local observations: [human0, robot0, human1, robot1, ...]
        obs_slots = []
        for e in range(E):
            obs_slots.append(human[e])
            obs_slots.append(robot[e])
        obs_tensor = torch.stack(obs_slots, dim=0)  # [N, obs_dim]
        
        # Shared observations: centralized [human||robot] for each slot
        share_tensor_per_env = torch.cat([human, robot], dim=1)  # [E, 2*obs_dim]
        share_slots = []
        for e in range(E):
            share_slots.append(share_tensor_per_env[e])  # human||robot
            share_slots.append(share_tensor_per_env[e])  # same for robot slot
        share_obs_tensor = torch.stack(share_slots, dim=0)  # [N, 2*obs_dim]
        
        return obs_tensor, share_obs_tensor

    def actions_to_env_format(self, actions_norm):
        """Convert normalized actions to environment format with clamping."""
        # CRITICAL FIX: Clamp actions to [-1, 1] before scaling
        actions_norm = actions_norm.clamp(-1.0, 1.0)
        
        N = actions_norm.shape[0]
        E = N // 2
        
        human_actions, robot_actions = [], []
        for e in range(E):
            human_act = actions_norm[2*e] * self.max_human_force
            robot_act = actions_norm[2*e+1] * self.max_robot_force
            human_actions.append(human_act)
            robot_actions.append(robot_act)
            
        return {
            "human": torch.stack(human_actions, dim=0),
            "robot": torch.stack(robot_actions, dim=0)
        }

    def select_actions(self, observations: Dict[str, torch.Tensor], add_noise: bool, noise_scale: float = 1.0):
        """Generate actions compatible with training loop."""
        obs_tensor, share_obs_tensor = self.build_obs_tensors(observations)
        masks = torch.ones(obs_tensor.shape[0], 1, device=self.device)
        
        # Get actions from policy (deterministic in eval mode)
        deterministic = self._is_eval_mode or not add_noise
        
        with torch.no_grad():
            values, actions_norm, action_log_probs, rnn_a_new, rnn_c_new = self.policy.get_actions(
                share_obs_tensor, obs_tensor, self.rnn_states_actor, self.rnn_states_critic, 
                masks, deterministic=deterministic
            )
        
        # Convert to environment format (with clamping)
        env_actions = self.actions_to_env_format(actions_norm)
        
        # Store for rollout collection (if not in eval mode)
        if not self._is_eval_mode:
            self._store_rollout_data(obs_tensor, share_obs_tensor, actions_norm, 
                                   action_log_probs, values, masks)
            # Update RNN states
            self.rnn_states_actor = rnn_a_new
            self.rnn_states_critic = rnn_c_new
        
        # Create detail info for StepTracer
        detail = {
            "mean_actions": {k: v.clone() for k, v in env_actions.items()},
            "noise_actions": {
                "human": torch.zeros_like(env_actions["human"]),
                "robot": torch.zeros_like(env_actions["robot"])
            }
        }
        
        return env_actions, detail

    def _store_rollout_data(self, obs_tensor, share_obs_tensor, actions_norm, 
                           action_log_probs, values, masks):
        """Store data for current rollout step."""
        # This will be called by add_experience_to_buffer with reward/done info
        self._current_step_data = {
            'obs': obs_tensor,
            'share_obs': share_obs_tensor,
            'actions': actions_norm,
            'action_log_probs': action_log_probs,
            'value_preds': values,
            'masks': masks,
            'rnn_states_actor': self.rnn_states_actor.clone(),
            'rnn_states_critic': self.rnn_states_critic.clone()
        }

    def add_experience_to_buffer(self, obs, actions, rewards, next_obs, dones):
        """Add experience to rollout buffer."""
        if self._is_eval_mode:
            return
            
        # Process rewards and dones
        E = self.num_envs
        reward_slots, done_slots = [], []
        
        for e in range(E):
            # Average human and robot rewards (or use separate - can be configured)
            r_avg = (rewards["human"][e].item() + rewards["robot"][e].item()) / 2
            reward_slots.extend([r_avg, r_avg])  # Same reward for both agents
            
            # OR logic for done
            d = bool(dones["human"][e] or dones["robot"][e])
            done_slots.extend([d, d])
        
        rewards_tensor = torch.tensor(reward_slots, device=self.device, dtype=torch.float32).unsqueeze(-1)
        next_masks = torch.tensor([0.0 if d else 1.0 for d in done_slots], device=self.device, dtype=torch.float32).unsqueeze(-1)
        
        # Insert into buffer (removed active_masks)
        self.buffer.insert(
            t=self.rollout_step,
            obs=self._current_step_data['obs'],
            share_obs=self._current_step_data['share_obs'],
            actions=self._current_step_data['actions'],
            action_log_probs=self._current_step_data['action_log_probs'],
            value_preds=self._current_step_data['value_preds'],
            rewards=rewards_tensor,
            masks=next_masks,
            rnn_states_actor=self._current_step_data['rnn_states_actor'],
            rnn_states_critic=self._current_step_data['rnn_states_critic']
        )
        
        # Handle RNN state resets for done environments
        if any(done_slots):
            done_indices = torch.tensor([i for i, d in enumerate(done_slots) if d], device=self.device, dtype=torch.long)
            self.rnn_states_actor.index_fill_(0, done_indices, 0.0)
            self.rnn_states_critic.index_fill_(0, done_indices, 0.0)
        
        self.rollout_step += 1

    def update(self):
        """Perform rMAPPO update when rollout is complete."""
        if self._is_eval_mode:
            return {}
            
        if self.rollout_step < self.T:
            return {}  # Not ready to update yet
        
        # Bootstrap with final values
        next_obs_dict = getattr(self, '_next_obs', None)
        if next_obs_dict is not None:
            _, share_obs_tensor = self.build_obs_tensors(next_obs_dict)
            masks = torch.ones(share_obs_tensor.shape[0], 1, device=self.device)
            
            with torch.no_grad():
                last_values = self.policy.get_values(share_obs_tensor, self.rnn_states_critic, masks)
        else:
            last_values = torch.zeros(self.num_envs * self.num_agents, 1, device=self.device)
        
        # Compute returns and advantages
        gamma = self.params.get('ppo', {}).get('gamma', 0.99)
        gae_lambda = self.params.get('ppo', {}).get('gae_lambda', 0.95)
        self.buffer.compute_returns_and_adv(last_values, gamma, gae_lambda)
        
        # Train
        train_info = self.trainer.train(self.buffer)
        
        # Reset buffer
        self.buffer.after_update()
        self.rollout_step = 0
        
        # Convert to expected format
        stats = {
            "training/policy_updates": 1,
            "training/value_updates": 1,
            "loss/actor": {"human": train_info.get('policy_loss', 0.0), "robot": train_info.get('policy_loss', 0.0)},
            "loss/critic": {"human": train_info.get('value_loss', 0.0), "robot": train_info.get('value_loss', 0.0)},
            "model/entropy": train_info.get('dist_entropy', 0.0),
            "model/ratio": train_info.get('ratio', 1.0),
            "grad_norm/actor": {"human": train_info.get('actor_grad_norm', 0.0), "robot": train_info.get('actor_grad_norm', 0.0)},
            "grad_norm/critic": {"human": train_info.get('critic_grad_norm', 0.0), "robot": train_info.get('critic_grad_norm', 0.0)},
        }
        
        return stats

    def store_next_obs(self, next_obs):
        """Store next observations for bootstrapping."""
        self._next_obs = next_obs


class RMAPPOTrainer:
    """Main rMAPPO trainer with unified infrastructure."""
    
    def __init__(self, args):
        self.args = args
        print(f"[TRAINER] Initializing rMAPPO Trainer...")
        
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[TRAINER] rMAPPO Trainer initialized successfully")

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
        self.log_dir = f"logs/rmappo_final/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[SETUP] Log directory created: {self.log_dir}")

        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"rmappo_final_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")

    def _setup_training_components(self):
        """Initialize training components."""
        print(f"[SETUP] Setting up rMAPPO training components...")
        
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
        print(f"[SETUP] Creating rMAPPO TrainingRunner and MilestoneEvaluator...")
        
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
        print(f"[TRAIN] Starting rMAPPO training:")
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
    """Main entry point for rMAPPO training."""
    print("="*80)
    print("rMAPPO Training with Dedicated Infrastructure")
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
        print(f"[MAIN] Creating rMAPPO Trainer...")
        trainer = RMAPPOTrainer(args_cli)
        
        print(f"[MAIN] Starting training...")
        trainer.train()
        
        print(f"[MAIN] Training completed successfully")
        
    finally:
        print(f"[MAIN] Closing Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()