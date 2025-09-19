#!/usr/bin/env python3

"""
Surgical Robot MADDPG Shared Network Training
Enhanced version with configurable network architecture and exponential noise decay.
MODIFIED: Added per-env scaling functionality for automatic parameter scaling.
MODIFIED: Added eval mode control and exit reason tracking.
FIXED: Proper global reproducibility setup for persistent generator strategy.

Features:
- Configurable network layers, dropout, and orthogonal initialization
- Exponential noise scheduling: fast early decay, slow later convergence
- Unified global step tracking for consistent logging
- TopK model management with milestone evaluation
- WandB integration with per-agent metrics
- Single evaluation chain: MADDPGTrainer -> MilestoneEvaluator
- Per-env scaling: YAML configs are per-env baselines, auto-scaled by --num_envs
- Exit reason tracking for debugging premature termination
- Eval mode support for clean milestone evaluation
- FIXED: Global reproducibility setup with proper seed management
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
from utils.training_helpers import (
    WandBLogger, TrainingConfiguration, create_argument_parser, 
    MetricsHub, TopKModelManager, TrainingRunner, MilestoneEvaluator, save_final_shared_networks
)


def setup_global_reproducibility(seed: int, strict_determinism: bool = False):
    """
    Setup global reproducibility for consistent training results.
    
    Args:
        seed: Base seed for all random number generators
        strict_determinism: Enable strict determinism (may slow training)
    """
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
    
    print(f"[INFO] Environment configuration:")
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Episode length: {env_cfg.episode_length_s}s")
    print(f"  Decimation: {env_cfg.decimation}")
    print(f"  Possible agents: {env_cfg.possible_agents}")
    print(f"  Action spaces: {env_cfg.action_spaces}")
    print(f"  Observation spaces: {env_cfg.observation_spaces}")
    
    env = gym.make(args.task, cfg=env_cfg)
    
    if hasattr(env, 'max_episode_length'):
        print(f"[INFO] Environment max_episode_length: {env.max_episode_length}")
    
    return env, env_cfg


def initialize_maddpg_algorithm(env, config, args):
    """Create and initialize MADDPG algorithm with configurable networks."""
    from surgical_project.algorithms.marl.maddpg import MADDPG
    
    device = config.get_compute_device()
    
    # Get environment count
    num_envs = args.num_envs
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_envs'):
        num_envs = env.unwrapped.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs

    maddpg = MADDPG(
        num_envs=num_envs,
        env=env,
        params=config.params,
        device=device
    )
    
    print(f"[INFO] MADDPG algorithm initialized:")
    print(f"  Device: {device}")
    print(f"  Agent IDs: {maddpg.agent_ids}")
    print(f"  Environments: {num_envs}")
    
    return maddpg


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


class MADDPGTrainer:
    """
    Streamlined MADDPG trainer for configurable networks and noise scheduling.
    MODIFIED: Added per-env scaling functionality.
    MODIFIED: Added eval mode control and exit reason tracking.
    FIXED: Proper global reproducibility setup.
    
    Features:
    - Configuration-driven network architecture
    - Exponential noise decay scheduling
    - Milestone evaluation with TopK model selection
    - Comprehensive WandB logging with per-agent metrics
    - Single evaluation chain (no old RewardLogger/MilestoneManager)
    - Per-env scaling: YAML configs auto-scaled by num_envs
    - Exit reason tracking for debugging premature termination
    - Eval mode support for clean milestone evaluation
    - Global reproducibility with persistent generators
    """
    
    def __init__(self, args):
        self.args = args
        
        print(f"[TRAINER] Initializing MADDPGTrainer with persistent generators and per-env scaling...")
        
        # Setup phases
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[TRAINER] MADDPGTrainer initialized successfully")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.milestone_episodes}")
        
        self._print_configuration_summary()

    def _setup_configuration(self):
        """Load and setup training configuration with per-env scaling."""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        
        # FIXED: Setup global reproducibility with proper seed management
        setup_global_reproducibility(self.args.seed, strict_determinism=False)
        self.config.params['seed'] = self.args.seed
        
        print(f"[SETUP] Configuration loaded, global reproducibility set: {self.args.seed}")
        
        # === NEW: Apply per-env scaling ===
        self._apply_per_env_scaling()

    def _apply_per_env_scaling(self):
        """
        Interpret YAML numbers as 'per-env' baselines and scale them by CLI --num_envs.
        Scales:
          - maddpg_config.min_buffer_size
          - maddpg_config.max_replay_buffer_len
          - training_monitor.milestone_episodes (each entry)
        
        Note: max_global_steps is NOT scaled as it represents total training steps, not per-env steps.
        """
        num_envs = int(self.args.num_envs)

        maddpg_cfg = self.config.params.get("maddpg_config", {})
        monitor_cfg = self.config.params.get("training_monitor", {})

        base_min_buffer = int(maddpg_cfg.get("min_buffer_size", 3600))
        base_max_buffer = int(maddpg_cfg.get("max_replay_buffer_len", 24000))
        base_milestones = list(monitor_cfg.get("milestone_episodes", []))

        scaled_min_buffer = base_min_buffer * num_envs
        scaled_max_buffer = base_max_buffer * num_envs
        scaled_milestones = [int(m * num_envs) for m in base_milestones]

        maddpg_cfg["min_buffer_size"] = int(scaled_min_buffer)
        maddpg_cfg["max_replay_buffer_len"] = int(scaled_max_buffer)
        self.config.params["maddpg_config"] = maddpg_cfg

        monitor_cfg["milestone_episodes"] = scaled_milestones
        self.config.params["training_monitor"] = monitor_cfg

        print("[SETUP][PER-ENV SCALING]")
        print(f"  num_envs = {num_envs}")
        print(f"  min_buffer_size: {base_min_buffer} â†’ {scaled_min_buffer}")
        print(f"  max_replay_buffer_len: {base_max_buffer} â†’ {scaled_max_buffer}")
        print(f"  milestone_episodes: {base_milestones} â†’ {scaled_milestones}")

    def _setup_environment(self):
        """Create and configure the environment."""
        print(f"[SETUP] Creating environment: {self.args.task}")
        self.env, self.env_cfg = setup_environment(self.args, self.config)
        
        # Inject configuration to environment
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, "params"):
            actual_env.params = self.config.params
            print(f"[SETUP] Injected configuration parameters to environment")
        else:
            print("[WARNING] Environment does not support config injection")
        
        # Inject StepTracer for console logging
        inject_step_tracer(self.env, self.config, self.args.num_envs)

    def _setup_logging_and_wandb(self):
        """Setup logging directory and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/maddpg_final/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[SETUP] Log directory created: {self.log_dir}")

        # Initialize WandB
        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_final_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")
        else:
            print(f"[SETUP] WandB disabled")

    def _setup_training_components(self):
        """Initialize training components and metrics hub with unified max_global_steps logic."""
        print(f"[SETUP] Setting up training components...")
        
        # Create MetricsHub
        self.metrics_hub = MetricsHub()
        
        # Connect WandB to MetricsHub
        if self.wandb_logger.enabled:
            self.wandb_logger.attach_metrics_hub(self.metrics_hub)
            print(f"[SETUP] WandB attached to MetricsHub")
        
        # Create MADDPG algorithm
        self.maddpg = initialize_maddpg_algorithm(self.env, self.config, self.args)
        
        # Create TopK manager
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        # UNIFIED max_global_steps logic: CLI takes priority over YAML
        maddpg_cfg = self.config.params.get('maddpg_config', {})
        yaml_max_steps = int(maddpg_cfg.get('max_global_steps', 200000))
        
        # Check if CLI provided a value (0 means not specified)
        if self.args.max_global_steps > 0:
            self.max_global_steps = self.args.max_global_steps
            print(f"[SETUP] Using CLI max_global_steps: {self.max_global_steps}")
        else:
            self.max_global_steps = yaml_max_steps
            print(f"[SETUP] Using YAML max_global_steps: {self.max_global_steps}")
        
        # Ensure we have a valid value
        if self.max_global_steps <= 0:
            self.max_global_steps = 200000  # Default fallback
            print(f"[SETUP] Using default max_global_steps: {self.max_global_steps}")
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
        
        print(f"[SETUP] Training components configured:")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Top-K models: {self.args.top_k_models}")
        print(f"  Milestone episodes: {self.milestone_episodes}")

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and milestone evaluator."""
        print(f"[SETUP] Creating TrainingRunner and MilestoneEvaluator...")
        
        # Pass the resolved max_global_steps to TrainingRunner
        self.runner = TrainingRunner(
            env=self.env,
            maddpg=self.maddpg,
            replay=self.maddpg.replay,
            metrics_hub=self.metrics_hub,
            reward_logger=None,  # No reward_logger dependency
            agent_ids=self.maddpg.agent_ids,
            max_global_steps=self.max_global_steps  # Pass the resolved value
        )
        
        # Create MilestoneEvaluator
        self.evaluator = MilestoneEvaluator(
            env=self.env,
            maddpg=self.maddpg,
            topk_mgr=self.top_k_manager,
            metrics_hub=self.metrics_hub,
            log_dir=self.log_dir,
            agent_ids=self.maddpg.agent_ids
        )
        
        print(f"[SETUP] TrainingRunner and MilestoneEvaluator created successfully")

    def _setup_milestone_management(self):
        """Initialize milestone tracking variables."""
        print(f"[SETUP] Setting up milestone management...")
        self.max_milestone_triggered = 0  # Highest milestone reached
        print(f"[SETUP] Milestone management configured for {len(self.milestone_episodes)} milestones")

    def _print_configuration_summary(self):
        """Print network and exploration configuration summary."""
        networks_cfg = self.config.params.get('networks', {})
        if networks_cfg:
            print(f"[CONFIG] Network Architecture:")
            actor_cfg = networks_cfg.get('actor', {})
            critic_cfg = networks_cfg.get('critic', {})
            print(f"  Actor layers: {actor_cfg.get('hidden_layers', 'default')}")
            print(f"  Critic layers: {critic_cfg.get('hidden_layers', 'default')}")
            print(f"  Orthogonal init: {actor_cfg.get('orthogonal_init', False)}")
            print(f"  Gains: Hidden {actor_cfg.get('ortho_gain_hidden', 1.0)}, Output {actor_cfg.get('ortho_gain_output', 0.01)}")
        
        exploration_cfg = self.config.params.get('exploration', {})
        if exploration_cfg:
            print(f"[CONFIG] Exploration Schedule:")
            print(f"  Noise range: {exploration_cfg.get('sigma_start', 0.7)} â†’ {exploration_cfg.get('sigma_end', 0.1)}")
            print(f"  Decay rate: {exploration_cfg.get('decay_k', 6.0)}")

    def set_eval_mode(self, is_eval: bool):
        """
        Set evaluation mode for MADDPG.
        During evaluation: disable buffer updates and parameter updates.
        
        Args:
            is_eval: True for evaluation mode, False for training mode
        """
        self.maddpg.set_eval_mode(is_eval)
        mode_str = "EVALUATION" if is_eval else "TRAINING"
        print(f"[TRAINER] Mode set to: {mode_str}")

    def evaluate_milestone_if_due(self):
        """Check and trigger milestone evaluation if threshold crossed."""
        if not self.milestone_episodes:
            return
            
        # Find the highest crossed milestone
        candidate = 0
        for milestone in sorted(self.milestone_episodes):
            if self.runner.global_episodes >= milestone:
                candidate = milestone
            else:
                break
                
        # If crossed new threshold, trigger evaluation
        if candidate > self.max_milestone_triggered:
            print(f"[MILESTONE] Crossed threshold: episodes {self.runner.global_episodes} >= milestone {candidate}")
            print(f"[MILESTONE] Triggering evaluation (previous max: {self.max_milestone_triggered})")
            
            # Set eval mode before evaluation
            self.set_eval_mode(True)
            
            try:
                # Direct call to evaluator
                result = self.evaluator.run_evaluation(candidate, self.runner.global_step)
                if result.get("skip_episode_once", False):
                    self.runner.mark_skip_episode_once()
            finally:
                # Always restore training mode
                self.set_eval_mode(False)
            
            # Refresh current observations after evaluation
            env = getattr(self.env, "unwrapped", self.env)
            if hasattr(env, "_get_observations"):
                self.runner._current_obs = env._get_observations()
            else:
                obs_dict, _ = self.env.reset()
                self.runner._current_obs = obs_dict
            print(f"[MILESTONE] Refreshed current observations after evaluation")
            
            self.max_milestone_triggered = candidate
            print(f"[MILESTONE] Updated max_milestone_triggered to {self.max_milestone_triggered}")

    def train(self) -> None:
        """
        Main training loop with configurable networks and noise scheduling.
        MODIFIED: Added exit reason tracking for debugging premature termination.
        FIXED: Uses persistent generators for reproducible yet diverse noise.
        """
        print(f"[TRAIN] Starting training with persistent generators and noise scheduling:")
        print(f"  - TrainingRunner: rolloutâ†’replayâ†’updateâ†’logâ†’count + exponential noise decay")
        print(f"  - MilestoneEvaluator: milestoneâ†’evalâ†’topkâ†’log")
        print(f"  - Network layers configurable via YAML")
        print(f"  - Noise schedule: Ïƒ_start={self.runner.sigma_start} â†’ Ïƒ_end={self.runner.sigma_end}")
        print(f"  - Max steps: {self.max_global_steps}")
        print(f"  - Per-env scaling applied: buffer sizes and milestones auto-scaled by {self.args.num_envs}x")
        print(f"  - FIXED: Persistent generators for reproducible yet diverse per-(agent,env) noise")
        
        reason = "UNKNOWN"  # Track exit reason for debugging
        
        try:
            # Initialize WandB data points
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                    "replay/buffer_size": 0,
                    "exploration/noise_scale": self.runner.sigma_start,
                }
                self.metrics_hub.push_update(0, initial_stats)
            
            # Reset environment
            print(f"[TRAIN] Resetting environment...")
            obs_dict, _ = self.env.reset()
            print(f"[TRAIN] Environment reset complete")
            self.runner._current_obs = obs_dict  
            
            # Main training loop
            while self.runner.global_step < self.max_global_steps:
                # Execute one training step
                obs_dict = self.runner.execute_training_step()
                
                # Milestone checking
                self.evaluate_milestone_if_due()
                
                # Progress reporting
                if self.runner.global_step % 2000 == 0:
                    print(f"[Step {self.runner.global_step}] Episodes so far: {self.runner.global_episodes}")
                    if self.top_k_manager.top_models:
                        scores = [m[0] for m in self.top_k_manager.top_models]
                        print(f"  Top-{len(scores)} Score Range: {min(scores):.2f} ~ {max(scores):.2f}")
                
                # Periodic sanity check
                if (self.runner.global_step % 1000) == 0:
                    assert self.runner.global_step <= self.max_global_steps, \
                        f"Step overflow: {self.runner.global_step} > {self.max_global_steps}"
                
                # Check termination condition
                if self.runner.global_step >= self.max_global_steps:
                    print(f"\n[TRAINING LIMIT] Reached max_global_steps={self.max_global_steps}")
                    break
            
            reason = "REACHED_MAX_STEPS"  # Normal completion
            
            # Training complete
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.runner.global_step}")
            print(f"  Total episodes: {self.runner.global_episodes}")
            print(f"  Max milestone triggered: {self.max_milestone_triggered}")
            print(f"  Final noise scale: {self.runner._calculate_noise_scale():.4f}")
            
            print("\n" + "=" * 70, "\nTraining Complete!\n" + "=" * 70)
            print("\nFinal Top-K Models:")
            for i, (performance, _, milestone) in enumerate(self.top_k_manager.get_top_models()):
                print(f"  #{i+1} Milestone {milestone}: {performance:.2f}")
            print(f"\nResults saved in: {self.log_dir}")
            
            # Save final model
            save_final_shared_networks(
                log_directory=self.log_dir,
                maddpg=self.maddpg,
                global_step=self.runner.global_step,
                global_episodes=self.runner.global_episodes,
                max_milestone_triggered=self.max_milestone_triggered
            )
            
            print(f"[TRAIN] Final results saved successfully")
        
        except KeyboardInterrupt:
            reason = "KEYBOARD_INTERRUPT"
            print(f"\nTraining interrupted by user")
            raise  # Re-raise for proper handling
        except Exception as e:
            reason = f"EXCEPTION:{type(e).__name__}:{str(e)[:50]}"
            print(f"\nTraining failed with exception: {e}")
            raise  # Re-raise for proper handling
        finally:
            # ALWAYS print exit information for debugging
            print(f"\n[TRAINING EXIT]")
            print(f"  Reason: {reason}")
            print(f"  Global step: {self.runner.global_step} / {self.max_global_steps}")
            print(f"  Global episodes: {self.runner.global_episodes}")
            print(f"  Max milestone: {self.max_milestone_triggered}")
            
            # Cleanup resources
            self.env.close()
            self.wandb_logger.finalize_run()
            print("[TRAIN] Cleanup completed")
            print("\nPersistent generator training with per-env scaling completed")


def main():
    """Main entry point for MADDPG training."""
    print("="*80)
    print("MADDPG Persistent Generator Training with Per-Env Scaling")
    print("Enhanced with reproducible yet diverse per-(agent,env) noise generation")
    print("="*80)
    
    # Parse arguments
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max steps: {args_cli.max_global_steps if args_cli.max_global_steps > 0 else 'from YAML'}")
    print(f"  WandB: {args_cli.wandb}")
    print(f"  Config: {args_cli.config}")
    print(f"  Per-env scaling: YAML configs will be auto-scaled by {args_cli.num_envs}x")
    print(f"  FIXED: Persistent generators for reproducible yet diverse noise")
    
    # Launch Isaac Sim
    print(f"[MAIN] Launching Isaac Sim...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    try:
        # Create and start trainer
        print(f"[MAIN] Creating MADDPGTrainer...")
        trainer = MADDPGTrainer(args_cli)
        
        print(f"[MAIN] Starting training...")
        trainer.train()
        
        print(f"[MAIN] Training completed successfully")
        
    finally:
        print(f"[MAIN] Closing Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()