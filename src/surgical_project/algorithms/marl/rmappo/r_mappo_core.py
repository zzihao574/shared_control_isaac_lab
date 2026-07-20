"""
rMAPPO core components: Actor-Critic networks, Policy wrapper, and Algorithm trainer.
Naming clarification: RMAPPOPolicy (policy) + RMAPPOAlgorithm (algorithm trainer).
MODIFIED: Removed ValueNorm class and all valuenorm branches, keeping only PopArt.
"""

import numpy as np
import torch
import torch.nn as nn
import math
import os
from .mappo_utils import init, check, get_shape_from_obs_space, ACTLayer, PopArt
from .rnn import RNNLayer


def _flat(x):
    """Flatten tensor for monitoring - keep last dimension for multi-dim actions."""
    return x.view(-1, x.size(-1)) if x.dim() > 2 else x.view(-1)


def _h_mean_norm(h):
    """Calculate mean norm of RNN hidden states."""
    if h.dim() == 3: 
        h = h[-1]  # Take last layer
    return h.view(-1, h.size(-1)).norm(p=2, dim=1).mean()


# =============================================================================
# ACTOR-CRITIC NETWORKS
# =============================================================================

class R_Actor(nn.Module):
    """Actor network for rMAPPO with RNN only."""
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args['hidden_size']
        
        self._gain = args.get('gain', 0.01)
        self._use_orthogonal = args.get('use_orthogonal', True)
        self._recurrent_N = args.get('recurrent_N', 1)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        
        # Basic linear layer as base
        self.base = nn.Sequential(
            nn.Linear(obs_shape[0], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Always use RNN
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # Action layer with Tanh-Gaussian
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, use_tanh=True)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        """Compute actions from the given inputs."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, available_actions=None, deterministic=deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks):
        """Compute log probability and entropy of given actions."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions=None, active_masks=None
        )

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """Critic network for rMAPPO with RNN only."""
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args['hidden_size']
        self._use_orthogonal = args.get('use_orthogonal', True)
        self._recurrent_N = args.get('recurrent_N', 1)
        self._use_popart = args.get('use_popart', False)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        # Basic linear layer as base
        self.base = nn.Sequential(
            nn.Linear(cent_obs_shape[0], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Always use RNN
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # Value output layer
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute value function predictions."""
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


# =============================================================================
# POLICY WRAPPER
# =============================================================================

class RMAPPOPolicy:
    """rMAPPO Policy class - manages networks and optimizers."""

    def __init__(self, obs_space_desc, cent_obs_space_desc, act_space_desc, device, args):
        self.device = device
        self.args = args
        
        # Extract learning rates and optimizer settings
        self.actor_lr = float(args.get('actor_lr', 3e-4))
        self.critic_lr = float(args.get('critic_lr', 3e-4)) 
        self.opt_eps = float(args.get('opt_eps', 1e-5))
        self.weight_decay = float(args.get('weight_decay', 0.0))

        # Strong assertion for action dimension
        if isinstance(act_space_desc, dict):
            act_dim = act_space_desc.get('shape', (None,))[0]
        else:
            act_dim = act_space_desc[0] if isinstance(act_space_desc, (tuple, list)) else None
        
        if act_dim is None:
            raise RuntimeError("[RMAPPO POLICY] Cannot infer action dimension from act_space_desc. "
                             "Please provide valid action space description.")
        
        self.act_dim = act_dim
        print(f"[RMAPPO POLICY] Action dimension: {self.act_dim}")

        # Create mock space objects for compatibility
        self.obs_space = self._create_mock_space(obs_space_desc)
        self.cent_obs_space = self._create_mock_space(cent_obs_space_desc)
        self.act_space = self._create_mock_space(act_space_desc)

        # Initialize networks
        self.actor = R_Actor(args, self.obs_space, self.act_space, device)
        self.critic = R_Critic(args, self.cent_obs_space, device)

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr, 
            eps=self.opt_eps,
            weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opt_eps,
            weight_decay=self.weight_decay
        )

    def _create_mock_space(self, space_desc):
        """Create a mock space object from description."""
        class MockSpace:
            def __init__(self, desc):
                if isinstance(desc, dict) and 'shape' in desc:
                    self.shape = desc['shape']
                else:
                    self.shape = desc
                self.__class__.__name__ = "Box"  # For ACTLayer compatibility
                
        return MockSpace(space_desc)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, deterministic=False):
        """Compute actions and value function predictions."""
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, deterministic
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions."""
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks):
        """Get action logprobs / entropy and value function predictions for update."""
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks
        )

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """Compute actions using the given inputs."""
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, deterministic
        )
        return actions, rnn_states_actor


# =============================================================================
# ALGORITHM TRAINER
# =============================================================================

def huber_loss(e, d):
    """Huber loss function."""
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)


def mse_loss(e):
    """MSE loss function."""
    return e ** 2 / 2


class RMAPPOAlgorithm:
    """rMAPPO Algorithm Trainer - executes PPO updates with gradient clipping and LR decay."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.get('clip_param', 0.2)
        self.ppo_epoch = args.get('ppo_epoch', 10)
        self.num_mini_batch = args.get('num_mini_batch', 4)
        self.data_chunk_length = args.get('data_chunk_length', 16)
        self.entropy_coef = args.get('entropy_coef', 0.01)
        
        # Separate gradient clipping thresholds
        self.max_grad_norm_actor = float(args.get('max_grad_norm_actor', 5.0))
        self.max_grad_norm_critic = float(args.get('max_grad_norm_critic', 10.0))
        
        self.huber_delta = args.get('huber_delta', 1.0)

        self._use_clipped_value_loss = args.get('use_clipped_value_loss', False)
        self._use_popart = args.get('use_popart', False)

        # Metrics are returned to the runner and emitted once per rollout.
        self.metrics_hub = args.get('metrics_hub', None)
        self.agent_id = str(args.get('agent_id', 'agent'))
        self.global_step = 0

        # MODIFIED: Only PopArt is supported, ValueNorm removed
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        else:
            self.value_normalizer = None

        # Learning rate decay setup
        self.global_update_step = 0
        self.actor_lr_init = self.policy.actor_optimizer.param_groups[0]["lr"]
        self.critic_lr_init = self.policy.critic_optimizer.param_groups[0]["lr"]
        
        # The caller injects the already-resolved training configuration here.
        # Do not reopen the default YAML: checkpoint/CLI conditions must remain
        # the single source of truth.
        decay_cfg = args.get("lr_decay", {})
        
        self._lr_decay_enabled = bool(decay_cfg.get('enabled', False))
        self._lr_final_factor = float(decay_cfg.get('final_factor', 0.1))
        
        if self._lr_decay_enabled:
            print(f"[LR DECAY] Enabled: final_factor={self._lr_final_factor}")
            print(f"[LR DECAY] Initial LR: actor={self.actor_lr_init:.6f}, critic={self.critic_lr_init:.6f}")

    def _maybe_decay_lr(self, now_update_idx: int):
        """Apply cosine learning rate decay if enabled."""
        if not self._lr_decay_enabled:
            return
        
        actor_min = self.actor_lr_init * self._lr_final_factor
        critic_min = self.critic_lr_init * self._lr_final_factor
        
        # Cosine decay with ~1000 updates as the decay horizon
        progress = min(1.0, now_update_idx / 1000.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        actor_lr = actor_min + (self.actor_lr_init - actor_min) * cosine
        critic_lr = critic_min + (self.critic_lr_init - critic_min) * cosine
        
        # Update optimizer learning rates
        for pg in self.policy.actor_optimizer.param_groups:
            pg["lr"] = actor_lr
        for pg in self.policy.critic_optimizer.param_groups:
            pg["lr"] = critic_lr
        
    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """Calculate value function loss using Huber loss."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        # MODIFIED: Only PopArt branch remains, ValueNorm removed
        if self._use_popart:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        # Always use Huber loss
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """Single PPO update step with gradient clipping and comprehensive monitoring."""
        # Input data
        share_obs_batch = check(sample["share_obs"]).to(**self.tpdv)
        obs_batch = check(sample["obs"]).to(**self.tpdv)
        rnn_states_batch = check(sample["rnn_states_actor"]).to(**self.tpdv)
        rnn_states_critic_batch = check(sample["rnn_states_critic"]).to(**self.tpdv)
        actions_batch = check(sample["actions"]).to(**self.tpdv)
        value_preds_batch = check(sample["value_preds"]).to(**self.tpdv)
        return_batch = check(sample["returns"]).to(**self.tpdv)
        rnn_masks_batch = check(sample["rnn_masks"]).to(**self.tpdv)
        old_action_log_probs_batch = check(sample["action_log_probs"]).to(**self.tpdv)
        adv_targ = check(sample["advantages"]).to(**self.tpdv)

        # Shape validation
        L, B = obs_batch.shape[:2]
        
        assert actions_batch.shape[:2] == (L, B), \
            f"actions {actions_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert rnn_masks_batch.shape[:2] == (L, B), \
            f"rnn_masks {rnn_masks_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        
        # Flatten for network processing
        share_obs_flat = share_obs_batch.view(L * B, -1)
        obs_flat = obs_batch.view(L * B, -1)
        actions_flat = actions_batch.view(L * B, -1)
        rnn_masks_flat = rnn_masks_batch.view(L * B, -1)

        # Forward pass through networks
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_flat, obs_flat, rnn_states_batch, rnn_states_critic_batch,
            actions_flat, rnn_masks_flat
        )
        
        # Flatten target tensors for loss computation
        value_preds_batch = value_preds_batch.view(L * B, -1)
        return_batch = return_batch.view(L * B, -1)
        old_action_log_probs_batch = old_action_log_probs_batch.view(L * B, -1)
        adv_targ = adv_targ.view(L * B, -1)
        
        # ===== MONITORING SYSTEM WITH MEAN VALUES =====
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = (old_action_log_probs_batch - action_log_probs)
        act_used = _flat(actions_batch)

        vals = values.view(-1)
        rets = return_batch.view(-1)

        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        # Gradient clipping for actor
        gn_actor = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), 
            self.max_grad_norm_actor
        ).item()
        
        self.policy.actor_optimizer.step()

        # Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.critic_optimizer.zero_grad()

        value_loss.backward()

        # Gradient clipping for critic
        gn_critic = torch.nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(),
            self.max_grad_norm_critic
        ).item()

        self.policy.critic_optimizer.step()

        # Calculate PPO monitoring metrics
        with torch.no_grad():
            # Clipping fraction
            clipped = (imp_weights > 1.0 + self.clip_param) | (imp_weights < 1.0 - self.clip_param)
            clipfrac = clipped.float().mean()
            
            # Approximate KL divergence
            approx_kl = (old_action_log_probs_batch - action_log_probs).mean().clamp_min(0)

        return {
            "value_loss": value_loss.item(),
            "critic_grad_norm": float(gn_critic),
            "policy_loss": policy_loss.item(),
            "dist_entropy": dist_entropy.item(),
            "actor_grad_norm": float(gn_actor),
            "imp_weights": imp_weights.mean().item(),
            "ratio_max": ratio.max().item(),
            "clipfrac": clipfrac.item(),
            "approx_kl": approx_kl.item(),
            "adv_mean": adv_targ.mean().item(),
            "adv_std": adv_targ.std(unbiased=False).item(),
            "ret_abs_mean": rets.abs().mean().item(),
            "v_abs_mean": vals.abs().mean().item(),
            "ret_absmax": rets.abs().max().item(),
            "v_absmax": vals.abs().max().item(),
            "saturation": (act_used.abs() > 0.98).float().mean().item(),
            "logstd_raw": self.policy.actor.act.logstd_mean(effective=False),
            "logstd_effective": self.policy.actor.act.logstd_mean(effective=True),
            "rnn_actor_h_norm": _h_mean_norm(rnn_states_batch).item(),
            "rnn_critic_h_norm": _h_mean_norm(rnn_states_critic_batch).item(),
        }

    def train(self, buffer, update_actor=True, generator=None):
        """Perform multi-epoch PPO training with LR decay at the end."""
        metric_names = (
            'value_loss', 'policy_loss', 'dist_entropy',
            'actor_grad_norm', 'critic_grad_norm', 'ratio', 'ratio_max',
            'clipfrac', 'approx_kl', 'adv_mean', 'adv_std',
            'ret_abs_mean', 'v_abs_mean', 'ret_absmax', 'v_absmax',
            'saturation', 'logstd_raw', 'logstd_effective',
            'rnn_actor_h_norm', 'rnn_critic_h_norm',
        )
        train_info = {name: 0.0 for name in metric_names}

        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(
                self.num_mini_batch, self.data_chunk_length, generator=generator
            )

            for sample in data_generator:
                update_info = self.ppo_update(sample, update_actor)

                update_info["ratio"] = update_info.pop("imp_weights")
                for name in metric_names:
                    train_info[name] += update_info[name]

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        # Apply LR decay after completing full PPO update
        self._maybe_decay_lr(self.global_update_step)
        self.global_update_step += 1
        train_info['actor_lr'] = self.policy.actor_optimizer.param_groups[0]['lr']
        train_info['critic_lr'] = self.policy.critic_optimizer.param_groups[0]['lr']

        return train_info

    def prep_training(self):
        """Set networks to training mode."""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.policy.actor.eval()
        self.policy.critic.eval()
