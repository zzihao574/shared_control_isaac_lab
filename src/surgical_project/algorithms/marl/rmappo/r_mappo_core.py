"""
rMAPPO core components: Actor-Critic networks, Policy wrapper, and Algorithm trainer.
Naming clarification: RMAPPOPolicy (policy) + RMAPPOAlgorithm (algorithm trainer).
MODIFIED: Fixed gradient norm type conversion and updated ACTLayer to use Tanh-Gaussian.
"""

import numpy as np
import torch
import torch.nn as nn
from .mappo_utils import init, check, get_shape_from_obs_space, ACTLayer, PopArt
from .rnn import RNNLayer


# =============================================================================
# ACTOR-CRITIC NETWORKS
# =============================================================================

class R_Actor(nn.Module):
    """
    Actor network for rMAPPO with RNN only.
    Simplified: removed CNN/MLP choices, always use RNN.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args['hidden_size']
        
        self._gain = args.get('gain', 0.01)
        self._use_orthogonal = args.get('use_orthogonal', True)
        self._recurrent_N = args.get('recurrent_N', 1)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        
        # Simplified: directly create a basic linear layer as base
        self.base = nn.Sequential(
            nn.Linear(obs_shape[0], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Always use RNN (no conditional)
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # Action layer with Tanh-Gaussian enabled by default
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
    """
    Critic network for rMAPPO with RNN only.
    Simplified: removed CNN/MLP choices, always use RNN.
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args['hidden_size']
        self._use_orthogonal = args.get('use_orthogonal', True)
        self._recurrent_N = args.get('recurrent_N', 1)
        self._use_popart = args.get('use_popart', False)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        # Simplified: directly create a basic linear layer as base
        self.base = nn.Sequential(
            nn.Linear(cent_obs_shape[0], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Always use RNN (no conditional)
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
    """
    rMAPPO Policy class - 策略包装器
    职责：管理网络和优化器，提供推理接口
    """

    def __init__(self, obs_space_desc, cent_obs_space_desc, act_space_desc, device, args):
        self.device = device
        self.args = args
        
        # Extract learning rates and optimizer settings (with explicit type conversion)
        self.actor_lr = float(args.get('actor_lr', 3e-4))
        self.critic_lr = float(args.get('critic_lr', 3e-4)) 
        self.opt_eps = float(args.get('opt_eps', 1e-5))
        self.weight_decay = float(args.get('weight_decay', 0.0))

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
# ALGORITHM TRAINER (RENAMED)
# =============================================================================

def get_grad_norm(parameters):
    """Calculate gradient norm."""
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def huber_loss(e, d):
    """Huber loss function."""
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)


def mse_loss(e):
    """MSE loss function."""
    return e ** 2 / 2


class ValueNorm:
    """Simple value normalization (placeholder implementation)."""
    def __init__(self, input_shape, device=torch.device("cpu")):
        self.device = device
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
    def update(self, values):
        """Update normalization statistics."""
        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.numel()
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, values):
        """Normalize values."""
        return (values - self.mean) / (torch.sqrt(self.var) + 1e-8)
        
    def denormalize(self, values):
        """Denormalize values."""
        return values * torch.sqrt(self.var) + self.mean


class RMAPPOAlgorithm:
    """
    rMAPPO Algorithm Trainer (renamed from RMAPPOTrainer)
    职责：执行PPO算法更新，包括loss计算、梯度更新等
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.get('clip_param', 0.2)
        self.ppo_epoch = args.get('ppo_epoch', 10)
        self.num_mini_batch = args.get('num_mini_batch', 4)
        self.data_chunk_length = args.get('data_chunk_length', 16)
        self.value_loss_coef = args.get('value_loss_coef', 0.5)
        self.entropy_coef = args.get('entropy_coef', 0.01)
        self.max_grad_norm = args.get('max_grad_norm', 0.5)
        self.huber_delta = args.get('huber_delta', 1.0)

        # Keep these options but simplify others
        self._use_max_grad_norm = True  # Always use grad norm clipping
        self._use_clipped_value_loss = args.get('use_clipped_value_loss', False)
        self._use_popart = args.get('use_popart', False)
        self._use_valuenorm = args.get('use_valuenorm', False)

        assert (self._use_popart and self._use_valuenorm) == False, (
            "use_popart and use_valuenorm cannot be both True")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """Calculate value function loss. Always use Huber loss."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        # Always use Huber loss (removed switch)
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Single PPO update step. Modified to use dict access and shape validation.
        FIXED: Added comprehensive shape guards to prevent batch mismatch errors.
        """
        # Use dict access - keep original 3D format for validation
        share_obs_batch = check(sample["share_obs"]).to(**self.tpdv)  # [L, B, share_obs_dim]
        obs_batch = check(sample["obs"]).to(**self.tpdv)  # [L, B, obs_dim]
        rnn_states_batch = check(sample["rnn_states_actor"]).to(**self.tpdv)  # [B, H]
        rnn_states_critic_batch = check(sample["rnn_states_critic"]).to(**self.tpdv)  # [B, H]
        actions_batch = check(sample["actions"]).to(**self.tpdv)  # [L, B, act_dim]
        value_preds_batch = check(sample["value_preds"]).to(**self.tpdv)  # [L, B, 1]
        return_batch = check(sample["returns"]).to(**self.tpdv)  # [L, B, 1]
        masks_batch = check(sample["masks"]).to(**self.tpdv)  # [L, B, 1]
        old_action_log_probs_batch = check(sample["action_log_probs"]).to(**self.tpdv)  # [L, B, 1]
        adv_targ = check(sample["advantages"]).to(**self.tpdv)  # [L, B, 1]

        # FIXED: Shape validation guards to catch dimension mismatches early
        L, B = obs_batch.shape[:2]
        
        assert actions_batch.shape[:2] == (L, B), \
            f"[ppo_update] actions {actions_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert masks_batch.shape[:2] == (L, B), \
            f"[ppo_update] masks {masks_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert share_obs_batch.shape[:2] == (L, B), \
            f"[ppo_update] share_obs {share_obs_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert value_preds_batch.shape[:2] == (L, B), \
            f"[ppo_update] value_preds {value_preds_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert return_batch.shape[:2] == (L, B), \
            f"[ppo_update] returns {return_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert old_action_log_probs_batch.shape[:2] == (L, B), \
            f"[ppo_update] old_action_log_probs {old_action_log_probs_batch.shape[:2]} != obs {obs_batch.shape[:2]}"
        assert adv_targ.shape[:2] == (L, B), \
            f"[ppo_update] advantages {adv_targ.shape[:2]} != obs {obs_batch.shape[:2]}"
        
        # Flatten for network processing
        share_obs_flat = share_obs_batch.view(L * B, -1)  # [L*B, share_obs_dim]
        obs_flat = obs_batch.view(L * B, -1)  # [L*B, obs_dim]
        actions_flat = actions_batch.view(L * B, -1)  # [L*B, act_dim]
        masks_flat = masks_batch.view(L * B, -1)  # [L*B, 1]

        # FIXED: Post-flatten validation to ensure strict consistency
        assert actions_flat.size(0) == obs_flat.size(0), \
            f"[ppo_update] flatten mismatch: actions {actions_flat.size(0)} vs obs {obs_flat.size(0)}"
        assert masks_flat.size(0) == obs_flat.size(0), \
            f"[ppo_update] flatten mismatch: masks {masks_flat.size(0)} vs obs {obs_flat.size(0)}"
        assert share_obs_flat.size(0) == obs_flat.size(0), \
            f"[ppo_update] flatten mismatch: share_obs {share_obs_flat.size(0)} vs obs {obs_flat.size(0)}"

        # Forward pass through networks with flattened inputs
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_flat, obs_flat, rnn_states_batch, rnn_states_critic_batch,
            actions_flat, masks_flat
        )
        
        # FIXED: Additional validation after forward pass
        assert action_log_probs.size(0) == actions_flat.size(0), \
            f"[ppo_update] forward pass mismatch: action_log_probs {action_log_probs.size(0)} vs actions {actions_flat.size(0)}"
        assert values.size(0) == actions_flat.size(0), \
            f"[ppo_update] forward pass mismatch: values {values.size(0)} vs actions {actions_flat.size(0)}"
        
        # Flatten target tensors for loss computation
        value_preds_batch = value_preds_batch.view(L * B, -1)  # [L*B, 1]
        return_batch = return_batch.view(L * B, -1)  # [L*B, 1]
        old_action_log_probs_batch = old_action_log_probs_batch.view(L * B, -1)  # [L*B, 1]
        adv_targ = adv_targ.view(L * B, -1)  # [L*B, 1]
        
        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        # Always use gradient clipping
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)

        self.policy.actor_optimizer.step()

        # Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        # Always use gradient clipping
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

        self.policy.critic_optimizer.step()

        # Calculate PPO monitoring metrics
        with torch.no_grad():
            # Clipping fraction
            clipped = (imp_weights > 1.0 + self.clip_param) | (imp_weights < 1.0 - self.clip_param)
            clipfrac = clipped.float().mean()
            
            # Approximate KL divergence
            approx_kl = (old_action_log_probs_batch - action_log_probs).mean().clamp_min(0)

        # FIXED: Convert tensor grad norms to float to avoid serialization issues
        return {
            "value_loss": value_loss.item(),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "policy_loss": policy_loss.item(),
            "dist_entropy": dist_entropy.item(),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "imp_weights": imp_weights.mean().item(),
            "clipfrac": clipfrac.item(),
            "approx_kl": approx_kl.item(),
        }

    def train(self, buffer, update_actor=True):
        """
        Perform multi-epoch PPO training.
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['clipfrac'] = 0
        train_info['approx_kl'] = 0

        for _ in range(self.ppo_epoch):
            # Always use recurrent generator (simplified)
            data_generator = buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:
                update_info = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += update_info["value_loss"]
                train_info['policy_loss'] += update_info["policy_loss"]
                train_info['dist_entropy'] += update_info["dist_entropy"]
                train_info['actor_grad_norm'] += update_info["actor_grad_norm"]
                train_info['critic_grad_norm'] += update_info["critic_grad_norm"]
                train_info['ratio'] += update_info["imp_weights"]
                train_info['clipfrac'] += update_info["clipfrac"]
                train_info['approx_kl'] += update_info["approx_kl"]

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Set networks to training mode."""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.policy.actor.eval()
        self.policy.critic.eval()