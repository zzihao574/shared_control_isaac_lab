"""
rMAPPO core components: Actor-Critic networks, Policy wrapper, and Algorithm trainer.
Simplified naming: RMAPPOPolicy (policy) + RMAPPOTrainer (algorithm).
"""

import numpy as np
import torch
import torch.nn as nn
from .utils import init, check, get_shape_from_obs_space, ACTLayer, PopArt
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
        self._use_policy_active_masks = args.get('use_policy_active_masks', True)
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

        # Action layer
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """Compute actions from the given inputs."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """Compute log probability and entropy of given actions."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
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
# POLICY WRAPPER - 重命名为更清晰的名称
# =============================================================================

class RMAPPOPolicy:
    """
    rMAPPO Policy class - 策略包装器
    职责：管理网络和优化器，提供推理接口
    """

    def __init__(self, obs_space_desc, cent_obs_space_desc, act_space_desc, device, args):
        self.device = device
        self.args = args
        
        # Extract learning rates and optimizer settings
        self.actor_lr = args.get('actor_lr', 3e-4)
        self.critic_lr = args.get('critic_lr', 3e-4) 
        self.opt_eps = args.get('opt_eps', 1e-5)
        self.weight_decay = args.get('weight_decay', 0.0)

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

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates."""
        def update_linear_schedule(optimizer, episode, episodes, initial_lr):
            lr = initial_lr - (initial_lr * (episode / float(episodes)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False):
        """Compute actions and value function predictions."""
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions."""
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, 
                        masks, available_actions=None, active_masks=None):
        """Get action logprobs / entropy and value function predictions for update."""
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """Compute actions using the given inputs."""
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor


# =============================================================================
# ALGORITHM TRAINER - 重命名为更清晰的名称
# =============================================================================

def get_gard_norm(parameters):
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


class RMAPPOTrainer:
    """
    rMAPPO Algorithm Trainer 
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
        self._use_value_active_masks = args.get('use_value_active_masks', True)
        self._use_policy_active_masks = args.get('use_policy_active_masks', True)

        assert (self._use_popart and self._use_valuenorm) == False, (
            "use_popart and use_valuenorm cannot be both True")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
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

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """Single PPO update step."""
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch
        )
        
        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        # Always use gradient clipping
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)

        self.policy.actor_optimizer.step()

        # Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        # Always use gradient clipping
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform multi-epoch PPO training.
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns - self.value_normalizer.denormalize(buffer.value_preds)
        else:
            advantages = buffer.returns - buffer.value_preds
        
        # Advantage normalization
        advantages_copy = advantages.clone()
        advantages_copy[buffer.active_masks == 0.0] = float('nan')
        mean_advantages = torch.nanmean(advantages_copy)
        std_advantages = torch.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            # Always use recurrent generator (simplified)
            data_generator = buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                    self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

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