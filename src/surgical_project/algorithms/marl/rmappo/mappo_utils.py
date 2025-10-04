"""
Utility functions and base modules for rMAPPO.
Contains: basic utilities, continuous action distributions, action layer, and PopArt.
Simplified to support only continuous actions with RNN networks.
MODIFIED: Clean deterministic branch, confirmed logstd_mean() method for monitoring.
STABLE: Core functionality remains, gradient monitoring support via logstd_mean().
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, TanhTransform


# ============================================================================
# BASIC UTILITY FUNCTIONS
# ============================================================================

def init(module, weight_init, bias_init, gain=1):
    """Initialize module weights and biases."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    """Convert numpy array to tensor if needed."""
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_shape_from_obs_space(obs_space):
    """Extract shape from observation space description."""
    if hasattr(obs_space, 'shape'):
        return obs_space.shape
    elif isinstance(obs_space, dict) and 'shape' in obs_space:
        return obs_space['shape']
    else:
        raise ValueError(f"Cannot extract shape from obs_space: {obs_space}")


# ============================================================================
# CONTINUOUS ACTION DISTRIBUTIONS
# ============================================================================

class FixedNormal(torch.distributions.Normal):
    """Fixed Normal distribution for continuous actions."""
    
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class AddBias(nn.Module):
    """Add learnable bias to input."""
    
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class DiagGaussian(nn.Module):
    """Diagonal Gaussian distribution for continuous actions."""
    
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros_like(action_mean)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class TanhDiagGaussian(nn.Module):
    """Tanh-Gaussian distribution for bounded continuous actions in [-1, 1]."""
    
    def __init__(self, in_dim, out_dim, use_orthogonal=True, gain=0.01):
        super().__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            init_method(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
            return m
        
        self.fc_mean = init_(nn.Linear(in_dim, out_dim))
        self.logstd = AddBias(torch.full((out_dim,), -2.0))

    def base_dist(self, x):
        """Get base normal distribution before tanh transform."""
        mean = self.fc_mean(x)
        logstd = self.logstd(torch.zeros_like(mean))
        
        # Model constraint - allowed boundary clamping
        logstd = torch.clamp(logstd, -3.0, -1.0)
        std = logstd.exp()
        
        return FixedNormal(mean, std)

    def dist(self, x):
        """Get transformed distribution (Normal -> Tanh)."""
        base_distribution = self.base_dist(x)
        return TransformedDistribution(
            base_distribution, 
            [TanhTransform(cache_size=1)]
        )


# ============================================================================
# ACTION LAYER
# ============================================================================

class ACTLayer(nn.Module):
    """Action layer for continuous actions with configurable distribution type and gradient monitoring."""
    
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, use_tanh=True):
        super(ACTLayer, self).__init__()
        
        # Only support Box action space (continuous)
        if action_space.__class__.__name__ != "Box":
            raise ValueError(f"Only Box action space supported, got {action_space.__class__.__name__}")
        
        action_dim = action_space.shape[0]
        self.use_tanh = use_tanh
        
        if use_tanh:
            self._dist = TanhDiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        else:
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
    
    def logstd_mean(self):
        """Get mean log standard deviation for gradient monitoring system."""
        if self.use_tanh:
            # For TanhDiagGaussian, access AddBias logstd
            if hasattr(self._dist, "logstd"):
                return float(self._dist.logstd._bias.data.mean().item())
            return 0.0
        else:
            # For DiagGaussian, access AddBias logstd  
            if hasattr(self.action_out, "logstd"):
                return float(self.action_out.logstd._bias.data.mean().item())
            return 0.0
    
    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input."""
        if self.use_tanh:
            if deterministic:
                # For deterministic actions, use mean of base distribution then tanh
                base_mean = self._dist.base_dist(x).mean
                actions = torch.tanh(base_mean)
            else:
                # For stochastic actions, sample from transformed distribution
                d = self._dist.dist(x)
                actions = d.sample()
            
            # Compute log probability for sampled/deterministic actions
            d = self._dist.dist(x)
            action_log_probs = d.log_prob(actions).sum(-1, keepdim=True)
            
        else:
            action_logit = self.action_out(x)
            actions = action_logit.mode() if deterministic else action_logit.sample()
            action_log_probs = action_logit.log_probs(actions)
            
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """Compute action probabilities from inputs."""
        if self.use_tanh:
            # Return deterministic policy action: tanh(base mean)
            base_mean = self._dist.base_dist(x).mean
            return torch.tanh(base_mean)
        else:
            action_logits = self.action_out(x)
            return action_logits.probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """Compute log probability and entropy of given actions."""
        if self.use_tanh:
            d = self._dist.dist(x)
            action_log_probs = d.log_prob(action).sum(-1, keepdim=True)
            
            # For entropy, use base distribution entropy (more stable)
            base_entropy = self._dist.base_dist(x).entropy().sum(-1)
            
            if active_masks is not None:
                if len(base_entropy.shape) == len(active_masks.shape):
                    dist_entropy = (base_entropy * active_masks).sum() / active_masks.sum()
                else:
                    dist_entropy = (base_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = base_entropy.mean()
            
        else:
            action_logit = self.action_out(x)
            action_log_probs = action_logit.log_probs(action)
            
            if active_masks is not None:
                if len(action_logit.entropy().shape) == len(active_masks.shape):
                    dist_entropy = (action_logit.entropy() * active_masks).sum() / active_masks.sum()
                else:
                    dist_entropy = (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logit.entropy().mean()

        return action_log_probs, dist_entropy


# ============================================================================
# POPART NORMALIZATION
# ============================================================================

class PopArt(torch.nn.Module):
    """
    PopArt normalization for value function.
    Preserves the rank of the targets while normalizing the scale.
    """
    
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters to initial values."""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        """Forward pass through PopArt layer."""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        """Update PopArt statistics and adapt weights and biases."""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        
        old_mean, old_stddev = self.mean, self.stddev

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        self.weight = self.weight * old_stddev / self.stddev
        self.bias = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    def debiased_mean_var(self):
        """Get debiased mean and variance."""
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        """Normalize input vector."""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """Denormalize input vector."""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()

        return out