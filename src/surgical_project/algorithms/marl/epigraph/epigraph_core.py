"""
Epigraph Core Networks for Safe MARL.
Contains: ZEncoder, ActorRNN, CriticVlRNN, CriticVhRNN, RootFinder.
Reuses RNN layer from rMAPPO, implements independent TanhGaussian distribution.

KEY FIX: TanhGaussian.log_prob now includes proper Jacobian correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================================
# REUSED UTILITIES (compatible with rMAPPO style)
# ============================================================================

def init_orthogonal(module, gain=1.0):
    """Initialize module with orthogonal weights (rMAPPO style)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    return module


def init_xavier(module, gain=1.0):
    """Initialize module with Xavier uniform (rMAPPO style)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    return module


# ============================================================================
# TANH GAUSSIAN DISTRIBUTION (Epigraph independent implementation)
# ============================================================================

class TanhGaussian:
    """
    Tanh-Gaussian distribution for bounded continuous actions in [-1, 1].
    Independent implementation with proper Jacobian correction.
    
    CRITICAL: log_prob includes Jacobian determinant correction for tanh transform.
    """
    def __init__(self, mean, std):
        """
        Args:
            mean: [B, act_dim] - raw mean before tanh
            std: [B, act_dim] - standard deviation
        """
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)
    
    def sample(self):
        """Sample action from tanh-transformed distribution."""
        # Sample from base normal
        z = self.normal.rsample()  # reparameterization trick
        # Apply tanh transform
        return torch.tanh(z)
    
    def mode(self):
        """Return deterministic action (mean after tanh)."""
        return torch.tanh(self.mean)
    
    def log_prob(self, action):
        """
        Compute log probability of action under tanh-transformed distribution.
        
        CRITICAL FIX: Includes Jacobian correction for tanh transform.
        
        Args:
            action: [B, act_dim] - action in [-1, 1]
        
        Returns:
            log_prob: [B, 1] - sum over action dimensions
        """
        # Inverse tanh to get pre-transform value
        # atanh(y) = 0.5 * log((1+y)/(1-y))
        eps = 1e-6
        action_clamped = torch.clamp(action, -1 + eps, 1 - eps)
        z = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))
        
        # Log prob of base normal
        log_prob_normal = self.normal.log_prob(z).sum(dim=-1, keepdim=True)
        
        # Jacobian correction for tanh transform
        # |det(J)| = product of (1 - tanh^2(z_i))
        # log|det(J)| = sum of log(1 - tanh^2(z_i)) = sum of log(1 - action^2)
        log_det_jacobian = torch.log(1 - action_clamped ** 2 + eps).sum(dim=-1, keepdim=True)
        
        # Final log prob = log p(z) - log|det(J)|
        return log_prob_normal - log_det_jacobian
    
    def entropy(self):
        """
        Approximate entropy using base distribution entropy.
        Exact entropy for tanh-transformed distribution is intractable.
        """
        # Use base normal entropy as approximation
        return self.normal.entropy().sum(dim=-1)


# ============================================================================
# Z ENCODER (Epigraph-specific)
# ============================================================================

class ZEncoder(nn.Module):
    """
    Encode scalar z into multi-channel feature representation.
    From DEF-MARL: (z - mean) / scale -> Linear(1->nz) -> tanh
    """
    def __init__(self, nz=8, z_mean=0.0, z_scale=0.2, use_orthogonal=True):
        super().__init__()
        self.nz = nz
        self.register_buffer("z_mean", torch.tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_scale", torch.tensor(z_scale, dtype=torch.float32))
        
        self.fc = nn.Linear(1, nz)
        init_fn = init_orthogonal if use_orthogonal else init_xavier
        init_fn(self.fc, gain=1.0)
    
    def forward(self, z):
        """
        Args:
            z: [B, 1] or [B] - scalar z values
        
        Returns:
            z_enc: [B, nz] - encoded features
        """
        if z.dim() == 1:
            z = z.unsqueeze(-1)  # [B] -> [B, 1]
        
        # Normalize z
        z_norm = (z - self.z_mean) / self.z_scale
        
        # Linear + tanh
        z_enc = torch.tanh(self.fc(z_norm))
        return z_enc


# ============================================================================
# ACTOR RNN (Epigraph-specific: obs + z_enc)
# ============================================================================

class ActorRNN(nn.Module):
    """
    Actor network with RNN for Epigraph.
    Input: obs_i ⊕ z_enc -> MLP -> RNN -> TanhGaussian
    """
    def __init__(self, obs_dim, act_dim, z_nz, hidden_size=256, 
                 recurrent_N=1, use_orthogonal=True, gain=0.01):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.z_nz = z_nz
        self.hidden_size = hidden_size
        
        # Input dimension: obs + z_enc
        in_dim = obs_dim + z_nz
        
        # MLP base
        init_fn = init_orthogonal if use_orthogonal else init_xavier
        self.fc1 = init_fn(nn.Linear(in_dim, hidden_size), gain=gain)
        self.fc2 = init_fn(nn.Linear(hidden_size, hidden_size), gain=gain)
        
        # RNN layer
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Action distribution head
        self.fc_mean = init_fn(nn.Linear(hidden_size, act_dim), gain=gain)
        
        # Learnable log std (initialized to -2.0 for stable exploration)
        # Clamped to [-20, 2] range as per RMAPPO
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.0))
        self.log_std_min = -20.0
        self.log_std_max = 2.0
    
    def forward(self, obs, z_enc, rnn_states, masks, deterministic=False):
        """
        Forward pass through actor network.
        
        Args:
            obs: [B, obs_dim] - agent observation
            z_enc: [B, z_nz] - encoded z features
            rnn_states: [B, hidden_size] - RNN hidden states
            masks: [B, 1] - episode continuation mask
            deterministic: bool - if True, return mean action
        
        Returns:
            actions: [B, act_dim] - sampled or deterministic actions
            action_log_probs: [B, 1] - log probabilities
            rnn_states_out: [B, hidden_size] - updated RNN states
        """
        # Concatenate obs and z_enc
        x = torch.cat([obs, z_enc], dim=-1)  # [B, obs_dim + z_nz]
        
        # MLP base
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN forward
        # Reshape for GRU: [1, B, hidden_size]
        x = x.unsqueeze(0)
        h = rnn_states.unsqueeze(0)
        
        # Apply mask to hidden state (episode reset)
        h = h * masks.unsqueeze(0)
        
        # GRU forward
        x, h = self.rnn(x, h)
        
        # Remove sequence dimension
        x = x.squeeze(0)  # [B, hidden_size]
        h = h.squeeze(0)  # [B, hidden_size]
        
        # Layer normalization
        x = self.norm(x)
        
        # Compute mean and std
        mean = self.fc_mean(x)  # [B, act_dim]
        
        # Clamp log_std to prevent numerical issues
        log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std_clamped).expand_as(mean)  # [B, act_dim]
        
        # Create TanhGaussian distribution
        dist = TanhGaussian(mean, std)
        
        # Sample or get mode
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        # Compute log probability
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, h
    
    def evaluate_actions(self, obs, z_enc, rnn_states, masks, actions):
        """
        Evaluate actions for PPO update (compute log_prob and entropy).
        
        Args:
            obs: [L*B, obs_dim]
            z_enc: [L*B, z_nz]
            rnn_states: [B, hidden_size]
            masks: [L*B, 1]
            actions: [L*B, act_dim]
        
        Returns:
            action_log_probs: [L*B, 1]
            dist_entropy: [L*B, 1]
        """
        # Concatenate obs and z_enc
        x = torch.cat([obs, z_enc], dim=-1)
        
        # MLP base
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN sequence processing
        B = rnn_states.size(0)
        L = x.size(0) // B
        
        x = x.view(L, B, -1)
        masks_seq = masks.view(L, B, 1)
        
        h = rnn_states.unsqueeze(0)
        
        outs = []
        for t in range(L):
            h = h * masks_seq[t].unsqueeze(0)
            out_t, h = self.rnn(x[t].unsqueeze(0), h)
            outs.append(out_t)
        
        x = torch.cat(outs, dim=0).reshape(L * B, -1)
        x = self.norm(x)
        
        # Compute mean and std
        mean = self.fc_mean(x)
        log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std_clamped).expand_as(mean)
        
        # Create distribution
        dist = TanhGaussian(mean, std)
        
        # Evaluate actions
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().unsqueeze(-1)  # [L*B, 1]
        
        return action_log_probs, dist_entropy


# ============================================================================
# CRITIC Vl RNN (Task value, centralized)
# ============================================================================

class CriticVlRNN(nn.Module):
    """
    Task value function V^l (centralized).
    Input: share_obs ⊕ global_z_enc -> MLP -> RNN -> V
    """
    def __init__(self, share_obs_dim, z_nz, hidden_size=256,
                 recurrent_N=1, use_orthogonal=True, gain=0.01):
        super().__init__()
        
        self.share_obs_dim = share_obs_dim
        self.z_nz = z_nz
        self.hidden_size = hidden_size
        
        in_dim = share_obs_dim + z_nz
        
        # MLP base
        init_fn = init_orthogonal if use_orthogonal else init_xavier
        self.fc1 = init_fn(nn.Linear(in_dim, hidden_size), gain=gain)
        self.fc2 = init_fn(nn.Linear(hidden_size, hidden_size), gain=gain)
        
        # RNN
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Value head
        self.v_out = init_fn(nn.Linear(hidden_size, 1), gain=gain)
    
    def forward(self, share_obs, z_enc, rnn_states, masks):
        """
        Forward pass through critic Vl.
        
        Args:
            share_obs: [B, share_obs_dim] - centralized observation
            z_enc: [B, z_nz] - global z encoding
            rnn_states: [B, hidden_size] - RNN hidden states
            masks: [B, 1] - continuation masks
        
        Returns:
            values: [B, 1] - task values
            rnn_states_out: [B, hidden_size] - updated states
        """
        x = torch.cat([share_obs, z_enc], dim=-1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN
        x = x.unsqueeze(0)
        h = rnn_states.unsqueeze(0)
        h = h * masks.unsqueeze(0)
        
        x, h = self.rnn(x, h)
        
        x = x.squeeze(0)
        h = h.squeeze(0)
        
        x = self.norm(x)
        values = self.v_out(x)
        
        return values, h
    
    def evaluate_values(self, share_obs, z_enc, rnn_states, masks):
        """
        Evaluate values for sequence (PPO update).
        
        Args:
            share_obs: [L*B, share_obs_dim]
            z_enc: [L*B, z_nz]
            rnn_states: [B, hidden_size]
            masks: [L*B, 1]
        
        Returns:
            values: [L*B, 1]
        """
        x = torch.cat([share_obs, z_enc], dim=-1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN sequence processing
        B = rnn_states.size(0)
        L = x.size(0) // B
        
        x = x.view(L, B, -1)
        masks_seq = masks.view(L, B, 1)
        
        h = rnn_states.unsqueeze(0)
        
        outs = []
        for t in range(L):
            h = h * masks_seq[t].unsqueeze(0)
            out_t, h = self.rnn(x[t].unsqueeze(0), h)
            outs.append(out_t)
        
        x = torch.cat(outs, dim=0).reshape(L * B, -1)
        x = self.norm(x)
        
        values = self.v_out(x)
        return values


# ============================================================================
# CRITIC Vh RNN (Safety value, per-agent)
# ============================================================================

class CriticVhRNN(nn.Module):
    """
    Safety value function V^h (per-agent, decentralized).
    Input: obs_i ⊕ z_enc_i -> MLP -> RNN -> V
    """
    def __init__(self, obs_dim, z_nz, hidden_size=256,
                 recurrent_N=1, use_orthogonal=True, gain=0.01):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.z_nz = z_nz
        self.hidden_size = hidden_size
        
        in_dim = obs_dim + z_nz
        
        # MLP base
        init_fn = init_orthogonal if use_orthogonal else init_xavier
        self.fc1 = init_fn(nn.Linear(in_dim, hidden_size), gain=gain)
        self.fc2 = init_fn(nn.Linear(hidden_size, hidden_size), gain=gain)
        
        # RNN
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Value head
        self.v_out = init_fn(nn.Linear(hidden_size, 1), gain=gain)
    
    def forward(self, obs, z_enc, rnn_states, masks):
        """
        Forward pass through critic Vh.
        
        Args:
            obs: [B, obs_dim] - agent observation
            z_enc: [B, z_nz] - encoded z (per-agent)
            rnn_states: [B, hidden_size] - RNN hidden states
            masks: [B, 1] - continuation masks
        
        Returns:
            values: [B, 1] - safety values
            rnn_states_out: [B, hidden_size] - updated states
        """
        x = torch.cat([obs, z_enc], dim=-1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN
        x = x.unsqueeze(0)
        h = rnn_states.unsqueeze(0)
        h = h * masks.unsqueeze(0)
        
        x, h = self.rnn(x, h)
        
        x = x.squeeze(0)
        h = h.squeeze(0)
        
        x = self.norm(x)
        values = self.v_out(x)
        
        return values, h
    
    def evaluate_values(self, obs, z_enc, rnn_states, masks):
        """
        Evaluate values for sequence (PPO update).
        
        Args:
            obs: [L*B, obs_dim]
            z_enc: [L*B, z_nz]
            rnn_states: [B, hidden_size]
            masks: [L*B, 1]
        
        Returns:
            values: [L*B, 1]
        """
        x = torch.cat([obs, z_enc], dim=-1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # RNN sequence processing
        B = rnn_states.size(0)
        L = x.size(0) // B
        
        x = x.view(L, B, -1)
        masks_seq = masks.view(L, B, 1)
        
        h = rnn_states.unsqueeze(0)
        
        outs = []
        for t in range(L):
            h = h * masks_seq[t].unsqueeze(0)
            out_t, h = self.rnn(x[t].unsqueeze(0), h)
            outs.append(out_t)
        
        x = torch.cat(outs, dim=0).reshape(L * B, -1)
        x = self.norm(x)
        
        values = self.v_out(x)
        return values


# ============================================================================
# ROOT FINDER (Epigraph-specific)
# ============================================================================

class RootFinder:
    """
    Batch root finder for solving Vh(o, z) - z = 0.
    Uses bisection method with vectorized PyTorch operations.
    From DEF-MARL chandrupatla + root_finder implementation.
    """
    def __init__(self, z_min=-0.6, z_max=0.6, max_iter=32, tol=1e-4):
        """
        Args:
            z_min: Lower bound for z search
            z_max: Upper bound for z search
            max_iter: Maximum iterations for bisection
            tol: Convergence tolerance
        """
        self.z_min = z_min
        self.z_max = z_max
        self.max_iter = max_iter
        self.tol = tol
    
    @torch.no_grad()
    def solve(self, vh_fn, obs, rnn_states_vh, masks):
        """
        Solve for z* such that Vh(obs, z*) - z* = 0 using bisection.
        
        Args:
            vh_fn: Callable - function(z, obs, rnn_states, masks) -> Vh(obs, z)
            obs: [B, obs_dim] - observations
            rnn_states_vh: [B, hidden_size] - RNN states for Vh
            masks: [B, 1] - continuation masks
        
        Returns:
            z_star: [B, 1] - root solutions
        """
        B = obs.size(0)
        device = obs.device
        
        # Initialize bounds
        a = torch.full((B, 1), self.z_min, device=device)
        b = torch.full((B, 1), self.z_max, device=device)
        
        # Evaluate function at bounds
        fa = self._eval_residual(vh_fn, a, obs, rnn_states_vh, masks)
        fb = self._eval_residual(vh_fn, b, obs, rnn_states_vh, masks)
        
        # Bisection iteration
        for i in range(self.max_iter):
            # Midpoint
            c = 0.5 * (a + b)
            fc = self._eval_residual(vh_fn, c, obs, rnn_states_vh, masks)
            
            # Check convergence
            if torch.abs(fc).max() < self.tol:
                break
            
            # Update bounds based on sign
            # If fa and fc have same sign, move a to c
            # Otherwise, move b to c
            same_sign = (fa * fc) > 0
            a = torch.where(same_sign, c, a)
            fa = torch.where(same_sign, fc, fa)
            b = torch.where(same_sign, b, c)
            fb = torch.where(same_sign, fb, fc)
        
        # Return best estimate
        z_star = 0.5 * (a + b)
        return z_star
    
    def _eval_residual(self, vh_fn, z, obs, rnn_states_vh, masks):
        """
        Evaluate residual function: R(z) = Vh(obs, z) - z
        
        Args:
            vh_fn: Callable
            z: [B, 1] - z values
            obs: [B, obs_dim] - observations
            rnn_states_vh: [B, hidden_size] - RNN states
            masks: [B, 1] - masks
        
        Returns:
            residual: [B, 1] - Vh(obs, z) - z
        """
        vh_values = vh_fn(z, obs, rnn_states_vh, masks)
        return vh_values - z