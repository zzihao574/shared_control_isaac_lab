"""
On-policy rollout buffer with Epigraph extensions.
Extends rMAPPO buffer with: z values, dual rewards (task/safe), dual values (Vl/Vh).
Maintains full compatibility with rMAPPO buffer structure.

KEY FIX: term_masks now correctly represents truncation (time-limit) only,
allowing proper bootstrap for time-limited episodes while preventing bootstrap
for naturally terminated episodes.
"""

import torch
import math


class RolloutBufferZ:
    """
    Extended rollout buffer for Epigraph algorithm.
    
    New fields compared to rMAPPO:
    - zs: [T, N, 1] - per-agent z values during training
    - rewards_task: [T, N, 1] - task rewards (r_l)
    - rewards_safe: [T, N, 1] - safety rewards (r_h)
    - value_preds_l: [T, N, 1] - task value predictions (Vl)
    - value_preds_h: [T, N, 1] - safety value predictions (Vh)
    - rnn_states_vh: [T, N, H] - RNN states for Vh critic
    - advantages_task/safe: [T, N, 1] - dual advantages
    - returns_task/safe: [T, N, 1] - dual returns
    
    Shapes: (T, N, ...) where T = rollout_horizon, N = num_envs * num_agents
    """
    
    def __init__(self, T, N, obs_dim, share_obs_dim, act_dim, rnn_hidden_dim, device):
        """
        Initialize extended buffer.
        
        Args:
            T: Rollout horizon length
            N: Total number of agents across all envs (num_envs * num_agents)
            obs_dim: Per-agent observation dimension
            share_obs_dim: Centralized observation dimension
            act_dim: Action dimension
            rnn_hidden_dim: RNN hidden state dimension
            device: torch device
        """
        self.T, self.N = T, N
        self.device = device
        
        # ========== Standard rMAPPO fields ==========
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.share_obs = torch.zeros(T, N, share_obs_dim, device=device)
        self.actions = torch.zeros(T, N, act_dim, device=device)
        self.action_log_probs = torch.zeros(T, N, 1, device=device)
        self.rewards = torch.zeros(T, N, 1, device=device)
        self.masks = torch.ones(T, N, 1, device=device)
        
        # Term masks for time-limit bootstrap (CRITICAL FIX)
        # term_masks = 1 ONLY for truncated episodes (time limit)
        # term_masks = 0 for naturally terminated episodes
        self.term_masks = torch.zeros(T, N, 1, device=device)
        
        # RNN states (actor and critic Vl)
        self.rnn_states_actor = torch.zeros(T, N, rnn_hidden_dim, device=device)
        self.rnn_states_critic = torch.zeros(T, N, rnn_hidden_dim, device=device)
        
        # ========== Epigraph-specific fields ==========
        # Z values (per-agent during training)
        self.zs = torch.zeros(T, N, 1, device=device)
        
        # Dual rewards
        self.rewards_task = torch.zeros(T, N, 1, device=device)
        self.rewards_safe = torch.zeros(T, N, 1, device=device)
        
        # Dual value predictions
        self.value_preds_l = torch.zeros(T, N, 1, device=device)  # Task values (Vl)
        self.value_preds_h = torch.zeros(T, N, 1, device=device)  # Safety values (Vh)
        
        # RNN states for Vh critic
        self.rnn_states_vh = torch.zeros(T, N, rnn_hidden_dim, device=device)
        
        # Dual advantages and returns (computed after rollout)
        self.advantages_task = torch.zeros(T, N, 1, device=device)
        self.advantages_safe = torch.zeros(T, N, 1, device=device)
        self.returns_task = torch.zeros(T, N, 1, device=device)
        self.returns_safe = torch.zeros(T, N, 1, device=device)
        
        # Insertion counter
        self.step = 0
    
    def insert(self, t, *, obs, share_obs, actions, action_log_probs,
               rewards, masks, rnn_states_actor, rnn_states_critic, term_masks,
               zs, rewards_task, rewards_safe, value_preds_l, value_preds_h, rnn_states_vh):
        """
        Insert experience at timestep t.
        
        Args:
            t: Current timestep index
            obs: [N, obs_dim] - agent observations
            share_obs: [N, share_obs_dim] - centralized observations
            actions: [N, act_dim] - executed actions
            action_log_probs: [N, 1] - action log probabilities
            rewards: [N, 1] - total rewards (for logging)
            masks: [N, 1] - continuation masks (0 at episode end)
            rnn_states_actor: [N, H] - actor RNN states
            rnn_states_critic: [N, H] - critic Vl RNN states
            term_masks: [N, 1] - truncation masks (1 for truncated, 0 for terminated)
            zs: [N, 1] - z values (per-agent)
            rewards_task: [N, 1] - task rewards (r_l)
            rewards_safe: [N, 1] - safety rewards (r_h)
            value_preds_l: [N, 1] - task value predictions (Vl)
            value_preds_h: [N, 1] - safety value predictions (Vh)
            rnn_states_vh: [N, H] - critic Vh RNN states
        """
        assert t == self.step, f"Insert step mismatch: {t} vs {self.step}"
        
        # Standard fields
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.actions[t].copy_(actions)
        self.action_log_probs[t].copy_(action_log_probs)
        self.rewards[t].copy_(rewards)
        self.masks[t].copy_(masks)
        self.term_masks[t].copy_(term_masks)
        
        self.rnn_states_actor[t].copy_(rnn_states_actor)
        self.rnn_states_critic[t].copy_(rnn_states_critic)
        
        # Epigraph-specific fields
        self.zs[t].copy_(zs)
        self.rewards_task[t].copy_(rewards_task)
        self.rewards_safe[t].copy_(rewards_safe)
        self.value_preds_l[t].copy_(value_preds_l)
        self.value_preds_h[t].copy_(value_preds_h)
        self.rnn_states_vh[t].copy_(rnn_states_vh)
        
        self.step += 1
    
    @torch.no_grad()
    def compute_gae_dual(self, last_vl, last_vh, gamma, gae_lambda, lambda_safe=1.0):
        """
        Compute dual GAE for task and safety objectives.
        
        CRITICAL FIX: Properly handles term_masks for bootstrap:
        - term_masks = 1: truncated episode (time limit) -> allow bootstrap
        - term_masks = 0: naturally terminated -> no bootstrap
        
        Args:
            last_vl: [N, 1] - final task values (bootstrap)
            last_vh: [N, 1] - final safety values (bootstrap)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            lambda_safe: Weight for combining advantages (default 1.0)
        
        Computes:
            - advantages_task: GAE for task rewards using Vl
            - advantages_safe: GAE for safety rewards using Vh
            - returns_task: Task returns (adv_task + Vl)
            - returns_safe: Safety returns (adv_safe + Vh)
        """
        T, N = self.T, self.N
        assert self.step == T, f"Buffer not full when computing returns: {self.step} vs {T}"
        
        # Validate hyperparameters
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma={gamma} not in [0,1]")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda={gae_lambda} not in [0,1]")
        
        # ========== Compute Task GAE (using Vl) ==========
        advantages_task = torch.zeros(T, N, 1, device=self.device)
        gae_task = torch.zeros(N, 1, device=self.device)
        
        last_v_task = last_vl
        
        for t in reversed(range(T)):
            mask = self.masks[t]        # 0 at any episode boundary
            term_mask = self.term_masks[t]  # 1 only for truncated
            
            # Bootstrap: only use next value for truncated episodes
            # For naturally terminated: next_v = 0
            # For truncated: next_v = last_v (bootstrap)
            next_v_task = term_mask * last_v_task
            
            # TD residual
            delta_task = self.rewards_task[t] + gamma * next_v_task - self.value_preds_l[t]
            
            # GAE recursion
            gae_task = delta_task + gamma * gae_lambda * mask * gae_task
            
            advantages_task[t] = gae_task
            last_v_task = self.value_preds_l[t]
        
        self.advantages_task.copy_(advantages_task)
        self.returns_task = self.advantages_task + self.value_preds_l
        
        # ========== Compute Safety GAE (using Vh) ==========
        advantages_safe = torch.zeros(T, N, 1, device=self.device)
        gae_safe = torch.zeros(N, 1, device=self.device)
        
        last_v_safe = last_vh
        
        for t in reversed(range(T)):
            mask = self.masks[t]
            term_mask = self.term_masks[t]
            
            # Bootstrap with time-limit correction
            next_v_safe = term_mask * last_v_safe
            
            # TD residual
            delta_safe = self.rewards_safe[t] + gamma * next_v_safe - self.value_preds_h[t]
            
            # GAE recursion
            gae_safe = delta_safe + gamma * gae_lambda * mask * gae_safe
            
            advantages_safe[t] = gae_safe
            last_v_safe = self.value_preds_h[t]
        
        self.advantages_safe.copy_(advantages_safe)
        self.returns_safe = self.advantages_safe + self.value_preds_h
        
        # ========== Normalize Task Advantages ==========
        # Combined advantage: A = A_task - lambda_safe * A_safe
        # We normalize task advantages for stable training
        flat_adv_task = self.advantages_task.view(T * N, 1)
        valid = self.masks.view(T * N, 1) > 0.5
        
        if valid.sum() > 0:
            valid_adv = flat_adv_task[valid]
            mean_task = valid_adv.mean()
            std_task = valid_adv.std().clamp_min(1e-6)
            
            if std_task < 1e-8:
                print(f"[WARNING] Very small task advantage std: {float(std_task):.3e}")
            else:
                self.advantages_task = (self.advantages_task - mean_task) / std_task
        else:
            raise ValueError("No valid advantages to normalize")
        
        # ========== Normalize Safety Advantages ==========
        flat_adv_safe = self.advantages_safe.view(T * N, 1)
        
        if valid.sum() > 0:
            valid_adv = flat_adv_safe[valid]
            mean_safe = valid_adv.mean()
            std_safe = valid_adv.std().clamp_min(1e-6)
            
            if std_safe < 1e-8:
                print(f"[WARNING] Very small safety advantage std: {float(std_safe):.3e}")
            else:
                self.advantages_safe = (self.advantages_safe - mean_safe) / std_safe
        else:
            raise ValueError("No valid advantages to normalize")
    
    def recurrent_generator(self, num_mini_batch, data_chunk_length, generator=None):
        """
        Generate mini-batches for recurrent PPO training.
        
        Args:
            num_mini_batch: Number of mini-batches per epoch
            data_chunk_length: Length of RNN sequence chunks
            generator: Optional torch.Generator for reproducibility
        
        Yields:
            batch_data: Dictionary containing all fields for one mini-batch
        """
        T, N = self.T, self.N
        L = data_chunk_length
        
        chunks_per_slot = T // L
        total_chunks = N * chunks_per_slot
        
        # Use ceil to avoid dropping tail data
        mb_size = math.ceil(total_chunks / num_mini_batch)
        
        # Reproducible shuffling with external generator
        if generator is not None:
            perm = torch.randperm(total_chunks, device=torch.device("cpu"), generator=generator)
        else:
            perm = torch.randperm(total_chunks, device=torch.device("cpu"))
        
        for mb in range(num_mini_batch):
            start = mb * mb_size
            end = min((mb + 1) * mb_size, total_chunks)
            
            assert end > start, f"Empty slice: start={start}, end={end}"
            idx = perm[start:end]
            assert idx.numel() > 0, "Empty index tensor"
            
            # Collect chunks
            obs_lst, s_obs_lst, act_lst, logp_lst = [], [], [], []
            mask_lst = []
            rnn_a0_lst, rnn_c0_lst, rnn_vh0_lst = [], [], []
            
            # Epigraph-specific
            z_lst = []
            r_task_lst, r_safe_lst = [], []
            vl_lst, vh_lst = [], []
            ret_task_lst, ret_safe_lst = [], []
            adv_task_lst, adv_safe_lst = [], []
            
            for k in idx:
                slot = int(k) // chunks_per_slot
                ck = int(k) % chunks_per_slot
                t0, t1 = ck * L, (ck + 1) * L
                
                # Standard fields
                obs_lst.append(self.obs[t0:t1, slot])
                s_obs_lst.append(self.share_obs[t0:t1, slot])
                act_lst.append(self.actions[t0:t1, slot])
                logp_lst.append(self.action_log_probs[t0:t1, slot])
                mask_lst.append(self.masks[t0:t1, slot])
                
                rnn_a0_lst.append(self.rnn_states_actor[t0, slot])
                rnn_c0_lst.append(self.rnn_states_critic[t0, slot])
                rnn_vh0_lst.append(self.rnn_states_vh[t0, slot])
                
                # Epigraph-specific
                z_lst.append(self.zs[t0:t1, slot])
                r_task_lst.append(self.rewards_task[t0:t1, slot])
                r_safe_lst.append(self.rewards_safe[t0:t1, slot])
                vl_lst.append(self.value_preds_l[t0:t1, slot])
                vh_lst.append(self.value_preds_h[t0:t1, slot])
                ret_task_lst.append(self.returns_task[t0:t1, slot])
                ret_safe_lst.append(self.returns_safe[t0:t1, slot])
                adv_task_lst.append(self.advantages_task[t0:t1, slot])
                adv_safe_lst.append(self.advantages_safe[t0:t1, slot])
            
            # Stack into mini-batch
            batch_data = {
                # Standard rMAPPO fields
                "obs": torch.stack(obs_lst, dim=1),                    # [L, B, obs_dim]
                "share_obs": torch.stack(s_obs_lst, dim=1),            # [L, B, share_obs_dim]
                "actions": torch.stack(act_lst, dim=1),                # [L, B, act_dim]
                "action_log_probs": torch.stack(logp_lst, dim=1),      # [L, B, 1]
                "masks": torch.stack(mask_lst, dim=1),                 # [L, B, 1]
                "rnn_states_actor": torch.stack(rnn_a0_lst, dim=0),    # [B, H]
                "rnn_states_critic": torch.stack(rnn_c0_lst, dim=0),   # [B, H]
                
                # Epigraph-specific fields
                "zs": torch.stack(z_lst, dim=1),                       # [L, B, 1]
                "rewards_task": torch.stack(r_task_lst, dim=1),        # [L, B, 1]
                "rewards_safe": torch.stack(r_safe_lst, dim=1),        # [L, B, 1]
                "value_preds_l": torch.stack(vl_lst, dim=1),           # [L, B, 1]
                "value_preds_h": torch.stack(vh_lst, dim=1),           # [L, B, 1]
                "returns_task": torch.stack(ret_task_lst, dim=1),      # [L, B, 1]
                "returns_safe": torch.stack(ret_safe_lst, dim=1),      # [L, B, 1]
                "advantages_task": torch.stack(adv_task_lst, dim=1),   # [L, B, 1] - Normalized
                "advantages_safe": torch.stack(adv_safe_lst, dim=1),   # [L, B, 1] - Normalized
                "rnn_states_vh": torch.stack(rnn_vh0_lst, dim=0),      # [B, H]
            }
            
            yield batch_data
    
    def after_update(self):
        """Reset buffer after training update."""
        self.step = 0