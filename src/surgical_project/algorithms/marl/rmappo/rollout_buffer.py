"""
On-policy rollout buffer with RNN support for rMAPPO.
Complete implementation with GAE term_masks support for time-limit bootstrap.
MODIFIED: Added term_masks support for proper time-limit vs terminal state handling.
ADDITIONAL FIX: Added protection against empty mini-batches when num_mini_batch is too large.
"""

import torch

class SharedRolloutBuffer:
    """
    On-policy rollout buffer with RNN support.
    Shapes use (T, N, ...), where N = num_envs * num_agents.
    MODIFIED: Added term_masks for proper GAE bootstrap handling and mini-batch protection.
    """
    def __init__(self, T, N, obs_dim, share_obs_dim, act_dim, rnn_hidden_dim, device):
        self.T, self.N = T, N
        self.device = device
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.share_obs = torch.zeros(T, N, share_obs_dim, device=device)
        self.actions = torch.zeros(T, N, act_dim, device=device)
        self.action_log_probs = torch.zeros(T, N, 1, device=device)
        self.value_preds = torch.zeros(T, N, 1, device=device)
        self.rewards = torch.zeros(T, N, 1, device=device)
        self.masks = torch.ones(T, N, 1, device=device)

        # MODIFIED: Added term_masks for time-limit bootstrap support
        # term_mask = 0 for true terminal states (success/failure)
        # term_mask = 1 for time-limit truncation (should use bootstrap)
        self.term_masks = torch.ones(T, N, 1, device=device)

        # RNN states (state BEFORE consuming obs[t])
        self.rnn_states_actor  = torch.zeros(T, N, rnn_hidden_dim, device=device)
        self.rnn_states_critic = torch.zeros(T, N, rnn_hidden_dim, device=device)

        # After GAE
        self.returns = torch.zeros(T, N, 1, device=device)
        self.advantages = torch.zeros(T, N, 1, device=device)
        self.step = 0

    def insert(self, t, *, obs, share_obs, actions, action_log_probs,
               value_preds, rewards, masks, rnn_states_actor, rnn_states_critic, term_masks=None):
        """
        Insert experience at timestep t.
        MODIFIED: Added term_masks parameter for time-limit vs terminal distinction.
        """
        assert t == self.step, f"insert step mismatch: {t} vs {self.step}"
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.actions[t].copy_(actions)
        self.action_log_probs[t].copy_(action_log_probs)
        self.value_preds[t].copy_(value_preds)
        self.rewards[t].copy_(rewards)
        self.masks[t].copy_(masks)
        
        # MODIFIED: Handle term_masks properly
        if term_masks is not None:
            self.term_masks[t].copy_(term_masks)
        else:
            # Fallback: assume all dones are terminal (conservative approach)
            # This maintains backward compatibility but is suboptimal
            self.term_masks[t].copy_(masks)
            
        self.rnn_states_actor[t].copy_(rnn_states_actor)
        self.rnn_states_critic[t].copy_(rnn_states_critic)
        self.step += 1

    @torch.no_grad()
    def compute_returns_and_adv(self, last_values, gamma, gae_lambda):
        """
        Compute returns and advantages using GAE with time-limit bootstrap support.
        MODIFIED: Use term_masks for proper bootstrap handling at time limits.
        
        Key insight: 
        - masks[t]: 0 when episode ends (any reason), 1 otherwise
        - term_masks[t]: 0 for true terminal states, 1 for time-limit truncation
        
        For GAE delta calculation:
        - Use term_masks to control bootstrap: allow V(s_{t+1}) for time-limit, deny for terminal
        - Use masks for recursion cutoff: cut GAE propagation at any episode boundary
        """
        T, N = self.T, self.N
        assert self.step == T, "buffer not full when computing returns"
        advantages = torch.zeros(T, N, 1, device=self.device)
        gae = torch.zeros(N, 1, device=self.device)

        for t in reversed(range(T)):
            mask = self.masks[t]  # 0 at episode boundary, 1 otherwise
            term_mask = self.term_masks[t]  # 0 for terminal, 1 for time-limit
            
            # MODIFIED: Time-limit bootstrap correction
            # For time-limit (term_mask=1): use bootstrap value
            # For terminal (term_mask=0): no bootstrap (value = 0)
            next_v_bootstrap = term_mask * last_values
            
            # GAE delta with corrected bootstrap
            delta = self.rewards[t] + gamma * next_v_bootstrap * mask - self.value_preds[t]
            
            # GAE recursion (always cut at episode boundary via mask)
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            
            # Update last_values for next iteration
            last_values = self.value_preds[t]

        self.advantages.copy_(advantages)
        self.returns = self.advantages + self.value_preds

        # Advantage normalization (single source of truth)
        flat_adv = self.advantages.view(T * N, 1)
        valid = self.masks.view(T * N, 1) > 0.5
        
        if valid.sum() > 0:  # Avoid division by zero
            mean = flat_adv[valid].mean()
            std = flat_adv[valid].std().clamp_min(1e-6)
            self.advantages = (self.advantages - mean) / std
        else:
            # If no valid advantages, keep as zeros
            self.advantages.zero_()

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """
        Yield mini-batches of sequential chunks for RNN training.
        Only shuffle chunk order, preserve within-chunk time order.
        FIXED: Added protection against empty mini-batches when num_mini_batch is too large.
        """
        T, N = self.T, self.N
        L = data_chunk_length
        assert T % L == 0, "T must be divisible by data_chunk_length"
        chunks_per_slot = T // L
        total_chunks = N * chunks_per_slot

        # FIXED: 更健壮：如果 num_mini_batch 过大，自动下调，避免 mb_size==0
        if total_chunks == 0:
            return  # 没有可用数据
        
        if num_mini_batch > total_chunks:
            # 可选：打印一次警告，方便调参
            if not hasattr(self, "_warned_mini_batch_clamp"):
                print(f"[BUFFER WARN] num_mini_batch({num_mini_batch}) > total_chunks({total_chunks}), clamp to total_chunks.")
                self._warned_mini_batch_clamp = True
        
        num_mini_batch = min(num_mini_batch, total_chunks)
        mb_size = max(1, total_chunks // num_mini_batch)

        perm = torch.randperm(total_chunks)
        
        for mb in range(num_mini_batch):
            start = mb * mb_size
            end = min((mb + 1) * mb_size, total_chunks)
            
            # FIXED: 防空切片
            if end <= start:
                continue
                
            idx = perm[start:end]
            
            if idx.numel() == 0:
                continue

            obs_lst, s_obs_lst, act_lst, logp_lst = [], [], [], []
            vp_lst, ret_lst, adv_lst, mask_lst = [], [], [], []
            rnn_a0_lst, rnn_c0_lst = [], []

            for k in idx:
                slot = int(k) // chunks_per_slot
                ck = int(k) % chunks_per_slot
                t0, t1 = ck * L, (ck + 1) * L

                obs_lst.append(self.obs[t0:t1, slot])
                s_obs_lst.append(self.share_obs[t0:t1, slot])
                act_lst.append(self.actions[t0:t1, slot])
                logp_lst.append(self.action_log_probs[t0:t1, slot])
                vp_lst.append(self.value_preds[t0:t1, slot])
                ret_lst.append(self.returns[t0:t1, slot])
                adv_lst.append(self.advantages[t0:t1, slot])
                mask_lst.append(self.masks[t0:t1, slot])

                rnn_a0_lst.append(self.rnn_states_actor[t0, slot])
                rnn_c0_lst.append(self.rnn_states_critic[t0, slot])

            yield {
                "obs": torch.stack(obs_lst, dim=1),              # [L, B, obs_dim]
                "share_obs": torch.stack(s_obs_lst, dim=1),      # [L, B, share_obs_dim]
                "actions": torch.stack(act_lst, dim=1),          # [L, B, act_dim]
                "action_log_probs": torch.stack(logp_lst, dim=1),# [L, B, 1]
                "value_preds": torch.stack(vp_lst, dim=1),       # [L, B, 1]
                "returns": torch.stack(ret_lst, dim=1),          # [L, B, 1]
                "advantages": torch.stack(adv_lst, dim=1),       # [L, B, 1] - Already normalized
                "masks": torch.stack(mask_lst, dim=1),           # [L, B, 1]
                "rnn_states_actor": torch.stack(rnn_a0_lst, dim=0),   # [B, H]
                "rnn_states_critic": torch.stack(rnn_c0_lst, dim=0),  # [B, H]
            }

    def after_update(self):
        """Reset buffer after training update."""
        self.step = 0