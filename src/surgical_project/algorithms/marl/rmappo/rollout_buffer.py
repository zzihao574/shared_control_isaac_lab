"""
On-policy rollout buffer with RNN support for rMAPPO.
Complete implementation from user's migration plan.
PHASE 1: Ensured compatibility with eval mode and done_mask shape fixes.
Note: term_masks/trunc_masks additions will be in Phase 2.
"""

import torch

class SharedRolloutBuffer:
    """
    On-policy rollout buffer with RNN support.
    Shapes use (T, N, ...), where N = num_envs * num_agents.
    PHASE 1: Fixed advantage normalization to be the single source of truth.
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

        # PHASE 2: Add term_masks for time-limit bootstrap support
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
        PHASE 1: Ensures masks are properly shaped [N,1] from done_mask fixes.
        PHASE 2: Added term_masks support for time-limit bootstrap.
        """
        assert t == self.step, f"insert step mismatch: {t} vs {self.step}"
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.actions[t].copy_(actions)
        self.action_log_probs[t].copy_(action_log_probs)
        self.value_preds[t].copy_(value_preds)
        self.rewards[t].copy_(rewards)
        self.masks[t].copy_(masks)  # PHASE 1: Now guaranteed to be [N,1] shape
        
        # PHASE 2: Handle term_masks
        if term_masks is not None:
            self.term_masks[t].copy_(term_masks)
        else:
            # Fallback: assume all dones are terminal (conservative)
            self.term_masks[t].copy_(masks)
            
        self.rnn_states_actor[t].copy_(rnn_states_actor)
        self.rnn_states_critic[t].copy_(rnn_states_critic)
        self.step += 1

    @torch.no_grad()
    def compute_returns_and_adv(self, last_values, gamma, gae_lambda):
        """
        Compute returns and advantages using GAE with time-limit bootstrap support.
        PHASE 1: This is the SINGLE SOURCE OF TRUTH for advantage normalization.
        PHASE 2: Added term_masks support for proper time-limit handling.
        last_values: V(s_T) for each rollout slot [N, 1]
        GAE computed backward over t = T-1 ... 0
        """
        T, N = self.T, self.N
        assert self.step == T, "buffer not full when computing returns"
        advantages = torch.zeros(T, N, 1, device=self.device)
        gae = torch.zeros(N, 1, device=self.device)

        for t in reversed(range(T)):
            mask = self.masks[t]  # 0 at episode boundary for next step
            term_mask = self.term_masks[t]  # PHASE 2: 0 for terminal, 1 for time-limit
            
            # PHASE 2: Time-limit bootstrap - allow bootstrap for time-limit, deny for terminal
            next_v_bootstrap = torch.where(term_mask == 0, torch.zeros_like(last_values), last_values)
            
            delta = self.rewards[t] + gamma * next_v_bootstrap * mask - self.value_preds[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            last_values = self.value_preds[t]

        self.advantages.copy_(advantages)
        self.returns = self.advantages + self.value_preds

        # PHASE 1: SINGLE SOURCE advantage normalization - removed from RMAPPOAlgorithm.train()
        # This is the only place where advantages get normalized
        flat_adv = self.advantages.view(T * N, 1)
        valid = self.masks.view(T * N, 1) > 0.5
        mean = flat_adv[valid].mean()
        std = flat_adv[valid].std().clamp_min(1e-6)
        self.advantages = (self.advantages - mean) / std

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """
        Yield mini-batches of sequential chunks for RNN training.
        Only shuffle chunk order, preserve within-chunk time order.
        PHASE 1: Compatible with fixed mask shapes.
        """
        T, N = self.T, self.N
        L = data_chunk_length
        assert T % L == 0, "T must be divisible by data_chunk_length"
        chunks_per_slot = T // L
        total_chunks = N * chunks_per_slot

        perm = torch.randperm(total_chunks)
        mb_size = total_chunks // num_mini_batch

        for mb in range(num_mini_batch):
            idx = perm[mb * mb_size:(mb + 1) * mb_size]

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
                adv_lst.append(self.advantages[t0:t1, slot])  # PHASE 1: Already normalized
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