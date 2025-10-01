"""
On-policy rollout buffer with RNN support for rMAPPO.
Complete implementation with GAE term_masks support for time-limit bootstrap.
MODIFIED: Removed all finite_check functions, relying on PyTorch natural failure.
STABLE: Core assertions for shape validation remain.
"""

import torch


class SharedRolloutBuffer:
    """
    On-policy rollout buffer with RNN support.
    Shapes use (T, N, ...), where N = num_envs * num_agents.
    STABLE: Shape assertions remain, all finite_check removed.
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

        # term_mask = 0 for true terminal states, 1 for time-limit truncation
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
        """Insert experience at timestep t."""
        assert t == self.step, f"insert step mismatch: {t} vs {self.step}"
            
        # Insert data into buffers
        self.obs[t].copy_(obs)
        self.share_obs[t].copy_(share_obs)
        self.actions[t].copy_(actions)
        self.action_log_probs[t].copy_(action_log_probs)
        self.value_preds[t].copy_(value_preds)
        self.rewards[t].copy_(rewards)
        self.masks[t].copy_(masks)
        
        # Handle term_masks properly
        if term_masks is not None:
            self.term_masks[t].copy_(term_masks)
        else:
            # Conservative approach: assume all dones are terminal
            self.term_masks[t].copy_(masks)
            
        self.rnn_states_actor[t].copy_(rnn_states_actor)
        self.rnn_states_critic[t].copy_(rnn_states_critic)
        self.step += 1

    @torch.no_grad()
    def compute_returns_and_adv(self, last_values, gamma, gae_lambda):
        """Compute returns and advantages using GAE."""
        T, N = self.T, self.N
        assert self.step == T, "buffer not full when computing returns"
        
        # Parameter validation
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma={gamma} not in [0,1]")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda={gae_lambda} not in [0,1]")
        
        advantages = torch.zeros(T, N, 1, device=self.device)
        gae = torch.zeros(N, 1, device=self.device)

        for t in reversed(range(T)):
            mask = self.masks[t]  # 0 at episode boundary, 1 otherwise
            term_mask = self.term_masks[t]  # 0 for terminal, 1 for time-limit
            
            # Time-limit bootstrap correction
            next_v_bootstrap = term_mask * last_values
            
            # GAE delta with corrected bootstrap
            delta = self.rewards[t] + gamma * next_v_bootstrap - self.value_preds[t]
            
            # GAE recursion
            gae = delta + gamma * gae_lambda * mask * gae
            
            advantages[t] = gae
            
            # Update last_values for next iteration
            last_values = self.value_preds[t]

        self.advantages.copy_(advantages)
        
        self.returns = self.advantages + self.value_preds

        # Advantage normalization with special handling for small std
        flat_adv = self.advantages.view(T * N, 1)
        valid = self.masks.view(T * N, 1) > 0.5
        
        if valid.sum() > 0:
            valid_adv = flat_adv[valid]
            
            mean = valid_adv.mean()
            std = valid_adv.std().clamp_min(1e-6)
            
            # Special handling: small std -> warning + skip normalization
            if std < 1e-8:
                print(f"[WARNING] Very small advantage std: {float(std):.3e}. Skip normalization.")
            else:
                self.advantages = (self.advantages - mean) / std
        else:
            raise ValueError("No valid advantages to normalize (all masks are zero).")

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """Yield mini-batches - hard assertions, no auto-downgrade."""
        T, N = self.T, self.N
        L = data_chunk_length
        assert T % L == 0, "T must be divisible by data_chunk_length"
        chunks_per_slot = T // L
        total_chunks = N * chunks_per_slot

        # Hard assertions
        assert total_chunks > 0, "Total chunks must be > 0"
        assert 1 <= num_mini_batch <= total_chunks, \
            f"num_mini_batch={num_mini_batch} exceeds total_chunks={total_chunks}"
        
        mb_size = max(1, total_chunks // num_mini_batch)
        perm = torch.randperm(total_chunks)
        
        for mb in range(num_mini_batch):
            start = mb * mb_size
            end = min((mb + 1) * mb_size, total_chunks)
            
            # Hard assertion - no empty slices
            assert end > start, f"Empty slice detected: start={start}, end={end}"
            idx = perm[start:end]
            assert idx.numel() > 0, "Empty index tensor"

            obs_lst, s_obs_lst, act_lst, logp_lst = [], [], [], []
            vp_lst, ret_lst, adv_lst, mask_lst = [], [], [], []
            rnn_a0_lst, rnn_c0_lst = [], []

            for k in idx:
                slot = int(k) // chunks_per_slot
                ck = int(k) % chunks_per_slot
                t0, t1 = ck * L, (ck + 1) * L

                obs_chunk = self.obs[t0:t1, slot]
                s_obs_chunk = self.share_obs[t0:t1, slot]
                act_chunk = self.actions[t0:t1, slot]
                logp_chunk = self.action_log_probs[t0:t1, slot]
                vp_chunk = self.value_preds[t0:t1, slot]
                ret_chunk = self.returns[t0:t1, slot]
                adv_chunk = self.advantages[t0:t1, slot]
                mask_chunk = self.masks[t0:t1, slot]
                rnn_a0 = self.rnn_states_actor[t0, slot]
                rnn_c0 = self.rnn_states_critic[t0, slot]

                obs_lst.append(obs_chunk)
                s_obs_lst.append(s_obs_chunk)
                act_lst.append(act_chunk)
                logp_lst.append(logp_chunk)
                vp_lst.append(vp_chunk)
                ret_lst.append(ret_chunk)
                adv_lst.append(adv_chunk)
                mask_lst.append(mask_chunk)
                rnn_a0_lst.append(rnn_a0)
                rnn_c0_lst.append(rnn_c0)

            # Stack all chunks into mini-batch
            batch_data = {
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

            yield batch_data

    def after_update(self):
        """Reset buffer after training update."""
        self.step = 0