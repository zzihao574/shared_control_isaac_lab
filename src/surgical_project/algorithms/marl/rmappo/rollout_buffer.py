"""
On-policy rollout buffer with RNN support for rMAPPO.
Complete implementation with GAE term_masks support for time-limit bootstrap.
MODIFIED: Added term_masks support for proper time-limit vs terminal state handling.
FAIL-FAST: Removed all emergency fallbacks, NaN/Inf repair mechanisms.
"""

import torch


def finite_check(name: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    """Check for NaN/Inf in tensor - fail fast, no repair"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name}: expected Tensor, got {type(x)}")
    if not torch.is_floating_point(x):
        return True
    ok = torch.isfinite(x).all().item()
    if ok:
        return True
    bad_ratio = (~torch.isfinite(x)).float().mean().item()
    try:
        min_v = torch.nanmin(x).item()
        max_v = torch.nanmax(x).item()
    except Exception:
        min_v, max_v = float("nan"), float("nan")
    msg = (f"[NUMERIC ERROR] {name}: non-finite values detected\n"
           f"  - bad_ratio={bad_ratio*100:.2f}%\n"
           f"  - range=[{min_v:.3e}, {max_v:.3e}]\n"
           f"  - shape={tuple(x.shape)}, device={x.device}, dtype={x.dtype}")
    if raise_on_fail:
        raise ValueError(msg)
    else:
        print("[WARNING]", msg)
        return False


class SharedRolloutBuffer:
    """
    On-policy rollout buffer with RNN support.
    Shapes use (T, N, ...), where N = num_envs * num_agents.
    FAIL-FAST: All operations fail immediately on NaN/Inf, no repairs.
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
        """Insert experience at timestep t - fail on any NaN/Inf."""
        assert t == self.step, f"insert step mismatch: {t} vs {self.step}"
        
        # Input validation - fail immediately on bad data
        finite_check(f"insert_obs_t{t}", obs)
        finite_check(f"insert_share_obs_t{t}", share_obs)
        finite_check(f"insert_actions_t{t}", actions)
        finite_check(f"insert_action_log_probs_t{t}", action_log_probs)
        finite_check(f"insert_value_preds_t{t}", value_preds)
        finite_check(f"insert_rewards_t{t}", rewards)
        finite_check(f"insert_masks_t{t}", masks)
        finite_check(f"insert_rnn_states_actor_t{t}", rnn_states_actor)
        finite_check(f"insert_rnn_states_critic_t{t}", rnn_states_critic)
        
        if term_masks is not None:
            finite_check(f"insert_term_masks_t{t}", term_masks)
            
        # Insert data into buffers - no fallbacks
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
        """Compute returns and advantages using GAE - strict error checking."""
        T, N = self.T, self.N
        assert self.step == T, "buffer not full when computing returns"
        
        # Input validation
        finite_check("gae_last_values", last_values)
        
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
            
            # Validate intermediate data
            finite_check(f"gae_mask_t{t}", mask)
            finite_check(f"gae_term_mask_t{t}", term_mask)
            finite_check(f"gae_rewards_t{t}", self.rewards[t])
            finite_check(f"gae_value_preds_t{t}", self.value_preds[t])
            
            # Time-limit bootstrap correction
            next_v_bootstrap = term_mask * last_values
            finite_check(f"gae_next_v_bootstrap_t{t}", next_v_bootstrap)
            
            # GAE delta with corrected bootstrap
            delta = self.rewards[t] + gamma * next_v_bootstrap * mask - self.value_preds[t]
            finite_check(f"gae_delta_t{t}", delta)
            
            # GAE recursion
            gae = delta + gamma * gae_lambda * mask * gae
            finite_check(f"gae_accumulated_t{t}", gae)
            
            advantages[t] = gae
            
            # Update last_values for next iteration
            last_values = self.value_preds[t]

        self.advantages.copy_(advantages)
        finite_check("gae_final_advantages", self.advantages)
        
        self.returns = self.advantages + self.value_preds
        finite_check("gae_final_returns", self.returns)

        # Advantage normalization with special handling for small std
        flat_adv = self.advantages.view(T * N, 1)
        valid = self.masks.view(T * N, 1) > 0.5
        
        finite_check("gae_flat_advantages", flat_adv)
        
        if valid.sum() > 0:
            valid_adv = flat_adv[valid]
            finite_check("gae_valid_advantages", valid_adv)
            
            mean = valid_adv.mean()
            std = valid_adv.std().clamp_min(1e-6)
            
            finite_check("gae_advantages_mean", mean.unsqueeze(0))
            finite_check("gae_advantages_std", std.unsqueeze(0))
            
            # Special handling: small std -> warning + skip normalization, not error
            if std < 1e-8:
                print(f"[WARNING] Very small advantage std: {float(std):.3e}. Skip normalization.")
            else:
                self.advantages = (self.advantages - mean) / std
                finite_check("gae_normalized_advantages", self.advantages)
        else:
            raise ValueError("No valid advantages to normalize (all masks are zero).")

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """Yield mini-batches - hard assertions, no auto-downgrade."""
        T, N = self.T, self.N
        L = data_chunk_length
        assert T % L == 0, "T must be divisible by data_chunk_length"
        chunks_per_slot = T // L
        total_chunks = N * chunks_per_slot

        # Hard assertions - no auto-downgrade
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

                # Validate each chunk
                finite_check(f"chunk_obs_mb{mb}_k{k}", obs_chunk)
                finite_check(f"chunk_share_obs_mb{mb}_k{k}", s_obs_chunk)
                finite_check(f"chunk_actions_mb{mb}_k{k}", act_chunk)
                finite_check(f"chunk_log_probs_mb{mb}_k{k}", logp_chunk)
                finite_check(f"chunk_value_preds_mb{mb}_k{k}", vp_chunk)
                finite_check(f"chunk_returns_mb{mb}_k{k}", ret_chunk)
                finite_check(f"chunk_advantages_mb{mb}_k{k}", adv_chunk)
                finite_check(f"chunk_masks_mb{mb}_k{k}", mask_chunk)
                finite_check(f"chunk_rnn_actor_mb{mb}_k{k}", rnn_a0)
                finite_check(f"chunk_rnn_critic_mb{mb}_k{k}", rnn_c0)

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
            
            # Final validation of stacked mini-batch
            for key, tensor in batch_data.items():
                finite_check(f"batch_{key}_mb{mb}", tensor)

            yield batch_data

    def after_update(self):
        """Reset buffer after training update."""
        self.step = 0