"""
RNN modules for rMAPPO.
CRITICAL FIX: RNNLayer.forward now uses per-timestep masking instead of segmentation.
This guarantees output length always equals L*B, eliminating batch shape mismatches.
FAIL-FAST: Removed all NaN/Inf repair mechanisms, emergency outputs, and nan_to_num.
"""

import torch
import torch.nn as nn


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


class RNNLayer(nn.Module):
    """GRU RNN layer with 2D external hidden state and per-timestep masking."""
    
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super().__init__()
        self._recurrent_N = recurrent_N
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_N)
        for name, p in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0.0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        Forward pass with strict error checking - no emergency fallbacks.
        x:     [N, feat] (acting)  or [L*B, feat] (training)
        hxs:   [N, H]    (acting)  or [B, H]      (training)
        masks: [N, 1]    (acting)  or [L*B, 1]    (training)
        return:
        out:     [N, H] or [L*B, H]
        hxs_out: [N, H] or [B, H]
        """
        # Input validation - fail immediately on bad inputs
        finite_check("rnn_input_x", x)
        finite_check("rnn_input_hxs", hxs)
        
        assert hxs.dim() == 2, f"expected 2D hxs, got {hxs.shape}"
        layers = self._recurrent_N
        B = hxs.size(0)

        # Normalize masks to 2D float
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if masks.dtype not in (torch.float32, torch.float64):
            masks = masks.float()
        
        finite_check("rnn_input_masks", masks)

        # ---- Acting: single step ----
        if x.dim() == 2 and x.size(0) == B:
            assert masks.dim() == 2 and masks.size(0) == B, f"bad masks {masks.shape}"
            h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()     # [layers,B,H]
            m = masks.view(B, 1)                                                # [B,1]
            m = m.view(1, B, 1).expand(layers, B, 1).contiguous()               # [layers,B,1]
            
            finite_check("rnn_hidden_before_mask", h)
            h = h * m                                                           # reset where mask==0
            finite_check("rnn_hidden_after_mask", h)
            
            out, h = self.rnn(x.unsqueeze(0), h)                                # out:[1,B,H]
            finite_check("rnn_output_acting", out)
            finite_check("rnn_hidden_output_acting", h)
            
            out = out.squeeze(0)                                                # [B,H]
            return self.norm(out), h[-1]                                        # [B,H], [B,H]

        # ---- Training: sequence L×B ----
        assert x.dim() == 2 and x.size(0) % B == 0, f"bad x shape {x.shape} vs B={B}"
        L = x.size(0) // B
        assert masks.dim() == 2 and masks.size(0) == L * B, f"bad masks {masks.shape}"

        x = x.view(L, B, x.size(1))            # [L,B,feat]
        m = masks.view(L, B, 1)                # [L,B,1]

        h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()  # [layers,B,H]
        finite_check("rnn_initial_hidden_training", h)
        
        outs = []

        for t in range(L):
            finite_check(f"rnn_x_t{t}", x[t])
            
            # Build per-timestep mask, target shape [layers,B,1]
            mt = m[t].view(1, B, 1).expand(layers, B, 1).contiguous()
            finite_check(f"rnn_mask_t{t}", mt)
            
            h = h * mt
            finite_check(f"rnn_hidden_masked_t{t}", h)
            
            out_t, h = self.rnn(x[t].unsqueeze(0), h)                     # out_t:[1,B,H]
            finite_check(f"rnn_output_t{t}", out_t)
            finite_check(f"rnn_hidden_output_t{t}", h)
            
            outs.append(out_t)

        out = torch.cat(outs, dim=0).reshape(L * B, -1)                   # [L*B,H]
        finite_check("rnn_concatenated_output", out)
        
        out = self.norm(out)
        finite_check("rnn_normalized_output", out)
        
        return out, h[-1]                                                 # [L*B,H], [B,H]