"""
RNN modules for rMAPPO.
CRITICAL FIX: RNNLayer.forward now uses per-timestep masking instead of segmentation.
This guarantees output length always equals L*B, eliminating batch shape mismatches.
"""

import torch
import torch.nn as nn


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
        x:     [N, feat] (acting)  or [L*B, feat] (training)
        hxs:   [N, H]    (acting)  or [B, H]      (training)
        masks: [N, 1]    (acting)  or [L*B, 1]    (training)   # 允许 [*,] 一维，下面会规范
        return:
        out:     [N, H] or [L*B, H]
        hxs_out: [N, H] or [B, H]
        """
        assert hxs.dim() == 2, f"expected 2D hxs, got {hxs.shape}"
        layers = self._recurrent_N
        B = hxs.size(0)

        # 规范 masks 形状为二维 [:,1]，并转 float
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if masks.dtype not in (torch.float32, torch.float64):
            masks = masks.float()

        # ---- Acting: single step ----
        if x.dim() == 2 and x.size(0) == B:
            assert masks.dim() == 2 and masks.size(0) == B, f"bad masks {masks.shape}"
            h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()     # [layers,B,H]
            m = masks.view(B, 1)                                                # [B,1]
            m = m.view(1, B, 1).expand(layers, B, 1).contiguous()               # [layers,B,1]  ← 顺序正确
            h = h * m                                                           # reset where mask==0
            out, h = self.rnn(x.unsqueeze(0), h)                                # out:[1,B,H]
            out = out.squeeze(0)                                                # [B,H]
            return self.norm(out), h[-1]                                        # [B,H], [B,H]

        # ---- Training: sequence L×B ----
        assert x.dim() == 2 and x.size(0) % B == 0, f"bad x shape {x.shape} vs B={B}"
        L = x.size(0) // B
        assert masks.dim() == 2 and masks.size(0) == L * B, f"bad masks {masks.shape}"

        x = x.view(L, B, x.size(1))            # [L,B,feat]
        m = masks.view(L, B, 1)                # [L,B,1]

        h = hxs.unsqueeze(0).expand(layers, B, hxs.size(1)).contiguous()  # [layers,B,H]
        outs = []

        for t in range(L):
            # 构造本步 mask，目标形状 [layers,B,1]
            mt = m[t].view(1, B, 1).expand(layers, B, 1).contiguous()     # ✅ 修正后的顺序
            h = h * mt
            out_t, h = self.rnn(x[t].unsqueeze(0), h)                     # out_t:[1,B,H]
            outs.append(out_t)

        out = torch.cat(outs, dim=0).reshape(L * B, -1)                   # [L*B,H]
        out = self.norm(out)
        return out, h[-1]                                                 # [L*B,H], [B,H]
