"""
RNN modules for rMAPPO.
CRITICAL FIX: RNNLayer.forward now enforces external 2D hidden states and internal 3D handling.
This eliminates the "batch dimension confusion" where num_layers was mistaken for batch size.
PHASE 1: No changes needed - this module works correctly with Phase 1 modifications.
"""

import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    """RNN layer with GRU implementation - FIXED for 2D external hidden states."""
    
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        FIXED: External 2D, Internal 3D handling
        x:     [N, feat] (acting)  or  [L*B, feat] (training)
        hxs:   [N, H]    (acting)  or  [B, H]      (training)
        masks: [N, 1]    (acting)  or  [L*B, 1]    (training)
        return:
          x_out:   same time flatten as input  ->  [N, H] or [L*B, H]
          hxs_out: [N, H] or [B, H]  (始终 2D；多层取最后一层)
        """
        assert hxs.dim() == 2, f"RNN expects 2D hidden state externally, got {hxs.shape}"
        batch_N = hxs.size(0)
        layers = self._recurrent_N

        def _gru_step(x2d, h2d, m2d):
            # x2d: [N, feat], h2d: [N, H], m2d: [N, 1]
            h0 = h2d.unsqueeze(0).expand(layers, batch_N, h2d.size(1)).contiguous()  # [layers, N, H]
            m0 = m2d.view(batch_N, 1).unsqueeze(0).expand(layers, batch_N, 1).contiguous()  # [layers,N,1]
            out, h_n = self.rnn(x2d.unsqueeze(0), h0 * m0)   # out:[1,N,H], h_n:[layers,N,H]
            return out.squeeze(0), h_n[-1]                   # -> [N,H], [N,H]

        # Acting 单步分支：x = [N, feat]
        if x.dim() == 2 and x.size(0) == batch_N:
            assert masks.dim() == 2 and masks.size(0) == batch_N, f"bad masks {masks.shape}"
            out, hxs_out = _gru_step(x, hxs, masks)
            return self.norm(out), hxs_out

        # 训练序列分支：x = [L*B, feat]，此时 batch_N= B
        assert x.dim() == 2 and x.size(0) % batch_N == 0, f"bad x shape {x.shape} vs B={batch_N}"
        L = x.size(0) // batch_N
        assert masks.dim() == 2 and masks.size(0) == L * batch_N, f"bad masks {masks.shape}"

        x = x.view(L, batch_N, x.size(1))           # [L, B, feat]
        masks = masks.view(L, batch_N, 1)           # [L, B, 1]

        # 找分段（任一序列断点），分段起点对 h 乘 mask
        has_zeros = ((masks[1:] == 0.0).any(dim=1).nonzero().squeeze().cpu())
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        has_zeros = [0] + has_zeros + [L]

        h = hxs.unsqueeze(0).expand(layers, batch_N, hxs.size(1)).contiguous()  # [layers,B,H]
        outputs = []
        for i in range(len(has_zeros) - 1):
            t0, t1 = has_zeros[i], has_zeros[i + 1]
            m0 = masks[t0].unsqueeze(0).expand(layers, batch_N, 1).contiguous() # [layers,B,1]
            h = h * m0
            out, h = self.rnn(x[t0:t1], h)                                      # out:[Δt,B,H]
            outputs.append(out)

        out = torch.cat(outputs, dim=0).reshape(L * batch_N, -1)                # [L*B, H]
        out = self.norm(out)
        hxs_out = h[-1]                                                         # [B, H]
        return out, hxs_out