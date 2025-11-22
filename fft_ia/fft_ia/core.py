import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class FFTInspiredAttention(nn.Module):
    """
    FFT-Inspired Attention (FFT-IA)
    O(N log N) via fixed radix-2 butterfly factorization + Softmax Fidelity
    Paper: "FFT-Inspired Attention (FFT-IA): O(N log N) Complexity via Hierarchical Structural Pruning and Softmax Fidelity"
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_qk = nn.ModuleList()
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # One Q/K projection per log2(N) stage → dynamic content-dependency
        max_stages = 16  # supports up to N=2^16 = 65536
        for _ in range(max_stages):
            self.to_qk.append(nn.Linear(dim, inner_dim * 2, bias=False))

        self.dropout = nn.Dropout(dropout)

    def butterfly_pairs(self, seq_len, stage):
        """
        Fixed radix-2 butterfly connectivity (Decimation-in-Time style)
        stage ∈ [0, log2(N)-1], stride = 2^stage
        """
        stride = 1 << stage
        pairs = []
        for i in range(0, seq_len, stride * 2):
            for j in range(stride):
                a = i + j
                b = i + j + stride
                if b < seq_len:
                    pairs.append((a, b))
        return pairs

    def forward(self, x):
        """
        x: (B, N, D)
        Returns: (B, N, D)
        """
        B, N, D = x.shape
        if not (N & (N - 1)) == 0 or N < 4:
            raise ValueError(f"FFT-IA requires power-of-2 sequence length >=4, got {N}")

        v = self.to_v(x)  # (B, N, H*Dh)
        h = v

        logn = (N.bit_length() - 1)
        for stage in range(logn):
            # Dynamic Q/K re-projection from current hidden state h
            qk = self.to_qk[stage](h)  # (B, N, 2*H*Dh)
            q, k = qk.chunk(2, dim=-1)

            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v_stage = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            # Fixed butterfly pairs for this stage
            pairs = self.butterfly_pairs(N, stage)
            new_h = h.clone()

            for a, b in pairs:
                # Local 2-token softmax (Softmax Fidelity)
                dots = torch.einsum('b h i d, b h j d -> b h i j', q[:, :, a:a+1], k[:, :, [a,b]])
                dots = dots * self.scale
                attn = dots.softmax(dim=-1)  # local softmax over {a,b}
                attn = self.dropout(attn)

                # Weighted aggregation (adaptive pooling)
                out_a = torch.einsum('b h i j, b h j d -> b h i d', attn[..., 0:1], v_stage[:, :, [a,b]])
                out_b = torch.einsum('b h i j, b h j d -> b h i d', attn[..., 1:2], v_stage[:, :, [a,b]])

                # Write back
                new_h[:, a] = rearrange(out_a.squeeze(2), 'b h d -> b (h d)')
                new_h[:, b] = rearrange(out_b.squeeze(2), 'b h d -> b (h d)')

            h = new_h

        out = rearrange(h, 'b n (h d) -> b n (h d)', h=self.heads)
        return out
