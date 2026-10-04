"""mu-scale GPT approximating the parameter-golf architecture family.

Deliberate simplifications vs. the official train_gpt.py (documented in the report):
  * MHA instead of GQA, no per-tensor int8 storage during training
  * SwiGLU MLP with 8/3*dim hidden instead of the official mlp_mult knob
Kept from the official design: RMSNorm, RoPE (base 10000), bias-free linears,
QK gain, logit softcap 30, untied/tied embedding switch, optional depth recurrence
(weight sharing of the final block, i.e. the "3-Layer Recurrence" family of fixes).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cos = None
        self._sin = None

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._cos.size(0) < seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()
            self._sin = f.sin()
        return self._cos[:seq_len].to(dtype), self._sin[:seq_len].to(dtype)


def apply_rotary(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, qk_gain_init=1.5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.q_gain = nn.Parameter(torch.full((n_heads,), qk_gain_init))

    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, H, hd)
        k = k.view(B, T, H, hd)
        v = v.view(B, T, H, hd)
        cos, sin = self.rotary(T, x.device, q.dtype)
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
        q = apply_rotary(q, c, s) * self.q_gain[None, None, :, None]
        k = apply_rotary(k, c, s)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim, mult=None):
        super().__init__()
        hidden = int(8 * dim / 3) if mult is None else int(dim * mult)
        hidden = ((hidden + 63) // 64) * 64
        self.hidden = hidden
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.n2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, vocab=1024, dim=256, n_layers=4, n_heads=4, tie=True,
                 depth_share=1, logit_softcap=30.0):
        super().__init__()
        assert dim % n_heads == 0
        self.cfg = dict(vocab=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                        tie=bool(tie), depth_share=depth_share, logit_softcap=logit_softcap)
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.tie = bool(tie)
        self.depth_share = int(depth_share)
        self.softcap = float(logit_softcap)
        if self.tie:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2 * n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == last and self.depth_share > 1:
                for _ in range(self.depth_share - 1):
                    x = blk(x)
        logits = self.head(self.norm_f(x))
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
