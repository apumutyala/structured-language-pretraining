"""Small decoder-only transformer (pre-LN, rotary positions) with optional belief heads.

Arm A "plain": next-token CE on the answer token only.
Arm B "structured": same answer loss + aux_weight * mean over layers l=1..L of
CE(head_l(h_l[final]), slot support labels). h_l[final] is the residual stream
after block l at the final prompt position (the ":" after "answer", whose
next-token prediction is the answer). Slot 0 = the queried proposition; slots
1..K_MAX = every distinct proposition in order of first mention (padding ignored).
Aux targets are always the TRUE exact support labels, also in deceptive contexts.
The aux heads are separate linear maps; they are not used for the answer and are
excluded from all probes (probes are fit fresh on frozen hidden states).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, Dh)."""
    T, Dh = x.shape[-2], x.shape[-1]
    half = Dh // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    ang = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * freq[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    """Causal multi-head self-attention.

    Heads [0, n_tied) are *content* heads: key = query projection (tied QK) and no rotary
    rotation, so identical representations score highly from initialization (an inductive
    bias for in-context lookup). Remaining heads are standard untied heads with rotary
    positions (positional / previous-token routing). Same architecture in both arms.
    """
    def __init__(self, d, h, dropout, n_tied=0):
        super().__init__()
        self.h = h
        self.n_tied = n_tied
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, mask, pad_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.n_tied:
            t = self.n_tied
            k = torch.cat([q[:, :t], rope(k[:, t:])], 1)
            q = torch.cat([q[:, :t], rope(q[:, t:])], 1)
        else:
            q, k = rope(q), rope(k)
        # causal mask only: padding is on the right, so real positions never see pads
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class IdentityRecall(nn.Module):
    """Fixed (non-learned) bigram-identity attention: position i attends uniformly to earlier
    positions j<i whose (previous token, token) pair equals its own (or to itself if none);
    it reads the residual stream after block 1, so matched occurrences carry local context. Only the value and
    output maps are learned. This hard-codes the in-context matching primitive that small
    from-scratch transformers failed to discover within the CPU budget (see RESULTS.md, pilots).
    Used identically in both arms (architectural prior, not supervision)."""
    def __init__(self, d, n_out=1):
        super().__init__()
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ids):
        T = ids.shape[1]
        prev = torch.cat([torch.full_like(ids[:, :1], -1), ids[:, :-1]], 1)
        # bigram identity: same token AND same preceding token (two-word names)
        same = (ids[:, :, None] == ids[:, None, :]) & (prev[:, :, None] == prev[:, None, :])
        earlier = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device), -1)
        m = same & earlier[None]
        none = ~m.any(-1, keepdim=True)
        m = m | (none & torch.eye(T, dtype=torch.bool, device=ids.device)[None])
        w = m.float() / m.float().sum(-1, keepdim=True)
        return self.o(w @ self.v(x))


class Block(nn.Module):
    def __init__(self, d, h, mlp_mult, dropout, n_tied=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout, n_tied)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, pad_mask):
        y = self.attn(self.ln1(x), mask, pad_mask)
        x = x + self.drop(y)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, mlp_mult=4, max_len=128, dropout=0.0,
                 n_slots=9, n_classes=3, aux=False, aux_seed=None, n_tied=0, identity_recall=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        # positions are encoded with RoPE inside attention; no learned absolute embedding
        self.pos = None
        self.blocks = nn.ModuleList(Block(d, heads, mlp_mult, dropout, n_tied) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.idr = IdentityRecall(d) if identity_recall else None
        self.idr_after = 1  # applied to the residual stream after block 1
        self.apply(self._init)
        self.n_slots, self.n_classes = n_slots, n_classes
        self.aux = None
        if aux:
            rng_state = torch.random.get_rng_state()
            # Built after the shared trunk, from a separate generator, so the trunk init is
            # bit-identical between arms for the same seed.
            g = torch.Generator().manual_seed(int(aux_seed if aux_seed is not None else 0) + 7777)
            self.aux = nn.ModuleList(nn.Linear(d, n_slots * n_classes) for _ in range(layers))
            with torch.no_grad():
                for lin in self.aux:
                    lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.02)
                    lin.bias.zero_()
            torch.random.set_rng_state(rng_state)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def trunk_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("aux."))

    def forward(self, ids, lengths):
        """ids: (B,T) right-padded; lengths: (B,). Returns logits at final position and
        list of hidden states at final position for layers 0..L (0 = embeddings)."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        pad_mask = pos[None] >= lengths[:, None]
        last = lengths - 1
        bidx = torch.arange(B, device=ids.device)
        hs = [x[bidx, last]]
        for blk in self.blocks:
            x = blk(x, mask, pad_mask)
            if self.idr is not None and len(hs) == self.idr_after:
                x = x + self.idr(x, ids)
            hs.append(x[bidx, last])
        logits = self.out(self.ln_f(hs[-1]))
        return logits, hs

    def forward_train(self, ids, lengths, prim, sup_b, sup_pos):
        """Teacher-forced pass. Returns logits at supervised positions (sup_b, sup_pos) and the
        per-layer hidden states at each episode's primary answer position `prim`."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        pad_mask = pos[None] >= lengths[:, None]
        bidx = torch.arange(B, device=ids.device)
        hs = [x[bidx, prim]]
        for blk in self.blocks:
            x = blk(x, mask, pad_mask)
            if self.idr is not None and len(hs) == self.idr_after:
                x = x + self.idr(x, ids)
            hs.append(x[bidx, prim])
        logits = self.out(self.ln_f(x[sup_b, sup_pos]))
        return logits, hs

    def aux_loss(self, hs, slot_labels):
        """Mean over layers 1..L of CE over valid slots."""
        losses = []
        for l, head in enumerate(self.aux, start=1):
            lg = head(hs[l]).view(-1, self.n_slots, self.n_classes)
            losses.append(F.cross_entropy(lg.reshape(-1, self.n_classes), slot_labels.reshape(-1),
                                          ignore_index=-100))
        return torch.stack(losses).mean()
