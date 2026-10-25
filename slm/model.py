"""Decoder-only transformer: RoPE, RMSNorm (pre-norm), SwiGLU, no biases, optional grouped-query
attention and tied embeddings. The same layout as Llama, so SmolLM2 weights load directly.

Shape legend: B batch, T new tokens, L attended keys, D d_model, H heads, KVH kv heads, Dh head dim, V vocab.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from slm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize in float32: the mean of squares can overflow in fp16.
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x32.type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim / 2)."""
    freqs = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate (x1, x2) pairs where x1/x2 are the two halves of the head dim; this is HF Llama's
    rotate_half layout, not the interleaved one. x: (B, H, T, Dh)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Preallocated keys/values with one sequence per slot: k, v are (n_layers, n_slots, KVH, max_len, Dh)
    and lengths[s] counts the tokens cached in slot s.

    A forward pass runs on any subset of slots through view(slots), and every row continues at its
    own position: this is what lets continuous batching mix sequences of different lengths.
    """

    def __init__(self, cfg: ModelConfig, n_slots: int, max_len: int, device=None, dtype=torch.float32):
        shape = (cfg.n_layers, n_slots, cfg.kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)
        self.max_len = max_len

    def view(self, slots: list[int]) -> "CacheView":
        return CacheView(self, slots)


class CacheView:
    def __init__(self, cache: KVCache, slots: list[int]):
        self.cache = cache
        self.slots = torch.tensor(slots, device=cache.k.device)
        # Reading the cache with an index list copies it on every layer and step; contiguous slots
        # (the engine keeps active requests in 0..n-1) are read as a slice, which is a free view.
        first = slots[0]
        contiguous = slots == list(range(first, first + len(slots)))
        self.read = slice(first, first + len(slots)) if contiguous else self.slots

    def prepare(self, T: int) -> torch.Tensor:
        """Return positions (B, T) of the new tokens and build the attention mask (B, 1, T, L).

        Keys are cut at L = the longest sequence in this batch after the step, not at max_len, so
        decode cost follows the actual lengths. Each query sees the keys at or before its position,
        which also masks shorter rows' unused tail and stale entries left by a slot's earlier sequence.
        """
        start = self.cache.lengths[self.slots]
        self.positions = start[:, None] + torch.arange(T, device=start.device)
        self.length = int(start.max()) + T
        if self.length > self.cache.max_len:
            raise ValueError(f"sequence length {self.length} exceeds cache max_len {self.cache.max_len}")
        keys = torch.arange(self.length, device=start.device)
        self.mask = (keys[None, None, :] <= self.positions[:, :, None]).unsqueeze(1)
        return self.positions

    def update(self, layer: int, k, v):
        """Write new k/v (B, KVH, T, Dh) at each row's positions; return keys/values (B, KVH, L, Dh)."""
        rows = self.slots[:, None].expand_as(self.positions)
        self.cache.k[layer][rows, :, self.positions] = k.transpose(1, 2)
        self.cache.v[layer][rows, :, self.positions] = v.transpose(1, 2)
        return self.cache.k[layer][self.read, :, :self.length], self.cache.v[layer][self.read, :, :self.length]

    def advance(self, T: int):
        self.cache.lengths[self.slots] += T


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.kv_heads, self.head_dim = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, (cfg.n_heads + 2 * cfg.kv_heads) * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: CacheView | None, layer: int):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split([self.n_heads * self.head_dim, self.kv_heads * self.head_dim,
                                     self.kv_heads * self.head_dim], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        gqa = self.kv_heads != self.n_heads
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:
            k, v = cache.update(layer, k, v)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=cache.mask, enable_gqa=gqa)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache, layer):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.context_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style: scale residual output projections so the residual stream doesn't grow with depth.
        for block in self.blocks:
            for proj in (block.attn.out, block.mlp.down):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, cache: CacheView | None = None):
        """idx: (B, T) token ids -> logits (B, T, V).

        Without a cache the whole sequence is processed causally (training, evaluation). With a
        cache view, row b's tokens continue at that slot's cached length.
        """
        T = idx.shape[1]
        if cache is None:
            if T > self.cfg.context_len:
                raise ValueError(f"sequence exceeds context_len {self.cfg.context_len}")
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        else:
            positions = cache.prepare(T)
            if cache.length > self.cfg.context_len:
                raise ValueError(f"sequence exceeds context_len {self.cfg.context_len}")
            # Per-row positions: (B, T, Dh/2) -> (B, 1, T, Dh/2) to broadcast over heads.
            cos, sin = self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        if cache is not None:
            cache.advance(T)
        return self.lm_head(self.norm(x))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        return n - self.embed.weight.numel() if non_embedding else n
