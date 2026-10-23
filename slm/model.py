"""Decoder-only transformer: RoPE, RMSNorm (pre-norm), SwiGLU, no biases, tied embeddings.

Shape legend: B batch, T sequence, D d_model, H heads, Dh head dim, V vocab.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from slm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
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
    """Rotate (x1, x2) pairs, where x1/x2 are the two halves of the head dim. x: (B, H, T, Dh)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Preallocated keys/values for all layers, each (B, H, max_len, Dh); `pos` = tokens cached so far."""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device=None, dtype=torch.float32):
        shape = (cfg.n_layers, batch_size, cfg.n_heads, max_len, cfg.d_model // cfg.n_heads)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, layer: int, k, v):
        """Store new k/v (B, H, T, Dh) at the current position; return all cached k/v for this layer."""
        end = self.pos + k.shape[2]
        self.k[layer, :, :, self.pos:end] = k
        self.v[layer, :, :, self.pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None, layer: int):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        # A multi-token query is a full sequence (training or prefill) and needs the causal mask;
        # a single decode token may attend to every cached position.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
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
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
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
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.context_len, cfg.rope_theta)
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

    def forward(self, idx, cache: KVCache | None = None):
        """idx: (B, T) token ids -> logits (B, T, V). With a cache, positions continue from cache.pos."""
        start = cache.pos if cache is not None else 0
        T = idx.shape[1]
        if start + T > self.cfg.context_len:
            raise ValueError(f"sequence length {start + T} exceeds context_len {self.cfg.context_len}")
        if start > 0 and T > 1:
            # is_causal in SDPA assumes queries and keys start at the same position.
            raise ValueError("multi-token input is only supported on an empty cache (prefill)")
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        if cache is not None:
            cache.pos += T
        return self.lm_head(self.norm(x))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        return n - self.embed.weight.numel() if non_embedding else n
