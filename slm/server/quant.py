"""Post-training int8 quantization of a loaded GPT.

Variants (compared in benchmarks/quantization.py):
  none          float weights
  int8          our weight-only int8 for every linear layer in the transformer blocks
  int8-emb      int8 + the embedding table (shared with lm_head when tied) in int8 too
  dynamic       torch's quantize_dynamic on the block linears (int8 weights and activations; CPU only)
  dynamic-head  dynamic + lm_head
"""

import io
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from slm.model import GPT

VARIANTS = ("none", "int8", "int8-emb", "dynamic", "dynamic-head")


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: w ≈ q * scale[:, None], with q in [-127, 127]."""
    w = w.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
    return torch.round(w / scale[:, None]).to(torch.int8), scale


class Int8Linear(nn.Module):
    """Weight-only int8 linear layer: one scale per output channel, activations stay in float.

    x @ (q * s)^T == (x @ q^T) * s, so the scale is applied to the output instead of rebuilding a
    scaled float weight; only the int8 -> float cast of q happens per call.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        q, scale = quantize_rows(weight)
        self.register_buffer("weight_int8", q)
        self.register_buffer("scale", scale.to(weight.dtype))

    def forward(self, x):
        return F.linear(x, self.weight_int8.to(x.dtype)) * self.scale


class Int8Embedding(nn.Module):
    """Row lookup in an Int8Linear's table; with tied embeddings it shares lm_head's int8 buffers."""

    def __init__(self, table: Int8Linear):
        super().__init__()
        self.weight_int8, self.scale = table.weight_int8, table.scale

    def forward(self, idx):
        return self.weight_int8[idx].to(self.scale.dtype) * self.scale[idx, None]


def _replace_linears(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child.weight.data))
        else:
            _replace_linears(child)


def quantize(model: GPT, variant: str) -> nn.Module:
    """Quantize in place (our variants) or return a quantized copy (dynamic variants).

    Call after moving the model to its device: .to() would give tied modules separate buffer copies.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant.startswith("dynamic"):
        weight = model.norm.weight
        if weight.device.type != "cpu" or weight.dtype != torch.float32:
            raise ValueError("quantize_dynamic needs a float32 model on CPU")
        names = {name for name, m in model.named_modules()
                 if isinstance(m, nn.Linear) and (name != "lm_head" or variant == "dynamic-head")}
        # torch.ao.quantization is deprecated in favour of torchao; it is kept here only as a reference point.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.qint8)
    if variant != "none":
        _replace_linears(model.blocks)
    if variant == "int8-emb":
        head = Int8Linear(model.lm_head.weight.data)
        table = head if model.cfg.tie_embeddings else Int8Linear(model.embed.weight.data)
        model.lm_head, model.embed = head, Int8Embedding(table)
    return model


def weight_bytes(model: nn.Module) -> int:
    """Size of the serialized weights: counts packed int8 weights of dynamic layers and stores
    shared (tied) tensors once."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()
