import copy

import pytest
import torch
from torch import nn

from slm.config import ModelConfig
from slm.model import GPT, KVCache
from slm.server.quant import VARIANTS, Int8Embedding, Int8Linear, quantize, quantize_rows, weight_bytes

CFG = ModelConfig(vocab_size=512, context_len=32, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, ffn_hidden=128)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval().requires_grad_(False)


def test_quantize_rows_error_is_within_half_a_step():
    w = torch.randn(16, 64)
    q, scale = quantize_rows(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert ((q.float() * scale[:, None] - w).abs() <= scale[:, None] / 2 + 1e-7).all()


def test_int8_linear_matches_float():
    linear = nn.Linear(64, 32, bias=False)
    x = torch.randn(4, 64)
    rel = (Int8Linear(linear.weight.data)(x) - linear(x)).norm() / linear(x).norm()
    assert rel < 0.01


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_keep_logits_close(model, variant):
    idx = torch.randint(0, 512, (2, 16))
    q = quantize(copy.deepcopy(model), variant)
    expected, actual = model(idx), q(idx)
    assert (actual.argmax(-1) == expected.argmax(-1)).float().mean() > 0.9
    assert (actual - expected).abs().max() < 0.05 * expected.abs().max()


def test_tied_int8_embedding_shares_storage_and_shrinks_weights(model):
    q = quantize(copy.deepcopy(model), "int8-emb")
    assert isinstance(q.embed, Int8Embedding) and q.embed.weight_int8 is q.lm_head.weight_int8
    sizes = {v: weight_bytes(quantize(copy.deepcopy(model), v)) for v in ("none", "int8", "int8-emb")}
    assert sizes["int8-emb"] < sizes["int8"] < sizes["none"]
    assert sizes["int8-emb"] < 0.35 * sizes["none"]  # ~1/4 plus scales and float norms


def test_quantized_model_works_with_kv_cache(model):
    q = quantize(copy.deepcopy(model), "int8-emb")
    idx = torch.randint(0, 512, (1, 10))
    cache = KVCache(CFG, n_slots=1, max_len=32)
    steps = [q(idx[:, :6], cache.view([0]))] + [q(idx[:, t:t + 1], cache.view([0])) for t in range(6, 10)]
    assert torch.allclose(torch.cat(steps, dim=1), q(idx), atol=1e-5)


def test_dynamic_needs_float32_cpu(model):
    with pytest.raises(ValueError):
        quantize(copy.deepcopy(model).half(), "dynamic")
