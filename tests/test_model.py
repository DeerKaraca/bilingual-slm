import pytest
import torch

from slm.config import ModelConfig
from slm.model import GPT, KVCache

TINY = ModelConfig(vocab_size=100, context_len=32, n_layers=2, d_model=32, n_heads=4, ffn_hidden=64)
TINY_GQA = ModelConfig(vocab_size=100, context_len=32, n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, ffn_hidden=64)


@pytest.fixture(scope="module", params=[TINY, TINY_GQA], ids=["mha", "gqa"])
def model(request):
    torch.manual_seed(0)
    return GPT(request.param).eval()


def test_default_size_is_about_40m_params():
    n = GPT(ModelConfig()).num_params()
    assert 40e6 < n < 45e6


def test_gqa_shrinks_kv_projection():
    full, grouped = GPT(TINY).blocks[0].attn.qkv, GPT(TINY_GQA).blocks[0].attn.qkv
    assert full.out_features == 3 * 32 and grouped.out_features == 32 + 2 * 2 * 8


def test_logits_shape(model):
    assert model(torch.randint(0, 100, (3, 10))).shape == (3, 10, 100)


def test_is_causal(model):
    idx = torch.randint(0, 100, (1, 12))
    changed = idx.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 100
    assert torch.allclose(model(idx)[:, :8], model(changed)[:, :8], atol=1e-6)


@torch.no_grad()
def test_kv_cache_matches_full_forward(model):
    idx = torch.randint(0, 100, (2, 20))
    full = model(idx)
    cache = KVCache(model.cfg, n_slots=2, max_len=32)
    view = cache.view([0, 1])
    steps = [model(idx[:, :12], view)]  # prefill
    steps += [model(idx[:, t:t + 1], view) for t in range(12, 20)]  # one token at a time
    assert cache.lengths.tolist() == [20, 20]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


@torch.no_grad()
def test_batched_decode_with_different_lengths(model):
    """Continuous batching: rows at different positions, in non-adjacent slots, decoded together."""
    torch.manual_seed(1)
    a, b = torch.randint(0, 100, (1, 9)), torch.randint(0, 100, (1, 15))
    cache = KVCache(model.cfg, n_slots=4, max_len=32)
    cache.k.normal_()  # stale data from an earlier sequence must be masked out
    model(a[:, :-1], cache.view([3]))
    model(b[:, :-1], cache.view([1]))

    view = cache.view([3, 1])
    logits = model(torch.cat([a[:, -1:], b[:, -1:]]), view)[:, -1]
    assert view.length == 15  # attention is cut at the longest row, not at max_len
    assert torch.allclose(logits[0], model(a)[0, -1], atol=1e-5)
    assert torch.allclose(logits[1], model(b)[0, -1], atol=1e-5)
    assert cache.lengths.tolist() == [0, 15, 0, 9]


def test_rejects_too_long_sequences(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))
    cache = KVCache(model.cfg, n_slots=1, max_len=8)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long), cache.view([0]))


def test_embeddings_are_tied(model):
    assert model.lm_head.weight is model.embed.weight
