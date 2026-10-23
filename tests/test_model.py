import pytest
import torch

from slm.config import ModelConfig
from slm.model import GPT, KVCache

TINY = ModelConfig(vocab_size=100, context_len=32, n_layers=2, d_model=32, n_heads=4, ffn_hidden=64)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(TINY).eval()


def test_default_size_is_about_40m_params():
    n = GPT(ModelConfig()).num_params()
    assert 40e6 < n < 45e6


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
    cache = KVCache(TINY, batch_size=2, max_len=TINY.context_len)
    steps = [model(idx[:, :12], cache)]  # prefill
    steps += [model(idx[:, t:t + 1], cache) for t in range(12, 20)]  # one token at a time
    assert cache.pos == 20
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_rejects_invalid_lengths(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))
    cache = KVCache(TINY, batch_size=1, max_len=TINY.context_len)
    model(torch.zeros(1, 4, dtype=torch.long), cache)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, dtype=torch.long), cache)


def test_embeddings_are_tied(model):
    assert model.lm_head.weight is model.embed.weight
