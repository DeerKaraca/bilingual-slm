"""Parity with the reference HF Llama implementation on tiny random models (no download needed)."""

import json

import pytest
import torch

from slm.server.loading import config_from_hf, load_hf_model

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_logits_match_transformers_llama(tmp_path, tie):
    torch.manual_seed(0)
    hf_cfg = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,  # GQA, like SmolLM2
        max_position_embeddings=64, rope_theta=100_000.0, rms_norm_eps=1e-5, tie_word_embeddings=tie,
    )
    reference = transformers.LlamaForCausalLM(hf_cfg).eval()
    reference.save_pretrained(tmp_path)  # writes config.json + model.safetensors

    ours = load_hf_model(tmp_path, max_context=64).eval()
    idx = torch.randint(0, 128, (2, 20))
    with torch.no_grad():
        expected = reference(idx).logits
        actual = ours(idx)
    assert ours.cfg.kv_heads == 2 and ours.cfg.norm_eps == 1e-5 and ours.cfg.rope_theta == 100_000.0
    assert torch.allclose(actual, expected, atol=1e-4)


def test_rejects_unsupported_configs():
    base = json.loads(transformers.LlamaConfig(num_attention_heads=4, hidden_size=64).to_json_string())
    config_from_hf(base, 64)
    assert config_from_hf({**base, "rope_theta": 5.0, "rope_parameters": None}, 64).rope_theta == 5.0  # old layout
    for change in [
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {"rope_parameters": {"rope_type": "llama3", "rope_theta": 5e5}},
        {"attention_bias": True},
        {"model_type": "mistral"},
    ]:
        with pytest.raises(ValueError):
            config_from_hf({**base, **change}, 64)
