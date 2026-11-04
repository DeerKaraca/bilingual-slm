from dataclasses import asdict

import pytest
import torch

from slm.config import ModelConfig
from slm.export import export
from slm.model import GPT
from slm.server.loading import load_hf
from slm.tokenizer import train_tokenizer

CORPUS = ["İstanbul'da hava çok güzel.", "The weather is nice today.", "Merhaba dünya!"] * 20


@pytest.fixture(params=[True, False], ids=["tied", "untied"])
def exported(tmp_path, request):
    tok = train_tokenizer(iter(CORPUS), vocab_size=300)
    tok.save(str(tmp_path / "tokenizer.json"))
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), context_len=64, n_layers=2, d_model=32, n_heads=4,
                      ffn_hidden=64, tie_embeddings=request.param)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    torch.save({"model": model.state_dict(), "config": {"model": asdict(cfg)}, "step": 1}, tmp_path / "final.pt")
    export(tmp_path / "final.pt", tmp_path / "tokenizer.json", tmp_path / "hf")
    return model, tok, tmp_path / "hf"


def test_roundtrip_through_our_loader_is_exact(exported):
    model, tok, out = exported
    loaded = load_hf(str(out), max_context=64)
    idx = torch.randint(0, tok.get_vocab_size(), (2, 16))
    with torch.no_grad():
        assert torch.equal(loaded.model.eval()(idx), model(idx))
    assert loaded.eos_ids == {tok.token_to_id("<|endoftext|>")} and loaded.chat_template is None


def test_transformers_loads_the_export(exported):
    transformers = pytest.importorskip("transformers")
    model, tok, out = exported
    reference = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    idx = torch.randint(0, tok.get_vocab_size(), (2, 16))
    with torch.no_grad():
        assert torch.allclose(reference(idx).logits, model(idx), atol=1e-4)
    hf_tok = transformers.AutoTokenizer.from_pretrained(out)
    text = "İstanbul'da hava güzel"
    assert hf_tok(text)["input_ids"] == tok.encode(text).ids
