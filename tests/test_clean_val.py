import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from slm.config import ModelConfig
from slm.eval.bpb import evaluate_bpb, token_byte_lengths
from slm.model import GPT
from slm.tokenizer import train_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from clean_val_eval import subset_bpb, token_losses  # noqa: E402


def test_per_token_losses_reproduce_evaluate_bpb():
    tok = train_tokenizer(iter(["İstanbul\x27da hava güzel.", "The weather is nice."] * 20), vocab_size=300)
    torch.manual_seed(0)
    vocab = tok.get_vocab_size()
    model = GPT(ModelConfig(vocab_size=vocab, context_len=16, n_layers=1, d_model=32, n_heads=4, ffn_hidden=64)).eval()
    rng = np.random.default_rng(0)
    tokens = rng.integers(3, vocab, 200).astype(np.uint16)
    tokens[[0, 50, 120]] = 0  # three documents
    byte_lengths = token_byte_lengths(tok)
    losses = token_losses(model, tokens, batch_size=4)

    doc_start = np.flatnonzero(tokens == 0)
    target_doc = np.searchsorted(doc_start, np.arange(1, len(tokens)), side="right") - 1
    target_bytes = byte_lengths.numpy()[tokens[1:]]
    keep_all = np.ones(3, dtype=bool)
    assert subset_bpb(losses, target_doc, target_bytes, keep_all) == pytest.approx(
        evaluate_bpb(model, tokens, byte_lengths, seq_len=16, batch_size=4), rel=1e-5)

    # Dropping a document leaves exactly the targets of the others.
    keep = np.array([True, False, True])
    m = keep[target_doc] & (target_bytes > 0) & ~np.isnan(losses)
    assert subset_bpb(losses, target_doc, target_bytes, keep) == pytest.approx(
        losses[m].sum() / np.log(2) / target_bytes[m].sum())
