import csv
import math

import numpy as np
import pytest
import torch
from torch import nn

from slm import train as train_mod
from slm.config import load_config
from slm.data.loader import MixtureLoader
from slm.eval.bpb import evaluate_bpb, token_byte_lengths
from slm.tokenizer import train_tokenizer
from slm.train import lr_at, parse_overrides, train, use_compile

CORPUS = ["İstanbul'da hava çok güzel.", "The weather is nice today."] * 20


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer(iter(CORPUS), vocab_size=300)


@pytest.fixture
def setup(tmp_path, tokenizer):
    """Random token files for both languages plus a matching tiny tokenizer; returns config overrides."""
    vocab = tokenizer.get_vocab_size()
    rng = np.random.default_rng(0)
    for lang in ("tr", "en"):
        (tmp_path / "tokens" / lang).mkdir(parents=True)
        for split, n in [("train", 20_000), ("val", 2_000)]:
            rng.integers(0, vocab, n).astype(np.uint16).tofile(tmp_path / "tokens" / lang / f"{split}.bin")
    tokenizer.save(str(tmp_path / "tok.json"))
    return {
        "model": {"vocab_size": vocab},
        "data": {"tokens_dir": str(tmp_path / "tokens"), "tokenizer": str(tmp_path / "tok.json")},
        "train": {"total_tokens": 6 * 4 * 2 * 128, "warmup_steps": 2, "eval_every": 2, "checkpoint_every_minutes": 0},
    }


def smoke_config(setup, out_dir):
    return load_config("configs/smoke.yaml", {**setup, "out_dir": str(out_dir)})


@pytest.fixture
def id_tokens(tmp_path):
    """Token files whose values are their own positions, so a window identifies its language and offset."""
    for lang, first in [("tr", 0), ("en", 1000)]:
        (tmp_path / lang).mkdir()
        np.arange(first, first + 1000, dtype=np.uint16).tofile(tmp_path / lang / "train.bin")
    return tmp_path


def read_all(loader, n_batches):
    return torch.cat([loader.next_batch()[0] for _ in range(n_batches)])


def test_loader_exact_mixture_without_repetition(id_tokens):
    loader = MixtureLoader(id_tokens, 0.3, batch_size=10, seq_len=8, seed=0, n_sequences=100)
    x = read_all(loader, 10)
    starts = x[:, 0]
    assert (starts < 1000).sum() == 30  # exactly 30% Turkish
    assert len(set(starts.tolist())) == 100  # no window is read twice
    assert ((starts % 1000) % 8 == 0).all()  # windows are aligned, so they never overlap
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)  # each row is one contiguous window


def test_loader_order_is_shared_across_mixtures(id_tokens):
    tr_of = lambda f: [s for s in read_all(MixtureLoader(id_tokens, f, 10, 8, 0, 100), 10)[:, 0].tolist() if s < 1000]
    assert tr_of(0.1) == tr_of(1.0)[:10]


def test_loader_resume(id_tokens):
    loader = MixtureLoader(id_tokens, 0.5, 4, 8, 0, 40)
    read_all(loader, 3)
    resumed = MixtureLoader(id_tokens, 0.5, 4, 8, 0, 40)
    resumed.load_state_dict(loader.state_dict())
    assert torch.equal(resumed.next_batch()[0], loader.next_batch()[0])


def test_loader_checks_capacity_and_opens_only_needed_languages(id_tokens):
    with pytest.raises(ValueError):
        MixtureLoader(id_tokens, 1.0, 4, 8, 0, n_sequences=125)  # only 124 windows per language
    assert set(MixtureLoader(id_tokens, 0.0, 4, 8, 0, 8).data) == {"en"}
    assert set(MixtureLoader(id_tokens, 1.0, 4, 8, 0, 8).data) == {"tr"}


def test_compile_disabled_on_old_gpus(monkeypatch):
    assert use_compile(True, torch.device("cpu"))
    assert not use_compile(False, torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (6, 0))  # P100
    assert not use_compile(True, torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))  # T4
    assert use_compile(True, torch.device("cuda"))


def test_bpb_of_uniform_model(tokenizer):
    """A model with constant logits loses ln(V) nats on every token, so BPB is exactly computable."""
    vocab = tokenizer.get_vocab_size()

    class Uniform(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(*x.shape, vocab)

    byte_lengths = token_byte_lengths(tokenizer)
    tokens = np.array(tokenizer.encode(" ".join(CORPUS)).ids, dtype=np.uint16)
    seq_len = 16
    targets = tokens[1:(len(tokens) - 1) // seq_len * seq_len + 1]
    n_bytes = byte_lengths[torch.from_numpy(targets.astype(np.int64))].sum().item()
    expected = len(targets) * math.log2(vocab) / n_bytes
    assert evaluate_bpb(Uniform(), tokens, byte_lengths, seq_len, batch_size=3) == pytest.approx(expected)


def test_lr_schedule(setup, tmp_path):
    cfg = smoke_config(setup, tmp_path)
    assert lr_at(0, cfg) == pytest.approx(cfg.train.lr / 2)
    assert lr_at(1, cfg) == pytest.approx(cfg.train.lr)
    assert lr_at(cfg.total_steps, cfg) == pytest.approx(cfg.train.min_lr)


def test_parse_overrides():
    assert parse_overrides(["train.lr=3e-4", "hub_repo=a/b", "train.compile=false"]) == {
        "train": {"lr": 3e-4, "compile": False},
        "hub_repo": "a/b",
    }


def read_log(run_dir):
    with open(run_dir / "log.csv") as f:
        return [{k: v for k, v in row.items() if k not in ("tokens_per_sec", "elapsed_min")} for row in csv.DictReader(f)]


def test_resume_matches_uninterrupted_run(setup, tmp_path, monkeypatch):
    cpu = torch.device("cpu")
    train(smoke_config(setup, tmp_path / "a"), cpu)

    # Crash at the start of step 3; checkpoints are saved after every step.
    real_lr_at = train_mod.lr_at

    def crash_at_step_3(step, cfg):
        if step == 3:
            raise KeyboardInterrupt
        return real_lr_at(step, cfg)

    monkeypatch.setattr(train_mod, "lr_at", crash_at_step_3)
    with pytest.raises(KeyboardInterrupt):
        train(smoke_config(setup, tmp_path / "b"), cpu)
    monkeypatch.setattr(train_mod, "lr_at", real_lr_at)
    train(smoke_config(setup, tmp_path / "b"), cpu)

    a, b = tmp_path / "a" / "smoke", tmp_path / "b" / "smoke"
    weights_a = torch.load(a / "final.pt")["model"]
    weights_b = torch.load(b / "final.pt")["model"]
    assert all(torch.equal(weights_a[k], weights_b[k]) for k in weights_a)
    assert read_log(a) == read_log(b)
    assert len(read_log(a)) == 3
