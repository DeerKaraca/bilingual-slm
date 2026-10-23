"""Batches for the language mixture, read without repetition.

Each language's token file is cut into non-overlapping windows of seq_len + 1 tokens (stride
seq_len, so inputs never overlap), shuffled once with a fixed seed and consumed in order.
Every run therefore sees each window at most once, whatever its mixture ratio, and runs with
the same seed read the same windows in the same order (tr010's Turkish is a prefix of tr100's).
"""

import math
from pathlib import Path

import numpy as np
import torch


def load_tokens(tokens_dir: str | Path, lang: str, split: str) -> np.memmap:
    return np.memmap(Path(tokens_dir) / lang / f"{split}.bin", dtype=np.uint16, mode="r")


def tr_count(n: int, tr_fraction: float) -> int:
    """Number of Turkish sequences among the first n. Spreads them evenly with an exact ratio."""
    return math.floor(n * tr_fraction)


class MixtureLoader:
    """Sequence i is Turkish iff tr_count(i + 1) > tr_count(i); the only resume state is how
    many sequences have been read."""

    def __init__(self, tokens_dir, tr_fraction: float, batch_size: int, seq_len: int, seed: int, n_sequences: int):
        self.tr_fraction = tr_fraction
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.consumed = 0
        needed = {"tr": tr_count(n_sequences, tr_fraction)}
        needed["en"] = n_sequences - needed["tr"]
        self.data, self.order = {}, {}
        for lang, n in needed.items():
            if n == 0:
                continue  # e.g. tr100 runs without English files
            self.data[lang] = load_tokens(tokens_dir, lang, "train")
            available = (len(self.data[lang]) - 1) // seq_len
            if n > available:
                raise ValueError(f"{lang}: run needs {n} sequences but only {available} are available")
            self.order[lang] = np.random.default_rng(seed).permutation(available)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and next-token targets, both (B, T) int64."""
        rows = []
        for i in range(self.consumed, self.consumed + self.batch_size):
            n_tr = tr_count(i, self.tr_fraction)
            lang, k = ("tr", n_tr) if tr_count(i + 1, self.tr_fraction) > n_tr else ("en", i - n_tr)
            start = self.order[lang][k] * self.seq_len
            rows.append(self.data[lang][start:start + self.seq_len + 1])
        self.consumed += self.batch_size
        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict:
        return {"consumed": self.consumed}

    def load_state_dict(self, state: dict):
        self.consumed = state["consumed"]
