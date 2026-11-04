"""Batches for the language mixture.

Each language's token file is cut into non-overlapping windows of seq_len + 1 tokens (stride
seq_len, so inputs never overlap), shuffled once with a fixed seed and consumed in order.
By default every run sees each window at most once, whatever its mixture ratio, and runs with
the same seed read the same windows in the same order (tr010's Turkish is a prefix of tr100's).

With `unique_sequences` set, only the first N windows of that order are used and the run cycles
through them: the first pass is in the same order as without the cap, every later pass is
reshuffled. This repeats a fixed amount of data for a compute-matched control.
Only train.bin is ever read; validation lives in separate val.bin files.
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

    def __init__(self, tokens_dir, tr_fraction: float, batch_size: int, seq_len: int, seed: int, n_sequences: int,
                 unique_sequences: int | None = None):
        self.tr_fraction = tr_fraction
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.consumed = 0
        needed = {"tr": tr_count(n_sequences, tr_fraction)}
        needed["en"] = n_sequences - needed["tr"]
        self.data, self.order, self.pool = {}, {}, {}
        self.passes: dict[str, tuple[int, np.ndarray]] = {}  # lang -> (pass number, its window order)
        for lang, n in needed.items():
            if n == 0:
                continue  # e.g. tr100 runs without English files
            self.data[lang] = load_tokens(tokens_dir, lang, "train")
            available = (len(self.data[lang]) - 1) // seq_len
            self.pool[lang] = min(unique_sequences or available, available)
            if unique_sequences and unique_sequences > available:
                raise ValueError(f"{lang}: {unique_sequences} unique sequences requested, only {available} exist")
            if unique_sequences is None and n > available:
                raise ValueError(f"{lang}: run needs {n} sequences but only {available} are available")
            self.order[lang] = np.random.default_rng(seed).permutation(available)

    def window(self, lang: str, k: int) -> int:
        """Index of the k-th window of `lang` this run reads."""
        pass_number, j = divmod(k, self.pool[lang])
        if pass_number == 0:
            return int(self.order[lang][j])
        if self.passes.get(lang, (None,))[0] != pass_number:
            rng = np.random.default_rng([self.seed, pass_number])
            self.passes[lang] = (pass_number, rng.permutation(self.order[lang][:self.pool[lang]]))
        return int(self.passes[lang][1][j])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and next-token targets, both (B, T) int64."""
        rows = []
        for i in range(self.consumed, self.consumed + self.batch_size):
            n_tr = tr_count(i, self.tr_fraction)
            lang, k = ("tr", n_tr) if tr_count(i + 1, self.tr_fraction) > n_tr else ("en", i - n_tr)
            start = self.window(lang, k) * self.seq_len
            rows.append(self.data[lang][start:start + self.seq_len + 1])
        self.consumed += self.batch_size
        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict:
        return {"consumed": self.consumed}

    def load_state_dict(self, state: dict):
        self.consumed = state["consumed"]
