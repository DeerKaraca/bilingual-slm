"""Random-window batches from the per-language token files, mixed by tr_fraction."""

from pathlib import Path

import numpy as np
import torch


def load_tokens(tokens_dir: str | Path, lang: str, split: str) -> np.memmap:
    return np.memmap(Path(tokens_dir) / lang / f"{split}.bin", dtype=np.uint16, mode="r")


class MixtureLoader:
    """Each sequence comes from one language: Turkish with probability tr_fraction, else English.

    Windows start at uniformly random offsets, so the only state needed to resume is the RNG.
    """

    def __init__(self, tokens_dir, tr_fraction: float, batch_size: int, seq_len: int, seed: int):
        self.tr_fraction = tr_fraction
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        # Only open what this mix needs, so e.g. tr100 runs without English files.
        self.data = {
            lang: load_tokens(tokens_dir, lang, "train")
            for lang, share in [("tr", tr_fraction), ("en", 1 - tr_fraction)]
            if share > 0
        }

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and next-token targets, both (B, T) int64."""
        is_tr = self.rng.random(self.batch_size) < self.tr_fraction
        rows = []
        for tr in is_tr:
            data = self.data["tr" if tr else "en"]
            start = self.rng.integers(0, len(data) - self.seq_len - 1)
            rows.append(data[start:start + self.seq_len + 1])
        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict):
        self.rng.bit_generator.state = state["rng"]
