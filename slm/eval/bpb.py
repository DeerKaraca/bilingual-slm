"""Validation bits-per-byte: a tokenizer-independent metric, comparable across languages.

BPB = (sum of cross-entropy in nats) / ln(2) / (UTF-8 bytes of the predicted tokens).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


def token_byte_lengths(tok: Tokenizer) -> torch.Tensor:
    """(V,) number of UTF-8 bytes each token stands for; 0 for special tokens.

    In byte-level BPE every character of a token's string encodes exactly one byte.
    """
    lengths = torch.zeros(tok.get_vocab_size(), dtype=torch.long)
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    for token, idx in tok.get_vocab().items():
        lengths[idx] = 0 if token in special else len(token)
    return lengths


@torch.no_grad()
def evaluate_bpb(model, tokens: np.ndarray, byte_lengths: torch.Tensor, seq_len: int, batch_size: int) -> float:
    """BPB over the whole split, cut into non-overlapping windows (the tail shorter than one window is skipped).

    Special-token targets (document separators) carry no bytes and are left out of the loss.
    The model has no dropout, so it is not switched to eval mode (which would recompile a compiled model).
    """
    device = next(model.parameters()).device
    n_windows = (len(tokens) - 1) // seq_len
    nats, n_bytes = 0.0, 0
    for first in range(0, n_windows, batch_size):
        idx = np.arange(first, min(first + batch_size, n_windows))[:, None] * seq_len + np.arange(seq_len + 1)
        batch = torch.from_numpy(tokens[idx].astype(np.int64)).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        y_bytes = byte_lengths.to(device)[y.flatten()]
        nats += loss[y_bytes > 0].sum().item()
        n_bytes += y_bytes.sum().item()
    return nats / math.log(2) / n_bytes
