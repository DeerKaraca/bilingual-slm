"""Validation/train n-gram overlap: how much of the validation set also occurs in a run's training windows.

    python -m slm.eval.overlap --tokens-dir data/tokens --lang tr --seed 0 --unique-sequences 175718

Validation and training documents are disjoint by construction (taken separately, exact duplicates
removed across both), but web text has near-duplicates: the same article on several sites,
boilerplate. A model that sees its training windows several times could memorise those and score
an artificially low validation BPB. This reports the share of validation 13-grams (the n used for
contamination checks in the GPT-3 paper) that also appear in the training windows the run uses.
"""

import argparse
import json

import numpy as np

from slm.data.loader import load_tokens

PRIME = np.uint64(1_000_003)


def ngram_hashes(rows: np.ndarray, n: int) -> np.ndarray:
    """64-bit polynomial hashes (wrap-around arithmetic) of the n-grams inside each row of a 2-D
    token array, flattened. N-grams never cross row boundaries."""
    t = rows.astype(np.uint64)
    width = t.shape[1] - n + 1
    h = np.zeros((t.shape[0], width), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * PRIME + t[:, k:k + width]
    return h.ravel()


def overlap(tokens_dir, lang: str, seed: int, unique_sequences: int, seq_len: int = 1024, n: int = 13,
            chunk: int = 8192) -> dict:
    data = load_tokens(tokens_dir, lang, "train")
    # The windows MixtureLoader uses with this seed and cap: the first unique_sequences of its order.
    windows = np.random.default_rng(seed).permutation((len(data) - 1) // seq_len)[:unique_sequences]
    offsets = np.arange(seq_len + 1)
    train = np.unique(np.concatenate([  # chunked to bound memory
        ngram_hashes(data[(windows[i:i + chunk] * seq_len)[:, None] + offsets], n)
        for i in range(0, len(windows), chunk)
    ]))
    val = ngram_hashes(np.asarray(load_tokens(tokens_dir, lang, "val"))[None, :], n)
    return {"lang": lang, "seed": seed, "unique_sequences": unique_sequences, "n": n,
            "val_ngrams": int(len(val)), "overlap_fraction": float(np.isin(val, train).mean())}


def contaminated_documents(tokens_dir, lang: str, eot: int, seeds: list[int], unique_sequences: int,
                           seq_len: int = 1024, n: int = 13, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mark validation documents that share any n-gram with the training windows of any of `seeds`.

    Returns (doc_start, contaminated): the token offset of every document (each starts with `eot`)
    and a boolean per document. Memory stays small by keeping only the 2M validation hashes and
    streaming the ~180M training hashes past them in chunks.
    """
    val = np.asarray(load_tokens(tokens_dir, lang, "val"))
    doc_start = np.flatnonzero(val == eot)
    hashes = ngram_hashes(val[None, :], n)
    doc_of = np.searchsorted(doc_start, np.arange(len(hashes)), side="right") - 1
    # n-grams that cross into the next document belong to neither: never count them.
    within = np.searchsorted(doc_start, np.arange(len(hashes)) + n - 1, side="right") - 1 == doc_of
    unique = np.unique(hashes[within])
    found = np.zeros(len(unique), dtype=bool)

    data = load_tokens(tokens_dir, lang, "train")
    offsets = np.arange(seq_len + 1)
    for seed in seeds:
        windows = np.random.default_rng(seed).permutation((len(data) - 1) // seq_len)[:unique_sequences]
        for i in range(0, len(windows), chunk):
            train = ngram_hashes(data[(windows[i:i + chunk] * seq_len)[:, None] + offsets], n)
            pos = np.minimum(np.searchsorted(unique, train), len(unique) - 1)
            found[pos[unique[pos] == train]] = True

    hit = np.zeros(len(hashes), dtype=bool)
    hit[within] = found[np.searchsorted(unique, hashes[within])]
    contaminated = np.zeros(len(doc_start), dtype=bool)
    contaminated[np.unique(doc_of[hit])] = True
    return doc_start, contaminated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", default="data/tokens")
    p.add_argument("--lang", default="tr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unique-sequences", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n", type=int, default=13)
    args = p.parse_args()
    print(json.dumps(overlap(args.tokens_dir, args.lang, args.seed, args.unique_sequences, args.seq_len, args.n)))


if __name__ == "__main__":
    main()
