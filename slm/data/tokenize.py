"""Encode parquet text shards into flat uint16 token files for training.

    python -m slm.data.tokenize --text-dir data/text --tokenizer assets/tokenizer.json --out data/tokens

Output per language and split: {split}.bin (uint16 token ids, every document preceded by
<|endoftext|>) and {split}.json with token and UTF-8 byte counts, which BPB needs.
"""

import argparse
import json
from itertools import batched
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm.data.prepare import read_texts
from slm.tokenizer import EOT


def encode_split(tok: Tokenizer, lang_dir: Path, split: str, out_dir: Path) -> dict:
    eot = tok.token_to_id(EOT)
    assert tok.get_vocab_size() <= 2**16, "token ids must fit in uint16"
    n_tokens = n_bytes = n_docs = 0
    with open(out_dir / f"{split}.bin", "wb") as f:
        for texts in batched(read_texts(lang_dir, split), 4096):
            encodings = tok.encode_batch_fast(list(texts))  # parallel in Rust; no offsets needed
            ids = np.fromiter((i for enc in encodings for i in [eot, *enc.ids]), dtype=np.uint16)
            ids.tofile(f)
            n_tokens += len(ids)
            n_bytes += sum(len(t.encode()) for t in texts)
            n_docs += len(texts)
    meta = {"tokens": n_tokens, "bytes": n_bytes, "docs": n_docs}
    (out_dir / f"{split}.json").write_text(json.dumps(meta))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-dir", type=Path, default=Path("data/text"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("data/tokens"))
    p.add_argument("--langs", nargs="+", default=["tr", "en"])
    args = p.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    for lang in args.langs:
        out_dir = args.out / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ["val", "train"]:
            print(lang, split, encode_split(tok, args.text_dir / lang, split, out_dir))


if __name__ == "__main__":
    main()
