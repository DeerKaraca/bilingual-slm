"""Tokenizer fertility per language on the validation split, compared with GPT-2's tokenizer.

    python -m slm.eval.fertility --text-dir data/text --tokenizer assets/tokenizer.json --out reports/tokenizer.md
"""

import argparse
from collections.abc import Iterable
from itertools import batched
from pathlib import Path

from tokenizers import Tokenizer

from slm.data.prepare import read_texts


def measure(tok: Tokenizer, texts: Iterable[str]) -> dict:
    """Tokens per whitespace-separated word (lower is better) and UTF-8 bytes per token (higher is better)."""
    n_tokens = n_words = n_bytes = 0
    for batch in batched(texts, 1024):
        n_tokens += sum(len(e.ids) for e in tok.encode_batch_fast(list(batch)))
        n_words += sum(len(t.split()) for t in batch)
        n_bytes += sum(len(t.encode()) for t in batch)
    return {"tokens_per_word": n_tokens / n_words, "bytes_per_token": n_bytes / n_tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-dir", type=Path, default=Path("data/text"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--langs", nargs="+", default=["tr", "en"])
    p.add_argument("--out", type=Path, default=Path("reports/tokenizer.md"))
    args = p.parse_args()

    ours = Tokenizer.from_file(str(args.tokenizer))
    tokenizers = {
        f"ours ({ours.get_vocab_size() // 1024}k)": ours,
        "GPT-2 (50k)": Tokenizer.from_pretrained("gpt2"),
    }
    lines = [
        "| Tokenizer | Language | Tokens / word ↓ | Bytes / token ↑ |",
        "|---|---|---|---|",
    ]
    for name, tok in tokenizers.items():
        for lang in args.langs:
            m = measure(tok, read_texts(args.text_dir / lang, "val"))
            lines.append(f"| {name} | {lang} | {m['tokens_per_word']:.2f} | {m['bytes_per_token']:.2f} |")

    table = "\n".join(lines)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# Tokenizer fertility (validation split)\n\n{table}\n")


if __name__ == "__main__":
    main()
