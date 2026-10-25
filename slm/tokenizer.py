"""Shared Turkish-English byte-level BPE tokenizer.

    python -m slm.tokenizer --text-dir data/text --mb-per-lang 300 --out assets/tokenizer.json
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from slm.data.prepare import MB, read_texts, take_bytes
from slm.special_tokens import SPECIAL_TOKENS


# Letter runs keep their leading space; digits are split one by one; punctuation is grouped.
# Unlike GPT-2's pattern there are no English contraction rules ('s, 'd, ...), which would
# cut Turkish suffixes after an apostrophe: "İstanbul'da" -> "İstanbul" "'" "da", not "'d" "a".
SPLIT_PATTERN = r" ?[\p{L}\p{M}]+| ?\p{N}| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"


def new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterator[str], vocab_size: int) -> Tokenizer:
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes: no unknown tokens
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def balanced_texts(text_dir: Path, langs: list[str], bytes_per_lang: int) -> Iterator[str]:
    """Equal number of UTF-8 bytes from each language's train split."""
    for lang in langs:
        yield from take_bytes(read_texts(text_dir / lang, "train"), bytes_per_lang)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-dir", type=Path, default=Path("data/text"))
    p.add_argument("--langs", nargs="+", default=["tr", "en"])
    p.add_argument("--mb-per-lang", type=float, default=300)
    p.add_argument("--vocab-size", type=int, default=32_768)
    p.add_argument("--out", type=Path, default=Path("assets/tokenizer.json"))
    args = p.parse_args()

    texts = balanced_texts(args.text_dir, args.langs, int(args.mb_per_lang * MB))
    tok = train_tokenizer(texts, args.vocab_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out))
    print(f"saved {args.out} (vocab {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
