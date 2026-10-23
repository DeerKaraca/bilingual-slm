"""Stream raw corpora from the HF Hub, clean them and save them as parquet text shards.

    python -m slm.data.prepare --langs tr en --hub-repo USER/bilingual-slm-data

Output per language: val-00000.parquet, train-00000.parquet, ... and stats.json.
Validation documents are taken first and share the dedup set with training,
so no exact duplicate can leak from train into val.
"""

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from slm.data.clean import doc_hash, is_good, normalize

# Hub globs of the parquet files for each language's source.
SOURCES = {
    "tr": "datasets/epfml/FineWeb2-HQ/tur_Latn/*.parquet",
    "en": "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet",
}
MB = 1024 * 1024


def stream_docs(lang: str, seed: int, stats: Counter) -> Iterator[str]:
    """Yield cleaned, exact-deduplicated documents of one language."""
    fs = HfFileSystem()
    files = sorted(fs.glob(SOURCES[lang]))
    # Files are ordered by crawl date; shuffle so we don't only read the oldest dumps.
    random.Random(seed).shuffle(files)
    seen = set()
    for path in files:
        with fs.open(path) as f:
            # Reading only the "text" column skips the large embedding columns in FineWeb2-HQ.
            for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=1024):
                for text in batch.column("text").to_pylist():
                    text = normalize(text)
                    h = doc_hash(text)
                    if not is_good(text):
                        stats["dropped_filter"] += 1
                    elif h in seen:
                        stats["dropped_duplicate"] += 1
                    else:
                        seen.add(h)
                        stats["kept"] += 1
                        yield text


def take_bytes(docs: Iterator[str], target_bytes: int) -> Iterator[str]:
    """Yield documents until their UTF-8 size reaches target_bytes; the source iterator stays usable."""
    total = 0
    for text in docs:
        yield text
        total += len(text.encode())
        if total >= target_bytes:
            return


def write_shards(texts: Iterable[str], out_dir: Path, prefix: str, shard_bytes: int) -> dict:
    """Write texts to parquet files of roughly shard_bytes each; return counts."""
    buf, buf_bytes, n_docs, n_bytes, n_shards = [], 0, 0, 0, 0

    def flush():
        nonlocal buf, buf_bytes, n_shards
        table = pa.table({"text": buf})
        pq.write_table(table, out_dir / f"{prefix}-{n_shards:05d}.parquet", compression="zstd")
        buf, buf_bytes, n_shards = [], 0, n_shards + 1

    for text in texts:
        size = len(text.encode())
        buf.append(text)
        buf_bytes += size
        n_docs += 1
        n_bytes += size
        if buf_bytes >= shard_bytes:
            flush()
    if buf:
        flush()
    return {"docs": n_docs, "bytes": n_bytes, "shards": n_shards}


def prepare_language(lang: str, out_dir: Path, val_bytes: int, train_bytes: int, shard_bytes: int, seed: int):
    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    docs = stream_docs(lang, seed, stats)
    info = {
        "val": write_shards(take_bytes(docs, val_bytes), lang_dir, "val", shard_bytes),
        "train": write_shards(take_bytes(docs, train_bytes), lang_dir, "train", shard_bytes),
        "filter": dict(stats),
    }
    (lang_dir / "stats.json").write_text(json.dumps(info, indent=2))
    print(lang, json.dumps(info))
    return lang_dir


def upload(lang_dir: Path, repo: str):
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=lang_dir, path_in_repo=f"text/{lang_dir.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["tr", "en"], choices=list(SOURCES))
    p.add_argument("--out", type=Path, default=Path("data/text"))
    p.add_argument("--val-mb", type=float, default=10)  # ~2.5M tokens
    p.add_argument("--train-mb", type=float, default=3200)  # > 650M tokens, so no run repeats data
    p.add_argument("--shard-mb", type=float, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hub-repo", help="HF dataset repo to upload to (e.g. USER/bilingual-slm-data)")
    args = p.parse_args()

    for lang in args.langs:
        lang_dir = prepare_language(
            lang, args.out, int(args.val_mb * MB), int(args.train_mb * MB), int(args.shard_mb * MB), args.seed
        )
        if args.hub_repo:
            upload(lang_dir, args.hub_repo)


if __name__ == "__main__":
    main()
