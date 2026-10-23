import pyarrow.parquet as pq

from slm.data.clean import doc_hash, is_good, letter_ratio, normalize
from slm.data.prepare import take_bytes, write_shards

GOOD_TR = "Bugün hava çok güzel, parkta uzun bir yürüyüş yaptık ve kuşları izledik. " * 5


def test_normalize_composes_turkish_letters():
    decomposed = "ş"  # "s" + combining cedilla
    assert normalize(f"  {decomposed}  ") == "ş"
    assert doc_hash(normalize(decomposed)) == doc_hash("ş")


def test_filters():
    assert is_good(GOOD_TR)
    assert not is_good("çok kısa")
    assert not is_good("1234 5678 | 90.12 | " * 50)
    assert letter_ratio("ab 12") == 0.5


def test_take_bytes_keeps_source_iterator():
    docs = iter(["aaaa", "bbbb", "cccc", "dddd"])
    assert list(take_bytes(docs, 6)) == ["aaaa", "bbbb"]
    assert list(take_bytes(docs, 100)) == ["cccc", "dddd"]


def test_take_bytes_counts_utf8_bytes():
    # "ş" is 2 bytes in UTF-8, so 3 of them already exceed 5 bytes.
    assert list(take_bytes(iter(["ş"] * 10), 5)) == ["ş"] * 3


def test_write_shards_roundtrip(tmp_path):
    texts = [f"doc {i}" for i in range(10)]
    info = write_shards(texts, tmp_path, "train", shard_bytes=20)
    files = sorted(tmp_path.glob("train-*.parquet"))
    assert info == {"docs": 10, "bytes": sum(len(t) for t in texts), "shards": len(files)}
    assert len(files) > 1
    read_back = [t for f in files for t in pq.read_table(f)["text"].to_pylist()]
    assert read_back == texts
