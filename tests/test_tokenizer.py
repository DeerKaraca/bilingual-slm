import json

import numpy as np
import pytest

from slm.data.prepare import write_shards
from slm.data.tokenize import encode_split
from slm.eval.fertility import measure
from slm.tokenizer import EOT, SPECIAL_TOKENS, new_tokenizer, train_tokenizer

CORPUS = [
    "İstanbul'da hava bugün çok güzel, 2024 yılında 15 kez yağmur yağdı.",
    "Öğrenciler kütüphanede ödevlerini yapıyorlar; ışık ve ağaç gölgesi.",
    "The students were reading in the library while it rained 3 times.",
    "Machine learning models don't need much data to start learning.",
] * 20


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(iter(CORPUS), vocab_size=400)


def pieces(text):
    return [p for p, _ in new_tokenizer().pre_tokenizer.pre_tokenize_str(text)]


def test_pretokenizer_keeps_turkish_suffixes_and_splits_digits():
    # Pieces are shown in byte-level form: "Ġ" is a space, Turkish letters become 2 symbols.
    assert len(pieces("İstanbul'da")) == 3  # "İstanbul" "'" "da"
    assert pieces(" 2024") == ["Ġ2", "0", "2", "4"]


def test_special_tokens_come_first(tok):
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))


@pytest.mark.parametrize("text", [CORPUS[0], CORPUS[1], "Ünseen wörds & emoji 🙂 ŞĞÜ"])
def test_roundtrip_is_lossless(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_measure(tok):
    m = measure(tok, CORPUS[:4])
    assert m["tokens_per_word"] > 0 and m["bytes_per_token"] >= 1


def test_encode_split_writes_uint16_with_doc_separators(tok, tmp_path):
    text_dir, out_dir = tmp_path / "text", tmp_path / "tokens"
    text_dir.mkdir()
    out_dir.mkdir()
    write_shards(CORPUS[:3], text_dir, "val", shard_bytes=100)

    meta = encode_split(tok, text_dir, "val", out_dir)
    ids = np.fromfile(out_dir / "val.bin", dtype=np.uint16)
    eot = tok.token_to_id(EOT)

    assert meta == json.loads((out_dir / "val.json").read_text())
    assert meta["tokens"] == len(ids) and meta["docs"] == 3
    assert meta["bytes"] == sum(len(t.encode()) for t in CORPUS[:3])
    assert (ids == eot).sum() == 3 and ids[0] == eot
    docs = [tok.decode(d.tolist()) for d in np.split(ids, np.flatnonzero(ids == eot))[1:]]
    assert docs == [tok.decode(tok.encode(t).ids) for t in CORPUS[:3]]
