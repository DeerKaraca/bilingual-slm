import numpy as np
import pytest

from slm.data.loader import MixtureLoader
from slm.eval.overlap import contaminated_documents, overlap

SEQ, N = 8, 4


@pytest.fixture
def tokens(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "tr").mkdir()
    rng.integers(0, 500, 1001).astype(np.uint16).tofile(tmp_path / "tr" / "train.bin")
    return tmp_path


def write_val(tokens, parts):
    np.concatenate(parts).astype(np.uint16).tofile(tokens / "tr" / "val.bin")


def window_tokens(tokens, index):
    train = np.fromfile(tokens / "tr" / "train.bin", dtype=np.uint16)
    return train[index * SEQ:index * SEQ + SEQ + 1]


def test_loader_never_reads_validation(tokens):
    write_val(tokens, [np.full(1000, 60_000)])  # a token value that never occurs in train.bin
    loader = MixtureLoader(tokens, 1.0, batch_size=25, seq_len=SEQ, seed=0, n_sequences=500, unique_sequences=50)
    for _ in range(20):  # 10 passes over the capped pool
        x, y = loader.next_batch()
        assert (x != 60_000).all() and (y != 60_000).all()


def test_contaminated_documents(tmp_path):
    (tmp_path / "tr").mkdir()
    # Two training windows (seq_len 8). The first holds "5 6 | 0 7 8": an n-gram across a document boundary.
    np.array([5, 6, 0, 7, 8, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 37], dtype=np.uint16).tofile(
        tmp_path / "tr" / "train.bin")
    val = [0, 1, 2, 3, 5, 6,  # ends with "5 6" ...
           0, 7, 8, 9, 4,  # ... and the next starts with "0 7 8": the shared 4-grams span the boundary
           0, 30, 31, 32, 33, 99]  # shares "30 31 32 33" with the second window
    np.array(val, dtype=np.uint16).tofile(tmp_path / "tr" / "val.bin")
    starts, contaminated = contaminated_documents(tmp_path, "tr", eot=0, seeds=[0], unique_sequences=2, seq_len=8, n=4)
    assert starts.tolist() == [0, 6, 11]
    assert contaminated.tolist() == [False, False, True]


def test_overlap_counts_ngrams_from_the_pool_only(tokens):
    order = np.random.default_rng(0).permutation(1000 // SEQ)
    inside, outside = order[0], order[-1]  # a window in a pool of 10, and one that is not
    fresh = np.arange(1000, 1020)  # never in train.bin

    write_val(tokens, [window_tokens(tokens, inside), fresh])
    # 9 copied tokens hold 6 of the 26 validation 4-grams; n-grams spanning into `fresh` cannot match.
    assert overlap(tokens, "tr", seed=0, unique_sequences=10, seq_len=SEQ, n=N)["overlap_fraction"] == pytest.approx(6 / 26)

    write_val(tokens, [window_tokens(tokens, outside), fresh])
    assert overlap(tokens, "tr", seed=0, unique_sequences=10, seq_len=SEQ, n=N)["overlap_fraction"] == 0
