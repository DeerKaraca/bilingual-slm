import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from loadtest import make_jobs, percentile, summarize  # noqa: E402


def test_nearest_rank_percentile():
    values = list(range(1, 21))  # 20 samples
    assert percentile(values, 0.5) == 10
    assert percentile(values, 0.95) == 19
    assert percentile([7.0], 0.95) == 7.0


def test_summarize_pools_latencies_and_takes_median_throughput():
    def run(wall, latencies):
        return {"wall": wall, "requests": [{"ttft": 0.1, "latency": x, "itl": 0.01, "tokens": 10} for x in latencies]}

    runs = [run(1.0, [1, 2]), run(2.0, [3, 4]), run(4.0, [5, 100])]
    s = summarize(runs)
    assert s["n_requests"] == 6
    assert s["tokens_per_s"] == 10  # per-run throughputs 20, 10, 5 -> median 10
    assert s["latency_p95_s"] == 100 and s["latency_p50_s"] == 3


def test_jobs_are_reproducible_and_mixed():
    jobs = make_jobs(16)
    assert jobs == make_jobs(16)
    assert len({n for _, n in jobs}) > 1 and len({len(p) for p, _ in jobs}) > 1
