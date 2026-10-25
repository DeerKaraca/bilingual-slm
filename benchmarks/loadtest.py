"""Load test for the inference server: KV cache on/off latency and batching throughput.

    python benchmarks/loadtest.py                       # starts a local server per mode, writes reports/serving.*
    python benchmarks/loadtest.py --url http://localhost:8000 --concurrency 1 4 8   # an already running server

Every request streams with temperature 0 and ignore_eos, so it generates exactly max_tokens.
Server and client share one laptop CPU whose speed drifts as it heats up, so configurations are
measured in interleaved rounds and the median over rounds is reported.
"""

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

SENTENCE = ("The history of science is the study of how people came to understand the natural world, "
            "from early astronomy and medicine to modern physics, chemistry and biology. ")


def make_jobs(n: int, seed: int = 0) -> list[tuple[str, int]]:
    """n (prompt, max_tokens) pairs with mixed lengths: the case where continuous batching matters."""
    rng = random.Random(seed)
    return [(SENTENCE * rng.randint(1, 6), rng.choice([16, 32, 64, 128])) for _ in range(n)]


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int) -> dict:
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True}
    start, ttft = time.perf_counter(), None
    async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if ttft is None and line.startswith("data: {"):
                ttft = time.perf_counter() - start
    latency = time.perf_counter() - start
    return {"ttft": ttft, "latency": latency, "itl": (latency - ttft) / max(max_tokens - 1, 1), "tokens": max_tokens}


async def closed_loop(url: str, concurrency: int, jobs: list[tuple[str, int]]) -> dict:
    """`concurrency` clients send the jobs back to back (each waits for its previous response)."""
    pending, results = list(jobs), []
    async with httpx.AsyncClient(timeout=None) as client:
        async def client_loop():
            while pending:
                results.append(await one_request(client, url, *pending.pop(0)))

        start = time.perf_counter()
        await asyncio.gather(*(client_loop() for _ in range(concurrency)))
        wall = time.perf_counter() - start
    latencies = sorted(r["latency"] for r in results)
    return {
        "tokens_per_s": sum(r["tokens"] for r in results) / wall,
        "ttft_p50_s": statistics.median(r["ttft"] for r in results),
        "itl_p50_ms": statistics.median(r["itl"] for r in results) * 1000,
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
    }


@contextmanager
def local_server(mode: str, args):
    cmd = [sys.executable, "-m", "slm.server", "--model", args.model, "--mode", mode, "--threads", str(args.threads),
           "--max-batch", str(args.max_batch), "--quant", args.quant, "--port", str(args.port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.time() + 300
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode}: {' '.join(cmd)}")
            try:
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.time() > deadline:
                raise TimeoutError("server did not start")
            time.sleep(1)
        asyncio.run(closed_loop(url, 1, [("warm up", 4)]))  # first calls pay one-off allocation costs
        yield url
    finally:
        proc.terminate()
        proc.wait()


def median_of(runs: list[dict]) -> dict:
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def run_local(args) -> dict:
    """Interleaved rounds: every round restarts each mode's server and runs its scenarios once."""
    latency_jobs = {n: [(SENTENCE * 2, n)] for n in args.output_lengths}
    throughput_jobs = make_jobs(args.requests)
    runs: dict[str, list[dict]] = {}
    for rnd in range(args.rounds):
        for mode in ("no_cache", "sequential", "static", "continuous"):
            with local_server(mode, args) as url:
                if mode in ("no_cache", "sequential"):  # KV cache off vs on, one request at a time
                    for n, jobs in latency_jobs.items():
                        runs.setdefault(f"latency|{mode}|{n}", []).append(asyncio.run(closed_loop(url, 1, jobs)))
                if mode != "no_cache":  # no_cache is far too slow for the throughput workload
                    for c in args.concurrency:
                        runs.setdefault(f"throughput|{mode}|{c}", []).append(
                            asyncio.run(closed_loop(url, c, throughput_jobs)))
            print(f"round {rnd + 1}/{args.rounds} {mode} done", flush=True)
    return {key: median_of(r) for key, r in runs.items()}


def to_markdown(meta: dict, results: dict) -> str:
    lines = [
        f"# Serving benchmark: {meta['model']}", "",
        f"CPU ({meta['cpu']}), server pinned to {meta['threads']} torch threads, client on the same machine; "
        f"quant={meta['quant']}, max batch {meta['max_batch']}. Greedy decoding with ignore_eos, so every request "
        f"generates exactly max_tokens. Values are medians over {meta['rounds']} interleaved rounds.", "",
        "## KV cache: one request at a time", "",
        "| Output tokens | Inter-token latency, no cache (ms) | with KV cache (ms) | Request latency, no cache (s) | with KV cache (s) | Speed-up |",
        "|---|---|---|---|---|---|",
    ]
    for n in meta["output_lengths"]:
        off, on = results[f"latency|no_cache|{n}"], results[f"latency|sequential|{n}"]
        lines.append(f"| {n} | {off['itl_p50_ms']:.0f} | {on['itl_p50_ms']:.0f} | {off['latency_p50_s']:.1f} | "
                     f"{on['latency_p50_s']:.1f} | {off['latency_p50_s'] / on['latency_p50_s']:.1f}x |")
    lines += [
        "", f"## Batching: {meta['requests']} requests with mixed prompt and output lengths", "",
        "| Mode | Concurrent clients | Throughput (tok/s) | TTFT p50 (s) | Inter-token p50 (ms) | Latency p50 (s) | Latency p95 (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in ("sequential", "static", "continuous"):
        for c in meta["concurrency"]:
            r = results[f"throughput|{mode}|{c}"]
            lines.append(f"| {mode} | {c} | {r['tokens_per_s']:.1f} | {r['ttft_p50_s']:.2f} | {r['itl_p50_ms']:.0f} | "
                         f"{r['latency_p50_s']:.1f} | {r['latency_p95_s']:.1f} |")
    return "\n".join(lines) + "\n"


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="test an already running server (throughput scenario only)")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--quant", default="none")
    p.add_argument("--threads", type=int, default=3, help="server torch threads; 4-core CPU: leave one core for the client")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--output-lengths", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("reports/serving"))
    args = p.parse_args()

    if args.url:
        for c in args.concurrency:
            print(json.dumps({"concurrency": c, **asyncio.run(closed_loop(args.url, c, make_jobs(args.requests)))}))
        return

    meta = {k: getattr(args, k) for k in ("model", "quant", "threads", "max_batch", "concurrency", "requests",
                                          "output_lengths", "rounds")}
    meta["cpu"] = cpu_name()
    results = run_local(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    args.out.with_suffix(".md").write_text(to_markdown(meta, results))
    print(to_markdown(meta, results))


if __name__ == "__main__":
    main()
