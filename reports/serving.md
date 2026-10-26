# Serving benchmark: HuggingFaceTB/SmolLM2-135M-Instruct

CPU (AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx), server pinned to 3 torch threads, client on the same machine; quant=none, max batch 8. Greedy decoding with ignore_eos, so every request generates exactly max_tokens. 4 rounds with a rotated mode order: throughput is the median over rounds, latencies pool the requests of all rounds.

## KV cache: one request at a time

| Output tokens | Inter-token latency, no cache (ms) | with KV cache (ms) | Request latency, no cache (s) | with KV cache (s) | Speed-up |
|---|---|---|---|---|---|
| 16 | 456 | 72 | 7.2 | 1.5 | 4.8x |
| 32 | 500 | 74 | 16.0 | 2.7 | 5.9x |
| 64 | 554 | 72 | 35.3 | 5.0 | 7.0x |
| 128 | 687 | 73 | 87.7 | 9.6 | 9.1x |

## Batching: 16 requests with mixed prompt and output lengths

Latency p95 is a nearest-rank estimate from 64 requests per row (16 per round x 4 rounds), so it is approximate.

| Mode | Concurrent clients | Throughput (tok/s) | TTFT p50 (s) | Inter-token p50 (ms) | Latency p50 (s) | Latency p95 (s) |
|---|---|---|---|---|---|---|
| sequential | 1 | 9.9 | 0.79 | 75 | 4.9 | 15.3 |
| sequential | 4 | 9.9 | 15.20 | 85 | 19.3 | 47.2 |
| sequential | 8 | 9.8 | 33.20 | 84 | 38.9 | 70.7 |
| static | 1 | 7.8 | 0.85 | 111 | 4.7 | 15.5 |
| static | 4 | 12.4 | 4.03 | 204 | 13.3 | 25.3 |
| static | 8 | 15.3 | 7.03 | 265 | 20.7 | 33.1 |
| continuous | 1 | 9.3 | 0.82 | 110 | 4.9 | 15.5 |
| continuous | 4 | 16.9 | 1.49 | 195 | 11.2 | 32.8 |
| continuous | 8 | 21.5 | 2.17 | 245 | 15.5 | 45.8 |
