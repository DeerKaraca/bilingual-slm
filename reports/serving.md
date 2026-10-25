# Serving benchmark: HuggingFaceTB/SmolLM2-135M-Instruct

CPU (AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx), server pinned to 3 torch threads, client on the same machine; quant=none, max batch 8. Greedy decoding with ignore_eos, so every request generates exactly max_tokens. Values are medians over 4 rounds with a rotated mode order.

## KV cache: one request at a time

| Output tokens | Inter-token latency, no cache (ms) | with KV cache (ms) | Request latency, no cache (s) | with KV cache (s) | Speed-up |
|---|---|---|---|---|---|
| 16 | 571 | 85 | 9.1 | 1.8 | 5.1x |
| 32 | 565 | 85 | 18.0 | 3.1 | 5.9x |
| 64 | 650 | 83 | 41.4 | 5.7 | 7.3x |

## Batching: 16 requests with mixed prompt and output lengths

| Mode | Concurrent clients | Throughput (tok/s) | TTFT p50 (s) | Inter-token p50 (ms) | Latency p50 (s) | Latency p95 (s) |
|---|---|---|---|---|---|---|
| sequential | 1 | 10.1 | 0.80 | 85 | 4.8 | 11.9 |
| sequential | 4 | 9.6 | 14.70 | 95 | 18.3 | 35.2 |
| sequential | 8 | 9.6 | 36.85 | 90 | 41.2 | 53.1 |
| static | 1 | 8.5 | 0.86 | 98 | 5.6 | 14.2 |
| static | 4 | 13.8 | 3.50 | 194 | 14.9 | 22.8 |
| static | 8 | 17.3 | 6.09 | 280 | 20.0 | 28.1 |
| continuous | 1 | 9.0 | 0.84 | 93 | 5.0 | 14.1 |
| continuous | 4 | 16.8 | 1.30 | 189 | 10.4 | 28.2 |
| continuous | 8 | 21.1 | 2.09 | 260 | 13.9 | 37.4 |
