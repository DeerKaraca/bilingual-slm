# Quantization benchmark: HuggingFaceTB/SmolLM2-135M

CPU, 4 threads; speed = median of 5 rounds with a rotated variant order; quality on 16,384 validation tokens per language in windows of 512. Top-1 agreement = share of positions where the next-token argmax matches float32.

| Variant | Weights (MB) | Prefill 256 tok (ms) | Decode tok/s B=1 | Decode tok/s B=8 | TR BPB | EN BPB | TR top-1 agree | EN top-1 agree |
|---|---|---|---|---|---|---|---|---|
| none | 513 | 1322 | 11.5 | 49.6 | 2.1533 | 0.9144 | 100.0% | 100.0% |
| int8 | 210 | 1309 | 6.7 | 36.1 | 2.1564 | 0.9161 | 95.2% | 96.8% |
| int8-emb | 129 | 1312 | 6.3 | 33.5 | 2.1565 | 0.9165 | 94.8% | 96.1% |
| dynamic | 210 | 981 | 19.1 | 81.9 | 3.1793 | 1.2270 | 36.0% | 50.1% |
| dynamic-head | 237 | 923 | 23.3 | 118.3 | 3.2557 | 1.2595 | 34.1% | 47.6% |
