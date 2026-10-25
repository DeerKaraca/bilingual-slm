# Quantization benchmark: HuggingFaceTB/SmolLM2-135M

CPU, 4 threads; speed = median of 5 interleaved rounds; quality on 16,384 validation tokens per language in windows of 512. Top-1 agreement = share of positions where the next-token argmax matches float32.

| Variant | Weights (MB) | Prefill 256 tok (ms) | Decode tok/s B=1 | Decode tok/s B=8 | TR BPB | EN BPB | TR top-1 agree | EN top-1 agree |
|---|---|---|---|---|---|---|---|---|
| none | 513 | 1262 | 10.7 | 45.3 | 2.1533 | 0.9144 | 100.0% | 100.0% |
| int8 | 210 | 1346 | 6.6 | 33.9 | 2.1564 | 0.9161 | 95.2% | 96.8% |
| int8-emb | 129 | 1422 | 6.1 | 32.0 | 2.1565 | 0.9165 | 94.8% | 96.1% |
| dynamic | 210 | 1004 | 17.5 | 70.9 | 3.1793 | 1.2270 | 36.0% | 50.1% |
| dynamic-head | 237 | 946 | 21.5 | 105.8 | 3.2557 | 1.2595 | 34.1% | 47.6% |
