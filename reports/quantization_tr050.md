# Quantization benchmark: export/bilingual-slm-42m-tr050

CPU, 4 threads; speed = median of 5 rounds with a rotated variant order; quality on 16,384 validation tokens per language in windows of 512. Top-1 agreement = share of positions where the next-token argmax matches float32.

| Variant | Weights (MB) | Prefill 256 tok (ms) | Decode tok/s B=1 | Decode tok/s B=8 | TR BPB | EN BPB | TR top-1 agree | EN top-1 agree |
|---|---|---|---|---|---|---|---|---|
| none | 162 | 468 | 30.6 | 158.8 | 1.1629 | 1.2501 | 100.0% | 100.0% |
| int8 | 89 | 432 | 25.0 | 107.4 | 1.1629 | 1.2501 | 99.0% | 99.2% |
| int8-emb | 41 | 511 | 13.0 | 74.4 | 1.1629 | 1.2502 | 98.7% | 99.0% |
| dynamic | 89 | 369 | 42.4 | 178.3 | 1.1965 | 1.2831 | 78.8% | 81.6% |
| dynamic-head | 105 | 295 | 64.9 | 305.2 | 1.1969 | 1.2833 | 78.6% | 81.6% |
