## Activation outliers (max |x| / median |x| at linear inputs)

| Layer | median over layers | worst layer |
|---|---|---|
| qkv | 73 | 152 |
| out | 59 | 203 |
| gate_up | 31 | 241 |
| down | 210 | 40490 |

## quantize_dynamic variants (en, float32 BPB 0.9478)

| Setup | Eval batch | BPB | Top-1 agree |
|---|---|---|---|
| per-tensor weights (default) | 1 | 1.2089 | 55.4% |
| per-tensor weights (default) | 4 | 1.2657 | 53.5% |
| per-channel weights | 2 | 1.1947 | 57.9% |
| per-tensor, mlp.down kept in float | 2 | 1.0184 | 75.4% |
