## Activation outliers (max |x| / median |x| at linear inputs)

| Layer | median over layers | worst layer |
|---|---|---|
| qkv | 14 | 17 |
| out | 30 | 70 |
| gate_up | 13 | 16 |
| down | 198 | 805 |

## quantize_dynamic variants (en, float32 BPB 1.2822)

| Setup | Eval batch | BPB | Top-1 agree |
|---|---|---|---|
| per-tensor weights (default) | 1 | 1.3123 | 83.8% |
| per-tensor weights (default) | 4 | 1.3257 | 79.6% |
| per-channel weights | 2 | 1.3144 | 82.2% |
| per-tensor, mlp.down kept in float | 2 | 1.2844 | 96.0% |
