# Research results

Analysis as pre-registered in [analysis_plan.md](../analysis_plan.md).

Complete runs: tr000, tr010, tr030, tr030_seed1, tr050, tr100, tr_only_control.

| Run | Turkish share | Turkish tokens | English tokens | TR BPB | EN BPB |
|---|---|---|---|---|---|
| tr000 | 0% | 0M | 600M | 2.7328 | 1.1184 |
| tr010 | 10% | 60M | 540M | 1.3159 | 1.1252 |
| tr030 | 30% | 180M | 420M | 1.2158 | 1.1465 |
| tr030_seed1 | 30% | 180M | 420M | 1.2170 | 1.1472 |
| tr050 | 50% | 300M | 300M | 1.1768 | 1.1793 |
| tr100 | 100% | 600M | 0M | 1.1200 | 1.7456 |
| tr_only_control | 100% | 180M | 0M | 1.2780 | 1.9698 |

**Seed noise** (|tr030 - tr030_seed1|): TR 0.0012, EN 0.0007 BPB.

**Q2, transfer from English:** control minus tr030 = +0.0616 TR BPB (+4.8% of the control) -> English helps Turkish (positive transfer).

**Q3, interference (indicative; monolingual curve is not schedule-matched):**

| Run | Language | Tokens of it seen | Final BPB | Monolingual curve | Gap | Reading |
|---|---|---|---|---|---|---|
| tr010 | TR | 60M | 1.3159 | 1.6290 | -0.3131 | below the monolingual curve (curve not schedule-matched) |
| tr030 | TR | 180M | 1.2158 | 1.2618 | -0.0460 | below the monolingual curve (curve not schedule-matched) |
| tr050 | TR | 300M | 1.1768 | 1.1871 | -0.0103 | below the monolingual curve (curve not schedule-matched) |
| tr010 | EN | 540M | 1.1252 | 1.1237 | +0.0015 | interference likely |
| tr030 | EN | 420M | 1.1465 | 1.1467 | -0.0002 | inconclusive at this noise level |
| tr050 | EN | 300M | 1.1793 | 1.1886 | -0.0093 | below the monolingual curve (curve not schedule-matched) |
