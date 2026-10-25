# Analysis plan (fixed before any final results)

Written while the first training session was still running, before any run had finished.
Its git commit date is the record that metrics, comparisons and decision rules were chosen
before seeing results. Anything not listed here is reported as exploratory.

## Runs

| Run | Turkish share | Total tokens | Turkish tokens | English tokens |
|---|---|---|---|---|
| tr000 | 0% | 600M | 0 | 600M |
| tr010 | 10% | 600M | 60M | 540M |
| tr030 | 30% | 600M | 180M | 420M |
| tr030_seed1 | 30% (seed 1) | 600M | 180M | 420M |
| tr050 | 50% | 600M | 300M | 300M |
| tr100 | 100% | 600M | 600M | 0 |
| tr_only_control | 100% | 180M | 180M | 0 |

Same tokenizer, architecture, batch size, optimizer and training code (tag `train-v1`) for all runs;
every run has its own warmup + cosine schedule over its own budget. Exact token counts are
step-quantized (tr030: 175,718 Turkish sequences, control: 175,616; 0.06% apart).

## Metric

Validation bits-per-byte on the full held-out Turkish (10.5 MB) and English (10.5 MB) sets,
evaluated every 100 steps. **Final** = the evaluation at the last step, i.e. after the learning
rate has fully decayed. No smoothing.

## Questions, comparisons and decision rules

**Noise.** n = 2 seeds exist only for tr030, so run-to-run noise is estimated as
`noise_L = |BPB_L(tr030) - BPB_L(tr030_seed1)|` at the final step, per language L. With two
seeds this is a range, not a standard deviation; it is reported as error bars on tr030
(min-max of the two seeds) and used as the yardstick below. Where tr030 enters a difference,
the mean of the two seeds is used.

**Q1. How does the Turkish share affect each language?** Final TR and EN BPB plotted against
the Turkish share (0, 0.1, 0.3, 0.5, 1). Descriptive; no test.

**Q2. How much does English transfer to Turkish? (primary question)**
`transfer = BPB_TR(tr_only_control) - BPB_TR(tr030)` at the final step. Both models saw the same
Turkish tokens with fully decayed schedules; tr030 additionally saw 420M English tokens.
- transfer > 2 x noise_TR: English **helps** Turkish (positive transfer).
- transfer < -2 x noise_TR: English **hurts** Turkish (interference).
- otherwise: **inconclusive** at this noise level.
Also reported relative to the control: `transfer / BPB_TR(control)`.

**Q3. When does one language start to hurt the other?** Final BPB of language L plotted against
the tokens of L each run saw, together with the training curve of the monolingual run of L
(tr100 for Turkish, tr000 for English). Curve points before the end of a run are **not**
schedule-matched (their learning rate has not decayed), so a monolingual curve is biased upwards
and this comparison is indicative only; the only schedule-matched comparison is Q2.
A mixture whose final point lies above the monolingual curve at the same tokens of L is
labelled "interference likely" only if the gap exceeds 2 x noise_L.

**Limitation stated in advance.** There is no English-only control at matched English tokens,
so for English the effect of seeing fewer English tokens cannot be separated from interference
by Turkish; Q3 for English stays descriptive.

## Figures

1. Training curves: TR BPB and EN BPB of every mixture, x = total tokens and x = tokens seen in
   the evaluated language.
2. Final TR and EN BPB vs Turkish share, with tr030 seed range as error bars.
3. TR BPB of tr030 (both seeds) and tr_only_control vs Turkish tokens seen, with the final
   transfer gap marked.
4. Seed noise: TR and EN BPB of tr030 and tr030_seed1 over training, and their difference.

Produced by `python -m slm.eval.analysis` from each run's `log.csv`.
