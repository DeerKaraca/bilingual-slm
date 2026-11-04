# Analysis plan addendum: compute-matched control (fixed before these runs)

**Written after** the seven runs of [analysis_plan.md](analysis_plan.md) were analysed
([research/results.md](research/results.md)) and **before** the two runs below were started. The
git commit date of this file is the record. It does not change the original plan or its verdicts.

## Why

The pre-registered comparison (Q2) found that tr030 beats the Turkish-only control by +0.062 TR BPB.
But tr030 took 2,288 optimizer steps and the control 686, so part of the gain could come from more
training rather than from English. This control matches tr030's compute exactly.

## Runs

| Run | Turkish data | Total tokens | Steps | Seed |
|---|---|---|---|---|
| tr_only_control_x3 | the same 175,718 Turkish windows tr030 saw, repeated ~3.33 times | 600M | 2,288 | 0 |
| tr_only_control_x3_seed1 | the same 175,718 windows tr030_seed1 saw, repeated ~3.33 times | 600M | 2,288 | 1 |

Training code: tag `train-v2`, which only adds a cap on distinct windows; without the cap it
produces bit-identical batches to `train-v1` (tested). The first pass reads the windows in tr030's
order, later passes reshuffle them. Same model, optimizer, schedule and evaluation as every other run.

~3.3 repetitions are below the ~4 epochs that Muennighoff et al. (2023), *Scaling Data-Constrained
Language Models*, found to be almost as good as fresh data, so the control is a strong one: it gets
tr030's compute and close to the full value of repeating its Turkish data.

## Decision rule

Final-step Turkish validation BPB, each side averaged over its two seeds:

`diff = BPB_TR(x3 control) - BPB_TR(tr030)`, `noise = |BPB_TR(tr030) - BPB_TR(tr030_seed1)|` (0.0012).

- diff > 2 x noise: the mixture is better at equal compute; the gain is attributed to **English**.
- |diff| <= 2 x noise: **the two cannot be distinguished** at this noise level.
- diff < -2 x noise: at equal compute, **repeating Turkish beats adding English**.

The seed range of the x3 pair is reported next to it; if it is larger than tr030's, that is stated,
but the rule above is not changed.

## Checks reported alongside

- **Overfitting to repeated data:** the Turkish validation BPB curve of each x3 run, with the
  boundaries between passes marked. Reported: the lowest BPB and where it occurs, and
  `rise = final - lowest`. Overfitting is called visible if rise > 2 x noise.
- **Validation/train overlap:** share of Turkish validation 13-grams that occur in the 175,718
  repeated windows (`python -m slm.eval.overlap`), computed for both seeds before training. Exact
  duplicate documents are excluded by construction; this measures near-duplicates. Reported as is;
  a high share means the x3 BPB may be optimistic and is stated as a limitation.
- English validation BPB of the x3 runs is reported for completeness (they see no English).
