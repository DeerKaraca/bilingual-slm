"""Research results from the runs' log.csv files, following reports/analysis_plan.md.

    python -m slm.eval.analysis --runs-dir runs
    python -m slm.eval.analysis --hub-repo USER/bilingual-slm-runs   # download the log.csv files first

Writes results.md, results.json and four figures to reports/research/. Unfinished runs appear in
the curves but not in the final-step results, so the script can be run while training continues.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slm.config import load_config
from slm.plotting import CATEGORICAL, INK_2, SEQUENTIAL, end_label, figure, legend, line, save, style

MIXTURES = ["tr000", "tr010", "tr030", "tr050", "tr100"]
CONTROL, SEED = "tr_only_control", "tr030_seed1"
MONOLINGUAL = {"tr": "tr100", "en": "tr000"}
LANGS = {"tr": "Turkish", "en": "English"}


@dataclass
class Run:
    name: str
    tr_share: float
    complete: bool
    tokens: np.ndarray  # total training tokens at each evaluation
    bpb: dict[str, np.ndarray]  # language -> validation BPB at each evaluation

    def share(self, lang: str) -> float:
        return self.tr_share if lang == "tr" else 1 - self.tr_share

    def seen(self, lang: str) -> np.ndarray:
        """Tokens of `lang` seen at each evaluation (the loader keeps the mixture ratio exact)."""
        return self.tokens * self.share(lang)

    def final(self, lang: str) -> float:
        return float(self.bpb[lang][-1])


def load_run(runs_dir: Path, name: str) -> Run | None:
    path = runs_dir / name / "log.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cfg = load_config(Path("configs/experiments") / f"{name}.yaml")
    column = lambda key: np.array([float(r[key]) for r in rows])  # noqa: E731
    return Run(name, cfg.data.tr_fraction, int(rows[-1]["step"]) == cfg.total_steps,
               column("tokens"), {lang: column(f"{lang}_bpb") for lang in LANGS})


def download_logs(hub_repo: str, runs_dir: Path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in [*MIXTURES, CONTROL, SEED]:
        try:
            hf_hub_download(hub_repo, f"{name}/log.csv", local_dir=runs_dir)
        except EntryNotFoundError:
            pass  # not started yet


def verdict(diff: float, noise: float | None, above: str, below: str) -> str:
    """Pre-registered rule: a difference counts only beyond twice the seed-to-seed range."""
    if noise is None:
        return "no seed-noise estimate yet"
    if diff > 2 * noise:
        return above
    if diff < -2 * noise:
        return below
    return "inconclusive at this noise level"


def analyse(runs: dict[str, Run]) -> dict:
    done = {name: run for name, run in runs.items() if run.complete}
    result = {
        "complete": sorted(done),
        "final": {name: {"tr_share": r.tr_share, "tr_tokens": r.seen("tr")[-1], "en_tokens": r.seen("en")[-1],
                         **{f"{lang}_bpb": r.final(lang) for lang in LANGS}} for name, r in done.items()},
    }
    noise = None
    if "tr030" in done and SEED in done:
        noise = {lang: abs(done["tr030"].final(lang) - done[SEED].final(lang)) for lang in LANGS}
        tr030 = {lang: (done["tr030"].final(lang) + done[SEED].final(lang)) / 2 for lang in LANGS}
    elif "tr030" in done:
        tr030 = {lang: done["tr030"].final(lang) for lang in LANGS}
    else:
        tr030 = None
    result["noise"], result["tr030_mean"] = noise, tr030

    if tr030 and CONTROL in done:
        transfer = done[CONTROL].final("tr") - tr030["tr"]
        result["transfer"] = {
            "bpb": transfer,
            "relative_to_control": transfer / done[CONTROL].final("tr"),
            "verdict": verdict(transfer, noise and noise["tr"], "English helps Turkish (positive transfer)",
                               "English hurts Turkish (interference)"),
        }

    result["interference"] = []
    for lang in LANGS:
        mono = runs.get(MONOLINGUAL[lang])
        if mono is None:
            continue
        for name in MIXTURES:
            run = done.get(name)
            if run is None or not 0 < run.share(lang) < 1:
                continue
            x = run.seen(lang)[-1]
            if x > mono.seen(lang)[-1]:
                continue  # the monolingual curve has not reached this many tokens yet
            reference = float(np.interp(x, mono.seen(lang), mono.bpb[lang]))
            gap = run.final(lang) - reference
            result["interference"].append({
                "run": name, "lang": lang, "tokens": x, "final_bpb": run.final(lang), "monolingual_curve_bpb": reference,
                "gap": gap, "verdict": verdict(gap, noise and noise[lang], "interference likely",
                                               "below the monolingual curve (curve not schedule-matched)"),
            })
    return result


def plot_curves(runs: dict[str, Run], out: Path):
    """Figure 1: BPB during training, against total tokens and against tokens of the evaluated language."""
    fig, axes = figure(2, 2, width=10, height=6.6, sharey="row")
    for row, (lang, lang_name) in enumerate(LANGS.items()):
        for col, xlabel in enumerate(["total tokens (M)", f"{lang_name} tokens seen (M)"]):
            ax = axes[row, col]
            for name, color in zip(MIXTURES, SEQUENTIAL):
                run = runs.get(name)
                if run is None or (col == 1 and run.share(lang) == 0):
                    continue
                x = run.tokens if col == 0 else run.seen(lang)
                line(ax, x / 1e6, run.bpb[lang], color, f"{run.tr_share:.0%} Turkish")
            style(ax, f"{lang_name} validation BPB", xlabel, "bits per byte" if col == 0 else "")
    legend(axes[0, 0], loc="upper right")
    save(fig, out)


def plot_final_vs_share(runs: dict[str, Run], out: Path):
    """Figure 2: final BPB of both languages against the Turkish share; tr030 shows its seed range."""
    done = [runs[n] for n in MIXTURES if n in runs and runs[n].complete]
    if not done:
        return
    fig, axes = figure()
    ax = axes[0, 0]
    seed = runs.get(SEED) if runs.get(SEED) and runs[SEED].complete else None
    for (lang, lang_name), color in zip(LANGS.items(), CATEGORICAL):
        xs = [r.tr_share for r in done]
        ys = [r.final(lang) for r in done]
        if seed and 0.3 in xs:
            i = xs.index(0.3)
            pair = sorted([ys[i], seed.final(lang)])
            ys[i] = sum(pair) / 2
            ax.errorbar([0.3], [ys[i]], yerr=[[ys[i] - pair[0]], [pair[1] - ys[i]]], color=color, capsize=4,
                        linewidth=1.5, fmt="none")
        line(ax, xs, ys, color, lang_name)
        end_label(ax, xs[-1], ys[-1], f"{lang_name}  {ys[-1]:.3f}")
    style(ax, "Final validation BPB vs Turkish share of the training mix", "Turkish share", "bits per byte")
    ax.set_xticks([0, 0.1, 0.3, 0.5, 1.0], ["0%", "10%", "30%", "50%", "100%"])
    ax.set_xlim(-0.03, 1.25)
    legend(ax, loc="upper center")
    save(fig, out)


def plot_transfer(runs: dict[str, Run], result: dict, out: Path):
    """Figure 3: tr030 (both seeds) and the Turkish-only control at equal Turkish tokens."""
    fig, axes = figure()
    ax = axes[0, 0]
    for name, label, color, dashed in [("tr030", "30% Turkish (seed 0)", CATEGORICAL[0], False),
                                       (SEED, "30% Turkish (seed 1)", CATEGORICAL[0], True),
                                       (CONTROL, "Turkish only, same Turkish tokens", CATEGORICAL[1], False)]:
        if name in runs:
            line(ax, runs[name].seen("tr") / 1e6, runs[name].bpb["tr"], color, label, dashed)
    if "transfer" in result:
        x = runs[CONTROL].seen("tr")[-1] / 1e6
        y_mix, y_ctrl = result["tr030_mean"]["tr"], runs[CONTROL].final("tr")
        ax.annotate("", xy=(x, y_mix), xytext=(x, y_ctrl), arrowprops={"arrowstyle": "<->", "color": INK_2, "lw": 1})
        end_label(ax, x, (y_mix + y_ctrl) / 2, f"transfer {result['transfer']['bpb']:+.3f} BPB")
    style(ax, "Turkish BPB at equal Turkish tokens: mixture vs Turkish-only", "Turkish tokens seen (M)", "bits per byte")
    legend(ax, loc="upper right")
    save(fig, out)


def plot_seed_noise(runs: dict[str, Run], out: Path):
    """Figure 4: the two tr030 seeds over training and their difference."""
    if "tr030" not in runs or SEED not in runs:
        return
    a, b = runs["tr030"], runs[SEED]
    n = min(len(a.tokens), len(b.tokens))
    fig, axes = figure(1, 2, width=10, height=3.8)
    for (lang, lang_name), color in zip(LANGS.items(), CATEGORICAL):
        line(axes[0, 0], a.tokens / 1e6, a.bpb[lang], color, f"{lang_name}, seed 0")
        line(axes[0, 0], b.tokens / 1e6, b.bpb[lang], color, f"{lang_name}, seed 1", dashed=True)
        line(axes[0, 1], a.tokens[:n] / 1e6, b.bpb[lang][:n] - a.bpb[lang][:n], color, lang_name)
    axes[0, 1].axhline(0, color=INK_2, linewidth=0.8)
    style(axes[0, 0], "tr030: two seeds", "total tokens (M)", "bits per byte")
    style(axes[0, 1], "Seed 1 minus seed 0", "total tokens (M)", "BPB difference")
    legend(axes[0, 0], loc="upper right")
    legend(axes[0, 1], loc="upper right")
    save(fig, out)


def to_markdown(result: dict) -> str:
    lines = ["# Research results", "", "Analysis as pre-registered in [analysis_plan.md](../analysis_plan.md).", "",
             f"Complete runs: {', '.join(result['complete']) or 'none yet'}.", "",
             "| Run | Turkish share | Turkish tokens | English tokens | TR BPB | EN BPB |", "|---|---|---|---|---|---|"]
    for name, f in sorted(result["final"].items(), key=lambda kv: (kv[0] == CONTROL, kv[1]["tr_share"], kv[0])):
        lines.append(f"| {name} | {f['tr_share']:.0%} | {f['tr_tokens'] / 1e6:.0f}M | {f['en_tokens'] / 1e6:.0f}M | "
                     f"{f['tr_bpb']:.4f} | {f['en_bpb']:.4f} |")
    if result["noise"]:
        n = result["noise"]
        lines += ["", f"**Seed noise** (|tr030 - tr030_seed1|): TR {n['tr']:.4f}, EN {n['en']:.4f} BPB."]
    if "transfer" in result:
        t = result["transfer"]
        lines += ["", f"**Q2, transfer from English:** control minus tr030 = {t['bpb']:+.4f} TR BPB "
                      f"({t['relative_to_control']:+.1%} of the control) -> {t['verdict']}."]
    if result["interference"]:
        lines += ["", "**Q3, interference (indicative; monolingual curve is not schedule-matched):**", "",
                  "| Run | Language | Tokens of it seen | Final BPB | Monolingual curve | Gap | Reading |",
                  "|---|---|---|---|---|---|---|"]
        for q in result["interference"]:
            lines.append(f"| {q['run']} | {q['lang'].upper()} | {q['tokens'] / 1e6:.0f}M | {q['final_bpb']:.4f} | "
                         f"{q['monolingual_curve_bpb']:.4f} | {q['gap']:+.4f} | {q['verdict']} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--hub-repo", help="download each run's log.csv from this HF model repo first")
    p.add_argument("--out", type=Path, default=Path("reports/research"))
    args = p.parse_args()

    if args.hub_repo:
        download_logs(args.hub_repo, args.runs_dir)
    runs = {name: run for name in [*MIXTURES, CONTROL, SEED] if (run := load_run(args.runs_dir, name))}
    result = analyse(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, args.out / "curves.png")
    plot_final_vs_share(runs, args.out / "final_vs_share.png")
    plot_transfer(runs, result, args.out / "transfer.png")
    plot_seed_noise(runs, args.out / "seed_noise.png")
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    (args.out / "results.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
