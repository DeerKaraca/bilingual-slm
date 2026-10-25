"""Serving charts for the README from reports/serving.json.

    python benchmarks/plots.py
"""

import json
from pathlib import Path

from slm.plotting import CATEGORICAL, end_label, figure, legend, line, save, style


def line_chart(series: dict[str, tuple[list, list]], title, xlabel, ylabel, out: Path, value_fmt):
    fig, axes = figure()
    ax = axes[0, 0]
    style(ax, title, xlabel, ylabel)
    for (name, (xs, ys)), color in zip(series.items(), CATEGORICAL):
        line(ax, xs, ys, color, name)
        end_label(ax, xs[-1], ys[-1], f"{name}  {value_fmt.format(ys[-1])}")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(max(ys) for _, ys in series.values()) * 1.15)
    ax.set_xlim(right=xs[-1] + (xs[-1] - xs[0]) * 0.35)  # room for the end labels
    legend(ax, loc="upper left")
    save(fig, out)


def main(report=Path("reports/serving.json")):
    data = json.loads(report.read_text())
    meta, r = data["meta"], data["results"]
    lengths, conc = meta["output_lengths"], meta["concurrency"]
    line_chart(
        {label: (lengths, [r[f"latency|{mode}|{n}"]["latency_p50_s"] for n in lengths])
         for label, mode in [("no KV cache", "no_cache"), ("KV cache", "sequential")]},
        "Request latency vs output length (one request at a time)", "output tokens", "latency (s)",
        report.with_name("serving_kv_cache.png"), "{:.1f} s",
    )
    line_chart(
        {mode: (conc, [r[f"throughput|{mode}|{c}"]["tokens_per_s"] for c in conc])
         for mode in ("sequential", "static", "continuous")},
        "Generation throughput vs concurrent clients", "concurrent clients", "tokens / s",
        report.with_name("serving_throughput.png"), "{:.0f} tok/s",
    )


if __name__ == "__main__":
    main()
