"""Shared chart style for README figures: light-mode tokens of a validated reference palette."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: distinguishable for all-pairs CVD
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # one hue, light -> dark, for ordered series


def figure(rows: int = 1, cols: int = 1, width: float = 6.4, height: float = 3.8, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=150, facecolor=SURFACE, squeeze=False, **kwargs)
    for ax in axes.flat:
        style(ax)
    return fig, axes


def style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Recessive grid and axes; text in ink colors, never in series colors."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def line(ax, xs, ys, color, label=None, dashed=False):
    """2px line with markers ringed in the surface color so overlapping points stay separate."""
    ax.plot(xs, ys, color=color, linewidth=1.5 if dashed else 2, linestyle="--" if dashed else "-", marker="o",
            markersize=4 if len(xs) > 12 else 6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)


def legend(ax, **kwargs):
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
