"""RQ5 repair trade-off scatter (prototype for tab:rq5-repair's companion plot).

Two axes: x = correctness (Incorrect vs Correct), y = latency.
Each kernel contributes three points: original (baseline), incorrect (the
divergent/broken optimization), and repaired. Lines trace original -> incorrect
-> repaired so the reader sees the trade-off: the unsafe optimization is fast
but wrong, and repair restores correctness while staying faster than baseline.

Produces two variants:
  repair_tradeoff_abs.{pdf,png}  - absolute latency, log y (faithful to "latency")
  repair_tradeoff_norm.{pdf,png} - latency / baseline, linear y (kernels comparable)

Data: kernel-testing/repair/RESULTS.md (verified on trn1, 2026-06-10).
Run:  uv run --with matplotlib python plot_repair_tradeoff.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# kernel -> (color, baseline_ms, incorrect_ms, repaired_ms)
KERNELS = [
    ("reference/rms",  "#2a9d8f", 0.095, 0.047, 0.064),
    ("nkibench/bmm",   "#6699cc", 8.07,  3.12,  6.94),
    ("nkibench/mamba", "#e76f51", 0.132, 0.057, 0.098),
]

X_INCORRECT, X_CORRECT = 0.0, 1.0
NUDGE = {"reference/rms": -0.06, "nkibench/bmm": 0.0, "nkibench/mamba": 0.06}


def draw(ax, normalize: bool):
    for name, color, base, bad, rep in KERNELS:
        denom = base if normalize else 1.0
        y_orig, y_bad, y_rep = base / denom, bad / denom, rep / denom
        dx = NUDGE[name]
        xo, xb, xr = X_CORRECT + dx, X_INCORRECT + dx, X_CORRECT + dx

        ax.plot([xo, xb, xr], [y_orig, y_bad, y_rep],
                color=color, linewidth=0.9, alpha=0.5, zorder=1)
        ax.scatter(xo, y_orig, marker="o", s=55, facecolors="none",
                   edgecolors=color, linewidths=1.5, zorder=3)
        ax.scatter(xb, y_bad, marker="X", s=75, color=color, zorder=3)
        ax.scatter(xr, y_rep, marker="s", s=55, color=color, zorder=3)

    if not normalize:
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms, log scale)", fontsize=9)
    else:
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("Latency relative to baseline", fontsize=9)
        ax.set_ylim(0.3, 1.08)

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([X_INCORRECT, X_CORRECT])
    ax.set_xticklabels(["Incorrect", "Correct"], fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    kernel_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=7,
               label=n) for n, c, *_ in KERNELS
    ]
    variant_handles = [
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="none",
               linestyle="none", markersize=7, label="Original"),
        Line2D([0], [0], marker="X", color="0.3", linestyle="none",
               markersize=8, label="Incorrect"),
        Line2D([0], [0], marker="s", color="0.3", linestyle="none",
               markersize=7, label="Repaired"),
    ]
    leg1 = ax.legend(handles=kernel_handles, loc="upper left", fontsize=7.5,
                     framealpha=0.95, title="Kernel", title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, loc="lower left", fontsize=7.5,
              framealpha=0.95, title="Variant", title_fontsize=7.5)


for normalize, tag in [(False, "abs"), (True, "norm")]:
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    draw(ax, normalize)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"repair_tradeoff_{tag}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", HERE / f"repair_tradeoff_{tag}.pdf")
