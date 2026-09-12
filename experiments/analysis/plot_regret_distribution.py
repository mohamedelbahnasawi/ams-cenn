"""Per-cell regret distribution figure (replaces the worst-case bar chart + the per-cell regret table).

For every method, the 28 per-cell ratios MSE / best-MSE-in-cell (unrounded seed-mean MSE from the result
files; --tex parses the manuscript table instead) are drawn as
points with a box (median, quartiles) on a log axis; the per-dataset worst-case statistic R_m of the paper
(mean ratio over horizons within a dataset, then max over datasets) is marked separately. Methods are
ordered by R_m; AMS-CeNN is highlighted.

Usage:  .venv/Scripts/python.exe experiments/analysis/plot_regret_distribution.py [--tex PATH]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis import plotting                                    # noqa: E402
from experiments.analysis.worst_case_regret_28cell import (                  # noqa: E402
    DEFAULT_TEX, parse_main_results, ratios_per_cell, per_dataset_stat)
from experiments.analysis import make_figures as mf                          # noqa: E402
from experiments.config import FIGURES_DIR, CENN_MAIN_VARIANT, BASELINES_ALL  # noqa: E402

OUT_DIRS = [Path(FIGURES_DIR)]
import os
if os.environ.get("FIG_OUT_DIR"):
    OUT_DIRS.append(Path(os.environ["FIG_OUT_DIR"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=None,
                    help="parse tab:main_results from this .tex instead of the unrounded result files")
    args = ap.parse_args()
    if args.tex:
        models, mse = parse_main_results(args.tex)
    else:  # unrounded seed-mean MSE from the result files (what the paper's statistics use)
        raw = mf.seed_mean_mse()
        keep = [f"CeNN_{CENN_MAIN_VARIANT}"] + BASELINES_ALL
        mse = {(mf.label_of(m), d, h): v for (m, d, h), v in raw.items() if m in keep}
        models = [mf.label_of(m) for m in keep]
    datasets = sorted({d for (_, d, _) in mse})
    horizons = sorted({h for (_, _, h) in mse})
    ratios = ratios_per_cell(models, mse, datasets, horizons)

    rm = {m: per_dataset_stat(m, ratios, datasets, horizons) for m in models}   # percent
    order = sorted(models, key=lambda m: rm[m])
    # axis variable: 1 + regret in percent, on a log scale (so 0% sits at 1 and 1%, 10%, 100% are equally spaced)
    cells = {m: 1 + np.array([(ratios[(m, d, h)] - 1) * 100 for d in datasets for h in horizons]) for m in models}

    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(plotting.COL_W, 3.1))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        y = i
        r = cells[m]
        hl = (m == "AMS-CeNN")
        col = plotting.CENN_C if hl else "#6c757d"
        ax.boxplot([r], positions=[y], vert=False, widths=0.55, showfliers=False, whis=(0, 100),
                   patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.25 if not hl else 0.35, edgecolor=col, lw=0.9),
                   whiskerprops=dict(color=col, lw=0.8), capprops=dict(color=col, lw=0.8),
                   medianprops=dict(color=col, lw=1.6))
        jitter = rng.uniform(-0.16, 0.16, size=r.size)
        ax.scatter(r, y + jitter, s=7, color=col, alpha=0.8, zorder=3, linewidths=0)
        ax.scatter([1 + rm[m]], [y], marker="D", s=22, facecolor="white", edgecolor="black",
                   lw=0.8, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.6)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "AMS-CeNN":
            lab.set_color(plotting.CENN_C); lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    tick_pct = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_xticks([1 + t for t in tick_pct])
    ax.set_xticklabels([f"{t}%" for t in tick_pct], fontsize=7.6)
    ax.set_xlim(0.93, 800)
    ax.minorticks_off()
    ax.set_xlabel("MSE above the best method in the cell (log scale)", fontsize=7.8)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], marker="D", s=22, facecolor="white", edgecolor="black", lw=0.8,
               label=r"per-dataset worst case $R_m$")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_regret_distribution.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_regret_distribution.png", dpi=200, bbox_inches="tight")
        print("wrote", d / "fig_regret_distribution.pdf")
    for m in order:
        r = cells[m]
        print(f"{m:<13} R_m {rm[m]:6.1f}%  median {np.median(r - 1):5.1f}%  "
              f"p90 {np.percentile(r - 1, 90):6.1f}%  max {(r.max() - 1):6.1f}%")


if __name__ == "__main__":
    main()
