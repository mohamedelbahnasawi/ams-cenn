"""Refined appendix figures for the manuscript:
  fig06_lookback_sweep   lookback sensitivity, full model vs the variant without the linear residual
  fig09_gate_adaptation  direct test of gate adaptivity (gate variation + accuracy effect)
House style (SciencePlots ieee, serif, no in-figure titles, short legends). Writes PDF+PNG to the figures
folder. Usage: python make_appendix_figs.py   (from the repository root)
"""
import glob
import json
import statistics
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import RESULTS_DIR, EXPERIMENTS_DIR, FIGURES_DIR  # noqa: E402
from experiments.analysis import plotting                    # noqa: E402

OUT = Path(FIGURES_DIR)
CENN_C, GREY = "#D55E00", "#5A5A5A"


def lookback(datasets=("ETTh1", "ETTh2", "Weather"), horizon=96, seeds=(1, 42, 123),
             full_variant="AMS-Anc", nores_variant="C1C2-Anc"):
    """Lookback sweep: headline AMS-Anc and its no-residual arm C1C2-Anc, three common seeds
    at every L (the L=512 headline has five seeds; only the three shared ones enter the mean)."""
    Ls = [96, 192, 336, 512]

    def m(variant, L):
        root = str(RESULTS_DIR) if L == 512 else str(EXPERIMENTS_DIR / f"L{L}" / "results")
        vs = []
        for d in datasets:
            for s in seeds:
                fs = glob.glob(f"{root}/CeNN_{variant}__{d}__H{horizon}__seed{s}.json")
                assert fs, (variant, d, L, s)
                vs.append(json.load(open(fs[0]))["mse"])
        return statistics.mean(vs)

    full = [m(full_variant, L) for L in Ls]
    nores = [m(nores_variant, L) for L in Ls]
    print("full", np.round(full, 4), "w/o residual", np.round(nores, 4))
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(Ls, full, color=CENN_C, marker="o", ms=4, lw=1.5, label="AMS-CeNN")
    ax.plot(Ls, nores, color=GREY, marker="s", ms=3.6, lw=1.3, ls="--", label="AMS-CeNN w/o residual")
    ax.set_xscale("log", base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(v) for v in Ls]); ax.minorticks_off()
    ax.set_xlabel("lookback $L$"); ax.set_ylabel("MSE")
    ax.set_ylim(0.235, 0.29)
    ax.legend(loc="upper right", frameon=False)
    plotting.savefig(fig, OUT, "fig06_lookback_sweep")


def gate():
    # Values measured in the gate-adaptivity test (conference-era configuration); see plot_robustness_gate.fig_gate.
    cats = ["benchmarks", "chirp", "heteroscedastic"]
    pointwise = [1.0e-4, 2.8e-4, 2.1e-4]        # AMS-CeNN gate, temporal std of alpha
    context = [np.nan, np.nan, 1.37e-2]         # context-aware gate, run on the heteroscedastic task only
    mse = {"fixed rate": 1.0575, "AMS-CeNN gate": 1.0524, "context-aware gate": 1.0402}
    plotting.apply_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.3), gridspec_kw=dict(height_ratios=[1.15, 1]))
    x = np.arange(len(cats)); w = 0.36
    a1.bar(x - w / 2, pointwise, w, color=CENN_C, label="AMS-CeNN gate")
    a1.bar(x[2] + w / 2, context[2], w, color=GREY, label="context-aware gate")
    a1.axhline(0.05, color="k", lw=0.9, ls=":", label="adaptation threshold")
    a1.set_yscale("log"); a1.set_ylim(5e-5, 4e-1)
    a1.set_xticks(x); a1.set_xticklabels(cats)
    a1.set_ylabel(r"temporal s.d. of the gate $\alpha$")
    a1.set_xlabel("task")
    a1.legend(loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=False, ncol=1, fontsize=7.2, handlelength=1.4)
    a1.set_title("(a) gate variation", fontsize=9, loc="left")
    labels = ["fixed", "AMS-CeNN", "context-aware"]; vals = list(mse.values())
    a2.bar(labels, vals, color=[GREY, CENN_C, GREY], width=0.55)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    a2.set_ylim(1.0, 1.075); a2.set_ylabel("MSE, heteroscedastic task"); a2.set_xlabel("gate")
    a2.set_title("(b) accuracy without the residual", fontsize=9, loc="left")
    plotting.savefig(fig, OUT, "fig09_gate_adaptation")


if __name__ == "__main__":
    lookback()
    gate()
    print("wrote", OUT)
