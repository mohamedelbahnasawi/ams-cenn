"""Generate the two robustness and gate figures for the paper:
  fig08_robustness.pdf  — robustness degradation curves (contamination robustness)
  fig09_gate_adaptation.pdf — gate-variation: pointwise vs context gate (the "Adaptive" answer)
Outputs go to experiments/aggregated/figures/.
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import numpy as np

ROBUSTNESS_CSV = "experiments/_robustness/robustness_degradation.csv"
from experiments.config import CENN_MAIN_VARIANT, FIGURES_DIR  # noqa: E402
OUT = str(FIGURES_DIR); os.makedirs(OUT, exist_ok=True)
AMS = f"CeNN_{CENN_MAIN_VARIANT}"
MODELS = [AMS, "DLinear", "TSMixer", "PatchTST", "TCN"]
LABEL = {AMS: "AMS-CeNN", "DLinear": "DLinear", "TSMixer": "TSMixer",
         "PatchTST": "PatchTST", "TCN": "TCN"}
# (csv key, panel title, x-axis label)
KINDS = [("gauss", "Gaussian noise", "noise standard deviation (z-units)"),
         ("spike", "Outlier spikes", "fraction of spiked steps"),
         ("mask", "Missing blocks", "block length (steps)"),
         ("scale", "Gain error", "multiplicative gain error"),
         ("shift", "Level shift", "shift (z-units)")]
COL = {AMS: "#E8772E", "DLinear": "#4C72B0", "TSMixer": "#55A868",
       "PatchTST": "#8172B3", "TCN": "#937860"}


def load():
    # ratios[(model,kind)][level] = mean over datasets of mean_ratio
    acc = defaultdict(lambda: defaultdict(list))
    with open(ROBUSTNESS_CSV) as f:
        for r in csv.DictReader(f):
            acc[(r["model"], r["kind"])][float(r["level"])].append(float(r["mean_ratio"]))
    out = {}
    for k, lv in acc.items():
        out[k] = {level: float(np.mean(v)) for level, v in lv.items()}
    return out


def fig_robustness(data):
    """2 x 3 grid: the three contamination families on the top row, the two distribution-shift
    families on the bottom row, legend in the last slot. Large type for print."""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6))
    axes = axes.ravel()
    for ax, (kind, title, xlabel) in zip(axes, KINDS):
        for m in MODELS:
            d = data.get((m, kind), {})
            if not d:
                continue
            levels = sorted(d)
            xs = [0.0] + levels
            ys = [1.0] + [d[l] for l in levels]   # anchor clean = ratio 1
            ax.plot(xs, ys, marker="o", ms=3.2, lw=2.0 if m == AMS else 1.2,
                    color=COL[m], label=LABEL[m], zorder=3 if m == AMS else 2,
                    alpha=1.0 if m == AMS else 0.85)
        ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("MSE$_{\\rm perturbed}$ / MSE$_{\\rm clean}$", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(handles, labels, loc="center", fontsize=9, frameon=False, title="Model", title_fontsize=9)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    p = os.path.join(OUT, "fig08_robustness.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_gate():
    # alpha temporal-std (gate variation) measured across regimes + the context-gate variant.
    # pointwise gate: ~1e-4 floor everywhere; context gate: 0.0137 on SynthHetero (still << 0.05).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    # left: alpha temporal-std, pointwise vs context, log scale, with 0.05 keep threshold
    cats = ["standard\nbenchmarks", "synth\nchirp", "synth\nhetero"]
    pointwise = [1.0e-4, 2.8e-4, 2.1e-4]
    context = [np.nan, np.nan, 1.37e-2]
    x = np.arange(len(cats)); w = 0.38
    a1.bar(x - w/2, pointwise, w, label="pointwise gate", color="#4C72B0")
    ctx_vals = [0 if v != v else v for v in context]
    a1.bar(x + w/2, ctx_vals, w, label="context gate (fix)", color="#E8772E")
    a1.axhline(0.05, color="crimson", lw=1.4, ls="--", label="adaptation threshold (0.05)")
    a1.set_yscale("log")
    a1.set_ylim(5e-5, 1e-1)
    a1.set_xticks(x); a1.set_xticklabels(cats, fontsize=7.5)
    a1.set_ylabel(r"gate temporal std of $\alpha$", fontsize=9)
    a1.set_title("Gate variation", fontsize=10, fontweight="bold")
    a1.legend(fontsize=6.8, loc="upper left", framealpha=0.9)
    a1.tick_params(labelsize=7.5)
    # right: accuracy on SynthHetero — fixed vs pointwise vs context (marginal benefit)
    labels = ["fixed-$\\alpha$", "pointwise", "context"]
    mse = [1.0575, 1.0524, 1.0402]
    cols = ["#937860", "#4C72B0", "#E8772E"]
    a2.bar(labels, mse, color=cols, width=0.6)
    a2.set_ylim(1.0, 1.07)
    a2.set_ylabel("MSE (SynthHetero, no-skip)", fontsize=9)
    a2.set_title("Accuracy effect", fontsize=10, fontweight="bold")
    for i, v in enumerate(mse):
        a2.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7.5)
    a2.tick_params(labelsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig09_gate_adaptation.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_robustness(load())
    fig_gate()
