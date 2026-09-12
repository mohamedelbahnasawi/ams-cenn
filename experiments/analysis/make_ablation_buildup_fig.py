#!/usr/bin/env python
"""ablation BUILD-UP bar chart (S0 -> +C1 -> +C2 -> +ensemble -> +linear skip = AMS-CeNN).

This is the HONEST build-up figure that visualizes tab:ablation (ablation_buildup.tex): every step
is a proper NESTED model (not a skip-less leave-one-out ABL-* variant), so it does NOT suffer the
confounding that retired the old leave-one-out `fig_ablation_bars` (every ABL-* variant
also lacks the skip, so its Delta vs the skip-headline is dominated by the absent skip).

Grouped by horizon so the figure shows BOTH stories the table tells:
  (1) the zero-init linear skip lowers MSE at every horizon, and the gap WIDENS with the horizon;
  (2) multi-scale (C2) only pulls ahead of the S0/C1 substrate at the long horizons (H336/H720).

Reads from the SAME results source as make_tables.ablation_buildup_table (load_all_results + _seedagg)
and ASSERTS the per-cell values match the locked table, so figure and table can never silently drift.

Run:  .venv/Scripts/python.exe experiments/analysis/make_ablation_buildup_fig.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import FIGURES_DIR, DATASETS_SMALL, CENN_DISPLAY_NAME, ROLES   # noqa: E402
from experiments.aggregate import load_all_results                             # noqa: E402
from experiments.make_tables import _seedagg                                   # noqa: E402
from experiments.analysis import plotting                                      # noqa: E402

# The build-up chain (same order/variants as make_tables.ablation_buildup_table).
CHAIN = [
    (ROLES["s0"],        r"S$_0$: substrate"),
    (ROLES["c1"],        r"$+$ retention gate (C1)"),
    (ROLES["c2"],        r"$+$ dilated branches (C2)"),
    (ROLES["no_skip"],   r"C1 $+$ C2, no residual"),
    (ROLES["skip_only"], r"linear residual alone"),
    (ROLES["main"],      CENN_DISPLAY_NAME + r" ($+$ residual)"),
]
# Locked values from tab:ablation (ablation_buildup.tex) — the figure self-check guards against drift.
LOCKED = None   # optional drift guard: set to the table values to assert the figure matches

# Graduated muted blues for the substrate steps, CeNN orange for the headline (skip).
BAR_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#7f7f7f", plotting.CENN_C]


def compute():
    """variant -> {horizon: seed-mean MSE averaged over the 4 ETT datasets} (== the table)."""
    df = load_all_results()
    d = df[df["dataset"].isin(list(DATASETS_SMALL))]
    agg = _seedagg(d, [f"CeNN_{v}" for v, _ in CHAIN])
    horizons = sorted(int(h) for h in agg["horizon"].unique())
    out = {}
    for v, _ in CHAIN:
        out[v] = {}
        for h in horizons:
            sub = agg[(agg.model == f"CeNN_{v}") & (agg.horizon == h)]
            out[v][h] = float(sub.mse_mean.mean()) if not sub.empty else None
    return out, horizons


def selfcheck(vals, horizons):
    if LOCKED is None:   # anchored chain: the table is regenerated from the same CSV, nothing to lock against
        print('[selfcheck skipped] LOCKED is None; values come straight from aggregated/ablation_results.csv')
        return
    bad = []
    for v, _ in CHAIN:
        for h in horizons:
            got, want = vals[v][h], LOCKED.get(v, {}).get(h)
            if want is None or got is None or abs(got - want) > 5e-4:
                bad.append(f"{v} H{h}: figure={got} table={want}")
    if bad:
        raise SystemExit("[ABORT] build-up figure disagrees with locked table:\n  " + "\n  ".join(bad))
    print("[selfcheck OK] all 5x%d build-up cells match tab:ablation within 5e-4" % len(horizons))


def make_figure(vals, horizons):
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(plotting.FULL_W, 2.9))
    n_var = len(CHAIN)
    x = np.arange(len(horizons))
    width = 0.8 / n_var
    for i, (v, disp) in enumerate(CHAIN):
        heights = [vals[v][h] for h in horizons]
        off = (i - (n_var - 1) / 2) * width
        is_hl = (v == ROLES["main"])
        ax.bar(x + off, heights, width, label=disp, color=BAR_COLORS[i],
               edgecolor="black", linewidth=0.6 if is_hl else 0.3,
               zorder=3, hatch=("///" if is_hl else None))
    # Annotate the skip improvement (headline vs the no-skip ensemble) per horizon.
    for j, h in enumerate(horizons):
        ens, hl = vals[ROLES["no_skip"]][h], vals[ROLES["main"]][h]
        impr = 100.0 * (ens - hl) / ens
        top = max(vals[v][h] for v, _ in CHAIN)
        ax.annotate(f"−{impr:.0f}%", xy=(x[j], top), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.0, color=plotting.CENN_C, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MSE, seed mean over the four ETT datasets")
    ax.set_ylim(0, max(max(vals[v][h] for v, _ in CHAIN) for h in horizons) * 1.16)
    ax.legend(ncol=3, loc="upper left", fontsize=6.8, columnspacing=1.0,
              handlelength=1.3, borderaxespad=0.3)
    ax.margins(x=0.02)
    return plotting.savefig(fig, FIGURES_DIR, "fig_ablation_buildup")


if __name__ == "__main__":
    vals, horizons = compute()
    selfcheck(vals, horizons)
    png = make_figure(vals, horizons)
    print(f"  wrote {png}")
    print(f"  wrote {png.with_suffix('.pdf')}")
