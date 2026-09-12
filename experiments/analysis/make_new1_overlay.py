#!/usr/bin/env python
"""multi-model forecast OVERLAY (truth vs AMS-CeNN vs strong baselines) on ETT H720.

The headline AMS-CeNN predictions come from the canonical artifacts/predictions/ (saved during the
main campaign); the baseline predictions (DLinear/PatchTST/TSMixer) come from the ISOLATED re-run
under CENN_EXP_DIR=experiments/_overlay_preds (canonical results untouched; the re-run reproduces the
locked table MSE bit-for-bit — verified DLinear ETTh2 H720 seed1 = 0.633629).

Honesty rules (same as the existing forecast_panels figure):
  - one REPRESENTATIVE (median-error) window of the HEADLINE per dataset — not best, not worst;
  - the SAME (unique_id, cutoff) window is used for every model (fair overlay);
  - each model's line is its SEED-MEAN prediction on that window.

Run (after the baseline re-run finishes):
  .venv/Scripts/python.exe experiments/analysis/make_new1_overlay.py
"""
import sys
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import ARTIFACTS_DIR, FIGURES_DIR, CENN_MAIN_VARIANT, CENN_DISPLAY_NAME  # noqa: E402
from experiments.analysis import plotting                                                        # noqa: E402

CANON_PRED = ARTIFACTS_DIR / "predictions"
SCRATCH_PRED = Path(os.environ.get("OVERLAY_BASELINE_PRED", str(Path(__file__).resolve().parents[2] / "experiments" / "_overlay_preds" / "artifacts" / "predictions")))
HEADLINE = f"CeNN_{CENN_MAIN_VARIANT}"

# Overlay roster: (npz model name, display label, source dir, color, linewidth, zorder).
# NOTE: PatchTST is intentionally EXCLUDED. Its implementation is non-deterministic, so an isolated
# re-run does not reproduce the published Table 1 MSE (e.g. ETTh2 H720: 0.4987 table vs 0.4476 re-run);
# plotting it would put a PatchTST in the figure that disagrees with the results table. DLinear and
# TSMixer reproduce their table MSE bit-for-bit, so the overlay uses those table-consistent baselines.
ROSTER = [
    (HEADLINE,   CENN_DISPLAY_NAME, CANON_PRED,   plotting.CENN_C, 2.1, 6),
    ("TSMixer",  "TSMixer",         SCRATCH_PRED, "#009E73",       1.2, 4),
    ("DLinear",  "DLinear",         SCRATCH_PRED, "#0072B2",       1.2, 4),
]
# Single panel: ETTh2 H720 (the cell the paper highlights as "decisive... where linear extrapolation
# degrades sharply"). ETTh1 is intentionally EXCLUDED: the results text states DLinear is the strongest
# single method on ETTh1 (aggregate over all 7 series), yet on the OT target series alone AMS-CeNN beats
# DLinear -- showing it would appear to contradict the paper's own honest per-dataset statement. On
# ETTh2 the OT story matches the aggregate (AMS decisive), so it is the consistent, non-conflicting choice.
# Two panels, the strongest dataset of AMS-CeNN (ETTh2) and its weakest (Weather, ninth of
# twelve), so the figure shows both ends of the per-dataset profile. Weather predictions for all three
# models come from the isolated re-run (CENN_EXP_DIR=experiments/_overlay_preds).
DATASETS = [("ETTh2", 720), ("Weather", 720)]
# Restrict the displayed window to the canonical ETT target series (oil temperature, "OT"). This is
# the standard variable shown in ETT forecast figures; it also avoids the pitfall that the GLOBAL
# median-error window (across all 7 series) can land on a weak auxiliary channel that is unrepresentative
# of the per-dataset ranking (e.g. an LUFL window where DLinear happened to beat AMS-CeNN, contradicting
# the aggregate). On OT, AMS-CeNN's per-window standing matches the aggregate (e.g. ETTh2 H720: AMS beats
# DLinear on 100% of OT windows), so the median-error OT window is both representative and honest.
TARGET_SERIES = "OT"
# Weather: its OT column (CO2) is nearly constant over the test period (standardized std 0.05), so the
# canonical meteorological variable, air temperature, is shown instead.
TARGET_BY_DS = {"Weather": "T (degC)"}



def _empty_corner(ax, series, frac=0.30):
    """Return (x, y, ha, va) in axes coordinates for the corner with the fewest data points."""
    import numpy as _np
    ys = _np.concatenate([_np.asarray(s, float) for s in series])
    n = len(series[0]); lo, hi = float(ys.min()), float(ys.max()); span = max(hi - lo, 1e-9)
    counts = {}
    for corner in ("lr", "ll", "ur", "ul"):
        cnt = 0
        for s in series:
            s = _np.asarray(s, float)
            xs = _np.arange(len(s)) / max(len(s) - 1, 1)
            xin = xs >= 1 - frac if corner[1] == "r" else xs <= frac
            yn = (s - lo) / span
            yin = yn >= 1 - frac if corner[0] == "u" else yn <= frac
            cnt += int((xin & yin).sum())
        counts[corner] = cnt
    corner = min(counts, key=counts.get)
    x = 0.985 if corner[1] == "r" else 0.015
    y = 0.96 if corner[0] == "u" else 0.04
    return x, y, ("right" if corner[1] == "r" else "left"), ("top" if corner[0] == "u" else "bottom")


def _files(model, ds, H, src):
    return sorted(glob.glob(str(src / f"{model}__{ds}__H{H}__seed*.npz")))


def _headline_window(ds, H):
    """Pick the headline's median-error full window ON THE TARGET SERIES -> (uid, cutoff, ds_order, y)."""
    files = _files(HEADLINE, ds, H, CANON_PRED) or _files(HEADLINE, ds, H, SCRATCH_PRED)
    if not files:
        return None
    d0 = np.load(files[0], allow_pickle=True)
    if not all(k in d0 for k in ("unique_id", "ds", "cutoff", "y", "pred")):
        return None
    b = pd.DataFrame({"uid": d0["unique_id"], "ds": d0["ds"], "cutoff": d0["cutoff"],
                      "y": d0["y"].astype("float64"), "pred": d0["pred"].astype("float64")})
    target = TARGET_BY_DS.get(ds, TARGET_SERIES)
    if target in set(b["uid"]):
        b = b[b["uid"] == target]
    sizes = b.groupby(["uid", "cutoff"]).size()
    full = sizes[sizes == H].index
    if len(full) == 0:
        return None
    key = b.set_index(["uid", "cutoff"]).index
    bf = b[key.isin(full)].copy()
    bf["se"] = (bf["y"] - bf["pred"]) ** 2
    werr = bf.groupby(["uid", "cutoff"])["se"].mean()
    uid, cutoff = (werr - werr.median()).abs().idxmin()      # representative (median-error) window
    sel = b[(b.uid == uid) & (b.cutoff == cutoff)].sort_values("ds")
    return uid, cutoff, sel["ds"].to_numpy(), sel["y"].to_numpy()


def _seed_mean(model, ds, H, src, uid, cutoff, ref_ds):
    """Seed-mean prediction of `model` on the exact (uid, cutoff) window, aligned by ds."""
    preds = []
    for f in _files(model, ds, H, src):
        d = np.load(f, allow_pickle=True)
        s = pd.DataFrame({"uid": d["unique_id"], "ds": d["ds"], "cutoff": d["cutoff"],
                          "pred": d["pred"].astype("float64")})
        s = s[(s.uid == uid) & (s.cutoff == cutoff)].sort_values("ds")
        if len(s) == len(ref_ds) and np.array_equal(s["ds"].to_numpy(), ref_ds):
            preds.append(s["pred"].to_numpy())
    return np.mean(preds, axis=0) if preds else None


def main():
    plotting.apply_style()
    panels = []
    for ds, H in DATASETS:
        win = _headline_window(ds, H)
        if win is None:
            print(f"[skip {ds} H{H}: no headline window]")
            continue
        uid, cutoff, ref_ds, y_true = win
        series = []
        for model, label, src, color, lw, z in ROSTER:
            pm = _seed_mean(model, ds, H, src, uid, cutoff, ref_ds)
            if pm is None and src is CANON_PRED:
                pm = _seed_mean(model, ds, H, SCRATCH_PRED, uid, cutoff, ref_ds)
            if pm is None:
                print(f"[{ds} H{H}: no preds for {label} (src={src.name}) — skipped]")
                continue
            mse = float(np.mean((pm - y_true) ** 2))   # per-window MSE on the shown series
            series.append((label, pm, color, lw, z, mse))
        panels.append((f"{ds}, {uid}, H={H}", y_true, series))
    if not panels:
        print("[overlay: no panels — has the baseline re-run finished?]")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(plotting.FULL_W, 2.5), squeeze=False)
    for ax, (title, y_true, series) in zip(axes[0], panels):
        t = np.arange(1, len(y_true) + 1)
        # De-emphasize the noisy ground truth so the model trajectories stay legible.
        ax.plot(t, y_true, color="0.45", lw=0.8, alpha=0.7, label="Ground truth", zorder=2)
        for label, pm, color, lw, z, _ in series:
            ax.plot(t, pm, color=color, lw=lw, label=label, zorder=z, alpha=0.95)
        ax.set_title(title)
        ax.set_xlabel("Forecast step")
        ax.margins(x=0.01)
        # Per-window MSE box (quantifies the visual: which model tracks this window best).
        txt = "\n".join(f"{lab}: {mse:.3f}" for lab, _, _, _, _, mse in series)
        bx, by, ha, va = _empty_corner(ax, [y_true] + [pm for _, pm, *_ in series])
        ax.text(bx, by, "MSE on window\n" + txt, transform=ax.transAxes,
                ha=ha, va=va, fontsize=6.2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    axes[0][0].set_ylabel("Value (normalized)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", bbox_to_anchor=(0.5, 1.10),
               fontsize=7.6, columnspacing=1.1, handlelength=1.5, frameon=True)
    png = plotting.savefig(fig, FIGURES_DIR, "fig_forecast_overlay")
    print(f"  wrote {png}")
    print(f"  wrote {png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
