"""Forecast decomposition figure: what the certified cellular path contributes on top of the linear path.

By eq. (9) of the manuscript the forecast is  y = LVN^{-1}( G(x~) + Skip(x~) ), with LVN^{-1}(z) = (z - delta)/gamma
+ x_L.  Because LVN^{-1} is affine, the forecast splits EXACTLY in the original scale into
    linear path      y_lin  = (Skip(x~) - delta)/gamma + x_L      (what the residual alone would output)
    cellular path    y_cell = G(x~)/gamma                        (the correction the certified block adds)
    y = y_lin + y_cell.
This script captures Skip(x~), the model input x and the full output y with forward hooks on a trained
checkpoint (no retraining), reconstructs y_lin and y_cell, and draws one representative (median-error)
test window of the target series per dataset: truth, full forecast, linear path, cellular correction.
It also prints the energy split of the two paths over ALL test windows (shares of ||y - x_L||^2, i.e. of
the forecast's deviation from the last value, so that the level itself is not counted as 'linear').

Usage:
  .venv/Scripts/python.exe experiments/analysis/plot_forecast_decomposition.py \
      --cells ETTh2:720 Weather:720 --seed 1 [--ckpt-root experiments/checkpoints]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import SPLITS, FIGURES_DIR, CENN_MAIN_VARIANT, CENN_DISPLAY_NAME  # noqa: E402
from experiments.runner import load_dataset                                                # noqa: E402
from experiments.analysis import plotting                                                  # noqa: E402
from neuralforecast import NeuralForecast                                                  # noqa: E402

TARGET = "OT"
TARGET_BY_DS = {"Weather": "T (degC)"}   # Weather's OT column is nearly constant in the test period



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


def capture(ckpt: Path, dataset: str):
    """Run the checkpoint over the test windows; return dict of tensors [W, H, V] / [W, L, V] and gamma/delta."""
    nf = NeuralForecast.load(path=str(ckpt))
    for a in ("dataset", "uids", "last_dates", "ds"):
        if not hasattr(nf, a):
            setattr(nf, a, None)
    inner = nf.models[0].model
    assert getattr(inner, "skip", None) is not None and getattr(inner, "revin", False), "needs LVN + residual"
    cap = {"x": [], "skip": [], "y": []}
    h_in = inner.register_forward_pre_hook(lambda m, inp: cap["x"].append(inp[0].detach().float().cpu()))
    h_sk = inner.skip.register_forward_hook(lambda m, i, o: cap["skip"].append(o.detach().transpose(1, 2).float().cpu()))
    h_out = inner.register_forward_hook(lambda m, i, o: cap["y"].append(o.detach().float().cpu()))
    try:
        Y_df = load_dataset(dataset)
        val_size, test_size = SPLITS[dataset]
        nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None,
                            refit=False, use_fitted=True)
    finally:
        h_in.remove(); h_sk.remove(); h_out.remove()
    x = torch.cat(cap["x"]); skip = torch.cat(cap["skip"]); y = torch.cat(cap["y"])
    gamma = inner.revin_weight.detach().float().cpu(); delta = inner.revin_bias.detach().float().cpu()
    eps = inner.revin_eps
    assert inner.revin_mode == "last_only", inner.revin_mode
    x_last = x[:, -1:, :]
    y_lin = (skip - delta) / (gamma + eps * eps) + x_last            # exactly the model's inverse LVN of Skip alone
    y_cell = y - y_lin                                                 # = G(x~)/gamma by linearity
    uids = sorted(Y_df["unique_id"].unique())                          # NeuralForecast channel order
    return dict(x=x, y=y, y_lin=y_lin, y_cell=y_cell, uids=uids, Y_df=Y_df)


def truth_for(cap, w: int, c: int, H: int):
    """Ground truth of window w, channel c: locate the input segment in the series and take the next H values."""
    s = cap["Y_df"][cap["Y_df"]["unique_id"] == cap["uids"][c]].sort_values("ds")["y"].to_numpy(np.float64)
    seg = cap["x"][w, :, c].numpy().astype(np.float64)
    L = seg.size
    # exact match of the lookback segment (inputs are the globally standardized series, no pipeline scaler)
    cand = np.flatnonzero(np.isclose(s[:len(s) - L - H + 1], seg[0]))
    for i in cand:
        if np.allclose(s[i:i + L], seg, atol=1e-5):
            return s[i + L:i + L + H]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["ETTh2:720", "Weather:720"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-root", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints")
    ap.add_argument("--ckpt-fallback", type=Path,
                    default=Path(__file__).resolve().parents[1] / "_overlay_preds" / "checkpoints")
    args = ap.parse_args()
    out_dirs = [Path(FIGURES_DIR)] + ([Path(os.environ["FIG_OUT_DIR"])] if os.environ.get("FIG_OUT_DIR") else [])

    plotting.apply_style()
    fig, axes = plt.subplots(2, len(args.cells), figsize=(plotting.FULL_W, 3.4), squeeze=False,
                             sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.12))
    for col, cell in enumerate(args.cells):
        ax, ax2 = axes[0][col], axes[1][col]
        ds, H = cell.split(":"); H = int(H)
        name = f"CeNN_{CENN_MAIN_VARIANT}__{ds}__H{H}__seed{args.seed}"
        ckpt = args.ckpt_root / name
        if not ckpt.exists():
            ckpt = args.ckpt_fallback / name
        if not ckpt.exists():
            print(f"[skip {cell}: no checkpoint {name}]"); ax.set_visible(False); ax2.set_visible(False); continue
        cap = capture(ckpt, ds)
        target = TARGET_BY_DS.get(ds, TARGET)
        c = cap["uids"].index(target) if target in cap["uids"] else 0
        W = cap["y"].shape[0]
        # energy split over all test windows and channels, of the deviation from the last value
        dev = cap["y"] - cap["x"][:, -1:, :]
        lin = cap["y_lin"] - cap["x"][:, -1:, :]; cel = cap["y_cell"]
        e = float((dev ** 2).sum()); e_l = float((lin ** 2).sum()); e_c = float((cel ** 2).sum())
        e_x = 2 * float((lin * cel).sum())
        print(f"{ds} H{H} seed{args.seed}: {W} windows; energy of (forecast - last value): "
              f"linear {100 * e_l / e:.1f}%, cellular {100 * e_c / e:.1f}%, cross {100 * e_x / e:.1f}%")
        # representative window on the target channel: median MSE of the full forecast
        errs = np.full(W, np.nan); truths = {}
        for w in range(W):
            t = truth_for(cap, w, c, H)
            if t is not None:
                truths[w] = t; errs[w] = float(np.mean((cap["y"][w, :, c].numpy() - t) ** 2))
        ok = np.flatnonzero(~np.isnan(errs))
        w = int(ok[np.argmin(np.abs(errs[ok] - np.nanmedian(errs)))])
        t = np.arange(1, H + 1)
        ax.plot(t, truths[w], color="0.45", lw=0.8, alpha=0.8, label="Ground truth", zorder=2)
        ax.plot(t, cap["y_lin"][w, :, c], color="#0072B2", lw=1.1, ls="--", label="Linear path", zorder=3)
        ax.plot(t, cap["y"][w, :, c], color=plotting.CENN_C, lw=1.5, label=f"{CENN_DISPLAY_NAME} forecast", zorder=4)
        ax.fill_between(t, cap["y_lin"][w, :, c], cap["y"][w, :, c], color=plotting.CENN_C, alpha=0.18,
                        lw=0, label="Cellular correction", zorder=1)
        ax.set_title(f"{ds}, {cap['uids'][c]}, H={H}")
        ax.margins(x=0.01)
        ax2.axhline(0, color="0.6", lw=0.6)
        ax2.plot(t, cap["y_cell"][w, :, c], color=plotting.CENN_C, lw=1.0)
        ax2.fill_between(t, 0, cap["y_cell"][w, :, c], color=plotting.CENN_C, alpha=0.25, lw=0)
        ax2.set_xlabel("Forecast step"); ax2.margins(x=0.01)
        if col == 0:
            ax2.set_ylabel("Cellular correction", fontsize=7.6)
        mse_full = errs[w]; mse_lin = float(np.mean((cap["y_lin"][w, :, c].numpy() - truths[w]) ** 2))
        bx, by, ha, va = _empty_corner(ax, [truths[w], cap["y_lin"][w, :, c].numpy(), cap["y"][w, :, c].numpy()])
        ax.text(bx, by, f"MSE on window\nfull {mse_full:.3f}\nlinear path {mse_lin:.3f}",
                transform=ax.transAxes, ha=ha, va=va, fontsize=6.2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    axes[0][0].set_ylabel("Value (standardized)")
    for col in range(len(args.cells)):
        if axes[0][col].get_visible() is False:
            axes[1][col].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=7.6,
               columnspacing=1.1, handlelength=1.5, frameon=True)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_forecast_decomposition.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_forecast_decomposition.png", dpi=200, bbox_inches="tight")
        print("wrote", d / "fig_forecast_decomposition.pdf")


if __name__ == "__main__":
    main()
