"""What the certified cellular path contributes, window by window, on every dataset.

By eq. (9) the forecast splits exactly into the linear path y_lin (the residual alone, passed through the
inverse LVN) and the cellular correction y_cell = y - y_lin.  For every test window we compute the MSE of
the full forecast and of the linear path alone against the truth (all channels of the window pooled) and
report the relative change  (MSE_full - MSE_lin) / MSE_lin  in percent: negative = the cellular path helps.
We also report the share of the energy of (y - x_L) carried by the cellular path per dataset.

One trained checkpoint per dataset (seed 1, H=96 by default), no retraining. Nothing is stored beyond
[W, H, V] outputs, so the high-cardinality datasets fit in memory.

Usage:  .venv/Scripts/python.exe experiments/analysis/plot_cell_contribution.py [--horizon 96] [--seed 1]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import SPLITS, FIGURES_DIR, CENN_MAIN_VARIANT, DATASETS_ALL  # noqa: E402
from experiments.runner import load_dataset                                        # noqa: E402
from experiments.analysis import plotting                                          # noqa: E402
from neuralforecast import NeuralForecast                                          # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoints"


def run_cell(dataset: str, H: int, seed: int):
    ckpt = CKPT / f"CeNN_{CENN_MAIN_VARIANT}__{dataset}__H{H}__seed{seed}"
    if not ckpt.exists():
        return None
    nf = NeuralForecast.load(path=str(ckpt))
    for a in ("dataset", "uids", "last_dates", "ds"):
        if not hasattr(nf, a):
            setattr(nf, a, None)
    inner = nf.models[0].model
    assert inner.revin_mode == "last_only" and getattr(inner, "skip", None) is not None
    cap = {"x_last": [], "x0": [], "skip": [], "y": []}
    def _grab_in(m, inp):
        cap["x_last"].append(inp[0][:, -1:, :].detach().float().cpu())
        cap["x0"].append(inp[0][:, :, 0].detach().float().cpu())
    h_in = inner.register_forward_pre_hook(_grab_in)
    h_sk = inner.skip.register_forward_hook(lambda m, i, o: cap["skip"].append(o.detach().transpose(1, 2).float().cpu()))
    h_out = inner.register_forward_hook(lambda m, i, o: cap["y"].append(o.detach().float().cpu()))
    Y_df = load_dataset(dataset)
    val_size, test_size = SPLITS[dataset]
    try:
        nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None,
                            refit=False, use_fitted=True)
    finally:
        h_in.remove(); h_sk.remove(); h_out.remove()
    x_last = torch.cat(cap["x_last"]); x0 = torch.cat(cap["x0"]); skip = torch.cat(cap["skip"]); y = torch.cat(cap["y"])
    gamma = inner.revin_weight.detach().float().cpu(); delta = inner.revin_bias.detach().float().cpu(); eps = inner.revin_eps
    y_lin = (skip - delta) / (gamma + eps * eps) + x_last
    y_cell = y - y_lin
    del skip
    W, Hh, V = y.shape
    L = x0.shape[1]
    # ground truth by index arithmetic, verified against the captured input of channel 0; streamed per
    # window so that the high-cardinality datasets never hold a [W, H, V] float64 truth array
    uids = sorted(Y_df["unique_id"].unique())
    series = np.stack([Y_df[Y_df["unique_id"] == u].sort_values("ds")["y"].to_numpy(np.float32) for u in uids], axis=1)  # [T, V]
    T = series.shape[0]
    start0 = T - test_size - L
    y_np = y.numpy(); ylin_np = y_lin.numpy(); x0_np = x0.numpy()
    mse_full = np.empty(W); mse_lin = np.empty(W); bad = 0
    for w in range(W):
        s = start0 + w
        if not np.allclose(series[s:s + L, 0], x0_np[w], atol=1e-4):
            bad += 1
        tr = series[s + L:s + L + Hh]
        mse_full[w] = float(np.mean((y_np[w] - tr) ** 2))
        mse_lin[w] = float(np.mean((ylin_np[w] - tr) ** 2))
    if bad:
        raise RuntimeError(f"{dataset}: {bad}/{W} windows failed the alignment check")
    rel = 100 * (mse_full - mse_lin) / mse_lin
    e = float(((y - x_last) ** 2).sum()); e_c = float((y_cell ** 2).sum()); e_l = float(((y_lin - x_last) ** 2).sum())
    return dict(rel=rel, share_cell=100 * e_c / e, share_lin=100 * e_l / e, W=W,
                mean_full=float(mse_full.mean()), mean_lin=float(mse_lin.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS_ALL))
    args = ap.parse_args()
    out_dirs = [Path(FIGURES_DIR)] + ([Path(os.environ["FIG_OUT_DIR"])] if os.environ.get("FIG_OUT_DIR") else [])
    res = {}
    cache = Path(FIGURES_DIR).parent / "cell_contribution"
    cache.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        f = cache / f"{ds}_H{args.horizon}_seed{args.seed}.npz"
        if f.exists():
            d = np.load(f)
            r = dict(rel=d["rel"], share_cell=float(d["share_cell"]), share_lin=float(d["share_lin"]), W=int(d["W"]),
                     mean_full=float(d["mean_full"]), mean_lin=float(d["mean_lin"]))
        else:
            r = run_cell(ds, args.horizon, args.seed)
            if r is None:
                print(f"[skip {ds}: no checkpoint]"); continue
            np.savez(f, **r)
        res[ds] = r
        print(f"{ds} H{args.horizon} seed{args.seed}: {r['W']} windows; mean window MSE full {r['mean_full']:.4f} "
              f"vs linear path {r['mean_lin']:.4f} ({100 * (r['mean_full'] / r['mean_lin'] - 1):+.2f}%); "
              f"cell helps on {100 * float((r['rel'] < 0).mean()):.0f}% of windows; median change {np.median(r['rel']):+.2f}%; "
              f"energy share cellular {r['share_cell']:.1f}%, linear {r['share_lin']:.1f}%")
    names = list(res)
    plotting.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(plotting.FULL_W, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(names):
        r = res[ds]["rel"]
        ax.boxplot([r], positions=[i], vert=False, widths=0.55, showfliers=False, whis=(5, 95), patch_artist=True,
                   boxprops=dict(facecolor=plotting.CENN_C, alpha=0.30, edgecolor=plotting.CENN_C, lw=0.9),
                   whiskerprops=dict(color=plotting.CENN_C, lw=0.8), capprops=dict(color=plotting.CENN_C, lw=0.8),
                   medianprops=dict(color=plotting.CENN_C, lw=1.6))
        sub = r if r.size <= 400 else rng.choice(r, 400, replace=False)
        ax.scatter(sub, i + rng.uniform(-0.18, 0.18, sub.size), s=4, color="0.35", alpha=0.35, linewidths=0, zorder=2)
        ax.scatter([100 * (res[ds]["mean_full"] / res[ds]["mean_lin"] - 1)], [i], marker="D", s=24,
                   facecolor="white", edgecolor="black", lw=0.8, zorder=4)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Change in window MSE from the cellular correction (%; negative = it helps)", fontsize=8.5)
    ax.set_xlim(-30, 30)
    ax.scatter([], [], marker="D", s=24, facecolor="white", edgecolor="black", lw=0.8, label="dataset mean")
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    ax.grid(axis="y", visible=False)
    ax2.barh(range(len(names)), [res[d]["share_cell"] for d in names], color=plotting.CENN_C, alpha=0.85, height=0.6)
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels([]); ax2.invert_yaxis()
    ax2.set_xlabel("Energy share of the cellular\ncorrection (%)", fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    for i, d in enumerate(names):
        ax2.text(res[d]["share_cell"] + 0.2, i, f"{res[d]['share_cell']:.1f}", va="center", fontsize=7.2)
    ax2.set_xlim(0, max(res[d]["share_cell"] for d in names) * 1.35)
    fig.tight_layout(w_pad=1.5)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_cell_contribution.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_cell_contribution.png", dpi=200, bbox_inches="tight")
        print("wrote", d / "fig_cell_contribution.pdf")


if __name__ == "__main__":
    main()
