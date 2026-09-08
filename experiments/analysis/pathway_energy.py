"""Prediction-energy decomposition of the two AMS-CeNN pathways.

WHAT THIS MEASURES
------------------
The model forms its forecast as  y = trunk(x) + skip(x)  (model.py: `out = self._head(...)`
then `out = out + self.skip(...)`). This script captures both terms on real test windows from a
trained checkpoint -- no retraining -- and reports how the forecast's energy divides between them.

The decomposition is exact:

    ||y||^2 = ||skip||^2 + ||trunk||^2 + 2<skip, trunk>

THE CROSS-TERM IS THE POINT. The two pathways are NOT orthogonal, so a bare "the skip explains
X%" is meaningless: the interaction term can be large and negative (the trunk partly cancelling
the skip) or large and positive (the two reinforcing). Reporting shares without it would overstate
whichever path is listed first. All three terms are printed as shares of ||y||^2, and they sum to
100% by construction.

WHAT IT DOES NOT MEASURE
------------------------
Energy is not accuracy. A pathway can carry a large share of the output norm while contributing
little to (or actively harming) predictive skill. This complements the frozen-pathway experiment,
which measures accuracy causally; it does not replace it.

USAGE
    python -m experiments.analysis.pathway_energy \
        --model CeNN_C1C2-Skip-K2 --dataset ETTh2 --horizons 96 192 336 720 --seeds 1 42 123
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from experiments.config import SPLITS                      # noqa: E402
from experiments.runner import load_dataset                # noqa: E402
from neuralforecast import NeuralForecast                  # noqa: E402

CKPT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"


def decompose_cell(model_name: str, dataset: str, horizon: int, seed: int):
    """Return (skip_share, trunk_share, cross_share, n_windows) for one trained cell."""
    ckpt = CKPT_ROOT / f"{model_name}__{dataset}__H{horizon}__seed{seed}"
    if not ckpt.exists():
        return None
    nf = NeuralForecast.load(path=str(ckpt))
    # Checkpoints are saved with save_dataset=False, so load() leaves the in-session fit attrs
    # unset while use_fitted's pre-eval snapshot expects them to exist. The eval rebuilds them
    # from the passed df and nf is discarded afterwards, so None defaults are safe. (Same
    # workaround as experiments/run_robustness.py.)
    for _attr in ("dataset", "uids", "last_dates", "ds"):
        if not hasattr(nf, _attr):
            setattr(nf, _attr, None)
    m = nf.models[0]
    inner = m.model
    if getattr(inner, "skip", None) is None or getattr(inner, "stack", None) is None:
        raise RuntimeError(f"{model_name} has no two-path structure to decompose")

    cap: dict[str, list] = {"skip": [], "total": []}

    def grab_skip(_mod, _inp, out):
        # skip is Linear(L->H) applied to x.transpose(1,2); the model transposes back before
        # adding, so mirror that here to compare like with like.
        cap["skip"].append(out.detach().transpose(1, 2).float().cpu())

    def grab_total(_mod, _inp, out):
        cap["total"].append(out.detach().float().cpu())

    h1 = inner.skip.register_forward_hook(grab_skip)
    h2 = inner.register_forward_hook(grab_total)
    try:
        Y_df = load_dataset(dataset)
        val_size, test_size = SPLITS[dataset]
        nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size,
                            n_windows=None, refit=False, use_fitted=True)
    finally:
        h1.remove()
        h2.remove()

    if not cap["skip"] or not cap["total"]:
        raise RuntimeError("hooks captured nothing -- forward path changed?")
    skip = torch.cat(cap["skip"])
    total = torch.cat(cap["total"])
    if skip.shape != total.shape:
        raise RuntimeError(f"shape mismatch skip {tuple(skip.shape)} vs total {tuple(total.shape)}")
    trunk = total - skip                       # exact: forward is total = trunk + skip

    e_total = float((total ** 2).sum())
    e_skip = float((skip ** 2).sum())
    e_trunk = float((trunk ** 2).sum())
    e_cross = 2.0 * float((skip * trunk).sum())
    # identity check: the three parts must reconstruct the total
    resid = abs(e_skip + e_trunk + e_cross - e_total) / max(e_total, 1e-12)
    if resid > 1e-4:
        raise RuntimeError(f"decomposition identity violated (rel. residual {resid:.2e})")
    return (100 * e_skip / e_total, 100 * e_trunk / e_total,
            100 * e_cross / e_total, int(total.shape[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CeNN_AMS-Anc")
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 123])
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{args.model} on {args.dataset}: prediction-energy shares of ||y||^2\n")
    print(f"{'H':>5}{'seed':>6}{'skip %':>10}{'trunk %':>10}{'cross %':>10}{'windows':>10}")
    rows = []
    for h in args.horizons:
        for s in args.seeds:
            r = decompose_cell(args.model, args.dataset, h, s)
            if r is None:
                print(f"{h:>5}{s:>6}{'  (no checkpoint)':>30}")
                continue
            sk, tr, cr, n = r
            rows.append((h, s, sk, tr, cr))
            print(f"{h:>5}{s:>6}{sk:10.1f}{tr:10.1f}{cr:10.1f}{n:10d}")
    if rows:
        a = np.array([[r[2], r[3], r[4]] for r in rows])
        print(f"\n{'mean':>11}{a[:,0].mean():10.1f}{a[:,1].mean():10.1f}{a[:,2].mean():10.1f}")
        print("\nShares sum to 100% by construction. A large negative cross-term means the "
              "pathways partly\ncancel; a large positive one means they reinforce. Energy is not "
              "accuracy -- see the frozen-\npathway experiment for the causal contribution.")


if __name__ == "__main__":
    main()
