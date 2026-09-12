"""Is there any nonlinear structure left after the anchored linear fit?

Motivation: under the anchored protocol skip-only (== NLinear) matches or beats the full
AMS-CeNN on ETT+Weather. If the linear residual is white, NO nonlinear pathway can add point
accuracy on these benchmarks by construction, and the trunk's small cost is a property of the
data, not of the design. Two complementary tests on the skip-only residuals of the TEST windows:

  1. BDS test (Brock-Dechert-Scheinkman) on the one-step-ahead residual series per channel:
     rejects iid at the 5% level if there is ANY remaining dependence (linear or nonlinear).
  2. Predictability probe: from a window of past residuals, predict the residual at lead h with
     (a) ridge regression (linear) and (b) a small MLP (nonlinear), out-of-sample R^2 on a
     chronological split. R^2 ~ 0 for both  -> nothing left to model.
     Linear ~ nonlinear > 0               -> more LINEAR capacity would help.
     Nonlinear > linear                   -> a nonlinear pathway COULD help.

Data source: the canonical SkipOnly-Anc (and AMS-Anc) checkpoints, clean stride-1 test windows,
exactly as the robustness harness evaluates them. Residual r_t = y_t - yhat_t.
Usage: python -m experiments.analysis.residual_nonlinearity --models CeNN_SkipOnly-Anc \
           --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather --horizon 96 --seed 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import SPLITS                              # noqa: E402
from experiments.runner import load_dataset                        # noqa: E402
from experiments.run_robustness import _cv                                 # noqa: E402
from neuralforecast import NeuralForecast                          # noqa: E402

OUT = Path(__file__).resolve().parent / "residual_nonlinearity_results.json"


def residual_series(model, dataset, horizon, seed, ckpt_root):
    """Return {uid: {'h1': r(t) one-step residual series, 'hH': r at lead=horizon}} on test windows."""
    ckpt = Path(ckpt_root) / f"{model}__{dataset}__H{horizon}__seed{seed}"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    val_size, test_size = SPLITS[dataset]
    Y = load_dataset(dataset)
    nf = NeuralForecast.load(str(ckpt))
    for a in ("dataset", "uids", "last_dates", "ds"):
        if not hasattr(nf, a):
            setattr(nf, a, None)
    alias = nf.models[0].alias
    cv = _cv(nf, Y, val_size, test_size)
    cv["lead"] = cv.groupby(["unique_id", "cutoff"]).cumcount() + 1
    cv["r"] = cv["y"] - cv[alias]
    out = {}
    for uid, g in cv.groupby("unique_id"):
        g1 = g[g["lead"] == 1].sort_values("cutoff")
        gH = g[g["lead"] == horizon].sort_values("cutoff")
        out[uid] = {"h1": g1["r"].to_numpy(np.float64), "hH": gH["r"].to_numpy(np.float64)}
    return out


def bds_pvalues(series_by_uid, max_dim=3):
    from statsmodels.tsa.stattools import bds
    ps = []
    for uid, s in series_by_uid.items():
        x = s["h1"]
        x = x[np.isfinite(x)]
        if len(x) < 200:
            continue
        try:
            _, p = bds(x, max_dim=max_dim)
            ps.append(float(np.min(np.atleast_1d(p))))
        except Exception:
            continue
    return ps


def predictability(series_by_uid, lookback=48, lead_key="h1", max_series=64, seed=0):
    """Out-of-sample R^2 of ridge (linear) vs MLP (nonlinear) predicting the residual at the
    lead from its own past `lookback` residuals; pooled over channels, chronological 70/30 split."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    uids = list(series_by_uid)
    if len(uids) > max_series:
        uids = list(rng.choice(uids, size=max_series, replace=False))
    Xtr, ytr, Xte, yte = [], [], [], []
    for uid in uids:
        r1 = series_by_uid[uid]["h1"]
        rl = series_by_uid[uid][lead_key]
        n = min(len(r1), len(rl))
        r1, rl = r1[:n], rl[:n]
        if n < lookback + 100:
            continue
        # window t-lookback..t-1 of one-step residuals -> residual at the lead ending at cutoff t
        X = np.stack([r1[i - lookback:i] for i in range(lookback, n)])
        y = rl[lookback:n]
        cut = int(0.7 * len(y))
        Xtr.append(X[:cut]); ytr.append(y[:cut]); Xte.append(X[cut:]); yte.append(y[cut:])
    if not Xtr:
        return None
    Xtr, ytr, Xte, yte = map(np.concatenate, (Xtr, ytr, Xte, yte))
    sd = ytr.std() + 1e-12
    Xtr, Xte, ytr_s, yte_s = Xtr / sd, Xte / sd, ytr / sd, yte / sd
    lin = Ridge(alpha=1.0).fit(Xtr, ytr_s)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True,
                       random_state=seed).fit(Xtr, ytr_s)
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
            "r2_linear": float(r2_score(yte_s, lin.predict(Xte))),
            "r2_mlp": float(r2_score(yte_s, mlp.predict(Xte)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="CeNN_SkipOnly-Anc")
    ap.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-root", default="experiments/checkpoints")
    a = ap.parse_args()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model in a.models.split(","):
        for ds in a.datasets.split(","):
            t0 = time.time()
            key = f"{model}|{ds}|H{a.horizon}|seed{a.seed}"
            try:
                ser = residual_series(model, ds, a.horizon, a.seed, a.ckpt_root)
            except FileNotFoundError as e:
                print(f"[skip] {key}: {e}", flush=True); continue
            ps = bds_pvalues(ser)
            rec = {"bds_n_channels": len(ps),
                   "bds_frac_reject_5pct": float(np.mean([p < 0.05 for p in ps])) if ps else None,
                   "bds_median_p": float(np.median(ps)) if ps else None,
                   "pred_h1": predictability(ser, lead_key="h1"),
                   "pred_hH": predictability(ser, lead_key="hH")}
            results[key] = rec
            OUT.write_text(json.dumps(results, indent=2))
            p1, pH = rec["pred_h1"], rec["pred_hH"]
            print(f"{key}: BDS reject {rec['bds_frac_reject_5pct']} (median p {rec['bds_median_p']}) | "
                  f"R2 h1 lin {p1['r2_linear']:+.3f} mlp {p1['r2_mlp']:+.3f} | "
                  f"R2 h{a.horizon} lin {pH['r2_linear']:+.3f} mlp {pH['r2_mlp']:+.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
