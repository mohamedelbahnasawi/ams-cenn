"""Robustness stress-test driver (Paper A): test-time input-perturbation robustness.

For each (model, dataset, horizon, seed):
  1. load the CANONICAL trained checkpoint (read-only) — NO retraining;
  2. run a CLEAN use_fitted cross_validation -> clean targets + clean MSE (the denominator);
  3. for every perturbation condition, run the model on a perturbed-input df and score its
     predictions against the CLEAN targets (join on window keys) -> perturbed MSE -> ratio.

Perturbation is df-level, pre-scaler (global z-score domain) so it is identical and fair across
all models. Results are written one atomic JSON per (cell, condition) under <out-root>/<cond>/results/,
skip-if-exists (resume-safe). This driver is self-pathing: it does NOT depend on CENN_EXP_DIR.

Usage:
  python -m experiments.run_robustness \
    --models CeNN_C1C2-Skip-K2,DLinear,TSMixer,PatchTST,S4D,TCN \
    --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather --horizons 96,336,720 --seeds 1,42,123 \
    --ckpt-root experiments/checkpoints --out-root experiments/_robustness
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# repo-root-relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.config import SPLITS, INPUT_SIZE          # noqa: E402
from experiments.runner import load_dataset                # noqa: E402
from experiments.perturb import perturb_df                 # noqa: E402
from neuralforecast import NeuralForecast                  # noqa: E402

# Perturbation sweep (level 0 = clean, computed once per cell). Mid-high levels per the
# pre-registered robustness criterion are the decisive ones (gauss>=0.5, spike>=0.03, mask>=48, scale>=0.25, shift>=1.0).
SWEEP = [
    ("gauss", 0.1), ("gauss", 0.25), ("gauss", 0.5), ("gauss", 1.0),
    ("spike", 0.01), ("spike", 0.03), ("spike", 0.05),
    ("mask", 16), ("mask", 48), ("mask", 96),
    ("scale", 0.1), ("scale", 0.25), ("scale", 0.5),
    ("shift", 0.5), ("shift", 1.0), ("shift", 2.0),
]
KEYS = ["unique_id", "ds", "cutoff"]


def _atomic_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _cv(nf, df, val_size, test_size):
    cv = nf.cross_validation(df=df, val_size=val_size, test_size=test_size,
                             n_windows=None, refit=False, use_fitted=True)
    if isinstance(cv.index, pd.MultiIndex):
        cv = cv.reset_index()
    elif cv.index.name is not None or "unique_id" not in cv.columns:
        cv = cv.reset_index()
    return cv


def run_cell(model, dataset, horizon, seed, ckpt_root: Path, out_root: Path):
    cond_name = lambda k, lv: f"{k}_{lv}"
    # skip-if-exists: all conditions already present?
    all_paths = {("clean", 0): out_root / "clean" / "results" / f"{model}__{dataset}__H{horizon}__seed{seed}.json"}
    for k, lv in SWEEP:
        all_paths[(k, lv)] = out_root / cond_name(k, lv) / "results" / f"{model}__{dataset}__H{horizon}__seed{seed}.json"
    if all(p.exists() for p in all_paths.values()):
        return "skip-all"

    ckpt_dir = ckpt_root / f"{model}__{dataset}__H{horizon}__seed{seed}"
    if not ckpt_dir.exists():
        return f"MISSING-CKPT {ckpt_dir.name}"

    val_size, test_size = SPLITS[dataset]
    Y = load_dataset(dataset)

    nf = NeuralForecast.load(str(ckpt_dir))
    # Checkpoints are saved with save_dataset=False, so load() does not set these in-session
    # fit attrs; use_fitted's pre-eval snapshot needs them to EXIST. The eval rebuilds them from
    # the passed df via _prepare_fit, and we discard nf afterward, so None defaults are safe.
    for _attr in ("dataset", "uids", "last_dates", "ds"):
        if not hasattr(nf, _attr):
            setattr(nf, _attr, None)
    alias = nf.models[0].alias

    # clean pass -> targets + denominator
    cv_clean = _cv(nf, Y, val_size, test_size)
    clean = cv_clean[KEYS + ["y"]].copy()
    err = cv_clean[alias].to_numpy() - cv_clean["y"].to_numpy()
    mse_clean = float(np.mean(err ** 2)); mae_clean = float(np.mean(np.abs(err)))
    prov = {"python": sys.version.split()[0], "ckpt": ckpt_dir.name}
    _atomic_write(all_paths[("clean", 0)], {
        "model": model, "dataset": dataset, "horizon": horizon, "seed": seed,
        "kind": "clean", "level": 0, "mse": mse_clean, "mae": mae_clean,
        "mse_clean": mse_clean, "mae_clean": mae_clean, "ratio_mse": 1.0,
        "alias": alias, "source": "robustness", "provenance": prov,
    })

    n_done = 1
    for k, lv in SWEEP:
        p = all_paths[(k, lv)]
        if p.exists():
            n_done += 1
            continue
        Y_pert = perturb_df(Y, kind=k, level=lv, seed=seed, test_size=test_size, input_size=INPUT_SIZE)
        cv_p = _cv(nf, Y_pert, val_size, test_size)
        # score perturbed predictions vs CLEAN targets (join on window keys)
        merged = cv_p[KEYS + [alias]].merge(clean, on=KEYS, how="inner")
        e = merged[alias].to_numpy() - merged["y"].to_numpy()
        mse_p = float(np.mean(e ** 2)); mae_p = float(np.mean(np.abs(e)))
        _atomic_write(p, {
            "model": model, "dataset": dataset, "horizon": horizon, "seed": seed,
            "kind": k, "level": lv, "mse": mse_p, "mae": mae_p,
            "mse_clean": mse_clean, "mae_clean": mae_clean,
            "ratio_mse": (mse_p / mse_clean) if mse_clean > 0 else float("nan"),
            "n_rows": int(len(merged)), "alias": alias, "source": "robustness", "provenance": prov,
        })
        n_done += 1
    return f"ok ({n_done}/{len(SWEEP)+1})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--horizons", default="96,336,720")
    ap.add_argument("--seeds", default="1,42,123")
    ap.add_argument("--ckpt-root", default="experiments/checkpoints")
    ap.add_argument("--out-root", default="experiments/_robustness")
    a = ap.parse_args()
    models = a.models.split(","); datasets = a.datasets.split(",")
    horizons = [int(x) for x in a.horizons.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    ckpt_root = Path(a.ckpt_root); out_root = Path(a.out_root)

    total = len(models) * len(datasets) * len(horizons) * len(seeds)
    i = 0; missing = []
    for ds in datasets:
        for m in models:
            for h in horizons:
                for s in seeds:
                    i += 1
                    t0 = time.time()
                    status = run_cell(m, ds, h, s, ckpt_root, out_root)
                    if status.startswith("MISSING"):
                        missing.append(f"{m}__{ds}__H{h}__seed{s}")
                    print(f"[{i}/{total}] {m} {ds} H{h} s{s}: {status}  ({time.time()-t0:.1f}s)", flush=True)
    if missing:
        print(f"\n=== {len(missing)} MISSING CHECKPOINTS ===")
        for x in missing:
            print("  ", x)
    print("\nR8 DRIVER DONE")


if __name__ == "__main__":
    main()
