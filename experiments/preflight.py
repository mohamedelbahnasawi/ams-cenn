#!/usr/bin/env python
"""Fail-free PREFLIGHT: validate every grid cell end-to-end before the H100 run.

For each (model, dataset, horizon) that the real campaign will run, this builds the
model and runs the FULL pipeline (fit + cross_validation eval) at max_steps=2 on the
REAL data at the REAL batch sizes — so precision errors, shape/dtype bugs, OOM, NaN
losses, and dataset-loading failures surface HERE (on the local GPU) instead of after
hours on the cluster. It uses a tiny eval window (test_size=3*h) and does NOT write to
experiments/results/ (no skip-if-exists pollution); status is logged to
experiments/preflight_status.json and the run is resumable.

What it catches: per-model build/precision (e.g. TimesNet bf16), per-dataset load/scaler/
shape, per-horizon patch-padding, training-step OOM at the real batch size, non-finite loss.
What it does NOT replace: the full-length training (accuracy) and the full stride-1 eval
throughput — those are validated by the real run; here we only prove every cell EXECUTES.

Run:
    .venv/Scripts/python.exe experiments/preflight.py            # full grid (MAIN 7x4 + ablation subset)
    .venv/Scripts/python.exe experiments/preflight.py --quick    # 1 dataset x 1 horizon per model
    .venv/Scripts/python.exe experiments/preflight.py --retry-failed
"""
import argparse, contextlib, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_float32_matmul_precision("medium")
import pandas as pd
from neuralforecast import NeuralForecast
from experiments.config import (DATASET_INFO, DATASETS_ALL, HORIZONS, EXPERIMENTS_DIR,
                                 CENN_VARIANTS_MAIN, CENN_VARIANTS_ABLATION,
                                 CENN_VARIANTS_APPENDIX, BASELINES_ALL)
import experiments.runner as R
from experiments.runner import load_dataset, build_cenn, BASELINE_BUILDERS

STATUS = EXPERIMENTS_DIR / "preflight_status.json"
ETT_SUBSET = ["ETTh1", "ETTh2"]          # cheap datasets for the ablation tier
ABL_HORIZONS = [96, 720]                  # shortest + longest (patch-padding edge)
CUDA = torch.cuda.is_available()


@contextlib.contextmanager
def _preflight_kwargs():
    """Scoped: validation runs at step 1 so early-stopping has its metric at max_steps=2
    (real runs use max_steps=1000 > val_check=50, so this is a preflight-only workaround,
    not a config change). Restored on exit."""
    orig = R._common_kwargs
    def patched(*a, **k):
        kw = orig(*a, **k); kw["val_check_steps"] = 1; return kw
    R._common_kwargs = patched
    try:
        yield
    finally:
        R._common_kwargs = orig


def build(model_name, h, n_series, dataset):
    if model_name.startswith("CeNN_"):
        m, alias, _ = build_cenn(model_name.removeprefix("CeNN_"), h, n_series, 1, 2, dataset_name=dataset)
    else:
        m, alias = BASELINE_BUILDERS[model_name](h, n_series, 1, 2, dataset_name=dataset)
    return m, alias


def validate(model_name, dataset, horizon):
    n_series = DATASET_INFO[dataset]["n_series"]; freq = DATASET_INFO[dataset]["freq"]
    Y = load_dataset(dataset)
    need = 512 + 4 * horizon + 64
    Y = pd.concat([g.tail(need) for _, g in Y.groupby("unique_id", sort=False)], ignore_index=True)
    with _preflight_kwargs():
        model, alias = build(model_name, horizon, n_series, dataset)
    nf = NeuralForecast(models=[model], freq=str(freq))
    cv = nf.cross_validation(df=Y, val_size=horizon, test_size=3 * horizon, n_windows=None, refit=False)
    if isinstance(cv.index, pd.MultiIndex):
        cv = cv.reset_index()
    mse = float(((cv[alias] - cv["y"]) ** 2).mean())
    if not (mse == mse and abs(mse) != float("inf")):   # NaN/inf guard
        raise ValueError(f"non-finite MSE={mse}")
    params = sum(p.numel() for p in model.parameters())
    prec = getattr(model, "trainer_kwargs", {}).get("precision", None)
    return {"mse": round(mse, 4), "params": params, "precision": prec}


def grid(quick):
    main_models = [f"CeNN_{v}" for v in CENN_VARIANTS_MAIN] + BASELINES_ALL
    abl_models = [f"CeNN_{v}" for v in CENN_VARIANTS_ABLATION + CENN_VARIANTS_APPENDIX]
    cells = []
    if quick:
        for m in main_models + abl_models:
            cells.append((m, "ETTh2", 96))
    else:
        for m in main_models:
            for d in DATASETS_ALL:
                for h in HORIZONS:
                    cells.append((m, d, h))
        for m in abl_models:
            for d in ETT_SUBSET:
                for h in ABL_HORIZONS:
                    cells.append((m, d, h))
    # de-dup preserving order
    seen = set(); out = []
    for c in cells:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    cells = grid(args.quick)
    print(f"device={torch.cuda.get_device_name(0) if CUDA else 'cpu'} | {len(cells)} cells | "
          f"max_steps=2, test_size=3*h (validation only)\n", flush=True)

    n_pass = n_fail = n_skip = 0
    for i, (m, d, h) in enumerate(cells):
        key = f"{m}__{d}__H{h}"
        prev = status.get(key)
        if prev and prev.get("status") == "PASS" and not args.retry_failed:
            n_skip += 1; continue
        if prev and prev.get("status") == "PASS" and args.retry_failed:
            n_skip += 1; continue
        t0 = time.time()
        try:
            info = validate(m, d, h)
            status[key] = {"status": "PASS", "t_s": round(time.time() - t0, 1), **info}
            n_pass += 1
            print(f"[{i+1}/{len(cells)}] PASS {key:46s} mse={info['mse']:.3f} "
                  f"params={info['params']} prec={info['precision']} ({status[key]['t_s']}s)", flush=True)
        except Exception as e:
            status[key] = {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:200]}"}
            n_fail += 1
            print(f"[{i+1}/{len(cells)}] FAIL {key:46s} {type(e).__name__}: {str(e)[:120]}", flush=True)
            traceback.print_exc()
        finally:
            import gc; gc.collect()
            if CUDA: torch.cuda.empty_cache()
        STATUS.write_text(json.dumps(status, indent=2))   # resumable after every cell

    print(f"\n{'='*70}\n  PREFLIGHT: {n_pass} PASS, {n_fail} FAIL, {n_skip} skipped (already PASS)")
    fails = [k for k, v in status.items() if v.get("status") == "FAIL"]
    if fails:
        print(f"  FAILED CELLS ({len(fails)}):")
        for k in fails:
            print(f"    {k}: {status[k]['error']}")
    print(f"  status -> {STATUS}\n{'='*70}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
