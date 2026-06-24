#!/usr/bin/env python
"""Two-point timing benchmark -> evidence-based GPU-hour estimate for the full grid.

For each (model, dataset, H=96) cell we time nf.cross_validation at two step counts
(S1, S2) with early-stop/validation DISABLED so it runs exactly that many steps. Then:
    per_step = (t2 - t1) / (S2 - S1)          # marginal training cost / step
    fixed    = t1 - S1 * per_step             # data load + full-test eval + setup (step-independent)
    t_full   = fixed + MAX_STEPS * per_step    # estimated real run time (1000 steps)
This separates the step-independent eval cost (large for Traffic/ECL) from training.
Cells are ordered cheap -> expensive; results print incrementally.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_float32_matmul_precision("medium")
from neuralforecast import NeuralForecast
from experiments.config import SPLITS, DATASET_INFO, MAX_STEPS
from experiments.runner import load_dataset, build_cenn, BASELINE_BUILDERS

S1, S2 = 50, 150               # two step counts for the slope
H = 96

# (label, kind, name, dataset) — kind in {"cenn","baseline"}; cheap -> expensive
CELLS = [
    ("DLinear/ETTh2",      "baseline", "DLinear",       "ETTh2"),
    ("C1/ETTh2",           "cenn",     "C1-BoundedTau", "ETTh2"),
    ("CeNN-Full/ETTh2",    "cenn",     "CeNN-Full",     "ETTh2"),
    ("TimesNet/ETTh2",     "baseline", "TimesNet",      "ETTh2"),
    ("C1/Weather",         "cenn",     "C1-BoundedTau", "Weather"),
    ("C1/Electricity",     "cenn",     "C1-BoundedTau", "Electricity"),
    ("DLinear/Traffic",    "baseline", "DLinear",       "Traffic"),
    ("C1/Traffic",         "cenn",     "C1-BoundedTau", "Traffic"),
    ("CeNN-Full/Traffic",  "cenn",     "CeNN-Full",     "Traffic"),
    ("TimesNet/Traffic",   "baseline", "TimesNet",      "Traffic"),
]


def build(kind, name, dataset, max_steps):
    n_series = DATASET_INFO[dataset]["n_series"]
    if kind == "cenn":
        model, alias, _ = build_cenn(name, H, n_series, seed=1, max_steps=max_steps, dataset_name=dataset)
    else:
        model, alias = BASELINE_BUILDERS[name](H, n_series, seed=1, max_steps=max_steps, dataset_name=dataset)
    # Disable early-stop/validation so the run executes EXACTLY max_steps (clean timing).
    model.early_stop_patience_steps = -1
    model.val_check_steps = max_steps + 1
    return model, alias


def time_run(kind, name, dataset, max_steps, Y, val_size, test_size, freq):
    model, alias = build(kind, name, dataset, max_steps)
    nf = NeuralForecast(models=[model], freq=str(freq))
    t0 = time.time()
    nf.cross_validation(df=Y, val_size=val_size, test_size=test_size, n_windows=None, refit=False)
    dt = time.time() - t0
    npar = sum(p.numel() for p in model.parameters())
    return dt, npar


def main():
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"Two-point timing: S1={S1}, S2={S2}, extrapolate to MAX_STEPS={MAX_STEPS}, H={H}\n", flush=True)

    # CUDA warmup so the first real cell isn't charged one-time init.
    try:
        Yw = load_dataset("ETTh2"); vs, ts = SPLITS["ETTh2"]
        time_run("cenn", "C1-BoundedTau", "ETTh2", 10, Yw, vs, ts, DATASET_INFO["ETTh2"]["freq"])
    except Exception as e:
        print(f"[warmup skipped: {e}]", flush=True)

    print(f"{'cell':22s} {'params':>9s} {'t@'+str(S1):>8s} {'t@'+str(S2):>8s} "
          f"{'s/step':>8s} {'fixed':>8s} {'t_full(1000)':>13s}", flush=True)
    results = {}
    for label, kind, name, dataset in CELLS:
        try:
            Y = load_dataset(dataset)
            vs, ts = SPLITS[dataset]
            freq = DATASET_INFO[dataset]["freq"]
            t1, npar = time_run(kind, name, dataset, S1, Y, vs, ts, freq)
            t2, _    = time_run(kind, name, dataset, S2, Y, vs, ts, freq)
            per_step = max((t2 - t1) / (S2 - S1), 0.0)
            fixed = max(t1 - S1 * per_step, 0.0)
            t_full = fixed + MAX_STEPS * per_step
            results[label] = t_full
            print(f"{label:22s} {npar:>9d} {t1:>8.1f} {t2:>8.1f} "
                  f"{per_step:>8.3f} {fixed:>8.1f} {t_full:>11.1f}s ({t_full/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"{label:22s}  ERROR {type(e).__name__}: {e}", flush=True)
    print("\nDONE. t_full = estimated wall-clock for one real run (1000 steps) on THIS GPU.", flush=True)


if __name__ == "__main__":
    main()
