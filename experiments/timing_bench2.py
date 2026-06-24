#!/usr/bin/env python
"""Baseline timing sweep (companion to timing_bench.py).

Times every re-run baseline on ETTh2 (per-model ranking) plus the two potentially
heavy ones (xLSTM recurrent, TimesNet) on Traffic. Two-point extrapolation to
MAX_STEPS, early-stop/val disabled. Auto-fallback bf16 -> fp32 on dtype errors, and
reports which precision each model actually used (a real config finding for the runs).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_float32_matmul_precision("medium")
from neuralforecast import NeuralForecast
import experiments.runner as R
from experiments.config import SPLITS, DATASET_INFO, MAX_STEPS
from experiments.runner import load_dataset, BASELINE_BUILDERS

S1, S2 = 50, 150
H = 96
CELLS = [
    ("TiDE", "ETTh2"), ("NHITS", "ETTh2"), ("iTransformer", "ETTh2"),
    ("TSMixer", "ETTh2"), ("TimeMixer", "ETTh2"), ("PatchTST", "ETTh2"),
    ("TCN", "ETTh2"), ("xLSTM", "ETTh2"), ("TimesNet", "ETTh2"),
    ("xLSTM", "Traffic"), ("TimesNet", "Traffic"),
]


def run_once(name, dataset, max_steps, precision, Y, vs, ts, freq):
    R.PRECISION = precision  # _common_kwargs reads this at call time
    n_series = DATASET_INFO[dataset]["n_series"]
    model, _ = BASELINE_BUILDERS[name](H, n_series, seed=1, max_steps=max_steps, dataset_name=dataset)
    model.early_stop_patience_steps = -1
    model.val_check_steps = max_steps + 1
    nf = NeuralForecast(models=[model], freq=str(freq))
    t0 = time.time()
    nf.cross_validation(df=Y, val_size=vs, test_size=ts, n_windows=None, refit=False)
    return time.time() - t0, sum(p.numel() for p in model.parameters())


def time_cell(name, dataset):
    Y = load_dataset(dataset); vs, ts = SPLITS[dataset]; freq = DATASET_INFO[dataset]["freq"]
    prec = "bf16-mixed"
    try:
        t1, npar = run_once(name, dataset, S1, prec, Y, vs, ts, freq)
    except RuntimeError as e:
        if "BFloat16" in str(e) or "dtype" in str(e) or "Half" in str(e):
            prec = "32-true"
            t1, npar = run_once(name, dataset, S1, prec, Y, vs, ts, freq)
        else:
            raise
    t2, _ = run_once(name, dataset, S2, prec, Y, vs, ts, freq)
    per = max((t2 - t1) / (S2 - S1), 0.0)
    fixed = max(t1 - S1 * per, 0.0)
    return npar, t1, t2, per, fixed, fixed + MAX_STEPS * per, prec


def main():
    print(f"torch {torch.__version__} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"Baseline sweep: S1={S1} S2={S2} -> MAX_STEPS={MAX_STEPS}, H={H}\n", flush=True)
    print(f"{'cell':22s} {'params':>10s} {'prec':>5s} {'t@'+str(S1):>7s} {'t@'+str(S2):>7s} {'s/step':>7s} {'t_full':>13s}", flush=True)
    for name, dataset in CELLS:
        try:
            npar, t1, t2, per, fixed, tfull, prec = time_cell(name, dataset)
            pl = "bf16" if prec == "bf16-mixed" else "fp32"
            print(f"{name+'/'+dataset:22s} {npar:>10d} {pl:>5s} {t1:>7.1f} {t2:>7.1f} {per:>7.3f} "
                  f"{tfull:>9.1f}s ({tfull/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"{name+'/'+dataset:22s}  ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
