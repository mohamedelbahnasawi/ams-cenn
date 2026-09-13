#!/usr/bin/env python
"""Efficiency profiler: MACs/FLOPs, params, latency, peak memory.

Convention: metrics describe one full multivariate forecast (all V series, h steps, from one
L-window), the standard LTSF efficiency axis. Channel-independent baselines pay the V x they
really cost.

FLOP counting uses torch.utils.flop_counter.FlopCounterMode, which counts aten.convolution
(including functional conv1d, used in the CeNN K-loop) and matmul uniformly across models.
ptflops misses functional conv1d; torch.profiler(with_flops) reports 0 for conv1d and FFT but
counts elementwise ops. FlopCounterMode does not count FFT (TimesNet period detection),
softmax, layernorm or elementwise ops, so the field is matmul+conv only. flops are FLOPs
(2 x MACs for mm/conv); macs = flops/2.

Per (model, dataset, horizon) writes one JSON to experiments/efficiency/. Seed-independent.
A fixed windows_batch_size (WBS) is used so the peak-training-memory comparison is fair
(models otherwise default to different wbs: CeNN/DLinear/PatchTST/NHITS/TiDE=1024,
iTransformer=32, TimesNet=64, TCN=128, an 8 to 32x spread). peak_infer_mb is wbs-independent.

Run:
    .venv/Scripts/python.exe experiments/profile_efficiency.py --models CeNN_CeNN-Full DLinear --datasets ETTh2 --horizons 96
    .venv/Scripts/python.exe experiments/profile_efficiency.py --all
"""
import argparse, contextlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from torch.utils.flop_counter import FlopCounterMode
torch.set_float32_matmul_precision("medium")
from neuralforecast import NeuralForecast
from experiments.config import (DATASET_INFO, DATASETS_ALL, HORIZONS, INPUT_SIZE,
                                 EFFICIENCY_DIR, CENN_VARIANTS, BASELINES_ALL)
import experiments.runner as R
from experiments.runner import load_dataset, build_cenn, BASELINE_BUILDERS

EFF_DIR = EFFICIENCY_DIR
EFF_DIR.mkdir(parents=True, exist_ok=True)
CUDA = torch.cuda.is_available()
LAT_REPEATS = 20
WBS = 32  # fixed windows_batch_size so peak training memory is comparable across models
OPS_COVERAGE = "matmul+conv (FlopCounterMode); FFT/softmax/layernorm/elementwise NOT counted"


@contextlib.contextmanager
def _profiling_kwargs(windows_batch_size):
    """Scoped patch of runner._common_kwargs: disable early stopping (1-step fit, no val set)
    and pin windows_batch_size. Restored on exit.

    build_cenn also passes windows_batch_size via _CENN_FIXED, which would collide with the
    patched kwarg, so it is stripped from _CENN_FIXED for the scope as well."""
    orig = R._common_kwargs
    orig_fixed = R._CENN_FIXED
    def patched(*a, **k):
        kw = orig(*a, **k)
        kw["early_stop_patience_steps"] = -1
        kw["val_check_steps"] = 10 ** 9
        kw["windows_batch_size"] = windows_batch_size
        return kw
    R._common_kwargs = patched
    R._CENN_FIXED = {k: v for k, v in orig_fixed.items() if k != "windows_batch_size"}
    try:
        yield
    finally:
        R._common_kwargs = orig
        R._CENN_FIXED = orig_fixed


def _build(model_name, horizon, n_series, dataset):
    if model_name.startswith("CeNN_"):
        variant = model_name.removeprefix("CeNN_")
        model, alias, _ = build_cenn(variant, horizon, n_series, seed=1, max_steps=1, dataset_name=dataset)
    else:
        model, alias = BASELINE_BUILDERS[model_name](horizon, n_series, seed=1, max_steps=1, dataset_name=dataset)
    return model, alias


def _fit_slice(Y, horizon):
    need = INPUT_SIZE + horizon + 64
    import pandas as pd
    return pd.concat([g.tail(need) for _, g in Y.groupby("unique_id", sort=False)], ignore_index=True)


def profile_one(model_name, dataset, horizon):
    n_series = DATASET_INFO[dataset]["n_series"]
    freq = DATASET_INFO[dataset]["freq"]
    Y = _fit_slice(load_dataset(dataset), horizon)

    with _profiling_kwargs(WBS):
        model, alias = _build(model_name, horizon, n_series, dataset)
    nf = NeuralForecast(models=[model], freq=str(freq))
    params = sum(p.numel() for p in model.parameters())

    # --- peak training memory: 1 real train step at the fixed WBS ---
    if CUDA:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    nf.fit(df=Y)
    peak_train_mb = (torch.cuda.max_memory_allocated() / 1e6) if CUDA else None

    # --- MACs/FLOPs for one full multivariate forecast via FlopCounterMode ---
    nf.predict()  # warmup
    with torch.no_grad(), FlopCounterMode(display=False) as fcm:
        nf.predict()
    flops = fcm.get_total_flops()                 # matmul + conv (incl conv1d); FLOPs (=2*MACs)

    # --- peak inference memory (wbs-independent) ---
    if CUDA:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    nf.predict()
    peak_infer_mb = (torch.cuda.max_memory_allocated() / 1e6) if CUDA else None

    # --- latency: wall-clock (incl. pipeline overhead, not model latency) + compute-only ---
    if CUDA: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(LAT_REPEATS):
        nf.predict()
    if CUDA: torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / LAT_REPEATS * 1000.0

    compute_ms = None
    if CUDA:
        acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with torch.no_grad(), profile(activities=acts) as prof:
            nf.predict()
        # CUDA-device kernel entries only; summing all key_averages also counts the
        # CPU-dispatch mirrors and 'Activity Buffer Request'.
        dev_us = sum(getattr(e, "self_device_time_total", 0) for e in prof.key_averages()
                     if getattr(e, "device_type", None) == DeviceType.CUDA)
        compute_ms = dev_us / 1000.0

    return {
        "model": model_name, "dataset": dataset, "horizon": horizon,
        "input_size": INPUT_SIZE, "n_series": n_series,
        "windows_batch_size": WBS,
        "params": params,
        "flops_fwd": flops, "macs_fwd": flops / 2.0,   # matmul+conv only; see ops_coverage
        "ops_coverage": OPS_COVERAGE,
        "latency_ms": round(latency_ms, 3),
        "latency_ms_note": "wall-clock incl. NeuralForecast pipeline overhead; NOT model latency — use compute_ms",
        "compute_ms": round(compute_ms, 4) if compute_ms is not None else None,
        "peak_train_mb": round(peak_train_mb, 1) if peak_train_mb is not None else None,
        "peak_infer_mb": round(peak_infer_mb, 1) if peak_infer_mb is not None else None,
        "precision": getattr(model, "trainer_kwargs", {}).get("precision", None),
        "device": torch.cuda.get_device_name(0) if CUDA else "cpu",
        "convention": "A_full_multivariate_forecast",
    }


def eff_path(model_name, dataset, horizon):
    return EFF_DIR / f"{model_name}__{dataset}__H{horizon}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["ETTh2"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[96])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all:
        models = [f"CeNN_{v}" for v in CENN_VARIANTS] + BASELINES_ALL
        datasets, horizons = DATASETS_ALL, HORIZONS
    else:
        models = args.models or ["CeNN_CeNN-Full"]
        datasets, horizons = args.datasets, args.horizons

    print(f"device={torch.cuda.get_device_name(0) if CUDA else 'cpu'} | WBS={WBS} | conv-aware FLOPs (FlopCounterMode)\n")
    print(f"{'model':24s} {'dataset':12s} {'H':>4s} {'params':>9s} {'MACs':>10s} {'cmp_ms':>7s} {'trainMB':>8s} {'infMB':>7s}")
    for m in models:
        for ds in datasets:
            for h in horizons:
                out = eff_path(m, ds, h)
                if out.exists() and not args.overwrite:
                    continue
                try:
                    r = profile_one(m, ds, h)
                    tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(r, indent=2)); tmp.replace(out)
                    print(f"{m:24s} {ds:12s} {h:>4d} {r['params']:>9d} {r['macs_fwd']/1e6:>8.2f}M "
                          f"{str(r['compute_ms']):>7s} {str(r['peak_train_mb']):>8s} {str(r['peak_infer_mb']):>7s}", flush=True)
                except Exception as e:
                    print(f"{m:24s} {ds:12s} {h:>4d}  ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
    print("\nDONE.")


if __name__ == "__main__":
    main()
