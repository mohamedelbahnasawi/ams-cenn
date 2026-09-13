"""Gate-variation probe: does the bounded gate respond to input non-stationarity?

On the standard benchmarks the bounded-tau gate is near-constant (mean alpha ~0.90, temporal
std ~1e-4). This script generates two synthetic multivariate datasets:
  - SynthChirp: non-stationary (frequency and amplitude drift), so every L=512 lookback
                contains changing dynamics;
  - SynthStat:  stationary control (fixed frequency and amplitude),
trains C1C2-Skip-K2 on each through the runner, and reads the gate artifact that _save_tau
writes (last_tau; alpha = 1 - tau, shape [C, L] per cell). The temporal std of alpha across the
lookback measures whether the gate tracks the within-window non-stationarity.

Criterion, fixed before the runs: the gate counts as adaptive if alpha temporal std >= 0.05 on
SynthChirp and < 0.01 on SynthStat; it counts as flat if the SynthChirp std stays < 0.01.

Run (CENN_EXP_DIR is read at config import, so set it before python starts):
  CENN_EXP_DIR=experiments/_gate_probe python -m experiments.run_gate_probe
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T_TOTAL = 4000
VAL, TEST = 400, 800
V = 7                       # match ETT cardinality
HORIZON = 96
SEEDS = [1, 42, 123]
DATASETS = ["SynthChirp", "SynthStat"]


def _zscore_train(y: np.ndarray, n_train: int) -> np.ndarray:
    mu = y[:n_train].mean(); sd = y[:n_train].std() + 1e-8
    return (y - mu) / sd


def gen_synth(kind: str, seed: int) -> pd.DataFrame:
    """Return a long-format df (unique_id, ds, y), z-scored on the train portion."""
    rng = np.random.default_rng(seed * 7919 + (0 if kind == "SynthChirp" else 1))
    t = np.arange(T_TOTAL)
    n_train = T_TOTAL - VAL - TEST
    ds = pd.date_range("2020-01-01", periods=T_TOTAL, freq="h")
    rows = []
    f0 = 1.0 / 24.0                       # base period ~ 24 steps (daily)
    for i in range(V):
        phase0 = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.1, size=T_TOTAL)
        if kind == "SynthChirp":
            # frequency grows linearly (chirp): instantaneous f = f0*(1 + 3*t/T) -> triples by the end.
            k = 3.0 * f0 / T_TOTAL
            inst_phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t) + phase0
            amp = 1.0 + 1.5 * (t / T_TOTAL)          # amplitude drift 1.0 -> 2.5
            y = amp * np.sin(inst_phase) + noise
        else:  # SynthStat: fixed frequency + amplitude
            y = np.sin(2 * np.pi * f0 * t + phase0) + noise
        y = _zscore_train(y, n_train)
        for tt in range(T_TOTAL):
            rows.append((f"S{i}", ds[tt], float(y[tt])))
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])


def inject(name: str, df: pd.DataFrame):
    from experiments import config
    from experiments import runner
    config.DATASET_INFO[name] = {"group": name, "n_series": V, "freq": "h"}
    config.SPLITS[name] = (VAL, TEST)
    runner.DATASET_INFO[name] = config.DATASET_INFO[name]   # runner imported these by value
    runner.SPLITS[name] = config.SPLITS[name]
    runner._dataset_cache[name] = df


def analyze():
    """Read the tau artifacts and print the summary against the criterion."""
    from experiments import config
    tau_dir = config.ARTIFACTS_DIR / "tau"
    print("\n" + "=" * 64)
    print("Gate variation (alpha = 1 - tau; temporal std across L)")
    print("=" * 64)
    summary = {}
    for name in DATASETS:
        stds, means = [], []
        for f in sorted(tau_dir.glob(f"CeNN_C1C2-Skip-K2__{name}__*.npz")):
            d = np.load(f)
            for key in d.files:                      # one [C, L] array per cell
                tau = d[key]                         # [C, L]
                alpha = 1.0 - tau
                stds.append(alpha.std(axis=1))       # temporal std per channel -> [C]
                means.append(alpha.mean())
        if stds:
            allstd = np.concatenate(stds)
            summary[name] = dict(med_std=float(np.median(allstd)),
                                 max_std=float(np.max(allstd)),
                                 mean_alpha=float(np.mean(means)),
                                 n_cells=len(means))
    for name, s in summary.items():
        print(f"  {name:12} mean_alpha={s['mean_alpha']:.3f}  "
              f"temporal-std median={s['med_std']:.5f}  max={s['max_std']:.5f}  (cells={s['n_cells']})")
    chirp = summary.get("SynthChirp"); stat = summary.get("SynthStat")
    print("-" * 64)
    if chirp and stat:
        keep = (chirp["max_std"] >= 0.05) and (stat["max_std"] < 0.01)
        retitle = chirp["max_std"] < 0.01
        print(f"  reference (standard benchmarks): temporal std ~1e-4")
        print(f"  SynthChirp max temporal-std = {chirp['max_std']:.5f}  (criterion: >= 0.05)")
        print(f"  SynthStat  max temporal-std = {stat['max_std']:.5f}  (control: < 0.01)")
        if keep:
            print("  gate tracks the non-stationarity: adaptive by the criterion.")
        elif retitle:
            print("  gate stays flat on the chirp: not adaptive by the criterion.")
        else:
            print("  gate moves, but below the 0.05 threshold.")
    else:
        print("  (insufficient artifacts; check that the runs completed)")
    print("=" * 64)


def main():
    from experiments import runner
    for kind in DATASETS:
        for seed in SEEDS:
            df = gen_synth(kind, seed)
            inject(kind, df)
            print(f"[gate] training AMS-CeNN on {kind} seed={seed} ...", flush=True)
            runner.run_single("CeNN_C1C2-Skip-K2", kind, HORIZON, seed,
                              cenn_K=2, save_artifacts=True)
    analyze()
    print("\ndone")


if __name__ == "__main__":
    main()
