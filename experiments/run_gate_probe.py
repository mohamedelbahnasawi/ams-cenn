"""Gate-variation probe (Paper A): does the bounded gate adapt to input non-stationarity?

Question: does the bounded-tau gate genuinely ADAPT to input non-stationarity, or is it the
near-uniform constant (mean alpha~0.90, temporal std~1e-4) observed on the standard benchmarks?

Method (no edit to the vendored CeNN math): generate two synthetic multivariate datasets —
  - SynthChirp:  strongly NON-STATIONARY (frequency + amplitude drift across the series), so every
                 L=512 lookback contains changing dynamics the gate could respond to;
  - SynthStat:   STATIONARY control (fixed frequency + amplitude),
train AMS-CeNN (C1C2-Skip-K2) on each via the existing runner, and read the gate artifact that
_save_tau already writes (last_tau -> alpha = 1 - tau, shape [C, L] per cell). The temporal std of
alpha across the L lookback measures whether the gate tracks the within-window non-stationarity.

PRE-REGISTERED verdict (see R8_R2_HARNESS_DESIGN.md):
  KEEP "Adaptive" iff alpha temporal-std >= 0.05 on SynthChirp AND < 0.01 on SynthStat.
  RETITLE if SynthChirp temporal-std stays < 0.01 (gate does not adapt even to a built non-stationarity).

Launch (CENN_EXP_DIR must be set BEFORE python starts; it is read at config import):
  CENN_EXP_DIR=experiments/_r2 python -m experiments.run_r2
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
        else:  # SynthStat — fixed frequency + amplitude
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
    """Read the tau artifacts and print the KEEP/RETITLE verdict."""
    from experiments import config
    tau_dir = config.ARTIFACTS_DIR / "tau"
    print("\n" + "=" * 64)
    print("GATE-VARIATION VERDICT (alpha = 1 - tau; temporal std across L)")
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
        print(f"  floor (standard benchmarks): temporal std ~1e-4")
        print(f"  SynthChirp max temporal-std = {chirp['max_std']:.5f}  (KEEP needs >= 0.05)")
        print(f"  SynthStat  max temporal-std = {stat['max_std']:.5f}  (control needs < 0.01)")
        if keep:
            print("  VERDICT: KEEP 'Adaptive' — gate genuinely tracks non-stationarity.")
        elif retitle:
            print("  VERDICT: RETITLE — gate stays flat even on a built non-stationarity; drop 'Adaptive'.")
        else:
            print("  VERDICT: BORDERLINE — gate moves but below the pre-registered keep threshold; lean retitle.")
    else:
        print("  (insufficient artifacts — check the runs completed)")
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
    print("\nR2 DONE")


if __name__ == "__main__":
    main()
