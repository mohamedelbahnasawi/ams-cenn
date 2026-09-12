"""Summaries of the lookback sweep, the stability ablations and the integrator check.

Reads the atomic result JSONs (experiments/results and experiments/L{96,192,336}/results) and prints:
  1. lookback sweep: mean MSE over ETTh1/ETTh2/Weather at H=96 for AMS-Anc and C1C2-Anc at L in {96,192,336,512}
  2. stability ablations: ETT mean MSE (4 datasets x 4 horizons, seeds 1/42/123) for AMS-Anc, AMS-Anc-GateUnbounded,
     AMS-Anc-CapOff
  3. integrators at K=2: ETT mean MSE at H=96 (seeds 1/42/123) for AMS-Anc and the Heun / exp. Euler / RK4 arms
Usage: python lvn_ablation_summary.py   (from the repository root)
"""
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import RESULTS_DIR, EXPERIMENTS_DIR  # noqa: E402

SEEDS = (1, 42, 123)
ETT = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


def cell(variant, dataset, h, L=512, seeds=SEEDS):
    root = str(RESULTS_DIR) if L == 512 else str(EXPERIMENTS_DIR / f"L{L}" / "results")
    vals = []
    for s in seeds:
        fs = glob.glob(f"{root}/CeNN_{variant}__{dataset}__H{h}__seed{s}.json")
        if fs:
            vals.append(json.load(open(fs[0]))["mse"])
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def mean_over(variant, datasets, horizons, L=512):
    ms, n = [], 0
    for d in datasets:
        for h in horizons:
            m, k = cell(variant, d, h, L)
            if m is None:
                return None, n
            ms.append(m); n += k
    return statistics.mean(ms), n


def main():
    print("== 1. lookback sweep (ETTh1/ETTh2/Weather, H=96, seeds 1/42/123)")
    for v in ("AMS-Anc", "C1C2-Anc"):
        row = []
        for L in (96, 192, 336, 512):
            m, n = mean_over(v, ("ETTh1", "ETTh2", "Weather"), (96,), L)
            row.append(f"L{L}: {m:.4f} (n={n})" if m is not None else f"L{L}: missing (n={n})")
        print(f"  {v:22s}", " | ".join(row))
    print("== 2. stability ablations (ETT, 4 horizons, seeds 1/42/123)")
    for v in ("AMS-Anc", "AMS-Anc-GateUnbounded", "AMS-Anc-CapOff"):
        m, n = mean_over(v, ETT, (96, 192, 336, 720))
        print(f"  {v:22s}", f"{m:.4f} (n={n})" if m is not None else f"missing (n={n})")
    print("== 3. integrators at K=2 (ETT, H=96, seeds 1/42/123)")
    for v in ("AMS-Anc", "AMS-Anc-Heun", "AMS-Anc-ExpEuler", "AMS-Anc-RK4"):
        m, n = mean_over(v, ETT, (96,))
        print(f"  {v:22s}", f"{m:.4f} (n={n})" if m is not None else f"missing (n={n})")


if __name__ == "__main__":
    main()
