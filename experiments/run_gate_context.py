"""Context-gate study: does a context-aware gate adapt and improve accuracy?

The pointwise gate stays near-uniform on the benchmarks: it sees only the instantaneous
LayerNorm-flat u_c(t), and it has little effect on accuracy. This script trains a
context/volatility-aware gate (CENN_GATE_TYPE=context) on a task where adaptive integration
should help, regime-switching input noise, and compares (single-scale, no skip, so the gate's
alpha drives the output directly):
  S0-StableBase             fixed alpha (no gate)
  C1-BoundedTau             pointwise adaptive gate
  C1-BoundedTau + context   context gate

Criterion, fixed before the runs: the gate counts as usefully adaptive if the context gate
(i) beats fixed-alpha S0 in MSE on SynthHetero and (ii) its alpha temporal std is >= 0.05.

Run (GPU): CENN_EXP_DIR=_gate_context python -m experiments.run_gate_context
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T_TOTAL = 4000
VAL, TEST = 400, 800
V = 7
HORIZON = 96
SEEDS = [1, 42, 123]
DATASET = "SynthHetero"
# (variant, gate_type) cells. gate_type "context" sets CENN_GATE_TYPE; others clear it.
CELLS = [
    ("S0-StableBase", "fixed"),
    ("C1-BoundedTau", "pointwise"),
    ("C1-BoundedTau", "context"),
    ("C1C2-Skip-K2", "pointwise"),   # same test on the skip architecture
    ("C1C2-Skip-K2", "context"),
]


def gen_hetero(seed: int) -> pd.DataFrame:
    """Periodic signal plus regime-switching input noise (alternating low/high sigma blocks)."""
    rng = np.random.default_rng(seed * 104729)
    t = np.arange(T_TOTAL)
    n_train = T_TOTAL - VAL - TEST
    ds = pd.date_range("2020-01-01", periods=T_TOTAL, freq="h")
    block = 150
    # regime sigma: alternating low (0.05) / high (3.0) blocks, same schedule across series.
    regime = (np.arange(T_TOTAL) // block) % 2
    sigma = np.where(regime == 0, 0.05, 3.0)
    rows = []
    for i in range(V):
        phase = rng.uniform(0, 2 * np.pi)
        signal = np.sin(2 * np.pi * t / 24.0 + phase) + 0.5 * np.sin(2 * np.pi * t / 168.0 + phase)
        y = signal + rng.normal(0, 1.0, size=T_TOTAL) * sigma
        mu, sd = y[:n_train].mean(), y[:n_train].std() + 1e-8
        y = (y - mu) / sd
        for tt in range(T_TOTAL):
            rows.append((f"S{i}", ds[tt], float(y[tt])))
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])


def inject(df: pd.DataFrame):
    from experiments import config, runner
    config.DATASET_INFO[DATASET] = {"group": DATASET, "n_series": V, "freq": "h"}
    config.SPLITS[DATASET] = (VAL, TEST)
    runner.DATASET_INFO[DATASET] = config.DATASET_INFO[DATASET]
    runner.SPLITS[DATASET] = config.SPLITS[DATASET]
    runner._dataset_cache[DATASET] = df


def run_cell(variant, gate_type, seed):
    from experiments.runner import build_cenn, load_dataset
    from experiments.config import SPLITS
    from neuralforecast import NeuralForecast
    from neuralforecast.cenn.model import CeNNCell1D
    if gate_type == "context":
        os.environ["CENN_GATE_TYPE"] = "context"
    else:
        os.environ.pop("CENN_GATE_TYPE", None)
    import torch
    torch.manual_seed(seed)
    model, name, _ = build_cenn(variant, HORIZON, V, seed, max_steps=1000, dataset_name=DATASET)
    nf = NeuralForecast(models=[model], freq="h")
    val, test = SPLITS[DATASET]
    cv = nf.cross_validation(df=load_dataset(DATASET), val_size=val, test_size=test,
                             n_windows=None, refit=False)
    if isinstance(cv.index, pd.MultiIndex):
        cv = cv.reset_index()
    elif "unique_id" not in cv.columns:
        cv = cv.reset_index()
    alias = nf.models[0].alias
    mse = float(((cv[alias] - cv["y"]) ** 2).mean())
    # alpha temporal-std (gate variation), max over channels/cells
    astd = np.nan
    stds = []
    for m in nf.models[0].modules():
        if isinstance(m, CeNNCell1D) and getattr(m, "adaptive_tau", False):
            lt = getattr(m, "last_tau", None)
            if lt is not None:
                alpha = (1.0 - lt).detach().float().mean(dim=0).cpu().numpy()  # [C, L]
                stds.append(alpha.std(axis=1))
    if stds:
        astd = float(np.max(np.concatenate(stds)))
    os.environ.pop("CENN_GATE_TYPE", None)
    return mse, astd


def main():
    results = {}
    for variant, gate in CELLS:
        mses, astds = [], []
        for seed in SEEDS:
            inject(gen_hetero(seed))
            print(f"[gate_context] {variant} gate={gate} seed={seed} ...", flush=True)
            mse, astd = run_cell(variant, gate, seed)
            mses.append(mse); astds.append(astd)
            print(f"   -> MSE={mse:.4f} alpha_std={astd}", flush=True)
        results[(variant, gate)] = (float(np.mean(mses)),
                                    float(np.nanmean(astds)) if any(a == a for a in astds) else float("nan"))

    print("\n" + "=" * 70)
    print("Context-gate results (SynthHetero: regime-switching input noise)")
    print("=" * 70)
    print(f"{'variant':16} {'gate':10} {'meanMSE':>9} {'alpha-std':>10}")
    for (v, g), (mse, astd) in results.items():
        print(f"{v:16} {g:10} {mse:>9.4f} {astd:>10.5f}")
    print("-" * 70)
    s0 = results.get(("S0-StableBase", "fixed"), (np.nan, np.nan))[0]
    ctx = results.get(("C1-BoundedTau", "context"), (np.nan, np.nan))
    pw = results.get(("C1-BoundedTau", "pointwise"), (np.nan, np.nan))
    print(f"fixed-alpha (S0) MSE      = {s0:.4f}")
    print(f"pointwise-gate MSE/std    = {pw[0]:.4f} / {pw[1]:.5f}")
    print(f"context-gate  MSE/std     = {ctx[0]:.4f} / {ctx[1]:.5f}")
    keep = (ctx[0] < s0) and (ctx[1] >= 0.05)
    if keep:
        print("context gate beats fixed alpha and varies (alpha std >= 0.05): adaptive by the criterion.")
    elif ctx[1] >= 0.05:
        print("context gate varies (alpha std >= 0.05) but does not beat fixed alpha.")
    else:
        print("context gate does not vary (alpha std < 0.05): not adaptive by the criterion.")
    print("=" * 70)
    print("\ndone")


if __name__ == "__main__":
    main()
