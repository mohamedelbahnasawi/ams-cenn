"""Context-gate study (Paper A): can the gate be made genuinely adaptive and beneficial?

The pointwise gate stays near-uniform because (a) it sees only the instantaneous LayerNorm-flat
u_c(t) and (b) it has no accuracy leverage. This probe tests a context/volatility-aware gate
(CENN_GATE_TYPE=context) on a task where adaptive integration SHOULD help: heteroscedastic
(regime-switching) input noise — smooth more in noisy regions, trust the drive in clean ones.

Compared (single-scale, NO skip, so the gate's alpha directly drives the output -> real leverage):
  S0-StableBase   : fixed alpha (no gate)            -> benefit floor
  C1-BoundedTau   : pointwise adaptive gate (current)
  C1-BoundedTau + context : the fixed gate (new)

PRE-REGISTERED: KEEP "Adaptive" iff the CONTEXT gate (i) beats the fixed-alpha S0 in MSE on
SynthHetero AND (ii) its alpha temporal-std >= 0.05 (genuinely tracks the regime). Otherwise the
gate is not usefully adaptive -> drop "Adaptive".

Launch (isolated, GPU): CENN_EXP_DIR=_gate_context python -m experiments.run_gate_context
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
    ("C1C2-Skip-K2", "pointwise"),   # does it matter in the shipped arch (skip dominates)?
    ("C1C2-Skip-K2", "context"),
]


def gen_hetero(seed: int) -> pd.DataFrame:
    """Periodic signal + REGIME-SWITCHING input noise (blocks of low/high sigma).
    Adaptive smoothing should recover the clean signal better in high-noise blocks."""
    rng = np.random.default_rng(seed * 104729)
    t = np.arange(T_TOTAL)
    n_train = T_TOTAL - VAL - TEST
    ds = pd.date_range("2020-01-01", periods=T_TOTAL, freq="h")
    block = 150
    # regime sigma: alternate low(0.05)/high(3.0) blocks (same schedule across series) — a very
    # strong SNR contrast (push #2) so adaptive smoothing of the noisy-regime lookback has the
    # largest possible accuracy job.
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
    print("GATE-CONTEXT VERDICT (SynthHetero: regime-switching input noise)")
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
        print("VERDICT: KEEP 'Adaptive' — context gate beats fixed AND genuinely varies (std>=0.05).")
    elif ctx[1] >= 0.05:
        print("VERDICT: gate VARIES but does NOT beat fixed -> 'input-conditioned' honest, not 'improves accuracy'.")
    else:
        print("VERDICT: DROP 'Adaptive' — context gate does not usefully adapt even here.")
    print("=" * 70)
    print("\nGATE-CONTEXT DONE")


if __name__ == "__main__":
    main()
