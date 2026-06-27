"""Test-time input perturbations for the robustness stress tests (Paper A).

Operates on a long-format NeuralForecast DataFrame (columns: unique_id, ds, y), in the
GLOBAL z-score domain that LongHorizon2 delivers (so a level is comparable across models
regardless of each model's per-window TemporalNorm — perturbing here is PRE-scaler and fair).

Only the **tail** of each series (the last test_size + input_size points) is perturbed, i.e.
the input-lookback region of the test windows. Correctness note: in a stride-1 sliding eval the
same timestep is a lookback-input for later windows and a horizon-target for earlier ones, so the
perturbed frame's `y` is corrupted for BOTH roles. The robustness driver therefore scores the
perturbed-run PREDICTIONS against the CLEAN-run TARGETS (join on window keys) — this module only
ever produces the corrupted model INPUT; it must never be used to supply scoring ground truth.

All perturbations are deterministic given `seed` and leave non-tail rows untouched.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

KINDS = ("gauss", "spike", "mask", "scale", "shift")


def _tail_idx(n: int, tail: int):
    """Index slice for the last `tail` rows of an n-row series (the test-window input region)."""
    start = max(0, n - tail)
    return start, n


def perturb_df(Y_df: pd.DataFrame, kind: str, level: float, seed: int,
               test_size: int, input_size: int = 512) -> pd.DataFrame:
    """Return a COPY of Y_df with the `y` tail of every series perturbed.

    kind: one of KINDS. level: perturbation strength (0 => returns an unperturbed copy).
    The tail length is test_size + input_size (covers every test window's lookback).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind {kind!r}; expected one of {KINDS}")
    out = Y_df.copy()
    if level == 0:
        return out
    tail = int(test_size + input_size)
    # group-stable RNG seeding: each series gets its own deterministic stream
    for gi, (uid, g) in enumerate(out.groupby("unique_id", sort=True)):
        idx = g.index.to_numpy()
        n = len(idx)
        s, e = _tail_idx(n, tail)
        reg = idx[s:e]                      # row labels of the perturb region (this series)
        m = len(reg)
        if m == 0:
            continue
        rng = np.random.default_rng((seed * 1_000_003) ^ (gi * 9973) ^ hash(kind) & 0xFFFFFFFF)
        y = out.loc[reg, "y"].to_numpy(dtype=np.float64).copy()

        if kind == "gauss":
            y = y + rng.normal(0.0, float(level), size=m)
        elif kind == "spike":
            # outlier injection: fraction `level` of region timesteps get a +/- spike of 5-8 z-units
            k = int(round(float(level) * m))
            if k > 0:
                pos = rng.choice(m, size=k, replace=False)
                mag = rng.uniform(5.0, 8.0, size=k) * rng.choice([-1.0, 1.0], size=k)
                y[pos] = y[pos] + mag
        elif kind == "mask":
            # missing blocks: zero-impute (z-score mean) contiguous blocks of length b at ~15%
            # coverage. Zero-impute (not NaN) so it is identical for mask-aware and mask-unaware
            # models — a fair cross-model missing-data test. level = block length b.
            b = int(level)
            if b > 0:
                target = int(0.15 * m)
                n_blocks = max(1, target // b)
                for _ in range(n_blocks):
                    st = int(rng.integers(0, max(1, m - b)))
                    y[st:st + b] = 0.0
        elif kind == "scale":
            # multiplicative gain error on the input region
            y = y * (1.0 + float(level))
        elif kind == "shift":
            # constant level/distribution shift (z-units) on the input region
            y = y + float(level)

        out.loc[reg, "y"] = y
    return out
