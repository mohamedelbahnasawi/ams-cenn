"""Test-time input perturbations for the robustness stress tests (Paper A).

Operates on a long-format NeuralForecast DataFrame (columns: unique_id, ds, y) in the global
z-score domain that LongHorizon2 delivers, i.e. before each model's own per-window scaler, so a
level means the same thing for every model.

Only the tail of each series (the last test_size + input_size points) is perturbed: the
input-lookback region of the test windows. In a stride-1 sliding evaluation the same timestep is
a lookback input for later windows and a horizon target for earlier ones, so `y` in the
perturbed frame is corrupted in both roles. The robustness driver therefore scores the
perturbed-run predictions against the clean-run targets (joined on window keys); this module
only produces the corrupted model input and must not supply scoring ground truth.

All perturbations are deterministic given `seed` and leave non-tail rows untouched.
"""
from __future__ import annotations
import numpy as np
import zlib
import pandas as pd

KINDS = ("gauss", "spike", "mask", "scale", "shift", "dead")


def dead_ids(Y_df: pd.DataFrame, level: float, seed: int) -> set:
    """Deterministic set of `level` series (channels) that are 'dead' for this seed.

    The other families corrupt every channel; this one models a single stuck sensor. Selection
    is a seeded permutation of the sorted unique_ids, so run_robustness.py can recover the same
    set for healthy-channel scoring."""
    uids = sorted(Y_df["unique_id"].unique())
    k = int(level)
    if k <= 0:
        return set()
    # The dead channels used by each stored cell are recorded in its "dead_ids" field.
    rng = np.random.default_rng(((seed * 1_000_003) ^ 7_919 ^ 0x1BADD1E) & 0xFFFFFFFF)
    return set(rng.permutation(uids)[:min(k, len(uids))].tolist())


def _tail_idx(n: int, tail: int):
    """Index slice for the last `tail` rows of an n-row series (the test-window input region)."""
    start = max(0, n - tail)
    return start, n


def perturb_df(Y_df: pd.DataFrame, kind: str, level: float, seed: int,
               test_size: int, input_size: int = 512) -> pd.DataFrame:
    """Return a copy of Y_df with the `y` tail of every series perturbed.

    kind: one of KINDS. level: perturbation strength (0 returns an unperturbed copy).
    The tail length is test_size + input_size (covers every test window's lookback).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind {kind!r}; expected one of {KINDS}")
    out = Y_df.copy()
    if level == 0:
        return out
    tail = int(test_size + input_size)
    dead = dead_ids(out, level, seed) if kind == "dead" else set()
    # group-stable RNG seeding: each series gets its own deterministic stream
    for gi, (uid, g) in enumerate(out.groupby("unique_id", sort=True)):
        idx = g.index.to_numpy()
        n = len(idx)
        s, e = _tail_idx(n, tail)
        reg = idx[s:e]                      # row labels of the perturb region (this series)
        m = len(reg)
        if m == 0:
            continue
        # zlib.crc32 is process-independent (hash() of a str is salted per process).
        rng = np.random.default_rng((seed * 1_000_003) ^ (gi * 9973) ^ zlib.crc32(kind.encode("utf-8")) & 0xFFFFFFFF)
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
            # models. level = block length b.
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
        elif kind == "dead":
            # dead sensor: the selected channels are stuck at the global mean (0 in the z-score
            # domain) over the whole input region; every other channel is untouched.
            if uid in dead:
                y[:] = 0.0

        out.loc[reg, "y"] = y
    return out
