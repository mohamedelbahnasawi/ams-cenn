"""Worst-case relative error at 28-cell granularity, with dataset-clustered bootstrap CIs.

The manuscript reports worst-case relative error per dataset: for each (model, dataset, horizon)
cell the ratio to the best model in that cell, averaged over the four horizons within a dataset,
then the max over the seven datasets. Averaging over horizons smooths per-cell peaks, so this
script also computes the statistic per cell (7 datasets x 4 horizons) with bootstrap uncertainty,
and reproduces the published per-dataset values first as a self-check. If the self-check fails,
the parser or the table changed and the 28-cell numbers should not be used.

Input: tab:main_results parsed from the manuscript .tex, so the statistic is computed from the
rounded numbers the table shows (plot_regret_distribution.py uses the unrounded result files).

Usage:
    python worst_case_regret_28cell.py [--tex PATH] [--boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import numpy as np

DEFAULT_TEX = Path("manuscript.tex")  # pass --tex to point at the manuscript source

# Published per-dataset worst-case relative errors (tab:profile), used as the self-check.
PUBLISHED = {
    "AMS-CeNN": 6.3, "TSMixer": 12.7, "TimeMixer": 11.0, "PatchTST": 22.6,
    "iTransformer": 22.7, "DLinear": 32.6, "TiDE": 34.7, "TCN": 63.3,
    "TimesNet": 67.6, "NHITS": 99.2, "xLSTM": 247.6, "S4D": 328.9,
}

CELL = re.compile(r"\\textcolor\{\w+\}\{\\textbf\{([^}]*)\}\}|\\textbf\{([^}]*)\}|([-\d.]+)")


def _clean(tok: str) -> float | None:
    """Strip \\textcolor/\\textbf wrappers from a table cell and return the float."""
    tok = tok.strip()
    if not tok or tok == "--":
        return None
    m = CELL.match(tok)
    if not m:
        return None
    val = next((g for g in m.groups() if g), None)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_main_results(tex_path: Path) -> tuple[list[str], dict]:
    """Return (models, {(model, dataset, horizon): mse}) parsed from tab:main_results."""
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    start = text.index(r"\label{tab:main_results}")
    end = text.index(r"\end{tabular}", start)
    block = text[start:end]

    header = next(l for l in block.splitlines() if "multicolumn" in l)
    models = re.findall(r"\\multicolumn\{2\}\{c\}\{([^}]*)\}", header)

    mse: dict[tuple[str, str, int], float] = {}
    for line in block.splitlines():
        line = line.strip()
        if "&" not in line or "multicolumn" in line or line.startswith("%"):
            continue
        cols = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cols) < 3:
            continue
        dataset, horizon = cols[0], cols[1]
        if not re.fullmatch(r"\d+", horizon):
            continue
        # each model contributes an (MSE, MAE) pair, in header order
        for i, model in enumerate(models):
            idx = 2 + 2 * i
            if idx < len(cols):
                v = _clean(cols[idx])
                if v is not None:
                    mse[(model, dataset, int(horizon))] = v
    return models, mse


def ratios_per_cell(models, mse, datasets, horizons) -> dict:
    """{(model, dataset, horizon): mse / best-in-cell}. Cells missing any model are skipped."""
    out = {}
    for d in datasets:
        for h in horizons:
            cells = {m: mse.get((m, d, h)) for m in models}
            if any(v is None for v in cells.values()):
                continue
            best = min(cells.values())
            for m, v in cells.items():
                out[(m, d, h)] = v / best
    return out


def per_dataset_stat(model, ratios, datasets, horizons) -> float:
    """Per-dataset definition (the one in the paper): mean ratio over horizons within a dataset, then max over datasets."""
    per_ds = []
    for d in datasets:
        rs = [ratios[(model, d, h)] for h in horizons if (model, d, h) in ratios]
        if rs:
            per_ds.append(statistics.mean(rs))
    return (max(per_ds) - 1.0) * 100.0 if per_ds else float("nan")


def per_cell_stat(model, ratios, datasets, horizons) -> float:
    """Per-cell definition: max ratio over all 28 (dataset, horizon) cells, no horizon averaging."""
    rs = [ratios[(model, d, h)] for d in datasets for h in horizons if (model, d, h) in ratios]
    return (max(rs) - 1.0) * 100.0 if rs else float("nan")


def regret_distribution(model, ratios, datasets, horizons) -> np.ndarray:
    """All 28 per-cell regrets (%) for one model."""
    return np.array([(ratios[(model, d, h)] - 1.0) * 100.0
                     for d in datasets for h in horizons if (model, d, h) in ratios])


def bootstrap_quantile_ci(model, ratios, datasets, horizons, q, n_boot, rng):
    """Dataset-clustered bootstrap CI for a quantile of the per-cell regret distribution.

    Not used for the max: the nonparametric bootstrap is inconsistent for extreme-order
    statistics (a resample never contains a value above the observed maximum, so the upper
    limit is an artifact). Interior quantiles (median, p75, p90) are fine; the max is reported
    as the worst observed cell without a CI.
    """
    draws = []
    ds = np.array(datasets, dtype=object)
    for _ in range(n_boot):
        sample = list(rng.choice(ds, size=len(ds), replace=True))
        vals = regret_distribution(model, ratios, sample, horizons)
        if vals.size:
            draws.append(np.percentile(vals, q))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff_ci(m_a, m_b, ratios, datasets, horizons, n_boot, rng):
    """Dataset-clustered bootstrap CI for the paired per-cell regret difference (a - b), a mean
    over matched cells. Returns (point estimate, lo, hi)."""
    ds = np.array(datasets, dtype=object)
    draws = []
    for _ in range(n_boot):
        sample = list(rng.choice(ds, size=len(ds), replace=True))
        da = regret_distribution(m_a, ratios, sample, horizons)
        db = regret_distribution(m_b, ratios, sample, horizons)
        if da.size and da.size == db.size:
            draws.append(float(np.mean(da - db)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(regret_distribution(m_a, ratios, datasets, horizons)
                         - regret_distribution(m_b, ratios, datasets, horizons))), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models, mse = parse_main_results(args.tex)
    datasets = sorted({d for (_, d, _) in mse})
    horizons = sorted({h for (_, _, h) in mse})
    print(f"parsed {len(models)} models x {len(datasets)} datasets x {len(horizons)} horizons "
          f"= {len(datasets) * len(horizons)} cells")
    print(f"models: {', '.join(models)}\n")

    ratios = ratios_per_cell(models, mse, datasets, horizons)
    complete = len({(d, h) for (_, d, h) in ratios})
    print(f"complete cells (all {len(models)} models present): {complete}\n")

    # ---- self-check against the published per-dataset numbers -------------------
    print("SELF-CHECK vs published per-dataset worst-case (tab:profile)")
    print(f"{'model':<14}{'ours':>8}{'published':>11}{'delta':>8}")
    ok = True
    for m in models:
        got = per_dataset_stat(m, ratios, datasets, horizons)
        exp = PUBLISHED.get(m)
        if exp is None:
            continue
        delta = got - exp
        # absolute tolerance for small values, relative for large ones: the table prints MSE to
        # three decimals, and a large ratio (S4D ~ 4.3x) amplifies that rounding
        agree = abs(delta) <= 0.15 or abs(delta) / max(exp, 1e-9) <= 0.005
        flag = "" if agree else "  <-- MISMATCH"
        if not agree:
            ok = False
        print(f"{m:<14}{got:8.1f}{exp:11.1f}{delta:8.2f}{flag}")
    print(f"\nself-check: {'PASS (all models agree to rounding)' if ok else 'FAIL (numbers below are inconsistent with the published table)'}\n")

    # ---- the 28-cell regret distribution ---------------------------------------
    rng = np.random.default_rng(args.seed)
    rows = []
    for m in models:
        vals = regret_distribution(m, ratios, datasets, horizons)
        med, p75, p90 = (float(np.percentile(vals, q)) for q in (50, 75, 90))
        mx = float(vals.max())
        worst = max(((ratios[(m, d, h)], d, h) for d in datasets for h in horizons
                     if (m, d, h) in ratios))
        mlo, mhi = bootstrap_quantile_ci(m, ratios, datasets, horizons, 50, args.boot, rng)
        rows.append((m, per_dataset_stat(m, ratios, datasets, horizons),
                     med, mlo, mhi, p75, p90, mx, f"{worst[1]} H{worst[2]}"))

    rows.sort(key=lambda r: r[7])
    print(f"PER-CELL REGRET DISTRIBUTION over 28 cells (%), "
          f"{args.boot} dataset-clustered bootstrap resamples")
    print(f"{'model':<14}{'per-ds':>8}{'median':>8}{'95% CI (median)':>19}"
          f"{'p75':>8}{'p90':>8}{'max':>8}   worst cell")
    for m, ds_stat, med, mlo, mhi, p75, p90, mx, where in rows:
        print(f"{m:<14}{ds_stat:8.1f}{med:8.1f}   [{mlo:5.1f}, {mhi:5.1f}]"
              f"{p75:8.1f}{p90:8.1f}{mx:8.1f}   {where}")

    print("\n'max' is the worst observed cell; no CI is given for it "
          "(the bootstrap is inconsistent for extreme-order statistics).")

    # ---- paired comparisons against a fixed comparator set ------------------------
    # The comparators are fixed in advance rather than chosen from the sorted table.
    COMPARATORS = ["TSMixer", "PatchTST", "DLinear", "TimeMixer"]
    print()
    print("Paired per-cell regret vs the fixed comparator set "
          "(AMS-CeNN minus comparator; negative = lower regret)")
    for comp in COMPARATORS:
        if comp not in models:
            continue
        diff, lo, hi = paired_diff_ci("AMS-CeNN", comp, ratios, datasets, horizons,
                                      args.boot, rng)
        sep = "separates" if (lo > 0 or hi < 0) else "does NOT separate"
        print(f"  vs {comp:<13}{diff:+7.2f} pp   95% CI [{lo:+6.2f}, {hi:+6.2f}]   {sep}")
    print("  With seven clusters the bootstrap admits few distinct resamples;")
    print("  the intervals are indicative rather than exact.")


if __name__ == "__main__":
    main()
