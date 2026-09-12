"""Worst-case relative error at 28-cell granularity, with dataset-clustered bootstrap CIs.

WHY THIS EXISTS
---------------
The manuscript reports worst-case relative error per DATASET: for each (model, dataset,
horizon) cell it takes the ratio to the best model in that cell, averages the four
horizon ratios within a dataset, then takes the max over the seven datasets. Averaging
over horizons smooths away per-cell peaks, so a model that is far behind on a single
(dataset, horizon) cell can still look consistent. A reviewer asked for the statistic at
full 28-cell granularity (7 datasets x 4 horizons) plus uncertainty.

This script computes BOTH definitions from the numbers the paper actually prints, and
reproduces the published per-dataset figures first as a self-check. If the self-check
fails, the parser or the table changed and the 28-cell number must not be trusted.

DATA SOURCE
-----------
tab:main_results in the manuscript .tex -- deliberately, not the repo's aggregated CSVs,
so that the statistic is computed from exactly the numbers a reader sees.
Parsing what is printed guarantees the reported statistic is consistent with the table a
referee reads.

USAGE
-----
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
    """PUBLISHED definition: mean ratio over horizons within a dataset, then max over datasets."""
    per_ds = []
    for d in datasets:
        rs = [ratios[(model, d, h)] for h in horizons if (model, d, h) in ratios]
        if rs:
            per_ds.append(statistics.mean(rs))
    return (max(per_ds) - 1.0) * 100.0 if per_ds else float("nan")


def per_cell_stat(model, ratios, datasets, horizons) -> float:
    """NEW definition: max ratio over all 28 (dataset, horizon) cells -- no horizon averaging."""
    rs = [ratios[(model, d, h)] for d in datasets for h in horizons if (model, d, h) in ratios]
    return (max(rs) - 1.0) * 100.0 if rs else float("nan")


def regret_distribution(model, ratios, datasets, horizons) -> np.ndarray:
    """All 28 per-cell regrets (%) for one model -- the full distribution, not just its max."""
    return np.array([(ratios[(model, d, h)] - 1.0) * 100.0
                     for d in datasets for h in horizons if (model, d, h) in ratios])


def bootstrap_quantile_ci(model, ratios, datasets, horizons, q, n_boot, rng):
    """Dataset-clustered bootstrap CI for a QUANTILE of the per-cell regret distribution.

    NOTE ON WHY THIS IS NOT DONE FOR THE MAX. The nonparametric bootstrap is inconsistent
    for extreme-order statistics: a resample can never contain a value above the observed
    maximum, so the bootstrap distribution of the max is bounded above by the point
    estimate and its upper "confidence" limit is an artifact, not an inference. Interior
    quantiles (median, p75, p90) are asymptotically normal and the bootstrap is valid for
    them, so uncertainty is reported there and the max is reported as a descriptive
    worst observed case.
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
    """Dataset-clustered bootstrap CI for the PAIRED per-cell regret difference (a - b).

    Valid where a CI on the difference of two maxima is not: the paired difference is a
    mean over matched cells, so the bootstrap applies normally. A CI excluding zero means
    model a is genuinely lower-regret across the cell population, not merely at its peak.
    """
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
        # Absolute tolerance for small values; relative tolerance for large ones. The table
        # prints MSE to three decimals, so a model whose ratio to best is large (S4D ~ 4.3x)
        # amplifies that rounding: 0.5 pp on 328 % is 0.16 % relative, not a methodology error.
        agree = abs(delta) <= 0.15 or abs(delta) / max(exp, 1e-9) <= 0.005
        flag = "" if agree else "  <-- MISMATCH"
        if not agree:
            ok = False
        print(f"{m:<14}{got:8.1f}{exp:11.1f}{delta:8.2f}{flag}")
    print(f"\nself-check: {'PASS (rounding-level agreement on all 12 models)' if ok else 'FAIL -- do not trust the numbers below'}\n")

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

    print("\nNOTE: 'max' is the worst observed cell, reported descriptively. No CI is given "
          "for it:\n      the bootstrap is inconsistent for extreme-order statistics "
          "(see bootstrap_quantile_ci).")

    # ---- paired comparisons against a PRE-SPECIFIED comparator set ---------------
    # Deliberately NOT "the next model in this sorted table": choosing the comparator by the
    # statistic being defended is selective inference and flatters the result. AMS-CeNN
    # separates from exactly one of the four nearest competitors; reporting all four is the
    # honest form.
    COMPARATORS = ["TSMixer", "PatchTST", "DLinear", "TimeMixer"]
    print()
    print("PAIRED per-cell regret vs a pre-specified comparator set "
          "(AMS-CeNN minus comparator; negative = lower regret)")
    for comp in COMPARATORS:
        if comp not in models:
            continue
        diff, lo, hi = paired_diff_ci("AMS-CeNN", comp, ratios, datasets, horizons,
                                      args.boot, rng)
        sep = "separates" if (lo > 0 or hi < 0) else "does NOT separate"
        print(f"  vs {comp:<13}{diff:+7.2f} pp   95% CI [{lo:+6.2f}, {hi:+6.2f}]   {sep}")
    print("  NOTE: with seven clusters the bootstrap admits few distinct resamples;")
    print("        treat these intervals as indicative rather than exact.")


if __name__ == "__main__":
    main()
