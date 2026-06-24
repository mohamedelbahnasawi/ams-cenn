"""Aggregate experiment results into tables and figures.

Scans experiments/results/ and experiments/published/ for JSON files,
combines them, and generates:
  - aggregated/main_results.csv (full results table)
  - aggregated/main_results.tex (LaTeX table for paper)
  - aggregated/ablation_results.csv
  - aggregated/summary.txt (human-readable status report)

Re-runnable anytime — always reads current state of results/.

Usage:
    python experiments/aggregate.py              # Full aggregation
    python experiments/aggregate.py --status     # Just show status (what's done/missing)
    python experiments/aggregate.py --latex      # Generate LaTeX tables only
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from experiments.config import (
    RESULTS_DIR, PUBLISHED_DIR, AGGREGATED_DIR, FIGURES_DIR,
    DATASET_INFO, DATASETS_ALL, DATASETS_SMALL,
    HORIZONS, SEEDS, SEEDS_MAIN,
    CENN_VARIANTS, CENN_MAIN_VARIANT,
    BASELINES_RERUN, BASELINES_PUBLISHED, BASELINES_ALL,
)


def load_all_results() -> pd.DataFrame:
    """Load all result JSONs (experiment + published) into a DataFrame.

    RESULTS-TAKE-PRECEDENCE: a published (literature) cell is loaded ONLY if we have no
    real run for that (model, dataset, horizon). Once a baseline is fully re-run under our
    L=512/stride-1 pipeline, its literature transcription in published/ is kept on disk as
    an appendix reproduction reference but is NOT merged into the main aggregation --
    otherwise the seed-mean would blend a single published value (seed=0) with our re-run
    seeds and corrupt the number. (Pre-declared policy: re-run numbers take precedence;
    published/ = appendix cross-check.)
    """
    records = []

    # Load experiment results (real runs). Count seeds per cell so we can warn if a published
    # number is suppressed by an UNDER-SEEDED real run (latent hazard: precedence keys on
    # (model,dataset,horizon), not on seed completeness -- a 1-of-3-seed run would still hide the
    # literature value silently. Harmless today since PatchTST, the only published model, is
    # complete; the warning is insurance for when published/ is expanded.)
    run_seeds = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("mse") is not None:  # Skip failed runs
                records.append(data)
                run_seeds.setdefault(
                    (data["model"], data["dataset"], data["horizon"]), set()
                ).add(data.get("seed"))
        except (json.JSONDecodeError, KeyError):
            continue

    # Load published baselines -- but only for cells we did NOT re-run ourselves.
    skipped_overridden = 0
    underseeded = []
    for f in sorted(PUBLISHED_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("mse") is None:
                continue
            key = (data["model"], data["dataset"], data["horizon"])
            if key in run_seeds:
                skipped_overridden += 1     # we have a real run -> literature value is appendix-only
                if len(run_seeds[key]) < 2:   # suspiciously thin override
                    underseeded.append(f"{key[0]}/{key[1]}/H{key[2]} ({len(run_seeds[key])} seed)")
                continue
            records.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    if skipped_overridden:
        print(f"[load_all_results] {skipped_overridden} published cells overridden by real "
              f"re-runs (kept on disk as appendix reproduction reference)")
    if underseeded:
        print(f"[load_all_results] WARNING: {len(underseeded)} published cells overridden by a "
              f"<2-seed real run -- main-table number rests on a thin re-run: {underseeded[:8]}"
              f"{' ...' if len(underseeded) > 8 else ''}")

    if not records:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Defense-in-depth: keep ONLY the LTSF benchmark datasets. Without this, stray result JSONs
    # from unrelated runs (whose error scale can differ by orders of magnitude) could silently merge
    # into the LTSF main/ablation tables and corrupt the aggregated numbers.
    if "dataset" in df.columns:
        before = len(df)
        df = df[df["dataset"].isin(DATASETS_ALL)].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[load_all_results] dropped {dropped} non-LTSF rows "
                  f"(datasets not in DATASETS_ALL); kept {len(df)}")
    return df


def show_status(df: pd.DataFrame):
    """Show what's completed and what's missing."""

    all_models = [f"CeNN_{v}" for v in CENN_VARIANTS] + BASELINES_ALL
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT STATUS REPORT")
    print(f"{'='*70}")
    print(f"  Total result files: {len(df)}")

    if df.empty:
        print("  No results yet.")
        return

    print(f"  Models with results: {sorted(df['model'].unique())}")
    print(f"  Datasets with results: {sorted(df['dataset'].unique())}")
    print()

    # Coverage matrix
    print("  Coverage (model × dataset, all horizons+seeds):")
    print(f"  {'Model':<20}", end="")
    for ds in DATASETS_ALL:
        print(f"{ds:>12}", end="")
    print()

    for model in all_models:
        model_df = df[df["model"] == model]
        print(f"  {model:<20}", end="")
        for ds in DATASETS_ALL:
            count = len(model_df[model_df["dataset"] == ds])
            expected = len(HORIZONS) * len(SEEDS)
            if model == f"CeNN_{CENN_MAIN_VARIANT}":
                expected = len(HORIZONS) * len(SEEDS_MAIN)
            if count == 0:
                print(f"{'—':>12}", end="")
            elif count >= expected:
                print(f"{'✓':>12}", end="")
            else:
                print(f"{count}/{expected}".rjust(12), end="")
        print()

    print()


def generate_main_table(df: pd.DataFrame):
    """Generate main results table (mean ± std across seeds)."""
    if df.empty:
        return

    # Group by (model, dataset, horizon), aggregate across seeds
    agg = df.groupby(["model", "dataset", "horizon"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    # Fill NaN std (single seed) with 0
    agg["mse_std"] = agg["mse_std"].fillna(0)
    agg["mae_std"] = agg["mae_std"].fillna(0)

    # Save CSV
    csv_path = AGGREGATED_DIR / "main_results.csv"
    agg.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return agg


def generate_ablation_table(df: pd.DataFrame):
    """Generate ablation table (CeNN variants only)."""
    if df.empty:
        return

    cenn_models = [f"CeNN_{v}" for v in CENN_VARIANTS]
    abl_df = df[df["model"].isin(cenn_models)]

    if abl_df.empty:
        print("  No CeNN ablation results yet.")
        return

    agg = abl_df.groupby(["model", "dataset", "horizon"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    agg["mse_std"] = agg["mse_std"].fillna(0)

    csv_path = AGGREGATED_DIR / "ablation_results.csv"
    agg.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return agg


def generate_latex_main(df: pd.DataFrame):
    """Generate LaTeX table for the main results."""
    if df.empty:
        return

    agg = df.groupby(["model", "dataset", "horizon"]).agg(
        mse_mean=("mse", "mean"),
    ).reset_index()

    # Pivot: rows = (dataset, horizon), columns = model
    pivot = agg.pivot_table(
        index=["dataset", "horizon"],
        columns="model",
        values="mse_mean",
    )

    # Find best and second best per row
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\caption{Multivariate long-horizon forecasting results (MSE). "
                 r"Results averaged over seeds. "
                 r"\textbf{Bold}: best. \underline{Underline}: second best.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{cc|" + "c" * len(pivot.columns) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Dataset & H & " + " & ".join(pivot.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    prev_ds = None
    for (ds, h), row in pivot.iterrows():
        vals = row.values
        sorted_vals = sorted([v for v in vals if pd.notna(v)])

        if len(sorted_vals) >= 2:
            best, second = sorted_vals[0], sorted_vals[1]
        elif len(sorted_vals) == 1:
            best, second = sorted_vals[0], None
        else:
            best, second = None, None

        if ds != prev_ds and prev_ds is not None:
            lines.append(r"\midrule")
        prev_ds = ds

        cells = []
        for v in vals:
            if pd.isna(v):
                cells.append("—")
            elif v == best:
                cells.append(f"\\textbf{{{v:.3f}}}")
            elif second is not None and v == second:
                cells.append(f"\\underline{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")

        line = f"{ds} & {h} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    tex_path = AGGREGATED_DIR / "main_results.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--status", action="store_true",
                        help="Show status only (what's done/missing)")
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX tables")
    args = parser.parse_args()

    df = load_all_results()

    if args.status:
        show_status(df)
        return

    print(f"\nLoaded {len(df)} results.")
    show_status(df)

    print("\nGenerating tables...")
    generate_main_table(df)
    generate_ablation_table(df)

    if args.latex or not args.status:
        generate_latex_main(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
