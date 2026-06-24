#!/usr/bin/env python
"""ONE entrypoint to (re)generate every asset for this work + a completeness report.

Runs, in order: aggregate.py (CSVs + coverage) -> make_tables.py (LaTeX + significance) ->
analysis/make_figures.py (figures from real data). Then prints/writes a CHECKLIST of every expected
asset (tables, figures, significance, published baselines, matrix coverage) marking each
present / MISSING / synthetic — so "I forgot to generate X" is impossible to miss after the H100 run.

Run:  .venv/Scripts/python.exe experiments/generate_paper_assets.py --anchor-dataset ETTh1 --anchor-horizon 96
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
PY = sys.executable
sys.path.insert(0, str(REPO))
from experiments.config import AGGREGATED_DIR, EFFICIENCY_DIR, PUBLISHED_DIR  # noqa: E402
TAB = AGGREGATED_DIR / "tables"
FIG = AGGREGATED_DIR / "figures"
EFF = EFFICIENCY_DIR
PUB = PUBLISHED_DIR


def run(cmd, label):
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    r = subprocess.run([PY] + cmd, cwd=str(REPO))
    if r.returncode != 0:
        print(f"  [WARN] {label} exited {r.returncode}")
    return r.returncode == 0


def _nonempty(p: Path):
    return p.exists() and p.stat().st_size > 0


def completeness_report(anchor_ds, anchor_h):
    """Enumerate every expected output asset and report present / MISSING. The safety net."""
    import glob
    import json
    sys.path.insert(0, str(REPO))
    from experiments.config import CENN_VARIANTS, BASELINES_PUBLISHED, HORIZONS, DATASETS_ALL

    checks = []  # (ok, label, detail)

    # --- tables ---
    for fn, lab in [("main_results.tex", "Main results table (tab:main_results)"),
                    ("ablation_results.tex", "Ablation table (tab:ablation)"),
                    ("efficiency_results.tex", "Efficiency table (tab:efficiency)")]:
        checks.append((_nonempty(TAB / fn), lab, str(TAB / fn)))

    # --- significance ---
    sig_f = TAB / "significance.json"
    if _nonempty(sig_f):
        sig = json.loads(sig_f.read_text())
        has = "friedman_p" in sig
        checks.append((has, "Significance (Friedman+Nemenyi+CD)",
                       f"p={sig.get('friedman_p')}, N={sig.get('n_complete_blocks')}, k={sig.get('k_methods')}"
                       if has else f"NOT COMPUTED: {sig.get('note', '?')}"))
    else:
        checks.append((False, "Significance (Friedman+Nemenyi+CD)", "significance.json missing"))

    # --- figures (wired to real data) ---
    for fn, lab in [("fig_pareto_acc_vs_macs.png", "Pareto fig (fig:pareto)"),
                    ("fig_efficiency_panels.png", "Efficiency panels"),
                    ("fig_k_integrator.png", "K-step ablation fig (fig:k_ablation)"),
                    ("fig_horizon_curves.png", "Per-horizon curves"),
                    ("fig_tau_heatmap.png", "Tau heatmap (fig:tau_viz)"),
                    ("fig_cd_diagram.png", "Critical-difference diagram"),
                    ("fig_ablation_bars.png", "Ablation bars (Delta-MSE)"),
                    ("fig_mse_boxplots.png", "MSE distribution boxplots"),
                    ("fig_forecast_grid.png", "Forecast-vs-truth grid (fig:forecast)"),
                    # mechanism / UQ diagnostics (pipelines now built; fed by --save-artifacts CeNN runs)
                    ("fig_contraction.png", "Contraction ||A_eff|| (C1 stability)"),
                    ("fig_scale_disagreement.png", "Scale-disagreement heatmap (C2 diagnostic)"),
                    ("fig_spread_vs_error.png", "Spread-vs-error (C2 UQ signal)")]:
        checks.append((_nonempty(FIG / fn), lab, str(FIG / fn)))

    # --- deferred (NOT generated here, NOT counted as missing): non-cosmetic UQ calibration
    # (quantile/conformal: PICP/MPIW/coverage/WIS/CRPS/reliability) is a future-work deliverable.
    # Tracked separately so it never inflates the current-paper miss count.
    reliability_deferred = not _nonempty(FIG / "fig_reliability.png")

    # --- efficiency coverage: CeNN variants must be profiled at the anchor (the bug that blanked them) ---
    cenn_eff = len(glob.glob(str(EFF / f"CeNN_*__{anchor_ds}__H{anchor_h}.json")))
    checks.append((cenn_eff >= len(CENN_VARIANTS) - 1,
                   f"CeNN efficiency @ {anchor_ds} H{anchor_h}",
                   f"{cenn_eff}/{len(CENN_VARIANTS)} CeNN variants profiled"))

    # --- PatchTST reproduction reference present? (appendix cross-check, NOT a main-table source) ---
    # BASELINES_PUBLISHED is empty (every main-table baseline is re-run under our pipeline).
    # published/ holds the PatchTST/64 literature numbers ONLY as the appendix reproduction reference
    # (Nie et al. ICLR'23). Expect 1 model x HORIZONS x DATASETS_ALL = 28.
    pub = len(glob.glob(str(PUB / "*.json")))
    expect_repro = len(HORIZONS) * len(DATASETS_ALL)
    checks.append((pub >= expect_repro,
                   "PatchTST reproduction reference (appendix)",
                   f"{pub} published JSONs (expect >= {expect_repro}; BASELINES_PUBLISHED={BASELINES_PUBLISHED})"))

    print(f"\n{'='*70}\n  PAPER-ASSET COMPLETENESS\n{'='*70}")
    miss = 0
    for ok, lab, detail in checks:
        mark = "OK  " if ok else "MISS"
        if not ok:
            miss += 1
        print(f"  [{mark}] {lab:42s} {detail}")
    if reliability_deferred:
        print(f"  [DEFER] {'Reliability/calibration (PICP/MPIW/CRPS)':42s} "
              f"deferred to future work -- not counted as missing")
    print(f"{'='*70}")
    print(f"  {len(checks)-miss}/{len(checks)} present; {miss} MISSING"
          f"{' (+1 future-work-deferred: reliability)' if reliability_deferred else ''}")
    if miss:
        print("  -> resolve MISSING items before treating the output assets as complete.")
    # also flag manuscript-side (paper-writing) TODOs not auto-checkable
    print("  Manuscript TODOs (need your approval, not auto-generated): K-set caption {2,4,8} "
          "(drop K=16); add \\figure envs for pareto/k/tau/horizon; AMSCeNN architecture diagram; "
          "forecast-grid wiring; update the 'Forward Euler only' limitation (integrators now explored).")
    report = EXP / "paper_assets_report.md"
    lines = ["# Paper-asset completeness report", ""]
    lines += [f"- [{'x' if ok else ' '}] {lab} — {detail}" for ok, lab, detail in checks]
    report.write_text("\n".join(lines))
    print(f"  report -> {report}")
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dataset", default="ETTh1")
    ap.add_argument("--anchor-horizon", type=int, default=96)
    ap.add_argument("--skip-run", action="store_true", help="only run the completeness report")
    args = ap.parse_args()

    if not args.skip_run:
        # Clear the target figures FIRST so a stale synthetic/_demo PNG can never masquerade as a
        # freshly-generated real figure in the completeness report (the exact "I forgot to regenerate
        # this figure" trap). A figure that can't be built from current data -> absent -> reported MISSING.
        for fn in ["fig_pareto_acc_vs_macs.png", "fig_efficiency_panels.png", "fig_k_integrator.png",
                   "fig_horizon_curves.png", "fig_tau_heatmap.png", "fig_cd_diagram.png",
                   "fig_ablation_bars.png", "fig_mse_boxplots.png", "fig_forecast_grid.png",
                   # UQ/diagnostic figures (generators exist, pipelines not built) — clear any stale
                   # _demo so the report honestly shows them MISSING until a real pipeline feeds them.
                   "fig_reliability.png", "fig_scale_disagreement.png", "fig_spread_vs_error.png",
                   "fig_contraction.png"]:
            (FIG / fn).unlink(missing_ok=True)
        run(["experiments/aggregate.py"], "1/3 aggregate (CSVs + coverage)")
        run(["experiments/make_tables.py", "--anchor-dataset", args.anchor_dataset,
             "--anchor-horizon", str(args.anchor_horizon)], "2/3 tables + significance")
        run(["experiments/analysis/make_figures.py", "--dataset", args.anchor_dataset,
             "--horizon", str(args.anchor_horizon)], "3/3 figures")

    completeness_report(args.anchor_dataset, args.anchor_horizon)


if __name__ == "__main__":
    main()
