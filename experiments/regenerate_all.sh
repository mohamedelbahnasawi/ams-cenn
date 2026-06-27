#!/usr/bin/env bash
# One-command regeneration of the AMS-CeNN tables/figures FROM the existing result JSONs.
# No retraining: reads experiments/results/ + artifacts/ and rebuilds aggregates, tables, and
# figures. Each step is non-fatal so a missing optional input never aborts the rest.
# Usage:  bash experiments/regenerate_all.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
run () { echo; echo "=== $* ==="; "$PY" "$@" || echo "[skipped/failed: $*]"; }

# 1. aggregate atomic JSONs -> status + LaTeX-ready tables
run experiments/aggregate.py --status
run experiments/aggregate.py --latex

# 2. main tables (main results, ablation, generic-trunk, cross-channel, efficiency)
run experiments/make_tables.py

# 3. core figures (CD diagram, no-champion heatmap, worst-case robustness, tau, contraction, forecasts)
run experiments/analysis/make_figures.py
run experiments/analysis/make_tau_all7.py

# 4. robustness + gate-variation + receptive-field figures
run experiments/analysis/aggregate_robustness.py
run experiments/analysis/plot_robustness_gate.py
run experiments/analysis/plot_receptive_field.py

echo; echo "=== regeneration done. Tables/figures written under experiments/aggregated/. ==="
