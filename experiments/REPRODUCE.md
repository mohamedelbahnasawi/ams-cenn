# Reproducing AMS-CeNN

This document describes how to regenerate the reported tables and figures from the per-run result
JSONs, and how to rerun the experiments from scratch. The model is AMS-CeNN integrated into a pinned fork of
NeuralForecast (v3.1.9): the NeuralForecast wrapper is `neuralforecast/models/cenn.py` and the model itself is in `neuralforecast/cenn/`.

## 1. Environment
- Python 3.10–3.12, PyTorch ≥ 2.0 (CPU works for analysis/regeneration; GPU for training).
- `pip install -e .` (or `uv pip install -e ".[dev]" --torch-backend auto`).
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set automatically by `runner.py`).
- Every result JSON carries a `provenance` block (python, torch, cuda, git commit).

## 2. Data
All seven benchmarks auto-download via `datasetsforecast.LongHorizon2` on first use (ETTh1/h2,
ETTm1/m2, Weather, Electricity, Traffic), delivered globally train-z-scored (the standard LTSF
protocol). No manual download needed.

## 3. Protocol (defined in `config.py`)
`input_size=512`, horizons `{96,192,336,720}`, fixed train/val/test splits, stride-1 sliding-window
evaluation, `max_steps=1000`. Seeds: baselines `{1,42,123}`; headline AMS-CeNN `{1,42,123,7,2026}`.
No pipeline scaler is applied to any method (`scaler_type="identity"`): the headline AMS-CeNN
variant `AMS-Anc` (the code's name for the last-value-normalization configuration, called LVN in the paper)
normalizes each window to its last value inside the model (`revin=True, revin_mode="last_only"`), and the baselines that carry an in-model instance normalization keep it.
`CENN_MAIN_VARIANT` and the `ROLES` map in `config.py` name the headline and every ablation row;
the earlier min-max configuration `C1C2-Skip-K2` is kept as the normalization-ablation row.

## 3a. Result data included in the repository
The per-seed result JSONs of every run reported in the paper are tracked in this repository:
- `experiments/results/` — one JSON per `(model, dataset, horizon, seed)`: test MSE/MAE, parameter
  count, validation loss, configuration and provenance (12 methods, all AMS-CeNN variants, 5 seeds
  for the headline);
- `experiments/efficiency/` — parameter count, multiply-accumulates, latency and memory per
  `(model, dataset, horizon)` (`_repeats/` holds the back-to-back latency repeats);
- `experiments/robustness/` — the input-perturbation study, one JSON per `(cell, condition)` plus
  the aggregated `robustness_degradation.csv`.
Per-window predictions, gate values and operator norms (`experiments/artifacts/`) and the trained
checkpoints (`experiments/checkpoints/`) are not tracked because of their size. With a checkpoint
(from a from-scratch run, Section 4) `python -m experiments.analysis.regen_artifacts` recomputes the
artifacts without retraining; the robustness driver, `pathway_energy.py` and `regen_artifacts.py`
all need checkpoints.

## 3b. Control and appendix result data (`experiments/controls/`)
- `experiments/results/` also holds the stability ablations on the headline (`CeNN_AMS-Anc-GateUnbounded`,
  `CeNN_AMS-Anc-CapOff`) and the integrator check at K=2 (`CeNN_AMS-Anc-Heun`, `-ExpEuler`, `-RK4`), ETT, four
  horizons, seeds {1,42,123};
- `experiments/controls/lookback/L{96,192,336}/results/` — the lookback sweep of the headline (`CeNN_AMS-Anc`) and its
  no-residual arm (`CeNN_C1C2-Anc`) on ETTh1/ETTh2/Weather at H=96 (the L=512 cells are in `experiments/results/`);
- `experiments/controls/variables/V{8,...,256}/results/`, `experiments/controls/budget_3000/results/`,
  `experiments/controls/frozen_residual_3000/results/` —
  the variable-count sweep on Electricity (`CENN_VAR_SUBSET=V`, `CeNN_C1C2-Ensemble` vs `DLinear`), the 3000-step
  training-budget control on Traffic (`CENN_MAX_STEPS=3000`) and the 3000-step frozen-residual control on ETTh2 H720
  (`CeNN_FrozenSkip-TrainTrunk`); these three use the earlier min-max configuration;
- `experiments/analysis/chaos_probe_results.json`, `residual_nonlinearity_results.json` — the synthetic nonlinear
  test and the residual-predictability test (`chaos_probe.py`, `residual_nonlinearity.py`);
- `experiments/controls/scaler_ab/` — the baselines re-run under the
  pipeline min-max scaler for the normalization-fairness table (`make_preproc_table.py`);
- `experiments/aggregated/cell_contribution/` — the per-window cellular-contribution cache behind the
  contribution figure (`plot_cell_contribution.py`; recomputing it needs checkpoints).

`residual_nonlinearity.py` additionally needs `pip install statsmodels scikit-learn`.

`python experiments/analysis/lvn_ablation_summary.py` prints the lookback, stability and integrator summaries
from these files; `make_appendix_figs.py` draws the lookback-sweep and gate-adaptivity figures.
To regenerate the data:
```bash
python experiments/runner.py --models CeNN_AMS-Anc-GateUnbounded CeNN_AMS-Anc-CapOff --datasets ETTh1 ETTh2 ETTm1 ETTm2 --seeds 1 42 123
python experiments/runner.py --models CeNN_AMS-Anc-Heun CeNN_AMS-Anc-ExpEuler CeNN_AMS-Anc-RK4 --datasets ETTh1 ETTh2 ETTm1 ETTm2 --seeds 1 42 123
for L in 96 192 336; do CENN_INPUT_SIZE=$L python experiments/runner.py --models CeNN_AMS-Anc CeNN_C1C2-Anc --datasets ETTh1 ETTh2 Weather --horizons 96 --seeds 1 42 123; done
for V in 8 16 32 64 128 256; do CENN_VAR_SUBSET=$V python experiments/runner.py --models CeNN_C1C2-Ensemble DLinear --datasets Electricity --seeds 1 42 123; done
CENN_MAX_STEPS=3000 CENN_EXP_DIR=experiments/controls/budget_3000 python experiments/runner.py --models CeNN_C1C2-Skip-K2 DLinear NHITS TSMixer --datasets Traffic --horizons 96 720 --seeds 1
```

## 4. Running the experiments from scratch (GPU)
```bash
# main table: AMS-CeNN (5 seeds) + the baseline suite (3 seeds), all 7 datasets x 4 horizons
python experiments/runner.py --models CeNN_AMS-Anc --seeds 1 42 123 7 2026
python experiments/runner.py --models DLinear TSMixer PatchTST iTransformer TiDE NHITS TimeMixer TCN TimesNet xLSTM S4D
# ablations (build-up, residual-only, generic-trunk, unroll depth, cross-channel, frozen residual)
python experiments/runner.py --models CeNN_S0-Anc CeNN_C1-Anc CeNN_C2-Anc CeNN_C1C2-Anc CeNN_SkipOnly-Anc CeNN_MLPSkip-Anc
python experiments/runner.py --models CeNN_AMS-Anc-K4 CeNN_AMS-Anc-K8 CeNN_FrozenSkip-TrainTrunk-Anc
python experiments/runner.py --models CeNN_AMS-Anc-STAR CeNN_AMS-Anc-Pointwise CeNN_AMS-Anc-VarMix CeNN_AMS-Anc-G4
# normalization-ablation row (the earlier configuration; its variant spec carries the pipeline min-max scaler, identity on Weather)
python experiments/runner.py --models CeNN_C1C2-Skip-K2 --seeds 1 42 123 7 2026
```
Each `(model,dataset,horizon,seed)` writes one JSON to `experiments/results/`; re-running skips
completed cells. The high-cardinality datasets (Electricity, Traffic) need a large-memory host.

## 5. Regenerating tables and figures from the result JSONs
This reads the tracked result JSONs and does not train anything.
```bash
bash experiments/regenerate_all.sh
```

## 6. Robustness, gate and receptive-field studies
- Robustness to input perturbations: `python -m experiments.run_robustness` reuses the trained
  checkpoints, perturbs the test-window input before any scaling and scores against clean targets.
  All perturbation draws are seeded from the cell's seed and the family name, so a re-run reproduces the
  released files exactly (the dead-channel sets are also recorded per cell);
  `python -m experiments.analysis.aggregate_robustness` and
  `python -m experiments.analysis.plot_robustness_gate` aggregate and plot the results.
- Gate variation: `python -m experiments.run_gate_probe` tests whether the bounded gate adapts on a
  synthetic non-stationary signal; `python -m experiments.run_gate_context` trains a context/volatility-aware
  gate variant on a heteroscedastic task.
- Receptive field: `python -m experiments.analysis.plot_receptive_field`.

## 7. Which script produces each table and figure
| Artifact | Generator |
|---|---|
| Aggregates, significance (Friedman/Nemenyi), LaTeX-ready tables | `aggregate.py --status --latex`, `make_tables.py` |
| Manuscript tables (main results, profile, per-cell regret, generic trunk, normalization row, cross-channel) and the prose numbers | `analysis/make_paper_tables.py`, written to `aggregated/paper_tables/` |
| Ablation build-up figure | `analysis/make_ablation_buildup_fig.py` |
| CD diagram, no-champion heatmap, worst-case robustness, contraction, forecast panels | `analysis/make_figures.py` |
| Gate-retention figure (all seven datasets) | `analysis/make_tau_all7.py` (needs `artifacts/tau`, recomputed from checkpoints by `analysis/regen_artifacts.py`) |
| Forecast overlay (AMS-CeNN vs DLinear/TSMixer) | `analysis/make_new1_overlay.py` (needs `artifacts/predictions`) |
| Robustness degradation curves (incl. dead-sensor family) + gate-variation figure | `analysis/aggregate_robustness.py`, then `analysis/plot_robustness_gate.py` |
| Pathway energy decomposition (residual / cell / cross term) | `analysis/pathway_energy.py` |
| Receptive-field figure | `analysis/plot_receptive_field.py` |
| Per-cell regret distribution figure | `analysis/plot_regret_distribution.py` |
| Lookback-sweep and gate-adaptivity figures | `analysis/make_appendix_figs.py` |
| Cellular-contribution figure | `analysis/plot_cell_contribution.py` (uses `aggregated/cell_contribution/`; recomputing the cache needs checkpoints) |
| Forecast decomposition (linear path vs cellular correction) | `analysis/plot_forecast_decomposition.py` (needs checkpoints) |
| Normalization-fairness table | `analysis/make_preproc_table.py`, written to `aggregated/tables/preproc_fairness.tex` |
| Lookback, stability and integrator summaries | `analysis/lvn_ablation_summary.py` |
| Per-cell worst-case regret statistic | `analysis/worst_case_regret_28cell.py --tex <manuscript.tex>` |
| Residual-predictability test, synthetic nonlinear test | `analysis/residual_nonlinearity.py`, `analysis/chaos_probe.py` (results JSONs included) |

## 8. Citation
Every tagged release is archived on Zenodo (concept DOI 10.5281/zenodo.21041026, all versions).
See `CITATION.cff` and the repository `README.md`.
