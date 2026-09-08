# Reproducing AMS-CeNN

A one-command path to regenerate the reported tables and figures from the atomic per-run result
JSONs, plus the from-scratch protocol. The model is AMS-CeNN integrated into a pinned fork of
NeuralForecast (v3.1.9); `neuralforecast/models/cenn.py` + the vendored math in `neuralforecast/cenn/`.

## 1. Environment
- Python 3.10–3.12, PyTorch ≥ 2.0 (CPU works for analysis/regeneration; GPU for training).
- `pip install -e .` (or `uv pip install -e ".[dev]" --torch-backend auto`).
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set automatically by `runner.py`).
- Every result JSON carries a `provenance` block (python, torch, cuda, git commit) for traceability.

## 2. Data
All seven benchmarks auto-download via `datasetsforecast.LongHorizon2` on first use (ETTh1/h2,
ETTm1/m2, Weather, Electricity, Traffic), delivered globally train-z-scored (the standard LTSF
protocol). No manual download needed.

## 3. Protocol (`config.py` is the single source of truth)
`input_size=512`, horizons `{96,192,336,720}`, fixed train/val/test splits, stride-1 sliding-window
evaluation, `max_steps=1000`. Seeds: baselines `{1,42,123}`; headline AMS-CeNN `{1,42,123,7,2026}`.
No pipeline scaler is applied to any method (`scaler_type="identity"`): the headline AMS-CeNN
variant `AMS-Anc` anchors each window to its last value inside the model (`revin=True,
revin_mode="last_only"`), and the baselines that carry an in-model instance normalization keep it.
`CENN_MAIN_VARIANT` and the `ROLES` map in `config.py` name the headline and every ablation row;
the earlier min-max configuration `C1C2-Skip-K2` is kept as the normalization-ablation row.

## 3a. Included result data (no retraining needed)
The per-seed result JSONs of every run reported in the paper are tracked in this repository:
- `experiments/results/` — one JSON per `(model, dataset, horizon, seed)`: test MSE/MAE, parameter
  count, validation loss, configuration and provenance (12 methods, all AMS-CeNN variants, 5 seeds
  for the headline);
- `experiments/efficiency/` — parameter count, multiply-accumulates, latency and memory per
  `(model, dataset, horizon)` (`_repeats/` holds the back-to-back latency repeats);
- `experiments/_robustness/` — the input-perturbation study, one JSON per `(cell, condition)` plus
  the aggregated `robustness_degradation.csv`.
Per-window predictions, gate values and operator norms (`experiments/artifacts/`) and the trained
checkpoints (`experiments/checkpoints/`) are not tracked because of their size. With a checkpoint
(from a from-scratch run, Section 4) `python -m experiments.analysis.regen_artifacts` recomputes the
artifacts without retraining; the robustness driver, `pathway_energy.py` and `regen_artifacts.py`
all need checkpoints.

## 4. From-scratch run (GPU) — atomic, skip-if-exists, resumable
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

## 5. One-command regeneration (from existing result JSONs — NO retraining)
```bash
bash experiments/regenerate_all.sh
```

## 6. Reproducibility studies
- **Robustness to input perturbations** — `python -m experiments.run_robustness` (reuses trained
  checkpoints, perturbs the test-window input pre-scaler, scores against clean targets), then
  `python -m experiments.analysis.aggregate_robustness` and
  `python -m experiments.analysis.plot_robustness_gate`.
- **Gate-variation** — `python -m experiments.run_gate_probe` (does the bounded gate adapt on a
  synthetic non-stationary signal?) and `python -m experiments.run_gate_context` (a context/volatility-aware
  gate variant on a heteroscedastic task).
- **Receptive field** — `python -m experiments.analysis.plot_receptive_field`.

## 7. Result → paper mapping
| Artifact | Generator |
|---|---|
| Aggregates, significance (Friedman/Nemenyi), LaTeX-ready tables | `aggregate.py --status --latex`, `make_tables.py` |
| Manuscript tables (main results, profile, per-cell regret, generic trunk, normalization row, cross-channel) and the prose numbers | `analysis/make_paper_tables.py` → `aggregated/paper_tables/` |
| Ablation build-up figure | `analysis/make_ablation_buildup_fig.py` |
| CD diagram, no-champion heatmap, worst-case robustness, contraction, forecast panels | `analysis/make_figures.py` |
| Gate-retention figure (all seven datasets) | `analysis/make_tau_all7.py` (needs `artifacts/tau`, recomputed from checkpoints by `analysis/regen_artifacts.py`) |
| Forecast overlay (AMS-CeNN vs DLinear/TSMixer) | `analysis/make_new1_overlay.py` (needs `artifacts/predictions`) |
| Robustness degradation curves (incl. dead-sensor family) + gate-variation figure | `analysis/aggregate_robustness.py` → `analysis/plot_robustness_gate.py` |
| Pathway energy decomposition (residual / cell / cross term) | `analysis/pathway_energy.py` |
| Receptive-field figure | `analysis/plot_receptive_field.py` |

## 8. Citation
Every tagged release is archived on Zenodo (concept DOI 10.5281/zenodo.21041026, all versions).
See `CITATION.cff` and the repository `README.md`.
