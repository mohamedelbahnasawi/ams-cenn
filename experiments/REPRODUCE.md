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

## 4. From-scratch run (GPU) — atomic, skip-if-exists, resumable
```bash
# main table: AMS-CeNN (5 seeds) + the baseline suite (3 seeds), all 7 datasets x 4 horizons
python experiments/runner.py --models CeNN_C1C2-Skip-K2 --seeds 1 42 123 7 2026
python experiments/runner.py --models DLinear TSMixer PatchTST iTransformer TiDE NHITS TimeMixer TCN TimesNet xLSTM S4D
# ablations (build-up, generic-trunk, cross-channel)
python experiments/runner.py --models CeNN_S0-StableBase CeNN_C1-BoundedTau CeNN_C2-MultiScaleEnsemble CeNN_C1C2-Ensemble CeNN_MLP-Skip
python experiments/runner.py --models CeNN_C1C2-Skip-K2-STAR CeNN_C1C2-Skip-K2-Pointwise CeNN_C1C2-Skip-K2-VarMix CeNN_C1C2-Skip-K2-G4
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
| Main results table; ablation; generic-trunk; cross-channel; efficiency | `make_tables.py` (+ `runner.py` variants) |
| CD diagram, no-champion heatmap, worst-case robustness, gate retention, contraction | `analysis/make_figures.py`, `analysis/make_tau_all7.py` |
| Robustness degradation curves + gate-variation figure | `analysis/aggregate_robustness.py` → `analysis/plot_robustness_gate.py` |
| Receptive-field figure | `analysis/plot_receptive_field.py` |

## 8. Citation
Tag the commit and archive on Zenodo for a citable DOI. See the repository `README.md`.
