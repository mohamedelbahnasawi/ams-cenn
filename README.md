# AMS-CeNN — Auditable Multi-Scale Cellular Neural Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21041026.svg)](https://doi.org/10.5281/zenodo.21041026)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)

A recurrent, dynamical-systems forecaster for multivariate long-horizon time series, built on a Cellular Neural Network (CeNN) and integrated into [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast).

AMS-CeNN is a parameter-auditable nonlinear residual forecaster with a stability-certified cellular recurrence. A bounded cellular recurrent module adds nonlinear dynamics around a full-lookback linear forecasting path; the linear path is the main source of accuracy. Every window is normalized inside the model to its last observed value (last-value normalization, LVN: an NLinear-style, per-channel, invertible transform), so no pipeline scaler is applied. The model has three parts.

- C1 is a bounded integration gate. A learned, per-channel gate whose output is bounded by construction; the bound gives a provable per-step contraction guarantee on the cellular hidden-state recurrence (a local guarantee, not one for the whole network) at no measurable accuracy cost. The learned gate converges to a near-uniform regime, so what it contributes is the stability certificate and the auditability, not input adaptation.
- C2 is a multi-scale dilation ensemble. Parallel CeNN branches at dilations {1, 2, 4, 8} refine the forecast at four temporal resolutions in a single forward pass.
- The third part is a zero-initialized linear residual on the last-value-normalized window: a full-lookback linear map on the LVN window, initialized to zero, so the model starts as a pure CeNN and can fall back to a near-linear forecast on linearly dominated series. In the ablations this residual carries the accuracy: on its own it reaches mean rank 3.75 of 12, and the cellular pathway on top of it is accuracy-neutral. LVN also confines a dead input channel to that channel (healthy-channel error ratio 0.998 under a stuck sensor).

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="AMS-CeNN architecture">
</p>

## Results

The model was evaluated on seven standard multivariate LTSF benchmarks (ETT, Weather, Electricity, Traffic) at horizons {96, 192, 336, 720} under one protocol (lookback `L = 512`; Friedman + Nemenyi over `N = 28` dataset–horizon blocks).

AMS-CeNN ranks 1st of 12 methods (Friedman mean rank 3.14). PatchTST (4.54), TSMixer (4.61) and DLinear (4.93) lie inside the Nemenyi critical difference of 3.15; the other eight are significantly worse. It is non-inferior to every baseline at a pre-registered 2 % margin.

Its worst-case relative error is the lowest of the 12 evaluated models (within 6.3 % of the best method on every dataset). It degrades less than the baseline median under additive noise, outliers and missing blocks, and confines a dead sensor to its own channel. It degrades more than the median under gain errors and level shifts, and we do not claim robustness to gross distribution shift.

The gate retention and the feedback operator norms can be read from the trained parameters and are tied to the contraction guarantee. The spectral cap binds in one of 8,960 audited branch-channel pairs across the 35 audited cells.

The ETTh1, H = 96 profile has 106,197 parameters, 5.3 M multiply-accumulates and a latency of 4.65 ms per multivariate forecast on an RTX 4090. Both the multiply-accumulate count and the latency are above those of the linear baselines, so we do not position the model as compute-efficient.

## Installation

This repository is a fork of NeuralForecast v3.1.9 with AMS-CeNN integrated; install it from source:

```bash
git clone https://github.com/mohamedelbahnasawi/ams-cenn
cd ams-cenn
pip install -e .
```

The model is also available on PyPI as `cenn-forecasting` (version 1.0.0, the `cenn/` directory of this repository): `pip install cenn-forecasting` gives the same core as `neuralforecast/cenn/` with `ams_cenn()` for the shipped configuration and a standalone forecaster; the benchmark harness, baselines and result data need this repository.

## Quickstart

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import CeNN

# The configuration reported in the paper (variant AMS-Anc)
model = CeNN(
    h=96,                                  # forecast horizon
    input_size=512,                        # lookback window
    n_series=7,                            # number of series / channels
    hidden_dim=64, N=1, neighborhood=3,    # cell width, cells per channel, template radius
    num_layers=4, dropout=0.25,            # four dilation branches in the ensemble
    K=2,                                   # forward-Euler integration steps
    adaptive_tau=True,                     # C1: bounded, input-conditioned gate (stability mechanism)
    multiscale_mode="parallel_ensemble",   # C2: parallel dilation ensemble {1, 2, 4, 8}
    linear_skip=True,                      # zero-initialized linear residual
    revin=True, revin_mode="last_only",    # in-model last-value normalization (LVN) + per-channel affine
    scaler_type="identity",                # no pipeline scaler
    alpha_min=0.5, alpha_max=0.99,         # gate bounds, which give the per-step contraction
    spectral_cap=True, spectral_rho=0.9,
)

# 106,197 parameters for n_series=7, h=96 (the ETTh1 profile in the paper)
nf = NeuralForecast(models=[model], freq="h")
nf.fit(df)                                 # df with columns [unique_id, ds, y]
forecasts = nf.predict()
```

See [`experiments/config.py`](experiments/config.py) and [`experiments/runner.py`](experiments/runner.py) for the exact configuration of every variant used in the paper.

## Reproducing the paper

The `experiments/` package trains the model and all baselines, writes one result per `(model, dataset, horizon, seed)`, and builds the tables and figures. [`experiments/REPRODUCE.md`](experiments/REPRODUCE.md) describes the protocol and lists which script produces each table and figure.

```bash
# Run the main model of the paper across the benchmark (5 seeds)
python -m experiments.runner --models CeNN_AMS-Anc \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic \
    --horizons 96 192 336 720 --seeds 1 42 123 7 2026

# Regenerate every table and figure from the result JSONs (no retraining)
bash experiments/regenerate_all.sh
```

Each run writes one JSON per `(model, dataset, horizon, seed)` under `experiments/results/`; existing results are skipped, so an interrupted run can be restarted. The per-seed result JSONs of every run reported in the paper are included in this repository (`experiments/results/`, `experiments/efficiency/`, `experiments/robustness/`, and the control studies under `experiments/controls/`: lookback sweep, variable-count sweep, training-budget and frozen-residual controls, normalization A/B), so every table and figure can be regenerated without retraining; see `experiments/REPRODUCE.md`, sections 3a and 3b. The robustness, gate-variation, cross-channel, and receptive-field studies are run by `experiments/run_robustness.py`, `experiments/run_gate_probe.py`, and the scripts under `experiments/analysis/`.

## Citation

If you use AMS-CeNN, please cite:

```bibtex
@article{elbahnasawi2026amscenn,
  title   = {{AMS-CeNN}: An Auditable, Stability-Certified Cellular Residual Network for Robust Long-Horizon Engineering Forecasting},
  author  = {El Bahnasawi, Mohamed and Zekaj, Jonida and Dubatouka, Palina and Gebser, Martin and Kyamakya, Kyandoghere},
  year    = {2026}
}
```

Software archive (all versions): [doi:10.5281/zenodo.21041026](https://doi.org/10.5281/zenodo.21041026). The citation will be updated once the paper is published.

## Acknowledgments

AMS-CeNN is built on [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast) (v3.1.9) and preserves its Apache-2.0 license. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
