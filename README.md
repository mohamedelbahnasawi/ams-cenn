# AMS-CeNN — Auditable Multi-Scale Cellular Neural Network

[![PyPI](https://img.shields.io/pypi/v/cenn-forecasting)](https://pypi.org/project/cenn-forecasting/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)

A recurrent, dynamical-systems forecaster for **multivariate long-horizon time series forecasting**, built on a Cellular Neural Network (CeNN) substrate and integrated into [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast).

AMS-CeNN is a **parameter-auditable nonlinear residual forecaster with a stability-certified cellular recurrence**: a bounded cellular recurrent module adds controlled nonlinear dynamics around a strong full-lookback linear forecasting path, which is the primary accuracy driver. It combines three components:

- **C1 — Bounded, stability-certified integration gate.** A learned, per-channel gate whose output is bounded by construction, giving a **provable per-step contraction guarantee** on the cellular hidden-state recurrence (a local guarantee, not a full-network one) at no measurable accuracy cost. The learned gate converges to a near-uniform regime: its value is the certified stability and auditability it provides, not input adaptation.
- **C2 — Multi-scale dilation ensemble.** Parallel CeNN branches at dilations {1, 2, 4, 8} that provide multi-resolution nonlinear refinement in a single forward pass.
- **Zero-initialized linear skip.** A DLinear-style residual initialized to zero, so the model begins as a pure CeNN and can fall back to a near-linear forecast on linearly dominated series — degrading gracefully rather than catastrophically. Ablations show this skip is the dominant accuracy driver.

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="AMS-CeNN architecture">
</p>

## Results

Evaluated across **seven standard multivariate LTSF benchmarks** (ETT, Weather, Electricity, Traffic) at horizons {96, 192, 336, 720}, under a uniform protocol (lookback `L = 512`; Friedman + Nemenyi over `N = 28` dataset–horizon blocks):

- **Accuracy** — ranks **4th of 12** methods (mean rank 5.21), within the critical-difference clique of the leader (no significant difference detected under the Nemenyi test); the rank survives a fully symmetric shared-scaler protocol (4th of 12 under a shared identity scaler).
- **Robustness** — attains the **lowest worst-case relative error of all 12 evaluated models** (within ≈ 7 % of the best method on every dataset), and degrades more gracefully than the baseline median under input contamination (noise, outliers, missing data); it is not, however, claimed robust to gross distribution shift.
- **Auditability** — the learned gate retention and feedback operator norms are directly readable from the trained parameters and tied to the contraction guarantee.
- **Footprint** — parameter-light (≈ 106 K parameters) and fast (≈ 1.7 ms per forecast); its multiply-accumulate count exceeds the linear baselines, so the model is not positioned as compute-efficient.

## Installation

```bash
pip install cenn-forecasting
```

Or from source — this repository is a fork of NeuralForecast v3.1.9 with AMS-CeNN integrated:

```bash
git clone https://github.com/mohamedelbahnasawi/ams-cenn
cd ams-cenn
pip install -e .
```

## Quickstart

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import CeNN

# Headline AMS-CeNN configuration (variant C1C2-Skip-K2)
model = CeNN(
    h=96,                                  # forecast horizon
    input_size=512,                        # lookback window
    n_series=7,                            # number of series / channels
    hidden_dim=64,
    K=2,                                   # forward-Euler integration steps
    adaptive_tau=True,                     # C1: bounded, input-conditioned gate (stability mechanism)
    multiscale_mode="parallel_ensemble",   # C2: parallel dilation ensemble {1, 2, 4, 8}
    linear_skip=True,                      # zero-initialized linear residual
    alpha_min=0.5, alpha_max=0.99,         # bounded gate -> per-step contraction
    spectral_cap=True, spectral_rho=0.9,
)

nf = NeuralForecast(models=[model], freq="h")
nf.fit(df)                                 # df with columns [unique_id, ds, y]
forecasts = nf.predict()
```

See [`experiments/config.py`](experiments/config.py) and [`experiments/runner.py`](experiments/runner.py) for the exact configuration of every variant used in the paper.

## Reproducing the paper

The full experiment suite (model + all baselines, atomic per-`(model, dataset, horizon, seed)` results, tables, and figures) is driven by the `experiments/` package. See [`experiments/REPRODUCE.md`](experiments/REPRODUCE.md) for the complete protocol and a result→figure/table mapping.

```bash
# Run the headline model across the benchmark (5 seeds)
python -m experiments.runner --models C1C2-Skip-K2 \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic \
    --horizons 96 192 336 720 --seeds 1 42 123 7 2026

# Regenerate every table and figure from the result JSONs (no retraining)
bash experiments/regenerate_all.sh
```

Each run writes one JSON per `(model, dataset, horizon, seed)` under `experiments/results/` (skip-if-exists, so campaigns are resumable). The robustness, gate-variation, cross-channel, and receptive-field studies are driven by `experiments/run_robustness.py`, `experiments/run_gate_probe.py`, and the scripts under `experiments/analysis/`.

## Citation

If you use AMS-CeNN, please cite:

```bibtex
@article{elbahnasawi2026amscenn,
  title   = {Robust, Stability-Certified Multi-Scale Cellular Neural Networks for Long-Horizon Time Series Forecasting},
  author  = {El Bahnasawi, Mohamed and Zekaj, Jonida and Dubatouka, Palina and Gebser, Martin and Kyamakya, Kyandoghere},
  year    = {2026}
}
```

> Citation details will be finalized upon publication.

## Acknowledgments

AMS-CeNN is built on [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast) (v3.1.9) and preserves its Apache-2.0 license. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
