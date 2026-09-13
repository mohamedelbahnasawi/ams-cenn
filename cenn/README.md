# cenn-forecasting

PyTorch implementation of AMS-CeNN, an auditable, stability-certified cellular residual network for
long-horizon multivariate time-series forecasting, built on the Cellular Neural Network (CeNN) of Chua and Yang.

AMS-CeNN adds three components to a full-lookback linear forecast:

- C1, a bounded retention gate. A learned, per-channel gate bounded in [0.5, 0.99] gives the cellular
  recurrence a per-step contraction guarantee that can be read from the trained parameters.
- C2, parallel dilated branches. Four cellular branches with dilation rates {1, 2, 4, 8} refine the forecast
  at four temporal resolutions.
- C3, last-value normalization with a zero-initialized linear residual. Each window is normalized to its
  last observed value inside the model. A full-lookback linear map on the normalized window carries the
  long-range structure, and the cellular pathway adds a bounded nonlinear correction on top.

The benchmark evaluation, the baselines and all per-seed result data live in the accompanying repository:
<https://github.com/mohamedelbahnasawi/ams-cenn> (archived at <https://doi.org/10.5281/zenodo.21041026>).

## Installation

```bash
pip install cenn-forecasting
```

## Quick start

```python
import torch
from cenn_forecasting import ams_cenn, count_parameters

model = ams_cenn(n_series=7, input_size=512, h=96)   # the configuration evaluated in the paper
print(count_parameters(model))                        # 106197

x = torch.randn(32, 512, 7)                           # [batch, lookback, series]
y = model(x)                                          # [32, 96, 7]
```

Any option of the underlying `CeNNModel1D` can be overridden, for example the variant without last-value
normalization used for autoregressive rollout in the paper's case study:

```python
model = ams_cenn(7, 512, 96, revin=False)
```

### Fit and forecast without NeuralForecast

```python
import numpy as np
from cenn_forecasting import AMSCeNNForecaster

y = np.load("series.npy")                 # [T, C], standardized by the caller
f = AMSCeNNForecaster(input_size=512, h=96, max_steps=1000).fit(y)
forecast = f.predict(y)                   # [96, C], continues the last 512 rows of y
```

`AMSCeNNForecaster` trains with MSE, AdamW (weight decay 0.01), a cosine learning-rate schedule, gradient
clipping and early stopping on a chronological validation split. It is a convenience wrapper; the numbers
reported in the paper come from the NeuralForecast-based runner in the repository above.

### Inside NeuralForecast

The repository is a pinned fork of NeuralForecast (v3.1.9) with the same core integrated as
`neuralforecast.models.CeNN`; see its README for the pipeline usage.

## The conference-era forecaster

The forecaster of the 0.x releases (a plain cellular cell with a fixed integration constant, no gate,
no branches, no residual) is unchanged and available as

```python
from cenn_forecasting.legacy import CeNNModel1D
```

## Architecture

```
Input [B, L, C]
    ├── last-value normalization (per channel)
    ├── zero-initialized linear residual  Linear(L -> H)            ─┐
    ├── input projection -> [B, hidden, L]                            │
    │     └── four cellular branches, dilation 1/2/4/8, K = 2 steps   │
    │           v' = α·v + (1-α)·(A ∗ tanh(v) + B ∗ u + I), y = tanh(v')│
    │           α bounded in [0.5, 0.99], ‖A‖ capped at 0.9            │
    ├── branch mean -> temporal projection Linear(L -> H) -> head     │
    └── sum ─────────────────────────────────────────────────────────┘
    └── inverse last-value normalization -> [B, H, C]
```

## Citation

Please cite the accompanying manuscript (under submission, 2026) and the archived software
(<https://doi.org/10.5281/zenodo.21041026>).

## References

- L. O. Chua and L. Yang, "Cellular Neural Networks: Theory," IEEE Transactions on Circuits and Systems, 1988.
- L. O. Chua and L. Yang, "Cellular Neural Networks: Applications," IEEE Transactions on Circuits and Systems, 1988.

## License

Apache 2.0
