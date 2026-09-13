# Changelog

## 1.0.0 (2026-09-12)

- AMS-CeNN, the auditable, stability-certified cellular residual network of the accompanying paper, is now
  the package's main model: `ams_cenn(n_series, input_size, h)` builds the shipped configuration (bounded
  retention gate, four parallel dilated branches, last-value normalization with a zero-initialized linear
  residual, two forward-Euler micro-steps); `CeNNModel1D` exposes every option used in the paper's ablations.
- `AMSCeNNForecaster`: fit on a NumPy array and forecast without the NeuralForecast pipeline.
- The conference-era forecaster of the 0.x releases is kept unchanged as `cenn_forecasting.legacy`
  (`from cenn_forecasting.legacy import CeNNModel1D`). The top-level `CeNNModel1D` now refers to the new core,
  whose constructor takes the same positional arguments but different keyword options; this is the reason
  for the major version bump.
- Tests: parameter count of the paper profile (106,197), level equivariance of last-value normalization,
  exact agreement with the core vendored in the experiment repository.

## 0.1.0 (2025)

- Initial release: conference-era CeNN forecaster (forward-Euler cellular cell, causal masking, depthwise or
  cross-channel templates, pre-norm residual blocks).
