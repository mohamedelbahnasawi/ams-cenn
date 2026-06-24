# CeNN Forecasting

A PyTorch implementation of **Cellular Neural Networks (CeNN)** adapted for multivariate time-series forecasting.

## Overview

This package provides a trainable CeNN model based on the theory of Chua & Yang (1988), adapted for temporal forecasting tasks. Key features:

- **Forward Euler discretization** of the CeNN state equation with learnable step size
- **Causal masking** to prevent information leakage from future time steps
- **Depthwise or cross-channel** convolution modes for univariate/multivariate coupling
- **Pre-norm residual blocks** with LayerNorm for stable deep stacking
- Compatible with [NeuralForecast](https://github.com/Nixtla/neuralforecast) via the `CeNN` wrapper class

## Installation

```bash
pip install cenn-forecasting
```

## Quick Start

### Standalone usage

```python
import torch
from cenn_forecasting import CeNNModel1D

model = CeNNModel1D(
    n_features=7,       # number of time series
    seq_length=96,      # input window length
    pred_length=24,     # forecast horizon
    hidden_dim=64,      # latent dimension
    N=2,                # number of CeNN blocks
    K=8,                # recurrent steps per layer
    dropout=0.1,
    num_layers=2,
    neighborhood=3,     # convolution kernel size (odd)
    alpha_init=0.9,     # initial integration parameter
)

x = torch.randn(32, 96, 7)  # [batch, time, features]
out = model(x)               # [32, 24, 7]
```

### With NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import CeNN
from neuralforecast.losses.pytorch import MSE

model = CeNN(
    h=96,
    input_size=512,
    n_series=7,
    hidden_dim=64,
    N=2, K=8,
    num_layers=2,
    loss=MSE(),
    max_steps=1000,
    scaler_type='robust',
    accelerator='gpu',
)

nf = NeuralForecast(models=[model], freq='h')
nf.fit(df=train_df, val_size=val_size)
forecasts = nf.predict()
```

## Architecture

```
Input [B, L, C]
    │
    ├── Linear projection → [B, L, hidden_dim]
    ├── Permute → [B, hidden_dim, L]
    │
    ├── CeNN Stack (N blocks)
    │     └── CeNN Block
    │           ├── ResidualNorm + CeNNLayer (×num_layers)
    │           │     └── CeNNCell (K iterations)
    │           │           ├── v' = α·v + (1-α)·(A·tanh(v) + B·u + I)
    │           │           └── y = tanh(v')
    │           └── Final LayerNorm
    │
    ├── Linear temporal projection → [B, hidden_dim, H]
    ├── Permute → [B, H, hidden_dim]
    └── MLP prediction head → [B, H, C]
```

## References

- L.O. Chua and L. Yang, "Cellular Neural Networks: Theory," IEEE Trans. Circuits and Systems, 1988.
- L.O. Chua and L. Yang, "Cellular Neural Networks: Applications," IEEE Trans. Circuits and Systems, 1988.

## License

Apache 2.0
