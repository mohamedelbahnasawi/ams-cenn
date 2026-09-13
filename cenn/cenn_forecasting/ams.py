"""AMS-CeNN: the configuration of the auditable, stability-certified cellular residual network
evaluated in the paper, and a small standalone forecaster around it.

``ams_cenn(...)`` builds the exact model evaluated in the paper (variant ``AMS-Anc`` of the experiment
runner): a bounded, input-conditioned retention gate (C1), four parallel dilated cellular branches with
dilation rates {1, 2, 4, 8} (C2), in-model last-value normalization with a zero-initialized full-lookback
linear residual (C3), two forward-Euler micro-steps, a linear temporal projection and a linear head.
For seven series, a lookback of 512 and a horizon of 96 the model has 106,197 parameters.

``AMSCeNNForecaster`` is a convenience wrapper for users who want to fit the model on a NumPy array
without the NeuralForecast pipeline: sliding windows, MSE loss, AdamW with a cosine schedule, early
stopping on a chronological validation split. The benchmark numbers in the paper come from the
NeuralForecast-based runner in the accompanying repository, not from this wrapper.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .model import CeNNModel1D

#: Configuration evaluated in the paper. Keys are ``CeNNModel1D`` keyword arguments.
AMS_CENN_CONFIG = dict(
    hidden_dim=64,
    N=1,
    K=2,
    num_layers=4,               # four dilation branches in the parallel ensemble
    dropout=0.25,
    neighborhood=3,             # template radius r = 1 -> n = 2r + 1 = 3 taps
    alpha_init=0.9,
    alpha_min=0.5,
    alpha_max=0.99,             # bounded retention gate (C1)
    spectral_cap=True,
    spectral_rho=0.9,           # feedback-template norm cap
    enforce_bistability=False,
    cross_channel=False,
    adaptive_tau=True,          # C1: input-conditioned gate
    dilation_schedule="none",
    multiscale_mode="parallel_ensemble",  # C2: dilations {1, 2, 4, 8}
    integrator="euler",
    var_mix=False,
    cross_var="none",
    pointwise_mix=False,
    channel_groups=1,
    patch_len=None,
    stride=None,
    head_type="linear",
    linear_skip=True,           # C3: zero-initialized full-lookback linear residual
    revin=True,
    revin_mode="last_only",     # C3: last-value normalization (LVN)
)


def ams_cenn(n_series: int, input_size: int, h: int, **overrides) -> CeNNModel1D:
    """Build AMS-CeNN for ``n_series`` channels, a lookback of ``input_size`` steps and a horizon of ``h``.

    Any keyword of ``CeNNModel1D`` can be overridden, e.g. ``ams_cenn(7, 512, 96, revin=False)`` gives the
    variant without last-value normalization used for autoregressive rollout in the paper's case study.
    Input to ``forward`` is ``[batch, input_size, n_series]``; output is ``[batch, h, n_series]``.
    """
    cfg = dict(AMS_CENN_CONFIG)
    cfg.update(overrides)
    return CeNNModel1D(n_features=n_series, seq_length=input_size, pred_length=h, **cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_windows(y: np.ndarray, input_size: int, h: int, stride: int = 1):
    """Sliding windows of a ``[T, C]`` array: returns ``X [N, L, C]`` and ``Y [N, H, C]``."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    T = len(y)
    idx = range(0, T - input_size - h + 1, stride)
    X = np.stack([y[i:i + input_size] for i in idx])
    Y = np.stack([y[i + input_size:i + input_size + h] for i in idx])
    return X, Y


class AMSCeNNForecaster:
    """Fit AMS-CeNN on a multivariate series and forecast ``h`` steps from the last ``input_size`` values.

    Parameters mirror the paper's training protocol where it matters (AdamW with weight decay 0.01,
    learning rate 1e-3 cosine-annealed, gradient clipping at 1.0, MSE loss, early stopping on a
    chronological validation split). Series are expected to be standardized by the caller; the model's
    in-model last-value normalization handles the per-window level.
    """

    def __init__(self, input_size: int, h: int, max_steps: int = 1000, batch_size: int = 256,
                 learning_rate: float = 1e-3, weight_decay: float = 0.01, grad_clip: float = 1.0,
                 val_fraction: float = 0.1, val_check_steps: int = 100, patience: int = 5,
                 device: Optional[str] = None, seed: int = 1, **model_overrides):
        self.input_size, self.h = input_size, h
        self.max_steps, self.batch_size = max_steps, batch_size
        self.learning_rate, self.weight_decay, self.grad_clip = learning_rate, weight_decay, grad_clip
        self.val_fraction, self.val_check_steps, self.patience = val_fraction, val_check_steps, patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model_overrides = model_overrides
        self.model: Optional[CeNNModel1D] = None
        self.history: list = []

    def fit(self, y: np.ndarray) -> "AMSCeNNForecaster":
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        n_series = y.shape[1]
        n_val = int(round(self.val_fraction * len(y)))
        train, val = (y, None) if n_val < self.input_size + self.h else (y[:-n_val], y[-n_val - self.input_size:])
        Xtr, Ytr = make_windows(train, self.input_size, self.h)
        Xva, Yva = make_windows(val, self.input_size, self.h) if val is not None else (None, None)
        torch.manual_seed(self.seed)
        self.model = ams_cenn(n_series, self.input_size, self.h, **self.model_overrides).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_steps, eta_min=1e-5)
        lossf = nn.MSELoss()
        Xt, Yt = torch.tensor(Xtr, device=self.device), torch.tensor(Ytr, device=self.device)
        best, best_state, bad = math.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for step in range(1, self.max_steps + 1):
            self.model.train()
            idx = torch.randint(0, len(Xt), (min(self.batch_size, len(Xt)),), generator=g)
            opt.zero_grad()
            loss = lossf(self.model(Xt[idx]), Yt[idx])
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step(); sched.step()
            if Xva is not None and step % self.val_check_steps == 0:
                v = self._mse(Xva, Yva)
                self.history.append((step, loss.item(), v))
                if v < best:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def _mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        self.model.eval()
        out, n = 0.0, 0
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i + 1024], device=self.device); yb = torch.tensor(Y[i:i + 1024], device=self.device)
            out += float(((self.model(xb) - yb) ** 2).sum()); n += yb.numel()
        return out / max(n, 1)

    @torch.no_grad()
    def predict(self, y_history: np.ndarray) -> np.ndarray:
        """Forecast ``h`` steps from the last ``input_size`` rows of ``y_history`` (``[T >= input_size, C]``)."""
        if self.model is None:
            raise RuntimeError("call fit() first")
        y = np.asarray(y_history, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        x = torch.tensor(y[-self.input_size:][None], device=self.device)
        self.model.eval()
        return self.model(x)[0].cpu().numpy()
