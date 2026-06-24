"""S4D (diagonal state-space) forecaster for this work's state-space "family-coverage" baseline.

The SSM core (S4DKernel, S4D, DropoutNd) is the AUTHORS' official standalone implementation,
vendored VERBATIM from https://github.com/state-spaces/s4 (models/s4/s4d.py + the DropoutNd helper)
-- Gu, Gupta, Goel, Re, "On the Parameterization and Initialization of Diagonal State Space Models"
(S4D), NeurIPS 2022; Apache-2.0. The only repo-specific dependency (`from src.models.nn import
DropoutNd`) is replaced by the vendored DropoutNd below; nothing else in the S4DKernel/S4D math is
changed. S4D is pure-PyTorch (diagonal) and needs NO custom CUDA kernels (unlike full S4 / Mamba).

Only the OUTER forecaster wrapper (S4DModel1D: input embedding, layer stacking with residual+norm,
and the linear forecast head) is ours -- the unavoidable glue to run S4D under the SAME L=512 /
stride-1 / per-window-scaler protocol as the other NeuralForecast baselines. The full CeNN-vs-SSM
and CeNN-xSSM study is future work; here S4D is a single representative SSM coverage baseline.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


# --- vendored from state-spaces/s4 (the `src.models.nn.DropoutNd` helper) -------------------------
class DropoutNd(nn.Module):
    """Official S4 DropoutNd: dropout that (optionally) ties the mask across the non-feature dims."""
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if not self.training:
            return X
        if not self.transposed:
            X = rearrange(X, "b ... d -> b d ...")
        mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
        mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
        X = X * mask * (1.0 / (1 - self.p))
        if not self.transposed:
            X = rearrange(X, "b d ... -> b ... d")
        return X


# --- vendored VERBATIM from state-spaces/s4 (models/s4/s4d.py) ------------------------------------
class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """returns: (..., c, L) where c is number of channels (default 1)"""
        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None
# --- end vendored state-spaces/s4 ----------------------------------------------------------------


class S4DModel1D(nn.Module):
    """OURS: the NeuralForecast-side forecaster around the official S4D layer. Embed -> n residual
    S4D blocks (official S4D layer + residual + LayerNorm, the standard S4 block) -> linear forecast
    head. Maps [B, L, V] -> [B, H, V*c_out] (same head shape as the CeNN core)."""
    def __init__(self, n_features: int, seq_length: int, pred_length: int, *,
                 d_model: int = 128, d_state: int = 64, n_layers: int = 2,
                 dropout: float = 0.1, c_out: int = 1):
        super().__init__()
        self.n_features, self.c_out = n_features, c_out
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_drop = nn.Dropout(dropout)
        self.s4_layers = nn.ModuleList(
            [S4D(d_model, d_state=d_state, dropout=dropout, transposed=True) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.pred_proj = nn.Linear(seq_length, pred_length)
        self.pred_head = nn.Linear(d_model, n_features * c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, L, V] -> [B, H, V*c_out]
        u = self.input_drop(self.input_proj(x)).transpose(1, 2)  # [B, d_model, L] = (B, H, L)
        for layer, norm in zip(self.s4_layers, self.norms):
            z, _ = layer(u)                                      # official S4D, (B, H, L)
            u = u + z                                            # residual
            u = norm(u.transpose(1, 2)).transpose(1, 2)          # LayerNorm over d_model
        y = self.pred_proj(u).transpose(1, 2)                    # [B, H, d_model]
        return self.pred_head(y)                                 # [B, H, V*c_out]
