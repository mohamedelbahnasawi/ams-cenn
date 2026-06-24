import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# C1: Adaptive Tau Gate  (Contribution 1: input-dependent integration speed)
# ---------------------------------------------------------------------------
class AdaptiveTauGate(nn.Module):
    """Input-dependent integration speed gate (channel-wise).

    Produces per-channel, per-timestep tau values in (0, 1) from the
    external input u.  When tau is large the Euler step trusts the drive
    (feedback + control) more; when tau is small the cell retains its
    previous hidden state (cautious integration).

    Uses element-wise (channel-independent) gating to respect the
    depthwise structure of CeNN convolutions.  Only 2*C parameters
    per gate instead of C*C+C for a full Linear layer.

    Parameters
    ----------
    channels : int
        Number of hidden channels (C).
    alpha_init : float
        Initial retention rate.  tau starts at (1 - alpha_init).
    """

    def __init__(self, channels: int, alpha_init: float = 0.9):
        super().__init__()
        tau_init = 1.0 - alpha_init  # e.g. 0.1 for alpha=0.9
        bias_init = math.log(tau_init) - math.log(1.0 - tau_init)
        # Per-channel element-wise gate: tau_c = sigmoid(weight_c * u_c + bias_c)
        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.full((channels,), bias_init))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : [B, C, L]  — external input in channels-first layout.
        Returns tau : [B, C, L] with values in (0, 1).
        """
        # weight and bias are [C], broadcast over B and L dims
        tau = torch.sigmoid(
            self.weight.unsqueeze(0).unsqueeze(-1) * u
            + self.bias.unsqueeze(0).unsqueeze(-1)
        )
        return tau


# ---------------------------------------------------------------------------
# VarMix: Dense topology-free cross-variable mixer
# ---------------------------------------------------------------------------
class VarMix(nn.Module):
    """Dense topology-free cross-variable mixer.

    Learns a V x V mixing matrix without assuming spatial adjacency
    or shift equivariance across variables.

    Identity-initialized with gated residual: starts as no-op,
    then learns useful mixing.  x_out = x + gate * (Linear(x) - x).
    """

    def __init__(self, n_vars: int, gate_init: float = 0.01):
        super().__init__()
        self.mix = nn.Linear(n_vars, n_vars)
        nn.init.eye_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        # Small positive init so gradients flow through both gate and mix
        # from the start (gate=0 would create a dead gradient).
        self.gate = nn.Parameter(torch.tensor([gate_init]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, V] -> [B, T, V]"""
        return x + self.gate * (self.mix(x) - x)


# ---------------------------------------------------------------------------
# CeNNCell1D  (extended with C1 adaptive_tau + C2 dilation)
# ---------------------------------------------------------------------------
class CeNNCell1D(nn.Module):
    def __init__(
        self,
        channels: int,
        neighborhood: int = 3,
        alpha_init: float = 0.9,
        enforce_bistability: bool = False,
        cross_channel: bool = False,
        channel_groups: int = 1,
        adaptive_tau: bool = False,   # C1
        dilation: int = 1,            # C2
    ):
        super().__init__()
        assert neighborhood % 2 == 1, "Neighborhood size must be odd"

        self.adaptive_tau = adaptive_tau
        self.dilation = dilation

        center  = neighborhood // 2
        padding = dilation * center   # C2: adjust padding for dilation

        # Channel grouping: channel_groups=1 → depthwise (default),
        # channel_groups=channels → full cross-channel,
        # intermediate values → grouped convolution.
        if cross_channel:
            groups = 1  # backward compat
        elif channel_groups > 1:
            assert channels % channel_groups == 0, (
                f"channels ({channels}) must be divisible by "
                f"channel_groups ({channel_groups})"
            )
            groups = channels // channel_groups
        else:
            groups = channels

        self.A = nn.Conv1d(
            channels, channels, neighborhood,
            padding=padding, dilation=dilation,
            groups=groups, bias=False,
        )
        self.B = nn.Conv1d(
            channels, channels, neighborhood,
            padding=padding, dilation=dilation,
            groups=groups, bias=False,
        )
        self.I = nn.Parameter(torch.zeros(1, channels, 1))

        self.enforce_bistability = enforce_bistability
        self.cross_channel = cross_channel

        # Causal mask: keep current and past, zero future taps
        causal_mask = torch.zeros(neighborhood)
        causal_mask[:center + 1] = 1.0
        self.register_buffer('causal_mask', causal_mask.view(1, 1, -1))

        # --- C1: Adaptive tau gate OR fixed alpha ---
        if adaptive_tau:
            self.tau_gate = AdaptiveTauGate(channels, alpha_init=alpha_init)
            self.last_tau: Optional[torch.Tensor] = None  # for visualization
        else:
            alpha_logit_init = math.log(alpha_init) - math.log(1.0 - alpha_init)
            self.alpha_logit = nn.Parameter(
                torch.tensor(alpha_logit_init, dtype=torch.float32)
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.zeros_(self.I)

    def precompute(self, u: torch.Tensor):
        """Compute input-independent terms ONCE before the K-loop.

        Returns a cache dict with alpha, beta, masked weights, and
        the pre-computed control signal (B*u + I), since u is constant
        across K iterations.
        """
        # --- Integration parameters (depend on u, not v) ---
        if self.adaptive_tau:
            tau = self.tau_gate(u)          # [B, C, L]
            self.last_tau = tau.detach()
            alpha = 1.0 - tau
            beta  = tau
        else:
            alpha = torch.sigmoid(self.alpha_logit)
            beta  = 1.0 - alpha

        # --- Masked weights (constant across K iterations) ---
        causal = self.causal_mask
        wA = self.A.weight * causal
        wB = self.B.weight * causal

        if self.enforce_bistability and not self.cross_channel:
            c = self.A.kernel_size[0] // 2
            left   = wA[:, :, :c]
            center = torch.clamp(wA[:, :, c], min=1.0).unsqueeze(-1)
            right  = wA[:, :, c + 1:]
            wA = torch.cat([left, center, right], dim=2)

        # --- Control signal B*u + I (constant across K iterations) ---
        control = F.conv1d(
            u, wB,
            padding=self.B.padding[0],
            dilation=self.dilation,
            groups=self.B.groups,
        ) + self.I

        return {
            "alpha": alpha, "beta": beta,
            "wA": wA, "control": control,
        }

    def step(self, v: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """One CeNN dynamics step using pre-computed cache.

        Only the feedback A*tanh(v) depends on the evolving state v.
        """
        feedback = F.conv1d(
            torch.tanh(v), cache["wA"],
            padding=self.A.padding[0],
            dilation=self.dilation,
            groups=self.A.groups,
        )
        v_next = cache["alpha"] * v + cache["beta"] * (feedback + cache["control"])
        return v_next, torch.tanh(v_next)

    def forward(
        self, v: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v : [B, C, L]  hidden state
        u : [B, C, L]  external input
        returns: (v_next, y_next)

        Backward-compatible single-step interface (used when K-loop
        is managed externally by CeNNLayer1D).
        """
        cache = self.precompute(u)
        return self.step(v, cache)


# ---------------------------------------------------------------------------
# CeNNLayer1D
# ---------------------------------------------------------------------------
class CeNNLayer1D(nn.Module):
    def __init__(self, channels: int, K: int, pointwise_mix: bool = False,
                 **cell_kwargs):
        super().__init__()
        self.K = K
        self.cell = CeNNCell1D(channels, **cell_kwargs)

        if pointwise_mix:
            self.pointwise = nn.Conv1d(channels, channels, 1)
            nn.init.eye_(self.pointwise.weight.squeeze(-1))
            nn.init.zeros_(self.pointwise.bias)
        else:
            self.pointwise = None

    def forward(self, u: torch.Tensor):  # u [B, C, L]
        # Pre-compute everything that doesn't depend on evolving state v.
        # tau, masked weights, and control signal (B*u + I) are constant
        # across all K iterations — compute once, reuse K times.
        cache = self.cell.precompute(u)

        v = torch.zeros_like(u)
        for _ in range(self.K):
            v, y = self.cell.step(v, cache)
        if self.pointwise is not None:
            y = self.pointwise(y)
        return y


# ---------------------------------------------------------------------------
# ResidualNorm1D
# ---------------------------------------------------------------------------
class ResidualNorm1D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x_t = x.transpose(1, 2)
        normed = self.norm(x_t).transpose(1, 2)
        return x + self.drop(sublayer(normed))


# ---------------------------------------------------------------------------
# C2: Dilation schedule helpers
# ---------------------------------------------------------------------------
def _build_dilation_schedule(num_layers: int, schedule: str) -> List[int]:
    """Return a list of dilation values, one per layer.

    Parameters
    ----------
    num_layers : int
    schedule : str
        "none"        — all layers use dilation=1  (baseline)
        "exponential" — [1, 2, 4, 8, …]  (doubles each layer)
    """
    if schedule == "none":
        return [1] * num_layers
    if schedule == "exponential":
        return [2 ** i for i in range(num_layers)]
    raise ValueError(f"Unknown dilation schedule: {schedule}")


# ---------------------------------------------------------------------------
# CeNNBlock1D  (extended with C2 dilation schedule)
# ---------------------------------------------------------------------------
class CeNNBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        K: int,
        num_layers: int,
        dropout: float,
        dilation_schedule: str = "none",  # C2
        channel_groups: int = 1,
        pointwise_mix: bool = False,
        **cell_kwargs,
    ):
        super().__init__()

        dilations = _build_dilation_schedule(num_layers, dilation_schedule)

        self.layers = nn.ModuleList([
            ResidualNorm1D(channels=channels, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.ce_nn_layers = nn.ModuleList([
            CeNNLayer1D(
                channels=channels, K=K,
                pointwise_mix=pointwise_mix,
                dilation=d, channel_groups=channel_groups,
                **cell_kwargs,
            )
            for d in dilations
        ])

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x):
        for layer, cenn in zip(self.layers, self.ce_nn_layers):
            x = layer(x, cenn)
        return self.final_norm(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# CeNNStack1D
# ---------------------------------------------------------------------------
class CeNNStack1D(nn.Module):
    def __init__(
        self,
        N: int,
        channels: int,
        K: int,
        dropout: float,
        num_layers: int = 3,
        dilation_schedule: str = "none",
        channel_groups: int = 1,
        pointwise_mix: bool = False,
        **cell_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CeNNBlock1D(
                channels=channels, K=K,
                num_layers=num_layers,
                dropout=dropout,
                dilation_schedule=dilation_schedule,
                channel_groups=channel_groups,
                pointwise_mix=pointwise_mix,
                **cell_kwargs,
            )
            for _ in range(N)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# CeNNModel1D  (top-level, now accepts adaptive_tau + dilation_schedule)
# ---------------------------------------------------------------------------
class CeNNModel1D(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        *,
        hidden_dim: int,
        N: int,
        K: int,
        dropout: float,
        num_layers: int,
        output_multiplier: int = 1,
        adaptive_tau: bool = False,           # C1
        dilation_schedule: str = "none",      # C2
        var_mix: bool = False,                # VM: cross-variable mixing
        pointwise_mix: bool = False,          # VM: channel mixing
        channel_groups: int = 1,              # Path A: grouped cross-channel
        patch_len: Optional[int] = None,      # Path A: patching
        stride: Optional[int] = None,         # Path A: patching
        head_type: str = "linear",            # Path A: "linear" or "mlp"
        **cell_kwargs,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.n_features = n_features
        self.output_multiplier = output_multiplier
        self.patch_len = patch_len

        if hidden_dim is None:
            hidden_dim = n_features

        # --- VarMix: cross-variable coupling (before embedding) ---
        self.var_mixer: Optional[VarMix] = None
        if var_mix and n_features > 1:
            self.var_mixer = VarMix(n_features)

        # --- Input embedding ---
        if patch_len is not None:
            # Patching: unfold input into patches, project each patch
            if stride is None:
                stride = patch_len  # non-overlapping by default
            self.stride = stride
            # Pad input so last patch is complete
            self.pad_len = (
                stride - (seq_length - patch_len) % stride
            ) % stride
            padded_len = seq_length + self.pad_len
            self.num_patches = (padded_len - patch_len) // stride + 1
            self.patch_proj = nn.Linear(n_features * patch_len, hidden_dim)
            self.input_dropout = nn.Dropout(dropout)
            stack_seq_len = self.num_patches
        else:
            self.input_proj = nn.Linear(n_features, hidden_dim)
            self.input_dropout = nn.Dropout(dropout)
            stack_seq_len = seq_length

        self.stack = CeNNStack1D(
            N=N,
            channels=hidden_dim,
            K=K,
            dropout=dropout,
            num_layers=num_layers,
            dilation_schedule=dilation_schedule,
            adaptive_tau=adaptive_tau,
            pointwise_mix=pointwise_mix,
            channel_groups=channel_groups,
            **cell_kwargs,
        )

        # --- Prediction head ---
        if head_type == "mlp":
            self.pred_proj = nn.Sequential(
                nn.Linear(stack_seq_len, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_length),
            )
        else:
            self.pred_proj = nn.Linear(stack_seq_len, pred_length)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features * output_multiplier),
        )

    def forward(self, x):  # x [B, L, V]
        # --- VarMix: cross-variable coupling ---
        if self.var_mixer is not None:
            x = self.var_mixer(x)

        if self.patch_len is not None:
            # Patching path
            B, L, C = x.shape
            # Pad if needed (replicate last timestep)
            if self.pad_len > 0:
                x = F.pad(x, (0, 0, 0, self.pad_len), mode='replicate')
            # Unfold into patches: [B, num_patches, patch_len, C]
            x = x.unfold(1, self.patch_len, self.stride)  # [B, num_patches, C, patch_len]
            x = x.permute(0, 1, 3, 2)                     # [B, num_patches, patch_len, C]
            x = x.reshape(B, self.num_patches, -1)         # [B, num_patches, patch_len * C]
            x = self.input_dropout(self.patch_proj(x))     # [B, num_patches, hidden_dim]
            u = x.permute(0, 2, 1)                         # [B, hidden_dim, num_patches]
        else:
            x = self.input_dropout(self.input_proj(x))
            u = x.permute(0, 2, 1)

        y = self.stack(u)

        y_proj = self.pred_proj(y)

        y_proj = y_proj.permute(0, 2, 1)
        B, H, C = y_proj.shape
        out = self.pred_head(y_proj.reshape(B * H, C)).reshape(
            B, H, self.n_features * self.output_multiplier
        )
        return out
