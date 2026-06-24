import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Read-out activations (the OUTPUT nonlinearity y=f(v) handed to the next layer / head).
# The FEEDBACK nonlinearity A*tanh(v) stays tanh always (bounded + 1-Lipschitz -> the
# contraction guarantee). Decoupling the read-out lets us use an unbounded, less-saturating
# activation (GELU/SiLU) for expressiveness WITHOUT touching stability (LayerNorm between
# layers bounds the unbounded read-out). "bounded dynamics, modern read-out".
_READOUTS = {
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
    "identity": lambda x: x,
}


# ---------------------------------------------------------------------------
# C1: Adaptive Tau Gate  (Contribution 1)
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

    def __init__(self, channels: int, alpha_init: float = 0.9,
                 alpha_min: float = 0.5, alpha_max: float = 0.99):
        super().__init__()
        # BOUNDED gate. Returns alpha = alpha_min + (alpha_max-alpha_min)*sigmoid(.)
        # so the retention rate stays in [alpha_min, alpha_max] (alpha_max < 1),
        # guaranteeing a per-step contraction toward v.
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # init so alpha ~= alpha_init at start (weight init 0)
        frac = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        bias_init = math.log(frac) - math.log(1.0 - frac)
        # Per-channel element-wise gate: alpha_c = bound(sigmoid(weight_c * u_c + bias_c))
        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.full((channels,), bias_init))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : [B, C, L]  — external input in channels-first layout.
        Returns alpha : [B, C, L] bounded in [alpha_min, alpha_max].
        """
        s = torch.sigmoid(
            self.weight.unsqueeze(0).unsqueeze(-1) * u
            + self.bias.unsqueeze(0).unsqueeze(-1)
        )
        return self.alpha_min + (self.alpha_max - self.alpha_min) * s


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
# STAR: O(V) cross-variable core (SOFTS-style Aggregate-Redistribute)
# ---------------------------------------------------------------------------
class STAR(nn.Module):
    """O(V) cross-variable core: STAR Aggregate-Redistribute (after SOFTS, Han et al. 2024).

    Adapted to per-timestep scalar variables: at each timestep the V variables are
    embedded (up: 1 -> d_core); a single learned SCALAR score per variable (score:
    d_core -> 1) gives ONE shared softmax weighting over the V variables (the SOFTS
    aggregate — one consensus weight per variable, not a per-channel distribution);
    the d_core core is the weighted mean over variables; each variable is then updated
    from [its own embedding ; the core] (redistribute). Variable interaction is mediated
    by the single core => O(V) per timestep, vs VarMix's dense O(V^2) V×V matrix.

    Weights are V-independent (shared across variables) => O(1) params, so the module
    transfers across datasets with different V. Note vs canonical SOFTS: SOFTS embeds a
    whole per-series token; here each variable contributes a scalar per timestep, so this
    is SOFTS STAR adapted to scalar per-timestep variables (forward compute is O(V), the
    parameter count is constant in V).

    redistribute is zero-initialized => delta == 0 => EXACT no-op at init, then learns.
    Interface matches VarMix: [B, T, V] -> [B, T, V].
    """

    def __init__(self, n_vars: int, d_core: int = 16, gate_init: float = 0.01):
        super().__init__()
        self.n_vars = n_vars  # interface parity with VarMix; weights are V-independent
        self.up = nn.Linear(1, d_core)               # per-variable scalar -> embedding
        self.score = nn.Linear(d_core, 1)            # SOFTS aggregate: one scalar score / variable
        self.redistribute = nn.Linear(2 * d_core, 1)  # [own ; core] -> scalar update
        nn.init.zeros_(self.redistribute.weight)      # delta == 0 at init -> exact no-op
        nn.init.zeros_(self.redistribute.bias)
        self.gate = nn.Parameter(torch.tensor([gate_init]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, V] -> [B, T, V]"""
        h = self.up(x.unsqueeze(-1))                  # [B, T, V, d_core]
        w = torch.softmax(self.score(h), dim=2)       # [B, T, V, 1]  one weight per variable, over V
        core = (h * w).sum(dim=2, keepdim=True)       # [B, T, 1, d_core]  (aggregate)
        core = core.expand(-1, -1, h.size(2), -1)     # [B, T, V, d_core]  (broadcast)
        combined = torch.cat([h, core], dim=-1)       # [B, T, V, 2*d_core]
        delta = self.redistribute(combined).squeeze(-1)  # [B, T, V]       (redistribute)
        return x + self.gate * delta                  # gated residual (no-op at init)


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
        alpha_min: float = 0.5,       # bounded-gate lower bound
        alpha_max: float = 0.99,      # bounded-gate upper bound (<1)
        spectral_cap: bool = True,    # cap ||A_eff|| < 1 (contraction)
        spectral_rho: float = 0.9,    # target operator-norm bound (<1)
        readout_act: str = "tanh",    # read-out nonlinearity (feedback stays tanh)
    ):
        super().__init__()
        assert neighborhood % 2 == 1, "Neighborhood size must be odd"
        if readout_act not in _READOUTS:
            raise ValueError(f"readout_act must be one of {set(_READOUTS)}, got {readout_act!r}")
        self.readout = _READOUTS[readout_act]

        self.adaptive_tau = adaptive_tau
        self.dilation = dilation
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.spectral_cap = spectral_cap
        self.spectral_rho = spectral_rho

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

        # --- C1: Adaptive tau gate OR fixed alpha (both bounded in [alpha_min, alpha_max]) ---
        if adaptive_tau:
            self.tau_gate = AdaptiveTauGate(
                channels, alpha_init=alpha_init,
                alpha_min=alpha_min, alpha_max=alpha_max,
            )
            self.last_tau: Optional[torch.Tensor] = None  # for visualization
        else:
            # Bound the fixed alpha into [alpha_min, alpha_max] too.
            frac = (alpha_init - alpha_min) / (alpha_max - alpha_min)
            frac = min(max(frac, 1e-4), 1.0 - 1e-4)
            alpha_logit_init = math.log(frac) - math.log(1.0 - frac)
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
        # --- Integration parameters (depend on u, not v); bounded in [alpha_min, alpha_max] ---
        if self.adaptive_tau:
            alpha = self.tau_gate(u)        # [B, C, L], bounded in [alpha_min, alpha_max]
            self.last_tau = (1.0 - alpha).detach()  # tau = 1 - alpha (for visualization)
            beta  = 1.0 - alpha
        else:
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_logit)
            beta  = 1.0 - alpha

        # --- Masked weights (constant across K iterations) ---
        causal = self.causal_mask
        wA = self.A.weight * causal
        wB = self.B.weight * causal

        # --- Spectral cap: clamp the feedback operator norm so ||A_eff|| <= rho < 1.
        # Per-output-channel L1 of the (masked) kernel upper-bounds the conv's
        # operator 2-norm; scaling it down enforces the contraction condition
        # alpha + beta*||A|| < 1 (alpha_max<1, rho<1) at EVERY forward, so training
        # cannot grow A into the expansive regime.
        if self.spectral_cap:
            # Sum over both in_channels_per_group (dim=1) and kernel_taps (dim=2) so the
            # total L1 across all in-channel contributions per output channel is bounded.
            # dim=2-only was correct for depthwise (in_per_group=1) but gave a 4x-loose
            # bound for ABL-ChannelGroups-G4 (channel_groups=4 -> in_per_group=4).
            kernel_l1 = wA.abs().sum(dim=(1, 2), keepdim=True)  # [Cout, 1, 1]
            wA = wA * (self.spectral_rho / kernel_l1.clamp(min=self.spectral_rho))

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

    def _feedback(self, v: torch.Tensor, cache: dict) -> torch.Tensor:
        """Compute A*tanh(v) using the cached (masked + capped) weight."""
        return F.conv1d(
            torch.tanh(v), cache["wA"],
            padding=self.A.padding[0],
            dilation=self.dilation,
            groups=self.A.groups,
        )

    def step(self, v: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-Euler step for the cell ODE dv/dt = -v + g(v), g = A*tanh(v) + control.

        With step size h = beta = 1-alpha: v_next = v + h*(-v + g) = alpha*v + beta*g.
        """
        v_next = cache["alpha"] * v + cache["beta"] * (self._feedback(v, cache) + cache["control"])
        return v_next, self.readout(v_next)

    def exp_euler_step(self, v: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exponential-Euler step: integrate the linear leak -v EXACTLY over the step.

        For dv/dt = -v + g (g = A*tanh(v) + control frozen over the step), the exact
        solution over step h is  v_next = e^{-h} v + (1 - e^{-h}) g.  With h = 1-alpha
        (the forward-Euler step size), e^{-h} = e^{alpha-1}. Unlike forward Euler
        (alpha = 1-h), this is unconditionally stable for the linear part, so the same
        accuracy holds at fewer K steps -> the MAC-reducer for the edge story.
        Costs one feedback conv per step, same as Euler.
        """
        alpha_exp = torch.exp(cache["alpha"] - 1.0)        # e^{-h}, h = 1-alpha
        beta_exp = 1.0 - alpha_exp
        g = self._feedback(v, cache) + cache["control"]
        v_next = alpha_exp * v + beta_exp * g
        return v_next, self.readout(v_next)

    def heun_step(self, v: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Heun (RK2) step: 2nd-order integration of dv/dt = -v + g(v).

        Predictor-corrector over the Euler map E(v) = alpha*v + beta*g(v):
            v_euler = E(v)                       # predictor
            v_corr  = E(v_euler)                  # corrector eval
            v_next  = 0.5 * (v + v_corr)          # RK2 average (uses v, NOT v_euler)
        Two feedback convs per step. 2nd-order accurate vs Euler's 1st-order, so it
        matches Euler accuracy at fewer K (the K-reducer for the accuracy/MAC story).
        """
        v_euler = cache["alpha"] * v + cache["beta"] * (self._feedback(v, cache) + cache["control"])
        v_corr  = cache["alpha"] * v_euler + cache["beta"] * (self._feedback(v_euler, cache) + cache["control"])
        v_next  = 0.5 * (v + v_corr)
        return v_next, self.readout(v_next)

    def rk4_step(self, v: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classical RK4 step: 4 feedback evals per K iteration (appendix / accuracy ceiling)."""
        def F(v_):
            return cache["alpha"] * v_ + cache["beta"] * (self._feedback(v_, cache) + cache["control"]) - v_
        k1 = F(v)
        k2 = F(v + 0.5 * k1)
        k3 = F(v + 0.5 * k2)
        k4 = F(v + k3)
        v_next = v + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return v_next, self.readout(v_next)

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
                 integrator: str = "euler", **cell_kwargs):
        super().__init__()
        self.K = K
        _VALID = {"euler", "exp_euler", "heun", "rk4"}
        if integrator not in _VALID:
            raise ValueError(f"integrator must be one of {_VALID}, got {integrator!r}")
        self.integrator = integrator
        self.cell = CeNNCell1D(channels, **cell_kwargs)

        if pointwise_mix:
            self.pointwise = nn.Conv1d(channels, channels, 1)
            nn.init.eye_(self.pointwise.weight.squeeze(-1))
            nn.init.zeros_(self.pointwise.bias)
        else:
            self.pointwise = None

    def forward(self, u: torch.Tensor):  # u [B, C, L]
        cache = self.cell.precompute(u)
        v = torch.zeros_like(u)
        if self.integrator == "euler":
            step_fn = self.cell.step
        elif self.integrator == "exp_euler":
            step_fn = self.cell.exp_euler_step
        elif self.integrator == "heun":
            step_fn = self.cell.heun_step
        else:  # rk4
            step_fn = self.cell.rk4_step
        for _ in range(self.K):
            v, y = step_fn(v, cache)
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
        dilation_schedule: str = "none",  # C2 dilated-template path
        channel_groups: int = 1,
        pointwise_mix: bool = False,
        integrator: str = "euler",        # integrator selector
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
                integrator=integrator,
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
# Parallel multi-scale ensemble block
# ---------------------------------------------------------------------------
class MultiScaleCeNNBlock1D(nn.Module):
    """Parallel CeNN branches at different dilations — outputs AVERAGED.

    Instead of passing dilated templates through successive layers, we run
    n_scales independent CeNN layers in parallel, each seeing the same input
    but at a different temporal scale, then average their outputs. A single
    ResidualNorm wraps the whole ensemble.

    Note: we MEAN (not sum) the branch outputs so the residual correction added
    by this block has the same per-step magnitude as a single CeNNLayer1D — i.e.
    comparable to the sequential CeNNBlock1D baseline. Summing would make
    the residual ~n_scales× larger, confounding the comparison.
    This is the parallel-ensemble *topology* arm; it differs from the sequential
    baseline in topology (and per-block graph depth).
    """

    def __init__(
        self,
        channels: int,
        K: int,
        dropout: float,
        n_scales: int = 4,
        channel_groups: int = 1,
        pointwise_mix: bool = False,
        integrator: str = "euler",
        **cell_kwargs,
    ):
        super().__init__()
        dilations = [2 ** i for i in range(n_scales)]
        self.res = ResidualNorm1D(channels=channels, dropout=dropout)
        self.branches = nn.ModuleList([
            CeNNLayer1D(
                channels=channels, K=K,
                pointwise_mix=pointwise_mix,
                integrator=integrator,
                dilation=d, channel_groups=channel_groups,
                **cell_kwargs,
            )
            for d in dilations
        ])
        self.final_norm = nn.LayerNorm(channels)

    def _ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Run all branches in parallel and average (mean) their outputs."""
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out / len(self.branches)

    def forward(self, x):
        x = self.res(x, self._ensemble)
        return self.final_norm(x.transpose(1, 2)).transpose(1, 2)

    def forward_branches(self, x):
        """Per-branch (per-scale) block outputs in isolation, for scale-disagreement UQ.
        Each = final_norm(x + branch_i(norm(x))) — the same residual+norm as forward() but
        with a single branch instead of the mean ensemble. Returns a list of [B, C, L].
        Must be called in eval mode (dropout active in train mode corrupts the spread)."""
        if self.training:
            raise RuntimeError("forward_branches() requires eval mode (model.eval()): "
                               "active dropout corrupts the scale-disagreement spread.")
        outs = []
        for branch in self.branches:
            xb = self.res(x, branch)
            outs.append(self.final_norm(xb.transpose(1, 2)).transpose(1, 2))
        return outs


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
        multiscale_mode: str = "none",    # "none" | "parallel_ensemble"
        channel_groups: int = 1,
        pointwise_mix: bool = False,
        integrator: str = "euler",        # integrator selector
        **cell_kwargs,
    ):
        super().__init__()
        _VALID_MS = {"none", "parallel_ensemble"}
        if multiscale_mode not in _VALID_MS:
            raise ValueError(f"multiscale_mode must be one of {_VALID_MS}, got {multiscale_mode!r}")

        if multiscale_mode == "parallel_ensemble":
            self.blocks = nn.ModuleList([
                MultiScaleCeNNBlock1D(
                    channels=channels, K=K,
                    dropout=dropout,
                    n_scales=num_layers,  # n_scales == num_layers: keeps block depth comparable
                    channel_groups=channel_groups,
                    pointwise_mix=pointwise_mix,
                    integrator=integrator,
                    **cell_kwargs,
                )
                for _ in range(N)
            ])
        else:
            self.blocks = nn.ModuleList([
                CeNNBlock1D(
                    channels=channels, K=K,
                    num_layers=num_layers,
                    dropout=dropout,
                    dilation_schedule=dilation_schedule,
                    channel_groups=channel_groups,
                    pointwise_mix=pointwise_mix,
                    integrator=integrator,
                    **cell_kwargs,
                )
                for _ in range(N)
            ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_branches(self, x):
        """Per-branch latents from a single parallel-ensemble block (for UQ).
        Only valid for multiscale_mode='parallel_ensemble' with N=1 (the CeNN-Full/C2 config)."""
        if len(self.blocks) != 1 or not isinstance(self.blocks[0], MultiScaleCeNNBlock1D):
            raise ValueError(
                "forward_branches requires a single parallel-ensemble block "
                "(multiscale_mode='parallel_ensemble', N=1)")
        return self.blocks[0].forward_branches(x)


# ---------------------------------------------------------------------------
# CeNNModel1D  (top-level, now accepts adaptive_tau + dilation_schedule)
# ---------------------------------------------------------------------------
class _MLPMixerTrunk(nn.Module):
    """Generic nonlinear spatiotemporal trunk — a capacity-comparable control for the CeNN stack.
    Replaces the CeNN dynamics with a generic MLP-Mixer trunk while holding input_proj, head, and
    the zero-init linear skip constant; this isolates the effect of the CeNN's specific inductive
    bias (stability guarantee + interpretability) from generic nonlinear capacity. Maps
    [B, C, L] -> [B, C, L] via stacked (time-mix, channel-mix) MLP blocks; NO recurrence,
    adaptive gate, dilation ensemble, or integrator. No multi-scale branches => forward_branches
    is intentionally absent (the control runs single-path)."""
    def __init__(self, channels: int, seq_len: int, num_layers: int, dropout: float,
                 token_hidden: int = 64):
        super().__init__()
        th = min(token_hidden, seq_len)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(channels),
                "time": nn.Sequential(nn.Linear(seq_len, th), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(th, seq_len)),
                "norm2": nn.LayerNorm(channels),
                "chan": nn.Sequential(nn.Linear(channels, channels), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(channels, channels)),
            }) for _ in range(max(2, num_layers))
        ])

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u [B, C, L] -> [B, C, L]
        x = u.transpose(1, 2)                              # [B, L, C]
        for b in self.blocks:
            y = b["norm1"](x).transpose(1, 2)              # [B, C, L]
            y = b["time"](y).transpose(1, 2)               # [B, L, C]  (time mixing, per channel)
            x = x + y
            x = x + b["chan"](b["norm2"](x))               # channel mixing, per time step
        return x.transpose(1, 2)                            # [B, C, L]


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
        dilation_schedule: str = "none",      # C2 (dilated-template path)
        multiscale_mode: str = "none",        # C2 (parallel-ensemble path)
        integrator: str = "euler",            # step integrator
        var_mix: bool = False,                # VM: dense V×V cross-variable mixing (legacy bool)
        cross_var: str = "none",              # input cross-var mixer {none, varmix, star}
        pointwise_mix: bool = False,          # VM: latent channel mixing
        channel_groups: int = 1,              # Path A: grouped cross-channel
        patch_len: Optional[int] = None,      # Path A: patching
        stride: Optional[int] = None,         # Path A: patching
        head_type: str = "linear",            # Path A: "linear" or "mlp"
        linear_skip: bool = False,            # direct raw-input linear path added to the output
        trunk_type: str = "cenn",             # "cenn" (default) or "mlp" (generic-trunk CONTROL)
        **cell_kwargs,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.n_features = n_features
        self.output_multiplier = output_multiplier
        self.linear_skip = linear_skip
        self.patch_len = patch_len
        # UQ scale-disagreement hook: set transiently by the experiment runner to
        # route forward() through a SINGLE dilation-scale branch. Default None => standard ensemble.
        self._uq_branch_idx: Optional[int] = None

        if hidden_dim is None:
            hidden_dim = n_features

        # --- Cross-variable coupling (before embedding) ---
        # Resolve the input mixer from cross_var; var_mix=True is the legacy alias
        # for cross_var="varmix" (cross_var takes precedence when set non-"none").
        resolved_cross_var = cross_var
        if cross_var == "none" and var_mix:
            resolved_cross_var = "varmix"
        if resolved_cross_var not in ("none", "varmix", "star"):
            raise ValueError(
                f"cross_var must be one of {{none, varmix, star}}, got {cross_var!r}"
            )
        self.var_mixer: Optional[nn.Module] = None
        if n_features > 1:
            if resolved_cross_var == "varmix":
                self.var_mixer = VarMix(n_features)          # dense O(V^2)
            elif resolved_cross_var == "star":
                self.var_mixer = STAR(n_features)            # O(V) core

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

        self.trunk_type = trunk_type
        if trunk_type == "mlp":
            # Generic-trunk control: a capacity-comparable MLP-Mixer replaces the CeNN dynamics;
            # input_proj + head + linear skip are unchanged. cenn_kwargs (gate/cap/integrator) are
            # intentionally unused here -- they have no meaning without the CeNN cell.
            # 2 blocks + a small token-mixing bottleneck keep the trunk near the CeNN stack's size
            # (the CeNN is unusually param-light due to weight-tying across steps/scales, so a
            # full-width MLP-Mixer would be ~4x heavier; this keeps the control to ~1.5-2x and both
            # param counts should be reported so the comparison is at comparable, not advantaged, capacity).
            self.stack = _MLPMixerTrunk(
                channels=hidden_dim, seq_len=stack_seq_len,
                num_layers=2, dropout=dropout, token_hidden=32,
            )
        elif trunk_type == "cenn":
            self.stack = CeNNStack1D(
                N=N,
                channels=hidden_dim,
                K=K,
                dropout=dropout,
                num_layers=num_layers,
                dilation_schedule=dilation_schedule,
                multiscale_mode=multiscale_mode,
                integrator=integrator,
                adaptive_tau=adaptive_tau,
                pointwise_mix=pointwise_mix,
                channel_groups=channel_groups,
                **cell_kwargs,
            )
        else:
            raise ValueError(f"trunk_type must be 'cenn' or 'mlp', got {trunk_type!r}")

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

        # Direct linear (DLinear/NLinear-style) path from the (scaler-normalized) input to the
        # forecast, added to the CeNN output. Zero-init => starts as pure CeNN and learns the linear
        # correction. Shared across variables (channel-independent). Rationale: the CeNN stack
        # smooths the signal before a linear temporal head, discarding detail a linear map keeps;
        # this parallel path recovers it. NOTE: only active for output_multiplier==1 (point losses);
        # silently disabled for probabilistic losses -- document if the paper extends to those.
        if linear_skip:
            self.skip = nn.Linear(seq_length, pred_length)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def _embed(self, x):  # x [B, L, V] -> u [B, hidden_dim, L_or_patches]
        # --- cross-variable coupling (VarMix / STAR) ---
        if self.var_mixer is not None:
            x = self.var_mixer(x)
        if self.patch_len is not None:
            # Patching path
            B, L, C = x.shape
            if self.pad_len > 0:                          # pad so the last patch is complete
                x = F.pad(x, (0, 0, 0, self.pad_len), mode='replicate')
            x = x.unfold(1, self.patch_len, self.stride)  # [B, num_patches, C, patch_len]
            x = x.permute(0, 1, 3, 2)                     # [B, num_patches, patch_len, C]
            x = x.reshape(B, self.num_patches, -1)         # [B, num_patches, patch_len * C]
            x = self.input_dropout(self.patch_proj(x))     # [B, num_patches, hidden_dim]
            return x.permute(0, 2, 1)                      # [B, hidden_dim, num_patches]
        x = self.input_dropout(self.input_proj(x))
        return x.permute(0, 2, 1)

    def _head(self, y):  # y [B, hidden_dim, L_or_patches] -> out [B, H, V*out_mult]
        y_proj = self.pred_proj(y).permute(0, 2, 1)
        B, H, C = y_proj.shape
        return self.pred_head(y_proj.reshape(B * H, C)).reshape(
            B, H, self.n_features * self.output_multiplier
        )

    def forward(self, x):  # x [B, L, V]
        # When _uq_branch_idx is set (transiently, under eval()/predict by the runner), emit ONE
        # dilation-scale branch's forecast so the full NeuralForecast pipeline (inverse-normalization
        # + (uid,ds,cutoff) keying) wraps it — the basis for the scale-disagreement / spread-vs-error
        # diagnostics. Default None => standard ensemble forward; the runner sets/resets via
        # try/finally so it can never leak into a metric eval.
        bi = self._uq_branch_idx
        if bi is not None:
            return self.forward_branches(x)[bi]
        out = self._head(self.stack(self._embed(x)))
        if self.linear_skip and self.output_multiplier == 1:
            # raw linear path: [B, L, V] -> per-variable Linear(L->H) (shared) -> [B, H, V]
            out = out + self.skip(x.transpose(1, 2)).transpose(1, 2)
        return out

    def forward_branches(self, x):  # x [B, L, V] -> [n_branches, B, H, V*out_mult]
        """Per-scale forecasts for the parallel multi-scale ensemble — the basis for
        the scale-disagreement uncertainty diagnostic. Each returned slice is the forecast
        from ONE dilation-scale branch in isolation; the spread across scales is the
        (interpretable) uncertainty signal. NOTE: this is a per-scale decomposition for
        UQ, NOT an exact additive split of forward() — the prediction head is nonlinear, and
        forward() means the branches in latent space while here each scale is headed separately.
        Only valid for multiscale_mode='parallel_ensemble' (raises otherwise).
        Call under model.eval() + torch.no_grad() — dropout must be off (enforced) or the
        per-scale spread mixes scale-disagreement with dropout noise and is non-reproducible."""
        if self.training:
            raise RuntimeError("forward_branches() requires eval mode (model.eval()): "
                               "active dropout corrupts the scale-disagreement spread.")
        with torch.no_grad():
            u = self._embed(x)
            return torch.stack([self._head(bl) for bl in self.stack.forward_branches(u)], dim=0)
