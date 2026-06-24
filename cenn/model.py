import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CeNNCell1D(nn.Module):    
    def __init__(
        self,
        channels: int,
        neighborhood: int = 3,
        alpha_init: float = 0.9,
        enforce_bistability: bool = False,
        cross_channel: bool = False  # allow cross-feature interaction
    ):
        super().__init__()
        assert neighborhood % 2 == 1, "Neighborhood size must be odd"

        # Use symmetric center padding; causality comes from the mask (future taps = 0).
        center  = neighborhood // 2
        padding = center
        groups  = 1 if cross_channel else channels  # full or depthwise

        self.A = nn.Conv1d(
            channels, channels, neighborhood, padding=padding,
            groups=groups, bias=False
        )
        self.B = nn.Conv1d(
            channels, channels, neighborhood, padding=padding,
            groups=groups, bias=False
        )
        self.I = nn.Parameter(torch.zeros(1, channels, 1))
        
        self.enforce_bistability = enforce_bistability
        self.cross_channel = cross_channel

        # Causal mask: keep current and past, zero future taps
        causal_mask = torch.zeros(neighborhood)
        causal_mask[:center + 1] = 1.0
        self.register_buffer('causal_mask', causal_mask.view(1, 1, -1))

        # Learnable alpha in (0,1): alpha = sigmoid(alpha_logit)
        # initialize alpha_logit = logit(alpha_init)
        alpha_logit_init = math.log(alpha_init) - math.log(1.0 - alpha_init)
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.zeros_(self.I)

    def forward(self, v: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v: hidden state [B, C, L]
        u: external input [B, C, L]
        returns: (next state v', output y)
        """
        # Integration parameters (Form A): alpha = 1 - h, beta = h
        alpha = torch.sigmoid(self.alpha_logit)  # α ∈ (0,1)
        beta  = 1.0 - alpha                      # β = 1 - α

        # CeNN output nonlinearity: tanh instead of the original PWL f(x)=clamp(x,-1,1).
        # tanh provides smooth gradients everywhere; the PWL has zero gradient for |x|>1
        # which can stall learning. Trade-off: tanh lacks the unity-gain linear region |x|<1.
        u = u.contiguous()
        y = torch.tanh(v).contiguous()

        # Apply causal mask to kernels (PixelCNN-style masking)
        wA = self.A.weight * self.causal_mask.contiguous()
        wB = self.B.weight * self.causal_mask.contiguous()

        # Optional: enforce center tap ≥ 1 (heuristic bistability for depthwise case)
        if self.enforce_bistability and not self.cross_channel:
                c = self.A.kernel_size[0] // 2
                left   = wA[:, :, :c]
                center = torch.clamp(wA[:, :, c], min=1.0).unsqueeze(-1)
                right  = wA[:, :, c+1:]
                wA = torch.cat([left, center, right], dim=2).contiguous()

        # Causal local convolutions with center padding (length preserved)
        feedback = F.conv1d(y, wA, padding=self.A.padding[0], groups=self.A.groups)
        control  = F.conv1d(u, wB, padding=self.B.padding[0], groups=self.B.groups)

        # Form A: move the leak into alpha; residual has ONLY the drive terms
        drive = feedback + control + self.I  # (A*y) + (B*u) + I

        # Forward Euler (Form A): v_{t+1} = alpha*v_t + beta*drive
        v_next = alpha * v + beta * drive
        y_next = torch.tanh(v_next)

        return v_next, y_next
class CeNNLayer1D(nn.Module):
    def __init__(self, channels: int, K: int, **cell_kwargs):
        super().__init__()
        self.K = K
        self.cell = CeNNCell1D(channels, **cell_kwargs)

    def forward(self, u: torch.Tensor):  # u [B, C, L]
        v = torch.zeros_like(u)
        for _ in range(self.K):         # hidden state
            v, y = self.cell(v, u)
        return y
    
class ResidualNorm1D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # x is [B, C, L] -> transpose to [B, L, C] for LayerNorm
        x_t = x.transpose(1, 2)
        normed = self.norm(x_t).transpose(1, 2)
        return x + self.drop(sublayer(normed))
        
class CeNNBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        K: int,
        num_layers: int,
        dropout: float,
        **cell_kwargs  # Pass all CeNN cell parameters directly
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ResidualNorm1D(
                channels=channels,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.ce_nn_layers = nn.ModuleList([
            CeNNLayer1D(
                channels=channels,
                K=K,
                **cell_kwargs
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x):
        for layer, cenn in zip(self.layers, self.ce_nn_layers):
            x = layer(x, cenn)  # ResidualNorm1D wraps CeNNLayer1D
        return self.final_norm(x.transpose(1, 2)).transpose(1, 2)

class CeNNStack1D(nn.Module):
    def __init__(self, N: int, channels: int, K: int, dropout: float, num_layers: int = 3, **cell_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            CeNNBlock1D(
                channels=channels, 
                K=K,
                num_layers=num_layers,
                dropout=dropout,
                **cell_kwargs
            ) for _ in range(N)
        ])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class CeNNModel1D(nn.Module):
    def __init__(self, n_features: int, seq_length: int, pred_length: int, *,
                 hidden_dim: int,
                 N: int, K: int, dropout: float,
                 num_layers: int, output_multiplier: int = 1, **cell_kwargs):
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.n_features = n_features
        self.output_multiplier = output_multiplier

        if hidden_dim is None:
            hidden_dim = n_features

        # Learnable input projection to latent_dim
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # CeNN stack works in latent space
        self.stack = CeNNStack1D(
            N=N,
            channels=hidden_dim,
            K=K,
            dropout=dropout,
            num_layers=num_layers,
            **cell_kwargs
        )

        # Prediction head: project time dimension, then a 2-layer MLP
        self.pred_proj = nn.Linear(seq_length, pred_length)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features * output_multiplier)
        )

    def forward(self, x):  # x [B, L, C]
        # 1) Project features to latent space
        x = self.input_dropout(self.input_proj(x))           # [B, L, hidden_dim]
        u = x.permute(0, 2, 1)           # -> [B, hidden_dim, L]

        # 2) CeNN stack
        y = self.stack(u)                 # -> [B, hidden_dim, L]

        # 3) Project temporal dimension to prediction length
        y_proj = self.pred_proj(y)        # [B, hidden_dim, pred_length]

        # 4) Apply prediction head to each time step
        y_proj = y_proj.permute(0, 2, 1)  # [B, pred_length, hidden_dim]
        B, H, C = y_proj.shape
        out = self.pred_head(y_proj.reshape(B * H, C)).reshape(B, H, self.n_features * self.output_multiplier)
        return out