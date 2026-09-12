"""Does a cellular recurrence capture nonlinear dynamics when it is allowed to run?

Motivation: on the LTSF benchmarks the anchored linear residual is unpredictable, so no nonlinear
pathway can add accuracy there. The shipped trunk is also near-linear by construction (K=2 steps
of a contractive map, input held as a constant bias). Before any redesign, measure on data that
is nonlinear BY CONSTRUCTION whether (a) more iterations and (b) relaxing the contraction
certificate buy anything over a linear map and over a generic MLP.

Data (standardized, chronological 70/10/20 split, stride-1 windows):
  Mackey-Glass  tau=17, univariate, the classic nonlinear-forecasting benchmark.
  Lorenz-96     F=8, N=8 coupled variables, chaotic (Lyapunov time ~ 0.6 t.u. = 12 steps at dt=0.05).
Arms (all with a last-value anchor so the comparison is about dynamics, not level handling):
  Linear        anchored full-window linear map (== NLinear)          the floor
  MLP           2-layer MLP on the flattened window                    generic nonlinear reference
  CeNN-K2-cap   trunk-only shipped substrate: K=2, cap, bounded gate    the paper's cell
  CeNN-K8-cap   more iterations, certificate kept
  CeNN-K32-cap  many iterations, certificate kept
  CeNN-K32-free many iterations, cap OFF, gate unbounded [0,1]        "let the dynamics run"
Direct multi-step head (no recursion), MSE, Adam 1e-3, early stopping on validation, 3 seeds.
Output: experiments/analysis/chaos_probe_results.json + a printed table.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuralforecast.cenn.model import CeNNModel1D   # noqa: E402

OUT = Path(__file__).resolve().parent / "chaos_probe_results.json"


# ----------------------------------------------------------------- data
def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, n_exp=10, seed=0, burn=1000):
    rng = np.random.default_rng(seed)
    x = np.zeros(n + burn + tau)
    x[:tau + 1] = 1.2 + 0.1 * rng.standard_normal(tau + 1)
    for t in range(tau, n + burn + tau - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** n_exp) - gamma * x[t]
    return x[burn + tau:][:, None].astype(np.float32)            # (n, 1)


def lorenz96(n, N=8, F=8.0, dt=0.05, seed=0, burn=2000):
    rng = np.random.default_rng(seed)
    x = F + 0.01 * rng.standard_normal(N)
    def f(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    out = np.zeros((n + burn, N))
    for i in range(n + burn):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        out[i] = x
    return out[burn:].astype(np.float32)                           # (n, N)


def windows(arr, L, H):
    X = np.stack([arr[i:i + L] for i in range(len(arr) - L - H + 1)])
    Y = np.stack([arr[i + L:i + L + H] for i in range(len(arr) - L - H + 1)])
    return X, Y


def split(arr, L, H):
    n = len(arr); a, b = int(0.7 * n), int(0.8 * n)
    mu, sd = arr[:a].mean(0), arr[:a].std(0) + 1e-8
    z = (arr - mu) / sd
    return [windows(z[s:e], L, H) for s, e in ((0, a), (a - L, b), (b - L, n))]


# ----------------------------------------------------------------- models
class Anchored(nn.Module):
    """Last-value anchor around any (B,L,V)->(B,H,V) core."""
    def __init__(self, core):
        super().__init__(); self.core = core
    def forward(self, x):
        last = x[:, -1:, :]
        return self.core(x - last) + last


class LinearCore(nn.Module):
    def __init__(self, L, H, V):
        super().__init__(); self.H, self.V = H, V; self.lin = nn.Linear(L, H)   # shared across variables (NLinear)
    def forward(self, x):
        return self.lin(x.transpose(1, 2)).transpose(1, 2)


class MLPCore(nn.Module):
    def __init__(self, L, H, V, hid=256):
        super().__init__(); self.H, self.V = H, V
        self.net = nn.Sequential(nn.Linear(L * V, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, H * V))
    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1)).view(-1, self.H, self.V)


def cenn_core(L, H, V, K, cap, free_gate, block_norm="layernorm"):
    return CeNNModel1D(n_features=V, seq_length=L, pred_length=H, hidden_dim=64, N=1, K=K, dropout=0.1,
                       num_layers=4, adaptive_tau=True, multiscale_mode="parallel_ensemble",
                       integrator="euler", linear_skip=False, spectral_cap=cap, block_norm=block_norm,
                       alpha_min=0.0 if free_gate else 0.5, alpha_max=1.0 if free_gate else 0.99)



class PatchTFCore(nn.Module):
    """Compact PatchTST-style core: channel-independent, patches of length P (stride P) embedded to
    d_model, a TransformerEncoder, flatten, linear head to H. RevIN is played by the outer anchor."""
    def __init__(self, L, H, V, d_model=64, n_layers=2, n_heads=4, d_ff=128, patch=8):
        super().__init__(); self.H, self.V, self.P = H, V, patch
        self.n_tok = L // patch
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(self.n_tok * d_model, H)
    def forward(self, x):                                   # (B, L, V)
        B, L, V = x.shape
        z = x.transpose(1, 2).reshape(B * V, L)             # channel-independent
        z = z[:, :self.n_tok * self.P].view(B * V, self.n_tok, self.P)
        z = self.enc(self.embed(z) + self.pos)              # (B*V, n_tok, d)
        y = self.head(z.reshape(B * V, -1))                 # (B*V, H)
        return y.view(B, V, self.H).transpose(1, 2)


class MixerCore(nn.Module):
    """Compact TSMixer-style core: n_blocks of [time-mixing MLP over L, channel-mixing MLP over V]
    with residuals, then a per-variable linear head L -> H."""
    def __init__(self, L, H, V, hid_t=128, hid_c=64, n_blocks=2):
        super().__init__(); self.H, self.V = H, V
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "t": nn.Sequential(nn.Linear(L, hid_t), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid_t, L)),
            "c": nn.Sequential(nn.Linear(V, hid_c), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid_c, V)),
        }) for _ in range(n_blocks)])
        self.head = nn.Linear(L, H)
    def forward(self, x):                                   # (B, L, V)
        for b in self.blocks:
            x = x + b["t"](x.transpose(1, 2)).transpose(1, 2)
            x = x + b["c"](x)
        return self.head(x.transpose(1, 2)).transpose(1, 2)


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def _size_to(target, build, widths):
    """Pick the width whose parameter count is closest to target."""
    best = min(widths, key=lambda w: abs(_n_params(build(w)) - target))
    return build(best)

def _hid_for_params(L, H, V, target):
    """Hidden width of the 2-layer MLP whose parameter count ~ target: hid*(L*V + H*V + 1 + hid) + H*V."""
    b = L * V + H * V + 1
    return max(8, int(round((-b + math.sqrt(b * b + 4 * (target - H * V))) / 2)))


ARMS = {
    "Linear":        lambda L, H, V: LinearCore(L, H, V),
    "MLP":           lambda L, H, V: MLPCore(L, H, V),
    "MLP-8k":        lambda L, H, V: MLPCore(L, H, V, hid=_hid_for_params(L, H, V, 8000)),   # parameter-matched to the CeNN (~8k)
    "CeNN-K2-cap":   lambda L, H, V: cenn_core(L, H, V, 2, True, False),
    "CeNN-K8-cap":   lambda L, H, V: cenn_core(L, H, V, 8, True, False),
    "CeNN-K32-cap":  lambda L, H, V: cenn_core(L, H, V, 32, True, False),
    "CeNN-K32-free": lambda L, H, V: cenn_core(L, H, V, 32, False, True),
    # Block-norm probe: LayerNorm inside the block replaced by a per-channel affine or nothing.
    "CeNN-K2-cap-affine":   lambda L, H, V: cenn_core(L, H, V, 2, True, False, "affine"),
    "CeNN-K2-cap-nonorm":   lambda L, H, V: cenn_core(L, H, V, 2, True, False, "none"),
    "CeNN-K32-free-affine": lambda L, H, V: cenn_core(L, H, V, 32, False, True, "affine"),
    "CeNN-K32-free-nonorm": lambda L, H, V: cenn_core(L, H, V, 32, False, True, "none"),
    # Transformer / mixer references: dense (native-ish size) and parameter-matched (~8k).
    "PatchTF":    lambda L, H, V: PatchTFCore(L, H, V, d_model=64, n_layers=2, n_heads=4, d_ff=128),
    "PatchTF-8k": lambda L, H, V: _size_to(8000, lambda w: PatchTFCore(L, H, V, d_model=w, n_layers=1, n_heads=2, d_ff=2 * w), [8, 12, 16, 20, 24, 28, 32]),
    "Mixer":      lambda L, H, V: MixerCore(L, H, V, hid_t=128, hid_c=64, n_blocks=2),
    "Mixer-8k":   lambda L, H, V: _size_to(8000, lambda w: MixerCore(L, H, V, hid_t=w, hid_c=max(4, w // 4), n_blocks=1), [8, 16, 24, 32, 40, 48, 56, 64]),
}


def train_eval(build, tr, va, te, steps, seed, device, bs=128, lr=1e-3, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Anchored(build()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, Ytr = (torch.tensor(a, device=device) for a in tr)
    Xva, Yva = (torch.tensor(a, device=device) for a in va)
    Xte, Yte = (torch.tensor(a, device=device) for a in te)
    best, best_state, bad, n = math.inf, None, 0, len(Xtr)
    for step in range(steps):
        model.train(); idx = torch.randint(0, n, (bs,), device=device)
        loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                v = float(((model(Xva) - Yva) ** 2).mean())
            if v < best - 1e-6:
                best, bad = v, 0; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        mse = float(((model(Xte) - Yte) ** 2).mean())
    return mse, best, step + 1, sum(p.numel() for p in model.parameters())


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in a.seeds.split(",")]
    datasets = {"MackeyGlass": (mackey_glass(a.n), 64, [16, 64]),
                "Lorenz96":    (lorenz96(a.n), 64, [16, 48])}
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for dname, (arr, L, Hs) in datasets.items():
        V = arr.shape[1]
        for H in Hs:
            tr, va, te = split(arr, L, H)
            for arm in a.arms.split(","):
                for seed in seeds:
                    key = f"{dname}|H{H}|{arm}|seed{seed}"
                    if key in results:
                        continue
                    t0 = time.time()
                    mse, vbest, nsteps, npar = train_eval(lambda: ARMS[arm](L, H, V), tr, va, te, a.steps, seed, device)
                    results[key] = {"mse": mse, "val": vbest, "steps": nsteps, "params": npar}
                    OUT.write_text(json.dumps(results, indent=2))
                    print(f"{key:40s} test MSE {mse:.5f}  (val {vbest:.5f}, {nsteps} steps, {npar} params, {time.time()-t0:.0f}s)", flush=True)
    # summary
    print("\n=== test MSE, mean over seeds (standardized) ===")
    for dname, (_, _, Hs) in datasets.items():
        for H in Hs:
            row = []
            for arm in ARMS:
                v = [results[k]["mse"] for k in results if k.startswith(f"{dname}|H{H}|{arm}|")]
                row.append(f"{arm} {np.mean(v):.4f}" if v else f"{arm} -")
            print(f"{dname:12s} H{H:<3d} " + " | ".join(row))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
