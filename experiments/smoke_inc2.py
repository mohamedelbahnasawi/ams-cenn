#!/usr/bin/env python
"""Inc-2 empirical latent-bug gate for the CeNN variant re-cut.

Run with the project venv:
    .venv/Scripts/python.exe experiments/smoke_inc2.py

For each buildable-NOW Inc-2 variant it:
  1. constructs the model via the venv-importable neuralforecast.models.CeNN wrapper
     (the real Inc-2 entry point), using each variant's full kwargs,
  2. ASSERTS the ablation knobs (spectral_cap, alpha_min/alpha_max) actually reached
     the CeNNCell1D — proves the wrapper->cell wiring is real, not silently dropped,
  3. runs a forward pass on tiny synthetic [B, L, V] data at L=512,
  4. runs 3 train steps (AdamW) on the inner CeNNModel1D,
  5. asserts output shape == [B, H, V] and that the loss is finite / non-NaN,
  6. prints PASS/FAIL per variant and exits non-zero if any FAIL.

This is the gate that proves the never-run patch / head / var_mix / pointwise /
channel_groups forward paths execute end-to-end before they are trusted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from neuralforecast.models import CeNN
from neuralforecast.losses.pytorch import MSE
from neuralforecast.cenn.model import CeNNCell1D

# Standard LTSF protocol dims (ETTh2): L=512, V=7, H=96. Small N/K/num_layers for speed.
B, L, V, H = 4, 512, 7, 96
FIXED = dict(hidden_dim=64, N=1, K=4, num_layers=2, dropout=0.0, neighborhood=3,
             alpha_init=0.9, enforce_bistability=False, cross_channel=False,
             loss=MSE(), max_steps=5, random_seed=1)

# Full kwargs per buildable-now variant (mirror of runner.VARIANT_SPECS).
VARIANTS = {
    "S0-StableBase":              dict(adaptive_tau=False, dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "C1-BoundedTau":              dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "ABL-SpectralCapOff":         dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=False, scaler_type="minmax"),
    "ABL-Patch":                  dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=16,   stride=8,    head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "ABL-GateParam-Unbounded":    dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.0, alpha_max=1.0,  spectral_cap=True,  scaler_type="minmax"),
    "ABL-CrossVar-Pointwise":     dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=True,  var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "ABL-CrossVar-VarMix":        dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=True,  channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "ABL-ChannelGroups-G4":       dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=4, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
    "CeNN-RawBase":               dict(adaptive_tau=False, dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.0, alpha_max=1.0,  spectral_cap=False, scaler_type="minmax"),
    "APP-C2Form-DilatedTemplate": dict(adaptive_tau=True,  dilation_schedule="exponential", pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=None, stride=None, head_type="linear", alpha_min=0.5, alpha_max=0.99, spectral_cap=True, scaler_type="minmax"),
    "APP-Head-MLP":               dict(adaptive_tau=True,  dilation_schedule="none",  pointwise_mix=False, var_mix=False, channel_groups=1, patch_len=16,   stride=8,    head_type="mlp",    alpha_min=0.5, alpha_max=0.99, spectral_cap=True,  scaler_type="minmax"),
}


def run_variant(name, kw):
    torch.manual_seed(0)
    model = CeNN(h=H, input_size=L, n_series=V, **FIXED, **kw)
    inner = model.model  # CeNNModel1D; forward takes [B, L, V] directly

    # --- wiring check: did the ablation knobs reach the cell? ---
    cells = [m for m in inner.modules() if isinstance(m, CeNNCell1D)]
    if not cells:
        return False, "no CeNNCell1D found in model"
    # Assert ALL cells, not just cells[0], to avoid false confidence with N>1.
    for i, c in enumerate(cells):
        if bool(c.spectral_cap) != bool(kw["spectral_cap"]):
            return False, (f"spectral_cap NOT wired in cell[{i}]: cell={c.spectral_cap} "
                           f"want={kw['spectral_cap']}")
        if abs(c.alpha_min - kw["alpha_min"]) > 1e-9 or abs(c.alpha_max - kw["alpha_max"]) > 1e-9:
            return False, (f"alpha bounds NOT wired in cell[{i}]: cell=[{c.alpha_min},{c.alpha_max}] "
                           f"want=[{kw['alpha_min']},{kw['alpha_max']}]")
    c0 = cells[0]

    x = torch.randn(B, L, V)
    y = torch.randn(B, H, V)

    # 1) forward
    inner.eval()
    with torch.no_grad():
        out = inner(x)
    if tuple(out.shape) != (B, H, V):
        return False, f"forward shape {tuple(out.shape)} != {(B, H, V)}"
    if not torch.isfinite(out).all():
        return False, "forward output non-finite"

    # 2) three train steps
    inner.train()
    opt = torch.optim.AdamW(inner.parameters(), lr=1e-3, weight_decay=0.1)
    last = None
    for step in range(3):
        opt.zero_grad()
        pred = inner(x)
        if tuple(pred.shape) != (B, H, V):
            return False, f"train-step shape {tuple(pred.shape)} != {(B, H, V)}"
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            return False, f"loss non-finite at step {step}: {float(loss)}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
        opt.step()
        last = float(loss)
    npar = sum(p.numel() for p in inner.parameters())
    return True, f"out={tuple(out.shape)} cap={c0.spectral_cap} a=[{c0.alpha_min},{c0.alpha_max}] params={npar} loss3={last:.4f}"


def check_dispatch():
    """Smoke the runner.build_cenn() dispatch path (3-tuple, scaler, ValueError guard)."""
    from experiments.runner import build_cenn, VARIANT_SPECS
    results = []
    # Test a representative subset: one MAIN, one patch, one channel_groups=4, one cap-off.
    for v in ["S0-StableBase", "ABL-Patch", "ABL-ChannelGroups-G4", "ABL-SpectralCapOff"]:
        try:
            m, name, scaler = build_cenn(v, h=H, n_series=V, seed=1, max_steps=5, K=8)
            assert name == f"CeNN_{v}", f"name mismatch: {name}"
            assert scaler == "minmax", f"scaler mismatch: {scaler}"
            cells = [c for c in m.model.modules() if isinstance(c, CeNNCell1D)]
            spec = VARIANT_SPECS[v]
            for c in cells:
                assert bool(c.spectral_cap) == bool(spec["spectral_cap"]), \
                    f"cap mismatch in build_cenn path for {v}"
            results.append((True, v, f"name={name} scaler={scaler}"))
        except Exception as e:
            results.append((False, v, f"CRASH {type(e).__name__}: {e}"))
    # Unknown variant must raise ValueError (no silent mis-build).
    try:
        build_cenn("NONEXISTENT-XYZ", h=H, n_series=V, seed=1, max_steps=5)
        results.append((False, "unknown-variant-guard", "did NOT raise ValueError"))
    except ValueError:
        results.append((True, "unknown-variant-guard", "ValueError raised correctly"))
    except Exception as e:
        results.append((False, "unknown-variant-guard", f"wrong exception: {e}"))
    return results


def main():
    print(f"torch {torch.__version__}  |  L={L} V={V} H={H} B={B}\n")
    n_fail = 0

    print("--- CeNN wrapper forward+train ---")
    for name, kw in VARIANTS.items():
        try:
            ok, msg = run_variant(name, kw)
        except Exception as e:
            ok, msg = False, f"CRASH {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{status}] {name:30s} {msg}")

    print("\n--- runner.build_cenn() dispatch path ---")
    dispatch_results = check_dispatch()
    for ok, name, msg in dispatch_results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{status}] {name:30s} {msg}")

    total = len(VARIANTS) + len(dispatch_results)
    print(f"\n{total - n_fail}/{total} checks PASS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
