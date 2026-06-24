#!/usr/bin/env python
"""Inc-3 smoke gate: integrator selector + parallel multi-scale ensemble (MA-A3).

Run with the project venv:
    .venv/Scripts/python.exe experiments/smoke_inc3.py

Tests:
  1. All 4 integrators (euler, exp_euler, heun, rk4) forward + 3 train steps.
  2. MultiScaleCeNNBlock1D forward (parallel ensemble, n_scales=4).
  3. CeNN-Full and C2-MultiScaleEnsemble via runner.build_cenn (dispatch path).
  4. K-sweep variants (K4-Heun, K2-Heun) via build_cenn.
  5. config CENN_VARIANTS is a subset of VARIANT_SPECS (every reported variant is buildable).
  6. CENN_MAIN_VARIANT (the headline) is buildable and in the curated reported set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from neuralforecast.models import CeNN
from neuralforecast.losses.pytorch import MSE
from neuralforecast.cenn.model import CeNNCell1D, MultiScaleCeNNBlock1D

B, L, V, H = 4, 512, 7, 96
FIXED = dict(hidden_dim=64, N=1, K=4, num_layers=4, dropout=0.0, neighborhood=3,
             alpha_init=0.9, enforce_bistability=False, cross_channel=False,
             alpha_min=0.5, alpha_max=0.99, spectral_cap=True, spectral_rho=0.9,
             loss=MSE(), max_steps=3, random_seed=1)


def run_forward_train(model_inner, label):
    x = torch.randn(B, L, V)
    y = torch.randn(B, H, V)
    model_inner.eval()
    with torch.no_grad():
        out = model_inner(x)
    if tuple(out.shape) != (B, H, V):
        return False, f"shape {tuple(out.shape)} != {(B,H,V)}"
    if not torch.isfinite(out).all():
        return False, "forward non-finite"
    model_inner.train()
    opt = torch.optim.AdamW(model_inner.parameters(), lr=1e-3)
    for step in range(3):
        opt.zero_grad()
        pred = model_inner(x)
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            return False, f"loss non-finite step {step}: {float(loss)}"
        loss.backward()
        opt.step()
    return True, f"out={tuple(out.shape)} loss={float(loss):.4f}"


def check_integrators():
    results = []
    for integrator in ["euler", "exp_euler", "heun", "rk4"]:
        try:
            m = CeNN(h=H, input_size=L, n_series=V, integrator=integrator, **FIXED)
            ok, msg = run_forward_train(m.model, integrator)
            results.append((ok, f"integrator:{integrator}", msg))
        except Exception as e:
            results.append((False, f"integrator:{integrator}", f"CRASH {type(e).__name__}: {e}"))
    return results


def check_star():
    """Inc-4 STAR cross-variable core: no-op at init, O(V), distinct from VarMix."""
    from neuralforecast.cenn.model import STAR, VarMix
    results = []
    try:
        torch.manual_seed(0)
        star = STAR(n_vars=V)
        x = torch.randn(B, L, V)
        out = star(x)
        assert tuple(out.shape) == (B, L, V), f"STAR shape {tuple(out.shape)}"
        assert torch.allclose(out, x, atol=1e-6), "STAR not a no-op at init (redistribute must be zero-init)"
        # STAR params are V-INDEPENDENT (O(1) in V); O(V) refers to forward COMPUTE.
        p5 = sum(p.numel() for p in STAR(n_vars=5).parameters())
        p50 = sum(p.numel() for p in STAR(n_vars=50).parameters())
        assert p5 == p50, f"STAR params depend on V ({p5} vs {p50}) — should be V-independent"
        # VarMix is O(V^2): its param count DOES grow with V (contrast)
        vm5 = sum(p.numel() for p in VarMix(5).parameters())
        vm50 = sum(p.numel() for p in VarMix(50).parameters())
        assert vm50 > vm5, "VarMix should grow with V"
        results.append((True, "STAR(no-op init; V-indep params)", f"params V=5:{p5}==V=50:{p50}; VarMix {vm5}->{vm50}"))
    except Exception as e:
        results.append((False, "STAR(no-op init, O(V))", f"CRASH {type(e).__name__}: {e}"))
    # Build ABL-CrossVar-STAR via build_cenn -> model must carry a STAR mixer; train + grad reaches gate
    try:
        from experiments.runner import build_cenn
        m, name, scaler = build_cenn("ABL-CrossVar-STAR", h=H, n_series=V, seed=1, max_steps=3)
        mixer = m.model.var_mixer
        assert isinstance(mixer, STAR), f"var_mixer is {type(mixer).__name__}, expected STAR"
        ok, msg = run_forward_train(m.model, "ABL-CrossVar-STAR")
        assert ok, f"train failed: {msg}"
        assert mixer.gate.grad is not None and float(mixer.gate.grad.abs().sum()) > 0, \
            "STAR gate received no gradient after training (should unfreeze past step 0)"
        results.append((True, "dispatch:ABL-CrossVar-STAR", f"mixer=STAR train-ok gate.grad!=0"))
    except Exception as e:
        results.append((False, "dispatch:ABL-CrossVar-STAR", f"CRASH {type(e).__name__}: {e}"))
    # Regression: migrated ABL-CrossVar-VarMix (var_mix=True -> cross_var='varmix') still builds VarMix
    try:
        from experiments.runner import build_cenn
        m, name, scaler = build_cenn("ABL-CrossVar-VarMix", h=H, n_series=V, seed=1, max_steps=3)
        assert isinstance(m.model.var_mixer, VarMix), \
            f"ABL-CrossVar-VarMix var_mixer is {type(m.model.var_mixer).__name__}, expected VarMix"
        results.append((True, "migration:ABL-CrossVar-VarMix", "still builds VarMix"))
    except Exception as e:
        results.append((False, "migration:ABL-CrossVar-VarMix", f"CRASH {type(e).__name__}: {e}"))
    return results


def check_multiscale():
    results = []
    # Direct MultiScaleCeNNBlock1D test
    try:
        block = MultiScaleCeNNBlock1D(
            channels=64, K=4, dropout=0.0, n_scales=4,
            alpha_min=0.5, alpha_max=0.99, spectral_cap=True, spectral_rho=0.9,
            neighborhood=3, alpha_init=0.9, enforce_bistability=False, cross_channel=False,
        )
        x = torch.randn(B, 64, L)
        out = block(x)
        assert tuple(out.shape) == (B, 64, L), f"block shape {tuple(out.shape)}"
        results.append((True, "MultiScaleCeNNBlock1D", f"shape={tuple(out.shape)}"))
    except Exception as e:
        results.append((False, "MultiScaleCeNNBlock1D", f"CRASH {type(e).__name__}: {e}"))

    # C2-MultiScaleEnsemble via wrapper
    try:
        m = CeNN(h=H, input_size=L, n_series=V, multiscale_mode="parallel_ensemble",
                 adaptive_tau=False, **FIXED)
        ok, msg = run_forward_train(m.model, "C2-ensemble")
        results.append((ok, "CeNN(multiscale=parallel_ensemble)", msg))
    except Exception as e:
        results.append((False, "CeNN(multiscale=parallel_ensemble)", f"CRASH {type(e).__name__}: {e}"))
    return results


def check_dispatch():
    from experiments.runner import build_cenn, VARIANT_SPECS
    from experiments.config import CENN_VARIANTS, CENN_MAIN_VARIANT

    results = []

    # Config/spec parity: config.CENN_VARIANTS is the CURATED reported/campaign set; VARIANT_SPECS
    # is the buildable SUPERSET (it also holds exploratory gate/CAP variants we don't sweep). The
    # invariant is subset, NOT equality: every reported variant must be buildable.
    cfg, spec = set(CENN_VARIANTS), set(VARIANT_SPECS)
    if cfg <= spec:
        results.append((True, "config CENN_VARIANTS subset of VARIANT_SPECS",
                        f"{len(cfg)} reported / {len(spec)} buildable"))
    else:
        results.append((False, "config CENN_VARIANTS subset of VARIANT_SPECS",
                        f"unbuildable_in_cfg={cfg-spec}"))

    # The headline variant must be buildable AND in the curated reported set (name-agnostic so this
    # never goes stale on the next headline change).
    ok = CENN_MAIN_VARIANT in spec and CENN_MAIN_VARIANT in cfg
    results.append((ok, "CENN_MAIN_VARIANT buildable & reported", f"got {CENN_MAIN_VARIANT!r}"))

    from experiments.runner import _VARIANT_K
    from neuralforecast.cenn.model import CeNNLayer1D

    # Build and forward key Inc-3 variants via build_cenn; ASSERT the effective K and
    # the integrator actually reached every CeNNLayer1D (the whole MAC story rests on this).
    expected_int = {
        "C2-MultiScaleEnsemble": "euler", "CeNN-Full": "euler",
        "K4-Euler": "euler", "K4-Heun": "heun", "K4-ExpEuler": "exp_euler",
        "K2-Heun": "heun", "APP-RK4": "rk4",
    }
    for v in ["C2-MultiScaleEnsemble", "CeNN-Full",
              "K4-Euler", "K4-Heun", "K4-ExpEuler", "K2-Heun", "APP-RK4"]:
        try:
            m, name, scaler = build_cenn(v, h=H, n_series=V, seed=1, max_steps=3)
            assert name == f"CeNN_{v}", f"name={name}"
            assert scaler == "minmax", f"scaler={scaler}"
            exp_K = _VARIANT_K.get(v, 8)
            layers = [c for c in m.model.modules() if isinstance(c, CeNNLayer1D)]
            assert layers, "no CeNNLayer1D found"
            assert all(c.K == exp_K for c in layers), \
                f"K not wired: {[c.K for c in layers]} != {exp_K}"
            assert all(c.integrator == expected_int[v] for c in layers), \
                f"integrator not wired: {[c.integrator for c in layers]} != {expected_int[v]}"
            x = torch.randn(B, L, V)
            m.model.eval()
            with torch.no_grad():
                out = m.model(x)
            assert tuple(out.shape) == (B, H, V), f"shape={tuple(out.shape)}"
            results.append((True, f"dispatch:{v}", f"K={exp_K} integ={expected_int[v]} out={tuple(out.shape)}"))
        except Exception as e:
            results.append((False, f"dispatch:{v}", f"CRASH {type(e).__name__}: {e}"))
    return results


def main():
    print(f"torch {torch.__version__}  |  L={L} V={V} H={H} B={B}\n")
    n_fail = 0
    all_results = []

    print("--- Integrator selector ---")
    for ok, name, msg in check_integrators():
        all_results.append((ok, name, msg))

    print("--- Parallel multi-scale ensemble ---")
    for ok, name, msg in check_multiscale():
        all_results.append((ok, name, msg))

    print("--- runner.build_cenn dispatch (Inc-3 variants) ---")
    for ok, name, msg in check_dispatch():
        all_results.append((ok, name, msg))

    print("--- Inc-4: STAR cross-variable core ---")
    for ok, name, msg in check_star():
        all_results.append((ok, name, msg))

    for ok, name, msg in all_results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{status}] {name:45s} {msg}")

    print(f"\n{len(all_results) - n_fail}/{len(all_results)} checks PASS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
