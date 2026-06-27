"""Incremental experiment runner with atomic results and skip-if-exists.

Each (model, dataset, horizon, seed) produces one JSON file in experiments/results/.
If the JSON already exists, the run is skipped. This means:
  - You can stop and resume anytime
  - Adding new seeds just means running with new seed values
  - Adding new baselines just means running with new model names
  - Nothing is ever overwritten unless you delete the JSON

Usage:
    # Sanity check (1 CeNN variant, 1 dataset, 1 horizon, 1 seed)
    python experiments/runner.py --sanity

    # Run all CeNN variants on ETT datasets
    python experiments/runner.py --cenn-all --datasets ETTh1 ETTh2 ETTm1 ETTm2

    # Run specific baseline
    python experiments/runner.py --models TCN --datasets ETTh1 --horizons 96

    # Run main CeNN with 5 seeds
    python experiments/runner.py --models CeNN_C1-BoundedTau --seeds 1 42 123 7 2026

    # Dry run (show what would be run without running)
    python experiments/runner.py --models CeNN_C1-BoundedTau --datasets ETTh1 --dry-run

    # Resume (just run the same command again — completed runs are skipped)
    python experiments/runner.py --cenn-all --datasets ETTh1 ETTh2 ETTm1 ETTm2
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path so imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Determinism: CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch/cuda init so the
# 5-seed CeNN-Full std is reproducible across resumes. Respect an externally-set value if present.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.set_float32_matmul_precision("medium")
# warn_only=True: ops without a deterministic kernel warn instead of crashing the run (some
# baselines, e.g. TimesNet FFT, lack deterministic impls) — best-effort determinism, never fatal.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Whether to save baseline (non-CeNN) checkpoints. CeNN checkpoints are always saved (needed for
# the paper's qualitative figures); baselines are optional. Set from --no-baseline-checkpoints in main().
SAVE_BASELINE_CKPTS = True
# When True, bypass skip-if-exists and re-run cells even if a result JSON exists. Results are
# deterministic, so the re-run rewrites identical metrics; its purpose is to REGENERATE ARTIFACTS
# (predictions/tau/branches) for cells whose metrics were saved without --save-artifacts. Set by --force.
FORCE_RERUN = False


def _provenance():
    """Per-run provenance so a mid-run code/data change is traceable and
    skip-if-exists can't blur pre/post-fix cells. Computed once at import."""
    import platform
    import subprocess as _sp
    prov = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None), "hostname": platform.node()}
    try:
        prov["git_commit"] = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                              cwd=str(REPO_ROOT), text=True,
                                              stderr=_sp.DEVNULL).strip()
    except Exception:
        prov["git_commit"] = None
    try:
        import datasetsforecast as _df
        prov["datasetsforecast"] = getattr(_df, "__version__", None)
    except Exception:
        prov["datasetsforecast"] = None
    return prov


_PROVENANCE = _provenance()

from experiments.config import (
    INPUT_SIZE, HORIZONS, SEEDS, SEEDS_MAIN, MAX_STEPS,
    PRECISION, SCALER_TYPE,
    DATASET_INFO, SPLITS, DATASETS_ALL, DATASETS_SMALL, DATASETS_MEDIUM,
    CENN_VARIANTS, CENN_MAIN_VARIANT, CENN_DATASET_SCALER,
    BASELINES_RERUN, BASELINES_ALL,
    RESULTS_DIR, CHECKPOINTS_DIR, ARTIFACTS_DIR,
    result_path, result_exists, batch_size_for_dataset,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
_dataset_cache = {}


def load_dataset(name: str):
    """Load dataset using LongHorizon2."""
    if name not in _dataset_cache:
        from datasetsforecast.long_horizon2 import LongHorizon2
        info = DATASET_INFO[name]
        Y_df = LongHorizon2.load(directory="./data", group=info["group"])
        if "index" in Y_df.columns:
            Y_df = Y_df.drop(columns=["index"])
        _dataset_cache[name] = Y_df
    return _dataset_cache[name]


# ---------------------------------------------------------------------------
# Model builders (reuse from paper_planning but with configurable input_size)
# ---------------------------------------------------------------------------
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE, MAE
from neuralforecast.models import (
    CeNN, PatchTST, iTransformer,
    DLinear, TiDE, NHITS, TSMixer, TimeMixer,
    TimesNet, TCN, S4D,
)

try:
    from neuralforecast.models import xLSTM
    _XLSTM_AVAILABLE = True
except (ImportError, Exception):
    _XLSTM_AVAILABLE = False

# Optional Weights & Biases logging (opt-in via --wandb). Guarded so a missing/blocked wandb
# can never break a run — the per-run JSON in experiments/results/ stays the source of truth.
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WandbLogger = None
    _WANDB_AVAILABLE = False


def _common_kwargs(h, seed, max_steps, dataset_name=None):
    """Shared training kwargs for all models."""
    bs = batch_size_for_dataset(dataset_name) if dataset_name else 32
    kw_extra = {}
    # Eval-throughput knob (H200): inference batching is numerically EXACT (results invariant to
    # inference_windows_batch_size), so a bigger eval batch only changes speed. Default (unset)
    # keeps each model's own default -> the 4090 campaign is byte-unaffected.
    if os.environ.get("CENN_INFER_WBS"):
        kw_extra["inference_windows_batch_size"] = int(float(os.environ["CENN_INFER_WBS"]))
    return dict(
        **kw_extra,
        h=h,
        input_size=INPUT_SIZE,
        loss=MSE(),
        valid_loss=MAE(),
        max_steps=max_steps,
        learning_rate=1e-3,
        num_lr_decays=0,
        early_stop_patience_steps=int(os.environ.get("CENN_PATIENCE", "5")),   # CENN_PATIENCE raises it for the
        # under-training probe so a higher CENN_MAX_STEPS isn't cut short by early stopping (scratch-dir only).
        val_check_steps=100,   # match the Nixtla published protocol (was 50 -> 2x early-stop freq)
        batch_size=bs,
        scaler_type=os.environ.get("BASELINE_SCALER_OVERRIDE", SCALER_TYPE),  # A/B fairness check: run baselines under minmax to test if the CeNN's per-dataset scaler gain is CeNN-specific (use with CENN_EXP_DIR scratch so canonical identity results are never clobbered)
        random_seed=seed,
        accelerator="auto",
        precision=PRECISION,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 0.01},
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": max_steps, "eta_min": 1e-5},
    )


# --- CeNN variants (Inc-2 + Inc-3 + Inc-4) ---
#
# VARIANT_SPECS keys ARE the canonical variant names: model_name == f"CeNN_{key}"
# and they round-trip through the result JSON filename. Dispatching an unknown name
# raises a clear ValueError. Each variant is _VARIANT_BASE plus ONLY the kwargs it
# changes, so every row's delta (the single factor it ablates) is self-documenting and
# nothing silently relies on a wrapper default. _VARIANT_BASE == C1-BoundedTau (the
# reference point). All variants use scaler_type='minmax'. K-sweep variants
# override K via _VARIANT_K (annotated). feedback-conv evals/step: euler=1, exp_euler=1,
# heun=2, rk4=4 — iso-MAC pair is K2-Heun (4 evals) vs K4-Euler (4 evals).
_VARIANT_BASE = dict(
    adaptive_tau=True, dilation_schedule="none", multiscale_mode="none",
    integrator="euler", var_mix=False, cross_var="none", pointwise_mix=False,
    channel_groups=1, patch_len=None, stride=None, head_type="linear",
    alpha_min=0.5, alpha_max=0.99, spectral_cap=True, scaler_type="minmax",
)
VARIANT_SPECS = {
    # ---- MAIN ----
    "S0-StableBase":              {**_VARIANT_BASE, "adaptive_tau": False},                                          # fixed bounded α (substrate)
    "C1-BoundedTau":              {**_VARIANT_BASE},                                                                 # = base (adaptive τ, reference)
    "C2-MultiScaleEnsemble":      {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},  # S0 + multi-scale ensemble
    # PRE-DECLARED (pre-results): C1+C2 = adaptive-tau + multi-scale ensemble,
    # patch-free, linear head — the C1C2 adaptive multi-scale base. Now the -SKIP ABLATION of the
    # headline AMS-CeNN (= this + a zero-init linear skip @ K=2, key C1C2-Skip-K2 below); C1/C2 are
    # its exact one-step contribution ablations.
    "C1C2-Ensemble":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble"},
    "CeNN-Full":                  {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8},  # efficiency/patching arm
    # ---- ABLATION (leave-one-out off C1-BoundedTau unless noted) ----
    "ABL-SpectralCapOff":         {**_VARIANT_BASE, "spectral_cap": False},                                          # −spectral cap
    "ABL-Patch":                  {**_VARIANT_BASE, "patch_len": 16, "stride": 8},                                   # +patch
    # ABL-GateParam-Unbounded: gate bounds widened to [0,1]; spectral_cap KEPT ON to isolate
    # the gate-bound contribution.
    "ABL-GateParam-Unbounded":    {**_VARIANT_BASE, "alpha_min": 0.0, "alpha_max": 1.0},
    "ABL-CrossVar-Pointwise":     {**_VARIANT_BASE, "pointwise_mix": True},                                          # latent 1×1 channel mix
    "ABL-CrossVar-VarMix":        {**_VARIANT_BASE, "cross_var": "varmix"},                                          # dense O(V²) V×V mix
    "ABL-CrossVar-STAR":          {**_VARIANT_BASE, "cross_var": "star"},                                            # Inc-4: O(V) STAR core
    "ABL-ChannelGroups-G4":       {**_VARIANT_BASE, "channel_groups": 4},                                            # grouped conv over hidden
    # K × Integrator sweep (base = C1; K via _VARIANT_K). The Euler curve K={8,4,2} uses
    # C1-BoundedTau as its K=8 point (config.cenn_K=8) — do NOT add a duplicate K8-Euler; the
    # K-sweep plot must key on (cenn_K, integrator), not the variant name.
    "K4-Euler":                   {**_VARIANT_BASE},                                                                 # K=4
    "K4-Heun":                    {**_VARIANT_BASE, "integrator": "heun"},                                           # K=4
    "K4-ExpEuler":                {**_VARIANT_BASE, "integrator": "exp_euler"},                                      # K=4
    "K2-Euler":                   {**_VARIANT_BASE},                                                                 # K=2 (completes the Euler K-curve)
    "K2-Heun":                    {**_VARIANT_BASE, "integrator": "heun"},                                           # K=2 (iso-MAC vs K4-Euler)

    # Scale-count ablation: vary the number of parallel scales. Anchored on C2-MultiScaleEnsemble
    # (= n_scales 4, the existing point); num_layers == n_scales in parallel_ensemble mode (overrides _CENN_FIXED).
    "ABL-Scales-2":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 2},  # dilations [1,2]
    "ABL-Scales-3":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 3},  # [1,2,4]
    "ABL-Scales-5":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 5},  # [1,2,4,8,16]
    # ---- APPENDIX ----
    # CeNN-RawBase: −gate bounds AND −spectral cap (weakest substrate).
    "CeNN-RawBase":               {**_VARIANT_BASE, "adaptive_tau": False, "alpha_min": 0.0, "alpha_max": 1.0, "spectral_cap": False},
    "APP-C2Form-DilatedTemplate": {**_VARIANT_BASE, "dilation_schedule": "exponential"},                            # dilated TEMPLATES (vs ensemble)
    "APP-Head-MLP":               {**_VARIANT_BASE, "patch_len": 16, "stride": 8, "head_type": "mlp"},               # MLP head (paired w/ patch)
    "APP-RK4":                    {**_VARIANT_BASE, "integrator": "rk4"},                                            # RK4 accuracy ceiling
    # ADDITIVITY (does the multi-scale ensemble add accuracy ON TOP of an MLP head, or are the two
    # redundant? Tested on the long-horizon setting where the dilated branches have room to operate.)
    "APP-MultiScale-Patch-MLP":   {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8, "head_type": "mlp"},  # CeNN-Full + MLP head
    "APP-MultiScale-MLP":         {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "head_type": "mlp"},     # multi-scale + MLP head, no patch

    # PRE-DECLARED CONFIRMATORY (pre-results): the HEADLINE config
    # (C2-MultiScaleEnsemble) at K=2. The flat K-surface was measured on the C1 chassis only;
    # this tests whether it transfers to the multi-scale headline (-> 4x cheaper inference at
    # equal accuracy if it holds; an honest K-requirement note if it breaks).
    "K2-C2Ensemble":              {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},

    # PRE-DECLARED CAPACITY-SCALING ANALYSIS (pre-results): the title-model architecture (C1C2)
    # at hidden_dim 128/256, intended for ECL/Traffic ONLY (the V->C bottleneck question: is the
    # big-V gap a capacity knob or a structural limit?). NOT in any CENN_VARIANTS list -- never
    # auto-runs; launched explicitly via --models after the in-protocol baseline gaps are known.
    # The main table stays uniform-capacity; these are analysis rows.
    "CAP-Ensemble-H128":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 128},
    "CAP-Ensemble-H256":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 256},
    # H512 >= V_ECL(321): the NO-COMPRESSION anchor of the capacity curve.
    # Traffic (V=862) stays 1.7:1 compressed at 512 -- state in the analysis. windows_batch_size
    # halved for VRAM headroom at 8x activations (training-sampling knob, disclosed; analysis row).
    "CAP-Ensemble-H512":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 512, "windows_batch_size": 128},
    # CAPACITY x SKIP (appendix): the HEADLINE arch (C1C2-Skip, K=2) at higher hidden_dim on the
    # high-V datasets (ECL V=321, Traffic V=862) to test whether AMS-CeNN's high-V rank gap is a
    # V->hidden CAPACITY bottleneck (uniform hidden=64 << V) or a STRUCTURAL limit. The CAP-Ensemble
    # rows above are the -skip ensemble; these add the linear skip so the curve is on the shipped arch.
    # Analysis rows ONLY (main table stays uniform hidden=64); NOT in any CENN_VARIANTS list. K=2 via
    # _VARIANT_K; wbs lowered at high hidden for VRAM (training-sampling knob, disclosed, eval unchanged).
    "CAP-Skip-H128":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 128},
    "CAP-Skip-H256":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 256, "windows_batch_size": 128},
    "CAP-Skip-H512":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 512, "windows_batch_size": 64},

    # ROOT-CAUSE GATE (pre-results): the CeNN stack smooths the signal before a
    # DLinear-style linear temporal head -> amplitude damping / short-horizon loss to DLinear.
    # Two candidate fixes, tested cheaply on ETTh2/ETTm2 H336/720 before any full re-run:
    #   linear_skip -> add a raw-input linear path (cause-level; never worse than the linear baseline)
    #   readout_act -> GELU/SiLU read-out (symptom-level; less saturation, stability untouched)
    # NOT in any CENN_VARIANTS list -- launched explicitly via --models for the gate only.
    "C1C2-Skip":                  {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
    "C1C2-GELUout":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "readout_act": "gelu"},
    "C1C2-SiLUout":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "readout_act": "silu"},
    "C1C2-Skip-GELUout":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "readout_act": "gelu"},
    # THE HEADLINE = AMS-CeNN: C1C2 ensemble + zero-init linear skip at K=2. Skip
    # rescues high-V and beats DLinear on complex/long-horizon; K=2 (not K=8) because the K-surface
    # is flat on THIS arch (C1C2-Skip K8 vs K2: dMSE 0.0000 over 42 cells) and 4x cheaper -- chosen
    # for a leaner backbone, NOT an efficiency claim (AMS is ~6x DLinear's MACs; efficiency is not a
    # headline pillar). K set via _VARIANT_K below; separate name from the K=8 gate cells of
    # C1C2-Skip (which serve as the K8 reference for the K-neutrality check).
    "C1C2-Skip-K2":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
    # Cross-channel interaction variants on the headline architecture (skip + multiscale, K=2): test
    # whether explicit cross-series coupling helps the high-cardinality regime (Electricity, Traffic).
    "C1C2-Skip-K2-STAR":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "star"},      # O(V) STAR aggregate-redistribute
    "C1C2-Skip-K2-Pointwise":     {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "pointwise_mix": True},    # latent 1x1 channel mix
    "C1C2-Skip-K2-VarMix":        {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "varmix"},    # dense O(V^2) V x V mix
    "C1C2-Skip-K2-G4":            {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "channel_groups": 4},      # grouped conv over hidden
    # CONTROL for "isn't this just a linear skip + ANY nonlinear block?": swap the CeNN dynamics for
    # a generic MLP-Mixer trunk, holding input_proj + head + zero-init skip constant. K/integrator/
    # multiscale are inherited but IGNORED by the MLP trunk (logged cenn_K is meaningless here). A tie
    # with AMS-CeNN shows the CeNN's inductive bias is not an ACCURACY win over a generic trunk.
    "MLP-Skip":                   {**_VARIANT_BASE, "trunk_type": "mlp", "linear_skip": True},
    # Closes the integrator gap on the EXACT headline arch (the K-sweep was on the C1 chassis): the
    # headline with exp_euler instead of euler at K=2. Expected ~= euler (K4 evidence: exp_euler within
    # 0.0004 of euler) -> confirms Euler-K2 leaves no accuracy on the table for the FINAL model.
    "C1C2-Skip-ExpEuler-K2":      {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "integrator": "exp_euler"},
    # K=4 MIDPOINT on the EXACT headline arch: completes a fully headline-native K-curve
    # {K8 (C1C2-Skip), K4 (this), K2 (C1C2-Skip-K2)} so the K-step figure is shown on the model
    # we ship, not only the C1 chassis. Same spec as C1C2-Skip/-K2; K=4 via _VARIANT_K. Run on
    # the ETT grid (4 datasets x 4 horizons x 3 seeds) where K8 & K2 both cover.
    "C1C2-Skip-K4":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
}

# Fixed CeNN architecture/training settings shared by every variant.
_CENN_FIXED = dict(
    hidden_dim=64, N=1, num_layers=4, dropout=0.25, neighborhood=3,
    alpha_init=0.9, enforce_bistability=False, cross_channel=False,
    spectral_rho=0.9, gradient_clip_val=1.0,
    # windows_batch_size=256 (NOT the NeuralForecast default 1024): the recurrent K-loop is
    # BPTT-unrolled (K x num_layers x [evals/step]), so memory scales with wbs x V x integrator
    # evals. At 1024, RK4 OOMs on 24GB and C2-ensemble on Traffic (V=862, 4 branches) would OOM
    # even on 80GB. 256 keeps the worst cells (RK4/ETT ~16GB, C2/Traffic ~6GB extrapolated) within
    # even 24GB -> trivially safe on the 80GB H100, while using more of the H100's memory/compute
    # (more data per step, better GPU utilization than 128). NOT a fairness axis (eval unchanged);
    # MAX_STEPS (not wbs) is the lever for training length. Kept identical dev (4090) <-> prod (H100)
    # for reproducibility. Re-verified fail-free at 256 by preflight before the H100 run.
    windows_batch_size=256,
    # multiscale_mode and integrator are NOT here: every VARIANT_SPECS entry sets them
    # explicitly (invariant verified by smoke), so they arrive via **spec. Putting them
    # here too would cause a 'multiple values for keyword argument' TypeError.
)


# K values for the K-sweep variants (overrides the default K=8).
_VARIANT_K = {
    "K4-Euler": 4, "K4-Heun": 4, "K4-ExpEuler": 4,
    "K2-Euler": 2, "K2-Heun": 2,
    "K2-C2Ensemble": 2,   # confirmatory: headline architecture at K=2
    "C1C2-Skip-K2": 2,    # NEW HEADLINE: linear-skip at K=2 (4x cheaper, accuracy-neutral)
    "C1C2-Skip-K2-STAR": 2, "C1C2-Skip-K2-Pointwise": 2,  # cross-channel variants on headline (K=2)
    "C1C2-Skip-K2-VarMix": 2, "C1C2-Skip-K2-G4": 2,
    "C1C2-Skip-ExpEuler-K2": 2,   # integrator-gap closer: headline arch, exp_euler @ K=2
    "C1C2-Skip-K4": 4,    # K=4 midpoint for the headline-native K-curve {K8,K4,K2}
    "CAP-Skip-H128": 2, "CAP-Skip-H256": 2, "CAP-Skip-H512": 2,   # headline arch (K=2) at higher capacity
}


def build_cenn(variant, h, n_series, seed, max_steps, K=8, dataset_name=None):
    """Build a CeNN model for a named variant from VARIANT_SPECS.

    `variant` is the canonical spec key (e.g. 'CeNN-Full') == the string after
    'CeNN_' in model_name and in the result JSON filename. Returns
    (model, name, scaler_type) so run_single can log the PER-VARIANT scaler
    (CeNN=minmax) instead of the global SCALER_TYPE (baselines=identity).
    K-sweep variants override K via _VARIANT_K (ignore the caller-supplied K).
    """
    if variant not in VARIANT_SPECS:
        raise ValueError(
            f"Unknown CeNN variant: {variant!r}. "
            f"Known: {sorted(VARIANT_SPECS)}."
        )
    spec = dict(VARIANT_SPECS[variant])
    scaler_type = spec.pop("scaler_type")
    # Per-dataset scaler (validation-selected): CeNN's minmax default underperforms on Weather;
    # identity recovers ~13% there. Applied to ALL CeNN variants on the dataset for a controlled
    # within-dataset comparison. Datasets not listed keep the per-variant default. See config.
    if dataset_name in CENN_DATASET_SCALER:
        scaler_type = CENN_DATASET_SCALER[dataset_name]
    # Weather-anomaly diagnosis: CENN_SCALER_OVERRIDE swaps the scaler for an A/B. Use with
    # CENN_EXP_DIR=_wdiag/<scaler> isolation so diagnostic cells never touch main results. The env
    # override WINS over the per-dataset default -- unset -> the per-dataset/per-variant scaler.
    scaler_type = os.environ.get("CENN_SCALER_OVERRIDE", scaler_type)
    name = f"CeNN_{variant}"

    kwargs = _common_kwargs(h, seed, max_steps, dataset_name=dataset_name)
    kwargs["scaler_type"] = scaler_type                  # per-variant override (CeNN=minmax)
    # weight_decay=0.01 inherited from _common_kwargs (symmetric across all re-run models)

    effective_K = _VARIANT_K.get(variant, K)  # K-sweep variants override the default K

    # Spec keys override _CENN_FIXED keys (e.g. ABL-Scales-* override num_layers, which is
    # n_scales in the parallel ensemble). Local copy — never mutate _CENN_FIXED itself
    # (profile_efficiency reads it). For all pre-existing variants no key collides, so this
    # is behavior-identical for them.
    fixed = {k: v for k, v in _CENN_FIXED.items() if k not in spec}

    model = CeNN(
        **kwargs,
        n_series=n_series,
        K=effective_K,
        **fixed,
        **spec,           # wires all params (patch/head/var_mix/groups/gate/cap/multiscale/integrator)
    )
    model.alias = name
    return model, name, scaler_type


# --- Baseline builders ---

BASELINE_BUILDERS = {}


def _register(name):
    def decorator(fn):
        BASELINE_BUILDERS[name] = fn
        return fn
    return decorator


@_register("PatchTST")
def build_patchtst(h, n_series, seed, max_steps, dataset_name=None):
    m = PatchTST(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        encoder_layers=3, n_heads=4, hidden_size=64,
        patch_len=16, stride=8, dropout=0.2,
    )
    m.alias = "PatchTST"
    return m, "PatchTST"


@_register("iTransformer")
def build_itransformer(h, n_series, seed, max_steps, dataset_name=None):
    m = iTransformer(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        n_series=n_series, hidden_size=128, n_heads=4,
        e_layers=2, d_ff=128, dropout=0.1,
    )
    m.alias = "iTransformer"
    return m, "iTransformer"


@_register("DLinear")
def build_dlinear(h, n_series, seed, max_steps, dataset_name=None):
    m = DLinear(**_common_kwargs(h, seed, max_steps, dataset_name=dataset_name))
    m.alias = "DLinear"
    return m, "DLinear"


@_register("S4D")
def build_s4d(h, n_series, seed, max_steps, dataset_name=None):
    # Official-S4D diagonal SSM baseline (this work's family coverage; full SSM study = future work).
    kw = _common_kwargs(h, seed, max_steps, dataset_name=dataset_name)
    kw["precision"] = "32-true"  # S4D's complex SSM kernel (cfloat FFT/einsum) -> fp32 like the authors
    # S4D is NORMALIZATION-SENSITIVE: under the global-identity protocol it diverges on the spiky
    # ETT*2 series (S4D-identity ETTh2 H96 = 1.83 vs ~0.25 for everything else). Use per-window
    # min-max -- the SAME validation-selected per-dataset policy as the CeNN (identity only on
    # Weather, where minmax compresses near-constant channels). The scale-robust linear/mixer
    # baselines keep global normalization (scaler A/B confirms minmax doesn't help them); only the
    # normalization-sensitive dynamical models (CeNN, S4D) get per-window minmax. Documented in the
    # paper. A BASELINE_SCALER_OVERRIDE (the A/B harness) still wins if explicitly set.
    if not os.environ.get("BASELINE_SCALER_OVERRIDE"):
        kw["scaler_type"] = CENN_DATASET_SCALER.get(dataset_name, "minmax")
    m = S4D(**kw, n_series=n_series, d_model=128, d_state=64, n_layers=2, dropout=0.1)
    m.alias = "S4D"
    return m, "S4D"


@_register("TiDE")
def build_tide(h, n_series, seed, max_steps, dataset_name=None):
    m = TiDE(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        num_encoder_layers=2, num_decoder_layers=2, hidden_size=256,
        decoder_output_dim=32, dropout=0.3,
    )
    m.alias = "TiDE"
    return m, "TiDE"


@_register("NHITS")
def build_nhits(h, n_series, seed, max_steps, dataset_name=None):
    m = NHITS(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1],
        stack_types=3 * ["identity"], mlp_units=3 * [[512, 512]],
        dropout_prob_theta=0.1,
    )
    m.alias = "NHITS"
    return m, "NHITS"


@_register("TSMixer")
def build_tsmixer(h, n_series, seed, max_steps, dataset_name=None):
    m = TSMixer(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        n_series=n_series, n_block=2, ff_dim=64, dropout=0.1,
    )
    m.alias = "TSMixer"
    return m, "TSMixer"


@_register("TimeMixer")
def build_timemixer(h, n_series, seed, max_steps, dataset_name=None):
    m = TimeMixer(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        n_series=n_series, e_layers=2, d_model=32, d_ff=32,
        dropout=0.1, down_sampling_layers=3, down_sampling_window=2,
    )
    m.alias = "TimeMixer"
    return m, "TimeMixer"


@_register("TimesNet")
def build_timesnet(h, n_series, seed, max_steps, dataset_name=None):
    kw = _common_kwargs(h, seed, max_steps, dataset_name=dataset_name)
    kw["precision"] = "32-true"  # TimesNet FFT ops don't support bf16 ("Unsupported dtype BFloat16")
    m = TimesNet(
        **kw,
        hidden_size=32, conv_hidden_size=32, encoder_layers=2,
        top_k=5, num_kernels=6, dropout=0.1,
    )
    m.alias = "TimesNet"
    return m, "TimesNet"


@_register("TCN")
def build_tcn(h, n_series, seed, max_steps, dataset_name=None):
    m = TCN(
        **_common_kwargs(h, seed, max_steps, dataset_name=dataset_name),
        kernel_size=3, dilations=[1, 2, 4, 8],
        encoder_hidden_size=64, encoder_activation="ReLU",
        context_size=10, decoder_hidden_size=64, decoder_layers=2,
    )
    m.alias = "TCN"
    return m, "TCN"


@_register("xLSTM")
def build_xlstm(h, n_series, seed, max_steps, dataset_name=None):
    if not _XLSTM_AVAILABLE:
        raise ImportError("xLSTM not available")
    m = xLSTM(**_common_kwargs(h, seed, max_steps, dataset_name=dataset_name))
    m.alias = "xLSTM"
    return m, "xLSTM"


# ---------------------------------------------------------------------------
# Core: run a single experiment and save atomic JSON
# ---------------------------------------------------------------------------

def run_single(model_name: str, dataset_name: str, horizon: int, seed: int,
               max_steps: int = MAX_STEPS, cenn_K: int = 8,
               save_artifacts: bool = False, wandb_cfg: dict | None = None) -> dict | None:
    """Run one (model, dataset, horizon, seed) and save result as JSON.

    Returns the result dict, or None if skipped (already exists).
    """
    # Skip if result already exists (unless --force, used to backfill artifacts for done cells)
    if not FORCE_RERUN and result_exists(model_name, dataset_name, horizon, seed):
        return None

    import pandas as pd

    Y_df = load_dataset(dataset_name)
    n_series = DATASET_INFO[dataset_name]["n_series"]
    freq = DATASET_INFO[dataset_name]["freq"]

    # Ablation F (variable count): CENN_VAR_SUBSET=K keeps K series via a FIXED nested random
    # permutation (V8 subset ⊂ V16 ⊂ ... so the ladder is monotone in information; same subset
    # across models+seeds). Outputs nest under V{K}/ (config) so subset cells never collide with
    # the full-dataset results. The orthogonal axis to CAP-* (which varies hidden at fixed V):
    # together they map the V/C compression plane. Unset -> full dataset, behavior unchanged.
    _subset = os.environ.get("CENN_VAR_SUBSET")
    if _subset:
        import numpy as np
        K = int(_subset)
        uids = sorted(Y_df["unique_id"].unique())
        perm = np.random.default_rng(20260613).permutation(len(uids))
        keep = {uids[i] for i in perm[:K]}
        Y_df = Y_df[Y_df["unique_id"].isin(keep)].copy()
        n_series = K
        print(f"[VAR_SUBSET={K}/{len(uids)}] ", end="")

    print(f"  Running {model_name} | {dataset_name} H={horizon} seed={seed}...",
          end=" ", flush=True)

    wb_logger = None
    wb_run = None
    try:
        # Build model
        if model_name.startswith("CeNN_"):
            variant = model_name.removeprefix("CeNN_")
            model, alias, run_scaler = build_cenn(variant, horizon, n_series, seed,
                                                   max_steps, K=cenn_K,
                                                   dataset_name=dataset_name)
            _trunk = VARIANT_SPECS[variant].get("trunk_type", "cenn")
            # For the generic-trunk control (trunk_type='mlp') the integration K / integrator /
            # multiscale are INHERITED from _VARIANT_BASE but mean nothing (no CeNN cell) -> null
            # them so the K-sweep / integrator figures never pick MLP-Skip up as a spurious data point.
            cenn_meta = {
                "trunk_type": _trunk,
                "cenn_K": (_VARIANT_K.get(variant, cenn_K) if _trunk == "cenn" else None),
                "integrator": (VARIANT_SPECS[variant]["integrator"] if _trunk == "cenn" else None),
                "multiscale_mode": VARIANT_SPECS[variant]["multiscale_mode"],
                "cross_var": VARIANT_SPECS[variant].get("cross_var", "none"),
            }
        else:
            builder = BASELINE_BUILDERS[model_name]
            model, alias = builder(horizon, n_series, seed, max_steps,
                                    dataset_name=dataset_name)
            # log the model's ACTUAL scaler, not the global default -- baselines may override it
            # (e.g. S4D uses per-window minmax; BASELINE_SCALER_OVERRIDE for the A/B). Provenance.
            run_scaler = model.hparams.get("scaler_type", SCALER_TYPE)
            cenn_meta = {}

        # Optional per-run W&B logger attached to the PL trainer. Strictly non-fatal: catch
        # BaseException so even a wandb sys.exit() (e.g. stale auth) can't kill the campaign,
        # and capture the Run object so a partial init can't leak an unfinished run. (No
        # reinit kwarg — each run is finish()ed in `finally`, so the next init starts clean.)
        if wandb_cfg is not None and _WANDB_AVAILABLE:
            try:
                wb_logger = WandbLogger(
                    project=wandb_cfg["project"], entity=wandb_cfg.get("entity"),
                    name=f"{model_name}__{dataset_name}__H{horizon}__seed{seed}",
                    group=dataset_name,
                )
                wb_run = wb_logger.experiment  # triggers wandb.init(); the Run we own + finish
                wb_run.config.update(
                    {"model": model_name, "dataset": dataset_name, "horizon": horizon,
                     "seed": seed, "input_size": INPUT_SIZE, "max_steps": max_steps,
                     "scaler_type": run_scaler, **cenn_meta},
                    allow_val_change=True,
                )
                model.trainer_kwargs["logger"] = wb_logger
            except BaseException as e:
                print(f"[wandb init failed: {type(e).__name__}: {e}; continuing]", end=" ")
                if wb_run is not None:
                    try:
                        wb_run.finish()
                    except BaseException:
                        pass
                wb_logger = wb_run = None

        nf = NeuralForecast(models=[model], freq=str(freq))

        # Train + evaluate on the standard LTSF fixed split (stride-1 sliding window
        # over the full test set). val_size/test_size from config.SPLITS; step_size
        # defaults to 1; refit=False trains once (matches Nixtla long-horizon experiments).
        # LongHorizon2 loads data already z-score normalized on train, so MSE/MAE are
        # reported on the normalized scale — comparable to published numbers.
        val_size, test_size = SPLITS[dataset_name]
        t0 = time.time()
        # Huge eval matrices (ECL/Traffic long horizons) go through the memory-bounded chunked
        # path: identical windows + metrics (forced-chunking A/B verified bit-identical), peak
        # RAM ~5-8 GB per chunk instead of 120+ GB for the single-shot frame+join.
        rows_est = (test_size - horizon + 1) * horizon * n_series
        chunked = None
        if rows_est > CV_CHUNK_ROWS:
            want_pred = save_artifacts and model_name == f"CeNN_{CENN_MAIN_VARIANT}"
            chunked = _chunked_cross_validation(
                nf, Y_df, val_size, test_size, horizon, alias,
                collect_windows=500 if want_pred else 0,
            )
        if chunked is not None:
            mse, mae, cv_df, cv_chunks = chunked
            if cv_df is None:
                cv_df = pd.DataFrame()
        else:
            cv_chunks = None
            cv_df = nf.cross_validation(
                df=Y_df,
                val_size=val_size,
                test_size=test_size,
                n_windows=None,
                refit=False,
            )
            if isinstance(cv_df.index, pd.MultiIndex):
                cv_df = cv_df.reset_index()
            mse = float(((cv_df[alias] - cv_df["y"]) ** 2).mean())
            mae = float((cv_df[alias] - cv_df["y"]).abs().mean())
        train_time = time.time() - t0

        if wb_run is not None:
            try:
                wb_run.summary["test_mse"] = mse
                wb_run.summary["test_mae"] = mae
                wb_run.summary["params"] = sum(p.numel() for p in model.parameters())
                wb_run.summary["train_time_s"] = round(train_time, 1)
            except Exception:
                pass

        # Save artifacts (predictions, tau values) if requested
        if save_artifacts and model_name.startswith("CeNN_"):
            _save_artifacts(nf, model_name, dataset_name, horizon, seed, cv_df, alias)
            _save_tau(nf, model_name, dataset_name, horizon, seed)
            _save_aeff(nf, model_name, dataset_name, horizon, seed)
            # Per-scale branch forecasts (scale-disagreement + spread-vs-error). The extra use_fitted
            # CV passes (one per dilation scale) are bounded to the headline multi-scale variant AND
            # to its first seed only — the diagnostics are single-model (not seed-ensemble; that is
            # future work), so one seed per (dataset,horizon) is sufficient and saves ~5x H100 eval time.
            # _save_branches also no-ops for non-ensemble variants. SKIPPED on chunked-CV cells:
            # each branch pass would be a single-shot huge CV (the exact OOM this path avoids);
            # the diagnostics are already covered by the small+Weather cells.
            if (model_name == f"CeNN_{CENN_MAIN_VARIANT}" and seed == SEEDS_MAIN[0]
                    and chunked is None):
                _save_branches(nf, model_name, dataset_name, horizon, seed, cv_df, alias,
                               Y_df, val_size, test_size)
            elif chunked is not None and model_name == f"CeNN_{CENN_MAIN_VARIANT}":
                print("[branches skipped: chunked-CV cell] ", end="")

        del cv_df

        # Save checkpoint. CeNN always (needed for forecast/tau figures + appendix re-eval);
        # baselines only unless --no-baseline-checkpoints (halves ckpt storage).
        # Crash-safe: write to a temp dir then swap, so a SIGTERM mid-write can
        # never destroy a previously-good checkpoint (nf.save(overwrite=True) does rm-then-write).
        if SAVE_BASELINE_CKPTS or model_name.startswith("CeNN_"):
            ckpt_dir = CHECKPOINTS_DIR / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}"
            tmp_dir = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                nf.save(path=str(tmp_dir), save_dataset=False, overwrite=True)
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                os.replace(tmp_dir, ckpt_dir)  # fast dir rename — tiny crash window vs the write
            except Exception as e:
                print(f"[checkpoint save failed: {e}]", end=" ")

        # Build result
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "horizon": horizon,
            "seed": seed,
            "input_size": INPUT_SIZE,
            "mse": mse,
            "mae": mae,
            "train_time_s": round(train_time, 1),
            "train_time_s_note": "wall time incl. train + stride-1 test eval (NOT training-only); "
                                 "use efficiency compute_ms for inference cost",
            "params": sum(p.numel() for p in model.parameters()),
            "source": "experiment",
            "provenance": _PROVENANCE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_steps": max_steps,
                # actual precision, not the global default (S4D + TimesNet override to 32-true).
                "precision": model.hparams.get("precision", PRECISION),
                "scaler_type": run_scaler,
                "weight_decay": model.optimizer_kwargs.get("weight_decay", 0.01),
                "val_size": SPLITS[dataset_name][0],
                "test_size": SPLITS[dataset_name][1],
                "eval": "standard_fixed_split_stride1",
                # provenance: evaluation chunk count (null = single-shot). Chunked cells use the
                # identical window set (hard-asserted); metrics agree to fp-association (~1e-16).
                "cv_chunks": cv_chunks,
                "batch_size": batch_size_for_dataset(dataset_name),
                **cenn_meta,  # cenn_K / integrator / multiscale_mode (CeNN runs only)
            },
        }

        # Save atomic JSON
        out_path = result_path(model_name, dataset_name, horizon, seed)
        tmp_path = out_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, out_path)  # atomic + cross-platform: os.replace OVERWRITES an existing
        # dest (needed for --force); Path.rename() raises FileExistsError on Windows if dest exists.

        print(f"MSE={mse:.6f}  MAE={mae:.6f}  Time={train_time:.1f}s")
        return result

    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

        # Save error result so we know it was attempted
        error_result = {
            "model": model_name,
            "dataset": dataset_name,
            "horizon": horizon,
            "seed": seed,
            "input_size": INPUT_SIZE,
            "mse": None,
            "mae": None,
            "error": str(e),
            "source": "experiment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Don't save error JSONs — let user retry by re-running
        return error_result

    finally:
        # Finish the W&B run (idempotent) so each cell is its own run and a crash/early-exit
        # can't leave it dangling. Use the captured Run, not the lazy .experiment property
        # (which would re-init a fresh empty run if PL already finalized the logger).
        if wb_run is not None:
            try:
                wb_run.finish()
            except BaseException:
                pass
        # Free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Chunked-CV engagement threshold, in evaluation rows = (test_size-h+1) * h * n_series.
# Above it, the single-shot cv_df (frame + ground-truth join) transiently needs ~120+ GB at
# Traffic x H720 (measured on the 128GB H200: trained fine at 92 GB, OOM-killed in assembly at
# 123.7 GB even with the categorical-id core patch). 2e8 keeps every small+Weather cell on the
# unchunked path (Weather H720 = 1.5e8) so the running 4090 campaign's behavior is unchanged.
CV_CHUNK_ROWS = int(float(os.environ.get("CENN_CV_CHUNK_ROWS", str(200_000_000))))  # float(): tolerate "2e8"
# Rows per chunk. Default sized for safety; on the H200 (125GB, measured 18.3GB peak at 1.2e8)
# CENN_CV_CHUNK_TARGET=3e8 gives fewer chunks -> less repeated per-chunk dataset prep. Affects
# only fp summation order (~1e-16), like chunking itself.
_CV_CHUNK_TARGET = int(float(os.environ.get("CENN_CV_CHUNK_TARGET", str(120_000_000))))


def _stride_windows(df_part, max_windows):
    """Deterministic whole-window ((unique_id,cutoff) group) striding, as in _save_artifacts."""
    wid = df_part.groupby(["unique_id", "cutoff"], sort=False, observed=True).ngroup()
    n_windows = int(wid.max()) + 1
    if n_windows <= max_windows:
        return df_part
    stride = (n_windows + max_windows - 1) // max_windows
    return df_part[(wid % stride) == 0]


def _chunked_cross_validation(nf, Y_df, val_size, test_size, h, alias,
                              collect_windows=0):
    """Protocol-identical, memory-bounded CV for huge eval matrices (H200 OOM workaround).

    Partitions the stride-1 test cutoffs into consecutive ranges. Chunk 0 trains ONCE with the
    exact protocol split: truncating every series to first E0 = (last cutoff of chunk 0)+h steps
    and passing test_size = E0 - first_cutoff makes the train extent E0 - val - test_size ==
    L - SPLITS_test - val — the full-split boundary, independent of chunking. Chunks 1..K-1 are
    use_fitted=True re-evaluations (eval-only; fitted state restored after each call — the same
    seam _save_branches uses). MSE/MAE stream as sums; each chunk's frame is freed before the
    next. Window set == the single-shot call's exactly (chunks partition the same cutoffs), so
    metrics are bit-identical (verified by forced-chunking A/B).

    Returns (mse, mae, artifact_df_or_None) or None when the layout doesn't qualify
    (unequal series lengths) -> caller falls back to single-shot.
    """
    import numpy as np
    import pandas as pd
    sizes = Y_df.groupby("unique_id", observed=True).size()
    L = int(sizes.iloc[0])
    if not (sizes == L).all():
        return None
    n_series = len(sizes)
    n_cutoffs = test_size - h + 1
    rows_total = n_cutoffs * h * n_series
    n_chunks = min(max(2, -(-rows_total // _CV_CHUNK_TARGET)), n_cutoffs)
    # Cutoff positions = 0-based index of each window's LAST INPUT step (NF convention:
    # the full call's cutoffs are {L-test-1, ..., L-h-1}). edges[j] (incl) .. edges[j+1]
    # (excl) partition that exact set across chunks.
    first_c = L - test_size - 1
    edges = np.linspace(first_c, L - h, n_chunks + 1).round().astype(int)
    Y = Y_df.sort_values(["unique_id", "ds"], kind="stable", ignore_index=True)
    pos = Y.groupby("unique_id", observed=True).cumcount().to_numpy()

    sse = sae = 0.0
    n_rows = 0
    keep_parts = []
    per_chunk_keep = max(1, collect_windows // n_chunks) if collect_windows else 0
    for j in range(n_chunks):
        a, b_excl = int(edges[j]), int(edges[j + 1])
        if b_excl <= a:
            continue
        b = b_excl - 1                    # last cutoff (last-input index) of this chunk
        E = b + h + 1                     # rows per series needed: inputs..targets of cutoff b
        t = E - a - 1                     # call's cutoffs = {E-t-1 .. E-h-1} = {a .. b} exactly
        # (degenerate single-chunk check: a=L-test-1, b=L-h-1 -> E=L, t=test_size == full call)
        df_j = Y[pos < E]
        cv = nf.cross_validation(
            df=df_j, val_size=(val_size if j == 0 else 0), test_size=t,
            n_windows=None, refit=False, use_fitted=(j > 0),
        )
        if isinstance(cv.index, pd.MultiIndex):
            cv = cv.reset_index()
        if cv["y"].isna().any():
            raise RuntimeError(f"chunked CV: NaN ground truth in chunk {j} (join misalignment)")
        err = cv[alias].to_numpy() - cv["y"].to_numpy()
        if np.isnan(err).any():
            raise RuntimeError(f"chunked CV: NaN forecasts in chunk {j} (model output blew up)")
        sse += float(np.square(err).sum())
        sae += float(np.abs(err).sum())
        n_rows += len(cv)
        if per_chunk_keep:
            keep_parts.append(_stride_windows(cv, per_chunk_keep).copy())
        del cv
        gc.collect()

    if n_rows != rows_total:
        raise RuntimeError(
            f"chunked CV row mismatch: evaluated {n_rows}, protocol expects {rows_total}")
    artifact_df = pd.concat(keep_parts, ignore_index=True) if keep_parts else None
    print(f"[chunked CV: {n_chunks} chunks x ~{rows_total // n_chunks:,} rows] ", end="")
    return sse / n_rows, sae / n_rows, artifact_df, n_chunks


def _save_artifacts(nf, model_name, dataset_name, horizon, seed, cv_df, alias):
    """Save per-window test predictions (+ index) for the forecast-grid figure / seed-ensemble band.

    SCOPE + SIZE GUARDS (the unguarded version wrote the FULL stride-1 prediction table for EVERY
    CeNN cell: ~730 MB/cell average, 1.06 TB across the 4090 campaign, and a single ECL/Traffic
    H720 cell is a ~1e9-row table => would fill the H200 VM disk mid-weekend):
      - predictions are saved ONLY for the headline variant (the only consumer is the
        forecast-grid figure + its 5-seed band, both headline-only);
      - windows are deterministically STRIDED to <= PRED_MAX_WINDOWS whole windows (same scheme
        as _save_branches; the grid picks one most-dynamic window, the band joins on
        (unique_id, ds), and the deterministic window-id stride keeps seeds aligned).
    tau/aeff/branches artifacts are unaffected (tiny / already gated)."""
    import numpy as np

    if model_name != f"CeNN_{CENN_MAIN_VARIANT}":
        return  # predictions feed headline-only figures; other variants' tables are dead weight
    pred_dir = ARTIFACTS_DIR / "predictions"
    # Self-describing .npz keeps unique_id/ds/cutoff so cross-seed alignment is a join on
    # (unique_id, ds), not a positional ordering assumption.
    pred_path = pred_dir / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}.npz"
    PRED_MAX_WINDOWS = 500
    try:
        keep = cv_df
        if {"unique_id", "cutoff"}.issubset(cv_df.columns):
            wid = cv_df.groupby(["unique_id", "cutoff"], sort=False).ngroup()
            n_windows = int(wid.max()) + 1
            if n_windows > PRED_MAX_WINDOWS:
                stride = (n_windows + PRED_MAX_WINDOWS - 1) // PRED_MAX_WINDOWS
                keep = cv_df[(wid % stride) == 0]
        # np.asarray: with the categorical unique_id (low-mem core path), .values would hand
        # np.savez a pd.Categorical; asarray coerces to a plain object array of strings.
        idx = {c: np.asarray(keep[c]) for c in ("unique_id", "ds", "cutoff") if c in keep.columns}
        np.savez(pred_path, pred=keep[alias].values, y=keep["y"].values, **idx)
    except Exception as e:
        print(f"[artifacts save failed: {e}]", end=" ")


def _save_tau(nf, model_name, dataset_name, horizon, seed):
    """Capture the adaptive-tau gate values (C1) for the tau heatmap figure.

    Each CeNNCell1D with adaptive_tau stores last_tau=[B,C,L] from its most recent forward (set
    during predict). We traverse the trained module, mean over the batch -> [C,L] per cell, and save
    one .npz to ARTIFACTS_DIR/tau/. Non-adaptive variants (no tau gate) simply produce nothing.
    Strictly non-fatal."""
    import numpy as np
    try:
        from neuralforecast.cenn.model import CeNNCell1D
        taus = {}
        for i, mod in enumerate(m for m in nf.models[0].modules()
                                if isinstance(m, CeNNCell1D) and getattr(m, "adaptive_tau", False)):
            lt = getattr(mod, "last_tau", None)
            if lt is not None:
                taus[f"cell{i}"] = lt.detach().float().mean(dim=0).cpu().numpy()  # [C, L]
        if not taus:
            return
        tau_dir = ARTIFACTS_DIR / "tau"
        tau_dir.mkdir(parents=True, exist_ok=True)
        np.savez(tau_dir / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}.npz", **taus)
    except Exception as e:
        print(f"[tau save failed: {e}]", end=" ")


def _save_aeff(nf, model_name, dataset_name, horizon, seed):
    """Capture the feedback operator norm ||A_eff|| per output channel (C1 stability / spectral cap)
    for the contraction figure. For each CeNNCell1D we recompute the masked feedback kernel exactly
    as precompute() does, take its per-output-channel L1 (the conv operator-norm upper bound the cap
    enforces), and record BOTH the learned (uncapped) norm and the effective (capped=min(raw,rho))
    norm. Pure weight inspection — NO forward pass, fully post-hoc. Strictly non-fatal."""
    import numpy as np
    try:
        from neuralforecast.cenn.model import CeNNCell1D
        raws, caps, rho, alpha_max = [], [], None, None
        for m in nf.models[0].modules():
            if not isinstance(m, CeNNCell1D):
                continue
            if alpha_max is None:
                alpha_max = float(getattr(m, "alpha_max", float("nan")))
            wA = (m.A.weight * m.causal_mask).detach()
            # NOTE: replicates precompute()'s masked-kernel L1 (dim=(1,2)) and cap exactly, but NOT
            # the optional enforce_bistability center-tap clamp. All VARIANT_SPECS set
            # enforce_bistability=False, so capped==effective here; revisit if a bistable variant
            # is ever profiled for this figure.
            raw = wA.abs().sum(dim=(1, 2)).float().cpu().numpy()       # [Cout] operator-norm UB
            if getattr(m, "spectral_cap", False):
                rho = float(m.spectral_rho)
                cap = np.minimum(raw, rho)                            # = raw*(rho/max(raw,rho))
            else:
                cap = raw.copy()
            raws.append(raw); caps.append(cap)
        if not raws:
            return
        aeff_dir = ARTIFACTS_DIR / "aeff"
        aeff_dir.mkdir(parents=True, exist_ok=True)
        np.savez(aeff_dir / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}.npz",
                 raw=np.concatenate(raws), capped=np.concatenate(caps),
                 rho=np.array(rho if rho is not None else np.nan, dtype="float32"),
                 alpha_max=np.array(alpha_max if alpha_max is not None else np.nan, dtype="float32"))
    except Exception as e:
        print(f"[aeff save failed: {e}]", end=" ")


def _save_branches(nf, model_name, dataset_name, horizon, seed, cv_df, alias,
                   Y_df, val_size, test_size):
    """Capture per-dilation-scale forecasts for the parallel multi-scale ensemble (C2) -> the
    scale-disagreement + spread-vs-error diagnostics (UQ). Re-evaluates the
    ALREADY-FITTED model once per branch via cross_validation(use_fitted=True) with the core
    forward routed to a single dilation branch (core._uq_branch_idx). Each branch forecast thus
    comes back inverse-normalized and keyed by (unique_id, ds, cutoff) through NF's own pipeline,
    so it joins to the ensemble cv_df (which carries y) with NO positional or scaling assumptions.
    Only for single-block parallel-ensemble variants; skip-if-exists; strictly non-fatal."""
    import numpy as np
    import pandas as pd
    out_path = ARTIFACTS_DIR / "branches" / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}.npz"
    if out_path.exists():
        return  # skip-if-exists (resume-safe; the extra use_fitted CV passes are the cost avoided)
    core = nf.models[0].model  # CeNNModel1D
    try:
        stack = getattr(core, "stack", None)
        blocks = getattr(stack, "blocks", None)
        if blocks is None or len(blocks) != 1 or not hasattr(blocks[0], "branches"):
            return  # not a single parallel-ensemble block -> no per-scale decomposition
        branches_mod = blocks[0].branches
        n_scales = len(branches_mod)
        # Read the actual per-branch dilation from the model (don't assume the 2**i schedule) so the
        # heatmap y-axis can never silently mislabel if the schedule ever changes.
        dilations = [int(getattr(b, "cell", b).dilation) for b in branches_mod]
        keys = ["unique_id", "ds", "cutoff"]
        base = cv_df[keys + ["y", alias]].rename(columns={alias: "ensemble"}).copy()

        for k in range(n_scales):
            try:
                core._uq_branch_idx = k
                bdf = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size,
                                          n_windows=None, refit=False, use_fitted=True)
            finally:
                core._uq_branch_idx = None
            if isinstance(bdf.index, pd.MultiIndex):
                bdf = bdf.reset_index()
            bk = bdf[keys + [alias]].rename(columns={alias: f"branch_{k}"})
            base = base.merge(bk, on=keys, how="inner")

        if base.empty:
            print(f"[branches: empty frame after join for {model_name} @ {dataset_name} "
                  f"H{horizon} -- key mismatch; skipping]", end=" ")
            return
        base = base.sort_values(keys)
        # Bound artifact size: keep a deterministic strided subset of windows. The heatmap averages
        # over windows and the scatter is already a sample, so a few hundred windows is ample; the
        # full test set (e.g. Traffic/H720, 862 series) would otherwise be many GB per artifact.
        wid = base.groupby(["unique_id", "cutoff"], sort=False).ngroup()
        n_windows = int(wid.max()) + 1
        MAX_WINDOWS = 500
        if n_windows > MAX_WINDOWS:
            stride = (n_windows + MAX_WINDOWS - 1) // MAX_WINDOWS
            base = base[(wid % stride) == 0]
        base["step"] = base.groupby(["unique_id", "cutoff"]).cumcount()   # horizon-step index
        branch_cols = [f"branch_{k}" for k in range(n_scales)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                 step=base["step"].to_numpy().astype("int32"),
                 y=base["y"].to_numpy().astype("float32"),
                 ensemble=base["ensemble"].to_numpy().astype("float32"),
                 branches=base[branch_cols].to_numpy().astype("float32"),  # [N_rows, n_scales]
                 dilations=np.array(dilations, dtype="int32"))
    except Exception as e:
        print(f"[branches save failed: {type(e).__name__}: {e}]", end=" ")
    finally:
        core._uq_branch_idx = None  # defensive: never leave the diagnostic route armed


# ---------------------------------------------------------------------------
# Batch runner: run multiple experiments sequentially
# ---------------------------------------------------------------------------

def run_batch(models: list[str], datasets: list[str], horizons: list[int],
              seeds: list[int], max_steps: int = MAX_STEPS,
              save_artifacts: bool = False, dry_run: bool = False,
              wandb_cfg: dict | None = None, main_seed_rule: bool = False):
    """Run all combinations, skipping already-completed ones."""

    # The headline CeNN variant gets 5 seeds (SEEDS_MAIN) ONLY for the full `--cenn-all` campaign.
    # For `--sanity`, `--cenn-main` (already SEEDS_MAIN), or any explicit `--models ... --seeds ...`
    # debugging, the passed seeds are respected verbatim — main_seed_rule gates the auto-upgrade so
    # it can't surprise those.
    main_model = f"CeNN_{CENN_MAIN_VARIANT}"

    def seeds_for(model):
        return SEEDS_MAIN if (main_seed_rule and model == main_model) else seeds

    # Count total and pending
    total = sum(len(seeds_for(m)) for m in models) * len(datasets) * len(horizons)
    pending = []
    for model in models:
        for dataset in datasets:
            for horizon in horizons:
                for seed in seeds_for(model):
                    if FORCE_RERUN or not result_exists(model, dataset, horizon, seed):
                        pending.append((model, dataset, horizon, seed))

    skipped = total - len(pending)
    print(f"\n{'='*70}")
    print(f"  Experiment batch: {total} total, {skipped} already done, "
          f"{len(pending)} to run")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Horizons: {horizons}")
    if main_seed_rule and main_model in models:
        print(f"  Seeds: {seeds} (+ {main_model} gets SEEDS_MAIN={SEEDS_MAIN})")
    else:
        print(f"  Seeds: {seeds}")
    print(f"{'='*70}\n")

    if dry_run:
        print("DRY RUN — would run:")
        for m, d, h, s in pending:
            print(f"  {m} | {d} | H={h} | seed={s}")
        print(f"\nTotal: {len(pending)} runs")
        return

    if not pending:
        print("All runs already completed. Nothing to do.")
        return

    completed = 0
    failed = 0
    for i, (model, dataset, horizon, seed) in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] ", end="")
        # Belt-and-suspenders: run_single catches its own body, but anything outside it
        # (e.g. load_dataset) must NOT abort a multi-hour batch — isolate every cell.
        try:
            result = run_single(model, dataset, horizon, seed,
                                max_steps=max_steps,
                                save_artifacts=save_artifacts,
                                wandb_cfg=wandb_cfg)
        except Exception as e:
            print(f"\n  CELL CRASHED ({model}|{dataset}|H{horizon}|seed{seed}): "
                  f"{type(e).__name__}: {e} — skipping, will retry on re-run")
            import traceback as _tb; _tb.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            failed += 1
            continue
        if result and result.get("mse") is not None:
            completed += 1
        elif result:
            failed += 1

    print(f"\n{'='*70}")
    print(f"  Done: {completed} completed, {failed} failed, {skipped} skipped")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Incremental experiment runner for AMS-CeNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # What to run
    parser.add_argument("--models", nargs="+",
                        help="Model names (e.g., CeNN_C1-BoundedTau PatchTST TCN)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL,
                        help="Dataset names")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS,
                        help="Forecast horizons")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Random seeds")

    # Presets
    parser.add_argument("--sanity", action="store_true",
                        help="Quick sanity check: CeNN_S0-StableBase + "
                             "CeNN_C1-BoundedTau + CeNN_CeNN-Full on ETTh2, H=96, seed=1")
    parser.add_argument("--cenn-all", action="store_true",
                        help="Run all CeNN variants (MAIN + ABLATION + APPENDIX)")
    parser.add_argument("--cenn-main", action="store_true",
                        help=f"Run the paper main CeNN variant ({CENN_MAIN_VARIANT}) "
                             f"with 5 seeds")
    parser.add_argument("--baselines-rerun", action="store_true",
                        help="Run baselines that need re-running "
                             "(TCN, TimesNet, TSMixer, TimeMixer, xLSTM)")

    # Options
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save predictions and tau values")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without running")
    parser.add_argument("--force", action="store_true",
                        help="Bypass skip-if-exists and re-run cells even if results exist "
                             "(deterministic: rewrites identical metrics). Use with --save-artifacts "
                             "or --cenn-main to BACKFILL artifacts for cells that lack them.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log each run to Weights & Biases (requires `wandb login`)")
    parser.add_argument("--wandb-project", default="cenn-ltsf",
                        help="W&B project name (default: cenn-ltsf)")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity/team (default: your default entity)")
    parser.add_argument("--no-baseline-checkpoints", action="store_true",
                        help="Skip saving baseline (non-CeNN) checkpoints to halve ckpt storage "
                             "(CeNN checkpoints are always saved for the paper's figures)")

    args = parser.parse_args()

    # Apply the baseline-checkpoint policy (module global read by run_single).
    global SAVE_BASELINE_CKPTS
    SAVE_BASELINE_CKPTS = not args.no_baseline_checkpoints
    global FORCE_RERUN
    FORCE_RERUN = args.force
    if FORCE_RERUN:
        print("[--force] skip-if-exists DISABLED: existing cells will re-run and their result JSON "
              "+ checkpoint will be OVERWRITTEN. Metrics are identical only if the env/code is "
              "unchanged (same torch/CUDA, CUBLAS_WORKSPACE_CONFIG). Intended use: artifact backfill.")

    # Resolve presets
    if args.sanity:
        # S0 (substrate) + C1 (adaptive tau) + CeNN-Full (exercises the parallel-ensemble
        # + patch path end-to-end on a fresh machine).
        args.models = ["CeNN_S0-StableBase", "CeNN_C1-BoundedTau", "CeNN_CeNN-Full"]
        args.datasets = ["ETTh2"]
        args.horizons = [96]
        args.seeds = [1]

    if args.cenn_all:
        args.models = [f"CeNN_{v}" for v in CENN_VARIANTS]
        args.save_artifacts = True  # CeNN runs: save per-seed predictions (seed-ensemble UQ)

    if args.cenn_main:
        args.models = [f"CeNN_{CENN_MAIN_VARIANT}"]
        args.seeds = SEEDS_MAIN  # 5 seeds
        args.save_artifacts = True  # the 5-seed main is THE deep-ensemble UQ source — must save

    if args.baselines_rerun:
        args.models = BASELINES_RERUN

    if not args.models:
        parser.error("Specify --models, --sanity, --cenn-all, --cenn-main, "
                     "or --baselines-rerun")

    wandb_cfg = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            parser.error("--wandb given but wandb isn't importable (install: `uv pip install wandb`).")
        authed = bool(os.environ.get("WANDB_API_KEY"))
        for _fn in (".netrc", "_netrc"):
            _f = Path.home() / _fn
            if _f.exists() and "api.wandb.ai" in _f.read_text(errors="ignore"):
                authed = True
        if not authed:
            parser.error("--wandb given but W&B isn't authenticated. "
                         "Run `wandb login` (or set WANDB_API_KEY), then re-run.")
        wandb_cfg = {"project": args.wandb_project, "entity": args.wandb_entity}

    run_batch(
        models=args.models,
        datasets=args.datasets,
        horizons=args.horizons,
        seeds=args.seeds,
        max_steps=args.max_steps,
        save_artifacts=args.save_artifacts,
        dry_run=args.dry_run,
        wandb_cfg=wandb_cfg,
        main_seed_rule=args.cenn_all,  # 5-seed auto-upgrade only for the full campaign
    )


if __name__ == "__main__":
    main()
