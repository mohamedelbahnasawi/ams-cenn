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

    # Resume: run the same command again; completed runs are skipped
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

# CUBLAS_WORKSPACE_CONFIG must be set before torch initializes CUDA for deterministic
# matmuls; an externally set value is respected.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.set_float32_matmul_precision("medium")
# warn_only=True: ops without a deterministic kernel (e.g. the TimesNet FFT) warn instead of
# raising.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Save baseline (non-CeNN) checkpoints. CeNN checkpoints are always saved (the figures need
# them). Set from --no-baseline-checkpoints in main().
SAVE_BASELINE_CKPTS = True
# Bypass skip-if-exists and re-run cells whose result JSON exists. Runs are deterministic, so
# this rewrites the same metrics; the point is to regenerate artifacts (predictions/tau/branches)
# for cells saved without --save-artifacts. Set by --force.
FORCE_RERUN = False
# Also save per-window predictions for non-CeNN baselines (forecast-overlay figure). Writes
# artifacts/predictions/*.npz only, never results/. Set by --save-baseline-preds.
SAVE_BASELINE_PREDS = False


def _provenance():
    """Python/torch/CUDA/git/datasetsforecast versions stored with every result. Computed once at import."""
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
# Model builders
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

# Optional Weights & Biases logging (--wandb). A missing wandb must not break a run; the
# per-run JSON in experiments/results/ is what the tables read.
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
    # Inference batch size only affects eval speed (results are invariant to
    # inference_windows_batch_size). Unset keeps each model's default.
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
        early_stop_patience_steps=int(os.environ.get("CENN_PATIENCE", "5")),   # CENN_PATIENCE: raise it
        # together with CENN_MAX_STEPS so longer training is not cut short by early stopping.
        val_check_steps=100,   # Nixtla long-horizon protocol
        batch_size=bs,
        scaler_type=os.environ.get("BASELINE_SCALER_OVERRIDE", SCALER_TYPE),  # scaler A/B for baselines; use with a CENN_EXP_DIR scratch dir
        random_seed=seed,
        accelerator="auto",
        precision=PRECISION,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 0.01},
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": max_steps, "eta_min": 1e-5},
    )


# --- CeNN variants ---
#
# VARIANT_SPECS keys are the variant names: model_name == f"CeNN_{key}", and the key is
# part of the result JSON filename. Each variant is _VARIANT_BASE (== C1-BoundedTau) plus
# only the kwargs it changes. Min-max-era variants use scaler_type='minmax'; anchored
# variants use identity. K-sweep variants set K via _VARIANT_K. Feedback-conv evals per
# step: euler=1, exp_euler=1, heun=2, rk4=4, so K2-Heun and K4-Euler are iso-MAC.
_VARIANT_BASE = dict(
    adaptive_tau=True, dilation_schedule="none", multiscale_mode="none",
    integrator="euler", var_mix=False, cross_var="none", pointwise_mix=False,
    channel_groups=1, patch_len=None, stride=None, head_type="linear",
    alpha_min=0.5, alpha_max=0.99, spectral_cap=True, scaler_type="minmax",
)
_ANCH = dict(revin=True, revin_mode="last_only", scaler_type="identity")   # anchored protocol: in-model last-value anchor, no pipeline scaler
VARIANT_SPECS = {
    # ---- MAIN ----
    "S0-StableBase":              {**_VARIANT_BASE, "adaptive_tau": False},                                          # fixed bounded α (substrate)
    "C1-BoundedTau":              {**_VARIANT_BASE},                                                                 # = base (adaptive τ, reference)
    "C2-MultiScaleEnsemble":      {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},  # S0 + multi-scale ensemble
    # C1+C2: adaptive tau + multi-scale ensemble, patch-free, linear head. The no-skip
    # ablation of C1C2-Skip-K2 below.
    "C1C2-Ensemble":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble"},
    "CeNN-Full":                  {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8},  # efficiency/patching arm
    # ---- ABLATION (leave-one-out off C1-BoundedTau unless noted) ----
    "ABL-SpectralCapOff":         {**_VARIANT_BASE, "spectral_cap": False},                                          # −spectral cap
    "ABL-Patch":                  {**_VARIANT_BASE, "patch_len": 16, "stride": 8},                                   # +patch
    # gate bounds widened to [0,1]; spectral_cap stays on so only the bound changes.
    "ABL-GateParam-Unbounded":    {**_VARIANT_BASE, "alpha_min": 0.0, "alpha_max": 1.0},
    "ABL-CrossVar-Pointwise":     {**_VARIANT_BASE, "pointwise_mix": True},                                          # latent 1×1 channel mix
    "ABL-CrossVar-VarMix":        {**_VARIANT_BASE, "cross_var": "varmix"},                                          # dense O(V²) V×V mix
    "ABL-CrossVar-STAR":          {**_VARIANT_BASE, "cross_var": "star"},                                            # O(V) STAR core
    "ABL-ChannelGroups-G4":       {**_VARIANT_BASE, "channel_groups": 4},                                            # grouped conv over hidden
    # K x integrator sweep (base = C1; K via _VARIANT_K). The K=8 Euler point is
    # C1-BoundedTau itself; the K-sweep plot keys on (cenn_K, integrator), not the name.
    "K4-Euler":                   {**_VARIANT_BASE},                                                                 # K=4
    "K4-Heun":                    {**_VARIANT_BASE, "integrator": "heun"},                                           # K=4
    "K4-ExpEuler":                {**_VARIANT_BASE, "integrator": "exp_euler"},                                      # K=4
    "K2-Euler":                   {**_VARIANT_BASE},                                                                 # K=2 (completes the Euler K-curve)
    "K2-Heun":                    {**_VARIANT_BASE, "integrator": "heun"},                                           # K=2 (iso-MAC vs K4-Euler)

    # Scale-count ablation around C2-MultiScaleEnsemble (n_scales=4). num_layers == n_scales
    # in parallel_ensemble mode (overrides _CENN_FIXED).
    "ABL-Scales-2":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 2},  # dilations [1,2]
    "ABL-Scales-3":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 3},  # [1,2,4]
    "ABL-Scales-5":               {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble", "num_layers": 5},  # [1,2,4,8,16]
    # ---- APPENDIX ----
    # CeNN-RawBase: no gate bounds, no spectral cap.
    "CeNN-RawBase":               {**_VARIANT_BASE, "adaptive_tau": False, "alpha_min": 0.0, "alpha_max": 1.0, "spectral_cap": False},
    "APP-C2Form-DilatedTemplate": {**_VARIANT_BASE, "dilation_schedule": "exponential"},                            # dilated templates (vs ensemble)
    "APP-Head-MLP":               {**_VARIANT_BASE, "patch_len": 16, "stride": 8, "head_type": "mlp"},               # MLP head (paired w/ patch)
    "APP-RK4":                    {**_VARIANT_BASE, "integrator": "rk4"},                                            # RK4 accuracy ceiling
    # Additivity check: does the multi-scale ensemble add accuracy on top of an MLP head?
    "APP-MultiScale-Patch-MLP":   {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8, "head_type": "mlp"},  # CeNN-Full + MLP head
    "APP-MultiScale-MLP":         {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "head_type": "mlp"},     # multi-scale + MLP head, no patch

    # C2-MultiScaleEnsemble at K=2: does the flat K-surface measured on C1 transfer to the
    # multi-scale model?
    "K2-C2Ensemble":              {**_VARIANT_BASE, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},

    # Capacity scaling: C1C2 at hidden_dim 128/256 on ECL/Traffic (is the high-V gap a capacity
    # or a structural limit?). Not in any CENN_VARIANTS list; run explicitly via --models.
    "CAP-Ensemble-H128":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 128},
    "CAP-Ensemble-H256":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 256},
    # hidden 512 >= V_ECL (321): no compression on ECL; Traffic (V=862) is still 1.7:1.
    # windows_batch_size halved for VRAM.
    "CAP-Ensemble-H512":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "hidden_dim": 512, "windows_batch_size": 128},
    # Same capacity curve with the linear skip (C1C2-Skip at K=2). Analysis rows only; not in
    # any CENN_VARIANTS list. windows_batch_size lowered at high hidden for VRAM.
    "CAP-Skip-H128":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 128},
    "CAP-Skip-H256":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 256, "windows_batch_size": 128},
    "CAP-Skip-H512":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "hidden_dim": 512, "windows_batch_size": 64},

    # Two candidate fixes for the amplitude damping of the CeNN stack ahead of the linear head:
    # a raw-input linear skip, or a GELU/SiLU read-out. Not in any CENN_VARIANTS list.
    "C1C2-Skip":                  {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
    "C1C2-GELUout":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "readout_act": "gelu"},
    "C1C2-SiLUout":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "readout_act": "silu"},
    "C1C2-Skip-GELUout":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "readout_act": "gelu"},
    # Min-max-era headline: C1C2 ensemble + zero-init linear skip at K=2. K=2 is
    # accuracy-neutral vs K=8 on this architecture and 4x cheaper. K set via _VARIANT_K;
    # C1C2-Skip above is the K=8 reference.
    "C1C2-Skip-K2":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
    # Cross-channel variants on the skip + multiscale architecture (K=2).
    "C1C2-Skip-K2-STAR":          {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "star"},      # O(V) STAR aggregate-redistribute
    "C1C2-Skip-K2-Pointwise":     {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "pointwise_mix": True},    # latent 1x1 channel mix
    # Single-seed probes that preceded the anchored protocol (run under a scratch CENN_EXP_DIR).
    # They remove two handicaps of the trunk vs the skip: trunk dropout 0.25, and the absence of
    # in-model instance normalization.
    "PROBE-Dropout0":             {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "dropout": 0.0},
    "PROBE-RevIN":                {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True},
    "PROBE-RevIN-Dropout0":       {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "dropout": 0.0},
    "PROBE-RevIN-Last":           {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "revin_mode": "last"},       # NLinear anchor + std
    "PROBE-RevIN-LastOnly":       {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "revin_mode": "last_only"},  # NLinear anchor, no std
    # Contamination probe arms: recover the bounded-input robustness min-max gave.
    "PROBE-LastMAD":              {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "revin_mode": "last_mad"},
    "PROBE-LastOnly-Squash":      {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "revin_mode": "last_only", "trunk_squash": True},
    "PROBE-LastMAD-Squash":       {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "revin": True, "revin_mode": "last_mad", "trunk_squash": True},
    # ---- Anchored protocol: in-model last-value anchor (NLinear form), no pipeline scaler,
    # applied to every variant. This is the headline protocol; C1C2-Skip-K2 stays as the
    # normalization-ablation row. All rows at K=2 unless the name says otherwise.
    # scaler_type=identity is in the spec, so CENN_DATASET_SCALER is a no-op here.
    "AMS-Anc":                    {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True},                       # headline (== PROBE-RevIN-LastOnly)
    "S0-Anc":                     {**_VARIANT_BASE, **_ANCH, "adaptive_tau": False},                                                              # substrate
    "C1-Anc":                     {**_VARIANT_BASE, **_ANCH},                                                                                     # + bounded gate
    "C2-Anc":                     {**_VARIANT_BASE, **_ANCH, "adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},                      # + multi-scale
    "C1C2-Anc":                   {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble"},                                             # -skip ablation
    # Stability ablations on the headline: gate bounds widened to [0, 1] (cap kept) / template-norm cap removed.
    "AMS-Anc-GateUnbounded":      {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "alpha_min": 0.0, "alpha_max": 1.0},
    "AMS-Anc-CapOff":             {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "spectral_cap": False},
    # Integrator check at the shipped K=2: Heun, exponential Euler, RK4 on the headline.
    "AMS-Anc-Heun":               {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "integrator": "heun"},
    "AMS-Anc-ExpEuler":           {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "integrator": "exp_euler"},
    "AMS-Anc-RK4":                {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "integrator": "rk4"},
    "SkipOnly-Anc":               {**_VARIANT_BASE, **_ANCH, "trunk_type": "none", "linear_skip": True},                                          # == NLinear
    "MLPSkip-Anc":                {**_VARIANT_BASE, **_ANCH, "trunk_type": "mlp", "linear_skip": True},                                           # generic-trunk control
    "AMS-Anc-K4":                 {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True},                       # K-curve
    "AMS-Anc-K8":                 {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True},                       # K-curve
    "FrozenSkip-TrainTrunk-Anc":  {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True,
                                   "warm_start_from": "SkipOnly-Anc", "freeze": "skip"},                                                         # pathway pilot
    # Seasonal-dilation probe: branches aligned to the data's periods instead of
    # powers of two. Resolved per dataset frequency in build_cenn (dilations_by_freq -> dilations).
    "AMS-Anc-Seas":               {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True,
                                   "dilations_by_freq": {"h": [1, 24, 48, 168], "15min": [1, 96, 192, 672], "10min": [1, 144, 288, 1008]}},
    # Hardware probe: the in-block LayerNorm replaced by a per-channel affine / nothing.
    "AMS-Anc-Affine":             {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "block_norm": "affine"},
    "AMS-Anc-NoNorm":             {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "block_norm": "none"},
    "AMS-Anc-STAR":               {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "star"},
    "AMS-Anc-Pointwise":          {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "pointwise_mix": True},
    "AMS-Anc-VarMix":             {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "varmix"},
    "AMS-Anc-G4":                 {**_VARIANT_BASE, **_ANCH, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "channel_groups": 4},
    "C1C2-Skip-K2-VarMix":        {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "cross_var": "varmix"},    # dense O(V^2) V x V mix
    "C1C2-Skip-K2-G4":            {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "channel_groups": 4},      # grouped conv over hidden
    # Generic-trunk control: the CeNN dynamics swapped for an MLP-Mixer trunk, with input_proj,
    # head and zero-init skip unchanged. K/integrator/multiscale are inherited but ignored by the
    # MLP trunk (cenn_K is nulled in the result JSON).
    "MLP-Skip":                   {**_VARIANT_BASE, "trunk_type": "mlp", "linear_skip": True},
    # Skip-only control: no trunk, the zero-init linear residual alone. With C1C2-Ensemble (trunk,
    # no residual) and MLP-Skip (generic trunk + residual) this completes the pathway decomposition.
    "Skip-Only":                  {**_VARIANT_BASE, "trunk_type": "none", "linear_skip": True},
    # Pathway-attribution arms, two-stage: stage 1 fits one path alone, stage 2 loads those
    # weights, freezes that path and trains only the other. Separates "the skip starves the trunk
    # of gradient" from "there is little left for the local nonlinear path after a linear fit".
    # Requires the stage-1 checkpoint.
    "FrozenSkip-TrainTrunk":      {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True,
                                   "warm_start_from": "Skip-Only", "freeze": "skip"},
    "FrozenTrunk-TrainSkip":      {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True,
                                   "warm_start_from": "C1C2-Ensemble-K2", "freeze": "trunk"},
    # Stage-1 source for FrozenTrunk-TrainSkip: C1 gate + C2 ensemble, no linear residual, K=2
    # (C1C2-Ensemble is K=8 and K2-C2Ensemble has no C1 gate, so neither can serve).
    "C1C2-Ensemble-K2":           {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble"},
    # C1C2-Skip-K2 with exp_euler instead of euler (the integrator sweep above was on the C1 chassis).
    "C1C2-Skip-ExpEuler-K2":      {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True, "integrator": "exp_euler"},
    # K=4 midpoint of the C1C2-Skip K-curve {K8 (C1C2-Skip), K4, K2 (C1C2-Skip-K2)}; K via _VARIANT_K.
    "C1C2-Skip-K4":               {**_VARIANT_BASE, "multiscale_mode": "parallel_ensemble", "linear_skip": True},
}

# Fixed CeNN architecture/training settings shared by every variant.
_CENN_FIXED = dict(
    hidden_dim=64, N=1, num_layers=4, dropout=0.25, neighborhood=3,
    alpha_init=0.9, enforce_bistability=False, cross_channel=False,
    spectral_rho=0.9, gradient_clip_val=1.0,
    # windows_batch_size=256 rather than the NeuralForecast default 1024: the K-loop is
    # BPTT-unrolled (K x num_layers x evals/step), so memory scales with wbs x V x evals. At 1024,
    # RK4 OOMs on 24 GB and the C2 ensemble on Traffic (V=862, 4 branches) exceeds 80 GB; at 256
    # the worst cells stay under 24 GB (RK4/ETT ~16 GB, C2/Traffic ~6 GB). Eval is unchanged.
    windows_batch_size=256,
    # multiscale_mode and integrator are set by every VARIANT_SPECS entry and arrive via **spec;
    # adding them here would raise 'multiple values for keyword argument'.
)


# K values for the K-sweep variants (overrides the default K=8).
_VARIANT_K = {
    "K4-Euler": 4, "K4-Heun": 4, "K4-ExpEuler": 4,
    "K2-Euler": 2, "K2-Heun": 2,
    "K2-C2Ensemble": 2,
    "C1C2-Skip-K2": 2,
    # Pathway-attribution arms and their stage-1 source: K=2 so the comparison against
    # C1C2-Skip-K2 isolates the pathway.
    "Skip-Only": 2, "C1C2-Ensemble-K2": 2,
    "FrozenSkip-TrainTrunk": 2, "FrozenTrunk-TrainSkip": 2,
    "PROBE-Dropout0": 2, "PROBE-RevIN": 2, "PROBE-RevIN-Dropout0": 2,  # probe arms on the headline (K=2)
    "PROBE-RevIN-Last": 2, "PROBE-RevIN-LastOnly": 2,
    "PROBE-LastMAD": 2, "PROBE-LastOnly-Squash": 2, "PROBE-LastMAD-Squash": 2,
    "AMS-Anc": 2, "S0-Anc": 2, "C1-Anc": 2, "C2-Anc": 2, "C1C2-Anc": 2, "SkipOnly-Anc": 2, "MLPSkip-Anc": 2,
    "AMS-Anc-K4": 4, "AMS-Anc-K8": 8, "FrozenSkip-TrainTrunk-Anc": 2,
    "AMS-Anc-GateUnbounded": 2, "AMS-Anc-CapOff": 2, "AMS-Anc-Heun": 2, "AMS-Anc-ExpEuler": 2, "AMS-Anc-RK4": 2,
    "AMS-Anc-STAR": 2, "AMS-Anc-Pointwise": 2, "AMS-Anc-VarMix": 2, "AMS-Anc-G4": 2, "AMS-Anc-Seas": 2, "AMS-Anc-Affine": 2, "AMS-Anc-NoNorm": 2,
    "C1C2-Skip-K2-STAR": 2, "C1C2-Skip-K2-Pointwise": 2,  # cross-channel variants on headline (K=2)
    "C1C2-Skip-K2-VarMix": 2, "C1C2-Skip-K2-G4": 2,
    "C1C2-Skip-ExpEuler-K2": 2,
    "C1C2-Skip-K4": 4,    # K=4 midpoint of the C1C2-Skip K-curve
    "CAP-Skip-H128": 2, "CAP-Skip-H256": 2, "CAP-Skip-H512": 2,   # headline arch (K=2) at higher capacity
}


def _warm_start(model, source_variant, dataset_name, horizon, seed, scope):
    """Load only the `scope` pathway ('skip' or 'trunk') from a stage-1 checkpoint.

    The loaded scope equals the frozen scope: the frozen path is inherited, everything else
    starts from this run's own initialization. Returns the sorted list of loaded tensor names.
    """
    ckpt_dir = CHECKPOINTS_DIR / f"CeNN_{source_variant}__{dataset_name}__H{horizon}__seed{seed}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"warm start needs the stage-1 checkpoint {ckpt_dir}. Run CeNN_{source_variant} on "
            f"{dataset_name} H{horizon} seed{seed} first."
        )
    src = NeuralForecast.load(path=str(ckpt_dir)).models[0].model.state_dict()
    want = {k: v for k, v in src.items()
            if ((k.startswith("skip.") or k.startswith("revin_")) if scope == "skip"
                else not (k.startswith("skip.") or k.startswith("revin_")))}
    tgt = model.model.state_dict()
    usable = {k: v for k, v in want.items() if k in tgt and tgt[k].shape == v.shape}
    if not usable:
        raise RuntimeError(
            f"warm start from {ckpt_dir} matched NO '{scope}' tensors "
            f"(source has {sorted(want)[:4]}...) -- key or shape mismatch"
        )
    model.model.load_state_dict(usable, strict=False)
    print(f"    [warm-start] {len(usable)}/{len(want)} '{scope}' tensors from "
          f"CeNN_{source_variant}; all other parameters at this run's own init")
    return sorted(usable)


def _apply_freeze(model, which):
    """Freeze one pathway. `which` is 'skip' (the zero-init linear residual) or 'trunk'
    (everything else: the cellular/MLP stack, the input projection and the head).

    Sets requires_grad=False and stores a snapshot of the frozen tensors on the model;
    `assert_frozen_unchanged` compares against it after fit.
    """
    if which not in ("skip", "trunk"):
        raise ValueError(f"freeze must be 'skip' or 'trunk', got {which!r}")
    inner = model.model
    frozen = []
    for name, p in inner.named_parameters():
        # The in-model anchor's affine (revin_weight/bias) is fitted in stage 1 with the skip and
        # frozen with it: it is the normalization of the linear path, not a trunk parameter.
        is_skip = name.startswith("skip.") or name.startswith("revin_")
        if (which == "skip" and is_skip) or (which == "trunk" and not is_skip):
            p.requires_grad = False
            frozen.append(name)
    if not frozen:
        raise RuntimeError(f"freeze='{which}' matched no parameters -- naming changed?")
    trainable = [n for n, p in inner.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"freeze='{which}' left nothing trainable")
    model._freeze_which = which
    model._freeze_guard = {n: p.detach().clone()
                           for n, p in inner.named_parameters() if not p.requires_grad}
    print(f"    [freeze] {which}: {len(frozen)} tensors frozen, {len(trainable)} trainable")
    return frozen


def assert_frozen_unchanged(model):
    """Raise if any frozen tensor differs from its pre-fit snapshot. No-op without a snapshot."""
    guard = getattr(model, "_freeze_guard", None)
    if not guard:
        return
    live = dict(model.model.named_parameters())
    drifted = [n for n, before in guard.items()
               if n not in live or not torch.equal(live[n].detach().cpu(), before.cpu())]
    if drifted:
        raise RuntimeError(
            f"frozen parameters changed during fit ({len(drifted)}/{len(guard)}): {drifted[:5]}. "
            f"The '{getattr(model, '_freeze_which', '?')}' arm is invalid."
        )
    print(f"    [freeze] all {len(guard)} frozen tensors unchanged after fit")


def build_cenn(variant, h, n_series, seed, max_steps, K=8, dataset_name=None):
    """Build a CeNN model for a named variant from VARIANT_SPECS.

    `variant` is the spec key (e.g. 'CeNN-Full'), the string after 'CeNN_' in model_name.
    Returns (model, name, scaler_type); the scaler is per-variant, not the global SCALER_TYPE.
    K-sweep variants take K from _VARIANT_K and ignore the caller-supplied K.
    """
    if variant not in VARIANT_SPECS:
        raise ValueError(
            f"Unknown CeNN variant: {variant!r}. "
            f"Known: {sorted(VARIANT_SPECS)}."
        )
    spec = dict(VARIANT_SPECS[variant])
    scaler_type = spec.pop("scaler_type")
    # Runner-level keys for the two-stage pathway-attribution arms -- popped so they never
    # reach the CeNN constructor (which would reject them as unknown kwargs).
    warm_start_from = spec.pop("warm_start_from", None)
    freeze_path = spec.pop("freeze", None)
    # Seasonal-dilation probe: pick the dilation list by the dataset's sampling frequency.
    dil_by_freq = spec.pop("dilations_by_freq", None)
    if dil_by_freq is not None:
        freq = DATASET_INFO.get(dataset_name, {}).get("freq")
        if freq not in dil_by_freq:
            raise ValueError(f"{variant}: no dilation list for frequency {freq!r} (dataset {dataset_name})")
        spec["dilations"] = dil_by_freq[freq]
    # Per-dataset scaler override (config.CENN_DATASET_SCALER), applied to every CeNN variant
    # on that dataset. Datasets not listed keep the per-variant default.
    if dataset_name in CENN_DATASET_SCALER:
        scaler_type = CENN_DATASET_SCALER[dataset_name]
    # CENN_SCALER_OVERRIDE swaps the scaler for an A/B run; use with a CENN_EXP_DIR scratch dir.
    scaler_type = os.environ.get("CENN_SCALER_OVERRIDE", scaler_type)
    name = f"CeNN_{variant}"

    kwargs = _common_kwargs(h, seed, max_steps, dataset_name=dataset_name)
    kwargs["scaler_type"] = scaler_type                  # per-variant scaler

    effective_K = _VARIANT_K.get(variant, K)  # K-sweep variants override the default K

    # Spec keys override _CENN_FIXED keys (e.g. ABL-Scales-* set num_layers). Local copy:
    # profile_efficiency reads _CENN_FIXED.
    fixed = {k: v for k, v in _CENN_FIXED.items() if k not in spec}

    model = CeNN(
        **kwargs,
        n_series=n_series,
        K=effective_K,
        **fixed,
        **spec,           # wires all params (patch/head/var_mix/groups/gate/cap/multiscale/integrator)
    )
    model.alias = name

    # Two-stage arms: warm-start first, freeze second, so the snapshot taken by _apply_freeze
    # records the loaded values.
    if warm_start_from is not None:
        if freeze_path is None:
            raise ValueError(
                f"{variant}: warm_start_from without freeze is ambiguous -- the warm-start scope "
                f"is defined as the path that gets frozen."
            )
        _warm_start(model, warm_start_from, dataset_name, h, seed, scope=freeze_path)
    if freeze_path is not None:
        _apply_freeze(model, freeze_path)

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
    # Official S4D diagonal SSM baseline.
    kw = _common_kwargs(h, seed, max_steps, dataset_name=dataset_name)
    kw["precision"] = "32-true"  # complex SSM kernel (cfloat FFT/einsum); fp32 as in the authors' code
    # S4D diverges on the spiky ETT*2 series under the identity scaler (ETTh2 H96 MSE 1.83), so it
    # gets the same per-window min-max policy as the CeNN (identity on Weather only).
    # BASELINE_SCALER_OVERRIDE still wins if set.
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

    # Variable-count ablation: CENN_VAR_SUBSET=K keeps K series chosen by a fixed permutation
    # (nested: the V8 subset is contained in V16, and so on; the same subset for every model and
    # seed). Outputs nest under V{K}/ (see config). Unset means the full dataset.
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
            # For the MLP trunk, K and integrator are inherited but unused; store them as null
            # so the K-sweep and integrator figures skip these rows.
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
            # log the model's own scaler; baselines may override the global default (S4D,
            # BASELINE_SCALER_OVERRIDE).
            run_scaler = model.hparams.get("scaler_type", SCALER_TYPE)
            cenn_meta = {}

        # Optional per-run W&B logger. Catch BaseException: wandb can sys.exit() on stale auth.
        # Keep the Run object so a partial init is finished in `finally`.
        if wandb_cfg is not None and _WANDB_AVAILABLE:
            try:
                wb_logger = WandbLogger(
                    project=wandb_cfg["project"], entity=wandb_cfg.get("entity"),
                    name=f"{model_name}__{dataset_name}__H{horizon}__seed{seed}",
                    group=dataset_name,
                )
                wb_run = wb_logger.experiment  # triggers wandb.init(); finished in `finally`
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
        # LongHorizon2 data is already z-scored on train, so MSE/MAE are on the normalized scale
        # used in the literature.
        val_size, test_size = SPLITS[dataset_name]
        t0 = time.time()
        # Large eval matrices (ECL/Traffic long horizons) go through the chunked path: same
        # windows and metrics, ~5-8 GB peak RAM per chunk instead of 120+ GB single-shot.
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

        # Frozen arms: raise if the frozen path moved. No-op otherwise.
        assert_frozen_unchanged(model)

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
            # Per-scale branch forecasts (scale-disagreement and spread-vs-error diagnostics).
            # One use_fitted CV pass per dilation scale, so only for the headline variant and its
            # first seed, and not on chunked-CV cells (each branch pass would be a single-shot CV).
            if (model_name == f"CeNN_{CENN_MAIN_VARIANT}" and seed == SEEDS_MAIN[0]
                    and chunked is None):
                _save_branches(nf, model_name, dataset_name, horizon, seed, cv_df, alias,
                               Y_df, val_size, test_size)
            elif chunked is not None and model_name == f"CeNN_{CENN_MAIN_VARIANT}":
                print("[branches skipped: chunked-CV cell] ", end="")

        # Baseline predictions for the forecast-overlay figure (artifacts/predictions/ only).
        if (SAVE_BASELINE_PREDS and save_artifacts and not model_name.startswith("CeNN_")
                and not cv_df.empty):
            _save_artifacts(nf, model_name, dataset_name, horizon, seed, cv_df, alias, force_any=True)

        del cv_df

        # Save checkpoint: always for CeNN, for baselines unless --no-baseline-checkpoints.
        # Write to a temp dir and swap, so a kill mid-write cannot destroy a good checkpoint
        # (nf.save(overwrite=True) removes first, then writes).
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
                os.replace(tmp_dir, ckpt_dir)  # directory rename
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
            # Validation MAE on the inverse-normalized scale, comparable across scalers and
            # hyperparameters. NeuralForecast deep-copies the model list at fit time, so the
            # trajectories live on nf.models[0], not on `model`.
            "best_valid_loss": (min(float(v) for _, v in nf.models[0].valid_trajectories)
                                if getattr(nf.models[0], "valid_trajectories", None) else None),
            "final_valid_loss": (float(nf.models[0].valid_trajectories[-1][1])
                                 if getattr(nf.models[0], "valid_trajectories", None) else None),
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
                # evaluation chunk count (null = single-shot); same window set either way.
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
        os.replace(tmp_path, out_path)  # atomic; overwrites an existing dest (needed for --force),
        # unlike Path.rename() on Windows.

        print(f"MSE={mse:.6f}  MAE={mae:.6f}  Time={train_time:.1f}s")
        return result

    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

        # Error result, returned to the caller but not written to disk
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
        # not saved: a re-run retries the cell
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


# Chunked-CV threshold in evaluation rows = (test_size-h+1) * h * n_series. Above it the
# single-shot cv_df (frame + ground-truth join) needs ~120 GB of host RAM at Traffic x H720.
# 2e8 keeps every ETT and Weather cell (Weather H720 = 1.5e8) on the unchunked path.
CV_CHUNK_ROWS = int(float(os.environ.get("CENN_CV_CHUNK_ROWS", str(200_000_000))))  # float(): tolerate "2e8"
# Rows per chunk (~18 GB peak at 1.2e8). Larger values mean fewer chunks and less repeated
# per-chunk dataset prep; only the fp summation order changes.
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
    """Memory-bounded cross-validation for large eval matrices, same protocol as single-shot.

    Partitions the stride-1 test cutoffs into consecutive ranges. Chunk 0 trains once with the
    protocol split: every series is truncated to E0 = (last cutoff of chunk 0) + h steps and
    test_size = E0 - first_cutoff, so the train extent E0 - val - test_size equals
    L - SPLITS_test - val, the full-split boundary. Chunks 1..K-1 are use_fitted=True
    re-evaluations. MSE/MAE accumulate as sums and each chunk's frame is freed before the next.
    The chunks partition the same cutoffs as the single-shot call, so the window set is the same.

    Returns (mse, mae, artifact_df_or_None, n_chunks), or None when series lengths differ
    (caller falls back to single-shot).
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
    # Cutoff positions = 0-based index of each window's last input step (NF convention:
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


def _save_artifacts(nf, model_name, dataset_name, horizon, seed, cv_df, alias, force_any=False):
    """Save per-window test predictions (with index) for the forecast-grid figure and the seed band.

    Only the headline variant is saved unless force_any, and windows are strided to at most
    PRED_MAX_WINDOWS whole windows (a full stride-1 table is ~730 MB per cell and ~1e9 rows for
    ECL/Traffic H720). The window-id stride is deterministic, so seeds stay aligned."""
    import numpy as np

    if not force_any and model_name != f"CeNN_{CENN_MAIN_VARIANT}":
        return  # only the headline variant's predictions are used
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
    """Save the adaptive-tau gate values (C1) for the tau heatmap figure.

    Each CeNNCell1D with adaptive_tau stores last_tau [B,C,L] from its most recent forward.
    Batch-mean gives [C,L] per cell, saved as one .npz under ARTIFACTS_DIR/tau/. Non-adaptive
    variants produce nothing. Errors are printed, not raised."""
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
    """Save the feedback operator norm ||A_eff|| per output channel for the contraction figure.

    For each CeNNCell1D, recompute the masked feedback kernel as precompute() does, take its
    per-output-channel L1 (the conv operator-norm bound the cap enforces), and record both the
    raw norm and the capped one, min(raw, rho). Weight inspection only, no forward pass.
    Errors are printed, not raised."""
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
            # Matches precompute()'s masked-kernel L1 (dim=(1,2)) and cap, but not the optional
            # enforce_bistability center-tap clamp (all variants set enforce_bistability=False).
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
    """Save per-dilation-scale forecasts of the parallel multi-scale ensemble (C2) for the
    scale-disagreement and spread-vs-error diagnostics.

    Re-evaluates the fitted model once per branch via cross_validation(use_fitted=True) with the
    forward routed to one dilation branch (core._uq_branch_idx). Each branch forecast comes back
    inverse-normalized and keyed by (unique_id, ds, cutoff), so it joins the ensemble cv_df
    without positional or scaling assumptions. Single-block parallel-ensemble variants only;
    skips if the artifact exists. Errors are printed, not raised."""
    import numpy as np
    import pandas as pd
    out_path = ARTIFACTS_DIR / "branches" / f"{model_name}__{dataset_name}__H{horizon}__seed{seed}.npz"
    if out_path.exists():
        return  # skip-if-exists
    core = nf.models[0].model  # CeNNModel1D
    try:
        stack = getattr(core, "stack", None)
        blocks = getattr(stack, "blocks", None)
        if blocks is None or len(blocks) != 1 or not hasattr(blocks[0], "branches"):
            return  # not a single parallel-ensemble block -> no per-scale decomposition
        branches_mod = blocks[0].branches
        n_scales = len(branches_mod)
        # Read the per-branch dilation from the model rather than assuming the 2**i schedule.
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

    # With main_seed_rule (--cenn-all) the headline variant runs SEEDS_MAIN; otherwise the
    # passed seeds are used as given.
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
        print("DRY RUN; would run:")
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
        # run_single catches its own body; this also catches failures outside it (e.g.
        # load_dataset) so one cell cannot abort the batch.
        try:
            result = run_single(model, dataset, horizon, seed,
                                max_steps=max_steps,
                                save_artifacts=save_artifacts,
                                wandb_cfg=wandb_cfg)
        except Exception as e:
            print(f"\n  CELL CRASHED ({model}|{dataset}|H{horizon}|seed{seed}): "
                  f"{type(e).__name__}: {e}; skipping, will retry on re-run")
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
    parser.add_argument("--save-baseline-preds", action="store_true",
                        help="Also save per-window predictions for non-CeNN baselines (forecast-overlay "
                             "figure). Writes artifacts/predictions/ only, never results/. "
                             "Use with --save-artifacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without running")
    parser.add_argument("--force", action="store_true",
                        help="Bypass skip-if-exists and re-run cells even if results exist "
                             "(deterministic: rewrites identical metrics). Use with --save-artifacts "
                             "or --cenn-main to backfill artifacts for cells that lack them.")
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
    global SAVE_BASELINE_PREDS
    SAVE_BASELINE_PREDS = args.save_baseline_preds
    if FORCE_RERUN:
        print("[--force] skip-if-exists disabled: existing cells re-run and their result JSON "
              "and checkpoint are overwritten. Metrics are identical only if the environment is "
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
        args.save_artifacts = True  # the 5-seed headline feeds the seed-ensemble figures

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
