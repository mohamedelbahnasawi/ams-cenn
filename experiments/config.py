"""Experiment configuration and constants.

All experiment settings live here; the runner and analysis scripts import from this module.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
# All experiment outputs live under _OUT. CENN_EXP_DIR redirects them to a scratch root
# (results/checkpoints/artifacts/aggregated/efficiency); unset means experiments/.
_OUT = Path(os.environ["CENN_EXP_DIR"]) if os.environ.get("CENN_EXP_DIR") else EXPERIMENTS_DIR
# Lookback sweep: a non-default CENN_INPUT_SIZE nests all outputs under controls/lookback/L{N}/
# so they cannot collide with the L=512 results. The aggregator reads only the L=512 dirs.
_L_OVERRIDE = os.environ.get("CENN_INPUT_SIZE")
if _L_OVERRIDE and _L_OVERRIDE != "512":
    _OUT = _OUT / "controls" / "lookback" / f"L{int(_L_OVERRIDE)}"
# Variable-count sweep: CENN_VAR_SUBSET=K nests outputs under controls/variables/V{K}/.
_V_OVERRIDE = os.environ.get("CENN_VAR_SUBSET")
if _V_OVERRIDE:
    _OUT = _OUT / "controls" / "variables" / f"V{int(_V_OVERRIDE)}"
RESULTS_DIR = _OUT / "results"
# PUBLISHED_DIR is not L-nested: published numbers are literature transcriptions, not run
# outputs. It still follows CENN_EXP_DIR.
PUBLISHED_DIR = (Path(os.environ["CENN_EXP_DIR"]) if os.environ.get("CENN_EXP_DIR")
                 else EXPERIMENTS_DIR) / "published"
CHECKPOINTS_DIR = _OUT / "checkpoints"
ARTIFACTS_DIR = _OUT / "artifacts"
AGGREGATED_DIR = _OUT / "aggregated"
FIGURES_DIR = AGGREGATED_DIR / "figures"
EFFICIENCY_DIR = _OUT / "efficiency"

# Ensure dirs exist
for d in [RESULTS_DIR, PUBLISHED_DIR, CHECKPOINTS_DIR, ARTIFACTS_DIR,
          AGGREGATED_DIR, FIGURES_DIR, EFFICIENCY_DIR,
          ARTIFACTS_DIR / "tau", ARTIFACTS_DIR / "predictions",
          ARTIFACTS_DIR / "fft", ARTIFACTS_DIR / "aeff", ARTIFACTS_DIR / "branches"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------
INPUT_SIZE = int(os.environ.get("CENN_INPUT_SIZE", "512"))  # L=512 (PatchTST/64 regime); CENN_INPUT_SIZE overrides it for the lookback ablation
HORIZONS = [96, 192, 336, 720]
SEEDS = [1, 42, 123]       # Default 3 seeds
SEEDS_MAIN = [1, 42, 123, 7, 2026]  # 5 seeds for main model
# Extra seeds, fixed before any 5/7-seed run. Seed sets are nested (SEEDS in SEEDS_MAIN in
# SEEDS_EXTENDED) so comparisons stay seed-paired.
SEEDS_EXTENDED = SEEDS_MAIN + [314, 2718]  # 7 seeds
N_WINDOWS = 3               # unused; evaluation uses the fixed splits in SPLITS with stride-1 windows
MAX_STEPS = int(os.environ.get("CENN_MAX_STEPS", "1000"))  # CENN_MAX_STEPS overrides it; use with a
# CENN_EXP_DIR scratch dir.
PRECISION = "bf16-mixed"
SCALER_TYPE = "identity"   # LongHorizon2 data is already train-z-scored; per-window 'standard' divides flat windows by a near-zero std

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "ETTh1":       {"group": "ETTh1",    "n_series": 7,   "freq": "h"},
    "ETTh2":       {"group": "ETTh2",    "n_series": 7,   "freq": "h"},
    "ETTm1":       {"group": "ETTm1",    "n_series": 7,   "freq": "15min"},
    "ETTm2":       {"group": "ETTm2",    "n_series": 7,   "freq": "15min"},
    "Weather":     {"group": "Weather",  "n_series": 21,  "freq": "10min"},
    "Electricity": {"group": "ECL",      "n_series": 321, "freq": "h"},
    "Traffic":     {"group": "TrafficL", "n_series": 862, "freq": "h"},
}

DATASETS_SMALL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
DATASETS_MEDIUM = ["Weather"]
DATASETS_LARGE = ["Electricity", "Traffic"]
DATASETS_ALL = DATASETS_SMALL + DATASETS_MEDIUM + DATASETS_LARGE

# ---------------------------------------------------------------------------
# Standard LTSF train/val/test splits, (val_size, test_size) in time points, from
# datasetsforecast LongHorizon2Info (ETT = 12/4/4 months; Weather/ECL/Traffic = 0.7/0.1/0.2).
# ---------------------------------------------------------------------------
SPLITS = {
    "ETTh1":       (2880, 2880),
    "ETTh2":       (2880, 2880),
    "ETTm1":       (11520, 11520),
    "ETTm2":       (11520, 11520),
    "Weather":     (5270, 10539),
    "Electricity": (2632, 5260),
    "Traffic":     (1756, 3508),
}

# ---------------------------------------------------------------------------
# CeNN variants. Keys are the string after 'CeNN_' in result filenames and must exist in
# runner.VARIANT_SPECS.
# ---------------------------------------------------------------------------
CENN_VARIANTS_MAIN = [
    "S0-StableBase",
    "C1-BoundedTau",
    "C2-MultiScaleEnsemble",   # parallel ensemble
    "CeNN-Full",               # patching arm
    "C1C2-Ensemble",           # C1+C2, patch-free; no-skip ablation
    "C1C2-Skip-K2",            # C1C2-Ensemble + zero-init linear skip at K=2 (min-max-era headline)
]
CENN_VARIANTS_ABLATION = [
    "ABL-SpectralCapOff",
    "ABL-Patch",
    "ABL-GateParam-Unbounded",
    # CrossVar axis: the explicit pre-embedding mixer on top of the dense V->H input_proj
    # (which already mixes variables): pointwise (latent 1x1), varmix (dense O(V^2)), star (O(V)).
    "ABL-CrossVar-Pointwise",
    "ABL-CrossVar-VarMix",
    "ABL-CrossVar-STAR",        # O(V) STAR aggregate-redistribute core (after SOFTS)
    "ABL-ChannelGroups-G4",
    # K x integrator sweep. The K=8 Euler point is C1-BoundedTau; the K-sweep figure keys on
    # (cenn_K, integrator).
    "K4-Euler",
    "K4-Heun",
    "K4-ExpEuler",
    "K2-Euler",
    "K2-Heun",
    # Scale-count sweep around the C2 anchor (n_scales=4 == C2-MultiScaleEnsemble)
    "ABL-Scales-2",
    "ABL-Scales-3",
    "ABL-Scales-5",
]
CENN_VARIANTS_APPENDIX = [
    "CeNN-RawBase",
    "APP-C2Form-DilatedTemplate",
    "APP-Head-MLP",
    "APP-RK4",                 # RK4 accuracy ceiling
    # C2 at K=2: does the K-flatness measured on C1 transfer to the multi-scale model?
    "K2-C2Ensemble",
    # Additivity: does the multi-scale ensemble add accuracy on top of an MLP head?
    "APP-MultiScale-Patch-MLP",
    "APP-MultiScale-MLP",
]
# Anchored protocol: in-model last-value anchor, no pipeline scaler, every variant. Headline
# is AMS-Anc; C1C2-Skip-K2 stays in the lists above as the normalization-ablation row.
CENN_VARIANTS_ANCHORED = [
    "AMS-Anc", "S0-Anc", "C1-Anc", "C2-Anc", "C1C2-Anc", "SkipOnly-Anc", "MLPSkip-Anc",
    "AMS-Anc-K4", "AMS-Anc-K8", "FrozenSkip-TrainTrunk-Anc",
    "AMS-Anc-STAR", "AMS-Anc-Pointwise", "AMS-Anc-VarMix", "AMS-Anc-G4",
]
# Flat union consumed by runner --cenn-all and aggregate.py.
CENN_VARIANTS = CENN_VARIANTS_MAIN + CENN_VARIANTS_ABLATION + CENN_VARIANTS_APPENDIX + CENN_VARIANTS_ANCHORED

# Headline model: 5 seeds, the main-table row and the prediction/branch/tau artifacts.
CENN_MAIN_VARIANT = "AMS-Anc"
# Display name of the headline in every figure and table. Scripts resolve it via
# `model == f"CeNN_{CENN_MAIN_VARIANT}"` (make_tables._disp, make_figures.label_of, aggregate).
CENN_DISPLAY_NAME = "AMS-CeNN"
# Role to variant map. Table/figure scripts resolve ablation rows through this, so a headline
# switch is one edit. All rows use the anchored protocol except minmax_headline, the earlier
# configuration kept as the normalization-ablation row.
ROLES = {
    "main": "AMS-Anc",
    "s0": "S0-Anc", "c1": "C1-Anc", "c2": "C2-Anc", "no_skip": "C1C2-Anc",
    "skip_only": "SkipOnly-Anc", "generic_trunk": "MLPSkip-Anc",
    "k4": "AMS-Anc-K4", "k8": "AMS-Anc-K8",
    "minmax_headline": "C1C2-Skip-K2",
    "cross_channel": {"STAR": "AMS-Anc-STAR", "pointwise": "AMS-Anc-Pointwise",
                      "varmix": "AMS-Anc-VarMix", "G4": "AMS-Anc-G4"},
    "pilot_frozen_skip": "FrozenSkip-TrainTrunk-Anc",
}
# DATASETS_HEADLINE: the low-to-moderate-V datasets (ETT + Weather), used only to scope the
# in-regime figures (MSE boxplots, tau profile). The main table and the CD diagram span all 7.
DATASETS_HEADLINE = DATASETS_SMALL + DATASETS_MEDIUM

# Per-dataset CeNN input scaler for the min-max-era variants. Min-max compresses Weather's
# near-constant channels, so Weather uses identity; identity diverges on ETTh2, so ETT keeps
# min-max. Applied to every CeNN variant on the named dataset (build_cenn).
CENN_DATASET_SCALER = {"Weather": "identity"}

# Integrator/K constants for the sweep drivers:
CENN_K_SWEEP     = [8, 4, 2]
CENN_INTEGRATORS = ["euler", "exp_euler", "heun", "rk4"]
# CeNN scaler_type is per-variant in runner.VARIANT_SPECS ('minmax' for the min-max-era
# variants, 'identity' for the anchored ones); baselines use the global SCALER_TYPE.

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
# Every baseline is re-run at L=512 under the common protocol. Published numbers for
# iTransformer/DLinear/NHITS/TiDE use other lookbacks and are not comparable. xLSTM is the
# NeuralForecast wrapper of the official xlstm package; xLSTMTime's published numbers (L=336)
# appear only in the as-published appendix table. S4D is the official state-spaces/s4 code,
# vendored, with SSM kernel params at wd=0 and fp32. PatchTST is re-run as well; the published
# PatchTST/64 numbers (Nie et al. ICLR'23, Table 3) stay in published/ for the appendix
# reproduction cross-check only (aggregate.load_all_results keeps them out of the main table).
BASELINES_RERUN = ["TCN", "TimesNet", "TSMixer", "TimeMixer",
                   "iTransformer", "DLinear", "NHITS", "TiDE", "xLSTM", "S4D", "PatchTST"]
# Baselines cited from the literature in the main table (none; published/ holds only the
# PatchTST reproduction reference).
BASELINES_PUBLISHED = []
# All baselines in the main comparison.
BASELINES_ALL = BASELINES_RERUN + BASELINES_PUBLISHED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def result_filename(model: str, dataset: str, horizon: int, seed: int) -> str:
    """Filename for a single experiment result."""
    return f"{model}__{dataset}__H{horizon}__seed{seed}.json"


def result_path(model: str, dataset: str, horizon: int, seed: int) -> Path:
    """Full path to a result JSON file."""
    return RESULTS_DIR / result_filename(model, dataset, horizon, seed)


def result_exists(model: str, dataset: str, horizon: int, seed: int) -> bool:
    """True if a parseable result JSON with a non-null 'mse' exists. A truncated or corrupt file
    counts as missing, so the cell is re-run."""
    p = result_path(model, dataset, horizon, seed)
    if not p.is_file():
        return False
    try:
        import json
        d = json.loads(p.read_text())
        return d.get("mse") is not None
    except Exception:
        print(f"[result_exists: corrupt/truncated {p.name} -> will re-run]")
        return False


def batch_size_for_dataset(dataset_name: str) -> int:
    """Adaptive batch size to avoid OOM on large multivariate datasets."""
    n_series = DATASET_INFO.get(dataset_name, {}).get("n_series", 7)
    if n_series > 500:    # Traffic (862)
        return 4
    elif n_series > 200:  # Electricity (321)
        return 8
    else:
        return 32
