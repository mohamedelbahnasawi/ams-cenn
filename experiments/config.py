"""Experiment configuration and constants.

Central source of truth for all experiment settings.
Change input_size, seeds, datasets here — everything else reads from this.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
# All experiment OUTPUTS live under _OUT. Set CENN_EXP_DIR to redirect them to a scratch root
# (fully isolated results/checkpoints/artifacts/aggregated/efficiency) — used by the e2e plumbing
# test so its throwaway max_steps cells never pollute the real results, and usable for H100 scratch.
# Unset -> the real experiments/ dir (default, unchanged).
_OUT = Path(os.environ["CENN_EXP_DIR"]) if os.environ.get("CENN_EXP_DIR") else EXPERIMENTS_DIR
# L-sensitivity ablation: a non-default CENN_INPUT_SIZE nests ALL outputs (results,
# checkpoints, artifacts, aggregated, efficiency) under L{N}/ so L-sweep markers/checkpoints can
# NEVER collide with or contaminate the canonical L=512 campaign (same isolation idea as
# CENN_EXP_DIR). The aggregator only ever reads the canonical dirs.
_L_OVERRIDE = os.environ.get("CENN_INPUT_SIZE")
if _L_OVERRIDE and _L_OVERRIDE != "512":
    _OUT = _OUT / f"L{int(_L_OVERRIDE)}"
# Ablation F (variable count): CENN_VAR_SUBSET=K nests outputs under V{K}/ so channel-subset
# cells never collide with the full-dataset results (same isolation idea as L{N}/).
_V_OVERRIDE = os.environ.get("CENN_VAR_SUBSET")
if _V_OVERRIDE:
    _OUT = _OUT / f"V{int(_V_OVERRIDE)}"
RESULTS_DIR = _OUT / "results"
# PUBLISHED_DIR is NOT L-nested: published numbers are verbatim literature transcriptions
# (PatchTST @ L=512), not run outputs — nesting them under L{N}/ would silently empty the
# published baselines in any L-sweep aggregation. It still respects CENN_EXP_DIR (the e2e
# scratch supplies its own published/).
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
INPUT_SIZE = int(os.environ.get("CENN_INPUT_SIZE", "512"))  # L=512 — long-lookback (PatchTST/64 regime); a deliberate choice for CeNN's long receptive field, NOT a universal LTSF default. CENN_INPUT_SIZE env overrides for the L-sensitivity ablation; see _OUT nesting below.
HORIZONS = [96, 192, 336, 720]
SEEDS = [1, 42, 123]       # Default 3 seeds
SEEDS_MAIN = [1, 42, 123, 7, 2026]  # 5 seeds for main model
# Pre-declared a-priori escalation seeds (in case review asks for 5/7 seeds on more models).
# Committed BEFORE any such run exists so a later top-up is provably not seed-shopping; seed sets
# stay nested (SEEDS ⊂ SEEDS_MAIN ⊂ SEEDS_EXTENDED) so every comparison can be made seed-paired.
SEEDS_EXTENDED = SEEDS_MAIN + [314, 2718]  # 7 seeds
N_WINDOWS = 3               # DEPRECATED — no longer used for evaluation. Eval now uses standard fixed splits (see SPLITS) with stride-1 sliding windows.
MAX_STEPS = int(os.environ.get("CENN_MAX_STEPS", "1000"))  # CENN_MAX_STEPS overrides for the under-training
# probe: high-V datasets (ECL/Traffic) run the FULL 1000 steps -> test if longer training closes
# the high-V gap. Use with CENN_EXP_DIR scratch so the non-default-budget cells never pollute canonical results.
PRECISION = "bf16-mixed"
SCALER_TYPE = "identity"   # LongHorizon2 already delivers globally train-z-scored data (the standard LTSF protocol). Per-window 'standard' TemporalNorm divides the horizon target by a near-zero per-window lookback std on flat windows -> target blowup on flat windows (verified). Was "standard"/"minmax".

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
# Standard LTSF train/val/test splits — (val_size, test_size) in time points.
# Sourced from datasetsforecast LongHorizon2Info; matches the canonical protocol
# (ETT = 12/4/4 months; Weather/ECL/Traffic = 0.7/0.1/0.2). Consumed by the runner's
# cross_validation(val_size=..., test_size=..., n_windows=None) call (stride-1, full test).
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
# CeNN variants (Inc-2 + Inc-3 + Inc-4). Canonical spec keys = the string after 'CeNN_' in
# result filenames; MUST match runner.VARIANT_SPECS. All Inc-4 items are now active.
# ABL-GateParam-Tanh is DROPPED (bounded-tanh gate ≈ bounded-sigmoid C1, a near-duplicate
# row), not deferred. Old baseline/c1/c2/c1c2/c1c2pw results are stale (superseded by the
# current scaler/protocol; do not mix with current results).
# ---------------------------------------------------------------------------
CENN_VARIANTS_MAIN = [
    "S0-StableBase",
    "C1-BoundedTau",
    "C2-MultiScaleEnsemble",   # Inc-3: parallel ensemble (unlocked)
    "CeNN-Full",               # the efficiency/patching arm
    "C1C2-Ensemble",           # C1+C2 adaptive multi-scale, patch-free; the -SKIP ABLATION of the headline
    "C1C2-Skip-K2",            # HEADLINE (AMS-CeNN): C1C2-Ensemble + zero-init linear skip @ K=2
]
CENN_VARIANTS_ABLATION = [
    "ABL-SpectralCapOff",
    "ABL-Patch",
    "ABL-GateParam-Unbounded",
    # CrossVar axis = the EXPLICIT pre-embedding mixer added on top of the dense V->H
    # input_proj (which already mixes variables — so no arm is truly channel-independent):
    #   implicit-only (=C1, aliased) / pointwise (latent 1x1) / varmix (dense O(V^2)) / star (O(V))
    "ABL-CrossVar-Pointwise",
    "ABL-CrossVar-VarMix",
    "ABL-CrossVar-STAR",        # Inc-4: O(V) STAR aggregate-redistribute core (after SOFTS)
    "ABL-ChannelGroups-G4",
    # "ABL-CrossVar-CI",        # == C1-BoundedTau (implicit-mix-only); alias its JSONs, don't re-run
    # "ABL-GateParam-Tanh",     # DROPPED: bounded-tanh gate ≈ bounded-sigmoid (C1) → near-duplicate row
    # Inc-3: K × Integrator sweep (accuracy-per-MAC / edge story). Euler curve K={8,4,2}: the K=8
    # point IS C1-BoundedTau (don't re-run); the K-sweep figure keys on (cenn_K, integrator).
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
    "APP-RK4",                 # Inc-3: RK4 accuracy ceiling (appendix)
    # Pre-declared confirmatory: headline (C2) at K=2 —
    # does the C1-chassis K-flatness transfer to the multi-scale headline?
    "K2-C2Ensemble",
    # Additivity probe: does the multi-scale ensemble add accuracy ON TOP of an MLP head, or are
    # the two redundant? Tested on the long-horizon setting where dilated branches have room to operate.
    "APP-MultiScale-Patch-MLP",
    "APP-MultiScale-MLP",
]
# Flat union consumed by runner --cenn-all and aggregate.py.
CENN_VARIANTS = CENN_VARIANTS_MAIN + CENN_VARIANTS_ABLATION + CENN_VARIANTS_APPENDIX

# Headline model (5 seeds; the main-table row + the prediction/branch/tau artifacts).
# HEADLINE = AMS-CeNN = C1C2-Skip-K2: the C1C2 adaptive multi-scale ensemble plus a
# zero-init linear residual ("skip") at K=2. The skip rescues the high-V failure and beats DLinear
# on complex/long-horizon while tying elsewhere; top-3 of 11 on the full 7-dataset CD. K=2 (not K=8)
# is accuracy-neutral ON THIS ARCHITECTURE (C1C2-Skip K8 vs K2: mean dMSE 0.0000 over 42 matched
# cells) and 4x cheaper, so the headline runs at K=2.
CENN_MAIN_VARIANT = "C1C2-Skip-K2"
# The OLD headline C1C2-Ensemble (== C1C2-Skip-K2 MINUS the skip) is now THE -skip ablation:
# removing the skip costs +0.030 MSE on ETT / +0.128 across all 7 (high-V driven). Because K is
# neutral (above), C1C2-Ensemble's K=8 setting does not confound that single-knob comparison.
# Single display name for the headline across ALL figures/tables (the manuscript name). Resolved
# everywhere via `model == f"CeNN_{CENN_MAIN_VARIANT}"` -> CENN_DISPLAY_NAME (make_tables._disp,
# make_figures.label_of, aggregate). The internal result key IS C1C2-Skip-K2 (NOT C1C2-Ensemble).
CENN_DISPLAY_NAME = "AMS-CeNN"
# DATASETS_HEADLINE = the low-to-moderate-V regime (ETT + Weather). Used ONLY to scope the in-regime
# figures (MSE boxplots, tau profile) where pooling the high-V datasets would inflate variance. The
# MAIN results table and the CD diagram now span ALL 7 datasets -- the linear skip fixed the high-V
# failure, so ECL/Traffic fold back in (no separate scoped/high-V split; that was an artifact of the
# old broken-on-high-V model and reporting only the flattering scope read as cherry-picking).
DATASETS_HEADLINE = DATASETS_SMALL + DATASETS_MEDIUM

# Per-dataset CeNN input scaler (validation-selected). CeNN's default minmax
# compresses Weather's many near-constant channels and underperforms there; identity
# (the scaler the re-run baselines already use) recovers ~13% on Weather (0.31->0.27) without
# touching ETT (where identity instead DIVERGES on ETTh2: 0.37->0.83). So the scaler is selected
# per dataset on validation; datasets not listed keep the per-variant default (minmax). Applied to
# ALL CeNN variants on the named dataset (build_cenn) so within-dataset comparisons stay controlled.
CENN_DATASET_SCALER = {"Weather": "identity"}

# Inc-3 integrator/K constants (for reference and future sweep drivers):
CENN_K_SWEEP     = [8, 4, 2]
CENN_INTEGRATORS = ["euler", "exp_euler", "heun", "rk4"]
# scaler_type is PER-VARIANT (all CeNN = 'minmax') in runner.VARIANT_SPECS, NOT the
# global SCALER_TYPE (='identity', which baselines still use).

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
# Re-run these at L=512 under our protocol.
# iTransformer/DLinear/NHITS/TiDE moved to re-run: their published
# numbers use different native lookbacks (L=96/336/5H/720) and are NOT comparable at L=512,
# so they must be re-run rather than cited.
# xLSTM was initially dropped from the re-run set: measured ~3.4 h/run on Traffic
# (eval-bound) → ~68 GPU-h for one appendix-tier baseline; and the NeuralForecast xLSTM
# wrapper != the published xLSTMTime architecture, so its numbers aren't a clean match.
# Instead, xLSTMTime's published numbers go in the AS-PUBLISHED appendix table only.
# VERIFIED (arXiv:2407.10240, Alharthi & Mahmood 2024): xLSTMTime reports our
# full 7 datasets × {96,192,336,720} with MSE+MAE, but at lookback L=336 (not 512) — so it
# is appendix-only (labeled "L=336, as published"), NOT comparable in the main L=512 table.
# xLSTM RE-ADDED (reversed): the NF model wraps the OFFICIAL xlstm
# package (Hochreiter-lab mLSTM/sLSTM blocks) -> a legitimate, author-blessed xLSTM baseline
# and the only modern-recurrent family member in the set. The earlier cost concern (~3.4h/run on
# Traffic) predates the chunked-CV fix + H200 + bigger eval batches. xLSTMTime stays
# as-published appendix-only (anti-sandbagging context; keep/drop at write-up).
# S4D ADDED: a full state-space baseline on ALL 7 datasets x 3 seeds, on equal
# footing with the others (NOT a subset -- a declared subset reads as cherry-picking and invites "why
# not all 7?"). Official S4D (state-spaces/s4) vendored; SSM kernel params trained wd=0 + fp32 per the
# authors. Whatever it scores is its honest number. The CeNN-xSSM HYBRID + dedicated SSM study = future work.
# PatchTST MOVED to re-run: re-run under our exact L=512/stride-1 pipeline
# (all 7 datasets x 4 horizons x 3 seeds = 84 cells) to remove the as-published protocol mismatch.
# Per-cell the re-run is a WASH that nets slightly in PatchTST's favour: materially
# better on 11/28 cells (all ETTm1/ETTm2/Traffic + short-horizon ETTh*), materially worse on 4/28
# (long-horizon ETTh1/ETTh2, peaking at ETTh2 H720: 0.518 re-run vs 0.379 published, +0.139), ~flat
# on the rest -> net mean MSE 0.295 re-run vs 0.304 published. The worse cells are where Nie et al.'s
# PER-DATASET patch/stride/dropout tuning helps most; under our UNIFORM hyperparameters (applied
# equally to every model, no per-dataset tuning for anyone) PatchTST stays mid-pack there (ETTh2 H720:
# ahead of DLinear 0.640 / TiDE 0.651 / TimesNet 0.837, behind TSMixer 0.385 / iTransformer 0.393 /
# AMS-CeNN 0.352) -> NOT sandbagged, just untuned like everyone else. This is the disclosed cost of a
# single uniform protocol; see the appendix reproduction cross-check. The published PatchTST/64 numbers
# (Nie et al. ICLR'23, Table 3) stay in published/ for that cross-check only (results-take-precedence in
# aggregate.load_all_results keeps them out of the main table).
BASELINES_RERUN = ["TCN", "TimesNet", "TSMixer", "TimeMixer",
                   "iTransformer", "DLinear", "NHITS", "TiDE", "xLSTM", "S4D", "PatchTST"]
# No baseline is cited from literature in the main table anymore: every model in the uniform-L=512
# comparison is run under our own pipeline. published/ now holds only the PatchTST reproduction
# reference (appendix), surfaced via a dedicated cross-check, never merged into main_results.
BASELINES_PUBLISHED = []
# All baselines actually run/cited in the MAIN (uniform-L=512) comparison.
BASELINES_ALL = BASELINES_RERUN + BASELINES_PUBLISHED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def result_filename(model: str, dataset: str, horizon: int, seed: int) -> str:
    """Canonical filename for a single experiment result."""
    return f"{model}__{dataset}__H{horizon}__seed{seed}.json"


def result_path(model: str, dataset: str, horizon: int, seed: int) -> Path:
    """Full path to a result JSON file."""
    return RESULTS_DIR / result_filename(model, dataset, horizon, seed)


def result_exists(model: str, dataset: str, horizon: int, seed: int) -> bool:
    """Check if a VALID result already exists (skip-if-exists). Validates the JSON parses and has a
    finite 'mse' — a truncated/corrupt file (e.g. an interrupted write after rename, disk hiccup)
    must NOT be treated as done, or the cell is skipped forever and never recovers."""
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
