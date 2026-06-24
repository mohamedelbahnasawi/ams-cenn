#!/usr/bin/env python
"""Drive the paper figures from REAL results.

Joins the seed-mean MSE (from experiments/results/, via aggregate.load_all_results) to the
efficiency profile (experiments/efficiency/*.json, from profile_efficiency.py) and feeds the
real rows to the plotting functions — replacing the synthetic `_demo()` path that every figure
in aggregated/figures/ was previously built from.

Produces (anchored on one dataset×horizon for a fair same-problem-size comparison; default ETTh1):
  - fig_pareto_acc_vs_macs   (accuracy vs MACs, Pareto frontier)
  - fig_efficiency_panels    (MACs / params / peak-mem vs MSE)
  - fig_k_integrator         (K × integrator accuracy-per-MAC; C1-BoundedTau = the K8-Euler point)

Run:
  .venv/Scripts/python.exe experiments/analysis/make_figures.py --dataset ETTh1 --horizon 96
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.config import (EFFICIENCY_DIR, FIGURES_DIR, ARTIFACTS_DIR, HORIZONS,   # noqa: E402
                                DATASETS_ALL, DATASETS_SMALL, DATASETS_HEADLINE,
                                CENN_MAIN_VARIANT, CENN_DISPLAY_NAME, CENN_VARIANTS,
                                CENN_VARIANTS_ABLATION, BASELINES_ALL, RESULTS_DIR, EXPERIMENTS_DIR)
from experiments.aggregate import load_all_results                      # noqa: E402
from experiments.runner import VARIANT_SPECS, _VARIANT_K                 # noqa: E402
from experiments.analysis import plotting                               # noqa: E402

EFF_DIR = EFFICIENCY_DIR
FIG_DIR = FIGURES_DIR

# Model family (drives color/marker in the figures). CeNN handled separately.
_FAMILY = {
    "DLinear": "linear",
    "NHITS": "mlp", "TiDE": "mlp", "TSMixer": "mlp", "TimeMixer": "mlp",
    "PatchTST": "transformer", "iTransformer": "transformer",
    "TimesNet": "conv", "TCN": "conv",
}
# CeNN variants worth emphasizing (stars) in the Pareto/panels.
_EMPH = {"CeNN_CeNN-Full", "CeNN_C1-BoundedTau"}


def family_of(model):
    return "CeNN" if model.startswith("CeNN_") else _FAMILY.get(model, "linear")


# Plain-English display names for CeNN variants — no internal codes in any figure.
# Keyed by the bare variant (CeNN_ stripped). Anything not here falls back to the bare name.
_VARIANT_LABEL = {
    "C1C2-Skip-K2": CENN_DISPLAY_NAME,            # the headline (also handled in label_of)
    "C1C2-Ensemble": f"{CENN_DISPLAY_NAME} (no skip)",
    "S0-StableBase": "S0: stable base",
    "C1-BoundedTau": "C1: bounded-τ gate",
    "C2-MultiScaleEnsemble": "C2: multi-scale",
    "CeNN-Full": "CeNN-Full",
    "CeNN-RawBase": "raw base (no gate/cap)",
    # leave-one-out architectural ablations
    "ABL-SpectralCapOff": "− spectral cap",
    "ABL-GateParam-Unbounded": "− bounded gate",
    "ABL-Patch": "+ patch embedding",
    "ABL-ChannelGroups-G4": "grouped channels (G=4)",
    "ABL-CrossVar-Pointwise": "cross-var: pointwise mix",
    "ABL-CrossVar-VarMix": "cross-var: dense mix",
    "ABL-CrossVar-STAR": "cross-var: STAR core",
    "ABL-Scales-2": "multi-scale: 2 scales",
    "ABL-Scales-3": "multi-scale: 3 scales",
    "ABL-Scales-5": "multi-scale: 5 scales",
    # K / integrator sweep (shown in the K-integrator figure, not the ablation bars)
    "K2-Euler": "K=2 (Euler)", "K2-Heun": "K=2 (Heun)",
    "K4-Euler": "K=4 (Euler)", "K4-Heun": "K=4 (Heun)", "K4-ExpEuler": "K=4 (Exp-Euler)",
    "C1C2-Skip-K4": "K=4 (Euler)", "C1C2-Skip": "K=8 (Euler)",
    "C1C2-Skip-ExpEuler-K2": "K=2 (Exp-Euler)",
}


def label_of(model):
    if model == f"CeNN_{CENN_MAIN_VARIANT}":
        return CENN_DISPLAY_NAME
    if model.startswith("CeNN_"):
        bare = model.removeprefix("CeNN_")
        return _VARIANT_LABEL.get(bare, bare)
    return model


def is_headline(model):
    return model == f"CeNN_{CENN_MAIN_VARIANT}"


def seed_mean_mse():
    """(model, dataset, horizon) -> seed-mean MSE, from the (DATASETS_ALL-filtered) results."""
    df = load_all_results()
    if df.empty or "mse" not in df.columns:
        return {}
    g = df.groupby(["model", "dataset", "horizon"])["mse"].mean()
    return {(m, d, int(h)): float(v) for (m, d, h), v in g.items()}


def scoped_mean_mse(datasets):
    """model -> mean MSE over (dataset in `datasets`) x horizons (balanced cell-mean), seed-averaged.
    This is the headline-regime accuracy used on the Pareto/efficiency y-axis."""
    df = load_all_results()
    if df.empty or "mse" not in df.columns:
        return {}
    df = df[df["dataset"].isin(datasets)]
    g = df.groupby(["model", "dataset", "horizon"])["mse"].mean().reset_index()
    return g.groupby("model")["mse"].mean().to_dict()


def seed_std_at(dataset, horizon):
    """(model) -> seed std of MSE at one (dataset,horizon) anchor (for k-sweep error bars)."""
    df = load_all_results()
    if df.empty:
        return {}
    d = df[(df["dataset"] == dataset) & (df["horizon"] == horizon)]
    return {m: float(s) for m, s in d.groupby("model")["mse"].std().items()}


def load_efficiency():
    """(model, dataset, horizon) -> efficiency dict."""
    eff = {}
    for f in sorted(glob.glob(str(EFF_DIR / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception as e:
            print(f"[skip unreadable efficiency {Path(f).name}: {type(e).__name__}]")
            continue
        eff[(d["model"], d["dataset"], int(d["horizon"]))] = d
    return eff


def build_rows(dataset, horizon, mse_scope, eff):
    """Join efficiency (at the dataset×horizon anchor) with the headline-regime mean MSE
    (`mse_scope`: model->mean MSE over ETT+Weather) for every model that has both. The y-axis is
    therefore the scoped mean accuracy, not a single anchor cell — labelled as such in the figure."""
    rows = []
    for (model, ds, h), e in eff.items():
        if ds != dataset or h != horizon:
            continue
        m = mse_scope.get(model)
        if m is None:
            print(f"[no scoped MSE for {model} — skipping in figure]")
            continue
        rows.append({
            "model": model, "label": label_of(model), "family": family_of(model),
            "macs": float(e["macs_fwd"]), "params": float(e["params"]),
            "peak_mem": float(e.get("peak_infer_mb") or 0.0), "mse": m,
            # Only the HEADLINE CeNN variant is emphasized+labelled; other CeNN variants render as
            # faded family-cloud stars (no text) so the headline efficiency story isn't drowned in
            # 25 overlapping ablation labels. Baselines keep their labels.
            "headline": is_headline(model),
            "emphasize": family_of(model) == "CeNN",
        })
    return rows


def build_ksweep(dataset, horizon, mse, eff, std=None):
    """K × integrator points. C1-BoundedTau IS the K8-Euler point (keyed on cenn_K/integrator,
    not the variant name). `std` (model->seed std at the anchor) adds error bars."""
    std = std or {}
    # variant -> (integrator, K).  C1-BoundedTau supplies the K=8 Euler anchor.
    kvars = {
        "C1-BoundedTau": ("euler", 8),
        "K4-Euler": ("euler", 4), "K2-Euler": ("euler", 2),
        "K4-Heun": ("heun", 4), "K2-Heun": ("heun", 2),
        "K4-ExpEuler": ("exp_euler", 4),
    }
    pts = []
    for v, (integ, k) in kvars.items():
        model = f"CeNN_{v}"
        e = eff.get((model, dataset, horizon))
        m = mse.get((model, dataset, horizon))
        if e is None or m is None:
            print(f"[k-sweep: missing {'eff' if e is None else 'mse'} for {v} @ {dataset} H{horizon}]")
            continue
        # sanity: the variant's declared K/integrator matches our table
        decl_k = _VARIANT_K.get(v, 8)
        decl_integ = VARIANT_SPECS[v]["integrator"]
        if decl_k != k or decl_integ != integ:
            print(f"[k-sweep WARN {v}: table says (K={k},{integ}) but VARIANT_SPECS says "
                  f"(K={decl_k},{decl_integ})]")
        pts.append({"integrator": integ, "K": k, "macs": float(e["macs_fwd"]), "mse": m,
                    "std": float(std.get(model, 0.0) or 0.0)})
    return pts


def horizon_curves(mse, datasets, models):
    """fig_horizon_curves (RQ2: error vs horizon), averaged over the headline regime `datasets`
    (ETT+Weather). series = model -> {mse:[mean over datasets per HORIZON], family, headline}.
    A model is plotted only if it has every (dataset,horizon) cell in the regime."""
    series = {}
    for m in models:
        per_h = []
        ok = True
        for h in HORIZONS:
            cells = [mse.get((m, d, h)) for d in datasets]
            if any(c is None for c in cells):
                ok = False
                break
            per_h.append(float(sum(cells) / len(cells)))
        if ok:
            series[label_of(m)] = {"mse": per_h, "family": family_of(m), "headline": is_headline(m)}
    if not series:
        print(f"[horizon-curves skipped: a model lacks a cell over {datasets} x {HORIZONS}]")
        return None
    return plotting.fig_horizon_curves(series, HORIZONS, FIG_DIR,
                                       scope_note="mean over ETT + Weather")


def tau_figure(dataset, horizon):
    """fig_tau_profile (C1 interpretability). The saved artifact is tau = 1 - alpha = the DRIVE weight
    (model.py L247: self.last_tau = 1 - alpha), shape (latent_channel, time). The RETENTION we plot is
    alpha = 1 - tau (~0.90: the gate retains ~90% of the cell state each step). NB: do NOT plot the raw
    saved tau as 'retention' -- that was a bug that mislabeled the ~0.10 drive weight as retention.
    Empirically alpha is ~constant over the lookback (time-std ~1e-4) but varies by latent channel
    (channel-std ~1e-2): a CHANNEL-specific retention rate, not a per-timestep one."""
    import numpy as np
    tau_dir = ARTIFACTS_DIR / "tau"
    for v in [CENN_MAIN_VARIANT, "C1-BoundedTau", "CeNN-Full"]:   # adaptive-tau variants
        hits = sorted(glob.glob(str(tau_dir / f"CeNN_{v}__{dataset}__H{horizon}__seed*.npz")))
        if hits:
            d = np.load(hits[0])
            cells = np.stack([d[c] for c in d.files])     # (n_cells, C, T)
            per_chan = cells.mean(axis=2)                 # (n_cells, C): time-averaged TAU (drive weight)
            mean_tau = per_chan.mean(axis=0)              # (C,) mean tau over cells/layers
            std_c = per_chan.std(axis=0)                  # (C,) spread (std of 1-tau == std of tau)
            return plotting.fig_tau_profile(1.0 - mean_tau, std_c, FIG_DIR,   # RETENTION alpha = 1 - tau
                                            model_label=label_of(f"CeNN_{v}"),
                                            note=f"{dataset}, H{horizon}")
    print(f"[tau profile skipped: no tau artifact for an adaptive-tau variant @ {dataset} H{horizon} "
          f"(run a {CENN_MAIN_VARIANT}/C1 cell with --save-artifacts)]")
    return None


def ablation_bars(df, datasets=None):
    """fig_ablation_bars: leave-one-out Δ MSE for each ablation variant vs the HEADLINE
    (AMS-CeNN = C1C2-Skip-K2), computed on MATCHED cells only — i.e. for each variant, restrict
    to the (dataset,horizon,seed) cells present in BOTH that variant and the headline, then take
    the mean difference. (The previous version subtracted each variant's mean over its OWN, larger
    cell-set, which is invalid: variants cover 72-124 different cells.) Positive Δ ⇒ removing that
    component HURTS ⇒ the component contributes. Optionally restrict to a dataset regime."""
    ref = f"CeNN_{CENN_MAIN_VARIANT}"
    d = df if datasets is None else df[df["dataset"].isin(datasets)]
    # cell-key -> mse, per model
    seedcol = "seed" if "seed" in d.columns else None
    keycols = ["dataset", "horizon"] + ([seedcol] if seedcol else [])
    by_model = {}
    for m, sub in d.groupby("model"):
        by_model[m] = {tuple(r): v for r, v in
                       zip(sub[keycols].itertuples(index=False, name=None), sub["mse"])}
    if ref not in by_model:
        print("[ablation-bars skipped: headline has no results]"); return None
    refcells = by_model[ref]
    # Include the -skip ablation (C1C2-Ensemble = the headline MINUS the zero-init linear skip) —
    # the single most important component bar for the AMS-CeNN headline — alongside the classic
    # ablations. (It lives in CENN_VARIANTS_MAIN, not _ABLATION, so it must be added explicitly.)
    SKIP_ABL = "C1C2-Ensemble"
    # Two clearly-separated questions in ONE figure (avoids the −remove / +add confusion):
    #   REMOVED  = take a real component OUT of AMS-CeNN -> Δ is its contribution.
    #   ALT      = swap in a design choice NOT used by AMS-CeNN -> Δ is "this alternative is worse by".
    # The K/integrator sweep is excluded (it has its own figure, fig_k_integrator).
    REMOVED = {"C1C2-Ensemble", "ABL-GateParam-Unbounded", "ABL-SpectralCapOff"}
    arch = [v for v in CENN_VARIANTS_ABLATION if not v.startswith(("K2-", "K4-", "K8-"))]
    items = []
    for v in arch + [SKIP_ABL]:
        vm = by_model.get(f"CeNN_{v}")
        if not vm:
            continue
        shared = set(vm) & set(refcells)
        if len(shared) < 8:                      # too few matched cells -> not comparable
            print(f"[ablation-bars: {v} has only {len(shared)} matched cells — skipped]")
            continue
        delta = sum(vm[k] - refcells[k] for k in shared) / len(shared)
        label = "− linear skip" if v == SKIP_ABL else label_of(f"CeNN_{v}")
        group = "Removed component" if v in REMOVED else "Alternative design"
        items.append((label, delta, v == SKIP_ABL, group))   # (label, Δ, highlight, group)
    if len(items) < 2:
        print("[ablation-bars skipped: <2 ablation variants with matched cells]"); return None
    return plotting.fig_ablation_bars(items, FIG_DIR)


def mse_boxplots(df, models, datasets=None):
    """fig_mse_boxplots: per-cell MSE distribution (seeds×dataset×horizon) per method, scoped to
    the headline regime `datasets` (ETT+Weather) so AMS-CeNN's spread reflects in-regime variance,
    not the ECL/Traffic datasets, which are excluded from the boxplots but DO appear in the
    consolidated all-7 main table (there is no separate high-V table anymore)."""
    d = df if datasets is None else df[df["dataset"].isin(datasets)]
    dist = {}
    for m in models:
        vals = d[d["model"] == m]["mse"].to_numpy()
        if len(vals) >= 3:
            dist[label_of(m)] = vals
    if len(dist) < 2:
        print("[mse-boxplots skipped: <2 methods with >=3 cells]"); return None
    note = "all 7 datasets" if datasets is None or len(datasets) >= 7 else "ETT + Weather"
    return plotting.fig_mse_boxplots(dist, FIG_DIR, scope_note=note)


def rank_heatmap(mse, models, datasets=None):
    """fig_rank_heatmap (§1 no-champion motivation): per-(model,dataset) integer rank (1=best of k),
    models = headline + all baselines, mean MSE over horizons then ranked within each dataset. Rows
    ordered by mean rank; AMS-CeNN boxed. Shows the no-champion pattern (different winner per dataset)."""
    import statistics
    ds = list(DATASETS_ALL if datasets is None else datasets)
    mean_md = {}
    for m in models:
        for d in ds:
            vals = [mse.get((m, d, h)) for h in HORIZONS]
            mean_md[(m, d)] = statistics.mean(vals) if all(v is not None for v in vals) else None
    rankmat = {}
    for d in ds:
        col = sorted([m for m in models if mean_md[(m, d)] is not None], key=lambda m: mean_md[(m, d)])
        for i, m in enumerate(col, 1):
            rankmat.setdefault(m, {})[d] = i
    present = [m for m in models if m in rankmat and len(rankmat[m]) == len(ds)]
    if len(present) < 2:
        print("[rank-heatmap skipped: <2 models with all datasets]"); return None
    order = sorted(present, key=lambda m: statistics.mean(rankmat[m][d] for d in ds))
    mat = [[rankmat[m][d] for d in ds] for m in order]
    cols = [d.replace("Electricity", "ECL") for d in ds]
    return plotting.fig_rank_heatmap([label_of(m) for m in order], cols, mat, FIG_DIR, k=len(present))


def robustness_fig(mse, models, datasets=None):
    """fig_robustness (pillar 1 = robust): each model's WORST-dataset relative error — % above the
    best-of-field per cell, mean over horizons, then the max over datasets. AMS-CeNN is the only model
    within ~7% of the best on every dataset; the no-champion payoff ('never far behind')."""
    import statistics
    ds = list(DATASETS_ALL if datasets is None else datasets)
    items = []
    for m in models:
        rels = []
        for d in ds:
            rs = []
            for h in HORIZONS:
                cells = [mse.get((x, d, h)) for x in models]
                if any(c is None for c in cells):
                    continue
                rs.append(mse[(m, d, h)] / min(c for c in cells if c is not None))
            if rs:
                rels.append(statistics.mean(rs))
        if len(rels) == len(ds):
            items.append((label_of(m), (max(rels) - 1) * 100))
    if len(items) < 2:
        print("[robustness skipped: <2 models with all datasets]"); return None
    return plotting.fig_robustness(items, FIG_DIR)


def lsweep_fig(datasets=("ETTh1", "ETTh2", "Weather"), horizon=96):
    """fig_lsweep: lookback sensitivity — headline vs −skip, mean MSE over `datasets` at one horizon,
    across L in {96,192,336,512}. Monotone 'more history -> lower error' = the multi-scale lookback
    story; the headline sits below the −skip arm at every L."""
    import statistics
    Ls = [96, 192, 336, 512]

    def m(variant, L):
        root = str(RESULTS_DIR) if L == 512 else str(EXPERIMENTS_DIR / f"L{L}" / "results")
        vs = [json.load(open(f))["mse"] for d in datasets
              for f in glob.glob(f"{root}/CeNN_{variant}__{d}__H{horizon}__*.json")]
        return statistics.mean(vs) if vs else None
    ser = {CENN_DISPLAY_NAME: [m("C1C2-Skip-K2", L) for L in Ls],
           f"{CENN_DISPLAY_NAME} (no skip)": [m("C1C2-Ensemble", L) for L in Ls]}
    if any(v is None for vv in ser.values() for v in vv):
        print("[lsweep skipped: missing L cells]"); return None
    return plotting.fig_line_sensitivity(
        Ls, ser, FIG_DIR, "fig_lsweep", r"Input lookback $L$", r"MSE ($\downarrow$)",
        f"Lookback sensitivity (mean over {'/'.join(datasets)}, H={horizon})",
        xlog2=True, note="more history $\\rightarrow$ lower error (monotone)")


def horizon_grid(mse, datasets, models):
    """fig_horizon_curves as per-dataset small-multiples (own y-scale per panel; replaces the 13-line
    single-axes spaghetti). `models` = AMS-CeNN + a few best + worst; the plotter highlights AMS-CeNN
    and clips an off-scale outlier (S4D) so the competitive pack stays readable."""
    panels = {}
    for d in datasets:
        key = d.replace("Electricity", "ECL")
        panels[key] = {}
        for m in models:
            vals = [mse.get((m, d, h)) for h in HORIZONS]
            if all(v is not None for v in vals):
                panels[key][label_of(m)] = vals
    if not any(panels.values()):
        print("[horizon-grid skipped: no complete model rows]"); return None
    return plotting.fig_horizon_grid(panels, HORIZONS, FIG_DIR, order=[label_of(m) for m in models])


def rank_boxplots(mse, models, datasets=None):
    """fig_rank_boxplots: each model's distribution of per-block RANKS (1 = best MSE) over the
    complete (dataset,horizon) blocks. Scale-free (unlike raw MSE, which pools datasets of very
    different magnitudes), so 'consistency' is legible: a uniformly strong model has low, tight
    ranks; an erratic one has high, wide ranks. Directly mirrors the Friedman/CD analysis."""
    ds = DATASETS_ALL if datasets is None else datasets
    ranks = {m: [] for m in models}
    n_blocks = 0
    for d in ds:
        for h in HORIZONS:
            cells = {m: mse.get((m, d, h)) for m in models}
            if any(v is None for v in cells.values()):
                continue                                   # only fully-populated blocks
            n_blocks += 1
            for i, m in enumerate(sorted(models, key=lambda mm: cells[mm]), 1):
                ranks[m].append(i)
    ranks = {label_of(m): v for m, v in ranks.items() if v}
    if len(ranks) < 2:
        print("[rank-boxplots skipped: <2 models with complete blocks]"); return None
    return plotting.fig_rank_boxplots(ranks, FIG_DIR, n_blocks=n_blocks, k=len(models))


def forecast_grid(datasets, horizon, model=None):
    """fig_forecast_grid: forecast-vs-truth (+ seed-ensemble band) per dataset for the headline CeNN,
    from the --save-artifacts prediction .npz. Picks the most-dynamic full-horizon window per dataset."""
    import numpy as np
    import pandas as pd
    model = model or f"CeNN_{CENN_MAIN_VARIANT}"
    pred_dir = ARTIFACTS_DIR / "predictions"
    panels = []
    for ds in datasets:
        files = sorted(glob.glob(str(pred_dir / f"{model}__{ds}__H{horizon}__seed*.npz")))
        if not files:
            continue
        d0 = np.load(files[0], allow_pickle=True)
        if not all(k in d0 for k in ("unique_id", "ds", "cutoff", "y", "pred")):
            continue
        b = pd.DataFrame({"uid": d0["unique_id"], "ds": d0["ds"], "cutoff": d0["cutoff"],
                          "y": d0["y"].astype("float64"), "pred": d0["pred"].astype("float64")})
        sizes = b.groupby(["uid", "cutoff"]).size()
        full = sizes[sizes == horizon].index
        if len(full) == 0:
            continue
        key = b.set_index(["uid", "cutoff"]).index
        bf = b[key.isin(full)].copy()
        bf["se"] = (bf["y"] - bf["pred"]) ** 2
        werr = bf.groupby(["uid", "cutoff"])["se"].mean()  # per-window forecast MSE
        # REPRESENTATIVE window = the one whose error is closest to the MEDIAN (not best, not worst,
        # not the most-dynamic) -> an honest typical example.
        uid, cutoff = (werr - werr.median()).abs().idxmin()
        sel = b[(b.uid == uid) & (b.cutoff == cutoff)].sort_values("ds")
        y_true, dsv = sel["y"].to_numpy(), sel["ds"].to_numpy()
        preds = [sel["pred"].to_numpy()]
        for f in files[1:]:                               # seed ensemble (align by ds)
            d = np.load(f, allow_pickle=True)
            s = pd.DataFrame({"uid": d["unique_id"], "ds": d["ds"], "cutoff": d["cutoff"],
                              "pred": d["pred"].astype("float64")})
            s = s[(s.uid == uid) & (s.cutoff == cutoff)].sort_values("ds")
            if len(s) == len(y_true) and np.array_equal(s["ds"].to_numpy(), dsv):
                preds.append(s["pred"].to_numpy())
        P = np.vstack(preds); t = np.arange(1, len(y_true) + 1)
        panels.append({"dataset": ds, "title": f"{ds}  (H={horizon}, median-error window)",
                       "t": t, "y_true": y_true,
                       "y_pred": P.mean(0), "lo": P.min(0), "hi": P.max(0)})
    if not panels:
        print("[forecast-grid skipped: no CeNN-Full prediction artifacts (run with --save-artifacts)]")
        return None
    return plotting.fig_forecast_grid(panels, FIG_DIR)


def forecast_panels(specs, model=None):
    """fig_forecast_grid for an explicit list of (dataset, horizon) pairs -> one REPRESENTATIVE
    (median-error) window per panel. Used for the long-term qualitative figure: 2 datasets x 2 horizons
    (short + extreme), which shows AMS-CeNN holding the periodic structure as the horizon grows, on the
    datasets it tracks (per-dataset accuracy incl. Weather/high-V is in the heatmap/robustness/table)."""
    import numpy as np
    import pandas as pd
    model = model or f"CeNN_{CENN_MAIN_VARIANT}"
    pred_dir = ARTIFACTS_DIR / "predictions"
    panels = []
    for ds, H in specs:
        files = sorted(glob.glob(str(pred_dir / f"{model}__{ds}__H{H}__seed*.npz")))
        if not files:
            continue
        d0 = np.load(files[0], allow_pickle=True)
        if not all(k in d0 for k in ("unique_id", "ds", "cutoff", "y", "pred")):
            continue
        b = pd.DataFrame({"uid": d0["unique_id"], "ds": d0["ds"], "cutoff": d0["cutoff"],
                          "y": d0["y"].astype("float64"), "pred": d0["pred"].astype("float64")})
        sizes = b.groupby(["uid", "cutoff"]).size()
        full = sizes[sizes == H].index
        if len(full) == 0:
            continue
        key = b.set_index(["uid", "cutoff"]).index
        bf = b[key.isin(full)].copy(); bf["se"] = (bf["y"] - bf["pred"]) ** 2
        werr = bf.groupby(["uid", "cutoff"])["se"].mean()
        uid, cutoff = (werr - werr.median()).abs().idxmin()      # representative window
        sel = b[(b.uid == uid) & (b.cutoff == cutoff)].sort_values("ds")
        y_true, dsv = sel["y"].to_numpy(), sel["ds"].to_numpy()
        preds = [sel["pred"].to_numpy()]
        for f in files[1:]:
            d = np.load(f, allow_pickle=True)
            s = pd.DataFrame({"uid": d["unique_id"], "ds": d["ds"], "cutoff": d["cutoff"],
                              "pred": d["pred"].astype("float64")})
            s = s[(s.uid == uid) & (s.cutoff == cutoff)].sort_values("ds")
            if len(s) == len(y_true) and np.array_equal(s["ds"].to_numpy(), dsv):
                preds.append(s["pred"].to_numpy())
        Pm = np.vstack(preds); t = np.arange(1, len(y_true) + 1)
        panels.append({"dataset": ds, "title": f"{ds}, H={H}", "t": t, "y_true": y_true,
                       "y_pred": Pm.mean(0), "lo": Pm.min(0), "hi": Pm.max(0)})
    if not panels:
        print("[forecast-panels skipped: no prediction artifacts]"); return None
    return plotting.fig_forecast_grid(panels, FIG_DIR)


def contraction_figure(dataset, horizon):
    """fig_contraction (C1 stability): learned vs spectral-capped feedback operator norm ||A_eff||
    per output channel, from an `aeff` artifact of a capped CeNN run (CeNN-Full). Demonstrates the
    cap pins every effective norm below the contraction bound rho<1 (post-hoc, no retrain)."""
    import numpy as np
    aeff_dir = ARTIFACTS_DIR / "aeff"
    for v in [CENN_MAIN_VARIANT, "C1-BoundedTau"]:
        hits = sorted(glob.glob(str(aeff_dir / f"CeNN_{v}__{dataset}__H{horizon}__seed*.npz")))
        if not hits:
            continue
        d = np.load(hits[0])
        raw, capped, rho = d["raw"], d["capped"], float(d["rho"])
        amax = float(d["alpha_max"]) if "alpha_max" in d.files else float("nan")
        order = np.argsort(raw)[::-1]                      # sort channels by learned norm, descending
        x = np.arange(len(raw))
        n_binding = int((raw > rho + 1e-6).sum())          # channels the cap actually pulls down
        # Worst-case (alpha=alpha_max) per-step contraction factor of the Euler map:
        # ||J|| <= alpha + (1-alpha)*||A_eff||. Contraction <=> alpha_max<1 AND ||A_eff||<1 (both
        # guaranteed). The factor is what actually bounds stability -- ||A_eff|| alone understates it,
        # since with alpha_max~0.99 the margin (1-factor) is dominated by the retention rate.
        factor_cap = capped + amax * (1.0 - capped) if amax == amax else None
        f_max = float(factor_cap.max()) if factor_cap is not None else float("nan")
        print(f"[contraction: {v} @ {dataset} H{horizon} -- {n_binding}/{len(raw)} channels exceed "
              f"rho={rho:g} (cap binding); max learned ||A||={raw.max():.3f}; "
              f"worst-case contraction factor (alpha_max={amax:g}) max={f_max:.4f} (<1 => contractive)]")
        if n_binding == 0:
            # Cap never binds: learned and effective norms coincide. The honest story is the factor:
            # ||A_eff|| sits low, but the contraction factor sits near alpha_max (slow contraction).
            curves = {r"$\|A_{\mathrm{eff}}\|$ (cap inactive)": (x, raw[order], plotting.ACC, "-")}
        else:
            # crimson = learned/uncapped (over the bound for some channels), blue = effective/capped.
            curves = {
                r"Learned $\|A\|$ (uncapped)": (x, raw[order], "#C44E52", "-"),
                r"Effective $\|A_{\mathrm{eff}}\|$ (capped)": (x, capped[order], plotting.ACC, "-"),
            }
        if factor_cap is not None:
            curves[r"Per-step factor $\leq 1$ (slow contraction)"] = \
                (x, factor_cap[order], "#117733", "-")
        return plotting.fig_contraction(
            curves, FIG_DIR, rho=rho,
            xlabel="Latent channel (sorted by learned $\\|A\\|$)",
            ylabel=r"Spectral norm $\|A_{\mathrm{eff}}\|$ / per-step factor")
    print(f"[contraction skipped: no aeff artifact for a capped CeNN variant @ {dataset} H{horizon} "
          f"(run CeNN-Full with --save-artifacts)]")
    return None


def scale_uq_figures(dataset, horizon, model=None):
    """fig_scale_disagreement + fig_spread_vs_error (C2 multi-scale UQ) from a `branches` artifact.
    Heatmap value = mean over windows of |branch_scale - cross-scale mean| at each horizon step;
    spread-vs-error pairs per-row cross-scale std with |ensemble - truth|. Returns a list (0-2)."""
    import numpy as np
    model = model or f"CeNN_{CENN_MAIN_VARIANT}"
    hits = sorted(glob.glob(str(ARTIFACTS_DIR / "branches" /
                                f"{model}__{dataset}__H{horizon}__seed*.npz")))
    if not hits:
        print(f"[scale-disagreement/spread-vs-error skipped: no branch artifact for {model} @ "
              f"{dataset} H{horizon} (multi-scale CeNN run with --save-artifacts)]")
        return []
    d = np.load(hits[0])
    branches = d["branches"]                               # [N_rows, n_scales]
    step = d["step"]; y = d["y"]; ens = d["ensemble"]
    dilations = [int(x) for x in d["dilations"]]
    n_scales = branches.shape[1]
    made = []

    # scale-disagreement heatmap: per (scale, horizon-step) deviation from the cross-scale mean
    H = int(step.max()) + 1
    dev = np.abs(branches - branches.mean(axis=1, keepdims=True))   # [N_rows, n_scales]
    spread = np.full((n_scales, H), np.nan)
    for t in range(H):
        mask = step == t
        if mask.any():
            spread[:, t] = dev[mask].mean(axis=0)
    sd = plotting.fig_scale_disagreement(spread, FIG_DIR, scales=dilations)
    if sd:
        made.append(sd)

    # spread-vs-error: cross-scale std vs |ensemble - truth|. Rows are NOT independent (H steps per
    # window, overlapping windows), so naive correlation significance is inflated. Report rho with a
    # window-level BLOCK-bootstrap 95% CI (resample whole windows) — the honest uncertainty on rho.
    row_spread = branches.std(axis=1); row_err = np.abs(ens - y)
    annotation = None
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(row_spread, row_err).correlation)
        wid = np.cumsum(step == 0) - 1                     # window id (a new window starts at step 0)
        n_w = int(wid.max()) + 1
        idx_by_w = [np.where(wid == w)[0] for w in range(n_w)]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(500):
            sel = np.concatenate([idx_by_w[w] for w in rng.integers(0, n_w, n_w)])
            r = spearmanr(row_spread[sel], row_err[sel]).correlation
            if r == r:                                     # drop NaN draws
                boots.append(r)
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
            if boots else (float("nan"), float("nan"))
        annotation = rf"Spearman $\rho$={rho:.2f} [{lo:.2f}, {hi:.2f}]"
        print(f"[spread-vs-error: {model} @ {dataset} H{horizon} -- rho={rho:.3f}, "
              f"95% block-bootstrap CI over {n_w} windows=[{lo:.3f}, {hi:.3f}], "
              f"{len(row_spread)} rows. (Placement: main-text vs appendix is the author's call.)]")
    except Exception as e:
        print(f"[spread-vs-error rho/CI failed: {type(e).__name__}: {e}]")
    se = plotting.fig_spread_vs_error(row_spread, row_err, FIG_DIR, annotation=annotation)
    if se:
        made.append(se)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1", help="anchor dataset for efficiency figures")
    ap.add_argument("--horizon", type=int, default=96)
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mse, eff = seed_mean_mse(), load_efficiency()
    if not eff:
        # Efficiency figures (pareto/panels/k-sweep) need eff JSONs, but the MSE-/artifact-driven
        # figures (horizon curves, tau, ablation, boxplots, forecast grid, UQ diagnostics) do NOT
        # -> warn and continue rather than returning, so a partial run still renders what it can.
        print(f"[no efficiency JSONs in {EFF_DIR}: skipping pareto/panels/k-sweep; "
              f"run profile_efficiency.py for those. Other figures continue.]")

    mse_scope = scoped_mean_mse(DATASETS_HEADLINE)         # headline-regime mean accuracy
    rows = build_rows(args.dataset, args.horizon, mse_scope, eff)
    pts = build_ksweep(args.dataset, args.horizon, mse, eff, std=seed_std_at(args.dataset, args.horizon))

    print(f"anchor {args.dataset} H{args.horizon}: {len(rows)} models with eff+scoped-MSE; "
          f"{len(pts)} K-sweep points")
    made = []
    # Pareto + efficiency-panels DROPPED: efficiency is NOT a pillar (AMS ~6x DLinear MACs);
    # the efficiency TABLE carries the honest compute disclosure. The K-integrator figure stays.
    # K-sweep figure DROPPED: the K-surface is flat (dMSE < 0.001 across K in {2,4,8}),
    # which makes a weak figure (tight axis misleads, honest axis is an empty flat line). Reported as a
    # one-line statement + a table row instead. The L-sweep is the sensitivity figure:
    lsf = lsweep_fig()
    if lsf:
        made.append(lsf)

    hl_models = [f"CeNN_{CENN_MAIN_VARIANT}"] + BASELINES_ALL
    # horizon SMALL-MULTIPLES (per-dataset, all 7): AMS-CeNN + 3 best + 2 worst -> no spaghetti.
    hgrid_models = [f"CeNN_{CENN_MAIN_VARIANT}", "TSMixer", "PatchTST", "DLinear", "S4D", "TimesNet"]
    hc = horizon_grid(mse, DATASETS_ALL, hgrid_models)
    if hc:
        made.append(hc)
    tau = tau_figure(args.dataset, args.horizon)
    if tau:
        made.append(tau)

    # results-driven appendix figures + qualitative forecast grid
    df = load_all_results()
    if not df.empty:
        # NOTE: the ablation BAR CHART is retired. Every ABL-* variant lacks the linear
        # skip, so each bar's Δ vs the skip-headline is dominated by the absent skip (~0.030) -> the
        # chart misleads. The honest ablation is the build-up TABLE (tab:ablation, make_tables.py).
        hm = rank_heatmap(mse, hl_models)                # §1 no-champion motivation (per-dataset ranks)
        if hm:
            made.append(hm)
        rb = robustness_fig(mse, hl_models)              # pillar 1 = robust (worst-case relative error)
        if rb:
            made.append(rb)
        bx = rank_boxplots(mse, hl_models)               # scale-free per-block ranks (appendix/supplementary)
        if bx:
            made.append(bx)
    # Qualitative figure: 2 datasets x 2 horizons (short + extreme) -> the LONG-TERM story
    # (AMS-CeNN holds the periodic structure to H720) on the datasets it tracks, with REPRESENTATIVE
    # (median-error) windows. Per-dataset accuracy incl. Weather/high-V is carried by the heatmap/
    # robustness/main table; ECL/Traffic have no single representative series (multivariate).
    fg = forecast_panels([("ETTh1", 96), ("ETTh1", 720), ("ETTm2", 96), ("ETTm2", 720)])
    if fg:
        made.append(fg)

    # C1 stability diagnostic (supports the 'stable' pillar) — KEPT.
    ct = contraction_figure(args.dataset, args.horizon)
    if ct:
        made.append(ct)
    # C2 UQ figures (scale-disagreement, spread-vs-error) CUT from this work: UQ/conformal
    # calibration is a future-work deliverable; rho=0.19 is weak.

    for p in made:
        print(f"  wrote {p}")
    if not made:
        print("No figures written (insufficient real data at this anchor).")


if __name__ == "__main__":
    main()
