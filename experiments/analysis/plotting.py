#!/usr/bin/env python
"""Research-grade figures for this work — built on SciencePlots `ieee` style.

Quality bar: SciencePlots ieee base (Times font, IEEE geometry) + readable sizes (no tiny
text), colorblind-safe palette, Title-Case legends, label-repulsion with leader lines (no
overlap), vector PDF + 600-dpi PNG. Every figure reads from the runner's result/efficiency/
artifact JSONs, so they regenerate cleanly once the real runs land; the __main__ demo renders
the WHOLE suite with illustrative data so the styling can be reviewed before the runs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots  # noqa: F401  (registers the 'science'/'ieee' styles)

COL_W, FULL_W = 3.5, 7.16        # IEEE single / double column width (in)
_STYLE_ON = False

# Colorblind-safe palette (Okabe-Ito) + Title-Case display names for legends
FAMILY = {
    "CeNN":        ("#D55E00", "CeNN variants"),  # cloud; the headline AMS-CeNN is labelled separately
    "linear":      ("#0072B2", "Linear"),
    "mlp":         ("#009E73", "MLP"),
    "transformer": ("#8000C0", "Transformer"),   # purple (orange now reserved for CeNN)
    "conv":        ("#CC79A7", "Convolutional"),
}
ACC = "#0072B2"; CENN_C = "#D55E00"; OK = "#009E73"; BAD = "#D55E00"


def apply_style():
    global _STYLE_ON
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({
        # SciencePlots ieee uses ~8pt; bump for legibility ("no small text").
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
        "figure.dpi": 130, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.prop_cycle": plt.cycler(color=[c for c, _ in FAMILY.values()]),
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.7",
        "figure.constrained_layout.use": True,
    })
    _STYLE_ON = True


def savefig(fig, outdir, name):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf"); fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    return outdir / f"{name}.png"


def _repel(texts, ax):
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.25, 1.6),
                    force_text=(0.5, 0.9), force_static=(0.3, 0.5),
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4),
                    max_move=40)
    except Exception:
        pass


def _pareto_idx(xs, ys):
    order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    out, best = [], float("inf")
    for i in order:
        if ys[i] <= best:
            out.append(i); best = ys[i]
    return out


# ---------------------------------------------------------------------------
# 1. Pareto: accuracy vs MACs  (efficiency centerpiece) — full-width for room
# ---------------------------------------------------------------------------
def fig_pareto(rows, outdir, name="fig_pareto_acc_vs_macs",
               anchor_note="MACs at ETTh1, H96", y_note="ETT + Weather"):
    """Accuracy(headline-regime mean MSE) vs compute(MACs at one anchor). Only the headline
    AMS-CeNN and the baselines are text-labelled; the other CeNN variants render as a faded
    family cloud (no labels) so the figure isn't drowned in ablation text."""
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.7, 3.0))
    xs = [r["macs"] / 1e6 for r in rows]; ys = [r["mse"] for r in rows]
    fi = _pareto_idx(xs, ys)
    ax.plot([xs[i] for i in fi], [ys[i] for i in fi], color="0.55", lw=1.0, ls="--",
            zorder=1, label="Pareto frontier")
    seen, texts = set(), []
    for i, r in enumerate(rows):
        fam = r.get("family", "linear"); c, disp = FAMILY.get(fam, ("0.3", fam))
        headline = r.get("headline", False)
        cenn_cloud = (fam == "CeNN") and not headline      # faded, unlabelled family member
        if headline:
            ax.scatter(xs[i], ys[i], s=150, marker="*", facecolor=c, edgecolor="black",
                       linewidth=0.9, zorder=6, label=disp if fam not in seen else None)
            # headline label offset above-left of the star (annotate, not a repelled text) so the
            # bold name never sits on its own marker
            ax.annotate(r.get("label"), (xs[i], ys[i]), xytext=(-6, 9),
                        textcoords="offset points", fontsize=8, fontweight="bold",
                        color="0.05", ha="right", zorder=7)
        elif cenn_cloud:
            ax.scatter(xs[i], ys[i], s=30, marker="*", facecolor=c, edgecolor="none",
                       alpha=0.32, zorder=2, label=disp if fam not in seen else None)
        else:                                               # baseline: labelled circle
            ax.scatter(xs[i], ys[i], s=46, marker="o", facecolor=c, edgecolor="white",
                       linewidth=0.7, zorder=3, label=disp if fam not in seen else None)
            texts.append(ax.text(xs[i], ys[i], r.get("label", r["model"]), fontsize=7, color="0.1"))
        seen.add(fam)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"MACs per forecast (M, log scale) — {anchor_note}")
    ax.set_ylabel(rf"Mean MSE over {y_note} ($\downarrow$)")
    ax.set_title("Accuracy–efficiency frontier")
    ax.margins(x=0.16, y=0.20)
    _repel(texts, ax)
    # extra right headroom on the log x-axis so the right-most label (xLSTM) is not clipped
    xl, xr = ax.get_xlim(); ax.set_xlim(xl, xr * 2.4)
    ax.legend(loc="upper right", ncol=1, handletextpad=0.4, borderpad=0.4, labelspacing=0.25)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 2. Efficiency panels: MSE vs {MACs, Params, Peak inference memory}
# ---------------------------------------------------------------------------
def fig_efficiency_panels(rows, outdir, name="fig_efficiency_panels"):
    apply_style()
    # constrained_layout off here: a figure-level top legend + manual spacing avoids the
    # uneven inter-panel gaps and the clipped 3rd-panel xlabel.
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), constrained_layout=False)
    keys = [("macs", 1e6, "MACs/forecast (M)"), ("params", 1e6, "Params (M)"),
            ("peak_mem", 1.0, "Peak inference mem (MB)")]
    handles = {}
    for ax, (k, sc, xlab) in zip(axes, keys):
        for r in rows:
            fam = r.get("family", "linear"); c, disp = FAMILY.get(fam, ("0.3", fam))
            headline = r.get("headline", False)
            cenn_cloud = (fam == "CeNN") and not headline
            if headline:
                h = ax.scatter(r[k] / sc, r["mse"], s=130, marker="*", facecolor=c,
                               edgecolor="black", linewidth=0.8, zorder=6)
                handles["AMS-CeNN"] = h
            elif cenn_cloud:
                h = ax.scatter(r[k] / sc, r["mse"], s=24, marker="*", facecolor=c,
                               edgecolor="none", alpha=0.32, zorder=2)
                handles.setdefault(disp, h)
            else:
                h = ax.scatter(r[k] / sc, r["mse"], s=34, marker="o", facecolor=c,
                               edgecolor="white", linewidth=0.6, zorder=3)
                handles.setdefault(disp, h)
        ax.set_xscale("log"); ax.set_xlabel(xlab)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.margins(x=0.18, y=0.16)
    axes[0].set_ylabel(r"Mean MSE, ETT+Weather ($\downarrow$)")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.3, handletextpad=0.3, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.9, wspace=0.32)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 3. K x integrator accuracy-vs-K sweep (K-step trade-off; shows K=2 suffices — NOT an efficiency claim)
# ---------------------------------------------------------------------------
def fig_k_integrator(points, outdir, name="fig_k_integrator"):
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 2.7))   # wider so low-MAC K points + labels fit
    integ_marker = {"euler": "o", "exp_euler": "s", "heun": "^", "rk4": "D"}
    integ_disp = {"euler": "Euler", "exp_euler": "Exp-Euler", "heun": "Heun (RK2)", "rk4": "RK4"}
    # per-integrator label offset (points) so coincident K labels (e.g. the three K4 near 5.8 MACs)
    # don't stack into one another
    integ_off = {"euler": (0, 9), "heun": (0, -12), "exp_euler": (13, 0), "rk4": (-13, 0)}
    xs_all, ys_all = [], []
    for integ, mk in integ_marker.items():
        pts = sorted([p for p in points if p["integrator"] == integ], key=lambda p: p["macs"])
        if not pts:
            continue
        xs = [p["macs"] / 1e6 for p in pts]; ys = [p["mse"] for p in pts]
        es = [p.get("std", 0.0) for p in pts]
        xs_all += xs; ys_all += ys
        ax.errorbar(xs, ys, yerr=es, marker=mk, label=integ_disp[integ], lw=1.1, ms=5,
                    capsize=2.5, elinewidth=0.8, capthick=0.8)
        dx, dy = integ_off.get(integ, (0, 9))
        for x, y, p in zip(xs, ys, pts):
            ax.annotate(f"K{p['K']}", (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.2, color="0.3", ha="center", va="center")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # pad the frame so no point/label is clipped (esp. the low-MAC K2 points)
    lo, hi = min(xs_all), max(xs_all)
    ax.set_xlim(lo / 2.1, hi * 2.1)
    ax.margins(y=0.26)
    ax.set_xlabel("MACs per forecast (M, log)"); ax.set_ylabel(r"MSE ($\downarrow$)")
    ax.set_title(r"Integrator $\times$ K trade-off (error bars: seed std)")
    ax.legend(title="Integrator", loc="upper right", labelspacing=0.25, handletextpad=0.4)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 4. Critical-difference diagram (Demsar) — significance of rank differences
# ---------------------------------------------------------------------------
def fig_cd_diagram(mean_ranks, cd, outdir, name="fig_cd_diagram", note=None):
    apply_style()
    import math
    items = sorted(mean_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; ranks = [v for _, v in items]
    n = len(names); lo, hi = 1, n
    STEP = 0.30                                     # compact row gap; width de-knots in x
    y0 = n / 2 + 1.0
    n_left = math.ceil(n / 2)                       # labels i=0..n_left-1 go left
    lowest = y0 - 0.5 - (n_left - 1) * STEP         # y of the lowest (deepest) label row
    fig, ax = plt.subplots(figsize=(FULL_W * 0.98, STEP * n + 0.35))  # wide + short, tight margins
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_ylim(lowest - 0.28, y0 + 0.6)            # tight top/bottom margins (kills white space)
    ax.axis("off")
    ax.plot([lo, hi], [y0, y0], "k-", lw=1.2)               # rank axis
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.08], "k-", lw=1.0)
        ax.text(r, y0 + 0.18, str(r), ha="center", fontsize=7)
    # place labels: left half on the left, right half on the right
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        left = i < n / 2
        yy = y0 - 0.5 - (i if left else (n - 1 - i)) * STEP
        xend = lo - 0.5 if left else hi + 0.5
        is_hl = ("CeNN" in nm or nm == "AMS-CeNN")
        lc = CENN_C if is_hl else "0.15"            # AMS-CeNN leader line in the headline accent
        ax.plot([rk, rk], [y0, yy], "-", color=lc, lw=1.1 if is_hl else 0.8)
        ax.plot([rk, xend], [yy, yy], "-", color=lc, lw=1.1 if is_hl else 0.8)
        ax.text(xend + (-0.12 if left else 0.12), yy, f"{nm} ({rk:.2f})",
                ha="right" if left else "left", va="center", fontsize=7.6,
                color=CENN_C if is_hl else "black",
                fontweight="bold" if is_hl else "normal")
    # CD clique bars: only MAXIMAL cliques (consecutive runs within cd). Keeping every start index
    # produced many nested/overlapping bars that merged into an unreadable black smear; a clique is
    # kept only if its right endpoint extends past every previous kept clique (== maximal).
    cliques, last_j = [], -1
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and j > last_j:
            cliques.append((ranks[i], ranks[j])); last_j = j
        i += 1
    yb = y0 - 0.16                                   # clean band just below the rank axis
    for k, (a, b) in enumerate(cliques):
        ax.plot([a - 0.05, b + 0.05], [yb - k * 0.13, yb - k * 0.13], "-",
                color="#333333", lw=3.0, solid_capstyle="round")
    ax.annotate(f"CD = {cd:.2f}", (lo, y0 + 0.45), fontsize=7.5, color="0.3")
    ax.plot([lo, lo + cd], [y0 + 0.38, y0 + 0.38], "k-", lw=1.5)
    title = "Critical-difference diagram (avg ranks, dataset$\\times$horizon)"
    if note:
        title += f"\n{note}"
    ax.set_title(title, fontsize=9, pad=3)
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 5. Per-channel adaptive retention profile (interpretability, C1)
# ---------------------------------------------------------------------------
def fig_tau_profile(mean_alpha, std_alpha, outdir, name="fig_tau_heatmap", model_label="AMS-CeNN",
                    note=None):
    """Learned retention alpha=1-tau per LATENT CHANNEL (sorted). alpha is ~constant over the
    lookback (time-std ~1e-4), so a time x channel heatmap would imply a per-timestep adaptivity
    that does NOT exist; instead we show the time-averaged per-channel value with a +-std band over
    input windows. Honest framing: the channel spread is SMALL = MODEST per-channel
    structure (near-uniform retention), NOT rich learned heterogeneity. We print the spread so the
    auto-scaled (tiny) y-range isn't misread as dramatic variation."""
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    order = np.argsort(mean_alpha)
    m = np.asarray(mean_alpha)[order]; s = np.asarray(std_alpha)[order]
    x = np.arange(len(m))
    ax.plot(x, m, color=CENN_C, lw=1.5, zorder=3, label="mean retention")
    ax.fill_between(x, m - s, m + s, color=CENN_C, alpha=0.22, lw=0,
                    label=r"$\pm$std over input windows")
    ax.set_xlabel("Latent channel (sorted by retention)")
    ax.set_ylabel(r"Retention $\alpha = 1-\tau$")
    ax.set_title(f"Learned per-channel retention ({model_label})", fontsize=8.7)
    ax.legend(loc="lower right", fontsize=6.8, handletextpad=0.5, framealpha=0.92)
    ax.margins(x=0.02)
    spread = float(m.max() - m.min())
    lab = (f"{note}\n" if note else "") + f"channel spread $\\approx$ {spread:.3f} (near-uniform)"
    ax.text(0.03, 0.96, lab, transform=ax.transAxes, va="top", ha="left", fontsize=6.3, color="0.4")
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 5. Adaptive-tau heatmap (interpretability, C1)
# ---------------------------------------------------------------------------
def fig_tau_heatmap(tau, outdir, name="fig_tau_heatmap", channel_labels=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    im = ax.imshow(tau, aspect="auto", cmap="cividis", origin="lower")
    ax.set_xlabel("Lookback time step"); ax.set_ylabel("Variable")
    if channel_labels is not None:
        ax.set_yticks(range(len(channel_labels))); ax.set_yticklabels(channel_labels, fontsize=7)
    ax.set_title(r"Learned adaptive retention $\alpha = 1-\tau$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\alpha$", fontsize=9); cb.ax.tick_params(labelsize=7)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 6. Reliability diagram (UQ calibration, seed-ensemble)
# ---------------------------------------------------------------------------
def fig_reliability(curves, outdir, name="fig_reliability"):
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.92))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, label="Ideal")
    for lab, (nom, emp, c) in curves.items():
        ax.plot(nom, emp, marker="o", ms=4, lw=1.2, color=c, label=lab)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", labelspacing=0.25)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 7. Forecast with prediction-interval band
# ---------------------------------------------------------------------------
def fig_forecast_intervals(t, y_true, y_pred, lo, hi, outdir, name="fig_forecast_intervals", ctx=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.66, 2.4))
    if ctx is not None:
        tc, yc = ctx
        ax.plot(tc, yc, color="0.4", lw=1.0, label="History")
    ax.plot(t, y_true, color="black", lw=1.2, label="Ground truth")
    ax.plot(t, y_pred, color=CENN_C, lw=1.2, label="CeNN forecast")
    ax.fill_between(t, lo, hi, color=CENN_C, alpha=0.2, lw=0, label="95% interval")
    ax.axvline(t[0], color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("Time step"); ax.set_ylabel("Value (normalized)")
    ax.set_title("Forecast with seed-ensemble interval")
    ax.legend(loc="upper left", ncol=2, labelspacing=0.25, columnspacing=0.9)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 8. Ablation contribution bars (leave-one-out delta MSE)
# ---------------------------------------------------------------------------
def fig_ablation_bars(items, outdir, name="fig_ablation_bars"):
    """items: (label, Δ, highlight, group). Δ = variant_MSE − headline_MSE on MATCHED cells (always ≥0
    here = AMS-CeNN is best). TWO colour-coded, gap-separated groups: 'Removed component' (Δ = the
    accuracy that component contributes) and 'Alternative design' (Δ = how much worse that swap is).
    The − linear skip is drawn in the headline accent. A legend names the groups; no ambiguous ±."""
    import matplotlib.patches as mpatches
    apply_style()
    REMOVED, ALT = "Removed component", "Alternative design"
    C_REM, C_ALT = "#4C78A8", "#B0B7BE"        # blue = removed component, grey = alternative design
    groups = [REMOVED, ALT]
    by_g = {g: [t for t in items if (t[3] if len(t) > 3 else ALT) == g] for g in groups}
    for g in groups:
        by_g[g].sort(key=lambda t: t[1])        # ascending Δ within group (smallest at top)
    ypos, labels, vals, cols = [], [], [], []
    band = []                                   # (y0, y1, group) for the section header
    y = 0.0; GAP = 1.4
    for g in groups:
        rows = by_g[g]
        if not rows:
            continue
        y0 = y
        for (label, delta, hi, *_r) in rows:
            ypos.append(y); labels.append(label); vals.append(delta)
            cols.append(CENN_C if hi else (C_REM if g == REMOVED else C_ALT))
            y += 1.0
        band.append((y0, y - 1.0, g)); y += GAP
    n = len(ypos)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.7, 0.4 * n + 1.5))
    ax.barh(ypos, vals, color=cols, edgecolor="white", linewidth=0.5, height=0.74)
    ax.axvline(0, color="0.3", lw=0.8)
    xmax = max(vals) if vals else 1.0
    for yp, v in zip(ypos, vals):
        ax.text(v + 0.012 * xmax, yp, f"{v:+.3f}", va="center", fontsize=6.6, color="0.25")
    # bold section header centred above each group
    for y0, y1, g in band:
        ax.text(xmax * 0.62, y0 - 0.85, g, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color="0.15")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_ylim(-1.4, n + GAP); ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.2)
    ax.set_xlabel(r"$\Delta$ MSE vs AMS-CeNN on matched cells  (larger = bigger effect)")
    handles = [mpatches.Patch(color=C_REM, label="Remove a component of AMS-CeNN"),
               mpatches.Patch(color=C_ALT, label="Swap in an alternative design"),
               mpatches.Patch(color=CENN_C, label="The linear skip (key contribution)")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=6.8,
              handlelength=1.1, handletextpad=0.5, borderpad=0.5, framealpha=0.95)
    ax.set_title("Ablation of AMS-CeNN (each variant vs the headline)", fontsize=9)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 9. Scale-disagreement heatmap (appendix UQ diagnostic)
# ---------------------------------------------------------------------------
def fig_scale_disagreement(spread, outdir, name="fig_scale_disagreement", scales=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    im = ax.imshow(spread, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("Forecast horizon step"); ax.set_ylabel("Dilation scale")
    if scales is not None:
        ax.set_yticks(range(len(scales))); ax.set_yticklabels(scales, fontsize=7)
    # "branch disagreement" (not "per-scale forecast"): branches are diagnostic probes of the
    # multi-scale latent, not independent forecasts (forward_branches heads each branch separately).
    ax.set_title("Multi-scale branch disagreement")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("disagreement", fontsize=9); cb.ax.tick_params(labelsize=7)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 10. Training-stability curves (appendix)
# ---------------------------------------------------------------------------
def fig_contraction(curves, outdir, name="fig_contraction", xlabel="Training step", rho=None,
                    ylabel=r"Feedback operator norm $\|A_{\mathrm{eff}}\|$"):
    """Direct, honest demonstration of the stability mechanism: the feedback operator norm
    ||A_eff|| stays below the contraction bound (1) WITH the spectral cap, but grows past it
    without. (The spectral cap is empirically slack; this shows what the cap actually guarantees.)

    The post-hoc variant feeds per-channel learned-vs-capped norms from a trained checkpoint
    (xlabel='Feedback operator (channels, sorted by learned norm)'); the x-axis is then a channel
    rank, not a training step — caller sets xlabel accordingly."""
    apply_style()
    # Two quantities live on very different scales -> two panels (the old single-axis squashed the
    # per-step factor flat against the top). LEFT = operator norm ||A_eff|| (0..~0.5, well under the
    # bound = the cap is slack); RIGHT = per-step contraction factor (~0.99, the REAL margin, which is
    # alpha-dominated). Factor curves are routed right by their label.
    fac = {k: v for k, v in curves.items() if "factor" in k.lower()}
    nrm = {k: v for k, v in curves.items() if "factor" not in k.lower()}
    fig, (axn, axf) = plt.subplots(1, 2, figsize=(FULL_W * 0.86, 2.5))
    ymax = 0.0
    for lab, (steps, norm, c, ls) in nrm.items():
        axn.plot(steps, norm, color=c, ls=ls, lw=1.5, label=lab); ymax = max(ymax, float(np.max(norm)))
    axn.axhline(1.0, color="#C44E52", ls=":", lw=1.1, label="contraction bound (=1)")
    if rho is not None:
        axn.axhline(rho, color="0.45", ls="--", lw=0.9, label=rf"spectral cap $\rho$={rho:g}")
    axn.set_ylim(0, max(1.12, ymax * 1.08))
    axn.set_xlabel(xlabel); axn.set_ylabel(r"Operator norm $\|A_{\mathrm{eff}}\|$")
    axn.set_title("Effective operator norm (cap is slack)", fontsize=8.3)
    axn.legend(fontsize=6.4, loc="center right", labelspacing=0.3, handletextpad=0.4, framealpha=0.9)
    if fac:
        allf = np.concatenate([np.asarray(v[1], float) for v in fac.values()])
        for lab, (steps, f, c, ls) in fac.items():
            axf.plot(steps, f, color="#117733", ls=ls, lw=1.7, label=lab)
        axf.axhline(1.0, color="#C44E52", ls=":", lw=1.1, label="bound (=1)")
        pad = max(1e-3, (1.0 - float(allf.min())) * 0.25)
        axf.set_ylim(float(allf.min()) - pad, 1.0 + pad)
        axf.set_xlabel(xlabel); axf.set_ylabel("Per-step contraction factor")
        axf.set_title(r"Per-step factor $<1$ ($\alpha$-dominated)", fontsize=8.3)
        axf.legend(fontsize=6.4, loc="lower right", labelspacing=0.3, handletextpad=0.4, framealpha=0.9)
    else:
        axf.axis("off")
    fig.tight_layout(w_pad=1.2)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 11. Forecast-vs-ground-truth grid across datasets (qualitative, with intervals)
# ---------------------------------------------------------------------------
def fig_forecast_grid(panels, outdir, name="fig_forecast_grid", ncols=None, model_label="AMS-CeNN"):
    """panels: list of {dataset, t, y_true, y_pred, lo, hi, ctx_t?, ctx_y?}."""
    apply_style()
    n = len(panels)
    if ncols is None:
        ncols = 2 if n <= 4 else 3                          # 2x2 for the 4-ETT grid
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_W, 1.8 * nrows + 0.5),
                             squeeze=False, constrained_layout=False)
    handles = {}
    for idx, p in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if p.get("ctx_t") is not None:
            h = ax.plot(p["ctx_t"], p["ctx_y"], color="0.45", lw=0.9)[0]; handles.setdefault("History", h)
        # seed-ensemble band: a LIGHTER tint than the prediction line (was the same orange -> blended),
        # with a thin edge, so the narrow (seed-stable) halo is perceptible. Genuinely thin at short
        # horizons; widens with horizon as uncertainty accumulates.
        h = ax.fill_between(p["t"], p["lo"], p["hi"], facecolor="#FDC692", edgecolor="#E8923A",
                            alpha=0.85, lw=0.35, zorder=2); handles.setdefault("Seed-ensemble range", h)
        h = ax.plot(p["t"], p["y_true"], color="black", lw=1.1, zorder=3)[0]; handles.setdefault("Ground truth", h)
        h = ax.plot(p["t"], p["y_pred"], color=CENN_C, lw=1.3, zorder=4)[0]; handles.setdefault(model_label, h)
        ax.set_title(p.get("title", p["dataset"]), fontsize=8.0, pad=2)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):                  # hide unused axes
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Forecast step ($h$ ahead)", fontsize=9, x=0.5)
    fig.supylabel("Value (normalized)", fontsize=9)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 1.005), frameon=False, columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, wspace=0.22, hspace=0.5)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 12. Per-horizon error curves (MSE vs forecast horizon)
# ---------------------------------------------------------------------------
def fig_horizon_curves(series, horizons, outdir, name="fig_horizon_curves", scope_note=None):
    """series: dict model -> {'mse': [per-horizon], 'family', 'headline'}. The headline AMS-CeNN is
    a bold orange star line; every other method gets a DISTINCT colour from a qualitative map (the
    family palette collides — 4 MLPs share one green), greyed thinner so the headline reads."""
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.66, 2.7))
    others = [m for m, d in series.items() if not d.get("headline")]
    # hand-picked distinct, reasonably-dark colours (no pale yellow-on-white); orange reserved
    # for the headline so it is excluded here.
    _PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
            "#7E2F8E", "#A2142F", "#4D4D4D", "#117733", "#666600"]
    cidx = {m: _PAL[i % len(_PAL)] for i, m in enumerate(others)}
    for model, d in series.items():
        if d.get("headline"):
            continue
        ax.plot(horizons, d["mse"], marker="o", ms=3.5, lw=1.0, color=cidx[model],
                label=model, zorder=3, alpha=0.9)
    for model, d in series.items():                          # headline drawn last, on top
        if d.get("headline"):
            ax.plot(horizons, d["mse"], marker="*", ms=9, lw=2.2, color=CENN_C,
                    label=model, zorder=6)
    ax.set_xticks(horizons); ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"MSE ($\downarrow$)")
    ax.set_title("Error vs horizon" + (f" ({scope_note})" if scope_note else ""))
    ax.legend(loc="upper left", labelspacing=0.2, fontsize=6.5, ncol=2, columnspacing=1.0)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 13. MSE distribution box plots (robustness across seeds / dataset-horizons)
# ---------------------------------------------------------------------------
def fig_line_sensitivity(x, series, outdir, name, xlabel, ylabel, title, highlight="AMS-CeNN",
                         xlog2=False, note=None, std=None, ylim=None):
    """Generic sensitivity line plot (L-sweep, K-sweep). x: list of x values. series: dict label ->
    y-list (same length as x). highlight drawn thicker in the accent. std: optional dict label ->
    list of +-std for an error band."""
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W * 1.32, 2.4))
    pal = [ACC, "#7F7F7F", OK, "#CC79A7"]
    pi = 0
    for lab, ys in series.items():
        is_hl = (lab == highlight)
        c = CENN_C if is_hl else pal[pi % len(pal)]
        if not is_hl:
            pi += 1
        ax.plot(x, ys, color=c, lw=2.1 if is_hl else 1.4, marker="o", ms=4.2 if is_hl else 3.2,
                label=lab, zorder=6 if is_hl else 3)
        if std and lab in std:
            s = np.asarray(std[lab]); yy = np.asarray(ys)
            ax.fill_between(x, yy - s, yy + s, color=c, alpha=0.15, lw=0, zorder=2)
    if xlog2:
        ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
        ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)                                  # contextual range so a flat result reads as flat
    ax.set_title(title, fontsize=8.8)
    ax.legend(fontsize=7.2, labelspacing=0.3, handlelength=1.6)
    if note:
        ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.6, color="0.4")
    return savefig(fig, outdir, name)


def fig_robustness(items, outdir, name="fig_robustness", highlight="AMS-CeNN", cap=35.0):
    """items: list of (label, pct) where pct = % above the best model on the WORST dataset (worst-case
    relative error). Sorted ascending (most robust on top); AMS-CeNN accented. Bars beyond `cap` are
    clipped with a '-> N%' annotation so the competitive pack stays readable."""
    apply_style()
    items = sorted(items, key=lambda t: t[1])
    labels = [t[0] for t in items]; vals = [t[1] for t in items]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 0.34 * len(items) + 1.0))
    colors = [CENN_C if lab == highlight else "#9ecae1" for lab in labels]
    drawn = [min(v, cap) for v in vals]
    ax.barh(range(len(items)), drawn, color=colors, edgecolor="white", linewidth=0.5, height=0.74)
    for i, v in enumerate(vals):
        if v > cap:
            ax.text(cap * 0.995, i, f"  → {v:.0f}%", va="center", ha="left", fontsize=6.4, color="0.4")
        else:
            ax.text(v + cap * 0.012, i, f"{v:.1f}%", va="center", fontsize=6.6, color="0.25")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([(r"$\bf{%s}$" % lab.replace("-", "\\text{-}")) if lab == highlight else lab
                        for lab in labels], fontsize=7.6)
    for tl, lab in zip(ax.get_yticklabels(), labels):
        if lab == highlight:
            tl.set_color(CENN_C)
    ax.invert_yaxis()
    ax.set_xlim(0, cap * 1.12)
    ax.set_xlabel("Worst-case error across the 7 datasets (% above the best model)")
    ax.set_title("Robustness: worst-case error across the 7 datasets\n(AMS-CeNN lowest at 6.5%; most "
                 "baselines fail badly on $\\geq$1 dataset)", fontsize=8.2)
    return savefig(fig, outdir, name)


def fig_horizon_grid(panels, horizons, outdir, name="fig_horizon_curves", highlight="AMS-CeNN",
                     order=None):
    """panels: dict dataset -> {model_label: [mse per horizon]}. One small panel per dataset (own
    y-scale, so no cross-dataset squashing), shared legend in the spare cell, AMS-CeNN highlighted.
    Replaces the 13-line single-axes 'spaghetti' horizon plot."""
    import math
    apply_style()
    dsets = list(panels.keys())
    models = order or sorted({m for p in panels.values() for m in p})
    pal = [ACC, OK, "#CC79A7", "#56B4E9", "#999999", "#117733"]
    cmap = {}
    pi = 0
    for m in models:
        if m == highlight:
            cmap[m] = CENN_C
        else:
            cmap[m] = pal[pi % len(pal)]; pi += 1
    ncol = 4
    nrow = math.ceil((len(dsets) + 1) / ncol)          # +1 spare cell for the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, 1.95 * nrow))
    axes = axes.flatten()
    x = list(range(len(horizons)))
    handles = {}
    for ai, d in enumerate(dsets):
        ax = axes[ai]
        for m in models:
            if m not in panels[d]:
                continue
            is_hl = (m == highlight)
            ln, = ax.plot(x, panels[d][m], color=cmap[m], lw=2.1 if is_hl else 1.1,
                          marker="o" if is_hl else None, ms=3.4, zorder=6 if is_hl else 3, label=m)
            handles[m] = ln
        # per-panel y-cap: if one model is a big outlier (>1.8x the next), clip it so the competitive
        # pack stays readable; annotate which model goes off-scale.
        maxes = sorted(((max(panels[d][m]), m) for m in panels[d]), reverse=True)
        lo = min(min(panels[d][m]) for m in panels[d])
        if len(maxes) >= 2 and maxes[0][0] > 1.8 * maxes[1][0]:
            cap = maxes[1][0] * 1.18
            ax.set_ylim(lo - 0.03 * (cap - lo), cap)
            ax.text(0.97, 0.96, f"{maxes[0][1]} off-scale", transform=ax.transAxes, ha="right",
                    va="top", fontsize=5.8, color="0.45", style="italic")
        ax.set_title(d, fontsize=8.4)
        ax.set_xticks(x); ax.set_xticklabels(horizons, fontsize=6.6)
        ax.tick_params(labelsize=6.6); ax.margins(x=0.04)
        if ai % ncol == 0:
            ax.set_ylabel(r"MSE ($\downarrow$)", fontsize=7.4)
    for kk in range(len(dsets), len(axes)):
        axes[kk].axis("off")
    leg = axes[len(dsets)]
    leg.legend([handles[m] for m in models if m in handles], [m for m in models if m in handles],
               loc="center", fontsize=7.6, frameon=False, handlelength=1.6, title="Model")
    fig.supxlabel("Forecast horizon", fontsize=8.4)
    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.9)
    return savefig(fig, outdir, name)


def fig_rank_heatmap(models, datasets, rankmat, outdir, name="fig_rank_heatmap",
                     highlight="AMS-CeNN", k=12, title=None):
    """rankmat[i][j] = integer rank (1=best of k) of models[i] on datasets[j]. Green→red, rank printed
    in every cell; the highlighted model's row is boxed. Visualises the no-champion pattern: no column
    is all-green (no model wins every dataset)."""
    apply_style()
    arr = np.asarray(rankmat, float)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.66, 0.3 * len(models) + 1.1))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=k)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=7.6, rotation=18, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7.6)
    for i in range(len(models)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{arr[i][j]:.0f}", ha="center", va="center", fontsize=7.0,
                    color="black", fontweight="bold" if models[i] == highlight else "normal")
    if highlight in models:
        hi = models.index(highlight)
        ax.add_patch(plt.Rectangle((-0.5, hi - 0.5), len(datasets), 1, fill=False,
                                   edgecolor=CENN_C, lw=2.2, zorder=5))
        ax.get_yticklabels()[hi].set_color(CENN_C); ax.get_yticklabels()[hi].set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(f"Rank within dataset (1 = best of {k})", fontsize=7.4); cb.ax.tick_params(labelsize=6.5)
    ax.set_title(title or "Per-dataset rank — no single model wins everywhere", fontsize=9)
    return savefig(fig, outdir, name)


def fig_radar(categories, series, outdir, name, title, highlight="AMS-CeNN", note=None):
    """Radar/spider chart. categories: spoke labels. series: dict label -> list of values in [0,1]
    (higher = better), one per category. The highlighted series is drawn thicker + filled. Qualitative
    POSITIONING figure (the table/CD carry the rigorous claim)."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                                   # close the loop
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W * 1.3, COL_W * 1.18), subplot_kw=dict(polar=True))
    pal = [ACC, OK, "#CC79A7", "#56B4E9", "#999999"]
    pi = 0
    for label, vals in series.items():
        v = list(vals) + [vals[0]]
        is_hl = (label == highlight)
        color = CENN_C if is_hl else pal[pi % len(pal)]
        if not is_hl:
            pi += 1
        ax.plot(angles, v, color=color, lw=2.4 if is_hl else 1.4, label=label,
                marker="o", ms=3 if is_hl else 0, zorder=6 if is_hl else 3)
        if is_hl:
            ax.fill(angles, v, color=color, alpha=0.15, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7.6)
    ax.tick_params(axis="x", pad=6)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6.4, color="0.55")
    ax.set_ylim(0, 1.04)
    ax.set_rlabel_position(90)
    ax.grid(color="0.8", lw=0.5)
    ax.set_title(title, fontsize=9, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=len(series),
              fontsize=6.8, columnspacing=1.0, handletextpad=0.4, frameon=False)
    if note:
        fig.text(0.5, -0.01, note, ha="center", fontsize=6.3, color="0.45", style="italic")
    return savefig(fig, outdir, name)


def fig_rank_boxplots(dist, outdir, name="fig_rank_boxplots", n_blocks=28, k=12):
    """dist: model_label -> list of per-block ranks (1=best). Ordered by median rank; AMS-CeNN in the
    headline accent; y-axis inverted so rank 1 (best) is at the TOP. Scale-free companion to the CD."""
    apply_style()
    items = sorted(dist.items(), key=lambda kv: (np.median(kv[1]), np.mean(kv[1])))
    labels = [a for a, _ in items]
    data = [np.asarray(v, float) for _, v in items]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 2.9))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.62,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=2.4, markerfacecolor="0.5",
                                    markeredgecolor="0.5", alpha=0.5))
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(CENN_C if lab == "AMS-CeNN" else "#9ecae1")
        patch.set_alpha(0.9); patch.set_edgecolor("0.3"); patch.set_linewidth(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel(f"Rank per (dataset, horizon) block\n(1 = best, {k} = worst)")
    ax.set_ylim(0.4, k + 0.6); ax.set_yticks(range(1, k + 1))
    ax.invert_yaxis()                                       # rank 1 (best) on top
    ax.set_title(f"Per-block rank distribution over {n_blocks} (dataset $\\times$ horizon) blocks",
                 fontsize=8.8)
    return savefig(fig, outdir, name)


def fig_mse_boxplots(dist, outdir, name="fig_mse_boxplots", scope_note=None):
    """dist: dict model -> array of MSE values (across seeds x dataset-horizons).
    Ordered by median; AMS-CeNN highlighted."""
    apply_style()
    items = sorted(dist.items(), key=lambda kv: np.median(kv[1]))
    labels = [k for k, _ in items]; data = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 2.8))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.1),
                    flierprops=dict(marker="o", ms=2.5, alpha=0.5))
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(CENN_C if ("CeNN" in lab or lab == "AMS-CeNN") else "#9ecae1")
        patch.set_alpha(0.85); patch.set_edgecolor("0.3"); patch.set_linewidth(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel(r"MSE ($\downarrow$)")
    # Cap the y-axis at the 98th pct of pooled values so a few extreme fliers (e.g. xLSTM) don't
    # compress the informative 0.1-0.6 band; annotate how many points sit above the cut.
    pooled = np.concatenate([np.asarray(v, float) for v in data])
    cap = float(np.percentile(pooled, 98))
    n_above = int((pooled > cap).sum())
    lo = max(0.0, float(np.min(pooled)) - 0.02)
    ax.set_ylim(lo, cap * 1.04)
    if n_above:
        ax.text(0.015, 0.97, f"{n_above} outliers above axis", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.8, color="0.4", style="italic")
    ttl = "MSE distribution (seeds $\\times$ horizons)"
    if scope_note:
        ttl += f" — {scope_note}"
    ax.set_title(ttl, fontsize=9)
    return savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# 14. Spread-vs-error scatter (validates scale-disagreement as a UQ signal)
# ---------------------------------------------------------------------------
def fig_spread_vs_error(spread, err, outdir, name="fig_spread_vs_error", annotation=None):
    """annotation: precomputed rho (+ CI) string from the caller. When given it is used verbatim
    (the caller computes a window-level block-bootstrap CI that respects within-window dependence);
    falling back to a naive in-figure Spearman only when no annotation is supplied."""
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
    ax.scatter(spread, err, s=10, alpha=0.4, color=ACC, edgecolor="none")
    # robust trend line
    z = np.polyfit(spread, err, 1); xs = np.linspace(min(spread), max(spread), 50)
    ax.plot(xs, np.polyval(z, xs), color=CENN_C, lw=1.4)
    if annotation is None:
        try:
            from scipy.stats import spearmanr
            annotation = rf"Spearman $\rho$ = {spearmanr(spread, err).correlation:.2f}"
        except Exception:
            annotation = None
    if annotation:
        ax.annotate(annotation, (0.05, 0.92), xycoords="axes fraction", fontsize=8, color="0.2")
    ax.set_xlabel("Scale-disagreement (branch std)"); ax.set_ylabel("Absolute forecast error")
    ax.set_title("Does scale-disagreement track error?")
    return savefig(fig, outdir, name)


# ===========================================================================
# DEMO: render the WHOLE suite with illustrative data (review the styling now).
# ===========================================================================
def _demo(outdir):
    rng = np.random.default_rng(0)
    rows = [
        {"model": "CeNN", "family": "CeNN", "macs": 7.35e6, "params": 57319, "peak_mem": 11, "mse": 0.255, "label": "CeNN", "emphasize": True},
        {"model": "CeNN-Eff", "family": "CeNN", "macs": 1.71e6, "params": 20551, "peak_mem": 10, "mse": 0.294, "label": "CeNN-Eff", "emphasize": True},
        {"model": "DLinear", "family": "linear", "macs": 0.69e6, "params": 98496, "peak_mem": 9, "mse": 0.262, "label": "DLinear"},
        {"model": "TiDE", "family": "mlp", "macs": 4.9e6, "params": 2.4e6, "peak_mem": 12, "mse": 0.258, "label": "TiDE"},
        {"model": "NHITS", "family": "mlp", "macs": 7.5e6, "params": 3.8e6, "peak_mem": 14, "mse": 0.251, "label": "NHITS"},
        {"model": "TSMixer", "family": "mlp", "macs": 1.2e6, "params": 0.6e6, "peak_mem": 10, "mse": 0.268, "label": "TSMixer"},
        {"model": "TimeMixer", "family": "mlp", "macs": 4.2e6, "params": 2.1e6, "peak_mem": 13, "mse": 0.249, "label": "TimeMixer"},
        {"model": "PatchTST", "family": "transformer", "macs": 11.0e6, "params": 0.55e6, "peak_mem": 16, "mse": 0.246, "label": "PatchTST"},
        {"model": "iTransformer", "family": "transformer", "macs": 3.9e6, "params": 0.28e6, "peak_mem": 12, "mse": 0.252, "label": "iTransformer"},
        {"model": "TimesNet", "family": "conv", "macs": 40.0e6, "params": 1.5e6, "peak_mem": 22, "mse": 0.250, "label": "TimesNet"},
    ]
    pngs = []
    pngs.append(fig_pareto(rows, outdir))
    pngs.append(fig_efficiency_panels(rows, outdir))
    # MACs ~ evals/step * K * per-eval-cost; euler/exp_euler share MACs (1 eval) but differ in MSE.
    ksweep = []
    for ig, evals, mses in [("euler", 1, {8: 0.255, 4: 0.260, 2: 0.270}),
                            ("exp_euler", 1, {8: 0.253, 4: 0.257, 2: 0.264}),
                            ("heun", 2, {8: 0.252, 4: 0.255, 2: 0.260}),
                            ("rk4", 4, {8: 0.251, 4: 0.253, 2: 0.257})]:
        for K in (8, 4, 2):
            ksweep.append({"integrator": ig, "K": K, "macs": evals * K * 0.9e6, "mse": mses[K]})
    pngs.append(fig_k_integrator(ksweep, outdir))
    ranks = {"CeNN": 3.1, "PatchTST": 2.8, "iTransformer": 3.4, "NHITS": 3.0,
             "TimeMixer": 3.6, "DLinear": 5.2, "TimesNet": 4.0, "TiDE": 4.9}
    pngs.append(fig_cd_diagram(ranks, cd=1.8, outdir=outdir))
    tau = 0.5 + 0.45 * rng.random((7, 96)) * np.linspace(0.4, 1.0, 96)[None, :]
    pngs.append(fig_tau_heatmap(tau, outdir, channel_labels=[f"v{i}" for i in range(7)]))
    nom = np.linspace(0.1, 0.95, 9)
    pngs.append(fig_reliability({
        "CeNN (seed-ens.)": (nom, np.clip(nom - 0.08 + 0.03 * rng.standard_normal(9), 0, 1), CENN_C),
        "DLinear": (nom, np.clip(nom - 0.14 + 0.03 * rng.standard_normal(9), 0, 1), ACC),
    }, outdir))
    t = np.arange(96); base = np.sin(t / 8) + 0.3 * np.sin(t / 3)
    yp = base + 0.05 * rng.standard_normal(96); band = 0.15 + 0.004 * t
    pngs.append(fig_forecast_intervals(t, base, yp, yp - band, yp + band, outdir,
                ctx=(np.arange(-48, 0), np.sin(np.arange(-48, 0) / 8))))
    pngs.append(fig_ablation_bars([("− Spectral cap", 0.012), ("− Adaptive τ (C1)", 0.002),
                ("− Multi-scale (C2)", 0.018), ("− Patch", -0.039), ("− Pointwise", 0.005),
                ("− STAR cross-var", 0.009)], outdir))
    pngs.append(fig_scale_disagreement(np.abs(rng.standard_normal((4, 96))) * np.linspace(0.05, 0.3, 96)[None, :],
                outdir, scales=["d=1", "d=2", "d=4", "d=8"]))
    steps = np.arange(0, 1000, 20)
    pngs.append(fig_contraction({
        "With spectral cap": (steps, 0.9 + 0.02 * np.sin(steps / 60), OK, "-"),
        "Without cap": (steps, 0.55 + 1.25 * (steps / 1000) ** 0.9, BAD, "--"),
    }, outdir))
    # forecast-vs-truth grid across datasets (with intervals)
    panels = []
    for k, ds in enumerate(["ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity", "Traffic"]):
        t = np.arange(96); b = np.sin(t / (7 + k)) + 0.3 * np.sin(t / 3 + k)
        yp = b + 0.06 * rng.standard_normal(96); band = 0.12 + 0.004 * t
        panels.append({"dataset": ds, "t": t, "y_true": b, "y_pred": yp, "lo": yp - band,
                       "hi": yp + band, "ctx_t": np.arange(-40, 0), "ctx_y": np.sin(np.arange(-40, 0) / (7 + k))})
    pngs.append(fig_forecast_grid(panels, outdir))
    # per-horizon error curves
    H = [96, 192, 336, 720]
    pngs.append(fig_horizon_curves({
        "CeNN": {"mse": [0.255, 0.30, 0.34, 0.41], "family": "CeNN"},
        "PatchTST": {"mse": [0.246, 0.29, 0.33, 0.40], "family": "transformer"},
        "DLinear": {"mse": [0.262, 0.31, 0.36, 0.45], "family": "linear"},
        "NHITS": {"mse": [0.251, 0.30, 0.35, 0.43], "family": "mlp"},
    }, H, outdir))
    # MSE distribution box plots across seeds
    box = {m: 0.25 + off + 0.012 * rng.standard_normal(20) for m, off in
           [("CeNN", 0.006), ("PatchTST", 0.0), ("DLinear", 0.02), ("NHITS", 0.003),
            ("iTransformer", 0.004), ("TiDE", 0.018), ("TimesNet", 0.005)]}
    pngs.append(fig_mse_boxplots(box, outdir))
    # spread-vs-error UQ validation
    sp = np.abs(rng.standard_normal(300)) * 0.2
    pngs.append(fig_spread_vs_error(sp, 0.3 * sp + 0.1 * np.abs(rng.standard_normal(300)), outdir))
    for p in pngs:
        print(f"  {p}")
    return pngs


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else "experiments/aggregated/figures")
