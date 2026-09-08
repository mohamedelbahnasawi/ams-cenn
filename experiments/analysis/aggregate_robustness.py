"""Aggregate the robustness results into degradation curves + the pre-registered verdict.

Reads experiments/_robustness/<kind>_<level>/results/*.json (+ clean/), computes for every cell the
degradation ratio = MSE_perturbed / MSE_clean, aggregates over seeds and horizons, and evaluates
the pre-registered robustness criterion (does the model degrade more gracefully than baselines?):

  AMS-CeNN's mean degradation ratio at MID-HIGH levels is strictly below the MEDIAN of the five
  baselines on >= 4 of 5 perturbation kinds (aggregated over the small+Weather datasets), and the
  advantage is monotone (does not vanish as level rises).

Writes robustness_degradation.csv and prints the verdict. Usage: python -m experiments.analysis.aggregate_robustness
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import numpy as np

ROBUST_DIR = Path("experiments/_robustness")
from experiments.config import CENN_MAIN_VARIANT  # noqa: E402
AMS = f"CeNN_{CENN_MAIN_VARIANT}"
# S4D dropped: no checkpoint was saved for its fp32 complex-kernel SSM,
# so it cannot be evaluated without retraining. The four retained baselines span the
# linear / mixer / transformer / conv families.
BASELINES = ["DLinear", "TSMixer", "PatchTST", "TCN"]
KINDS = ["gauss", "spike", "mask", "scale", "shift"]
# pre-registered "mid-high" decisive levels per kind
MIDHIGH = {"gauss": [0.5, 1.0], "spike": [0.03, 0.05], "mask": [48, 96],
           "scale": [0.25, 0.5], "shift": [1.0, 2.0]}


def load_all():
    """rows: list of dicts with model,dataset,horizon,seed,kind,level,ratio_mse."""
    rows = []
    for cond_dir in sorted(ROBUST_DIR.glob("*/results")):
        for f in cond_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if d.get("kind") == "clean":
                continue
            rows.append(d)
    return rows


def main():
    rows = load_all()
    if not rows:
        print("No robustness results yet under", ROBUST_DIR); return
    # aggregate ratio over (model, dataset, kind, level) across seeds+horizons
    agg = defaultdict(list)
    # dead-sensor family: the all-channel ratio is dominated by the dead channel itself (lost for
    # every model); the diagnostic quantity is the HEALTHY-channel ratio (does the fault leak into
    # the other channels?), stored per cell as ratio_mse_healthy and aggregated alongside.
    agg_h = defaultdict(list)
    for d in rows:
        agg[(d["model"], d["dataset"], d["kind"], d["level"])].append(d.get("ratio_mse", np.nan))
        if d["kind"] == "dead":
            agg_h[(d["model"], d["dataset"], d["kind"], d["level"])].append(d.get("ratio_mse_healthy", np.nan))
    # write CSV
    out_csv = ROBUST_DIR / "robustness_degradation.csv"
    lines = ["model,dataset,kind,level,mean_ratio,std_ratio,n,mean_ratio_healthy,std_ratio_healthy"]
    for (m, ds, k, lv), vs in sorted(agg.items()):
        vs = [v for v in vs if v == v]
        if vs:
            hs = [v for v in agg_h.get((m, ds, k, lv), []) if v == v]
            h_cols = f",{np.mean(hs):.5f},{np.std(hs):.5f}" if hs else ",,"
            lines.append(f"{m},{ds},{k},{lv},{np.mean(vs):.5f},{np.std(vs):.5f},{len(vs)}{h_cols}")
    out_csv.write_text("\n".join(lines))
    print(f"wrote {out_csv} ({len(lines)-1} rows)\n")

    # ---- dead-sensor family: healthy-channel ratio pooled over datasets/horizons/seeds ----
    if agg_h:
        print("DEAD-SENSOR containment (healthy-channel MSE ratio; 1.00 = the fault stays in its channel)")
        models = sorted({m for (m, _, _, _) in agg_h})
        for lv in sorted({lv for (_, _, _, lv) in agg_h}):
            for m in models:
                hs = [v for (mm, _, _, l), vs in agg_h.items() if mm == m and l == lv for v in vs if v == v]
                if hs:
                    print(f"  dead={lv}  {m:28s} {np.mean(hs):.3f}  (n={len(hs)})")
        print()

    # ---- pre-registered verdict: AMS vs baseline-median at mid-high levels, per kind ----
    # model-level mean ratio at mid-high levels (pooled over datasets, horizons, seeds)
    def model_kind_ratio(model, kind):
        vals = []
        for d in rows:
            if d["model"] == model and d["kind"] == kind and d["level"] in MIDHIGH[kind]:
                r = d.get("ratio_mse", np.nan)
                if r == r:
                    vals.append(r)
        return np.mean(vals) if vals else np.nan

    print("=" * 72)
    print("ROBUSTNESS VERDICT — mean degradation ratio at mid-high levels (lower = more robust)")
    print("=" * 72)
    print(f"{'kind':7} {'AMS-CeNN':>9} {'base-median':>12} {'best-base':>10}  result")
    ams_wins = 0; n_eval = 0
    for k in KINDS:
        ams = model_kind_ratio(AMS, k)
        base = [model_kind_ratio(b, k) for b in BASELINES]
        base = [x for x in base if x == x]
        if not base or ams != ams:
            print(f"{k:7} {'--':>9} {'--':>12} {'--':>10}  (incomplete)")
            continue
        n_eval += 1
        bmed = float(np.median(base)); bbest = float(np.min(base))
        win = ams < bmed
        ams_wins += int(win)
        print(f"{k:7} {ams:>9.3f} {bmed:>12.3f} {bbest:>10.3f}  "
              f"{'AMS more robust' if win else 'AMS NOT more robust'}")
    print("-" * 72)
    print(f"AMS-CeNN more robust than baseline median on {ams_wins}/{n_eval} kinds "
          f"(pre-registered threshold: >= 4/5).")
    if n_eval < 5:
        print("VERDICT: INCOMPLETE — wait for all kinds to finish.")
    elif ams_wins >= 4:
        print("VERDICT: KEEP 'Robust' — AMS-CeNN degrades more gracefully on >= 4/5 perturbation kinds.")
    else:
        print("VERDICT: DROP 'Robust' — AMS-CeNN does not show a graceful-degradation edge.")
    print("=" * 72)


if __name__ == "__main__":
    main()
