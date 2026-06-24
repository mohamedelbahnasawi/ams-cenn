"""Regenerate the tau-retention figure across ALL 7 datasets with the CORRECT quantity.

Bug being fixed: the artifact stores tau = 1 - alpha (drive weight, model.py L247); the old
figure plotted that ~0.10 value but labelled it "Retention alpha". RETENTION is alpha = 1 - tau ~= 0.90.

Data: ETT+Weather = headline C1C2-Skip-K2 (regenerated on the 4090); ECL/Traffic = C1C2-Ensemble
(headline minus the skip => identical C1 gate; retention is variant-independent, verified).
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ANALYSIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYSIS)
from plotting import apply_style, savefig, COL_W

TAU = os.environ.get("TAU_DIR", os.path.join(os.path.dirname(ANALYSIS), "_tau_scratch", "artifacts", "tau"))
OUT = os.environ.get("FIG_OUT_DIR", os.path.join(ANALYSIS, "figures"))
SEVEN = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"]
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#666666"]

pat = re.compile(r"CeNN_(.+?)__(.+?)__H(\d+)__seed(\w+)\.npz")
byds = defaultdict(list)
for f in sorted(glob.glob(os.path.join(TAU, "*.npz"))):
    m = pat.match(os.path.basename(f))
    if m:
        byds[m.group(2)].append(f)

profiles = {}
for ds in SEVEN:
    fs = byds.get(ds, [])
    if not fs:
        continue
    chan = []
    for f in fs:
        d = np.load(f)
        cells = [d[k] for k in d.files if d[k].dtype.kind == "f" and d[k].ndim >= 2]
        tau_c = np.mean([c.reshape(c.shape[0], -1).mean(axis=1) for c in cells], axis=0)  # tau per channel
        chan.append(1.0 - tau_c)                                                          # retention
    profiles[ds] = np.sort(np.mean(np.stack(chan), axis=0))                               # seed-mean, sorted

apply_style()
fig, ax = plt.subplots(figsize=(COL_W, 2.7))
for i, ds in enumerate(SEVEN):
    if ds not in profiles:
        continue
    p = profiles[ds]
    ax.plot(np.arange(len(p)), p, lw=1.1, color=COLORS[i], label=f"{ds} ({p.mean():.2f})")
ax.set_xlabel("Latent channel (sorted by retention)")
ax.set_ylabel(r"Retention $\alpha = 1-\tau$")
ax.set_title("Learned per-channel retention (AMS-CeNN)", fontsize=8.6)
ax.legend(loc="lower right", fontsize=5.8, ncol=2, handletextpad=0.4, columnspacing=0.8, framealpha=0.92)
ax.margins(x=0.02)
allmean = float(np.mean([profiles[d].mean() for d in profiles]))
savefig(fig, OUT, "fig05_tau_retention_v2")
print("saved fig05_tau_retention_v2 ; per-dataset mean retention alpha:")
for ds in SEVEN:
    if ds in profiles:
        print("  %-12s alpha=%.4f  (channels=%d)" % (ds, float(profiles[ds].mean()), len(profiles[ds])))
print("overall mean retention alpha = %.4f" % allmean)
