"""Receptive-field diagram of the shipped AMS-CeNN (K=2, template size 3 with the future tap masked,
dilation rates {1,2,4,8}).

Each cellular branch reads the current step and ONE past step at distance d per micro-step, so after
K micro-steps it has seen K*d + 1 steps (3, 5, 9, 17 at K=2; verified on the trained model with an
impulse test), whereas the temporal projection and the linear residual each span the full L=512
lookback. The x axis is logarithmic so that the short cellular reaches and the 512-step linear maps
are legible on one plot.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.config import FIGURES_DIR  # noqa: E402

OUT_DIRS = [str(FIGURES_DIR)]
if os.environ.get("FIG_OUT_DIR"):
    OUT_DIRS.append(os.environ["FIG_OUT_DIR"])
K, L = 2, 512
dils = [1, 2, 4, 8]
rf = {d: K * d + 1 for d in dils}   # 3, 5, 9, 17

rows = [("temporal projection", L, "#E8772E"), ("linear residual", L, "#E8772E")]
for d in reversed(dils):
    rows.append((f"cell branch, $d={d}$", rf[d], "#4C72B0"))

fig, ax = plt.subplots(figsize=(6.6, 2.9))
y = list(range(len(rows)))
for i, (lab, span, col) in zip(y, rows):
    ax.barh(i, span, left=1, color=col, height=0.62, zorder=3)
    ax.text(span * 1.12, i, f"{span} steps", va="center", ha="left", fontsize=10, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.invert_yaxis()
ax.set_xscale("log", base=2)
ax.set_xlim(1, L * 3.2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels([str(v) for v in [2, 4, 8, 16, 32, 64, 128, 256, 512]], fontsize=10)
ax.set_xlabel("steps of the $L=512$ window seen by the component (log scale)", fontsize=10)
ax.grid(axis="x", alpha=0.3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "fig10_receptive_field.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)
print("branch reaches:", rf)
