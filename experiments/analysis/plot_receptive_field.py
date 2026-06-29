"""K=2 receptive-field diagram (Paper A, R6 polish).

Shows, for the SHIPPED K=2 setting (kernel k=3, dilations {1,2,4,8}), that the nonlinear CeNN
branches are strictly LOCAL (RF = K*(k-1)*d + 1 -> 5,9,17,33 steps) while the linear temporal head
and the zero-init skip each span the FULL L=512 lookback. Visually supports the honest framing:
C2 is multi-resolution nonlinear refinement; the long-range structure is carried by the linear path.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"aggregated/figures/fig10_receptive_field.pdf"
K, k, L = 2, 3, 512
dils = [1, 2, 4, 8]
rf = {d: K * (k - 1) * d + 1 for d in dils}   # 5, 9, 17, 33

rows = [("Linear head / skip  ($L\\!\\to\\!H$)", L, "#E8772E")]
for d in reversed(dils):
    rows.append((f"CeNN branch  $d{{=}}{d}$", rf[d], "#4C72B0"))

fig, ax = plt.subplots(figsize=(7.0, 2.7))
y = range(len(rows))
for i, (lab, span, col) in zip(y, rows):
    ax.barh(i, span, color=col, height=0.62, zorder=3)
    ax.text(span + 6, i, f"{span} steps", va="center", fontsize=8.5)
ax.set_yticks(list(y))
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, L * 1.32)  # headroom so the "512 steps" label + "full lookback" note stay inside the axes
ax.set_xlabel("effective receptive field over the $L=512$ lookback (steps)", fontsize=9)
ax.axvline(L, color="0.6", lw=0.8, ls=":")
ax.text(L, len(rows) - 0.4, "  full lookback $L=512$", fontsize=7.5, color="0.4", va="top")
ax.set_title("Effective receptive field at the shipped $K{=}2$ setting", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT, "| branch RFs:", rf)
