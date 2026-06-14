"""
ieee_loss_curve.py
==================
IEEE-format training loss curve for the BioChat AI paper.
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# 0. rcParams (IEEE style)
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    "lines.linewidth": 1.2,
    "lines.markersize": 4,

    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,

    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.7,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────────────────────────────
steps = np.array([10, 20, 30, 40, 50])
loss  = np.array([3.0217, 2.9191, 2.8321, 2.6714, 2.6837])

steps_fine = np.linspace(steps[0], steps[-1], 500)
loss_fine  = np.interp(steps_fine, steps, loss)

min_idx = int(np.argmin(loss))
min_step = int(steps[min_idx])
min_loss = float(loss[min_idx])
initial_loss = float(loss[0])
improvement_pct = (initial_loss - min_loss) / initial_loss * 100

# ─────────────────────────────────────────────────────────────────────────────
# 2. Figure
# ─────────────────────────────────────────────────────────────────────────────
COL_W = 3.487
HEIGHT = COL_W / 1.618

fig, ax = plt.subplots(figsize=(COL_W, HEIGHT))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fill
# ─────────────────────────────────────────────────────────────────────────────
ax.fill_between(
    steps_fine, loss_fine, loss_fine.min() - 0.015,
    color="#1f6eb5", alpha=0.07, zorder=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Curve
# ─────────────────────────────────────────────────────────────────────────────
ax.plot(
    steps_fine, loss_fine,
    color="#1f6eb5",
    zorder=3,
    label="Training loss",
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Markers
# ─────────────────────────────────────────────────────────────────────────────
ax.plot(
    steps, loss,
    marker="o",
    linestyle="none",
    color="#1f6eb5",
    markerfacecolor="#ffffff",
    markeredgewidth=0.9,
    zorder=4,
    label="Checkpoints",
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Phase separator
# ─────────────────────────────────────────────────────────────────────────────
PHASE_BOUNDARY = 25
ax.axvline(
    x=PHASE_BOUNDARY,
    color="#888888",
    linewidth=0.6,
    linestyle=(0, (1, 2)),
    zorder=2,
)

# Phase labels (aligned using axes coords)
ax.text(
    0.27, 0.93, "Rapid\ndescent",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=6, style="italic",
    color="#555555", linespacing=1.2
)

ax.text(
    0.73, 0.93, "Convergence",
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=6, style="italic",
    color="#555555"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Annotations (FIXED ALIGNMENT)
# ─────────────────────────────────────────────────────────────────────────────
_arrow_kw = dict(
    arrowstyle="->,head_length=0.12,head_width=0.07",
    color="#444444",
    lw=0.6,
    shrinkA=0,
    shrinkB=2,
)

# Initial loss
ax.annotate(
    f"$L_0={initial_loss:.4f}$",
    xy=(steps[0], initial_loss),
    xytext=(0.28, 0.82),
    textcoords="axes fraction",
    fontsize=6.5,
    color="#1a1a1a",
    arrowprops=_arrow_kw,
    zorder=5,
)

# Minimum loss
ax.annotate(
    f"$L_{{\\mathrm{{min}}}}={min_loss:.4f}$",
    xy=(min_step, min_loss),
    xytext=(0.55, 0.25),
    textcoords="axes fraction",
    fontsize=6.5,
    color="#1a1a1a",
    arrowprops=_arrow_kw,
    zorder=5,
)

# ΔL box (aligned)
ax.text(
    0.80, 0.55,
    f"$\\Delta L={improvement_pct:.1f}\\%$",
    transform=ax.transAxes,
    ha="center", va="center",
    fontsize=6.5,
    bbox=dict(
        boxstyle="square,pad=0.25",
        facecolor="#f5f5f5",
        edgecolor="#aaaaaa",
        linewidth=0.5,
    ),
    zorder=5,
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Axes
# ─────────────────────────────────────────────────────────────────────────────
ax.set_xlabel("Training Step", labelpad=3)
ax.set_ylabel("Cross-Entropy Loss", labelpad=3)

ax.set_xlim(6, 54)
ax.set_ylim(2.60, 3.10)

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Legend (aligned)
# ─────────────────────────────────────────────────────────────────────────────
leg = ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    framealpha=0.90,
    edgecolor="#cccccc",
    borderpad=0.5,
    handletextpad=0.5,
    labelspacing=0.4,
)
leg.get_frame().set_linewidth(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Layout (fixed margins)
# ─────────────────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.165, bottom=0.18, right=0.97, top=0.93)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

fig.savefig("figures/loss_curve_ieee.pdf")
fig.savefig("figures/loss_curve_ieee.png", dpi=300)

plt.close(fig)

print("Saved:")
print("  figures/loss_curve_ieee.pdf")
print("  figures/loss_curve_ieee.png")
print(f"\nStats: L0={initial_loss:.4f}  Lmin={min_loss:.4f}  "
      f"@step {min_step}  DeltaL={improvement_pct:.1f}%")