"""
ieee_eval_metrics_aligned.py
===========================
Final polished version with perfectly aligned grouped bars (no hatches).
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# rcParams
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["High", "Medium", "Low"]

y_true = np.array([
    "High","High","High","Medium","High","High","Medium","High",
    "High","Medium","High","Low","High","High","Medium"
])

y_pred = np.array([
    "High","High","Medium","Medium","High","High","High","Medium",
    "High","Medium","High","Low","High","High","High"
])

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, labels=CLASS_NAMES, average=None)
recall = recall_score(y_true, y_pred, labels=CLASS_NAMES, average=None)
f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average=None)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
COL_W = 3.487
HEIGHT = COL_W / 1.618
fig, ax = plt.subplots(figsize=(COL_W, HEIGHT))

# ─────────────────────────────────────────────────────────────────────────────
# PERFECT BAR ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
n_classes = len(CLASS_NAMES)
n_metrics = 3

bar_w = 0.18
inner_gap = 0.04   # space between bars INSIDE group
group_gap = 0.25   # space BETWEEN groups

group_width = n_metrics * bar_w + (n_metrics - 1) * inner_gap

x_centers = np.arange(n_classes) * (group_width + group_gap)

colors = ["#4c72b0", "#55a868", "#c44e52"]  # soft Nature-style
labels = ["Precision", "Recall", "F1"]
metrics = [precision, recall, f1]

for i, (vals, color, label) in enumerate(zip(metrics, colors, labels)):
    offsets = x_centers - group_width/2 + i * (bar_w + inner_gap)

    bars = ax.bar(
        offsets,
        vals,
        width=bar_w,
        color=color,
        edgecolor="#1a1a1a",
        linewidth=0.4,
        label=label,
        zorder=3,
    )

    # value labels
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=5.5,
        )

# Accuracy line
ax.axhline(
    y=accuracy,
    color="#333333",
    linestyle="--",
    linewidth=0.8,
    zorder=2,
    label=f"Accuracy ({accuracy:.2f})"
)

# Axes
ax.set_xticks(x_centers)
ax.set_xticklabels(CLASS_NAMES)

ax.set_ylabel("Score", labelpad=3)
ax.set_ylim(0, 1.15)

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Legend
ax.legend(loc="lower right", frameon=False)

# Layout
fig.subplots_adjust(left=0.16, bottom=0.18, right=0.97, top=0.95)

# Save
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/metrics_bar_aligned.pdf")
fig.savefig("figures/metrics_bar_aligned.png", dpi=300)

plt.close(fig)

print("Saved: figures/metrics_bar_aligned.pdf + PNG")
