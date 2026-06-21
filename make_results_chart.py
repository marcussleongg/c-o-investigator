"""Grouped bar chart of the README Results table (mean reward by hop tier).

Run:  uv run --with matplotlib python make_results_chart.py
Writes results_chart.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Copied straight from README.md Results table (mean reward, [0, 1]).
tiers = ["neg", "1-hop", "2-hop", "3-hop", "Overall"]
base = [1.000, 0.500, 0.000, 0.000, 0.500]
trained = [0.833, 0.438, 0.000, 0.000, 0.425]
claude = [1.000, 0.469, 0.188, 0.000, 0.525]

x = np.arange(len(tiers))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w, base, w, label="Base Qwen3-8B", color="#8b949e")
b2 = ax.bar(x, trained, w, label="Trained Qwen3-8B", color="#2ea043")
b3 = ax.bar(x + w, claude, w, label="Claude", color="#2dd4bf")

ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_ylabel("Mean reward [0–1]")
ax.set_ylim(0, 1.08)
ax.set_title("COI eval — mean reward by hop tier (small eval set, n=20)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

fig.tight_layout()
fig.savefig("results_chart.png", dpi=150)
print("wrote results_chart.png")
