"""Figure 4 of the paper (Section VI-D): state-conditional action spread sigma_10 per task and dataset quality,
from results/action_spread.json (k = 10, sample seed 0; omiga_rev/diagnostics/spread.py). Style as make_fig_dataset.py.
Run: python analysis/make_fig_spread.py   (writes paper/figures/action_spread.png)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(REPO, "paper/figures")
TASKS = [("Hopper", "Hopper-v2-3x1"), ("HalfCheetah", "HalfCheetah-v2-6x1"), ("Ant", "Ant-v2-2x4")]
QUAL = ["expert", "medium-expert", "medium", "medium-replay"]
QLAB = ["Expert", "Medium-expert", "Medium", "Medium-replay"]
DS_COL = ["#5C1029", "#8C1D40", "#C0607A", "#E8B9C5"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI, COLW = 600, 3.5

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.5, "font.family": "DejaVu Sans",
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})

spread = {c: v["0"]["10"] for c, v in json.load(open(f"{REPO}/results/action_spread.json"))["spread"].items()}
cells = [f"{p}-{q}" for _, p in TASKS for q in QUAL]
vals = {"spread": [spread[c] for c in cells]}

W, GAP = 0.19, 0.3                                   # bar width, gap between task groups (in bar units of the x axis)
x = np.array([g * (4 * W + GAP) + k * W for g in range(3) for k in range(4)])
centers = [np.mean(x[4 * g:4 * g + 4]) for g in range(3)]

fig, ax = plt.subplots(1, 1, figsize=(COLW, 1.75))
v = np.array(vals["spread"])
ax.bar(x, v, width=W, color=[DS_COL[k % 4] for k in range(12)], edgecolor="white", linewidth=0.3)
ax.set_xticks(centers); ax.set_xticklabels([t for t, _ in TASKS])
ax.set_xlim(x[0] - 2.2 * W, x[-1] + 2.2 * W); ax.set_ylim(0, 0.45)
ax.set_ylabel(r"Action spread $\sigma_{10}$", labelpad=2); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DS_COL]
fig.legend(h, QLAB, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, -0.015),
           handlelength=1.0, handletextpad=0.4, columnspacing=0.9)
fig.subplots_adjust(left=0.135, right=0.995, top=0.97, bottom=0.30)
fig.savefig(f"{OUT}/action_spread.png", dpi=DPI)
print("wrote", OUT)
