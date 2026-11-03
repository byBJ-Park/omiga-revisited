"""Figures 1 and 2 of the paper (Section IV): the episode-return distribution of each dataset (dataset_returns) and
the out-of-bound action fraction per joint (dataset_actions), from the released hdf5 files in data/.
Drawn at one IEEE column width (3.5 in) with 7-8 pt text; panel labels "(a) Hopper" etc. under each panel. Returns:
median dot with the interquartile bar. Actions: the boundary row of each episode (d = 1, the final observation) is
excluded; only Hopper and HalfCheetah are drawn (no Ant action lies outside [-1, 1]; its numbers are in
results/dataset_summary.json, which this script also writes together with every number quoted in Section IV).
Run: python analysis/make_fig_dataset.py   (writes paper/figures/dataset_returns.png, dataset_actions.png)
"""
import json, os
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(REPO, "paper/figures")
QUAL = ["expert", "medium-expert", "medium", "medium-replay"]
QLAB = ["Expert", "Medium-expert", "Medium", "Medium-replay"]
VIOLIN_FILL, VIOLIN_EDGE, BAR_COLOR = "#E2D6DA", "#8A6A74", "#3A2A2F"   # returns figure: warm grey with a burgundy tint, desaturated burgundy outline, IQR bar
DS_COL = ["#5C1029", "#8C1D40", "#C0607A", "#E8B9C5"]       # actions figure: E, M-E, M, M-R, burgundy from dark to light
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TASKS = [("Hopper", "3x1", ["Thigh", "Leg", "Foot"]),
         ("HalfCheetah", "6x1", ["Back\nthigh", "Back\nshin", "Back\nfoot", "Front\nthigh", "Front\nshin", "Front\nfoot"]),
         ("Ant", "2x4", ["Agent 0\n(legs 4, 1)", "Agent 1\n(legs 2, 3)"])]
DPI = 600
COLW = 3.5                                                   # IEEE single-column width (in); include at \\columnwidth

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.5, "font.family": "DejaVu Sans",
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})


def style(ax, label):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel(label, fontsize=8, color=INK, labelpad=3)        # "(a) Hopper" under the panel, centred


data = {}
for task, conf, joints in TASKS:
    rets, oob, amax = [], [], []
    for q in QUAL:
        with h5py.File(f"{REPO}/data/{task}-v2-{conf}-{q}.hdf5", "r") as f:
            a = f["a"][:]; r = f["r"][:, 0]; d = f["d"][:, 0]
        ends = np.where(d > 0.5)[0]; starts = np.r_[0, ends[:-1] + 1]
        rets.append(np.array([r[s:e + 1].sum() for s, e in zip(starts, ends)]))     # boundary-row reward is 0
        a = a[d < 0.5]                                                               # drop the final-observation rows
        oob.append([100 * float(np.mean(np.abs(a[:, i]) > 1)) for i in range(a.shape[1])])
        amax.append(float(np.abs(a).max()))
    data[task] = (rets, oob, amax)

# Figure 1: episode returns, one shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(COLW, 1.95), sharey=True)
for p, (ax, (task, conf, joints)) in enumerate(zip(axes, TASKS)):
    rets = data[task][0]
    vp = ax.violinplot(rets, positions=range(4), widths=0.8, showextrema=False)
    for b in vp["bodies"]: b.set_facecolor(VIOLIN_FILL); b.set_alpha(1.0); b.set_edgecolor(VIOLIN_EDGE); b.set_linewidth(0.5)
    for k, R in enumerate(rets):
        q1, med, q3 = np.percentile(R, [25, 50, 75])
        ax.plot([k, k], [q1, q3], color=BAR_COLOR, lw=1.3, solid_capstyle="round")
        ax.plot(k, med, "o", ms=2.8, mfc="white", mec=BAR_COLOR, mew=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["E", "M-E", "M", "M-R"]); ax.set_xlim(-0.6, 3.6)
    style(ax, f"({'abc'[p]}) {task}")
lo = min(R.min() for t in data for R in data[t][0]); hi = max(R.max() for t in data for R in data[t][0])
axes[0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
axes[0].set_ylabel("Episode return", labelpad=2)
fig.subplots_adjust(left=0.135, right=0.995, top=0.97, bottom=0.215, wspace=0.07)
fig.align_xlabels(axes)
fig.savefig(f"{HERE}/dataset_returns.png", dpi=DPI); plt.close(fig)

# Figure 2: out-of-bound action fraction per joint, Hopper and HalfCheetah only (Ant has none: max |a| = 0.69),
# one shared y-axis; panel widths proportional to the joint count so every bar has the same physical width
TASKS2 = [t for t in TASKS if t[0] != "Ant"]
fig, axes = plt.subplots(1, 2, figsize=(COLW, 2.25), sharey=True,
                         gridspec_kw={"width_ratios": [len(t[2]) + 0.2 for t in TASKS2]})
for p, (ax, (task, conf, joints)) in enumerate(zip(axes, TASKS2)):
    oob = data[task][1]
    nj = len(joints); x = np.arange(nj); w = 0.2
    for k in range(4):
        ax.bar(x + (k - 1.5) * w, oob[k], width=w, color=DS_COL[k], edgecolor="white", linewidth=0.3, label=QLAB[k])
    ax.set_xticks(x); ax.set_xticklabels(joints, fontsize=6.5, linespacing=1.0); ax.set_xlim(-0.6, nj - 0.4)
    style(ax, f"({'ab'[p]}) {task}")
axes[0].set_ylabel("Actions outside [-1, 1] (%)", labelpad=2); axes[0].set_ylim(0, 70)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, -0.015),
           handlelength=1.0, handletextpad=0.4, columnspacing=0.9)
fig.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.34, wspace=0.05)
fig.align_xlabels(axes)                       # tick labels have one or two lines: put "(a)", "(b)" on one baseline
fig.savefig(f"{HERE}/dataset_actions.png", dpi=DPI); plt.close(fig)

out = {task: {q: dict(episodes=len(R), mean=float(R.mean()), median=float(np.median(R)), sd=float(R.std()),
                      q25=float(np.percentile(R, 25)), q75=float(np.percentile(R, 75)), min=float(R.min()), max=float(R.max()),
                      out_of_bound_pct=o, overall_out_of_bound_pct=float(np.mean(o)), max_abs_action=m)
              for q, R, o, m in zip(QUAL, *data[task])} for task in data}
json.dump(out, open(f"{REPO}/results/dataset_summary.json", "w"), indent=1)
print("wrote", HERE)
