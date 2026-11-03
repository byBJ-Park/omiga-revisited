"""Figure: one dataset seen in action space and in the latent space of the OMIGA actor.

(a) distribution of the dataset action components a (all joints pooled), the part outside (-1, 1) highlighted;
(b) distribution of the same components after the OMIGA transform phi(a) = atanh(clamp(a, -1 + 1e-7, 1 - 1e-7)) in
    float32: everything outside (-1, 1) lands on the two walls +-8.32.
Data: data/<CELL>.hdf5, episode-boundary rows excluded. Default cell Hopper expert (Fig. 3 of the paper); another
cell: --cell. Writes paper/figures/action_latent_<cell>.png (one column, 3.5 in, 600 dpi) and prints the numbers.
Run: python analysis/make_fig_action_latent.py
"""
import argparse, os
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL, OUTC, INK, INK2, GRID = "#C9CED6", "#8C1D40", "#0b0b0b", "#52514e", "#e4e3df"   # in-bound: light blue-grey; out-of-bound: burgundy
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.5, "font.family": "DejaVu Sans",
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
EPS = np.float32(1e-7)
phi = lambda a: np.arctanh(np.clip(np.asarray(a, np.float32), np.float32(-1) + EPS, np.float32(1) - EPS))
WALL = float(phi(np.float32(1.0)))
NAME = {"Hopper-v2-3x1": "Hopper", "HalfCheetah-v2-6x1": "HalfCheetah", "Ant-v2-2x4": "Ant"}

p = argparse.ArgumentParser(); p.add_argument("--cell", default="Hopper-v2-3x1-expert"); cell = p.parse_args().cell
with h5py.File(os.path.join(ROOT, f"data/{cell}.hdf5"), "r") as f:
    d = np.asarray(f["d"])[:, 0]; a = np.asarray(f["a"])[d < 0.5].reshape(-1).astype(np.float32)
t = phi(a); out = np.abs(a) >= 1 - EPS
label = next(f"{v} {cell[len(k) + 1:]}" for k, v in NAME.items() if cell.startswith(k))
print(f"{label}: {a.size} components, outside {100 * out.mean():.1f}% (above {100 * (a >= 1 - EPS).mean():.1f}%, "
      f"below {100 * (a <= -1 + EPS).mean():.1f}%), max |a| {np.abs(a).max():.2f}, wall {WALL:.3f}")


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


w = 100 / a.size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.3))
fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.12, hspace=0.55)

# (a) action space
lim = float(np.ceil(np.abs(a).max() * 10) / 10)
ab = np.linspace(-lim, lim, 2 * int(round(lim / 0.1)) + 1)          # 0.1-wide bins, edges on +-1
ax1.hist(a[~out], bins=ab, color=COL, weights=np.full((~out).sum(), w))
ax1.hist(a[out], bins=ab, color=OUTC, weights=np.full(out.sum(), w))
for x in (-1, 1): ax1.axvline(x, color=INK2, lw=0.5, ls=":")
ax1.set_xlim(-lim, lim); ax1.set_ylabel("Actions (%)"); style(ax1)
ax1.set_xlabel("(a) Action space", fontsize=8, color=INK, labelpad=2)

# (b) latent space
tb = np.linspace(-9, 9, 181)                                        # 0.1-wide bins
ax2.hist(t[~out], bins=tb, color=COL, weights=np.full((~out).sum(), w))
ax2.hist(t[out], bins=tb, color=OUTC, weights=np.full(out.sum(), w))
ax2.set_xlim(-9.5, 9.5); ax2.set_xticks([-WALL, -4, 0, 4, WALL]); ax2.set_xticklabels([f"\u2212{WALL:.2f}", "\u22124", "0", "4", f"{WALL:.2f}"])   # U+2212, the minus matplotlib uses in (a)
ax2.set_ylabel("Actions (%)"); style(ax2)
ax2.set_xlabel("(b) Latent space", fontsize=8, color=INK, labelpad=2)

fig.align_ylabels([ax1, ax2])                   # tick labels differ in width (4 vs 15): one x for both y-labels
out_path = os.path.join(ROOT, f"paper/figures/action_latent_{cell}.png")
fig.savefig(out_path, dpi=600); print("wrote", os.path.relpath(out_path, ROOT))
