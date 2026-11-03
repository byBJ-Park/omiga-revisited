"""Tables 4 and 7 of the paper (controlled comparison), evaluated under MuJoCo 2.0.

(1) table_factorial_target: behaviour cloning with the regression target (latent atanh(a) vs action a) crossed with
    the loss (Gaussian NLL with learned sigma vs squared error); omiga_rev/bc_target.py, arms A latent+NLL,
    L latent+MSE, G action+NLL, B action+MSE; results/factorial_target_eval_mj200.jsonl.
(2) table_factorial_weight: OMIGA's advantage-weighted policy loss with its tanh-Gaussian NLL actor (arm C) vs the
    same weight with a squared-error actor (arm D), on OMIGA's frozen critic; omiga_rev/bc_weighted.py;
    results/factorial_weight_eval_mj200.jsonl. Table 7 also lists the unweighted arms A and B of the same script.
100 fixed initial states (reset seeds 50000-50099), final checkpoint, three training seeds (0-2).
Out-of-range share = fraction of dataset action components with |a| > 1 (episode boundary rows excluded; needs data/).
Writes paper/tables/table_factorial_target.{md,tex} and table_factorial_weight.{md,tex}.
Run: python analysis/make_table_factorial.py
"""
import json, os, collections
import numpy as np, h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper/tables")
SEEDS = [0, 1, 2]


def load(path):
    runs = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        runs[(r["cell"], r["arm"])][r["seed"]] = r["fixed"]
    return runs


def oob(cell):
    with h5py.File(os.path.join(ROOT, f"data/{cell}.hdf5"), "r") as f:
        a = np.asarray(f["a"]); d = np.asarray(f["d"])[:, 0]
    return 100 * float((np.abs(a[d < 0.5]) > 1).mean())


def stat(runs, cell, arm):
    d = runs[(cell, arm)]
    assert sorted(d) == SEEDS, (cell, arm, sorted(d))
    ret = np.array([d[s]["ret_mean"] for s in SEEDS]); surv = np.mean([d[s]["survival"] for s in SEEDS])
    return ret.mean(), ret.std(ddof=1), surv, ret


def fmt(st, term):
    m, s, sv, _ = st
    return f"{m:.0f} ({s:.0f})" + (f", surv {sv:.2f}" if term else "")


def tex_fmt(st, term):
    m, s, sv, _ = st
    return f"{m:.0f} ({s:.0f})" + (f" [{sv:.2f}]" if term else "")


NAME = {"Hopper-v2-3x1": "Hopper", "HalfCheetah-v2-6x1": "HalfCheetah", "Ant-v2-2x4": "Ant"}
split = lambda cell: next((NAME[k], cell[len(k) + 1:]) for k in NAME if cell.startswith(k))

# ------------------------------------------------------------------------------------------------ (1) 2x2
P = load(f"{SRC}/factorial_target_eval_mj200.jsonl")
CELLS1 = ["Hopper-v2-3x1-expert", "Hopper-v2-3x1-medium-expert", "Hopper-v2-3x1-medium-replay",
          "HalfCheetah-v2-6x1-expert", "HalfCheetah-v2-6x1-medium-expert", "Ant-v2-2x4-medium"]
COLS1 = [("A", "atanh target, NLL"), ("L", "atanh target, MSE"), ("G", "action target, NLL"), ("B", "action target, MSE")]
md = ["# Regression target x loss (behaviour cloning, MuJoCo 2.0)", "",
      "Mean (sd) over 3 training seeds of the mean return on 100 fixed initial states; surv = share of episodes reaching "
      "1000 steps (seed mean). Out of range = share of dataset action components with |a| > 1.", "",
      "| task | dataset | out of range | " + " | ".join(c for _, c in COLS1) + " |", "|---|---|---|---|---|---|---|"]
tex = [r"\begin{table*}[t]", r"\centering",
       r"\caption{Behavior cloning with the regression target (the latent $\operatorname{atanh}(a)$ used by the OMIGA "
       r"actor, or the action $a$) and the loss (Gaussian negative log-likelihood with a learned scale, or squared "
       r"error) varied independently. The first column is the actor of the OMIGA code. Mean (standard deviation) over "
       r"three seeds of the return on 100 fixed initial states in MuJoCo 2.0; survival in brackets for the tasks with a "
       r"termination condition. Out of range is the share of dataset action components with $|a|>1$. The design and the "
       r"predictions were registered before training.}",
       r"\label{tab:param2x2}", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc}", r"\hline",
       r" & & & \multicolumn{2}{c}{Target $\operatorname{atanh}(a)$} & \multicolumn{2}{c}{Target $a$} \\",
       r"Task & Dataset & Out of range & NLL & MSE & NLL & MSE \\", r"\hline"]
for cell in CELLS1:
    task, q = split(cell); term = task != "HalfCheetah"; o = oob(cell)
    sts = [stat(P, cell, arm) for arm, _ in COLS1]
    md.append(f"| {task} | {q} | {o:.1f}% | " + " | ".join(fmt(s, term) for s in sts) + " |")
    tex.append(f"{task} & {q} & {o:.1f}\\% & " + " & ".join(tex_fmt(s, term) for s in sts) + r" \\")
md += ["", "Per-seed returns:", ""]
for cell in CELLS1:
    md.append(f"- {' '.join(split(cell))}: " + "; ".join(
        f"{c} " + " / ".join(f"{x:.0f}" for x in stat(P, cell, arm)[3]) for arm, c in COLS1))
tex += [r"\hline", r"\end{tabular*}", r"\end{table*}"]
open(f"{OUT}/table_factorial_target.md", "w").write("\n".join(md) + "\n")
open(f"{OUT}/table_factorial_target.tex", "w").write("\n".join(tex) + "\n")
print("\n".join(md)); print()

# ------------------------------------------------------------------------------------------------ (2) OMIGA actor
Q = load(f"{SRC}/factorial_weight_eval_mj200.jsonl")
CELLS2 = ["Hopper-v2-3x1-expert", "Hopper-v2-3x1-medium-expert", "Hopper-v2-3x1-medium", "Hopper-v2-3x1-medium-replay",
          "Ant-v2-2x4-medium"]
md = ["# OMIGA's weighted policy loss: tanh-Gaussian NLL actor vs squared-error actor (frozen critic, MuJoCo 2.0)", "",
      "Same advantage weight exp(w (Q - V) / alpha) from OMIGA's critic in both columns; mean (sd) over 3 seeds, "
      "100 fixed initial states.", "",
      "| task | dataset | out of range | OMIGA actor (tanh-Gaussian NLL) | squared-error actor | difference |",
      "|---|---|---|---|---|---|"]
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{OMIGA's advantage-weighted policy extraction with the actor of the OMIGA code (tanh-Gaussian "
       r"negative log-likelihood) and with a squared-error actor, both trained on the same frozen OMIGA critic. Mean "
       r"(standard deviation) over three seeds of the return on 100 fixed initial states in MuJoCo 2.0; survival in "
       r"brackets.}",
       r"\label{tab:omiga-actor}", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcc}", r"\hline",
       r"Task & Dataset & tanh-Gaussian NLL & Squared error \\", r"\hline"]
for cell in CELLS2:
    task, q = split(cell); o = oob(cell)
    c, d = stat(Q, cell, "C"), stat(Q, cell, "D")
    md.append(f"| {task} | {q} | {o:.1f}% | {fmt(c, True)} | {fmt(d, True)} | {d[0] - c[0]:+.0f} |")
    tex.append(f"{task} & {q} & {tex_fmt(c, True)} & {tex_fmt(d, True)} \\\\")
md += ["", "Per-seed returns:", ""]
for cell in CELLS2:
    md.append(f"- {' '.join(split(cell))}: OMIGA actor " + " / ".join(f"{x:.0f}" for x in stat(Q, cell, "C")[3])
              + "; squared-error actor " + " / ".join(f"{x:.0f}" for x in stat(Q, cell, "D")[3]))
tex += [r"\hline", r"\end{tabular*}", r"\end{table}"]
open(f"{OUT}/table_factorial_weight.md", "w").write("\n".join(md) + "\n")
open(f"{OUT}/table_factorial_weight.tex", "w").write("\n".join(tex) + "\n")
print("\n".join(md))
