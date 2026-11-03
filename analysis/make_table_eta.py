"""Table 8 of the paper: OMIGA+BC for each value of eta, MuJoCo 2.0.

Source: results/omiga_bc_eval_mj200.jsonl (omiga_rev/eval_mj200.py on the actors of omiga_rev/omiga_bc.py; seeds 0-4,
seed i trained on the critic of OMIGA seed i; eta = 0 is behaviour cloning in the same pipeline). Mean (sd over the
five seeds) of the 100-episode mean return; bold = highest mean in the row; last row = mean over the 12 datasets.
Writes paper/tables/table_eta.tex and paper/tables/table_eta.md (the .md also lists the survival rate).
Run: python analysis/make_table_eta.py
"""
import collections, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETAS = [0.0, 2.5, 5.0, 10.0, 40.0]
TASK = {"Hopper-v2-3x1": "Hopper", "HalfCheetah-v2-6x1": "HalfCheetah", "Ant-v2-2x4": "Ant"}
CELLS = [f"{p}-{q}" for p in TASK for q in ("expert", "medium-expert", "medium", "medium-replay")]


def load():
    g = collections.defaultdict(dict)                              # (cell, eta) -> seed -> (return, survival)
    for line in open(os.path.join(ROOT, "results/omiga_bc_eval_mj200.jsonl")):
        r = json.loads(line)
        g[(r["cell"], r["eta"])][r["seed"]] = (r["fixed"]["ret_mean"], r["fixed"]["survival"])
    return g


def main():
    g = load(); res = {}
    md = ["# OMIGA+BC for each eta (MuJoCo 2.0)", "",
          "mean (sd over seeds 0-4; seed i trains the critic and the actor), survival.", "",
          "| cell | " + " | ".join(f"eta {a:g}" for a in ETAS) + " |", "|---|" + "---|" * len(ETAS)]
    for cell in CELLS:
        out = []
        for a in ETAS:
            d = g[(cell, a)]
            assert sorted(d) == list(range(5)), (cell, a, sorted(d))
            ret = np.array([d[s][0] for s in range(5)]); surv = np.mean([d[s][1] for s in range(5)])
            res[(cell, a)] = (ret.mean(), ret.std(ddof=1))
            out.append(f"{ret.mean():.0f} ({ret.std(ddof=1):.0f}), {surv:.2f}")
        md.append(f"| {cell} | " + " | ".join(out) + " |")
    open(os.path.join(ROOT, "paper/tables/table_eta.md"), "w").write("\n".join(md) + "\n")

    fmt = lambda v: f"{v[0]:.0f} ({v[1]:.0f})".replace("-", "$-$")
    tex = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{OMIGA+BC for each value of $\eta$. With $\eta=0$ the critic term vanishes and OMIGA+BC reduces to "
           r"behavior cloning. Mean (standard deviation) over five seeds, evaluated in MuJoCo~2.0; bold marks the highest "
           r"mean in each row; the last row averages the means over the twelve datasets.}",
           r"\label{tab:eta-ablation}", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll" + "r" * len(ETAS) + "@{}}", r"\hline",
           r" & & \multicolumn{" + str(len(ETAS)) + r"}{c}{$\eta$} \\", r"\cline{3-" + str(2 + len(ETAS)) + "}",
           "Task & Dataset & " + " & ".join(f"{a:g}" for a in ETAS) + r" \\", r"\hline"]
    for cell in CELLS:
        prefix, q = next((k, cell[len(k) + 1:]) for k in TASK if cell.startswith(k))
        best = max(ETAS, key=lambda a: res[(cell, a)][0])
        vals = [(r"\textbf{" + fmt(res[(cell, a)]) + "}") if a == best else fmt(res[(cell, a)]) for a in ETAS]
        tex.append(f"{TASK[prefix] if q == 'expert' else ''} & {q} & " + " & ".join(vals) + r" \\")
        if q == "medium-replay":
            tex.append(r"\hline")
    avg = {a: np.mean([res[(c, a)][0] for c in CELLS]) for a in ETAS}
    tex += [r"\multicolumn{2}{@{}l}{Mean} & " + " & ".join((r"\textbf{%.0f}" if a == max(avg, key=avg.get) else "%.0f") % avg[a]
                                                         for a in ETAS).replace("-", "$-$") + r" \\",
            r"\hline", r"\end{tabular*}", r"\end{table*}"]
    out = os.path.join(ROOT, "paper/tables/table_eta.tex")
    open(out, "w").write("\n".join(tex) + "\n"); print("\n".join(md)); print("wrote", out)


if __name__ == "__main__":
    main()
