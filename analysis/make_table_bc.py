"""Table 3 of the paper: behaviour cloning against the published OMIGA results, MuJoCo 2.0.

BC: results/bc_eval_mj200.jsonl (omiga_rev/eval_mj200.py), seeds 0-4, 100 fixed initial states.
OMIGA: OMIGA paper Table 1 (mean, sd over five seeds). Data mean: results/dataset_summary.json.
Bold: the higher mean of BC and OMIGA; both when |difference| < the larger sd.
Writes paper/tables/table_bc.tex.   Run: python analysis/make_table_bc.py
"""
import collections, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = [("Hopper", "Hopper-v2-3x1"), ("HalfCheetah", "HalfCheetah-v2-6x1"), ("Ant", "Ant-v2-2x4")]
QUALS = ["expert", "medium-expert", "medium", "medium-replay"]
OMIGA = {("HalfCheetah", "expert"): (3383.61, 552.67), ("HalfCheetah", "medium-expert"): (2948.46, 518.89),
         ("HalfCheetah", "medium"): (3608.13, 237.37), ("HalfCheetah", "medium-replay"): (2504.70, 83.47),
         ("Hopper", "expert"): (859.63, 709.47), ("Hopper", "medium-expert"): (709.00, 595.66),
         ("Hopper", "medium"): (1189.26, 544.30), ("Hopper", "medium-replay"): (774.18, 494.27),
         ("Ant", "expert"): (2055.46, 1.58), ("Ant", "medium-expert"): (1720.33, 110.63),
         ("Ant", "medium"): (1418.44, 5.36), ("Ant", "medium-replay"): (1105.13, 88.87)}
DATA = {(e, q): v["mean"] for e, d in json.load(open(os.path.join(ROOT, "results/dataset_summary.json"))).items() for q, v in d.items()}

runs = collections.defaultdict(dict)
for line in open(os.path.join(ROOT, "results/bc_eval_mj200.jsonl")):
    r = json.loads(line)
    runs[r["cell"]][r["seed"]] = r["fixed"]

tex = [r"\begin{table*}[t]", r"\centering",
       r"\caption{Behavior cloning (BC) on the OMIGA Multi-Agent MuJoCo datasets, evaluated in MuJoCo 2.0, the engine "
       r"that generated the datasets and that the OMIGA code requires. Data mean is the average episode return in each "
       r"dataset. BC is reported as mean (standard deviation) over five seeds, each averaged over 100 episodes from a "
       r"fixed set of initial states. Survival is the fraction of BC evaluation episodes that reach the 1000-step limit, "
       r"reported for the two tasks with a termination condition. OMIGA denotes the results reported in the original "
       r"paper~\cite{wang2023offline} (five seeds). Bold marks the higher mean of BC and OMIGA; both are bold when their "
       r"difference is smaller than the larger of the two standard deviations.}",
       r"\label{tab:bc}",
       r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc}", r"\hline",
       r"Task & Dataset & Data mean & BC & BC/Data mean & Survival & OMIGA \\", r"\hline"]
for k, (env, prefix) in enumerate(ENVS):
    for j, q in enumerate(QUALS):
        d = runs[f"{prefix}-{q}"]
        assert sorted(d) == [0, 1, 2, 3, 4], (prefix, q, sorted(d))
        ret = np.array([d[s]["ret_mean"] for s in range(5)])
        m, s = ret.mean(), ret.std(ddof=1)
        om, osd = OMIGA[(env, q)]
        tie = abs(m - om) < max(s, osd)
        bc_txt, om_txt = f"{m:.0f} ({s:.0f})", f"{om:.0f} ({osd:.0f})"
        if tie or m > om:
            bc_txt = rf"\textbf{{{bc_txt}}}"
        if tie or om > m:
            om_txt = rf"\textbf{{{om_txt}}}"
        surv = "--" if env == "HalfCheetah" else f"{np.mean([d[s]['survival'] for s in range(5)]):.2f}"
        task = env if j == 0 else ""
        tex.append(f"{task} & {q} & {DATA[(env, q)]:.0f} & {bc_txt} & {m / DATA[(env, q)]:.2f} & {surv} & {om_txt} \\\\")
    tex.append(r"\hline")
tex += [r"\end{tabular*}", r"\end{table*}"]
open(os.path.join(ROOT, "paper/tables/table_bc.tex"), "w").write("\n".join(tex) + "\n")
print("\n".join(tex))
