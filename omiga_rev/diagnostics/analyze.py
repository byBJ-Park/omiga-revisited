"""Table 9 of the paper from the diagnostics: sigma_10 (results/action_spread.json), and from the overestimation
records (results/diagnostics/overestimation/<cell>-s<seed>-eta<eta>.jsonl) the over-estimation
E = sum(Q - R) / sum|R| and the deviation d per (seed, eta), summarized over seeds as mean and 95% t-interval, and
the paired change Delta E = E(eta = 10) - E(eta = 0).

  python -m omiga_rev.diagnostics.analyze           (writes results/critic_chain.json and paper/tables/table_critic_diag.tex)
"""
import glob, json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = [("Hopper", "Hopper-v2-3x1"), ("HalfCheetah", "HalfCheetah-v2-6x1"), ("Ant", "Ant-v2-2x4")]
QUALS = [("expert", "E"), ("medium-expert", "ME"), ("medium", "M"), ("medium-replay", "MR")]
ETAS = ["0", "2.5", "5", "10", "40"]


def tci(v):
    v = np.asarray(v, float)
    h = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(len(v), 2.0) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), [float(v.mean() - h), float(v.mean() + h)]


def main():
    spread = {c: v["0"]["10"] for c, v in json.load(open(os.path.join(ROOT, "results/action_spread.json")))["spread"].items()}
    src = os.path.join(ROOT, "results/diagnostics/overestimation")
    res = {}
    for _, prefix in TASKS:
        for q, _ in QUALS:
            c = f"{prefix}-{q}"; E, d = {}, {}
            for f in glob.glob(f"{src}/{c}-s*-eta*.jsonl"):
                rs = [json.loads(l) for l in open(f)]
                if not rs:
                    continue
                s, e = rs[0]["seed"], rs[0]["eta"]
                E[(s, e)] = sum(r["sum_err"] for r in rs) / sum(r["sum_abs_rtg"] for r in rs)
                d[(s, e)] = float(np.mean([r["mean_dist"] for r in rs]))
            ss = sorted({s for s, _ in E})
            res[c] = {"seeds": ss, "sigma10": spread[c],
                      "E": {e: tci([E[(s, e)] for s in ss if (s, e) in E]) for e in ETAS},
                      "d": {e: tci([d[(s, e)] for s in ss if (s, e) in d]) for e in ETAS},
                      "dE_10": tci([E[(s, "10")] - E[(s, "0")] for s in ss if (s, "10") in E and (s, "0") in E])}
    json.dump({"res": res}, open(os.path.join(ROOT, "results/critic_chain.json"), "w"), indent=1)

    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{Critic diagnostics. $\sigma_{10}$: state-conditional action spread. $\Delta E$: change in "
           r"overestimation from $\eta=0$ to $\eta=10$, with a 95\% $t$-interval over critic seeds. $d$: mean deviation "
           r"of the policy from the local data actions, in units of $\sigma_{10}$.}",
           r"\label{tab:chain}", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llccc}", r"\hline",
           r"Task & Dataset & $\sigma_{10}$ & $\Delta E$ & $d$ ($\eta=0$ / $10$) \\", r"\hline"]
    for task, prefix in TASKS:
        for j, (q, short) in enumerate(QUALS):
            r = res[f"{prefix}-{q}"]; m, ci = r["dE_10"]
            tex.append(f"{task if j == 0 else ''} & {short} & {r['sigma10']:.3f} & {m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] & "
                       f"{r['d']['0'][0]:.1f} / {r['d']['10'][0]:.1f} \\\\".replace("-", "$-$").replace("$-$$-$", "--"))
        tex.append(r"\hline")
    tex += [r"\end{tabular*}", r"\end{table}"]
    out = os.path.join(ROOT, "paper/tables/table_critic_diag.tex")
    open(out, "w").write("\n".join(tex) + "\n"); print("\n".join(tex)); print("wrote", out)


if __name__ == "__main__":
    main()
