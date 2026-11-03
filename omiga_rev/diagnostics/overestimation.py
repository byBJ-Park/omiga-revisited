"""Policy deviation and critic over-estimation along the OMIGA+BC policy's own episodes (Appendix E, Table 9), for one
(dataset, seed, eta): ten episodes from the first ten initial states of the evaluation protocol; per episode the sum
of Q_tot(s_t, pi(o_t)) - R_t (R_t the realized discounted return-to-go) over the first 500 steps, the sum of |R_t|,
and the mean distance of the policy's actions from the mean logged action of the ten nearest pool states, in units
of the dataset's action spread sigma_10.

Run in the evaluation environment (MuJoCo 2.0):
  python -m omiga_rev.diagnostics.overestimation --cell Ant-v2-2x4-medium --seed 0 --eta 2.5 \
      --critic checkpoints/omiga/OMIGA-Ant-v2-2x4-medium-s0/ckpt_final.pt \
      --actor checkpoints/omiga_bc/eta2.5/TD3BC-Ant-v2-2x4-medium-s0/actor_final.pt \
      --spread results/action_spread.json --out results/diagnostics/overestimation/Ant-v2-2x4-medium-s0-eta2.5.jsonl
"""
import argparse, json
import numpy as np
import torch
from .common import Pool, eligible, load_actor, load_critic, load_data, OmigaMJ200

GAMMA, TMAX, K = 0.99, 500, 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eta", required=True); p.add_argument("--critic", required=True); p.add_argument("--actor", required=True)
    p.add_argument("--data_dir", default="data"); p.add_argument("--spread", default="results/action_spread.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    spread = json.load(open(a.spread))["spread"][a.cell]["0"]["10"]
    env = OmigaMJ200(a.cell.split("-")[0] + "-v2"); n, ad = env.n, env.act
    O, A, D = load_data(f"{a.data_dir}/{a.cell}.hdf5"); od = O.shape[-1]
    pool = Pool(O, A, np.random.default_rng(3000 + a.seed).choice(eligible(D), 20000, replace=False))
    critic = load_critic(a.critic, od, ad, n); pi = load_actor(a.actor, od, ad, n)

    for ep in range(10):
        env.seed(50000 + ep); obs = env.reset(); S, Q, R, P = [], [], [], []
        for t in range(1000):
            o = torch.as_tensor(np.stack(obs))
            with torch.no_grad():
                act = pi(o); q = float(critic(o, act))
            S.append(np.stack(obs)); Q.append(q); P.append(act.numpy())
            obs, r, d, _ = env.step(list(act.numpy())); R.append(float(r))
            if d:
                break
        T = len(R); rtg = np.zeros(T); acc = 0.0
        for t in reversed(range(T)):
            acc = R[t] + GAMMA * acc; rtg[t] = acc
        m = min(T, TMAX + 1); Q = np.array(Q[:m]); rtg = rtg[:m]
        nb = pool.nn(np.stack(S[:m]), K)                                                  # [m, K]
        local = pool.A[nb].reshape(m, K, -1).clip(-1, 1).mean(1)                          # mean local data action
        dist = np.sqrt(((np.stack(P[:m]).reshape(m, -1) - local) ** 2).mean(1)) / spread
        rec = {"cell": a.cell, "seed": a.seed, "eta": a.eta, "ep": ep, "len": T, "ret": float(sum(R)), "n": int(m),
               "sum_err": float((Q - rtg).sum()), "sum_abs_rtg": float(np.abs(rtg).sum()), "mean_q": float(Q.mean()),
               "mean_rtg": float(rtg.mean()), "mean_dist": float(dist.mean())}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(f"{a.cell} s{a.seed} eta {a.eta} done", flush=True)


if __name__ == "__main__":
    main()
