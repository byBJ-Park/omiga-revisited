"""Evaluate saved actors under MuJoCo 2.0, the engine that generated the datasets (the protocol of every return in the
paper): 100 episodes from the environment seeds 50000..50099, at most 1000 steps, deterministic actions.

Actor kinds and their execution rule:
  det       DetActor (fc1, fc2, head), tanh(head)                      behaviour cloning, weighted MSE, OMIGA+BC
  gauss     tanh-Gaussian Actor (fc1, fc2, mu_head, sigma_head), tanh(clamp(mu, +-9)), as OMIGA's
            get_deterministic_action                                   latent-space NLL
  latmse    RawHead (fc1, fc2, head) regressed on atanh(a), tanh(clamp(head, +-9))   latent-space MSE
  actgauss  Gaussian in action space, clamp(mu, -1, 1)                 action-space NLL

Run in the evaluation environment (envs/eval_mj200.yml: gym 0.10.8 + mujoco-py 2.0, MuJoCo 2.0 on LD_LIBRARY_PATH):
  python -m omiga_rev.eval_mj200 --out results/<file>.jsonl --cell Hopper-v2-3x1-medium \
      det:checkpoints/.../actor_final.pt::<name> [...]
One record per policy is appended to --out; names already present are skipped.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs.mj200_env import OmigaMJ200

LIMIT, BASE = 1000, 50000
torch.set_num_threads(1)


class Trunk(nn.Module):
    def __init__(self, obs, act, n, h, heads):
        super().__init__()
        self.fc1 = nn.Linear(obs + n, h); self.fc2 = nn.Linear(h, h)
        for name in heads:
            setattr(self, name, nn.Linear(h, act))

    def feat(self, x): return F.relu(self.fc2(F.relu(self.fc1(x))))


HEADS = {"det": ["head"], "latmse": ["head"], "gauss": ["mu_head", "sigma_head"], "actgauss": ["mu_head", "sigma_head"]}
RULE = {"det": lambda m, x: torch.tanh(m.head(m.feat(x))),
        "latmse": lambda m, x: torch.tanh(m.head(m.feat(x)).clamp(-9.0, 9.0)),
        "gauss": lambda m, x: torch.tanh(m.mu_head(m.feat(x)).clamp(-9.0, 9.0)),
        "actgauss": lambda m, x: m.mu_head(m.feat(x)).clamp(-1.0, 1.0)}


def evaluate_policy(kind, path, scenario, episodes=100):
    env = OmigaMJ200(scenario)
    sd = torch.load(path, map_location="cpu")
    net = Trunk(env.obs_dim, env.act, env.n, sd["fc1.weight"].shape[0], HEADS[kind])
    net.load_state_dict(sd); net.eval()
    eye = torch.eye(env.n); rets, lens = [], []
    for i in range(episodes):
        env.seed(BASE + i); obs = env.reset(); R = 0.0; t = 0
        for t in range(1, LIMIT + 1):
            with torch.no_grad():
                act = RULE[kind](net, torch.cat([torch.as_tensor(np.asarray(obs, dtype=np.float32)), eye], -1)).numpy()
            obs, r, d, _ = env.step(list(act)); R += float(r)
            if d:
                break
        rets.append(R); lens.append(t)
    rets = np.asarray(rets)
    return {"ret_mean": float(rets.mean()), "ret_se": float(rets.std(ddof=1) / np.sqrt(len(rets))),
            "survival": float(np.mean(np.asarray(lens) >= LIMIT)), "len_mean": float(np.mean(lens)),
            "ret": rets.tolist(), "len": lens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cell", required=True, help="<scenario>-<conf>-<quality>, e.g. Hopper-v2-3x1-medium")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("policies", nargs="+", help="kind:path::name")
    a = p.parse_args()
    scenario = a.cell.split("-")[0] + "-v2"
    done = {json.loads(l)["run"] for l in open(a.out)} if os.path.exists(a.out) else set()
    for spec in a.policies:
        kind, rest = spec.split(":", 1); path, _, name = rest.partition("::")
        if name in done:
            continue
        t0 = time.time()
        fixed = evaluate_policy(kind, path, scenario, a.episodes)
        rec = {"run": name, "cell": a.cell, "kind": kind, "engine": "mujoco200", "policy": path, "fixed": fixed}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{name:40s} {kind:8s} ret {fixed['ret_mean']:8.1f}  surv {fixed['survival']:.2f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
