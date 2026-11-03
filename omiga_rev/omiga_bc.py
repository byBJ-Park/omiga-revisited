"""OMIGA+BC (Section VI of the paper): a TD3+BC actor on a frozen OMIGA critic.

The critic (Q_i and the mixer, online networks of ckpt_final.pt) is loaded and frozen; a deterministic policy shared
by the agents is trained with

    L = -lambda * Q_tot(s, pi(o)) + || pi(o) - a_data ||^2,   lambda = eta / E|Q_tot(s, pi(o))|,

where the expectation is the mean absolute value over the minibatch, treated as a constant. eta = 0 removes the
critic term (behaviour cloning in the same pipeline). Q_tot = sum_i w_i(s) Q_i(o_i, a_i) + b(s), so each agent's
gradient flows through its own Q_i only.

Writes <out_dir>/TD3BC-<scenario>-<conf>-<quality>-s<seed>/{log.json, actor_final.pt}. The in-training evaluations
in log.json use the MuJoCo 3 environment and are for monitoring only; the reported returns come from eval_mj200.py.

Usage (paper setting: 3 x 10^5 steps, seed equal to the critic seed):
    python -m omiga_rev.omiga_bc --ckpt checkpoints/omiga/OMIGA-Hopper-v2-3x1-medium-s0 --scenario Hopper-v2 \
        --agent_conf 3x1 --data_type medium --seed 0 --eta 5 --out_dir checkpoints/omiga_bc/eta5
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import ReplayBuffer
from .networks import MixNet, Q_critic
from .train_omiga import evaluate, make_env


class DetActor(nn.Module):
    """Deterministic actor shared by the agents, agent identity by one-hot (the trunk of OMIGA's Actor, one head)."""

    def __init__(self, num_obs, num_action, num_agent, hidden):
        super().__init__()
        self.fc1 = nn.Linear(num_obs + num_agent, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_action)

    def forward(self, o_id):
        x = F.relu(self.fc1(o_id))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.head(x))


def find_checkpoint(path):
    """A ckpt_final.pt, or a run directory (or its parent) that holds exactly one."""
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "*", "ckpt_final.pt")) + glob.glob(os.path.join(path, "ckpt_final.pt"))
        assert len(hits) == 1, f"expected one ckpt_final.pt under {path}, found {hits}"
        return hits[0]
    return path


class Extractor:
    def __init__(self, obs_dim, act_dim, n_agents, cfg, device):
        self.n = n_agents
        self.device = device
        self.cfg = cfg
        st = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
        self.q = Q_critic(obs_dim, act_dim, n_agents, cfg["hidden_sizes"], device).to(device)
        self.mix = MixNet(obs_dim, act_dim, n_agents, cfg["mix_hidden_sizes"], device).to(device)
        self.q.load_state_dict(st["networks"]["_q_network"])
        self.mix.load_state_dict(st["networks"]["_mix_network"])
        for p in list(self.q.parameters()) + list(self.mix.parameters()):
            p.requires_grad_(False)
        self.q.eval()
        self.mix.eval()
        self.actor = DetActor(obs_dim, act_dim, n_agents, cfg["hidden_sizes"]).to(device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.eye = torch.eye(n_agents, device=device)

    def _ids(self, B):
        return self.eye.expand(B, -1, -1)

    def q_tot(self, o, s, a):
        oa = torch.cat((o, a, self._ids(o.shape[0])), dim=-1)
        qi = self.q(oa)                                       # [B, n, 1]
        w, b = self.mix(s)                                    # [B, n, 1], [B, 1, 1]
        return ((w * qi).sum(dim=-2) + b.squeeze(-1)).squeeze(-1)   # [B]

    def update(self, o, s, a_data):
        a_pi = self.actor(torch.cat((o, self._ids(o.shape[0])), dim=-1))
        q = self.q_tot(o, s, a_pi)
        G = q.abs().mean().detach()
        lam = self.cfg["eta"] / G.clamp_min(1e-6)
        bc = ((a_pi - a_data) ** 2).mean()
        loss = -lam * q.mean() + bc
        self.opt.zero_grad()
        loss.backward()
        if self.cfg["grad_norm_clip"] > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["grad_norm_clip"])
        self.opt.step()
        return {"q_pi": float(q.mean().detach()), "bc_mse": float(bc.detach()), "lam": float(lam), "G": float(G)}

    @torch.no_grad()
    def diag(self, o, s, a_data):
        """Q at data actions vs at policy actions, and the policy's distance from the data actions."""
        a_pi = self.actor(torch.cat((o, self._ids(o.shape[0])), dim=-1))
        return {"q_data": float(self.q_tot(o, s, a_data).mean()),
                "q_pi_eval": float(self.q_tot(o, s, a_pi).mean()),
                "act_l2": float((a_pi - a_data).norm(dim=-1).mean())}

    @torch.no_grad()
    def step(self, o):                       # evaluate() protocol: [B, n, obs] -> cpu actions
        o = torch.from_numpy(o).to(self.device)
        return self.actor(torch.cat((o, self._ids(o.shape[0])), dim=-1)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="OMIGA ckpt_final.pt, or the run directory holding it")
    p.add_argument("--scenario", default="Hopper-v2")
    p.add_argument("--agent_conf", default="3x1")
    p.add_argument("--data_type", default="medium")
    p.add_argument("--agent_obsk", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--eta", type=float, default=2.5, help="weight of the critic term; 0 = behaviour cloning")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_sizes", type=int, default=256)
    p.add_argument("--mix_hidden_sizes", type=int, default=64)
    p.add_argument("--grad_norm_clip", type=float, default=1.0)
    p.add_argument("--total_iterations", type=int, default=300000)
    p.add_argument("--log_iterations", type=int, default=10000)
    p.add_argument("--num_evaluation", type=int, default=10)
    cfg = vars(p.parse_args())
    cfg["ckpt"] = find_checkpoint(cfg["ckpt"])

    tag = f"{cfg['scenario']}-{cfg['agent_conf']}-{cfg['data_type']}"
    run_name = f"TD3BC-{tag}-s{cfg['seed']}"
    out_dir = os.path.join(cfg["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    env_cfg = {"scenario": cfg["scenario"], "agent_conf": cfg["agent_conf"], "agent_obsk": cfg["agent_obsk"]}
    eval_env = make_env(env_cfg, cfg["seed"] + 10000)
    obs_dim = eval_env.observation_space[0].shape[0]
    act_dim = eval_env.action_space[0].shape[0]
    n_agents = len(eval_env.observation_space)

    data = ReplayBuffer(obs_dim, act_dim, n_agents, tag, cfg["data_dir"], device=cfg["device"])
    data.load()
    data.to_device()
    print(f"[{run_name}] critic {cfg['ckpt']}  dataset {data.size} transitions")

    ex = Extractor(obs_dim, act_dim, n_agents, cfg, device)
    logs = []
    t0 = time.time()
    for it in range(cfg["total_iterations"] + 1):
        if it % cfg["log_iterations"] == 0:
            o, s, a, *_ = data.sample(4096)
            rec = {"iteration": it, **ex.diag(o, s, a)}
            if it > 0:
                rec.update(last)
            ep_r, ep_std = evaluate(ex, eval_env, cfg["num_evaluation"])
            rec.update({"ep_r": ep_r, "ep_r_std": ep_std, "wall": time.time() - t0})
            logs.append(rec)
            print(f"[{run_name}] it={it:>7d}  return={ep_r:9.1f} +-{ep_std:7.1f}  "
                  f"q_pi={rec['q_pi_eval']:8.2f} q_data={rec['q_data']:8.2f} "
                  f"act_l2={rec['act_l2']:.3f}  {rec['wall']/60:.1f}min", flush=True)
            with open(os.path.join(out_dir, "log.json"), "w") as f:
                json.dump({"config": cfg, "logs": logs}, f)
        if it == cfg["total_iterations"]:
            break
        o, s, a, *_ = data.sample(cfg["batch_size"])
        last = ex.update(o, s, a)
    torch.save(ex.actor.state_dict(), os.path.join(out_dir, "actor_final.pt"))


if __name__ == "__main__":
    main()
