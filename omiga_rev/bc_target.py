"""Parameterisation 2x2: latent-vs-action target crossed with fixed-vs-learned scale.

The finished A-vs-B comparison moves two things at once -- the regression target becomes
atanh(a_D) instead of a_D, and a learned per-component sigma with a Gaussian likelihood
replaces a fixed-scale squared error. These four arms separate them:

                 fixed scale            learned sigma
  action space   B deterministic MSE    G action-space Gaussian NLL
  latent space   L latent MSE           A OMIGA tanh-Gaussian NLL

All four arms come from this one file. The pre-registration allowed reusing the existing
A and B runs if this file reproduced them; it did not -- arm B matched exactly but arm A
differed (act_err 0.060042 against 0.059109 at step 500), because this file computes the
density term by term while bc_weighted.py calls Actor.get_log_density, and the two
orderings do not round identically. So all four arms are retrained here and the
2x2 is internally paired, with the older A and B runs kept as a separate cross-check.

The clamps are kept where they belong rather than applied uniformly. mu's +-9 clamp
(MEAN_MAX) belongs to the latent path, because it exists to stop atanh's boundary target
dragging the pre-squash mean without limit, so L keeps it and G does not. log sigma's
[-5, 2] clamp belongs to the Gaussian likelihood, so G keeps it and L has no sigma at all.

G is a control defined here. It is NOT any published method's "unsquashed Gaussian" and
is not reported as one.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import ReplayBuffer
from .networks import Actor, MixNet, Q_critic, V_critic
from .omiga_bc import DetActor, find_checkpoint
from .train_omiga import evaluate, make_env

EPS = 1e-7
MEAN_MIN, MEAN_MAX = -9.0, 9.0
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
HALF_LOG_2PI = 0.5 * float(np.log(2 * np.pi))
# arm -> (target space, scale, needs a sigma head)
ARMS = {"B": ("action", "fixed", False), "A": ("latent", "learned", True),
        "L": ("latent", "fixed", False), "G": ("action", "learned", True)}


class RawHead(nn.Module):
    """Same trunk as DetActor but the head is left unsquashed.

    Arm L regresses this head on atanh(a_D) and squashes only at execution; arm B's
    DetActor squashes inside the loss. Keeping the trunk identical is what makes the
    two columns of the table comparable.
    """

    def __init__(self, num_obs, num_action, num_agent, hidden):
        super().__init__()
        self.fc1 = nn.Linear(num_obs + num_agent, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_action)

    def forward(self, o_id):
        return self.head(F.relu(self.fc2(F.relu(self.fc1(o_id)))))


class Roll:
    """Execution rule per arm, applied identically in training eval and in the final eval."""

    def __init__(self, net, arm, n, device):
        self.net, self.arm, self.device = net, arm, device
        self.eye = torch.eye(n, device=device)

    def act(self, x):
        if self.arm == "B":
            return self.net(x)
        if self.arm == "L":
            return torch.tanh(self.net(x).clamp(MEAN_MIN, MEAN_MAX))
        if self.arm == "A":
            return self.net.get_deterministic_action(x)
        h = F.relu(self.net.fc2(F.relu(self.net.fc1(x))))     # G
        return self.net.mu_head(h).clamp(-1.0, 1.0)

    @torch.no_grad()
    def step(self, o):
        o = torch.from_numpy(o).to(self.device)
        x = torch.cat((o, self.eye.expand(o.shape[0], -1, -1)), dim=-1)
        return self.act(x).cpu()


def build(arm, obs_dim, act_dim, n, hidden, device):
    if arm in ("A", "G"):
        return Actor(obs_dim, act_dim, n, hidden, device).to(device)
    if arm == "B":
        return DetActor(obs_dim, act_dim, n, hidden).to(device)
    return RawHead(obs_dim, act_dim, n, hidden).to(device)


def per_component_loss(arm, net, x, a):
    """Negative log-density or squared error per (agent, action dim), before weighting."""
    if arm == "B":
        return ((net(x) - a) ** 2).sum(dim=-1)
    if arm == "L":
        y = torch.atanh(a.clamp(-1. + EPS, 1. - EPS))
        return ((net(x).clamp(MEAN_MIN, MEAN_MAX) - y) ** 2).sum(dim=-1)
    h = F.relu(net.fc2(F.relu(net.fc1(x))))
    mu, ls = net.mu_head(h), net.sigma_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
    sig = ls.exp()
    # HALF_LOG_2PI is the Gaussian normalising constant. It does not change the gradient,
    # but leaving it out makes the reported loss incomparable to OMIGA's get_log_density,
    # which is how the first version of this file was caught differing from it.
    if arm == "G":                                   # Gaussian in action space, no Jacobian
        return (0.5 * ((a - mu) / sig) ** 2 + ls + HALF_LOG_2PI).sum(dim=-1)
    y = torch.atanh(a.clamp(-1. + EPS, 1. - EPS))    # A: OMIGA's own density
    mu = mu.clamp(MEAN_MIN, MEAN_MAX)
    return (0.5 * ((y - mu) / sig) ** 2 + ls + HALF_LOG_2PI
            + 2 * (np.log(2) - y - F.softplus(-2 * y))).sum(dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["L", "G"], choices=list(ARMS))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--agent_conf", required=True)
    p.add_argument("--data_type", required=True)
    p.add_argument("--weighted", type=int, default=0, choices=[0, 1])
    p.add_argument("--agent_obsk", type=int, default=1)
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--grad_norm_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--num_evaluation", type=int, default=10)
    p.add_argument("--out", required=True)
    cfg = vars(p.parse_args())

    ck = find_checkpoint(cfg["ckpt"])
    ccfg = json.load(open(os.path.join(os.path.dirname(ck), "log.json")))["config"]
    os.makedirs(cfg["out"], exist_ok=True)
    device = torch.device(cfg["device"])

    env = make_env({"scenario": cfg["scenario"], "agent_conf": cfg["agent_conf"],
                    "agent_obsk": cfg["agent_obsk"]}, cfg["seed"] + 10000)
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].shape[0]
    n = len(env.observation_space)
    tag = f"{cfg['scenario']}-{cfg['agent_conf']}-{cfg['data_type']}"

    data = ReplayBuffer(obs_dim, act_dim, n, tag, cfg["data_dir"], device=cfg["device"])
    data.load()
    N = data.o.shape[0]
    O, S, A = np.asarray(data.o), np.asarray(data.s), np.asarray(data.a)
    del data

    W_fn = None
    if cfg["weighted"]:
        st = torch.load(ck, map_location=device, weights_only=False)
        qt = Q_critic(obs_dim, act_dim, n, ccfg["hidden_sizes"], device).to(device)
        mt = MixNet(obs_dim, act_dim, n, ccfg["mix_hidden_sizes"], device).to(device)
        vn = V_critic(obs_dim, n, ccfg["hidden_sizes"], device).to(device)
        qt.load_state_dict(st["networks"]["_q_target_network"])
        mt.load_state_dict(st["networks"]["_mix_target_network"])
        vn.load_state_dict(st["networks"]["_v_network"])
        for m in (qt, mt, vn):
            m.eval()
            for q in m.parameters():
                q.requires_grad_(False)
        alph = float(ccfg.get("alpha", 10.0))

        @torch.no_grad()
        def W_fn(o, s, a, one):
            z = ((mt(s)[0] * qt(torch.cat((o, a, one), dim=-1))
                  - mt(s)[0] * vn(torch.cat((o, one), dim=-1))) / alph).clamp(-10., 10.)
            return torch.exp(z).squeeze(-1)

    eye = torch.eye(n, device=device)
    for arm in cfg["arms"]:
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        net = build(arm, obs_dim, act_dim, n, cfg["hidden"], device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        roll = Roll(net, arm, n, device)
        bg = torch.Generator().manual_seed(4242 + cfg["seed"])
        logs, t0 = [], time.time()

        def do_eval(step, extra):
            net.eval()
            ep_r, sd = evaluate(roll, env, cfg["num_evaluation"])
            net.train()
            rec = {"step": step, "ep_r": ep_r, "ep_r_std": sd,
                   "wall": time.time() - t0, **extra}
            logs.append(rec)
            print(f"[{arm} {tag} s{cfg['seed']}] step={step:>7d} return={ep_r:9.1f} +-{sd:7.1f} "
                  + " ".join(f"{k}={v:.4g}" for k, v in extra.items())
                  + f"  {rec['wall']/60:.1f}min", flush=True)

        do_eval(0, {})
        for step in range(1, cfg["steps"] + 1):
            idx = torch.randint(0, N, (cfg["batch_size"],), generator=bg).numpy()
            o = torch.as_tensor(O[idx], dtype=torch.float32, device=device)
            a = torch.as_tensor(A[idx], dtype=torch.float32, device=device)
            one = eye.expand(o.shape[0], -1, -1)
            x = torch.cat((o, one), dim=-1)
            per = per_component_loss(arm, net, x, a)
            if W_fn is not None:
                s = torch.as_tensor(S[idx], dtype=torch.float32, device=device)
                per = W_fn(o, s, a, one) * per
            loss = per.mean()
            opt.zero_grad(); loss.backward()
            if cfg["grad_norm_clip"] > 0:
                nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_norm_clip"])
            opt.step()
            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                with torch.no_grad():
                    ad = roll.act(x)
                    ex = {"loss": float(loss),
                          "act_err": float((ad - a).norm(dim=-1).mean()),
                          "sat_pi": float((ad.abs() > 0.99).float().mean()),
                          "sat_data": float((a.abs() > 0.99).float().mean()),
                          "false_sat": float(((ad.abs() > 0.99) &
                                              (a.abs() <= 0.99)).float().mean())}
                    if ARMS[arm][2]:
                        h = F.relu(net.fc2(F.relu(net.fc1(x))))
                        sg = net.sigma_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
                        mu = net.mu_head(h)
                        ex.update(sigma_mean=float(sg.mean()),
                                  sigma_p10=float(torch.quantile(sg.reshape(-1), 0.1)),
                                  sigma_p90=float(torch.quantile(sg.reshape(-1), 0.9)),
                                  mu_abs_max=float(mu.abs().max()))
                    elif arm == "L":
                        mu = net(x)
                        ex.update(mu_abs_max=float(mu.abs().max()),
                                  mu_abs_p90=float(torch.quantile(mu.abs().reshape(-1), 0.9)))
                do_eval(step, ex)
        name = f"{arm}-{tag}-s{cfg['seed']}"
        d = os.path.join(cfg["out"], name)
        os.makedirs(d, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(d, "actor_final.pt"))
        with open(os.path.join(d, "log.json"), "w") as f:
            json.dump({"config": {**cfg, "arm": arm, "target_space": ARMS[arm][0],
                                  "scale": ARMS[arm][1], "ckpt": ck,
                                  "critic_alpha": ccfg.get("alpha")}, "logs": logs}, f)
        del net, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
