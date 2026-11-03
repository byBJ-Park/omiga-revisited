"""Which part of OMIGA's actor limits it: the weighting, or the loss and parametrisation?

A 2x2 on one frozen OMIGA checkpoint. The data-selection criterion (OMIGA's advantage
weight) is held fixed down one axis; the way the actor is trained to fit the selected
actions changes down the other.

                      uniform weight        OMIGA advantage weight
    tanh-Gaussian NLL   A  plain NLL BC      C  OMIGA's own actor objective
    deterministic MSE   B  plain MSE BC      D  weighted MSE

    A  L = -E[ log pi(a_D | o) ]
    B  L =  E[ ||pi(o) - a_D||^2 ]
    C  L = -E[ W log pi(a_D | o) ]          <- omiga.py policy_loss, reproduced exactly
    D  L =  E[ W ||pi(o) - a_D||^2 ]

W is taken from the checkpoint, not re-derived:

    z = (1/alpha) * (w_target * Q_target(o, a_D, id) - w_target * V(o, id))
    z = clamp(z, -10, 10)
    W = exp(z).detach()                      # omiga.py lines 165-168

with w_target from the target mixer and V from the value network, all frozen. So C is
OMIGA's actor objective computed against OMIGA's own trained critic, and D differs from
C in the loss and the policy parametrisation only.

What this is and is not. It is a policy re-training experiment on a frozen critic, not
OMIGA: the real algorithm updates critic and policy together, so C is not guaranteed to
reproduce the checkpoint's own policy and the difference is reported rather than assumed.
D is a control that keeps OMIGA's value learning and weighting while changing the actor,
not "OMIGA with a bug fixed" -- swapping NLL for MSE changes what the actor is.

Every arm gets the same states, the same batch index stream, the same budget, the same
learning rate and the same evaluation, and each arm is run against each critic seed so
the comparison is checkpoint-paired rather than best-row against best-row.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import ReplayBuffer
from .networks import Actor, MixNet, Q_critic, V_critic
from .omiga_bc import DetActor, find_checkpoint
from .train_omiga import evaluate, make_env

ARMS = {"A": ("nll", False), "B": ("mse", False), "C": ("nll", True), "D": ("mse", True)}


class Roll:
    """run_omiga.evaluate's .step() interface, for both actor families."""

    def __init__(self, net, kind, n, device):
        self.net, self.kind, self.device = net, kind, device
        self.eye = torch.eye(n, device=device)

    @torch.no_grad()
    def step(self, o):
        o = torch.from_numpy(o).to(self.device)
        x = torch.cat((o, self.eye.expand(o.shape[0], -1, -1)), dim=-1)
        return (self.net(x) if self.kind == "mse" else
                self.net.get_deterministic_action(x)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--ckpt", required=True, help="a run_omiga results/<tag> dir or ckpt_final.pt")
    p.add_argument("--scenario", required=True)
    p.add_argument("--agent_conf", required=True)
    p.add_argument("--data_type", required=True)
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
    p.add_argument("--outer_clamp", type=float, default=0.0,
                   help="NLL arms: clamp the target action to +-(1 - this) BEFORE "
                        "get_log_density, which clamps again at 1e-7. omiga.py passes the "
                        "raw action (use 0.0); scripts/pct_bc.py clamps at 1e-6 first. In "
                        "float32 the latent target is 8.318 with no outer clamp and 7.248 "
                        "with 1e-6, so this is not a cosmetic difference.")
    p.add_argument("--data_on_gpu", type=int, default=0, choices=[0, 1])
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
    if cfg["data_on_gpu"]:
        data.to_device()
    N = data.o.shape[0]
    host = not cfg["data_on_gpu"]

    # ---- frozen critic pieces, exactly the ones omiga.py reads for the weight --------
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
    alpha = float(ccfg.get("alpha", 10.0))
    eye = torch.eye(n, device=device)

    @torch.no_grad()
    def omiga_weight(o, s, a):
        one = eye.expand(o.shape[0], -1, -1)
        q_t = qt(torch.cat((o, a, one), dim=-1))          # [B, n, 1]
        w_t, _ = mt(s)                                    # [B, n, 1]
        v = vn(torch.cat((o, one), dim=-1))               # [B, n, 1]
        z = (1.0 / alpha) * (w_t * q_t - w_t * v)
        z = torch.clamp(z, min=-10.0, max=10.0)
        return torch.exp(z).squeeze(-1), z.squeeze(-1)    # [B, n]

    results = {}
    for arm in cfg["arms"]:
        kind, weighted = ARMS[arm]
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        net = (DetActor(obs_dim, act_dim, n, cfg["hidden"]) if kind == "mse"
               else Actor(obs_dim, act_dim, n, cfg["hidden"], device)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        roll = Roll(net, kind, n, device)
        bg = torch.Generator().manual_seed(4242 + cfg["seed"])   # identical across arms
        logs, t0 = [], time.time()

        @torch.no_grad()
        def roll_diag(episodes=5, max_steps=1000):
            """Saturation rate and episode length in the states the policy reaches.

            false_sat is measured on dataset states, where the logged action is known. At
            execution there is no reference action, so what is recorded here is the share
            of executed torques at the box edge and how long the episode lasted.
            """
            net.eval()
            sats, lens = [], []
            for _ in range(episodes):
                obs, _, _ = env.reset()
                t = 0
                for t in range(max_steps):
                    act = roll.step(np.asarray(obs, dtype=np.float32)[None]).numpy()[0]
                    sats.append(float((np.abs(act) > 0.99).mean()))
                    obs, _, _, done, _, _ = env.step(act)
                    if done[0]:
                        break
                lens.append(t + 1)
            net.train()
            return {"roll_sat": float(np.mean(sats)), "roll_len": float(np.mean(lens))}

        def do_eval(step, extra):
            net.eval()
            ep_r, ep_sd = evaluate(roll, env, cfg["num_evaluation"])
            net.train()
            rec = {"step": step, "ep_r": ep_r, "ep_r_std": ep_sd,
                   "wall": time.time() - t0, **extra}
            logs.append(rec)
            print(f"[{arm} {tag} s{cfg['seed']}] step={step:>7d} return={ep_r:9.1f} "
                  f"+-{ep_sd:7.1f} " +
                  " ".join(f"{k}={v:.4g}" for k, v in extra.items()) +
                  f"  {rec['wall']/60:.1f}min", flush=True)
            return rec

        do_eval(0, {})
        for step in range(1, cfg["steps"] + 1):
            idx = torch.randint(0, N, (cfg["batch_size"],), generator=bg)
            if host:
                ii = idx.numpy()
                o = torch.as_tensor(np.asarray(data.o)[ii], dtype=torch.float32, device=device)
                s = torch.as_tensor(np.asarray(data.s)[ii], dtype=torch.float32, device=device)
                a = torch.as_tensor(np.asarray(data.a)[ii], dtype=torch.float32, device=device)
            else:
                i2 = idx.to(device)
                o, s, a = data.o[i2], data.s[i2], data.a[i2]
            one = eye.expand(o.shape[0], -1, -1)
            x = torch.cat((o, one), dim=-1)
            W, z = (omiga_weight(o, s, a) if weighted
                    else (torch.ones(o.shape[0], n, device=device),
                          torch.zeros(o.shape[0], n, device=device)))
            if kind == "mse":
                per = ((net(x) - a) ** 2).sum(dim=-1)                 # [B, n]
            else:
                ab = (a.clamp(-1 + cfg["outer_clamp"], 1 - cfg["outer_clamp"])
                      if cfg["outer_clamp"] > 0 else a)
                per = -net.get_log_density(x, ab)                     # [B, n]
            loss = (W * per).mean()
            opt.zero_grad(); loss.backward()
            if cfg["grad_norm_clip"] > 0:
                nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_norm_clip"])
            opt.step()
            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                with torch.no_grad():
                    ad = net(x) if kind == "mse" else net.get_deterministic_action(x)
                    sat_pi = float((ad.abs() > 0.99).float().mean())
                    sat_d = float((a.abs() > 0.99).float().mean())
                    # the diagnostic that separates "copying the expert's extreme torque"
                    # from "saturating where the data did not"
                    false_sat = float(((ad.abs() > 0.99) & (a.abs() <= 0.99)).float().mean())
                    Wf = W.reshape(-1)
                    wq = torch.quantile(Wf, torch.tensor([0.1, 0.5, 0.9], device=Wf.device))
                do_eval(step, {"loss": float(loss),
                               "W_mean": float(W.mean()), "W_sd": float(Wf.std()),
                               "W_p10": float(wq[0]), "W_p50": float(wq[1]),
                               "W_p90": float(wq[2]), "W_max": float(W.max()),
                               "W_frac_lt_0p5": float((Wf < 0.5).float().mean()),
                               "W_frac_lt_0p1": float((Wf < 0.1).float().mean()),
                               "z_clamp_frac": float((z.abs() >= 9.999).float().mean()),
                               "act_err": float((ad - a).norm(dim=-1).mean()),
                               "sat_pi": sat_pi, "sat_data": sat_d,
                               "false_sat": false_sat, **roll_diag()})
        name = f"{arm}-{tag}-s{cfg['seed']}"
        d = os.path.join(cfg["out"], name)
        os.makedirs(d, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(d, "actor_final.pt"))
        with open(os.path.join(d, "log.json"), "w") as f:
            json.dump({"config": {**cfg, "arm": arm, "kind": kind, "weighted": weighted,
                                  "ckpt": ck, "critic_alpha": alpha}, "logs": logs}, f)
        results[arm] = logs[-1]["ep_r"]
        del net, opt
        torch.cuda.empty_cache()
    print(f"[{tag} s{cfg['seed']}] " + "  ".join(f"{k}={v:.1f}" for k, v in results.items()),
          flush=True)


if __name__ == "__main__":
    main()
