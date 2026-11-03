"""Train an OMIGA critic (and OMIGA's own policy) on one released dataset.

Equivalent to the released ``run_mujoco.py``, with the legacy gym/mujoco-py vec-env machinery replaced by
``envs.ma_mujoco.MujocoMulti`` (one environment; upstream wrapped one env in ShareDummyVecEnv and only used index
0). The environment is used only for the periodic evaluations written to log.json; training reads the dataset
alone, so the trained critic does not depend on the environment version.

Writes <out_dir>/OMIGA-<scenario>-<conf>-<quality>-s<seed>/{log.json, ckpt_<it>.pt, ckpt_final.pt}. The checkpoint
holds every network (online and target) and optimizer; OMIGA+BC and the factorial scripts load _q_network,
_mix_network, _q_target_network, _mix_target_network and _v_network from it.

Usage (paper setting: 10^6 steps, alpha 10, seeds 0-4):
    python -m omiga_rev.train_omiga --scenario Hopper-v2 --agent_conf 3x1 --data_type medium --seed 0 \
        --out_dir checkpoints/omiga --ckpt_iterations 100000
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .omiga import OMIGA
from .dataset import ReplayBuffer
from .envs.ma_mujoco import MujocoMulti


def make_env(config, seed):
    env = MujocoMulti(env_args={
        "scenario": config["scenario"],
        "agent_conf": config["agent_conf"],
        "agent_obsk": config["agent_obsk"],
        "episode_limit": 1000,
    })
    env.seed(seed)
    return env


@torch.no_grad()
def evaluate(agent, env, num_evaluation=10, max_steps=1000):
    """Mean undiscounted episode return; mirrors upstream utils.util.evaluate."""
    returns = []
    for _ in range(num_evaluation):
        obs, _, _ = env.reset()
        ep_r = 0.0
        for _ in range(max_steps):
            # upstream fed a leading vec-env axis; agent.step expects [B, n_agent, obs]
            a = agent.step(np.asarray(obs, dtype=np.float32)[None]).numpy()[0]
            obs, _, r, done, _, _ = env.step(a)
            ep_r += r[0][0]
            if done[0]:
                break
        returns.append(ep_r)
    return float(np.mean(returns)), float(np.std(returns))


def save_checkpoint(agent, path, iteration):
    """Every nn.Module attribute of the agent and every optimizer, written atomically."""
    state = {"iteration": iteration, "networks": {}, "optimizers": {}}
    for name, obj in vars(agent).items():
        if isinstance(obj, torch.nn.Module):
            state["networks"][name] = obj.state_dict()
    for name, opt in getattr(agent, "_optimizers", {}).items():
        state["optimizers"][name] = opt.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(agent, path, load_optimizers=True):
    """Restore what save_checkpoint wrote. Returns the saved iteration."""
    state = torch.load(path, map_location=agent._device, weights_only=False)
    for name, sd in state["networks"].items():
        obj = getattr(agent, name, None)
        if obj is None:
            print(f"  [ckpt] WARNING: no attribute {name}, skipped")
            continue
        obj.load_state_dict(sd)
    if load_optimizers:
        for name, sd in state["optimizers"].items():
            if name in getattr(agent, "_optimizers", {}):
                agent._optimizers[name].load_state_dict(sd)
    return state["iteration"]


def run(config):
    tag = f"{config['scenario']}-{config['agent_conf']}-{config['data_type']}"
    run_name = f"OMIGA-{tag}-s{config['seed']}"
    out_dir = os.path.join(config["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    env = make_env(config, config["seed"])
    eval_env = make_env(config, config["seed"] + 10000)

    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    n_agents = len(env.observation_space)
    print(f"[{run_name}] obs_dim={obs_dim} action_dim={action_dim} n_agents={n_agents}")

    dataset = ReplayBuffer(obs_dim, action_dim, n_agents, tag, config["data_dir"],
                           device=config["device"])
    dataset.load()
    assert dataset.o.shape[1:] == (n_agents, obs_dim), (
        f"dataset obs {dataset.o.shape[1:]} != env ({n_agents}, {obs_dim})")
    assert dataset.a.shape[1:] == (n_agents, action_dim), (
        f"dataset act {dataset.a.shape[1:]} != env ({n_agents}, {action_dim})")
    print(f"[{run_name}] dataset OK: {dataset.size} transitions")
    if config["dataset_on_gpu"]:
        dataset.to_device()

    agent = OMIGA(obs_dim, action_dim, n_agents, eval_env, config)

    logs, t0 = [], time.time()
    for it in range(config["total_iterations"] + 1):
        batch = dataset.sample(config["batch_size"])
        train_result = agent.train_step(*batch)

        if it % config["log_iterations"] == 0:
            ep_r, ep_std = evaluate(agent, eval_env, config["num_evaluation"])
            rec = {k: float(v.detach().cpu()) for k, v in train_result.items()}
            rec.update({"iteration": it, "ep_r": ep_r, "ep_r_std": ep_std,
                        "elapsed": time.time() - t0})
            logs.append(rec)
            print(f"[{run_name}] it={it:>7d}  return={ep_r:9.1f} +-{ep_std:7.1f}  "
                  f"q_loss={rec['q_loss']:.4f} v_loss={rec['v_loss']:.4f}  ({rec['elapsed']/60:.1f} min)",
                  flush=True)
            with open(os.path.join(out_dir, "log.json"), "w") as f:
                json.dump({"config": config, "logs": logs}, f, indent=1)

        if config["ckpt_iterations"] > 0 and it > 0 and \
                it % config["ckpt_iterations"] == 0:
            save_checkpoint(agent, os.path.join(out_dir, f"ckpt_{it}.pt"), it)

    p = save_checkpoint(agent, os.path.join(out_dir, "ckpt_final.pt"), config["total_iterations"])
    print(f"[{run_name}] saved {p}", flush=True)

    env.close()
    eval_env.close()
    return logs


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, default="Hopper-v2")
    p.add_argument("--agent_conf", type=str, default="3x1")
    p.add_argument("--agent_obsk", type=int, default=1)
    p.add_argument("--data_type", type=str, default="medium")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--out_dir", type=str, default="checkpoints/omiga")
    p.add_argument("--total_iterations", type=int, default=int(1e6))
    p.add_argument("--log_iterations", type=int, default=int(5e3))
    p.add_argument("--num_evaluation", type=int, default=10)
    p.add_argument("--ckpt_iterations", type=int, default=0, help="also save every N steps; 0 = final only")
    p.add_argument("--dataset_on_gpu", type=int, default=1, choices=[0, 1])
    # OMIGA hyperparameters (released defaults; alpha is hardcoded to 10 in the released code)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden_sizes", type=int, default=256)
    p.add_argument("--mix_hidden_sizes", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad_norm_clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=10.0)
    return p


if __name__ == "__main__":
    run(vars(get_parser().parse_args()))
