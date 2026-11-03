"""OMIGA's multi-agent observation on gym 0.10.8 + mujoco-py 2.0 (MuJoCo 2.0), the stack that generated the datasets.

Same construction as ma_mujoco.py (MuJoCo 3): every agent observes the full single-agent observation plus its one-hot
id, standardised within the vector. The joint action is the per-agent actions concatenated in agent order and passed
to the env unchanged. Validated by one-step replay of dataset transitions (set_from_raw + step reproduces the next
stored observation to about 1e-7). Requires the evaluation environment (envs/eval_mj200.yml).
"""
import numpy as np
import gym

SPEC = {"Hopper-v2": (3, 1, 11), "HalfCheetah-v2": (6, 1, 17), "Ant-v2": (2, 4, 111)}   # n agents, act per agent, raw obs


class OmigaMJ200:
    def __init__(self, scenario):
        self.scenario = scenario; self.n, self.act, self.raw_dim = SPEC[scenario]
        self.env = gym.make(scenario)
        self.obs_dim = self.raw_dim + self.n

    def seed(self, s): self.env.seed(s)

    def obs_from_raw(self, raw):
        out = []
        for a in range(self.n):
            e = np.zeros(self.n); e[a] = 1.0; o = np.concatenate([raw, e]); out.append(((o - o.mean()) / o.std()).astype(np.float32))
        return out

    def reset(self): return self.obs_from_raw(np.asarray(self.env.reset(), dtype=np.float64).ravel())

    def step(self, acts):
        flat = np.concatenate([np.asarray(a, dtype=np.float64).ravel()[: self.act] for a in acts])
        raw, r, done, info = self.env.step(flat)
        return self.obs_from_raw(np.asarray(raw, dtype=np.float64).ravel()), r, done, info

    # --- restoring the simulator at a dataset state (critic diagnostics) ---
    def raw_from_obs(self, o_agent, agent):
        """Invert the per-vector standardisation using the known one-hot entries (1 at `agent`, 0 elsewhere)."""
        ids = o_agent[self.raw_dim:]; one = ids[agent]; zero = ids[(agent + 1) % self.n]
        sd = 1.0 / (one - zero); m = -zero * sd
        return o_agent[: self.raw_dim] * sd + m

    def set_from_raw(self, raw):
        u = self.env.unwrapped; nq = u.model.nq; nv = u.model.nv
        if self.scenario == "HalfCheetah-v2":
            qpos = np.concatenate([[0.0], raw[:nq - 1]]); qvel = raw[nq - 1:nq - 1 + nv]
        elif self.scenario == "Ant-v2":
            qpos = np.concatenate([[0.0, 0.0], raw[:nq - 2]]); qvel = raw[nq - 2:nq - 2 + nv]
        else:
            qpos = np.concatenate([[0.0], raw[:nq - 1]]); qvel = raw[nq - 1:nq - 1 + nv]
        u.set_state(qpos, qvel)
