"""Shared pieces of the critic diagnostics (Appendix E): dataset rows, the neighbour pool, and CPU copies of the
frozen OMIGA critic and of a deterministic actor. Runs in the evaluation environment (MuJoCo 2.0)."""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.mj200_env import OmigaMJ200  # noqa: F401  (re-exported for the diagnostic scripts)


class Critic(nn.Module):
    """OMIGA's Q_i (shared, one-hot id) and abs mixer: Q_tot = sum_i |w_i(s)| Q_i(o_i, a_i) + b(s), for one state."""

    def __init__(self, od, ad, n, h=256, mh=64):
        super().__init__()
        self.q = nn.ModuleDict({"fc1": nn.Linear(od + ad + n, h), "fc2": nn.Linear(h, h), "state_value": nn.Linear(h, 1)})
        self.m = nn.ModuleDict({"f_v": nn.Linear(od * n, mh), "w_v": nn.Linear(mh, n), "b_v": nn.Linear(mh, 1)})
        self.eye = torch.eye(n)

    def forward(self, o, a):                                     # o [n, od], a [n, ad] -> scalar
        x = torch.cat([o, a, self.eye], -1)
        qi = self.q.state_value(F.relu(self.q.fc2(F.relu(self.q.fc1(x)))))      # [n, 1]
        z = F.relu(self.m.f_v(o.reshape(1, -1)))
        return (self.m.w_v(z).abs().reshape(-1, 1) * qi).sum() + self.m.b_v(z).sum()


class BC(nn.Module):
    """Deterministic actor (DetActor of omiga_bc.py) for one state."""

    def __init__(self, od, ad, n, h=256):
        super().__init__()
        self.fc1 = nn.Linear(od + n, h); self.fc2 = nn.Linear(h, h); self.head = nn.Linear(h, ad); self.eye = torch.eye(n)

    def forward(self, o):                                        # o [n, od] -> [n, ad]
        return torch.tanh(self.head(F.relu(self.fc2(F.relu(self.fc1(torch.cat([o, self.eye], -1)))))))


def load_data(path):
    with h5py.File(path, "r") as f:
        O = f["o"][:].astype(np.float32); A = f["a"][:].astype(np.float32); D = f["d"][:].ravel() > 0.5
    return O, A, D


def eligible(D):
    """Non-end rows whose episode either terminates (fewer than 1000 steps) or has at least 300 steps left."""
    ends = np.nonzero(D)[0]; starts = np.r_[0, ends[:-1] + 1]; keep = np.zeros(len(D), bool)
    for s, e in zip(starts, ends):
        rows = np.arange(s, e)
        keep[rows] = True if (e - s) < 1000 else (e - rows) >= 300
    return np.nonzero(keep)[0]


class Pool:
    """20,000 eligible transitions, standardized joint observation, for nearest-neighbour lookups."""

    def __init__(self, O, A, idx):
        self.idx = idx; o = O[idx].reshape(len(idx), -1)
        self.mu, self.sd = o.mean(0), o.std(0) + 1e-6
        self.Z = torch.as_tensor((o - self.mu) / self.sd); self.A = A[idx]

    def nn(self, o_joint, k):
        z = torch.as_tensor((np.asarray(o_joint, np.float32).reshape(len(o_joint), -1) - self.mu) / self.sd)
        return torch.cdist(z, self.Z).topk(k, largest=False)[1].numpy()


def load_critic(ckpt, od, ad, n):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["networks"]
    cr = Critic(od, ad, n); cr.q.load_state_dict(st["_q_network"]); cr.m.load_state_dict(st["_mix_network"]); cr.eval()
    for p in cr.parameters():
        p.requires_grad_(False)
    return cr


def load_actor(path, od, ad, n):
    m = BC(od, ad, n); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
