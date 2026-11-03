"""State-conditional action spread sigma_k of each dataset (Appendix E, Fig. 4 and Table 9).

For a sample of 20,000 transitions and 2,000 query states drawn from it, sigma_k is the RMS deviation of the logged
joint actions of the k nearest states (standardized joint observation, the query itself excluded) from their mean.
The paper reports k = 10 and sample seed 0; k in {5, 10, 20, 50} and seeds 0-2 are computed to show that the grouping
of the datasets does not depend on these choices.

Run in the training environment:
  python -m omiga_rev.diagnostics.spread --data_dir data --out results/action_spread.json
"""
import argparse, json
import numpy as np, torch

CELLS = [f"{t}-{q}" for t in ("Hopper-v2-3x1", "HalfCheetah-v2-6x1", "Ant-v2-2x4") for q in ("expert", "medium-expert", "medium", "medium-replay")]
SMALL = {c for c in CELLS if c.startswith("Ant")} | {"Hopper-v2-3x1-medium-replay"}
KS, SEEDS = [5, 10, 20, 50], [0, 1, 2]


def spreads(O, A, seed):
    """sigma_k for every k in KS from one sample (20k transitions, 2k queries) drawn with `seed`."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(O), min(20000, len(O)), replace=False); o, a = O[idx], A[idx]
    Z = torch.as_tensor((o - o.mean(0)) / (o.std(0) + 1e-6)); At = torch.as_tensor(a)
    q = torch.as_tensor(rng.choice(len(o), 2000, replace=False))
    _, nn_i = torch.cdist(Z[q], Z).topk(max(KS) + 1, largest=False)
    out = {}
    for k in KS:
        neigh = At[nn_i[:, 1:k + 1]]                                   # drop self
        out[k] = (neigh - neigh.mean(1, keepdim=True)).pow(2).mean().sqrt().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data"); p.add_argument("--out", default="results/action_spread.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    res = {}
    for c in CELLS:
        import h5py
        with h5py.File(f"{a.data_dir}/{c}.hdf5", "r") as f:
            O = f["o"][:].astype(np.float32); A = f["a"][:].astype(np.float32)
        O = O.reshape(len(O), -1); A = A.reshape(len(A), -1)      # all rows, including the episode-end rows
        res[c] = {s: spreads(O, A, s) for s in SEEDS}
        print(c, {s: {k: round(v, 3) for k, v in res[c][s].items()} for s in SEEDS}, flush=True)
    sep = {}
    for k in KS:
        for s in SEEDS:
            lo = max(res[c][s][k] for c in SMALL); hi = min(res[c][s][k] for c in CELLS if c not in SMALL)
            sep[f"k{k}-s{s}"] = (lo, hi, hi / lo)
    json.dump({"spread": res, "separation": sep, "grouping_holds": all(lo < hi for lo, hi, _ in sep.values())},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
