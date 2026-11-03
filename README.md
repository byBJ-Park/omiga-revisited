# Same Critic, Different Policy: Revisiting OMIGA in Offline Multi-Agent Reinforcement Learning

Code and evaluation records for the paper (B. Park, U. Jung, K. Jang, 2026, manuscript under review).

The paper re-examines OMIGA (Wang et al., NeurIPS 2023) on the twelve Multi-Agent MuJoCo datasets released with it:

* behavior cloning, evaluated in the simulator version that generated the datasets, matches or exceeds the
  published OMIGA results on most datasets (Table 3).
* the cause is OMIGA's policy extraction: it fits a tanh-squashed Gaussian to the dataset actions through
  `atanh`, and the many logged actions at or beyond the action bound all collapse onto the same latent target
  ±8.32 (Fig. 3, Table 4).
* **OMIGA+BC** keeps OMIGA's critic frozen and replaces only the extraction with a TD3+BC objective in the action
  space. This single change recovers much of the gap to recent methods (Table 5, Table 8).

Every table and Figs. 1-4 of the paper can be regenerated from the records in `results/` without a GPU
(`scripts/make_paper.sh`). Every experiment can be rerun from the scripts in `scripts/`.

## Layout

```
omiga_rev/                  the code
  omiga.py, networks.py, dataset.py     OMIGA, ported from the released code (commit 39851c0), unchanged
  envs/ma_mujoco.py, obsk.py            Multi-Agent MuJoCo wrapper on gymnasium + MuJoCo 3 (training-time evaluation only)
  envs/mj200_env.py                     the same observation on gym 0.10.8 + mujoco-py 2.0 (every reported return)
  train_omiga.py                        OMIGA critic (and OMIGA's own policy), 10^6 steps
  bc_weighted.py                        behavior cloning (Table 3) and the weighting experiment (Table 7)
  bc_target.py                          regression target x loss (Table 4)
  omiga_bc.py                           OMIGA+BC: TD3+BC actor on the frozen critic (Tables 5, 8)
  eval_mj200.py                         evaluation protocol: MuJoCo 2.0, 100 episodes, initial-state seeds 50000-50099
  diagnostics/                          action spread, over-estimation and deviation (Appendix E)
scripts/                    drivers for the experiments (GPU) and the evaluation (CPU)
analysis/                   one script per table or figure, reading results/
results/                    evaluation records of every policy in the paper (jsonl), dataset statistics, diagnostics
paper/tables, paper/figures what the analysis scripts write
envs/                       the two conda environments
```

Two environments are needed, because the released datasets were generated with MuJoCo 2.0 and the Ant observation
(contact forces) differs under MuJoCo 3:

* `envs/train.yml` (Python 3.11, PyTorch, gymnasium 1.3, mujoco 3.11): training and analysis.
* `envs/eval_mj200.yml` (Python 3.9, gym 0.10.8, mujoco-py 2.0.2.13, MuJoCo 2.0): evaluation and the diagnostics.
  The comments in that file give the five install steps (MuJoCo 2.0 binaries, `LD_LIBRARY_PATH`, the environment,
  a `libOSMesa.so` link, then `pip install --no-build-isolation mujoco-py==2.0.2.13`).

## Data

The twelve Multi-Agent MuJoCo datasets are the ones released with OMIGA. Download them from the OMIGA authors'
Tsinghua Cloud folder, https://cloud.tsinghua.edu.cn/d/dcf588d659214a28a777/ (linked from
https://github.com/ZhengYinan-AIR/OMIGA), and place the hdf5 files in `data/` with their original names:

```
data/Hopper-v2-3x1-{expert,medium-expert,medium,medium-replay}.hdf5
data/HalfCheetah-v2-6x1-{expert,medium-expert,medium,medium-replay}.hdf5
data/Ant-v2-2x4-{expert,medium-expert,medium,medium-replay}.hdf5
```

They are read as released, nothing is converted. `data/` is git-ignored.

## Reproducing the paper from the records

```
conda env create -f envs/train.yml && conda activate omiga-train
bash scripts/make_paper.sh
```

| Paper | Script | Input |
|---|---|---|
| Fig. 1, Fig. 2 | `analysis/make_fig_dataset.py` | `data/` |
| Fig. 3 | `analysis/make_fig_action_latent.py` | `data/` |
| Table 3 | `analysis/make_table_bc.py` | `results/bc_eval_mj200.jsonl`, `results/dataset_summary.json` |
| Table 4, Table 7 | `analysis/make_table_factorial.py` | `results/factorial_{target,weight}_eval_mj200.jsonl`, `data/` |
| Table 5 | `analysis/make_table_compare.py` | `results/{bc,omiga_bc}_eval_mj200.jsonl` |
| Table 8 | `analysis/make_table_eta.py` | `results/omiga_bc_eval_mj200.jsonl` |
| Table 9, Fig. 4 | `omiga_rev/diagnostics/analyze.py`, `analysis/make_fig_spread.py` | `results/action_spread.json`, `results/diagnostics/` |

Each record in `results/*_eval_mj200.jsonl` holds the 100 episode returns and lengths of one policy, so other
statistics can be computed from them. `results/dataset_summary.json` holds the dataset statistics of Section IV
(episode-return statistics and the share of out-of-bound actions of each dataset, written by `make_fig_dataset.py`).
`data/` is needed only for the dataset figures and the out-of-range column of Tables 4 and 7.

## Rerunning the experiments

Every driver in `scripts/` runs one experiment over all its datasets and seeds: `bash scripts/<name>.sh` is the whole
experiment. The drivers read a few environment variables (`scripts/common.sh`): `PY` (python of the training
environment, default `python`), `PY_EVAL` (python of the evaluation environment), `GPUS` (GPU ids, default `0`),
`PAR` (jobs per GPU, default 3), `SEEDS` (default `0 1 2 3 4`). Runs whose checkpoint already exists are skipped,
logs go to `logs/`. For example, `GPUS="0 1" PAR=4 bash scripts/run_bc.sh` trains the 60 behavior-cloning actors
four at a time on each of two GPUs.

**Step 0: the OMIGA critics** (needed by every later step: the critic is frozen and reused, and the other scripts
read the dataset config from the checkpoint of the same seed)

```
bash scripts/train_critics.sh      # OMIGA, 12 datasets x seeds 0-4, 10^6 steps, released hyperparameters, alpha 10
                                   # -> checkpoints/omiga/OMIGA-<cell>-s<seed>/ckpt_final.pt   (~10 GPU-hours per run)
```

**Table 3: behavior cloning**

```
bash scripts/run_bc.sh             # deterministic actor, squared error, 10^5 steps, 12 datasets x seeds 0-4
                                   # -> checkpoints/bc/B-<cell>-s<seed>/actor_final.pt
bash scripts/eval_mj200.sh         # -> results/bc_eval_mj200.jsonl
python analysis/make_table_bc.py   # -> paper/tables/table_bc.tex
```

**Table 4 and Table 7: the controlled comparison** (regression target x loss, and the OMIGA weighting, seeds 0-2)

```
bash scripts/run_factorial.sh             # bc_target.py arms B A L G on 6 datasets -> checkpoints/factorial_target
                                          # bc_weighted.py arms A B C D on 8 datasets -> checkpoints/factorial_weight
bash scripts/eval_mj200.sh                # -> results/factorial_{target,weight}_eval_mj200.jsonl
python analysis/make_table_factorial.py   # -> paper/tables/table_factorial_{target,weight}.tex
```

**Table 5 and Table 8: OMIGA+BC**

```
bash scripts/run_omiga_bc.sh              # TD3+BC actor on the frozen critic of the same seed, eta in {0, 2.5, 5, 10, 40},
                                          # 3 x 10^5 steps, 12 datasets x seeds 0-4 -> checkpoints/omiga_bc/eta<eta>/
bash scripts/eval_mj200.sh                # -> results/omiga_bc_eval_mj200.jsonl
python analysis/make_table_compare.py     # Table 5 -> paper/tables/table_compare.tex
python analysis/make_table_eta.py         # Table 8 -> paper/tables/table_eta.tex
```

**Table 9 and Fig. 4: critic diagnostics (Appendix E)**

```
bash scripts/run_diagnostics.sh           # action spread (results/action_spread.json), then over-estimation and deviation
                                          # of every (dataset, seed, eta) in MuJoCo 2.0 -> results/diagnostics/, Table 9
python analysis/make_fig_spread.py        # Fig. 4 -> paper/figures/action_spread.png
```

**Fig. 1, Fig. 2, Fig. 3: dataset statistics** (no training, only `data/`)

```
python analysis/make_fig_dataset.py       # -> paper/figures/dataset_{returns,actions}.png, results/dataset_summary.json
python analysis/make_fig_action_latent.py # -> paper/figures/action_latent_Hopper-v2-3x1-expert.png
```

`scripts/eval_mj200.sh` evaluates every actor found under `checkpoints/` that is not yet in `results/` (MuJoCo 2.0,
CPU, 100 episodes, initial-state seeds 50000-50099), so it can be run once after all training or after each step.
It needs the evaluation environment: `PY_EVAL=<python of envs/eval_mj200.yml> PAR=16 bash scripts/eval_mj200.sh`.
`scripts/make_paper.sh` runs all the analysis scripts at once.

Settings are the ones listed in Appendix B of the paper and are the defaults of each script. Three of the five Ant
medium-replay critics diverged during training (|Q_tot| ≈ 1e10). They are kept, and the diagnostics skip them, as
stated in the paper.

Single runs, e.g.

```
python -m omiga_rev.train_omiga --scenario Hopper-v2 --agent_conf 3x1 --data_type medium --seed 0 --out_dir checkpoints/omiga
python -m omiga_rev.omiga_bc --ckpt checkpoints/omiga/OMIGA-Hopper-v2-3x1-medium-s0 --scenario Hopper-v2 --agent_conf 3x1 \
    --data_type medium --seed 0 --eta 5 --out_dir checkpoints/omiga_bc/eta5
python -m omiga_rev.eval_mj200 --out results/my_eval.jsonl --cell Hopper-v2-3x1-medium \
    det:checkpoints/omiga_bc/eta5/TD3BC-Hopper-v2-3x1-medium-s0/actor_final.pt::hopper-medium-eta5-s0      # eval env
```

## Notes on the code

The algorithm, networks and dataset code of OMIGA are the released files. The environment code was replaced
because the released gym 0.17 / mujoco-py stack no longer installs. The critic is trained from the dataset alone,
so it does not depend on the environment. The MuJoCo 3 environment serves only the evaluations logged during
training. All reported returns use the MuJoCo 2.0 environment (`omiga_rev/envs/mj200_env.py`), whose
observation and action mapping were validated by replaying dataset transitions.

## Acknowledgement

Parts of the code and of this README were written with the help of an AI assistant (Claude, Anthropic). All
experiments, checks and results are the authors' own.

## Reference

OMIGA: X. Wang, H. Xu, Y. Zheng, X. Zhan, "Offline multi-agent reinforcement learning with implicit global-to-local
value regularization," NeurIPS 2023. https://github.com/ZhengYinan-AIR/OMIGA
