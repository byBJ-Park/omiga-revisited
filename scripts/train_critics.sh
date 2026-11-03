#!/usr/bin/env bash
# OMIGA on the 12 datasets, seeds 0-4 (Section VI-B): 10^6 steps, released hyperparameters, alpha 10.
# Output: checkpoints/omiga/OMIGA-<cell>-s<seed>/{ckpt_final.pt, log.json}. About 10 GPU-hours per run.
source "$(dirname "$0")/common.sh"
job() {  # cell seed
  local d=checkpoints/omiga/OMIGA-$1-s$2
  [ -f $d/ckpt_final.pt ] && { echo "[skip] $d"; return; }
  read -r sc conf dt <<< "$(split_cell $1)"
  $PY -m omiga_rev.train_omiga --scenario $sc --agent_conf $conf --data_type $dt --seed $2 --device cuda:0 \
    --total_iterations 1000000 --log_iterations 5000 --num_evaluation 10 --ckpt_iterations 100000 \
    --out_dir checkpoints/omiga > logs/omiga-$1-s$2.log 2>&1 && echo "[done] $d" || echo "[FAIL] $d"
}
for c in $CELLS; do for s in $SEEDS; do echo "$c $s"; done; done | run_jobs
