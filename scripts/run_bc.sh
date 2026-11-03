#!/usr/bin/env bash
# Behaviour cloning of Table 3: deterministic actor, squared error, 10^5 steps, seeds 0-4 on the 12 datasets
# (omiga_rev/bc_weighted.py, arm B). The script loads the OMIGA checkpoint of the same seed to read its config; the
# unweighted arm B never uses the critic. Output: checkpoints/bc/B-<cell>-s<seed>/actor_final.pt.
source "$(dirname "$0")/common.sh"
job() {  # cell seed
  local d=checkpoints/bc/B-$1-s$2
  [ -f $d/actor_final.pt ] && { echo "[skip] $d"; return; }
  read -r sc conf dt <<< "$(split_cell $1)"
  $PY -m omiga_rev.bc_weighted --arms B --ckpt checkpoints/omiga/OMIGA-$1-s$2 --scenario $sc --agent_conf $conf \
    --data_type $dt --seed $2 --device cuda:0 --steps 100000 --out checkpoints/bc > logs/bc-$1-s$2.log 2>&1 \
    && echo "[done] $d" || echo "[FAIL] $d"
}
for c in $CELLS; do for s in $SEEDS; do echo "$c $s"; done; done | run_jobs
