#!/usr/bin/env bash
# OMIGA+BC (Section VI): on the frozen OMIGA critic of each seed, one actor with the same seed for every
# eta in {0, 2.5, 5, 10, 40}, 3 x 10^5 steps. Output: checkpoints/omiga_bc/eta<eta>/TD3BC-<cell>-s<seed>/actor_final.pt.
source "$(dirname "$0")/common.sh"
ETAS=${ETAS:-"0 2.5 5 10 40"}
job() {  # cell seed eta
  local d=checkpoints/omiga_bc/eta$3/TD3BC-$1-s$2
  [ -f $d/actor_final.pt ] && { echo "[skip] $d"; return; }
  read -r sc conf dt <<< "$(split_cell $1)"
  $PY -m omiga_rev.omiga_bc --ckpt checkpoints/omiga/OMIGA-$1-s$2 --scenario $sc --agent_conf $conf --data_type $dt \
    --seed $2 --eta $3 --device cuda:0 --total_iterations 300000 --out_dir checkpoints/omiga_bc/eta$3 \
    > logs/omiga_bc-$1-s$2-eta$3.log 2>&1 && echo "[done] $d" || echo "[FAIL] $d"
}
for e in $ETAS; do for c in $CELLS; do for s in $SEEDS; do echo "$c $s $e"; done; done; done | run_jobs
