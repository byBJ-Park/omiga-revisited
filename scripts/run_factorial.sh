#!/usr/bin/env bash
# The controlled comparison of Section V and Appendix C, seeds 0-2, 10^5 steps.
#   target  (Table 4)  omiga_rev/bc_target.py, arms B A L G (action/latent target x fixed/learned scale), unweighted,
#                      on Hopper E/M-E/M-R, HalfCheetah E/M-E, Ant M -> checkpoints/factorial_target
#   weight  (Table 7)  omiga_rev/bc_weighted.py, arms A B C D (latent NLL / action MSE x unweighted / OMIGA weight),
#                      on Hopper E/M-E/M/M-R, HalfCheetah E/M-E/M, Ant M -> checkpoints/factorial_weight
# Both load the OMIGA checkpoint of the same seed (the weighted arms use its critic).
source "$(dirname "$0")/common.sh"
SEEDS=${SEEDS_FACTORIAL:-"0 1 2"}
TARGET_CELLS="Hopper-v2-3x1-expert Hopper-v2-3x1-medium-expert Hopper-v2-3x1-medium-replay \
HalfCheetah-v2-6x1-expert HalfCheetah-v2-6x1-medium-expert Ant-v2-2x4-medium"
WEIGHT_CELLS="Hopper-v2-3x1-expert Hopper-v2-3x1-medium-expert Hopper-v2-3x1-medium Hopper-v2-3x1-medium-replay \
HalfCheetah-v2-6x1-expert HalfCheetah-v2-6x1-medium-expert HalfCheetah-v2-6x1-medium Ant-v2-2x4-medium"
job() {  # kind cell seed
  read -r sc conf dt <<< "$(split_cell $2)"
  if [ $1 = target ]; then
    $PY -m omiga_rev.bc_target --arms B A L G --weighted 0 --ckpt checkpoints/omiga/OMIGA-$2-s$3 --scenario $sc \
      --agent_conf $conf --data_type $dt --seed $3 --device cuda:0 --steps 100000 --out checkpoints/factorial_target
  else
    $PY -m omiga_rev.bc_weighted --arms A B C D --ckpt checkpoints/omiga/OMIGA-$2-s$3 --scenario $sc \
      --agent_conf $conf --data_type $dt --seed $3 --device cuda:0 --steps 100000 --out checkpoints/factorial_weight
  fi > logs/factorial_$1-$2-s$3.log 2>&1 && echo "[done] $1 $2 s$3" || echo "[FAIL] $1 $2 s$3"
}
{ for c in $TARGET_CELLS; do for s in $SEEDS; do echo "target $c $s"; done; done
  for c in $WEIGHT_CELLS; do for s in $SEEDS; do echo "weight $c $s"; done; done; } | run_jobs
