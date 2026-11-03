#!/usr/bin/env bash
# Critic diagnostics of Appendix E: the action spread (training environment, CPU), then the over-estimation and
# deviation records for every (dataset, seed, eta) in the evaluation environment, and Table 9.
# Ant medium-replay seeds 1-3 are skipped: those three critics diverged during training (|Q_tot| ~ 1e10).
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=
[ -f results/action_spread.json ] || $PY -m omiga_rev.diagnostics.spread --data_dir data --out results/action_spread.json
mkdir -p results/diagnostics/overestimation
for c in $CELLS; do for s in $SEEDS; do
  [ $c = Ant-v2-2x4-medium-replay ] && [ $s -ge 1 ] && [ $s -le 3 ] && continue
  for e in 0 2.5 5 10 40; do echo "$c $s $e"; done
done; done | xargs -P $PAR -L 1 bash -c '
  out=results/diagnostics/overestimation/$0-s$1-eta$2.jsonl; [ -s $out ] && exit 0
  $PY_EVAL -m omiga_rev.diagnostics.overestimation --cell $0 --seed $1 --eta $2 \
    --critic checkpoints/omiga/OMIGA-$0-s$1/ckpt_final.pt --actor checkpoints/omiga_bc/eta$2/TD3BC-$0-s$1/actor_final.pt \
    --data_dir data --spread results/action_spread.json --out $out'
$PY -m omiga_rev.diagnostics.analyze
