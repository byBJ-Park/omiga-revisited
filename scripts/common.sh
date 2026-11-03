# Shared settings of the drivers. Override on the command line, e.g. GPUS="0 1" PAR=4 bash scripts/train_critics.sh
# PY       python of the training environment (envs/train.yml)
# PY_EVAL  python of the evaluation environment (envs/eval_mj200.yml), with MuJoCo 2.0 on LD_LIBRARY_PATH
# GPUS     GPU ids the training jobs are spread over (round robin)
# PAR      jobs per GPU (training) or in total (evaluation, CPU)
# SEEDS    training seeds
cd "$(dirname "$0")/.."
export PY=${PY:-python} PY_EVAL=${PY_EVAL:-python} GPUS=${GPUS:-0} PAR=${PAR:-3} SEEDS=${SEEDS:-"0 1 2 3 4"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p checkpoints results logs

CELLS="Hopper-v2-3x1-expert Hopper-v2-3x1-medium-expert Hopper-v2-3x1-medium Hopper-v2-3x1-medium-replay \
HalfCheetah-v2-6x1-expert HalfCheetah-v2-6x1-medium-expert HalfCheetah-v2-6x1-medium HalfCheetah-v2-6x1-medium-replay \
Ant-v2-2x4-expert Ant-v2-2x4-medium-expert Ant-v2-2x4-medium Ant-v2-2x4-medium-replay"

# cell -> "scenario agent_conf data_type"
split_cell() { local c=$1; local sc=${c%%-v2-*}-v2; local rest=${c#*-v2-}; echo "$sc ${rest%%-*} ${rest#*-}"; }

# run_jobs <lines of "job args..."> : each line is run by the function `job`, PAR at a time per GPU, round robin
run_jobs() {
  local n=0 gpus=($GPUS)
  while read -r line; do
    [ -n "$line" ] || continue
    local g=${gpus[$((n % ${#gpus[@]}))]}; n=$((n + 1))
    while [ "$(jobs -rp | wc -l)" -ge $((PAR * ${#gpus[@]})) ]; do sleep 10; done
    CUDA_VISIBLE_DEVICES=$g job $line &
    sleep 2
  done
  wait
}
