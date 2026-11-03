#!/usr/bin/env bash
# Evaluate every saved actor in MuJoCo 2.0 (100 episodes, seeds 50000-50099) into results/*.jsonl. CPU only, PAR jobs
# at a time, in the evaluation environment (PY_EVAL). Records already in a file are skipped, so the script can be rerun.
#   results/bc_eval_mj200.jsonl                checkpoints/bc                 kind det
#   results/omiga_bc_eval_mj200.jsonl          checkpoints/omiga_bc/eta*      kind det
#   results/factorial_target_eval_mj200.jsonl  checkpoints/factorial_target   A gauss, L latmse, G actgauss, B det
#   results/factorial_weight_eval_mj200.jsonl  checkpoints/factorial_weight   A, C gauss; B, D det
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=
kind_of() { case $1 in A|C) echo gauss;; L) echo latmse;; G) echo actgauss;; *) echo det;; esac; }
{
  for d in checkpoints/bc/B-*-s[0-9]; do [ -f $d/actor_final.pt ] || continue
    n=$(basename $d); c=${n#B-}; c=${c%-s*}; echo "results/bc_eval_mj200.jsonl $c det:$d/actor_final.pt::$n"; done
  for d in checkpoints/omiga_bc/eta*/TD3BC-*-s[0-9]; do [ -f $d/actor_final.pt ] || continue
    e=$(basename $(dirname $d)); e=${e#eta}; n=$(basename $d); c=${n#TD3BC-}; c=${c%-s*}; s=${n##*-s}
    echo "results/omiga_bc_eval_mj200.jsonl $c det:$d/actor_final.pt::$c-eta$e-s$s"; done
  for k in target weight; do for d in checkpoints/factorial_$k/[A-Z]-*-s[0-9]; do [ -f $d/actor_final.pt ] || continue
    n=$(basename $d); a=${n%%-*}; c=${n#*-}; c=${c%-s*}
    echo "results/factorial_${k}_eval_mj200.jsonl $c $(kind_of $a):$d/actor_final.pt::$n"; done; done
} | xargs -P $PAR -L 1 bash -c 'flock logs/.lock.$(basename $0) $PY_EVAL -m omiga_rev.eval_mj200 --out $0 --cell $1 "$2"'
# the records carry eta/seed/arm fields for the analysis scripts
$PY_EVAL - <<'EOF'
import json, re
for name, pat, extra in (("bc", r"B-(?P<cell>.+)-s(?P<seed>\d)$", {}),
                         ("omiga_bc", r"(?P<cell>.+)-eta(?P<eta>[\d.]+)-s(?P<seed>\d)$", {"eta": float}),
                         ("factorial_target", r"(?P<arm>[A-Z])-(?P<cell>.+)-s(?P<seed>\d)$", {}),
                         ("factorial_weight", r"(?P<arm>[A-Z])-(?P<cell>.+)-s(?P<seed>\d)$", {})):
    path = f"results/{name}_eval_mj200.jsonl"
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        continue
    for r in rows:
        m = re.match(pat, r["run"]).groupdict()
        r["seed"] = int(m["seed"])
        if "arm" in m: r["arm"] = m["arm"]
        if "eta" in m: r["eta"] = float(m["eta"])
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))
EOF
