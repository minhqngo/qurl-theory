#!/usr/bin/env bash
# S2 (P1 note, section 7): exact test of Theorem 2(b)-(c) at s = 2 on Qwen2.5-0.5B-Instruct (S3's model).
#   bash ~/qurl/s2/s2_run.sh <config_tag> [s2_train.py arguments, applied to every arm]
#   e.g. bash ~/qurl/s2/s2_run.sh lr1e-6 --lr 1e-6
#        bash ~/qurl/s2/s2_run.sh lr1e-5 --lr 1e-5   (weight decay active: lr*wd = 1e-7 > 2^-25)
# Arms, run in sequence (finished arms are skipped on re-run):
#   R4         UAQ, plain optimizer settings              (reference)
#   R4_repeat  identical to R4: checks the whole pipeline is deterministic run to run
#   R5         no UAQ, Theorem 2(b)-(c) settings          (prediction: bit-identical to R4 at every step)
#   R5o        as R5 but ordinary clipping                (isolates rule (c): drifts whenever clipping binds)
#   R5x        no UAQ, learning rates matched only        (should drift: eps, weight decay, clipping unmatched)
#   R3         no UAQ, plain settings                     (should differ)
# Outputs: ~/qurl/s2/results/<config_tag>/<arm>/{log.jsonl,params_step*.pt}, <arm>.log, compare.txt
set -euo pipefail
TAG=$1; shift
S2="$HOME/qurl/s2"
RES="$S2/results/$TAG"
PY="$HOME/venvs/qurl/bin/python"
mkdir -p "$RES"
echo "== S2 $TAG ($*) $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee "$RES/gpu.txt"

run() {
  local arm_tag=$1; shift
  if [ -f "$RES/$arm_tag/done" ]; then echo "skip $arm_tag (done)"; return 0; fi
  echo "-- $arm_tag $(date -Is)"
  "$PY" "$S2/s2_train.py" --out "$RES/$arm_tag" "$@" > "$RES/$arm_tag.log" 2>&1
  touch "$RES/$arm_tag/done"
  tail -n 1 "$RES/$arm_tag.log"
}
run R4        --arm R4  "$@"
run R4_repeat --arm R4  "$@"
run R5        --arm R5  "$@"
run R5o       --arm R5o "$@"
run R5x       --arm R5x "$@"
run R3        --arm R3  "$@"
"$PY" "$S2/s2_compare.py" "$RES" --ref R4 | tee "$RES/compare.txt"
echo "== done $TAG $(date -Is)"
