#!/usr/bin/env bash
# S4: after a run's training, merge its final checkpoint into a Hugging Face model, evaluate it (eval.py -> eval.json),
# then delete the run's FSDP checkpoints (7B: about 90 GB each) unless KEEP_CKPT=1. run.sh calls it; rerun it by hand
# if it failed (it skips the merge when the merged model exists).
#   bash ~/qurl/s4/finish.sh <size> <tag>
# The merged model stays at $CKPT_ROOT/<tag>/hf_step<N> (FP32; for R4 it is in UAQ coordinates, the same function).
set -euo pipefail
SIZE=$1; TAG=$2
S4="$HOME/qurl/s4"
RES="$S4/results/$TAG"
CKPT="${CKPT_ROOT:-$S4/ckpt}/$TAG"
PY="$HOME/venvs/verl/bin/python"
DATA="${DATA_DIR:-$HOME/data/s4}"
export PYTHONPATH="$S4${PYTHONPATH:+:$PYTHONPATH}"
step=$(cat "$CKPT/latest_checkpointed_iteration.txt")
HF="$CKPT/hf_step$step"
echo "== finish $TAG ($SIZE): checkpoint step $step $(date -Is)"
if [ ! -f "$HF/config.json" ]; then
  "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$CKPT/global_step_$step/actor" --target_dir "$HF"
fi
"$PY" "$S4/eval.py" --model "$HF" --data "$DATA" --out "$RES/eval.json" --gpus "${NGPU:-8}" \
  --max-tokens "${MAX_RESPONSE:-8192}" --context "${CONTEXT:-32768}" --step "$step" ${EVAL_ARGS:-}
if [ "${KEEP_CKPT:-0}" != 1 ] && [ -s "$RES/eval.json" ]; then
  rm -rf "$CKPT"/global_step_*
  echo "deleted the FSDP checkpoints of $TAG (merged model kept at $HF)"
fi
echo "== finished $TAG $(date -Is)"
