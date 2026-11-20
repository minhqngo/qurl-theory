#!/usr/bin/env bash
# X3: one rerun of S3's arm R1 (BF16 rollouts) or R3 (QuRL: INT8 rollouts, TIS, ACR) with seed k. It keeps the
# actor's FP32 weights at every validation step and scores each in BF16 and in INT8 W8A8. The training call is the S3
# queue's own (uaq_queue.sh), with these additions:
#   trainer.val_before_train=True             VeRL's own step-0 validation, a check of score_gsm8k.py
#   trainer.save_freq=29, model-only          the actor at steps 29, 58, ..., 435 (2 GB each; the critic keeps one)
#   trainer.resume_mode=disable               a failed run starts over; it never resumes from model-only weights
# Each checkpoint is merged in FP32 (merge_fp32.py), not by VeRL's merger, which casts to BF16, and is scored in
# both formats. Only step 435's weights are kept. Resumable: a finished training is not repeated, merged checkpoints
# and existing scores are reused, and the run is done when all 15 steps are scored.
#   bash ~/qurl/rev/x3_run.sh <R1|R3> <seed>      (about 6 hours: 5 training, 1 scoring)
set -euo pipefail
source "$HOME/qurl/rev/common.sh"
ARM=$1
SEED=$2
case "$ARM" in R1|R3) ;; *) echo "arm must be R1 or R3" >&2; exit 2 ;; esac
TAG="rev_x3_${ARM}_s${SEED}"
OUT="$REV/results/x3/$TAG"
RUN="$S3/results/$TAG"
CKPT="$REV/ckpt/$TAG"
[ -f "$OUT/done" ] && { log "skip $TAG (done)"; exit 0; }
mkdir -p "$OUT"

# 1. Train, unless an earlier attempt finished training.
if [ ! -s "$RUN/summary.txt" ]; then
  if [ -e "$RUN" ]; then log "moving the unfinished $TAG aside"; mv "$RUN" "$RUN.incomplete.$(date +%s)"; fi
  rm -rf "$CKPT"
  log "train $TAG (arm $ARM, seed $SEED)"
  bash "$S3/arm.sh" "$ARM" "$TAG" 435 trainer.test_freq=29 data.seed="$SEED" \
    actor_rollout_ref.rollout.seed=$((42 + SEED)) \
    trainer.val_before_train=True trainer.save_freq=29 trainer.default_local_dir="$CKPT" \
    trainer.resume_mode=disable trainer.max_critic_ckpt_to_keep=1 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model]' 'critic.checkpoint.save_contents=[model]' < /dev/null
fi
for f in summary.txt metrics.jsonl env.txt arm.txt; do cp "$RUN/$f" "$OUT/" 2>/dev/null || true; done

# 2. Merge each saved actor checkpoint in FP32, then drop its FSDP shards (actor and critic).
for d in "$CKPT"/global_step_*; do
  [ -d "$d" ] || continue
  N=${d##*_}
  H="$CKPT/hf_step$N"
  if [ ! -s "$H/model.safetensors" ]; then
    log "merge step $N"
    "$PY" "$REV/merge_fp32.py" fsdp "$d/actor" "$H" --base "$BASE_MODEL" --require-fp32
  fi
  rm -rf "$d"
done

# 3. Score every kept checkpoint in both formats.
n=0
for H in $(ls -d "$CKPT"/hf_step* 2>/dev/null | sort -V); do
  N=${H##*hf_step}
  score "$H" bf16 "$OUT/step${N}_bf16.json"
  score "$H" int8 "$OUT/step${N}_int8.json"
  n=$((n + 1))
done
if [ "$n" -lt 15 ]; then log "$TAG: expected 15 checkpoints (steps 29-435), found $n"; exit 1; fi

# 4. Keep only the final weights.
for H in "$CKPT"/hf_step*; do [ "$H" = "$CKPT/hf_step435" ] || rm -rf "$H"; done
touch "$OUT/done"
log "$TAG done"
