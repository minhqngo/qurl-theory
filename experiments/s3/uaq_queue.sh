#!/usr/bin/env bash
# S3's UAQ experiment (P1 note, section 7): R1-R6 x seeds 1-3 at actor lr 1e-6, 435 steps each, run back to back.
#   nohup bash ~/qurl/s3/uaq_queue.sh > ~/qurl/s3/uaq_queue.log 2>&1 < /dev/null &    # start (or resume)
#   bash ~/qurl/s3/uaq_queue.sh --list                                                  # plan and status
#   bash ~/qurl/s3/uaq_queue.sh --progress                                              # progress of every run
# What it does, in order:
#   1. Waits until no other pilot.sh is running and the GPU has no compute process (so it can be started while the
#      Table 1 chain is still running; it begins when that chain ends).
#   2. Prerequisites: the s = 1.5 UAQ checkpoint for R2/R4 (written on the CPU if missing), and R5's exactness check on
#      this GPU (r5_check.sh). The queue stops unless R5 and the R4 repeat are bit-identical to R4.
#   3. The 18 runs, core arms first: R3, R4, R5 for seeds 1-3 (H1, 9 runs), then R1, R2, R6 for seeds 1-3 (H2, H3).
#      Seed k sets data.seed=k (prompt order) and actor_rollout_ref.rollout.seed=42+k (sampling); the six arms of a
#      seed share both, so arm-to-arm comparisons are paired by prompt order. Validation every epoch (29 steps).
# Resumable: a finished run (summary.txt present) is skipped; an unfinished run's folder is moved aside to
# <tag>.incomplete.<unix time> and the run starts over. The queue stops at the first failed run (set
# CONTINUE_ON_FAIL=1 to go on instead), so a systematic problem does not burn through the remaining runs.
# Environment: STEPS (default 435), QURL_S (default 1.5), BASE_LR (default 1e-6), DRY_RUN=1 (print, do not run).
set -uo pipefail
S3="$HOME/qurl/s3"
RESULTS="$S3/results"
STEPS=${STEPS:-435}
export QURL_S=${QURL_S:-1.5}
export BASE_LR=${BASE_LR:-1e-6}
PY="$HOME/venvs/verl/bin/python"
UAQ_MODEL="$HOME/qurl/models/qwen0.5b-instruct-uaq$QURL_S"

plan() {   # "arm seed tag" lines in run order
  for group in "R3 R4 R5" "R1 R2 R6"; do
    for seed in 1 2 3; do
      for arm in $group; do echo "$arm $seed uaq_${arm}_s${seed}"; done
    done
  done
}
status() {
  if [ -s "$RESULTS/$1/summary.txt" ]; then echo done
  elif [ -d "$RESULTS/$1" ]; then echo incomplete
  else echo pending; fi
}

case "${1:-}" in
  --tags) plan | awk '{print $3}' | tr '\n' ' '; echo; exit 0 ;;
  --list) plan | while read -r arm seed tag; do printf '%-4s seed %s  %-14s %s\n' "$arm" "$seed" "$tag" "$(status "$tag")"; done; exit 0 ;;
  --progress) bash "$S3/progress.sh" $(plan | awk '{print $3}'); exit 0 ;;
  "") ;;
  *) echo "usage: uaq_queue.sh [--list|--tags|--progress]" >&2; exit 2 ;;
esac

log() { echo "[$(date -Is)] $*"; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY RUN: $*"; else "$@"; fi; }

log "UAQ queue: $(plan | wc -l) runs, $STEPS steps each, s = $QURL_S, base lr $BASE_LR"
md5sum "$S3"/pilot.sh "$S3"/arm.sh "$S3"/qurl_verl/*.py "$S3"/qurl_verl/*.json 2>/dev/null | sed 's/^/  code: /'

# 1. Wait for the GPU.
while pgrep -f '[p]ilot.sh' >/dev/null || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  [ "${waited:-0}" = 1 ] || log "waiting for the GPU (another pilot.sh or GPU process is running)"
  waited=1
  sleep 60
done
log "GPU free"

# 2. Prerequisites.
if [ ! -f "$UAQ_MODEL/uaq.json" ]; then
  log "writing the UAQ checkpoint at $UAQ_MODEL"
  run env CUDA_VISIBLE_DEVICES= "$PY" "$S3/make_uaq_checkpoint.py" --s "$QURL_S" --out "$UAQ_MODEL" || { log "UAQ checkpoint failed"; exit 1; }
fi
if [ ! -s "$RESULTS/r5_check/compare.txt" ]; then
  log "running R5's exactness check (r5_check.sh)"
  run bash "$S3/r5_check.sh" || { log "r5_check.sh failed"; exit 1; }
fi
if [ "${DRY_RUN:-0}" != 1 ]; then
  n_ok=$(grep -c "BIT-IDENTICAL" "$RESULTS/r5_check/compare.txt" || true)
  if [ "$n_ok" -lt 2 ]; then
    log "R5's exactness check did not pass (expected R4 repeat and R5 bit-identical to R4); see results/r5_check/compare.txt"
    exit 1
  fi
  log "R5's exactness check passed"
fi

# 3. The runs.
plan | while read -r arm seed tag; do
  st=$(status "$tag")
  if [ "$st" = done ]; then log "skip $tag (done)"; continue; fi
  if [ "$st" = incomplete ]; then
    aside="$RESULTS/$tag.incomplete.$(date +%s)"
    log "moving the unfinished $tag aside to $(basename "$aside")"
    run mv "$RESULTS/$tag" "$aside"
  fi
  log "start $tag (arm $arm, seed $seed)"
  run bash "$S3/arm.sh" "$arm" "$tag" "$STEPS" trainer.test_freq=29 data.seed="$seed" \
    actor_rollout_ref.rollout.seed=$((42 + seed)) < /dev/null
  rc=$?
  log "end $tag: exit $rc"
  if [ $rc -ne 0 ] && [ "${CONTINUE_ON_FAIL:-0}" != 1 ]; then
    log "stopping the queue after a failed run; fix it and start the queue again (finished runs are skipped)"
    exit 1
  fi
done || exit 1
log "UAQ queue finished"
bash "$S3/progress.sh" $(plan | awk '{print $3}')
