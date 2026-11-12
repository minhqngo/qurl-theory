#!/usr/bin/env bash
# Seeds 4 and 5 of R4 and R5 for the pre-registered H1 test (P1 note, section 7; decided 2026-11-11).
#   nohup bash ~/qurl/s3/uaq_seeds45.sh > ~/qurl/s3/uaq_seeds45.log 2>&1 < /dev/null &    # start now; it waits
#   bash ~/qurl/s3/uaq_seeds45.sh --list                                                    # plan and status
#   bash ~/qurl/s3/uaq_seeds45.sh --progress                                                # progress of its runs
# What it does, in order:
#   1. Waits until uaq_queue.sh has exited, no pilot.sh is running and the GPU has no compute process.
#   2. Checks that all 18 runs of uaq_queue.sh finished (FORCE=1 skips this) and that R5's exactness check passed.
#   3. Runs R4 and R5 for seed 4, then for seed 5, with the same arm.sh call as uaq_queue.sh: seed k sets
#      data.seed=k and actor_rollout_ref.rollout.seed=42+k, and validation runs every epoch (29 steps).
# A separate file on purpose: uaq_queue.sh is running on the server, and overwriting a script that a running bash is
# reading can break it. Resumable and stops at the first failed run, like uaq_queue.sh.
# Environment: STEPS (default 435), QURL_S (1.5), BASE_LR (1e-6), FORCE=1, CONTINUE_ON_FAIL=1, DRY_RUN=1.
set -uo pipefail
S3="$HOME/qurl/s3"
RESULTS="$S3/results"
STEPS=${STEPS:-435}
export QURL_S=${QURL_S:-1.5}
export BASE_LR=${BASE_LR:-1e-6}

plan() {   # "arm seed tag" lines in run order; each seed's pair runs back to back
  for seed in 4 5; do
    for arm in R4 R5; do echo "$arm $seed uaq_${arm}_s${seed}"; done
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
  *) echo "usage: uaq_seeds45.sh [--list|--tags|--progress]" >&2; exit 2 ;;
esac

log() { echo "[$(date -Is)] $*"; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY RUN: $*"; else "$@"; fi; }

log "R4/R5 seeds 4-5 for H1: $(plan | wc -l) runs, $STEPS steps each, s = $QURL_S, base lr $BASE_LR"
md5sum "$S3"/uaq_seeds45.sh "$S3"/pilot.sh "$S3"/arm.sh "$S3"/qurl_verl/*.py "$S3"/qurl_verl/*.json 2>/dev/null | sed 's/^/  code: /'

# 1. Wait for the main queue and the GPU. The pattern matches only a bash process running uaq_queue.sh (and its
#    subshells), not any command line that merely mentions the name. The check in step 2 is the real guard: without
#    FORCE, nothing starts until all 18 runs of uaq_queue.sh have a summary.txt.
while pgrep -f '^([^ ]*/)?bash [^ ]*uaq_queue\.sh( |$)' >/dev/null || pgrep -f '[p]ilot\.sh' >/dev/null \
      || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  [ "${waited:-0}" = 1 ] || log "waiting for uaq_queue.sh to finish and the GPU to be free"
  waited=1
  sleep 60
done
log "uaq_queue.sh has exited and the GPU is free"

# 2. Prerequisites.
if [ "${FORCE:-0}" != 1 ]; then
  missing=""
  for t in $(bash "$S3/uaq_queue.sh" --tags); do
    [ -s "$RESULTS/$t/summary.txt" ] || missing="$missing $t"
  done
  if [ -n "$missing" ]; then
    log "uaq_queue.sh did not finish all its runs; missing:$missing. Not starting (FORCE=1 starts anyway)."
    exit 1
  fi
  log "all $(bash "$S3/uaq_queue.sh" --tags | wc -w) runs of uaq_queue.sh finished"
fi
n_ok=$(grep -c "BIT-IDENTICAL" "$RESULTS/r5_check/compare.txt" 2>/dev/null || true)
if [ "${n_ok:-0}" -lt 2 ]; then
  log "R5's exactness check has not passed (results/r5_check/compare.txt); not starting"
  exit 1
fi
if [ ! -f "$HOME/qurl/models/qwen0.5b-instruct-uaq$QURL_S/uaq.json" ]; then
  log "the UAQ checkpoint for R4 is missing (uaq_queue.sh writes it); not starting"
  exit 1
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
    log "stopping after a failed run; fix it and start this script again (finished runs are skipped)"
    exit 1
  fi
done || exit 1
log "seeds 4-5 finished"
bash "$S3/progress.sh" $(plan | awk '{print $3}')
