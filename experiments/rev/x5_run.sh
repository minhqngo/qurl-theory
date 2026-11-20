#!/usr/bin/env bash
# X5: the first 29 steps of S3's arm R3, R4 or R5 with a new seed, validated at step 29, to see whether the run
# stalls early. The call is the S3 queue's own (uaq_queue.sh) with 29 steps; seed k sets the prompt order
# (data.seed=k) and the sampling seed (42 + k), so the arms of a seed are paired. VeRL validates at the last step.
# With a constant learning rate and no warmup, these steps are the first 29 of a full run.
#   bash ~/qurl/rev/x5_run.sh <R3|R4|R5> <seed>   (about 25 minutes)
set -euo pipefail
source "$HOME/qurl/rev/common.sh"
ARM=$1
SEED=$2
case "$ARM" in R3|R4|R5) ;; *) echo "arm must be R3, R4 or R5" >&2; exit 2 ;; esac
TAG="rev_x5_${ARM}_s${SEED}"
OUT="$REV/results/x5/$TAG"
RUN="$S3/results/$TAG"
[ -f "$OUT/done" ] && { log "skip $TAG (done)"; exit 0; }
mkdir -p "$OUT"
if [ ! -s "$RUN/summary.txt" ]; then
  if [ -e "$RUN" ]; then log "moving the unfinished $TAG aside"; mv "$RUN" "$RUN.incomplete.$(date +%s)"; fi
  log "train $TAG (arm $ARM, seed $SEED, 29 steps)"
  bash "$S3/arm.sh" "$ARM" "$TAG" 29 trainer.test_freq=29 data.seed="$SEED" \
    actor_rollout_ref.rollout.seed=$((42 + SEED)) < /dev/null
fi
for f in summary.txt metrics.jsonl env.txt arm.txt; do cp "$RUN/$f" "$OUT/" 2>/dev/null || true; done
"$PY" - "$OUT/metrics.jsonl" <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
steps = {r["step"] for r in recs}
val = [r["data"]["val-core/openai/gsm8k/acc/mean@1"] for r in recs if "val-core/openai/gsm8k/acc/mean@1" in r["data"]]
if max(steps) < 29 or not val:
    sys.exit(f"incomplete: last step {max(steps)}, validations {len(val)}")
EOF
touch "$OUT/done"
log "$TAG done"
