#!/usr/bin/env bash
# E2 (P1 experiments report, S5): one S3 run of arm R3 (QuRL INT8 W8A8, TIS, ACR; lr 1e-6) that records the synced
# FP32 master weights for the tracking and persistence analyses. It is the S3 queue's own call,
#   arm.sh R3 <tag> <steps> trainer.test_freq=29 data.seed=<seed> actor_rollout_ref.rollout.seed=<42 + seed>,
# with E2's worker extension (e2_dump.E2WorkerExtension, a tap on S3's unchanged QuRLWorkerExtension) in vLLM.
#   bash ~/qurl/e2/e2_run.sh <tag> [steps]        e.g.  e2_smoke 3   (smoke test, a few minutes)
#                                                        e2_r3_s1     (the E2 run: 435 steps, about 5 h)
# Environment: SEED (default 1); QURL_E2_FULL_SYNCS, QURL_E2_ROW_LAYERS, QURL_E2_ROWS (see e2_dump.py).
# Outputs: S3's usual run files in ~/qurl/s3/results/<tag>/, the dumps in ~/qurl/e2/results/<tag>/dumps/, and a copy
# of the small run files next to the dumps.
set -euo pipefail
TAG=$1
STEPS=${2:-435}
SEED=${SEED:-1}
E2="$HOME/qurl/e2"
S3="$HOME/qurl/s3"
OUT="$E2/results/$TAG"
if [ -e "$OUT" ] || [ -e "$S3/results/$TAG" ]; then echo "results for $TAG exist already; use a new tag" >&2; exit 1; fi

# Full dumps: the default list for a full run, every sync for a short one.
if [ -z "${QURL_E2_FULL_SYNCS:-}" ]; then
  if [ "$STEPS" -ge 435 ]; then QURL_E2_FULL_SYNCS=1,2,3,5,9,17,33,65,66,129,257,258,435,436
  else QURL_E2_FULL_SYNCS=$(seq -s, 1 $((STEPS + 1))); fi
fi
export QURL_E2_FULL_SYNCS
n_full=$(echo "$QURL_E2_FULL_SYNCS" | tr ',' '\n' | grep -c .)

mkdir -p "$OUT"
# Disk: about 2.0 GB per full dump (494M FP32 parameters), 3.3 MB of row samples per sync, plus a margin.
need_gb=$(( n_full * 2 + (STEPS + 1) * 4 / 1000 + 5 ))
avail_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "need about $need_gb GB free under $OUT, have $avail_gb GB" >&2; rmdir "$OUT"; exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "the GPU is busy; wait for the running job to finish" >&2; rmdir "$OUT"; exit 1
fi

export QURL_E2_DIR="$OUT/dumps"
export PYTHONPATH="$E2${PYTHONPATH:+:$PYTHONPATH}"   # arm.sh prepends ~/qurl/s3 (qurl_verl) and keeps this
{
  echo "E2 run $TAG: arm R3, $STEPS steps, seed $SEED, $(date -Is)"
  echo "full dumps at syncs $QURL_E2_FULL_SYNCS ($n_full; sync k holds the weights after step k-1)"
  echo "row samples: layers ${QURL_E2_ROW_LAYERS:-0,6,12,18,23}, ${QURL_E2_ROWS:-16} rows per matrix, every sync"
  echo "disk: need about $need_gb GB, $avail_gb GB available"
  md5sum "$E2"/e2_dump.py "$E2"/e2_run.sh "$S3"/arm.sh "$S3"/pilot.sh "$S3"/qurl_verl/*.py | sed 's/^/  code: /'
} | tee "$OUT/e2_run.txt"

bash "$S3/arm.sh" R3 "$TAG" "$STEPS" trainer.test_freq=29 data.seed="$SEED" \
  actor_rollout_ref.rollout.seed=$((42 + SEED)) \
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=e2_dump.E2WorkerExtension" < /dev/null

for f in metrics.jsonl summary.txt env.txt arm.txt int8_refit.txt; do
  [ -f "$S3/results/$TAG/$f" ] && cp "$S3/results/$TAG/$f" "$OUT/"
done
grep -o "\[qurl e2\].*" "$S3/results/$TAG/train.log" > "$OUT/e2_dump.txt" || true
echo "E2 run $TAG done $(date -Is): $(ls "$OUT/dumps/full" 2>/dev/null | grep -c '_b00.pt') full dumps," \
  "$(ls "$OUT/dumps/rows" 2>/dev/null | grep -c '^sync') row files, $(du -sh "$OUT/dumps" | cut -f1)" | tee -a "$OUT/e2_run.txt"
