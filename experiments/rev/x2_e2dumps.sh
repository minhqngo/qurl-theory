#!/usr/bin/env bash
# X2: E2's instrumented run (arm R3, seed 1, 435 steps) recorded the full FP32 weights that VeRL synced into the INT8
# rollout engine at 14 syncs (sync k holds the weights after step k - 1). Each is rebuilt as a Hugging Face model
# (merge_fp32.py e2) and scored in BF16 and INT8 W8A8. This gives the INT8 scoring cost along one INT8-trained
# run, at no training cost. If no dumps with sync 436 are left on the server, the job records SKIPPED and ends.
#   bash ~/qurl/rev/x2_e2dumps.sh       (about 1 hour)
set -euo pipefail
source "$HOME/qurl/rev/common.sh"
OUT="$REV/results/x2_e2dumps"
[ -f "$OUT/done" ] && { log "skip x2_e2dumps (done)"; exit 0; }
mkdir -p "$OUT"
FULL=""
for d in "$HOME"/qurl/e2/results/*/dumps/full; do
  if compgen -G "$d/sync0436_b*.pt" > /dev/null; then FULL=$d; break; fi
done
if [ -z "$FULL" ]; then
  log "no E2 full dumps with sync 436 under ~/qurl/e2/results: X2 skipped"
  echo "no E2 full dumps with sync 436 under ~/qurl/e2/results" > "$OUT/SKIPPED"
  touch "$OUT/done"
  exit 0
fi
echo "$FULL" > "$OUT/source.txt"
ORIG=$(base_dir)
SYNCS=$(ls "$FULL" | sed -n -E 's/^sync([0-9]{4})_b[0-9]+\.pt$/\1/p' | sort -u)
log "X2: $(echo "$SYNCS" | wc -w) full syncs in $FULL"
for S in $SYNCS; do
  if [ -s "$OUT/sync${S}_bf16.json" ] && [ -s "$OUT/sync${S}_int8.json" ]; then continue; fi
  TMP="$REV/tmp/x2_sync$S"
  rm -rf "$TMP"
  "$PY" "$REV/merge_fp32.py" e2 "$FULL" "$((10#$S))" "$TMP" --base "$BASE_MODEL" --require-fp32
  score "$TMP" bf16 "$OUT/sync${S}_bf16.json" --tokenizer "$ORIG"
  score "$TMP" int8 "$OUT/sync${S}_int8.json" --tokenizer "$ORIG"
  rm -rf "$TMP"
done
touch "$OUT/done"
log "x2_e2dumps done"
