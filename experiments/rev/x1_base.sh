#!/usr/bin/env bash
# X1: step 0. The base model (every S3 run's start) and S3's UAQ checkpoint (R2 and R4's start, s = 1.5, FP32), each
# scored on GSM8K's test set in BF16 and in INT8 W8A8 (score_gsm8k.py). The base model's INT8 - BF16 difference is the
# cost of scoring in INT8 before any training. By Theorem 1 the UAQ checkpoint's INT8 score equals the base model's,
# up to the FP32 rounding of W / 1.5.
#   bash ~/qurl/rev/x1_base.sh          (about 10 minutes)
set -euo pipefail
source "$HOME/qurl/rev/common.sh"
OUT="$REV/results/x1_base"
[ -f "$OUT/done" ] && { log "skip x1_base (done)"; exit 0; }
mkdir -p "$OUT"
ORIG=$(base_dir)
score "$ORIG" bf16 "$OUT/orig_bf16.json"
score "$ORIG" int8 "$OUT/orig_int8.json"
score "$UAQ_MODEL" bf16 "$OUT/uaq1.5_bf16.json" --tokenizer "$ORIG"
score "$UAQ_MODEL" int8 "$OUT/uaq1.5_int8.json" --tokenizer "$ORIG"
touch "$OUT/done"
log "x1_base done"
