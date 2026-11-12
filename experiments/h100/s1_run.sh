#!/usr/bin/env bash
# S1 (P1 note, section 7): E0 step-0 identity test on the H100, at the paper's model size and with
# vLLM's CUTLASS INT8 kernel (the laptop run had to fall back to Triton INT8 on SM120).
#   bash ~/qurl/h100/s1_run.sh <hf_model_id> <tag> [online quant methods...]
#   e.g. bash ~/qurl/h100/s1_run.sh Qwen/Qwen2.5-Math-7B math7b fp8_per_tensor fp8_per_channel
# Outputs: ~/qurl/h100/results/<tag>/  (same file layout as experiments/e0/results)
set -euo pipefail
MODEL_ID=$1; TAG=$2; shift 2
QUANTS=${*:-fp8_per_tensor fp8_per_channel}
E0="$HOME/qurl/e0"
RES="$HOME/qurl/h100/results/$TAG"
WORK="$HOME/e0_work/$TAG"
mkdir -p "$RES" "$WORK"
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on this server for FlashInfer's JIT sampler
PY="$HOME/venvs/qurl/bin/python"
LOG="$RES/run.log"
echo "== S1 $MODEL_ID ($TAG) $(date -Is)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee -a "$LOG"

ORIG=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))")
[ -d "$WORK/uaq_s1.5" ] || "$PY" "$E0/make_uaq_checkpoints.py" "$MODEL_ID" "$WORK" 1.5 2 | tee "$RES/checkpoints.jsonl"
[ -f "$RES/responses.json" ] || "$PY" "$E0/e0_logprobs.py" generate "$ORIG" "$RES/responses.json" --n 512 >> "$LOG" 2>&1

# score <model> <responses> <out> [opts]: skipped if <out> exists (re-runs resume). Results are written
# atomically, so an existing <out> is complete even if the process then aborts during vLLM teardown.
sc() {
  if [ -s "$3" ]; then echo "skip $3 (exists)" >> "$LOG"; return 0; fi
  "$PY" "$E0/e0_logprobs.py" score "$@" >> "$LOG" 2>&1 || [ -s "$3" ]
}
sc "$ORIG" "$RES/responses.json" "$RES/score_none_orig.json"
sc "$ORIG" "$RES/responses.json" "$RES/score_none_orig_shuffled.json" --shuffle   # BF16 batch-invariance floor
sc "$WORK/uaq_s2" "$RES/responses.json" "$RES/score_none_s2.json"
sc "$WORK/uaq_s1.5" "$RES/responses.json" "$RES/score_none_s1.5.json"
for Q in $QUANTS; do
  QN=${Q//_/-}
  sc "$ORIG" "$RES/responses.json" "$RES/score_${QN}_orig.json" --quant "$Q" \
  && sc "$ORIG" "$RES/responses.json" "$RES/score_${QN}_orig_shuffled.json" --quant "$Q" --shuffle \
  && sc "$WORK/uaq_s2" "$RES/responses.json" "$RES/score_${QN}_s2.json" --quant "$Q" \
  && sc "$WORK/uaq_s1.5" "$RES/responses.json" "$RES/score_${QN}_s1.5.json" --quant "$Q" \
  || { echo "QUANT $Q FAILED (see run.log)" | tee -a "$LOG"; rm -f "$RES/score_${QN}_"*.json; }
done

# INT8 W8A8 (QuRL's scheme): compressed-tensors checkpoints, scored with the default (CUTLASS) INT8 kernel.
[ -d "$WORK/int8_w8a8_s1.5" ] || "$HOME/venvs/llmc/bin/python" "$E0/make_int8_w8a8.py" "$WORK" \
  "orig=$ORIG" "s2=$WORK/uaq_s2" "s1.5=$WORK/uaq_s1.5" >> "$RES/int8_quantize.log" 2>&1
"$PY" "$E0/compare_int8_codes.py" "$WORK" | tee "$RES/int8_codes.txt"
sc "$WORK/int8_w8a8_orig" "$RES/responses.json" "$RES/score_int8-w8a8_orig.json"
sc "$WORK/int8_w8a8_orig" "$RES/responses.json" "$RES/score_int8-w8a8_orig_shuffled.json" --shuffle
sc "$WORK/int8_w8a8_s2" "$RES/responses.json" "$RES/score_int8-w8a8_s2.json"
sc "$WORK/int8_w8a8_s1.5" "$RES/responses.json" "$RES/score_int8-w8a8_s1.5.json"
grep -iE "ScaledMM|Int8.*Kernel|Using .*kernel" "$LOG" | sort | uniq -c | tee "$RES/kernels.txt" || true

"$PY" "$E0/analyze_e0.py" "$RES" | tee "$RES/summary.txt"
echo "== done $(date -Is)" | tee -a "$LOG"
