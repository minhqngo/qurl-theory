#!/usr/bin/env bash
# FP8 determinism check (P1 note, section 7, S1 findings). At 1.5B, FP8's s = 2 runs differed from the original on
# 0.05-0.23% of tokens although every s = 2 rescaling is exact and the batch order was the same. This re-scores the
# original and s = 2 checkpoints under both FP8 modes, again in the same batch order:
#   repeat == first run  -> vLLM's FP8 path is run-to-run deterministic, so s = 2 vs. original is a systematic
#                           (non-homogeneous) step in that path;
#   repeat != first run  -> the FP8 path is not run-to-run deterministic, which explains s = 2 vs. original.
#   bash ~/qurl/h100/repeat_fp8.sh <hf_model_id> <tag>     (after s1_run.sh with the same arguments)
# Outputs: score_<fp8 mode>_{orig,s2}_repeat.json and a refreshed summary.txt in ~/qurl/h100/results/<tag>/
set -euo pipefail
MODEL_ID=$1; TAG=$2
E0="$HOME/qurl/e0"
RES="$HOME/qurl/h100/results/$TAG"
WORK="$HOME/e0_work/$TAG"
[ -f "$RES/responses.json" ] && [ -d "$WORK/uaq_s2" ] || { echo "run s1_run.sh $MODEL_ID $TAG first" >&2; exit 1; }
export VLLM_USE_FLASHINFER_SAMPLER=0   # same environment as s1_run.sh
PY="$HOME/venvs/qurl/bin/python"
LOG="$RES/repeat_fp8.log"
echo "== repeat_fp8 $MODEL_ID ($TAG) $(date -Is)" | tee -a "$LOG"
ORIG=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))")

# Same contract as s1_run.sh: skip finished outputs; an abort at vLLM teardown after the atomic write is fine.
sc() {
  if [ -s "$3" ]; then echo "skip $3 (exists)" >> "$LOG"; return 0; fi
  "$PY" "$E0/e0_logprobs.py" score "$@" >> "$LOG" 2>&1 || [ -s "$3" ]
}
for Q in fp8_per_tensor fp8_per_channel; do
  QN=${Q//_/-}
  sc "$ORIG" "$RES/responses.json" "$RES/score_${QN}_orig_repeat.json" --quant "$Q"
  sc "$WORK/uaq_s2" "$RES/responses.json" "$RES/score_${QN}_s2_repeat.json" --quant "$Q"
done
# Kernel choice and torch.compile cache hits/misses per run (a cache miss that re-autotunes could change numerics).
grep -iE "Selected .*Kernel for|compiled graph|Compiling a graph|cache directory" "$LOG" > "$RES/repeat_fp8_kernels.txt" || true

"$PY" "$E0/analyze_e0.py" "$RES" | tee "$RES/summary.txt"
echo "== done $(date -Is)" | tee -a "$LOG"
