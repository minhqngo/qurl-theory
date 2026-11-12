#!/usr/bin/env bash
# E0 for INT8 W8A8 (run inside WSL after run_e0.sh): bash run_e0_int8.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${E0_WORK:-$HOME/e0_work}
RES="$HERE/results"
ORIG=$("$HOME/venvs/qurl/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B-Instruct'))")

# 1) quantize original and UAQ checkpoints with llm-compressor (CPU venv)
[ -d "$WORK/int8_w8a8_s1.5" ] || "$HOME/venvs/llmc/bin/python" "$HERE/make_int8_w8a8.py" "$WORK" \
  "orig=$ORIG" "s2=$WORK/uaq_s2" "s1.5=$WORK/uaq_s1.5" > "$HERE/results_int8_quantize.log" 2>&1

# 2) compare stored INT8 codes and scales directly
"$HOME/venvs/qurl/bin/python" "$HERE/compare_int8_codes.py" "$WORK" | tee "$RES/int8_codes.txt"

# 3) score with vLLM (quantization read from the checkpoint config)
source "$HOME/venvs/qurl/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
# CUTLASS INT8 GEMM is not supported on SM120 (Blackwell consumer/laptop); fall back to the Triton INT8 kernel.
export VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel
S="$HERE/e0_logprobs.py"; R="$RES/responses.json"
python "$S" score "$WORK/int8_w8a8_orig" "$R" "$RES/score_int8-w8a8_orig.json" > "$HERE/results_int8_run.log" 2>&1
python "$S" score "$WORK/int8_w8a8_orig" "$R" "$RES/score_int8-w8a8_orig_shuffled.json" --shuffle >> "$HERE/results_int8_run.log" 2>&1
python "$S" score "$WORK/int8_w8a8_s2" "$R" "$RES/score_int8-w8a8_s2.json" >> "$HERE/results_int8_run.log" 2>&1
python "$S" score "$WORK/int8_w8a8_s1.5" "$R" "$RES/score_int8-w8a8_s1.5.json" >> "$HERE/results_int8_run.log" 2>&1
python "$HERE/analyze_e0.py" "$RES" | tee "$RES/summary.txt"
