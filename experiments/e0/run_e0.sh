#!/usr/bin/env bash
# E0 driver (run inside WSL):  bash run_e0.sh [MODEL_ID] [QUANTS...]
# Example: bash run_e0.sh Qwen/Qwen2.5-0.5B-Instruct fp8
set -euo pipefail
MODEL_ID=${1:-Qwen/Qwen2.5-0.5B-Instruct}
shift || true
QUANTS=${*:-fp8}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${E0_WORK:-$HOME/e0_work}
RES="$HERE/results"
mkdir -p "$WORK" "$RES"
source "$HOME/venvs/qurl/bin/activate"
# No CUDA toolkit (nvcc) in WSL: use vLLM's PyTorch sampler instead of FlashInfer's JIT-compiled one.
export VLLM_USE_FLASHINFER_SAMPLER=0

ORIG=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))")
python "$HERE/make_uaq_checkpoints.py" "$MODEL_ID" "$WORK" 1.5 2 | tee "$RES/checkpoints.jsonl"

if [ ! -f "$RES/responses.json" ]; then
  python "$HERE/e0_logprobs.py" generate "$ORIG" "$RES/responses.json" --n 512
fi

python "$HERE/e0_logprobs.py" score "$ORIG" "$RES/responses.json" "$RES/score_none_orig.json"
python "$HERE/e0_logprobs.py" score "$WORK/uaq_s2" "$RES/responses.json" "$RES/score_none_s2.json"
python "$HERE/e0_logprobs.py" score "$WORK/uaq_s1.5" "$RES/responses.json" "$RES/score_none_s1.5.json"
for Q in $QUANTS; do
  QN=${Q//_/-}   # file-name-safe (analyze_e0.py splits keys on "_")
  python "$HERE/e0_logprobs.py" score "$ORIG" "$RES/responses.json" "$RES/score_${QN}_orig.json" --quant "$Q" \
  && python "$HERE/e0_logprobs.py" score "$ORIG" "$RES/responses.json" "$RES/score_${QN}_orig_shuffled.json" --quant "$Q" --shuffle \
  && python "$HERE/e0_logprobs.py" score "$WORK/uaq_s2" "$RES/responses.json" "$RES/score_${QN}_s2.json" --quant "$Q" \
  && python "$HERE/e0_logprobs.py" score "$WORK/uaq_s1.5" "$RES/responses.json" "$RES/score_${QN}_s1.5.json" --quant "$Q" \
  || { echo "QUANT $Q FAILED (see log above)"; rm -f "$RES/score_${QN}_"*.json; }
done
python "$HERE/analyze_e0.py" "$RES" | tee "$RES/summary.txt"
