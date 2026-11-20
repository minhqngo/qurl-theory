# Shared settings and helpers for the revision experiments; the job scripts source this file.
REV="$HOME/qurl/rev"
S3="$HOME/qurl/s3"
PY="$HOME/venvs/verl/bin/python"          # S3's stack: VeRL 0.9.1 with vLLM 0.24
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
UAQ_MODEL="$HOME/qurl/models/qwen0.5b-instruct-uaq1.5"   # S3's R2/R4 starting checkpoint (FP32)
export VLLM_USE_FLASHINFER_SAMPLER=0      # no nvcc on the server for FlashInfer's JIT sampler (as in S1-S3)

log() { echo "[$(date -Is)] $*"; }

base_dir() {   # local snapshot of the base model
  "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_MODEL', local_files_only=True))"
}

# score <model dir> <bf16|int8> <out json> [score_gsm8k.py options]
# Skips an existing <out>. The JSON is written before vLLM shuts down, so a non-zero exit with a complete JSON (vLLM
# aborting at interpreter exit) counts as success. Records vLLM's INT8 kernel from the log in <out>.kernel.
score() {
  local model=$1 fmt=$2 out=$3
  shift 3
  if [ -s "$out" ]; then log "skip $(basename "$out") (exists)"; return 0; fi
  "$PY" "$REV/score_gsm8k.py" --model "$model" --format "$fmt" --out "$out" "$@" > "$out.log" 2>&1 || true
  if [ -s "$out" ]; then
    grep -o -m1 -E "(Cutlass|Triton|Machete|Marlin)[A-Za-z0-9]*Kernel" "$out.log" > "$out.kernel" || true
    log "$(basename "$out"): $(grep -m1 -o "$fmt accuracy.*" "$out.log" || echo scored) $(cat "$out.kernel" 2>/dev/null)"
    return 0
  fi
  log "scoring FAILED: $model ($fmt); log tail:"
  tail -n 25 "$out.log"
  return 1
}
