#!/usr/bin/env bash
# X6: a null distribution for S1's FP8 ratios at s = 1.5 (revision experiments; analysis x6_analysis.py, frozen).
# Every scalar s gives the same quantized model in exact arithmetic (Theorem 1). UAQ checkpoints are stored in BF16,
# so each s re-rounds W/s and s*gamma differently: checkpoints at nearby s are re-rounding draws of one mathematical
# model. For S1's Qwen2.5-Math-7B and -1.5B (tags math7b, math1.5b) this job
#   - builds UAQ checkpoints at s = 1.45 ... 1.55 (not 1.5) with S1's make_uaq_checkpoints.py, one at a time;
#   - scores S1's 512 responses on each with S1's e0_logprobs.py on vLLM 0.33 (~/venvs/qurl), exactly as S1 did:
#     BF16 (the draw's own policy, the reference of its ratios), FP8 per-tensor, FP8 per-channel and INT8 W8A8
#     (an llm-compressor checkpoint from ~/venvs/llmc; X6_INT8=0 leaves it out);
#   - re-scores the original and S1's s = 1.5 checkpoint the same way, to check that S1's environment is reproduced;
#   - deletes each draw's checkpoints once it is scored, keeping only the score JSONs and a small manifest, and
#     prunes the vLLM compile-cache entries it created (about 50 MB per checkpoint and mode, never reusable).
#   bash ~/qurl/rev/x6_run.sh        (about 2.5 h on the H100, 2 h without INT8; peak disk about 25 GB at 7B)
# Needs on the server: S1's responses (~/qurl/h100/results/{math7b,math1.5b}/responses.json), S1's scripts
# (~/qurl/e0), the venvs qurl and llmc, and both models in the Hugging Face cache (else they are downloaded,
# 15 + 3 GB). S1's checkpoints in ~/e0_work/<tag> (uaq_s1.5, int8_w8a8_orig, int8_w8a8_s1.5) are reused if present
# and rebuilt with S1's scripts and venvs otherwise (identical to S1's if the venvs are unchanged; x6_analysis.py's
# reproduction check compares the re-scores with S1's).
# Resumable: existing scores are skipped, and leftovers of an interrupted run are deleted. The job is done when every
# BF16 and FP8 score exists; INT8 W8A8 is optional, and its failures are listed but do not block "done".
# Outputs in ~/qurl/rev/results/x6_fp8null/: manifest.txt, progress.log, done, x6_analysis.txt (the frozen analysis,
# run here too), and per model <tag>/: score_<none|fp8-per-tensor|fp8-per-channel|int8-w8a8>_<orig|s1.5|s1.45|...>.json,
# checkpoints.jsonl (codes changed by each draw's re-rounding), responses.sha256, kernels.txt, run.log, build.log.
# Overrides (smoke tests): X6_MODELS ("<hf id>=<tag> ..."), X6_DRAWS, X6_INT8, X6_MIN_FREE_GB, REV_DIR, E0_DIR,
# S1_RESULTS, S1_WORK, VENVS, X_GPU_WAIT (seconds to wait for a busy GPU; 0 skips the check).
set -euo pipefail
export LC_NUMERIC=C
REV_DIR=${REV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
E0_DIR=${E0_DIR:-$HOME/qurl/e0}
S1_RESULTS=${S1_RESULTS:-$HOME/qurl/h100/results}
S1_WORK=${S1_WORK:-$HOME/e0_work}
VENVS=${VENVS:-$HOME/venvs}
PY="$VENVS/qurl/bin/python"          # S1's stack: vLLM 0.33
LLMC="$VENVS/llmc/bin/python"        # llm-compressor (CPU), S1's INT8 W8A8 checkpoints
MODELS=${X6_MODELS:-"Qwen/Qwen2.5-Math-7B=math7b Qwen/Qwen2.5-Math-1.5B=math1.5b"}
DRAWS=${X6_DRAWS:-"1.45 1.46 1.47 1.48 1.49 1.51 1.52 1.53 1.54 1.55"}
INT8=${X6_INT8:-1}
MIN_FREE_GB=${X6_MIN_FREE_GB:-10}    # free space to keep beyond the checkpoint about to be written
OUT="$REV_DIR/results/x6_fp8null"
WORK="$REV_DIR/tmp/x6_work"
export VLLM_USE_FLASHINFER_SAMPLER=0  # as in S1: no nvcc on the server for FlashInfer's JIT sampler

log() { echo "[$(date -Is)] $*" | tee -a "$OUT/progress.log"; }
if [ -f "$OUT/done" ]; then echo "[$(date -Is)] skip x6_fp8null (done)"; exit 0; fi
mkdir -p "$OUT" "$WORK"
trap 'rm -rf "$WORK"' EXIT           # only rebuildable checkpoints live there
rm -f "$OUT/incomplete.txt"

for f in e0_logprobs.py make_uaq_checkpoints.py make_int8_w8a8.py; do
  [ -f "$E0_DIR/$f" ] || { log "missing $E0_DIR/$f (S1's scripts): stopping"; exit 1; }
done
[ -x "$PY" ] || { log "missing $PY: stopping"; exit 1; }
if [ "$INT8" = 1 ] && [ ! -x "$LLMC" ]; then log "no $LLMC: INT8 W8A8 left out"; INT8=0; fi

GPU_WAIT=${X_GPU_WAIT:-1200}         # seconds to wait for a busy GPU; 0 skips the check (WSL lists host processes)
t=0
while [ "$GPU_WAIT" -gt 0 ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
  [ $t -eq 0 ] && log "waiting for the GPU (another process is using it)"
  t=$((t + 30))
  [ $t -le "$GPU_WAIT" ] || { log "the GPU is still busy after $GPU_WAIT s: stopping"; exit 1; }
  sleep 30
done

free_gb() { df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9; }
PEAK=0
track() {   # remember the largest size of the work directory (bytes)
  local b
  b=$(du -sb "$WORK" 2>/dev/null | cut -f1)
  [ "${b:-0}" -gt "$PEAK" ] && PEAK=$b
  return 0
}
need() {    # need <GB> <what>: stop rather than fill the disk
  local f
  f=$(free_gb)
  if [ "$f" -lt $(($1 + MIN_FREE_GB)) ]; then
    log "only $f GB free under $WORK; $2 needs $1 GB plus $MIN_FREE_GB GB margin: stopping"
    exit 1
  fi
}
complete_ckpt() { [ -f "$1/config.json" ] && compgen -G "$1/*.safetensors" > /dev/null; }
# vLLM keeps a torch.compile cache entry (about 50 MB) per checkpoint path and mode. The draws' paths are deleted, so
# their entries can never be reused: entries created after the job started are pruned after each draw
# (X6_PRUNE_CACHE=0 keeps them). The queue runs one job at a time, so no other job's entries are touched.
VCACHE="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/torch_compile_cache"
cache_entries() {
  if [ -d "$VCACHE" ]; then (cd "$VCACHE" && ls -1d -- * torch_aot_compile/* 2>/dev/null || true) | sort -u; fi
  return 0
}
CACHE0=$(cache_entries)
prune_cache() {
  local e
  [ "${X6_PRUNE_CACHE:-1}" = 1 ] || return 0
  comm -13 <(printf '%s\n' "$CACHE0") <(cache_entries) | while read -r e; do
    if [ -n "$e" ]; then rm -rf -- "${VCACHE:?}/$e"; fi
  done
  return 0
}
hf_path() {
  "$PY" - "$1" <<'EOF'
import sys
from huggingface_hub import snapshot_download
try:
    print(snapshot_download(sys.argv[1], local_files_only=True))
except Exception:
    print(snapshot_download(sys.argv[1]))
EOF
}

{
  echo "x6_fp8null manifest (x6_run.sh), started $(date -Is) on $(hostname)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  "$PY" -c "import vllm, torch; print('vllm (qurl venv):', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  if [ "$INT8" = 1 ]; then
    "$LLMC" -c "import llmcompressor, compressed_tensors as c; print('llmcompressor (llmc venv):', llmcompressor.__version__, '| compressed-tensors', c.__version__)" 2>/dev/null \
      | grep "^llmcompressor (llmc venv)" || echo "llmcompressor (llmc venv): version not readable"
  fi
  echo "settings: models $MODELS | draws $DRAWS | int8 $INT8 | S1 results $S1_RESULTS | S1 work $S1_WORK"
  md5sum "$E0_DIR/e0_logprobs.py" "$E0_DIR/make_uaq_checkpoints.py" "$E0_DIR/make_int8_w8a8.py" "$0" | sed 's/^/code: /'
  echo "free disk at start: $(free_gb) GB"
} >> "$OUT/manifest.txt"

INT8_FAILED=()

# sc <model dir> <out json> [e0_logprobs.py score options]: skipped if <out> exists; one retry. A non-zero exit with
# a complete <out> counts as success (S1's rule: vLLM may abort at interpreter teardown after the atomic write).
sc() {
  local model=$1 out=$2 try
  shift 2
  [ -s "$out" ] && return 0
  for try in 1 2; do
    echo "== score $(basename "$out") model=$model opts=$* try=$try $(date -Is)" >> "$LOGF"
    "$PY" "$E0_DIR/e0_logprobs.py" score "$model" "$RESP" "$out" "$@" >> "$LOGF" 2>&1 || true
    if [ -s "$out" ]; then log "  scored $(basename "$out")"; return 0; fi
    log "  scoring $(basename "$out") failed (try $try; see $LOGF)"
    sleep 20
  done
  return 1
}

# score_bf16_fp8 <BF16 checkpoint dir> <variant>: the draw's own BF16 policy and both FP8 modes (vLLM online).
# A failed score is left missing; the completeness check at the end decides whether the job is done.
score_bf16_fp8() {
  local m=$1 v=$2
  sc "$m" "$D/score_none_$v.json" || true
  sc "$m" "$D/score_fp8-per-tensor_$v.json" --quant fp8_per_tensor || true
  sc "$m" "$D/score_fp8-per-channel_$v.json" --quant fp8_per_channel || true
}

# score_int8 <BF16 checkpoint dir> <variant> [existing INT8 checkpoint]: optional; failures are recorded, not fatal.
score_int8() {
  local m=$1 v=$2 have=${3:-} q
  [ "$INT8" = 1 ] || return 0
  [ -s "$D/score_int8-w8a8_$v.json" ] && return 0
  q=$have
  if [ -z "$q" ]; then
    q="$W/int8_w8a8_$v"
    rm -rf "$q"
    need $((MODEL_GB * 6 / 10 + 1)) "the INT8 W8A8 checkpoint of $v"
    if ! "$LLMC" "$E0_DIR/make_int8_w8a8.py" "$W" "$v=$m" >> "$D/build.log" 2>&1 || ! complete_ckpt "$q"; then
      log "  INT8 W8A8 quantization of $v failed (see $D/build.log)"
      INT8_FAILED+=("$TAG/$v")
      rm -rf "$q"
      return 0
    fi
    track
  fi
  sc "$q" "$D/score_int8-w8a8_$v.json" || INT8_FAILED+=("$TAG/$v")
  if [ -z "$have" ]; then rm -rf "$q"; fi
  return 0
}

# build_uaq <s>: S1's make_uaq_checkpoints.py into $W/uaq_s<s>; appends its code-change line to checkpoints.jsonl.
build_uaq() {
  local s=$1
  need "$MODEL_GB" "the UAQ checkpoint at s = $s"
  rm -rf "$W/uaq_s$s"
  if ! HF_HUB_OFFLINE=1 "$PY" "$E0_DIR/make_uaq_checkpoints.py" "$ID" "$W" "$s" >> "$D/checkpoints.jsonl" \
       2>> "$D/build.log" || ! complete_ckpt "$W/uaq_s$s"; then
    log "  building the UAQ checkpoint at s = $s failed (see $D/build.log)"
    rm -rf "$W/uaq_s$s"
    return 1
  fi
  track
}

FAILED_MODELS=()
for spec in $MODELS; do
  ID=${spec%%=*}
  TAG=${spec#*=}
  D="$OUT/$TAG"
  W="$WORK/$TAG"
  RESP="$S1_RESULTS/$TAG/responses.json"
  LOGF="$D/run.log"
  T0=$(date +%s)
  if [ ! -s "$RESP" ]; then
    log "$TAG: S1's responses are missing ($RESP); copy them from experiments/h100/results/$TAG/. Model skipped."
    FAILED_MODELS+=("$TAG")
    continue
  fi
  mkdir -p "$D"
  rm -rf "$W"
  mkdir -p "$W"
  if ! ORIG=$(hf_path "$ID"); then
    log "$TAG: $ID is neither in the HF cache nor downloadable. Model skipped."
    FAILED_MODELS+=("$TAG")
    continue
  fi
  MODEL_GB=$(du -sL --block-size=1G "$ORIG" 2>/dev/null | cut -f1 || true)
  MODEL_GB=${MODEL_GB:-16}
  echo "$(sha256sum "$RESP" | cut -d' ' -f1)  $RESP" > "$D/responses.sha256"
  log "== $TAG ($ID, $MODEL_GB GB): $ORIG; responses $RESP"
  echo "model $TAG: $ID at $ORIG ($MODEL_GB GB); responses $RESP sha256 $(cut -d' ' -f1 "$D/responses.sha256")" \
    >> "$OUT/manifest.txt"

  # References: the original, then S1's s = 1.5 (S1's checkpoints when they exist, else rebuilt).
  score_bf16_fp8 "$ORIG" orig
  if complete_ckpt "$S1_WORK/$TAG/int8_w8a8_orig"; then I8=$S1_WORK/$TAG/int8_w8a8_orig; else I8=""; fi
  score_int8 "$ORIG" orig "$I8"
  echo "  int8 orig: ${I8:-rebuilt}" >> "$OUT/manifest.txt"
  if complete_ckpt "$S1_WORK/$TAG/uaq_s1.5"; then
    U15=$S1_WORK/$TAG/uaq_s1.5
  else
    U15=""
    if [ ! -s "$D/score_none_s1.5.json" ] || [ ! -s "$D/score_fp8-per-tensor_s1.5.json" ] || \
       [ ! -s "$D/score_fp8-per-channel_s1.5.json" ] || { [ "$INT8" = 1 ] && [ ! -s "$D/score_int8-w8a8_s1.5.json" ]; }; then
      build_uaq 1.5 && U15=$W/uaq_s1.5
    fi
  fi
  echo "  uaq s1.5: ${U15:-rebuilt or not needed}" >> "$OUT/manifest.txt"
  if [ -n "$U15" ]; then
    score_bf16_fp8 "$U15" s1.5
    if complete_ckpt "$S1_WORK/$TAG/int8_w8a8_s1.5"; then I8=$S1_WORK/$TAG/int8_w8a8_s1.5; else I8=""; fi
    score_int8 "$U15" s1.5 "$I8"
    echo "  int8 s1.5: ${I8:-rebuilt}" >> "$OUT/manifest.txt"
    if [ "$U15" = "$W/uaq_s1.5" ]; then rm -rf "$U15"; fi
  fi

  # The null draws, one at a time.
  for s0 in $DRAWS; do
    s=$(printf '%g' "$s0")
    v="s$s"
    todo=0
    for q in none fp8-per-tensor fp8-per-channel; do [ -s "$D/score_${q}_$v.json" ] || todo=1; done
    if [ "$INT8" = 1 ] && [ ! -s "$D/score_int8-w8a8_$v.json" ]; then todo=1; fi
    [ $todo -eq 1 ] || continue
    log "$TAG: draw s = $s"
    build_uaq "$s" || continue
    score_bf16_fp8 "$W/uaq_s$s" "$v"
    score_int8 "$W/uaq_s$s" "$v"
    rm -rf "$W/uaq_s$s"
    prune_cache
  done
  rm -rf "$W"
  prune_cache
  grep -E "^== score |Selected [A-Za-z0-9]+ for [A-Za-z0-9]+|Initializing a V1 LLM engine \(v" "$LOGF" 2>/dev/null \
    | sed -E 's/^.*(Selected [A-Za-z0-9]+ for [A-Za-z0-9]+).*$/  \1/; s/^.*Initializing a V1 LLM engine \((v[^)]*)\).*$/  vLLM \1/; s/ model=.*$//' \
    > "$D/kernels.txt" || true
  log "$TAG finished in $(( ($(date +%s) - T0) / 60 )) min"
done

# Completeness: every BF16 and FP8 score of every model; INT8 W8A8 is optional.
missing=()
for spec in $MODELS; do
  TAG=${spec#*=}
  for s0 in orig 1.5 $DRAWS; do
    if [ "$s0" = orig ]; then v=orig; else v="s$(printf '%g' "$s0")"; fi
    for q in none fp8-per-tensor fp8-per-channel; do
      [ -s "$OUT/$TAG/score_${q}_$v.json" ] || missing+=("$TAG/score_${q}_$v.json")
    done
    if [ "$INT8" = 1 ] && [ ! -s "$OUT/$TAG/score_int8-w8a8_$v.json" ]; then INT8_FAILED+=("$TAG/$v"); fi
  done
done
INT8_MISSING=$(printf '%s\n' "${INT8_FAILED[@]:-}" | grep . | sort -u | tr '\n' ' ' || true)
PEAK_GB=$(awk -v b="$PEAK" 'BEGIN { printf "%.1f", b / 2^30 }')
{
  echo "finished $(date -Is): peak size of the work directory $PEAK_GB GB; free disk now $(free_gb) GB"
  echo "INT8 W8A8 missing: ${INT8_MISSING:-none}"
} >> "$OUT/manifest.txt"

if [ ${#missing[@]} -gt 0 ]; then
  printf '%s\n' "${missing[@]}" > "$OUT/incomplete.txt"
  log "x6 incomplete: ${#missing[@]} BF16/FP8 scores missing (incomplete.txt); run the job again to resume"
  exit 1
fi

# The frozen analysis, run here as well (it reads S1's results from $S1_RESULTS); a failure does not undo the data.
if ! "$PY" -I "$REV_DIR/x6_analysis.py" --x6 "$OUT" --s1 "$S1_RESULTS" > /dev/null 2> "$OUT/x6_analysis.err"; then
  log "x6_analysis.py failed (see x6_analysis.err); the data are complete"
fi
[ -s "$OUT/x6_analysis.err" ] || rm -f "$OUT/x6_analysis.err"
{
  echo "complete $(date -Is)"
  echo "scores: $(ls "$OUT"/*/score_*.json | wc -l); INT8 W8A8 missing: ${INT8_MISSING:-none}; peak work dir $PEAK_GB GB"
} > "$OUT/done"
log "x6_fp8null done"
