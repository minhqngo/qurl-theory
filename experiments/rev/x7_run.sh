#!/usr/bin/env bash
# X7: S1's INT8 W8A8 identity check, repeated on vLLM 0.24. Every RL run (S3, E2, X3, X5) used VeRL 0.9.1 with
# vLLM 0.24 (~/venvs/verl) for its INT8 rollouts; S1 measured Theorem 1 on vLLM 0.33 (~/venvs/qurl).
# Model: Qwen2.5-0.5B-Instruct (S1's tag qwen0.5b-instruct, S3's model). Same 512 responses, same checkpoints and
# the same scorer, S1's e0_logprobs.py, unchanged (it runs on vLLM 0.24 as is), now with ~/venvs/verl/bin/python:
#   INT8 W8A8 (llm-compressor checkpoints, vLLM's CUTLASS INT8 kernel) and BF16 (the reference policies), each for
#   the original, the original with shuffled batches (noise floor), UAQ s = 2 and UAQ s = 1.5.
# S1's responses (~/qurl/h100/results/qwen0.5b-instruct/responses.json) and checkpoints (~/e0_work/qwen0.5b-instruct/
# {uaq_s2,uaq_s1.5,int8_w8a8_orig,int8_w8a8_s2,int8_w8a8_s1.5}) are reused when they exist. Missing ones are rebuilt
# with S1's own scripts and venvs (responses: e0_logprobs.py generate on ~/venvs/qurl; UAQ: make_uaq_checkpoints.py;
# INT8: make_int8_w8a8.py on ~/venvs/llmc) in ~/qurl/rev/tmp/x7_work, which is deleted at the end.
#   bash ~/qurl/rev/x7_run.sh           (about 10 minutes on the H100; 15 if the checkpoints must be rebuilt)
# Outputs in ~/qurl/rev/results/x7_s1_vllm024/: score_<none|int8-w8a8>_<orig|orig_shuffled|s2|s1.5>.json (S1's
# layout), manifest.txt (versions, sources, hashes), int8_codes.txt (S1's compare_int8_codes.py), kernels.txt,
# run.log, summary.txt (S1's analyze_e0.py), s1_analysis.txt (S1's analyze_s1.py), x7_analysis.txt, responses.json
# (only if S1's were missing and had to be regenerated), and done.
# Overrides (smoke tests): REV_DIR, E0_DIR, H100_DIR, S1_RESULTS, S1_WORK, VENVS, X7_N (responses to regenerate),
# X_GPU_WAIT (seconds to wait for a busy GPU; 0 skips the check). On a GPU without CUTLASS INT8 (e.g. SM120) set
# VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel, as in S1's laptop run.
set -euo pipefail
export LC_NUMERIC=C
REV_DIR=${REV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
E0_DIR=${E0_DIR:-$HOME/qurl/e0}
H100_DIR=${H100_DIR:-$HOME/qurl/h100}
S1_RESULTS=${S1_RESULTS:-$H100_DIR/results}
S1_WORK=${S1_WORK:-$HOME/e0_work}
VENVS=${VENVS:-$HOME/venvs}
VPY="$VENVS/verl/bin/python"         # the RL runs' stack: VeRL 0.9.1, vLLM 0.24 (all scoring)
QPY="$VENVS/qurl/bin/python"         # S1's stack: vLLM 0.33 (responses if missing, UAQ checkpoints, S1's analyses)
LLMC="$VENVS/llmc/bin/python"        # llm-compressor (CPU): INT8 W8A8 checkpoints if missing
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
TAG=qwen0.5b-instruct
OUT="$REV_DIR/results/x7_s1_vllm024"
WORK="$REV_DIR/tmp/x7_work"
LOGF="$OUT/run.log"
export VLLM_USE_FLASHINFER_SAMPLER=0  # as in S1 and S3: no nvcc on the server for FlashInfer's JIT sampler

log() { echo "[$(date -Is)] $*"; }
[ -f "$OUT/done" ] && { log "skip x7_s1_vllm024 (done)"; exit 0; }
mkdir -p "$OUT"
for f in e0_logprobs.py make_uaq_checkpoints.py make_int8_w8a8.py compare_int8_codes.py; do
  [ -f "$E0_DIR/$f" ] || { log "missing $E0_DIR/$f (S1's scripts): stopping"; exit 1; }
done
for p in "$VPY" "$QPY"; do [ -x "$p" ] || { log "missing $p: stopping"; exit 1; }; done

GPU_WAIT=${X_GPU_WAIT:-1200}         # seconds to wait for a busy GPU; 0 skips the check (WSL lists host processes)
t=0
while [ "$GPU_WAIT" -gt 0 ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
  [ $t -eq 0 ] && log "waiting for the GPU (another process is using it)"
  t=$((t + 30))
  [ $t -le "$GPU_WAIT" ] || { log "the GPU is still busy after $GPU_WAIT s: stopping"; exit 1; }
  sleep 30
done

complete_ckpt() { [ -f "$1/config.json" ] && compgen -G "$1/*.safetensors" > /dev/null; }
ORIG=$("$QPY" - "$MODEL_ID" <<'EOF'
import sys
from huggingface_hub import snapshot_download
try:
    print(snapshot_download(sys.argv[1], local_files_only=True))
except Exception:
    print(snapshot_download(sys.argv[1]))
EOF
)
M="$OUT/manifest.txt"
{
  echo "x7_s1_vllm024 manifest (x7_run.sh), $(date -Is) on $(hostname)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
  "$VPY" -c "import vllm, torch, transformers, compressed_tensors as c; print('scoring stack (verl venv): vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| compressed-tensors', c.__version__)"
  "$QPY" -c "import vllm; print('S1 stack (qurl venv): vllm', vllm.__version__)"
  echo "VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-}"
  echo "model: $MODEL_ID at $ORIG"
  md5sum "$E0_DIR/e0_logprobs.py" "$E0_DIR/make_uaq_checkpoints.py" "$E0_DIR/make_int8_w8a8.py" "$0" | sed 's/^/code: /'
} > "$M"

# 1. Responses: S1's, else regenerated with S1's script and venv (512 GSM8K test problems, vLLM 0.33, seed 0).
RESP="$S1_RESULTS/$TAG/responses.json"
if [ -s "$RESP" ]; then
  echo "responses: S1's $RESP" >> "$M"
else
  RESP="$OUT/responses.json"
  if [ ! -s "$RESP" ]; then
    log "S1's responses are missing: regenerating them with S1's e0_logprobs.py generate (vLLM 0.33)"
    "$QPY" "$E0_DIR/e0_logprobs.py" generate "$ORIG" "$RESP" --n "${X7_N:-512}" >> "$LOGF" 2>&1 || true
    [ -s "$RESP" ] || { log "generating the responses failed (see run.log)"; exit 1; }
  fi
  echo "responses: REGENERATED (S1's were missing) $RESP" >> "$M"
fi
echo "responses sha256: $(sha256sum "$RESP" | cut -d' ' -f1)" >> "$M"

# 2. Checkpoints: S1's where complete, else rebuilt with S1's scripts in $WORK (recreated on every start, so a
#    checkpoint half-written by an interrupted run is never used; rebuilding takes about a minute at 0.5B).
rm -rf "$WORK"
mkdir -p "$WORK"
declare -A CK
for v in s2 s1.5; do
  if complete_ckpt "$S1_WORK/$TAG/uaq_$v"; then CK[uaq_$v]=$S1_WORK/$TAG/uaq_$v
  else
    log "rebuilding uaq_$v with S1's make_uaq_checkpoints.py"
    if ! HF_HUB_OFFLINE=1 "$QPY" "$E0_DIR/make_uaq_checkpoints.py" "$MODEL_ID" "$WORK" "${v#s}" \
         >> "$OUT/checkpoints.jsonl" 2>> "$LOGF" || ! complete_ckpt "$WORK/uaq_$v"; then
      log "building uaq_$v failed (see run.log): stopping"
      exit 1
    fi
    CK[uaq_$v]=$WORK/uaq_$v
  fi
done
specs=()
for v in orig s2 s1.5; do
  if complete_ckpt "$S1_WORK/$TAG/int8_w8a8_$v"; then CK[int8_$v]=$S1_WORK/$TAG/int8_w8a8_$v
  else
    if [ "$v" = orig ]; then specs+=("orig=$ORIG"); else specs+=("$v=${CK[uaq_$v]}"); fi
    CK[int8_$v]=$WORK/int8_w8a8_$v
  fi
done
if [ ${#specs[@]} -gt 0 ]; then
  [ -x "$LLMC" ] || { log "missing $LLMC, needed to rebuild INT8 checkpoints: stopping"; exit 1; }
  log "rebuilding INT8 W8A8 checkpoints with S1's make_int8_w8a8.py: ${specs[*]}"
  if ! "$LLMC" "$E0_DIR/make_int8_w8a8.py" "$WORK" "${specs[@]}" >> "$OUT/int8_quantize.log" 2>&1; then
    log "building the INT8 W8A8 checkpoints failed (see int8_quantize.log): stopping"
    exit 1
  fi
fi
for k in uaq_s2 uaq_s1.5 int8_orig int8_s2 int8_s1.5; do
  complete_ckpt "${CK[$k]}" || { log "checkpoint $k missing at ${CK[$k]}: stopping"; exit 1; }
  case "${CK[$k]}" in "$S1_WORK"/*) src="S1's" ;; *) src="rebuilt" ;; esac
  echo "checkpoint $k: $src ${CK[$k]}" >> "$M"
done
for v in orig s2 s1.5; do
  for f in "${CK[int8_$v]}"/*.safetensors; do echo "sha256 int8_$v/$(basename "$f"): $(sha256sum "$f" | cut -d' ' -f1)"; done
done >> "$M"
# S1's check of the stored INT8 codes and scales (it expects <dir>/int8_w8a8_{orig,s2,s1.5}).
mkdir -p "$WORK/links"
for v in orig s2 s1.5; do ln -sfn "${CK[int8_$v]}" "$WORK/links/int8_w8a8_$v"; done
"$QPY" "$E0_DIR/compare_int8_codes.py" "$WORK/links" > "$OUT/int8_codes.txt" 2>> "$LOGF" || log "compare_int8_codes.py failed"

# 3. Scores on vLLM 0.24. Skipped if present; a non-zero exit with a complete file counts as success (S1's rule).
sc() {
  local model=$1 out=$2 try
  shift 2
  [ -s "$out" ] && return 0
  for try in 1 2; do
    echo "== score $(basename "$out") model=$model opts=$* try=$try $(date -Is)" >> "$LOGF"
    "$VPY" "$E0_DIR/e0_logprobs.py" score "$model" "$RESP" "$out" "$@" >> "$LOGF" 2>&1 || true
    if [ -s "$out" ]; then log "scored $(basename "$out")"; return 0; fi
    log "scoring $(basename "$out") failed (try $try; see run.log)"
    sleep 20
  done
  return 1
}
sc "${CK[int8_orig]}" "$OUT/score_int8-w8a8_orig.json" || true
sc "${CK[int8_orig]}" "$OUT/score_int8-w8a8_orig_shuffled.json" --shuffle || true
sc "${CK[int8_s2]}" "$OUT/score_int8-w8a8_s2.json" || true
sc "${CK[int8_s1.5]}" "$OUT/score_int8-w8a8_s1.5.json" || true
sc "$ORIG" "$OUT/score_none_orig.json" || true
sc "$ORIG" "$OUT/score_none_orig_shuffled.json" --shuffle || true
sc "${CK[uaq_s2]}" "$OUT/score_none_s2.json" || true
sc "${CK[uaq_s1.5]}" "$OUT/score_none_s1.5.json" || true

# 4. Which kernels and vLLM version each run used (per score: the engine line and vLLM's "Selected ... for ..." lines).
grep -E "^== score |Selected [A-Za-z0-9]+ for [A-Za-z0-9]+|Initializing a V1 LLM engine \(v" "$LOGF" \
  | sed -E 's/^.*(Selected [A-Za-z0-9]+ for [A-Za-z0-9]+).*$/  \1/; s/^.*Initializing a V1 LLM engine \((v[^)]*)\).*$/  vLLM \1/; s/ model=.*$//' \
  > "$OUT/kernels.txt" || true
n_int8=$(grep -c "for CompressedTensorsW8A8Int8" "$OUT/kernels.txt" || true)
n_cut=$(grep -c "Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8" "$OUT/kernels.txt" || true)
echo "INT8 kernel: CutlassInt8ScaledMMLinearKernel in $n_cut of $n_int8 INT8 engine starts" >> "$M"
[ "$n_int8" -gt 0 ] && [ "$n_cut" = "$n_int8" ] || log "WARNING: not every INT8 run selected CutlassInt8ScaledMMLinearKernel (kernels.txt)"

missing=()
for f in int8-w8a8_orig int8-w8a8_orig_shuffled int8-w8a8_s2 int8-w8a8_s1.5 none_orig none_orig_shuffled none_s2 none_s1.5; do
  [ -s "$OUT/score_$f.json" ] || missing+=("score_$f.json")
done
if [ ${#missing[@]} -gt 0 ]; then
  log "x7 incomplete, missing: ${missing[*]}; run the job again to resume"
  exit 1
fi

# 5. S1's own analyses on these files (same layout as S1's results), then X7's comparison with S1.
if [ -f "$E0_DIR/analyze_e0.py" ]; then
  "$QPY" "$E0_DIR/analyze_e0.py" "$OUT" > "$OUT/summary.txt" 2>> "$LOGF" || log "analyze_e0.py failed"
fi
if [ -f "$H100_DIR/analyze_s1.py" ]; then
  "$QPY" "$H100_DIR/analyze_s1.py" "$OUT" > "$OUT/s1_analysis.txt" 2>> "$LOGF" || log "analyze_s1.py failed"
fi
"$QPY" -I "$REV_DIR/x7_analysis.py" --x7 "$OUT" --s1 "$S1_RESULTS/$TAG" > /dev/null 2>> "$LOGF" \
  || log "x7_analysis.py failed (see run.log); the data are complete"
rm -rf "$WORK"
echo "complete $(date -Is); INT8 kernel: CutlassInt8ScaledMMLinearKernel in $n_cut of $n_int8 INT8 engine starts" > "$OUT/done"
log "x7_s1_vllm024 done"
