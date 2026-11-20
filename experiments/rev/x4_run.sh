#!/usr/bin/env bash
# X4 (revision): rounding-divergence test at s = 1.5 on S2's loop. Does the rounding between UAQ (R4) and the matched
# settings (R5) act like ulp noise? Run on the H100:
#   bash ~/qurl/rev/x4_run.sh
# Needs ~/qurl/s2/s2_train.py (S2's loop, imported unchanged) and ~/venvs/qurl; GSM8K is copied from S2's results
# or downloaded.
# Arms (s = 1.5, lr 1e-6, seed 0, 50 rollout batches of 8 prompts x 4 samples). They run in this order, and
# finished arms are skipped on a re-run:
#   R4             UAQ                                                    the question, first arm
#   R5             no UAQ, Theorem 2(b)-(c) settings                      the question, second arm
#   R5p            R5 with every W_QKV and gamma element +-1 ulp at step 0  ulp-noise control
#   R4_repeat      identical to R4                                        determinism check: must be bit-identical
#   R5_seed1       R5 with sampling seed 1, same prompts                  seed-change reference
#   R4p            R4 with every W'_QKV and gamma' element +-1 ulp at step 0  second ulp-noise control
#   R5_seed1_full  R5 with seed 1 for sampling and data order             full seed change (different prompts)
# Then x4_analysis.py writes x4_report.txt (last line: the verdict) and x4_metrics.json, and the job writes
# results/x4_rounding/done.
# Budget: about 20 min per arm on the H100 NVL (S2: 22.6 s per batch), so about 2.4 h for all 7 arms. Disk: about
# 120 MB per arm.
# Environment overrides:
#   X4_STEPS   rollout batches per arm (default 50)
#   X4_ARMS    arms to run, in order (default: all seven above)
#   X4_EXTRA   extra x4_train.py arguments for every arm and the probe (smoke tests: "--tiny --reward toy ...")
#   X4_RES     results directory (default: <this folder>/results/x4_rounding)
#   X4_PY      python (default ~/venvs/qurl/bin/python)
#   X4_S2_DIR  folder holding s2_train.py (default ~/qurl/s2)
set -euo pipefail
REV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export X4_S2_DIR="${X4_S2_DIR:-$HOME/qurl/s2}"
RES="${X4_RES:-$REV/results/x4_rounding}"
PY="${X4_PY:-$HOME/venvs/qurl/bin/python}"
STEPS="${X4_STEPS:-50}"
ARMS="${X4_ARMS:-R4 R5 R5p R4_repeat R5_seed1 R4p R5_seed1_full}"
read -r -a EXTRA <<< "${X4_EXTRA:-}"

echo "== X4 start $(date -Is): $STEPS steps per arm; arms: $ARMS; results: $RES"
[ -x "$PY" ] || { echo "!! no python at $PY"; exit 1; }
[ -f "$X4_S2_DIR/s2_train.py" ] || { echo "!! no $X4_S2_DIR/s2_train.py (S2's loop is imported from there)"; exit 1; }
for f in x4_train.py x4_analysis.py; do
  [ -f "$REV/$f" ] || { echo "!! no $REV/$f"; exit 1; }
done
mkdir -p "$RES"
rm -f "$RES/done"     # "done" marks a completed run of this script; a re-run that fails must not leave an old one
avail_kb=$(df -Pk "$RES" | awk 'NR == 2 {print $4}')
if [ "$avail_kb" -lt $((5 * 1024 * 1024)) ]; then
  echo "!! less than 5 GB free under $RES"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee "$RES/gpu.txt" || true
used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')
if [ -n "$used_mb" ] && [ "$used_mb" -gt 2000 ]; then
  echo "   warning: $used_mb MiB already in use on the GPU (results stay deterministic; timing will suffer)"
fi
{ (cd "$REV" && sha256sum x4_train.py x4_analysis.py x4_run.sh); sha256sum "$X4_S2_DIR/s2_train.py"; } | tee "$RES/sha256.txt"

DATA="$RES/gsm8k_train.jsonl"
if [ ! -f "$DATA" ]; then
  for f in "$X4_S2_DIR"/results/*/gsm8k_train.jsonl; do
    if [ -f "$f" ]; then cp "$f" "$DATA"; echo "   GSM8K copied from $f"; break; fi
  done
fi
PROBE="$RES/probe.npz"
if [ ! -f "$PROBE" ]; then
  echo "-- probe set $(date -Is)"
  if ! "$PY" -u "$REV/x4_train.py" --make-probe --probe "$PROBE" --data "$DATA" ${EXTRA[@]+"${EXTRA[@]}"} \
       2>&1 | tee "$RES/probe.log"; then
    echo "!! probe set failed; see $RES/probe.log"; exit 1
  fi
fi

run_arm() {
  local tag=$1; shift
  if [ -f "$RES/$tag/done" ]; then echo "-- skip $tag (done)"; return 0; fi
  echo "-- $tag start $(date -Is): $*"
  if ! "$PY" -u "$REV/x4_train.py" --out "$RES/$tag" --steps "$STEPS" --probe "$PROBE" --data "$DATA" "$@" \
       ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$RES/$tag.log"; then
    echo "!! $tag failed at $(date -Is); see $RES/$tag.log"; exit 1
  fi
  touch "$RES/$tag/done"
  echo "-- $tag done $(date -Is) (job elapsed $((SECONDS / 60)) min)"
}

for tag in $ARMS; do
  case "$tag" in
    R4|R4_repeat)  run_arm "$tag" --arm R4 ;;
    R5)            run_arm "$tag" --arm R5 ;;
    R5p)           run_arm "$tag" --arm R5 --perturb W,gamma ;;
    R4p)           run_arm "$tag" --arm R4 --perturb W,gamma ;;
    R5_seed1)      run_arm "$tag" --arm R5 --seed 1 --data-seed 0 ;;
    R5_seed1_full) run_arm "$tag" --arm R5 --seed 1 --data-seed 1 ;;
    *) echo "!! unknown arm $tag"; exit 1 ;;
  esac
done

echo "-- analysis $(date -Is)"
"$PY" -I "$REV/x4_analysis.py" "$RES"
touch "$RES/done"
echo "== X4 done $(date -Is) (job elapsed $((SECONDS / 60)) min); verdict in $RES/x4_report.txt"
