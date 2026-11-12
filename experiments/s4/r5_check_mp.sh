#!/usr/bin/env bash
# Gate G3 (S4): R5's exactness check on all GPUs (r5_check_mp.py) for one model size, at s = 2: three actor updates on
# fixed per-GPU batches in VeRL's FSDP2 engine for R4 (UAQ checkpoint, plain settings), an R4 repeat, R5 (Theorem
# 2(b)-(c) settings) and R5o (R5 with ordinary clipping). Expected: R4 repeat and R5 bit-identical to R4; R5o different.
#   bash ~/qurl/s4/r5_check_mp.sh <1.5b|7b>     (needs the GPUs to itself; 1.5B: minutes, 7B: about half an hour)
# Writes ~/qurl/s4/results/gates/r5_check_<size>/{R4,R4r,R5,R5o}/run.json, the logs, compare.txt, and PASS or FAIL.
set -euo pipefail
SIZE=$1
S4="$HOME/qurl/s4"
PY="$HOME/venvs/verl/bin/python"
TORCHRUN="$HOME/venvs/verl/bin/torchrun"
NGPU=${NGPU:-8}
OUT="$S4/results/gates/r5_check_$SIZE"
case "$SIZE" in
  7b)   BASE=Qwen/Qwen2.5-Math-7B;   M2="$HOME/qurl/models/qwen2.5-math-7b-uaq2" ;;
  1.5b) BASE=Qwen/Qwen2.5-Math-1.5B; M2="$HOME/qurl/models/qwen2.5-math-1.5b-uaq2" ;;
  *) echo "size must be 1.5b or 7b" >&2; exit 1 ;;
esac
[ -f "$M2/uaq.json" ] || { echo "no s = 2 UAQ checkpoint at $M2 (setup_s4.sh writes it)" >&2; exit 1; }
mkdir -p "$OUT"
rm -f "$OUT/PASS" "$OUT/FAIL"
cd "$S4"
export PYTHONPATH="$S4${PYTHONPATH:+:$PYTHONPATH}"
for arm in R4 R4r R5 R5o; do
  case $arm in
    R4|R4r) A=R4; M=$M2 ;;
    *) A=$arm; M=$BASE ;;
  esac
  "$TORCHRUN" --standalone --nproc_per_node "$NGPU" r5_check_mp.py run --arm "$A" --model "$M" --out "$OUT/$arm" \
    > "$OUT/$arm.log" 2>&1 || { echo "$arm failed; see $OUT/$arm.log" >&2; touch "$OUT/FAIL"; exit 1; }
  grep -h "grad norms (before" "$OUT/$arm.log" | sed "s/^/$arm: /"
done
for other in R4r R5 R5o; do "$PY" r5_check_mp.py compare "$OUT/R4" "$OUT/$other"; echo; done | tee "$OUT/compare.txt"
# PASS: the R4 repeat and R5 bit-identical to R4 (2 lines), R5o different (1 line).
if [ "$(grep -c "BIT-IDENTICAL" "$OUT/compare.txt")" = 2 ] && [ "$(grep -c "  DIFFERENT" "$OUT/compare.txt")" = 1 ]; then
  touch "$OUT/PASS"; echo "G3 $SIZE: PASS"
else
  touch "$OUT/FAIL"; echo "G3 $SIZE: FAIL"; exit 1
fi
