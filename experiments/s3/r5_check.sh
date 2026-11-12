#!/usr/bin/env bash
# R5's exactness check (r5_check.py) on this machine: three actor updates on one fixed batch in VeRL's FSDP2 engine,
# at s = 2, for R4 (UAQ checkpoint, plain settings), an R4 repeat, R5 (Theorem 2(b)-(c) settings) and R5o (R5 with
# ordinary clipping). Expected: R4 repeat and R5 bit-identical to R4; R5o different.
#   bash ~/qurl/s3/r5_check.sh        (a few minutes; needs the GPU to itself)
# Writes ~/qurl/s3/results/r5_check/{R4,R4r,R5,R5o}/ and compare.txt.
set -euo pipefail
S3="$HOME/qurl/s3"
PY="$HOME/venvs/verl/bin/python"
OUT="$S3/results/r5_check"
M2="$HOME/qurl/models/qwen0.5b-instruct-uaq2"
mkdir -p "$OUT"
if [ ! -f "$M2/uaq.json" ]; then
  CUDA_VISIBLE_DEVICES= "$PY" "$S3/make_uaq_checkpoint.py" --s 2 --out "$M2"      # on the CPU, about a minute
fi
cd "$S3"
for arm in R4 R4r R5 R5o; do
  case $arm in
    R4|R4r) A=R4; M=$M2 ;;
    *) A=$arm; M=Qwen/Qwen2.5-0.5B-Instruct ;;
  esac
  "$PY" r5_check.py run --arm "$A" --model "$M" --out "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  grep -h "grad norms (before" "$OUT/$arm.log" | sed "s/^/$arm: /"
done
for other in R4r R5 R5o; do "$PY" r5_check.py compare "$OUT/R4" "$OUT/$other"; echo; done | tee "$OUT/compare.txt"
