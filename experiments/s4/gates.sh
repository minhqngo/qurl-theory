#!/usr/bin/env bash
# S4 day 1: the gates that must pass before queue.sh may start, the R3 mechanism runs (which are also the timing
# pilots), and the base models' evaluation.
#   nohup bash ~/qurl/s4/gates.sh > ~/qurl/s4/gates.log 2>&1 < /dev/null &      (about 7 h after setup_s4.sh)
# Writes ~/qurl/s4/results/gates/: summary.txt, hours.txt, and PASSED if G2-G4 pass. Steps (README "Day 1"):
#   G1  machine: GPU count and type, flash-attn or SDPA (from setup_s4.sh's env.txt), the node checks' warnings and
#       failures (bootstrap_node.sh, check_node.py), the analysis plan's hash; recorded, and it stops the gates only
#       when check_node.py found a failure (then nothing can run)
#   G2  packed attention: VeRL's padding-free forward gives the same log-probs as one sequence at a time, with the
#       attention the runs will use (test_packed_attn.py, Qwen2.5-Math-1.5B)
#   G3  R5's exactness on all GPUs at s = 2, 1.5B and 7B (r5_check_mp.sh)
#   G4  the R3 mechanism runs at 1.5B and 7B (25 steps each, the analysis plan's runs s4m_<size>_R3_s1), with the
#       test dump: the INT8 codes after the first sync equal an independent quantization of the starting weights
#       (check_int8_dump.py), every replica re-quantized every weight at every sync, and the code-flip log is
#       written and plausible
#   G5  timing -> hours.txt: full runs (HOURS_7B, HOURS_1_5B) = 1.25 x 200 x the median step time, for response
#       growth, + 2 h for validations, checkpoints, merge and evaluation; mechanism runs (HOURS_7B_M, HOURS_1_5B_M)
#       = 1.1 x the R3 mechanism run's measured wall time
#   G6  evaluation of the two base models (step 0 of every run): results/base_7b/eval.json, results/base_1.5b/eval.json
# A failed gate stops the script (later gates are not run); fix it and rerun: passed steps are skipped.
set -uo pipefail
S4="$HOME/qurl/s4"
PY="$HOME/venvs/verl/bin/python"
G="$S4/results/gates"
mkdir -p "$G"
rm -f "$G/PASSED"
log() { echo "[$(date -Is)] $*" | tee -a "$G/summary.txt"; }
export PYTHONPATH="$S4${PYTHONPATH:+:$PYTHONPATH}"

log "== G1 machine"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | tee -a "$G/summary.txt"
cat "$S4/env.txt" >> "$G/summary.txt" 2>/dev/null || log "no env.txt: run setup_s4.sh first"
n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$n" = "${NGPU:-8}" ] || log "WARNING: $n GPUs visible, NGPU=${NGPU:-8}"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -qv "^9\.0" && \
  log "WARNING: not every GPU is an H100 (compute capability 9.0); S1-S3's INT8 kernel was CUTLASS on SM90"
# The node checks: bootstrap_node.sh's (driver, fabric, health; no PyTorch) and check_node.py's (under load). The
# latter normally ran in setup_s4.sh; it runs here if it did not.
NODE="$S4/results/node"
[ -s "$NODE/check_node.txt" ] || "$PY" "$S4/check_node.py" --out "$NODE/check_node.txt" > /dev/null 2>&1
for f in bootstrap check_node; do
  if [ -s "$NODE/$f.txt" ]; then
    log "$f: $(grep -c '^PASS' "$NODE/$f.txt") PASS, $(grep -c '^WARN' "$NODE/$f.txt") WARN, $(grep -c '^FAIL' "$NODE/$f.txt") FAIL"
    grep -E '^(WARN|FAIL)' "$NODE/$f.txt" | sed "s/^/  $f: /" | tee -a "$G/summary.txt"
  else
    log "WARNING: no results/node/$f.txt (README, Day 1)"
  fi
done
if grep -q '^FAIL' "$NODE/check_node.txt" 2>/dev/null; then
  log "G1 FAILED: check_node.py found a failure (results/node/check_node.txt)"; exit 1
fi
# The analysis plan was frozen on 2026-11-11 (README); a changed s4_analysis.py must be recorded and reported with both
# versions' results, so a mismatch is logged loudly but does not stop the runs.
if (cd "$S4" && sha256sum -c s4_analysis.sha256 > /dev/null 2>&1); then
  log "analysis plan: s4_analysis.py matches its frozen SHA-256"
else
  log "WARNING: s4_analysis.py differs from its frozen SHA-256 (s4_analysis.sha256): record the change (README)"
fi

log "== G2 packed attention"
if [ "$(grep -c '^PASS' "$G/packed_attn.txt" 2>/dev/null)" != 2 ]; then
  {  # float32 SDPA: the definitive check; bfloat16: SDPA and flash-attn (if installed) as the runs use them
    "$PY" "$S4/test_packed_attn.py" --model Qwen/Qwen2.5-Math-1.5B --attn sdpa --dtype float32
    "$PY" "$S4/test_packed_attn.py" --model Qwen/Qwen2.5-Math-1.5B --attn sdpa flash_attention_2 --dtype bfloat16 \
      --timing-tokens 16384
  } > "$G/packed_attn.txt" 2>&1
fi
grep -E "OK|WRONG|not available|packed forward|^PASS|^FAIL" "$G/packed_attn.txt" | tee -a "$G/summary.txt"
[ "$(grep -c '^PASS' "$G/packed_attn.txt")" = 2 ] || { log "G2 FAILED"; exit 1; }

for size in 1.5b 7b; do
  log "== G3 R5 exactness, $size"
  [ -e "$G/r5_check_$size/PASS" ] || bash "$S4/r5_check_mp.sh" "$size" >> "$G/summary.txt" 2>&1
  [ -e "$G/r5_check_$size/PASS" ] || { log "G3 $size FAILED (see $G/r5_check_$size/compare.txt)"; exit 1; }
  log "G3 $size PASS"
done

for size in 1.5b 7b; do
  tag=s4m_${size}_R3_s1
  log "== G4 R3 mechanism run, $size ($tag, 25 steps)"
  if [ ! -s "$S4/results/$tag/summary.txt" ]; then
    rm -rf "$S4/results/$tag" "${CKPT_ROOT:-$S4/ckpt}/$tag"
    SEED=1 DUMP=1 NO_FINISH=1 VAL_EVERY=-1 VAL_BEFORE=False SAVE_EVERY=-1 bash "$S4/run.sh" "$size" R3 "$tag" 25 \
      < /dev/null >> "$G/summary.txt" 2>&1 || { log "G4 $size: the run failed (results/$tag/train.log)"; exit 1; }
  fi
  base=$([ "$size" = 7b ] && echo Qwen/Qwen2.5-Math-7B || echo Qwen/Qwen2.5-Math-1.5B)
  "$PY" "$S4/check_int8_dump.py" "$S4/results/$tag/int8_dump" --model "$base" > "$G/int8_dump_$size.txt" 2>&1
  tail -5 "$G/int8_dump_$size.txt" | tee -a "$G/summary.txt"
  grep -q "^PASS" "$G/int8_dump_$size.txt" || { log "G4 $size FAILED: INT8 codes"; exit 1; }
  grep "INT8 refit:" "$S4/results/$tag/summary.txt" | tee -a "$G/summary.txt"
  [ -s "$S4/results/$tag/e3_int8.jsonl" ] || { log "G4 $size FAILED: no E3 log"; exit 1; }
  # Plausibility: at lr 1e-6 a step moves weights by a few percent of a grid step, so a few percent of codes flip
  # at most. Shares near 100% mean the history is wrong (as when it was read back from vLLM's slept memory).
  "$PY" - "$S4" "$S4/results/$tag" <<'EOF' | tee -a "$G/summary.txt" || { log "G4 $size FAILED: implausible flip shares"; exit 1; }
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from s4_analysis import OTHER, QKV, load_e3, pooled_flips
recs = load_e3(Path(sys.argv[2]))
q, n = pooled_flips(recs, QKV)
o, _ = pooled_flips(recs, OTHER)
print(f"code-flip shares over {n} syncs: q/k/v {q:.4%}, other projections {o:.4%}")
sys.exit(0 if n >= 20 and q < 0.2 and o < 0.2 else 1)
EOF
  log "G4 $size PASS"
done
touch "$G/PASSED"
log "gates G2-G4 passed: queue.sh may start"

log "== G5 timing"
"$PY" - "$S4/results" <<'EOF' | tee "$G/hours.txt" | tee -a "$G/summary.txt"
import datetime, re, statistics, sys
sys.path.insert(0, sys.argv[1] + "/..")
from summarize import load_metrics
for size, var in (("7b", "7B"), ("1.5b", "1_5B")):
    root = f"{sys.argv[1]}/s4m_{size}_R3_s1"
    by = load_metrics(root)
    t = [by[k]["timing_s/step"] for k in sorted(by) if "timing_s/step" in by[k]]
    med = statistics.median(t[1:])
    stamps = re.findall(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d)", open(f"{root}/arm.txt").read())
    wall = (datetime.datetime.fromisoformat(stamps[-1]) - datetime.datetime.fromisoformat(stamps[0])).total_seconds()
    print(f"HOURS_{var}={1.25 * 200 * med / 3600 + 2:.1f}")
    print(f"HOURS_{var}_M={1.1 * wall / 3600:.1f}")
EOF
log "(full runs: 1.25 x 200 x the median step time of the 25-step run + 2 h; mechanism runs: 1.1 x its wall time)"

log "== G6 base models"
for size in 1.5b 7b; do
  out="$S4/results/base_$size"
  mkdir -p "$out"
  base=$([ "$size" = 7b ] && echo Qwen/Qwen2.5-Math-7B || echo Qwen/Qwen2.5-Math-1.5B)
  [ -s "$out/eval.json" ] || "$PY" "$S4/eval.py" --model "$base" --out "$out/eval.json" --gpus "${NGPU:-8}" --step 0 \
    > "$out/eval.log" 2>&1 || log "G6 $size: evaluation failed (see $out/eval.log)"
  grep -E "^(aime24|math500):" "$out/eval.log" | sed "s/^/base $size /" | tee -a "$G/summary.txt"
done
log "day-1 gates done"
