#!/usr/bin/env bash
# S4's queue for the 5-day rental: the runs of the analysis plan, in priority order, one at a time on all 8 GPUs.
#   END="2026-11-30 08:00 UTC" nohup bash ~/qurl/s4/queue.sh > ~/qurl/s4/queue.log 2>&1 < /dev/null &   # start/resume
#   bash ~/qurl/s4/queue.sh --list         # plan and status of every run
#   bash ~/qurl/s4/queue.sh --progress     # one line per run: step, step time, score, mismatch, validation, eval
# Two kinds of run (README "Run order"):
#   m  mechanism run: 25 steps, seed 1, no validation, no checkpoints, no final evaluation (tag s4m_<size>_<arm>_s1).
#      Only its training metrics and code-flip log count (M1, early M2). The two R3 runs are made by gates.sh (G4).
#   f  full run: the paper's 200 steps, validation and checkpoints every 20 steps, final evaluation (s4_<size>_<arm>_s<k>).
# Order: M (mechanism runs 1.5B R4, R5, R6 and 7B R6) | A 7B R4/R5 (H1 at the paper's scale; their first 20 steps give
# M1 at 7B) | B 7B R3 (UAQ's effect, M2 over training, Table 4) | C 7B R6 (the paper's control) | D 1.5B R4/R5.
# Whatever the 5 days allow is the most useful prefix.
# Before the first run it requires results/gates/PASSED (gates.sh, gates G2-G4); FORCE=1 overrides.
# Per run: run.sh (training, summary; for full runs finish.sh = merge + evaluation). A failed run is retried once (a
# full run resumes from its last checkpoint, a mechanism run restarts); if it fails again it is marked FAILED and the
# queue moves on (STOP_ON_FAIL=1 stops instead), so a crash at night does not leave the GPUs idle. A hang is treated
# as a failure: watchdog.sh stops a run that writes no output for 60 min (or 3x its longest step; WATCH_MIN=0 disables).
# Time limit: with END set (any date `date -d` accepts), a run starts only if its projected length ends BUFFER_H hours
# (default 6) before END; otherwise it is skipped and a later, shorter run may still start. An R4/R5 pair starts only
# if both runs fit. Projected lengths: HOURS_7B, HOURS_1_5B (full runs) and HOURS_7B_M, HOURS_1_5B_M (mechanism
# runs), read from results/gates/hours.txt (gates.sh) unless set. Nothing is cut short to fit.
# DRY_RUN=1 prints the decisions without running anything (with FORCE=1 before the gates); with END set it simulates
# the clock: FORCE=1 DRY_RUN=1 END="now + 110 hours" HOURS_7B=28 HOURS_1_5B=11 HOURS_7B_M=4 HOURS_1_5B_M=2 bash queue.sh
set -uo pipefail
S4="$HOME/qurl/s4"
RESULTS="$S4/results"
PY="$HOME/venvs/verl/bin/python"
BUFFER_H=${BUFFER_H:-6}
MECH_STEPS=25
FULL_STEPS=200

plan() {   # "kind size arm seed tag runs" in run order; runs = 2 for the first run of an R4/R5 pair (the pair starts
           # only if both fit), 0 for its second (it follows the first), 1 otherwise
  {
    echo "m 1.5b R4 1 1"; echo "m 1.5b R5 1 1"; echo "m 1.5b R6 1 1"; echo "m 7b R6 1 1"   # M
    echo "f 7b R4 1 2"; echo "f 7b R5 1 0"                                                 # A
    echo "f 7b R3 1 1"                                                                     # B
    echo "f 7b R6 1 1"                                                                     # C
    echo "f 1.5b R4 1 2"; echo "f 1.5b R5 1 0"                                             # D
  } | while read -r kind size arm seed runs; do
    if [ "$kind" = m ]; then tag=s4m_${size}_${arm}_s${seed}; else tag=s4_${size}_${arm}_s${seed}; fi
    echo "$kind $size $arm $seed $tag $runs"
  done
}
status() {   # status <kind> <tag>
  local d="$RESULTS/$2"
  if [ "$1" = f ] && [ -s "$d/eval.json" ]; then echo "done"
  elif [ "$1" = m ] && [ -s "$d/summary.txt" ]; then echo "done"
  elif [ -s "$d/summary.txt" ]; then echo trained
  elif [ -e "$d/FAILED" ]; then echo failed
  elif [ -s "$d/metrics.jsonl" ]; then echo incomplete
  else echo pending; fi
}
evalline() {
  [ -s "$RESULTS/$1/eval.json" ] || return 0
  "$PY" - "$RESULTS/$1/eval.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
print("   eval: " + " | ".join(f"{b} " + " ".join(f"{k} {100 * v:.1f}%" for k, v in x.items() if k.startswith("avg@") or k == "greedy") for b, x in r.items()))
EOF
}

case "${1:-}" in
  --tags) plan | awk '{print $5}' | tr '\n' ' '; echo; exit 0 ;;
  --list) plan | while read -r kind size arm seed tag runs; do
            printf '%-4s %-5s %-3s seed %s  %-17s %s\n' "$kind" "$size" "$arm" "$seed" "$tag" "$(status "$kind" "$tag")"
          done; exit 0 ;;
  --progress)
    for tag in s4m_1.5b_R3_s1 s4m_7b_R3_s1 $(plan | awk '{print $5}'); do
      [ -d "$RESULTS/$tag" ] || continue
      steps=$FULL_STEPS; case $tag in s4m_*) steps=$MECH_STEPS ;; esac
      "$PY" "$S4/summarize.py" "$RESULTS/$tag" --brief --full-steps "$steps"; evalline "$tag"
    done; exit 0 ;;
  "") ;;
  *) echo "usage: queue.sh [--list|--tags|--progress]" >&2; exit 2 ;;
esac

log() { echo "[$(date -Is)] $*"; }
. "$S4/watchdog.sh"
END_S=""
if [ -n "${END:-}" ]; then
  END_S=$(date -d "$END" +%s) || { log "cannot parse END=$END"; exit 1; }
  if [ -z "${HOURS_7B:-}" ] && [ -f "$RESULTS/gates/hours.txt" ]; then . "$RESULTS/gates/hours.txt"; fi
  for v in HOURS_7B HOURS_1_5B HOURS_7B_M HOURS_1_5B_M; do
    [ -n "${!v:-}" ] || { log "END needs $v (gates.sh writes results/gates/hours.txt)"; exit 1; }
  done
  log "time limit: END $END; projected hours per run: full 7B $HOURS_7B, 1.5B $HOURS_1_5B; mechanism 7B" \
      "$HOURS_7B_M, 1.5B $HOURS_1_5B_M"
fi
hours() {   # hours <kind> <size>
  case "$1/$2" in f/7b) echo "$HOURS_7B" ;; f/*) echo "$HOURS_1_5B" ;; m/7b) echo "$HOURS_7B_M" ;; *) echo "$HOURS_1_5B_M" ;; esac
}
SIM_T=$(date +%s)   # DRY_RUN's simulated clock: each run it would start advances it by that run's projected hours
fits() {   # fits <kind> <size> <runs>: do <runs> runs of this kind and size, started now, end BUFFER_H h before END?
  [ -z "$END_S" ] && return 0
  local now; now=$([ "${DRY_RUN:-0}" = 1 ] && echo "$SIM_T" || date +%s)
  "$PY" -c "import sys; sys.exit(0 if $now + 3600 * ($3 * $(hours "$1" "$2") + $BUFFER_H) <= $END_S else 1)"
}

log "S4 queue: $(plan | wc -l) runs${END:+, end $END (buffer ${BUFFER_H} h)}"
md5sum "$S4"/run.sh "$S4"/finish.sh "$S4"/eval.py "$S4"/qurl_verl/*.py 2>/dev/null | sed 's/^/  code: /'
if [ ! -e "$RESULTS/gates/PASSED" ] && [ "${FORCE:-0}" != 1 ]; then
  log "gates have not passed (no $RESULTS/gates/PASSED): run gates.sh first, or FORCE=1"
  exit 1
fi

skip_pair=0
plan | while read -r kind size arm seed tag runs; do
  st=$(status "$kind" "$tag")
  if [ "$runs" = 0 ] && [ "$skip_pair" = 1 ]; then log "skip $tag (its pair's first run did not fit)"; continue; fi
  skip_pair=0
  case "$st" in
    done) log "skip $tag (done)"; continue ;;
    failed) log "skip $tag (FAILED: $(head -1 "$RESULTS/$tag/FAILED"))"; continue ;;
    trained) log "finish $tag (trained, not yet evaluated)"
             watched "$tag" bash "$S4/finish.sh" "$size" "$tag" || log "finish.sh failed for $tag"; continue ;;
  esac
  if [ "$runs" -gt 0 ] && [ "$st" = pending ] && ! fits "$kind" "$size" "$runs"; then
    log "skip $tag: $runs $size run(s) would not end ${BUFFER_H} h before $END"
    [ "$runs" = 2 ] && skip_pair=1
    continue
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    [ -n "$END_S" ] && SIM_T=$("$PY" -c "print(int($SIM_T + 3600 * $(hours "$kind" "$size")))")
    log "DRY RUN: $kind $tag${END_S:+ (would end $(date -d "@$SIM_T" '+%m-%d %H:%M'))}"
    continue
  fi
  ok=0
  for attempt in 1 2; do
    log "start $tag (kind $kind, size $size, arm $arm, seed $seed; attempt $attempt, status $st)"
    if [ "$kind" = m ]; then
      watched "$tag" env SEED="$seed" NO_FINISH=1 VAL_EVERY=-1 VAL_BEFORE=False SAVE_EVERY=-1 \
        bash "$S4/run.sh" "$size" "$arm" "$tag" "$MECH_STEPS"
    else
      watched "$tag" env SEED="$seed" bash "$S4/run.sh" "$size" "$arm" "$tag" "$FULL_STEPS"
    fi
    rc=$?
    log "end $tag: exit $rc"
    [ "$(status "$kind" "$tag")" = "done" ] && { ok=1; break; }
    if [ $rc -eq 0 ] && [ "$kind" = f ]; then   # trained but finish.sh failed: retry the finish only
      watched "$tag" bash "$S4/finish.sh" "$size" "$tag" && [ -s "$RESULTS/$tag/eval.json" ] && { ok=1; break; }
    fi
    st=incomplete
  done
  if [ $ok != 1 ]; then
    echo "exit $rc after 2 attempts, $(date -Is)" > "$RESULTS/$tag/FAILED"
    log "$tag FAILED twice; see results/$tag/train.log"
    [ "${STOP_ON_FAIL:-0}" = 1 ] && { log "stopping (STOP_ON_FAIL=1)"; exit 1; }
  fi
done || exit 1
log "S4 queue finished"
bash "$0" --progress
