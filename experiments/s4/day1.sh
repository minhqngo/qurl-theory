#!/usr/bin/env bash
# S4 day 1 after the node bootstrap (README "Day 1"): setup_s4.sh, gates.sh and then queue.sh, unattended, so the
# queue starts as soon as the gates pass, even at night.
#   END="2026-11-21 09:30 UTC" nohup bash ~/qurl/s4/day1.sh > ~/qurl/s4/day1.log 2>&1 < /dev/null &
# END is the rental's end; the queue keeps its own reserve before it (BUFFER_H, 6 h). Logs: ~/qurl/s4/setup.log,
# gates.log, queue.log; results/gates/queue_dry_run.txt holds the queue's plan from the measured hours.
# It stops before the queue when:
#   - the setup fails (setup.log; a check_node.py failure means a bad node);
#   - the gates do not pass (no results/gates/PASSED; results/gates/summary.txt says why);
#   - G5 projects more than 45 h per full 7B run: the README's contingency (responses <= 4,096 tokens for every run,
#     the two R3 mechanism runs repeated) is then decided by hand, before any queued run.
# Rerunning is safe: the setup and the gates skip their finished steps, and the queue resumes.
set -uo pipefail
S4="$HOME/qurl/s4"
log() { echo "[$(date -Is)] $*"; }
[ -n "${END:-}" ] || { log "set END to the rental's end, e.g. END=\"2026-11-21 09:30 UTC\""; exit 1; }
date -d "$END" > /dev/null 2>&1 || { log "cannot parse END=$END"; exit 1; }
log "rental end: $(date -u -d "$END" '+%Y-%m-%d %H:%M UTC')"

log "setup_s4.sh (log: $S4/setup.log)"
bash "$S4/setup_s4.sh" > "$S4/setup.log" 2>&1 || { log "the setup failed: see $S4/setup.log"; exit 1; }
log "setup done"

log "gates.sh (log: $S4/gates.log)"
bash "$S4/gates.sh" > "$S4/gates.log" 2>&1
grep -E "PASS|FAIL|WARNING" "$S4/results/gates/summary.txt" 2>/dev/null | tail -n 20
[ -e "$S4/results/gates/PASSED" ] || { log "the gates did not pass: see $S4/results/gates/summary.txt"; exit 1; }
HOURS_7B=
[ -s "$S4/results/gates/hours.txt" ] && . "$S4/results/gates/hours.txt"
log "G5: $(tr '\n' ' ' < "$S4/results/gates/hours.txt" 2>/dev/null)"
if [ -z "$HOURS_7B" ] || ! awk -v h="$HOURS_7B" 'BEGIN { exit !(h <= 45) }'; then
  log "G5 projects '${HOURS_7B:-nothing}' h per full 7B run (over 45, or none): the README's contingency decides;" \
      "the queue is not started"
  exit 1
fi

FORCE=1 DRY_RUN=1 END="$END" bash "$S4/queue.sh" > "$S4/results/gates/queue_dry_run.txt" 2>&1
log "the queue's plan: $S4/results/gates/queue_dry_run.txt"
grep -E "DRY RUN|skip" "$S4/results/gates/queue_dry_run.txt"
log "queue.sh (log: $S4/queue.log)"
END="$END" bash "$S4/queue.sh" >> "$S4/queue.log" 2>&1
log "the queue has finished (exit $?): bash ~/qurl/s4/queue.sh --list"
