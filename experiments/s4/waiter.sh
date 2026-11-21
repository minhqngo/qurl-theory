#!/usr/bin/env bash
# Start a waiter that reruns the S4 queue as soon as the running queue exits. The running queue keeps the plan and
# the END it started with; the rerun reads queue.sh afresh (so a plan changed by cp + mv takes effect) and skips
# finished runs.
#   END='2026-11-23 01:00 UTC' BUFFER_H=10 bash ~/qurl/s4/waiter.sh
# It refuses to start a waiter if no queue is running, if more than one is, or if a waiter for that queue already
# exists, so that two queues never run the same run on the same GPUs. A waiter left over from an earlier extension may
# still be the running queue's parent: it no longer waits for anything, and it ends with the queue.
# Overrides (tests): S4_DIR (default ~/qurl/s4), WAIT_POLL (seconds between checks, default 60).
set -uo pipefail
: "${END:?set END, e.g. END='2026-11-23 01:00 UTC'}"
: "${BUFFER_H:?set BUFFER_H, the reserve in hours}"
date -d "$END" > /dev/null 2>&1 || { echo "cannot parse END=$END"; exit 1; }
S4="${S4_DIR:-$HOME/qurl/s4}"
POLL="${WAIT_POLL:-60}"
[ -f "$S4/queue.sh" ] || { echo "no $S4/queue.sh"; exit 1; }

# The running queue: the processes whose command line ends in /qurl/s4/queue.sh, without arguments, that are not
# waiters (a waiter's command line also ends in queue.sh). A running queue is several such processes: bash runs the
# `plan | while read` loop, and other parts, in subshells that keep the queue's command line. The queue itself is the
# root of that tree, the one whose parent is not one of them.
mapfile -t QS < <(pgrep -af '/qurl/s4/queue\.sh$' | grep -v 'while kill -0' | awk '{print $1}')
ROOTS=()
for p in "${QS[@]}"; do
  pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -n "$pp" ] || continue   # exited meanwhile
  inside=0
  for q in "${QS[@]}"; do [ "$q" = "$pp" ] && inside=1; done
  [ "$inside" = 0 ] && ROOTS+=("$p")
done
if [ "${#ROOTS[@]}" -ne 1 ]; then
  echo "NOT started: expected exactly one running queue, found ${#ROOTS[@]} (of ${#QS[@]} queue processes)"
  [ "${#QS[@]}" -gt 0 ] && ps -o pid,ppid,etime,args -p "$(IFS=,; echo "${QS[*]}")"
  exit 1
fi
Q=${ROOTS[0]}
if pgrep -f "while kill -0 $Q " > /dev/null; then
  echo "NOT started: a waiter for queue $Q already exists:"
  pgrep -af "while kill -0 $Q "
  exit 1
fi
nohup bash -c "while kill -0 $Q 2>/dev/null; do sleep $POLL; done; END='$END' BUFFER_H=$BUFFER_H bash $S4/queue.sh" \
  >> "$S4/queue.log" 2>&1 < /dev/null &
W=$!
disown
echo "waiter $W started: when queue $Q exits, it reruns queue.sh with END='$END' BUFFER_H=$BUFFER_H"
ps -o pid,ppid,etime,args -p "$(IFS=,; echo "${QS[*]}")" | sed 's/^/  /'
