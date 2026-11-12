# S4 hang watchdog, sourced by queue.sh. A hung run (an NCCL collective that never returns, a stuck Ray actor) does
# not fail: it would hold all 8 GPUs until someone looked. GPU utilization cannot tell, since a hung NCCL kernel
# spins at 100%, so the signal is output: a run that has written no file under results/<tag> or its checkpoint
# directory for LIMIT minutes is stopped, and the queue's retry resumes it from its last checkpoint (README "Failures").
#   LIMIT = max(WATCH_MIN, 3 x the longest step time so far in its metrics); WATCH_MIN=60 by default, 0 disables.
# Every quiet phase writes more often than that: a training step prints its metrics (late 7B steps are expected at
# 10-20 min), validation and evaluation write vLLM's progress bars, checkpoints and the merge write files.
# A stop is recorded in results/<tag>/WATCHDOG. It kills the run's whole session (run.sh, the trainer, the vLLM
# replicas), then Ray (ray stop --force), then any process of this venv still on a GPU.
# Needs: RESULTS, S4, log(). Checked every WATCH_POLL seconds (default 300).
WATCH_MIN=${WATCH_MIN:-60}
WATCH_POLL=${WATCH_POLL:-300}

watch_limit() {   # watch_limit <tag>: minutes of silence that count as a hang
  cat "$RESULTS/$1"/metrics*.jsonl 2>/dev/null | grep -o '"timing_s/step": *[0-9.e+-]*' \
    | awk -F: -v m="$WATCH_MIN" '{s = $2 / 60 * 3; if (s > m) m = s} END {printf "%d\n", m + 0.999}'
}

watched() {   # watched <tag> <command...>: run it in its own session, stop it if it hangs; returns its exit status
  local tag=$1 pid limit waited p venv="$HOME/venvs/verl"
  shift
  setsid "$@" < /dev/null &
  pid=$!          # setsid in a background child (not a group leader) execs in place: pid = session and group id
  while kill -0 "$pid" 2>/dev/null; do
    for ((waited = 0; waited < WATCH_POLL; waited += 10)); do kill -0 "$pid" 2>/dev/null || break 2; sleep 10; done
    [ "$WATCH_MIN" = 0 ] && continue
    limit=$(watch_limit "$tag")
    [ -n "$(find "$RESULTS/$tag" "${CKPT_ROOT:-$S4/ckpt}/$tag" -mmin "-$limit" -print -quit 2>/dev/null)" ] && continue
    log "WATCHDOG: $tag has written nothing for $limit min: stopping it"
    echo "$(date -Is) no output for $limit min: stopped by the queue's watchdog" >> "$RESULTS/$tag/WATCHDOG"
    pkill -TERM -s "$pid" 2>/dev/null
    for ((waited = 0; waited < 60; waited += 5)); do pgrep -s "$pid" > /dev/null || break; sleep 5; done
    pkill -KILL -s "$pid" 2>/dev/null
    [ -x "$venv/bin/ray" ] && "$venv/bin/ray" stop --force > /dev/null 2>&1
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      ps -o args= -p "$p" 2>/dev/null | grep -q "$venv" && kill -KILL "$p" 2>/dev/null
    done
    break
  done
  wait "$pid"
}
