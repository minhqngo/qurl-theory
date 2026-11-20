#!/usr/bin/env bash
# Revision experiments (after the blind reviews of the P1 paper, November 2026): every job in order, one at a time, on
# the single H100.
#   nohup bash ~/qurl/rev/queue.sh > ~/qurl/rev/queue.log 2>&1 < /dev/null &    # start, or resume after a stop
#   bash ~/qurl/rev/queue.sh --list         # plan and status of every job
#   bash ~/qurl/rev/queue.sh --progress     # progress of the RL runs (S3's progress.sh)
# Order: quick checks first (X1, X7, X2), then the long RL runs that answer the reviewers' two main concerns
# (X3: the INT8 gap; X5: early stalls), then X4 and X6. Every job is resumable and skipped once done, so the queue
# can be stopped and started again at any time.
# Before the first job, it requires the GPU to be free and the frozen analyses to match registration.sha256: the
# analyses were fixed before any of these data existed.
# A failed job is logged and the queue moves on (STOP_ON_FAIL=1 stops instead). Failed jobs are listed at the end;
# start the queue again to retry them.
set -uo pipefail
REV="$HOME/qurl/rev"
S3="$HOME/qurl/s3"

plan() {   # "<job> <command ...>" in run order; <job> is the results path under results/
  echo "x1_base bash $REV/x1_base.sh"
  echo "x7_s1_vllm024 bash $REV/x7_run.sh"
  echo "x2_e2dumps bash $REV/x2_e2dumps.sh"
  for s in 1 2 3; do
    for arm in R1 R3; do echo "x3/rev_x3_${arm}_s$s bash $REV/x3_run.sh $arm $s"; done
  done
  for s in $(seq 6 35); do
    for arm in R4 R5 R3; do
      if [ "$arm" = R3 ] && [ "$s" -gt 15 ]; then continue; fi
      echo "x5/rev_x5_${arm}_s$s bash $REV/x5_run.sh $arm $s"
    done
  done
  echo "x4_rounding bash $REV/x4_run.sh"
  echo "x6_fp8null bash $REV/x6_run.sh"
}
status() {
  if [ -f "$REV/results/$1/done" ]; then
    if [ -f "$REV/results/$1/SKIPPED" ]; then echo skipped; else echo done; fi
  elif [ -e "$REV/results/$1" ]; then echo started
  else echo pending; fi
}

case "${1:-}" in
  --list)
    plan | while read -r job _; do printf '%-26s %s\n' "$job" "$(status "$job")"; done
    echo "done: $(plan | while read -r job _; do status "$job"; done | grep -c -E 'done|skipped') of $(plan | wc -l)"
    exit 0 ;;
  --progress)
    tags=$(plan | awk '{print $1}' | grep -E '^x[35]/' | sed 's#.*/##' | while read -r t; do
      [ -d "$S3/results/$t" ] && echo "$t"; done)
    [ -n "$tags" ] && bash "$S3/progress.sh" $tags
    exit 0 ;;
  "") ;;
  *) echo "usage: queue.sh [--list|--progress]" >&2; exit 2 ;;
esac

log() { echo "[$(date -Is)] $*"; }
log "revision queue: $(plan | wc -l) jobs"
md5sum "$REV"/*.py "$REV"/*.sh "$S3"/arm.sh "$S3"/pilot.sh "$S3"/qurl_verl/*.py 2>/dev/null | sed 's/^/  code: /'

if ! (cd "$REV" && sha256sum --quiet -c registration.sha256); then
  log "the frozen analyses do not match registration.sha256; stopping"
  exit 1
fi
log "frozen analyses match registration.sha256"

while pgrep -f '[p]ilot.sh|[m]ain_ppo' > /dev/null || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  [ "${waited:-0}" = 1 ] || log "waiting for the GPU (another job is running)"
  waited=1
  sleep 60
done
log "GPU free"

failed=()
while read -r job cmd; do
  st=$(status "$job")
  if [ "$st" = done ] || [ "$st" = skipped ]; then log "skip $job ($st)"; continue; fi
  log "start $job"
  bash -c "$cmd" < /dev/null
  rc=$?
  log "end $job: exit $rc"
  if [ $rc -ne 0 ]; then
    failed+=("$job")
    [ "${STOP_ON_FAIL:-0}" = 1 ] && { log "stopping (STOP_ON_FAIL=1)"; exit 1; }
  fi
done < <(plan)
if [ ${#failed[@]} -gt 0 ]; then log "revision queue finished with ${#failed[@]} failed jobs: ${failed[*]}"
else log "revision queue finished: every job done"; fi
bash "$0" --list
