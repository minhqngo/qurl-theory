#!/usr/bin/env bash
# Second NCCL diagnosis (2026-11-16). In nccl_diag.sh every test crashed (SIGSEGV) within seconds, whatever the
# transport (NVLink SHARP, cuMem, P2P or shared memory off) and with GPU 6 left out, while the fabric, NVLink and CUDA
# peer access are healthy (bootstrap): the crash is in NCCL's start-up on this system, not in a GPU.
#   bash ~/qurl/s4/nccl_diag2.sh 2>&1 | tee ~/qurl/s4/results/node/nccl_diag2.txt          (about 2-3 min)
# 1. The last NCCL lines before the crash, and the libraries NCCL may load (InfiniBand verbs, net plugins).
# 2. nccl_test.py on GPUs 0-1 with one start-up feature off at a time (InfiniBand, external net plugins, multi-node
#    NVLink detection, the RAS service), then all four off; then all 8 GPUs with the first setting that worked.
set -uo pipefail
S4="$HOME/qurl/s4"
TR="$HOME/venvs/verl/bin/torchrun"
D="$S4/results/node/nccl_diag"
mkdir -p "$D"

echo "== the last NCCL lines before the crash (nccl_diag.sh, GPUs 6 and 7)"
grep -h -E "NCCL (INFO|WARN)|Fatal Python|File \"" "$D/GPU_6_with_GPU_7.log" 2>/dev/null | tail -n 30 | cut -c1-220
echo "== libraries NCCL may load"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(empty)}"
ldconfig -p | grep -i -E "nccl|libibverbs|libmlx5|librdmacm" | head -n 10
echo "InfiniBand devices: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"

run() {   # run <label> <number of GPUs> [VAR=value ...]: 0 if every rank finished
  local label=$1 np=$2
  shift 2
  local log="$D/diag2_$(echo "$label" | tr -c 'A-Za-z0-9\n' '_').log"
  env NCCL_DEBUG=INFO PYTHONFAULTHANDLER=1 "$@" timeout 120 "$TR" --nproc_per_node="$np" "$S4/nccl_test.py" \
    > "$log" 2>&1
  local rc=$? n
  n=$(grep -c 'all-reduce correct True' "$log")
  echo "$label: exit $rc; ranks done $n of $np; last NCCL line: $(grep -h 'NCCL INFO' "$log" | tail -n 1 | cut -c1-150)"
  [ "$rc" = 0 ] && [ "$n" = "$np" ]
}

echo "== 2-GPU tests (GPUs 0 and 1; 2 min limit each)"
run "default" 2 CUDA_VISIBLE_DEVICES=0,1
grep -h -A 12 "Fatal Python error" "$D/diag2_default.log" 2>/dev/null | head -n 14 | cut -c1-160
first=""
for s in "InfiniBand off|NCCL_IB_DISABLE=1" \
         "no net plugin|NCCL_NET_PLUGIN=none" \
         "multi-node NVLink off|NCCL_MNNVL_ENABLE=0" \
         "RAS off|NCCL_RAS_ENABLE=0" \
         "all four off|NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_MNNVL_ENABLE=0 NCCL_RAS_ENABLE=0"; do
  label=${s%%|*}
  vars=${s#*|}
  # shellcheck disable=SC2086  # vars holds several VAR=value words on purpose
  if run "$label" 2 CUDA_VISIBLE_DEVICES=0,1 $vars && [ -z "$first" ]; then first=$s; fi
done

echo "== 8 GPUs"
if [ -n "$first" ]; then
  # shellcheck disable=SC2086
  run "8 GPUs, ${first%%|*}" 8 ${first#*|} && echo "FIX FOUND: ${first#*|}"
else
  echo "no setting let 2 GPUs finish: see the logs in $D (diag2_*.log)"
fi
