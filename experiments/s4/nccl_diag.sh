#!/usr/bin/env bash
# Diagnosis of an NCCL failure in check_node.py (on 2026-11-16: GPU 6's worker died with SIGSEGV in the NCCL test).
#   bash ~/qurl/s4/nccl_diag.sh 2>&1 | tee ~/qurl/s4/results/node/nccl_diag.txt          (a few minutes; needs the GPUs)
# 1. The system facts NCCL depends on: driver, Fabric Manager, fabric state, topology, NVLink, /dev/shm, memlock,
#    ptrace scope, IOMMU, kernel errors; the bootstrap's own NVLink/fabric/peer results; check_node.py's messages.
# 2. nccl_test.py (all-reduce, all-gather, reduce-scatter) on all GPUs: by default, and with one NCCL feature off at a
#    time (NVLink SHARP, cuMem, P2P, shared memory); then without GPU 6, and GPU 6 with GPU 7 alone.
# Each test's full NCCL_DEBUG=INFO log is in results/node/nccl_diag/.
set -uo pipefail
S4="$HOME/qurl/s4"
TR="$HOME/venvs/verl/bin/torchrun"
D="$S4/results/node/nccl_diag"
mkdir -p "$D"
pgrep -f "day1.sh|setup_s4.sh|gates.sh|queue.sh" > /dev/null && echo "WARNING: an S4 job is still running"

echo "== system"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv,noheader
echo "fabric manager: $(systemctl is-active nvidia-fabricmanager 2>&1)"
nvidia-smi -q | grep -i -A 3 "^ *fabric$" | grep -i -E "state|status" | sort | uniq -c
nvidia-smi topo -m | head -n 10
echo "inactive NVLink links: $(nvidia-smi nvlink -s 2>/dev/null | grep -c -i inactive)"
echo "/dev/shm: $(df -h /dev/shm | tail -1)"
echo "ulimit -l: $(ulimit -l); ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
echo "kernel command line IOMMU: $(grep -o -E '(intel_|amd_)?iommu=[^ ]*' /proc/cmdline | tr '\n' ' ')"
dmesg 2>/dev/null | grep -i -E "xid|nvrm.*(error|fail)|segfault" | tail -n 10 || echo "dmesg: not readable without sudo"
echo "== the bootstrap's GPU checks"
grep -i -E "nvlink|fabric|peer|xid|shm|cuda" "$S4/results/node/bootstrap.txt" | head -n 15
echo "== check_node.py's messages in setup.log"
grep -i -E "nccl|sigsegv|segfault|error|traceback|signal" "$S4/setup.log" | tail -n 20

echo "== NCCL tests (3 min limit each)"
run() {   # run <label> <number of GPUs> [VAR=value ...]
  local label=$1 np=$2
  shift 2
  local log="$D/$(echo "$label" | tr -c 'A-Za-z0-9\n' '_').log"
  env NCCL_DEBUG=INFO "$@" timeout 180 "$TR" --nproc_per_node="$np" "$S4/nccl_test.py" > "$log" 2>&1
  local rc=$?
  echo "$label: exit $rc; ranks done $(grep -c 'all-reduce correct True' "$log") of $np;" \
       "$(grep -m1 -i -E 'sigsegv|segfault|signal [0-9]|exitcode|nccl warn|error:' "$log" | cut -c1-160)"
}
run "default" 8
run "NVLink SHARP off" 8 NCCL_NVLS_ENABLE=0
run "cuMem off" 8 NCCL_CUMEM_ENABLE=0
run "P2P off" 8 NCCL_P2P_DISABLE=1
run "shared memory off" 8 NCCL_SHM_DISABLE=1
run "7 GPUs without GPU 6" 7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
run "GPU 6 with GPU 7" 2 CUDA_VISIBLE_DEVICES=6,7
echo "logs: $D"
