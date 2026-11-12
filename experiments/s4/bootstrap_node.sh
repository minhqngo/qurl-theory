#!/usr/bin/env bash
# S4 day 0: make a fresh Ubuntu 24.04 GPU node ready for setup_s4.sh. The rental's image (ubuntu-24.04-x86_64-
# standard-uefi) has no NVIDIA driver, so there is no nvidia-smi yet. Run this as the user who will run the experiments
# (it uses sudo for the system steps), inside tmux, after copying the code to ~/qurl/s4:
#   bash ~/qurl/s4/bootstrap_node.sh 2>&1 | tee -a ~/bootstrap.log
# On a fresh image it usually ends with "REBOOT NEEDED" (exit 3): sudo reboot, reconnect, run the same command again.
# Every step checks its own state first, so running it again is always safe. About 20-40 min including the reboot.
#  1 inventory: NVIDIA GPUs and NVSwitches on the PCI bus (visible before any driver), CPU, RAM, disks, Secure Boot
#  2 storage: the runs need about 1 TB beside ~/qurl (README). DATA_DISK=/dev/nvmeXn1 formats a blank disk (ext4) and
#    mounts it at /data; DATA_DIR=/path uses a filesystem that is already mounted. ~/qurl, ~/venvs, ~/data and ~/.cache
#    then move there and stay reachable through symlinks, so no path in the S4 scripts changes
#  3 packages (compiler, kernel headers, git, curl, tmux); automatic upgrades off for the rental: an unattended update
#    of the driver's libraries under a running job breaks it (driver/library version mismatch)
#  4 the NVIDIA driver from NVIDIA's apt repository, pinned to branch 580 (the oldest for CUDA 13.0; long-term support):
#    headless, open kernel modules built by DKMS; it brings nvidia-smi (in libnvidia-compute-580) and the persistence
#    daemon. On an NVSwitch board (HGX), Fabric Manager at exactly the driver's version, enabled and started: without
#    it CUDA fails with error 802 (system not ready). A driver >= 580.65.06 already there is kept; an older one is
#    replaced only with UPGRADE_DRIVER=1. Secure Boot must be off (DKMS modules are unsigned): it stops if it is on
#  5 the CUDA 13.0 toolkit at /usr/local/cuda-13.0 (nvcc for setup_s4.sh's flash-attn build)
#  6 GPU state: persistence mode on, MIG off, driver packages held; qurl-gpu-monitor.service logs every GPU every 60 s
#    to /var/log/qurl-gpu-monitor.csv for the whole rental (utilization, memory, power, temperature, SM clock)
#  7 checks without PyTorch: GPU count and model, driver, ECC errors and row remapping, NVLink links, PCIe width,
#    NVLink fabric state, CUDA initialization and peer access between every GPU pair, kernel Xid errors, disk, RAM,
#    /dev/shm, access to Hugging Face, GitHub, PyPI and NVIDIA. setup_s4.sh's check_node.py then loads the GPUs
# Writes ~/qurl/s4/results/node/bootstrap.txt. Exit 0 ready, 1 ready with warnings, 2 failed, 3 reboot needed.
#   NGPU=8  DRIVER_BRANCH=580  UPGRADE_DRIVER=1  CHECK_ONLY=1 (checks only)  DRY_RUN=1 (prints system changes, runs none)
#   Laptop tests only: S4=<dir>, PRETEND_NO_DRIVER=1, TEST_PCI_FILE=<lspci -n output>
set -uo pipefail
S4=${S4:-$HOME/qurl/s4}
OUT="$S4/results/node"
REPORT="$OUT/bootstrap.txt"
NGPU=${NGPU:-8}
BRANCH=${DRIVER_BRANCH:-580}
MIN_DRIVER=580.65.06          # CUDA 13.0's minimum driver (torch 2.11+cu130)
CUDA_DIR=/usr/local/cuda-13.0
MONITOR=/var/log/qurl-gpu-monitor.csv
NEED_GB=1000
DRY=${DRY_RUN:-0}
DET=$(mktemp)
lines=(); warn=0; fail=0; reboot=0

log() { echo "[$(date -Is)] $*"; }
verdict() {   # verdict PASS|INFO|WARN|FAIL text: printed now, and listed first in the report
  local v=$1; shift
  echo "$v $*"; lines+=("$v $*")
  case $v in WARN) warn=1 ;; FAIL) fail=1 ;; esac
}
detail() { { echo "== $1"; shift; "$@" 2>&1; echo; } >> "$DET"; }
if [ "$(id -u)" = 0 ]; then SUDO=(); else SUDO=(sudo); fi
root() { if [ "$DRY" = 1 ]; then echo "DRY_RUN (root): $*"; else "${SUDO[@]}" "$@"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "DRY_RUN: $*"; else "$@"; fi; }
aptq() { root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 apt-get -y -q -o DPkg::Lock::Timeout=900 "$@"; }
ver_ge() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # ver_ge A B: A >= B
q() { local o; o=$(nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null) && echo "$o"; }
gpu_ok() { [ "${PRETEND_NO_DRIVER:-0}" != 1 ] && nvidia-smi -L >/dev/null 2>&1; }

finish() {
  local rc=0
  if [ "$fail" = 1 ]; then rc=2; elif [ "$reboot" = 1 ]; then rc=3; elif [ "$warn" = 1 ]; then rc=1; fi
  mkdir -p "$OUT"
  { echo "# bootstrap_node.sh $(date -Is) on $(hostname): exit $rc$([ "$DRY" = 1 ] && echo ' (DRY_RUN)')"
    printf '%s\n' "${lines[@]}"; echo; cat "$DET"; } > "$REPORT"
  rm -f "$DET"
  echo
  echo "== summary (every line in $REPORT)"
  printf '%s\n' "${lines[@]}" | grep -E '^(WARN|FAIL)' || echo "no warnings or failures"
  case $rc in
    3) echo "REBOOT NEEDED: sudo reboot; reconnect after a few minutes (bare metal can take 10); run the same command again" ;;
    2) echo "FAILED: see the FAIL lines; fix them and run this again" ;;
    *) echo "node ready$([ "$rc" = 1 ] && echo ', with warnings'): next, bash ~/qurl/s4/setup_s4.sh 2>&1 | tee ~/qurl/s4/setup.log" ;;
  esac
  exit "$rc"
}

# --- storage ---
blank_disks() {   # whole disks with no partitions or holders, no filesystem signature, not mounted
  local d out=""
  for d in $(lsblk -dpno NAME,TYPE 2>/dev/null | awk '$2 == "disk" {print $1}'); do
    [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ] || continue
    [ -z "$(lsblk -no MOUNTPOINTS,FSTYPE "$d" | tr -d '[:space:]')" ] || continue
    out+="$d ($(lsblk -dno SIZE "$d" | tr -d ' ')) "
  done
  echo "${out:-none}"
}
prepare_disk() {   # format a blank disk as ext4 and mount it at /data (fstab, nofail: a missing disk never blocks boot)
  local d=$1 uuid
  if findmnt -rno SOURCE /data >/dev/null 2>&1; then
    [ "$(readlink -f "$(findmnt -rno SOURCE /data)")" = "$d" ] && { verdict PASS "storage: $d already mounted at /data"; return 0; }
    verdict FAIL "/data is already mounted from $(findmnt -rno SOURCE /data), not $d"; return 1
  fi
  [ -b "$d" ] && [ "$(lsblk -dno TYPE "$d")" = disk ] || { verdict FAIL "DATA_DISK=$d is not a whole disk"; return 1; }
  case " $(blank_disks) " in
    *" $d ("*) ;;
    *) verdict FAIL "DATA_DISK=$d is not blank (partitions, a filesystem, a RAID/LVM member or a mount): not formatted"; return 1 ;;
  esac
  log "formatting $d as ext4 (label qurl-data) and mounting it at /data"
  root mkfs.ext4 -q -F -L qurl-data -E lazy_itable_init=1,lazy_journal_init=1,nodiscard "$d" || { verdict FAIL "mkfs.ext4 $d failed"; return 1; }
  uuid=$(root blkid -s UUID -o value "$d")
  root mkdir -p /data
  grep -q "UUID=$uuid " /etc/fstab || echo "UUID=$uuid /data ext4 defaults,noatime,nofail,x-systemd.device-timeout=30s 0 2" | root tee -a /etc/fstab >/dev/null
  root systemctl daemon-reload
  root mount /data || { verdict FAIL "mounting /data failed"; return 1; }
  root chown "$(id -u):$(id -g)" /data
  verdict PASS "storage: $d formatted (ext4) and mounted at /data (in /etc/fstab)"
}
move_home_dirs() {   # the big directories onto $1, left reachable from $HOME through symlinks
  local base=$1 n src dst moved=0
  [ -d "$base" ] && [ -w "$base" ] || [ "$DRY" = 1 ] || { verdict FAIL "DATA_DIR=$base is not a writable directory"; return 1; }
  for n in qurl venvs data .cache; do
    src="$HOME/$n"; dst="$base/$n"
    [ -L "$src" ] && continue
    if [ -e "$dst" ]; then verdict WARN "$dst already exists and $src is not a link to it: left as is"; continue; fi
    if [ -e "$src" ]; then run mv "$src" "$dst" || { verdict FAIL "could not move $src to $dst"; return 1; }; moved=1
    else run mkdir -p "$dst"; fi
    run ln -s "$dst" "$src"
  done
  verdict PASS "storage: ~/qurl, ~/venvs, ~/data and ~/.cache are on $base (symlinks from ~)"
  [ "$moved" = 1 ] && echo "(~/qurl moved: a shell inside it should cd ~/qurl/s4 again)"
  return 0
}

# --- driver ---
ensure_repo() {   # NVIDIA's apt repository for this Ubuntu release: its key package, then one apt-get update
  [ -n "${REPO_READY:-}" ] && return 0
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    local t; t=$(mktemp -d)
    curl -fsSL -o "$t/cuda-keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
      || { verdict FAIL "could not download NVIDIA's repository key for $REPO"; rm -rf "$t"; return 1; }
    root dpkg -i "$t/cuda-keyring.deb" || { verdict FAIL "could not install NVIDIA's repository key"; rm -rf "$t"; return 1; }
    rm -rf "$t"
  fi
  aptq update >/dev/null || verdict WARN "apt-get update reported errors"
  REPO_READY=1
}
purge_driver() {   # an older driver (packages and/or NVIDIA's .run installer); container tooling is left alone
  local pk
  pk=$(dpkg-query -W -f='${Package} ${db:Status-Abbrev}\n' 2>/dev/null | awk '$2 ~ /^[hi]i/ {print $1}' \
       | grep -E '^(nvidia-|libnvidia-|cuda-drivers|xserver-xorg-video-nvidia|linux-modules-nvidia|linux-objects-nvidia|linux-signatures-nvidia)' \
       | grep -vE 'container|docker')
  log "removing the old driver: ${pk:-no packages}"
  root systemctl stop nvidia-fabricmanager nvidia-persistenced >/dev/null 2>&1
  if [ -n "$pk" ]; then
    root apt-mark unhold $pk >/dev/null
    aptq purge --allow-change-held-packages $pk || { verdict FAIL "could not remove the old driver's packages"; return 1; }
  fi
  command -v nvidia-uninstall >/dev/null && root nvidia-uninstall --silent
  aptq autoremove >/dev/null
  return 0
}
install_driver() {
  if [ "$sb" = on ]; then
    verdict FAIL "Secure Boot is on: the driver's kernel modules (built here by DKMS, unsigned) would not load. Turn Secure Boot off in the provider's console or BIOS (or ask the provider), then run this again"
    return 1
  fi
  [ "$ngpu_pci" -gt 0 ] || { verdict FAIL "no NVIDIA GPU on the PCI bus"; return 1; }
  ensure_repo || return 1
  aptq install --no-install-recommends "nvidia-driver-pinning-$BRANCH" || { verdict FAIL "nvidia-driver-pinning-$BRANCH did not install"; return 1; }
  local pk=("nvidia-headless-$BRANCH-open" nvidia-modprobe)
  [ "$nsw" -gt 0 ] && pk+=(nvidia-fabricmanager)
  aptq install --no-install-recommends "${pk[@]}" || { verdict FAIL "the driver packages did not install (apt's messages above)"; return 1; }
  if [ ! -e /etc/modprobe.d/qurl-blacklist-nouveau.conf ]; then   # nouveau must not take the GPUs at boot
    printf 'blacklist nouveau\noptions nouveau modeset=0\n' | root tee /etc/modprobe.d/qurl-blacklist-nouveau.conf >/dev/null
    root update-initramfs -u
  fi
  [ "$DRY" = 1 ] && return 0
  dkms status 2>/dev/null | grep -Eq "^nvidia.*$(uname -r).*installed" \
    || { verdict FAIL "DKMS did not build the nvidia module for kernel $(uname -r) (see /var/lib/dkms/nvidia/*/build/make.log)"; return 1; }
  local dv; dv=$(dpkg-query -W -f='${Version}' "libnvidia-compute-$BRANCH" 2>/dev/null)
  verdict PASS "driver ${dv%%-*} installed (NVIDIA's repository, branch $BRANCH pinned; headless, open kernel modules)"
  root systemctl enable nvidia-persistenced >/dev/null 2>&1
  [ "$nsw" -gt 0 ] && root systemctl enable nvidia-fabricmanager >/dev/null 2>&1
  return 0
}
load_driver() {   # without a reboot when nouveau lets go of the GPUs; otherwise the caller asks for a reboot
  [ "$DRY" = 1 ] && { echo "DRY_RUN: would try to load the module without a reboot"; return 1; }
  if lsmod | grep -q '^nouveau '; then root modprobe -r nouveau 2>/dev/null || return 1; fi
  root modprobe nvidia 2>/dev/null || return 1
  root modprobe nvidia_uvm 2>/dev/null
  gpu_ok || return 1
  root systemctl start nvidia-persistenced >/dev/null 2>&1
  verdict INFO "driver loaded without a reboot"
}
fabric_states() {   # each GPU's NVLink fabric "State/Status" from nvidia-smi -q
  nvidia-smi -q 2>/dev/null | awk '/^GPU [0-9A-Fa-f]+:/ {f = 0} /^ +Fabric *$/ {f = 1; next}
    f && /^ +State +:/ {sub(/^[^:]*: */, ""); st = $0}
    f && /^ +Status +:/ {sub(/^[^:]*: */, ""); print st "/" $0; f = 0}'
}
fabric_done() { local s; s=$(fabric_states); [ -n "$s" ] && ! grep -qv '^Completed/Success$' <<< "$s"; }
setup_fm() {   # NVSwitch boards: Fabric Manager at exactly the loaded driver's version, enabled and running
  [ "$nsw" -gt 0 ] || return 0
  local dv fv cand
  dv=$(q driver_version | head -1)
  fv=$(dpkg-query -W -f='${Version}' nvidia-fabricmanager 2>/dev/null); fv=${fv%%-*}
  if [ "$fv" != "$dv" ]; then
    ensure_repo || return 1
    cand=$(apt-cache madison nvidia-fabricmanager 2>/dev/null | awk -v v="$dv" 'index($3, v "-") == 1 {print $3; exit}')
    [ -n "$cand" ] || { verdict FAIL "NVIDIA's repository has no Fabric Manager $dv for driver $dv: install the matching one by hand"; return 1; }
    aptq install --no-install-recommends --allow-downgrades --allow-change-held-packages "nvidia-fabricmanager=$cand" \
      || { verdict FAIL "Fabric Manager $cand did not install"; return 1; }
  fi
  root systemctl enable nvidia-fabricmanager >/dev/null 2>&1
  systemctl is-active -q nvidia-fabricmanager || root systemctl restart nvidia-fabricmanager
  log "waiting for the NVLink fabric (Fabric Manager trains it; up to 3 min)"
  for _ in $(seq 36); do fabric_done && break; [ "$DRY" = 1 ] && break; sleep 5; done
  return 0   # the checks report the fabric's state
}
hold_driver() {   # no driver change during the rental
  local pk
  pk=$(dpkg-query -W -f='${Package} ${db:Status-Abbrev}\n' 2>/dev/null | awk '$2 ~ /^ii/ {print $1}' \
       | grep -E '^(nvidia-|libnvidia-)' | grep -vE 'container|docker')
  [ -n "$pk" ] && root apt-mark hold $pk >/dev/null && verdict INFO "held $(wc -w <<< "$pk") driver packages (apt-mark showhold)"
  return 0
}
install_monitor() {   # every GPU's state every 60 s, for the whole rental, independent of the S4 scripts
  local unit=/etc/systemd/system/qurl-gpu-monitor.service f
  [ -e "$unit" ] && return 0
  f=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.active
  nvidia-smi --query-gpu="$f" --format=csv,noheader >/dev/null 2>&1 || f=${f%,clocks_event_reasons.active}
  echo "$f" | root tee "$MONITOR" >/dev/null
  root tee "$unit" >/dev/null <<EOF
[Unit]
Description=QuRL S4: every GPU's state every 60 s
After=nvidia-persistenced.service
[Service]
ExecStart=/usr/bin/nvidia-smi --query-gpu=$f --format=csv,noheader -l 60
StandardOutput=append:$MONITOR
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
EOF
  root systemctl daemon-reload
  root systemctl enable --now qurl-gpu-monitor.service >/dev/null 2>&1 && verdict INFO "GPU log: $MONITOR, every 60 s"
}

# --- checks (read-only) ---
field_check() {   # field_check FIELD OK_REGEX LEVEL TEXT: LEVEL lists the GPUs whose value does not match
  local vals bad
  vals=$(q "$1")
  [ -n "$vals" ] || { verdict INFO "$4: not reported ($1)"; return; }
  bad=$(awk -v re="$2" '$0 !~ re {printf "GPU%d=%s ", NR - 1, $0}' <<< "$vals")
  if [ -z "$bad" ]; then verdict PASS "$4"; else verdict "$3" "$4: no, $bad"; fi
}
nvlink_check() {
  local s max bad
  s=$(nvidia-smi nvlink -s 2>/dev/null | awk '/^GPU [0-9]+:/ {g = $2; sub(/:$/, "", g); order[++k] = g; n[g] = 0; up[g] = 0}
       /Link [0-9]+:/ {n[g]++; if ($0 ~ /GB\/s/) up[g]++}
       END {for (i = 1; i <= k; i++) print order[i], up[order[i]], n[order[i]]}')
  if [ -z "$s" ] || [ "$(awk '{t += $3} END {print t + 0}' <<< "$s")" = 0 ]; then
    verdict WARN "no NVLink: the GPUs talk over PCIe, so FSDP's collectives are slower (README asks for SXM)"; return
  fi
  max=$(awk '$3 > m {m = $3} END {print m + 0}' <<< "$s")
  bad=$(awk -v m="$max" '$2 < m {printf "GPU%s %d of %d up; ", $1, $2, m}' <<< "$s")
  if [ -z "$bad" ]; then verdict PASS "NVLink: $max of $max links up on every GPU"; else verdict WARN "NVLink links down: $bad"; fi
}
fabric_check() {
  local dv fv s
  dv=$(q driver_version | head -1)
  fv=$(nv-fabricmanager --version 2>/dev/null | grep -o '[0-9][0-9.]*[0-9]' | head -1)
  if systemctl is-active -q nvidia-fabricmanager; then verdict PASS "Fabric Manager running (${fv:-?}, driver $dv)"
  else verdict FAIL "Fabric Manager is not running (journalctl -u nvidia-fabricmanager)"; fi
  [ "$fv" = "$dv" ] || verdict FAIL "Fabric Manager ${fv:-missing} does not match driver $dv"
  s=$(fabric_states | sort | uniq -c | awk '{printf "%s%d x %s", (NR > 1 ? ", " : ""), $1, $2}')
  if fabric_done; then verdict PASS "NVLink fabric: $s"; else verdict FAIL "NVLink fabric not ready: ${s:-no state} (CUDA fails with error 802 until it is)"; fi
}
cuda_probe() {   # the driver API through ctypes: no PyTorch needed
  local out line
  out=$(python3 - <<'EOF'
import ctypes, sys
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print(f"FAIL CUDA: libcuda.so.1 cannot be loaded ({e})"); sys.exit()
def name(r):
    s = ctypes.c_char_p()
    cu.cuGetErrorName(r, ctypes.byref(s))
    return s.value.decode() if s.value else str(r)
r = cu.cuInit(0)
if r:
    hint = ": the NVLink fabric is not ready (Fabric Manager)" if r == 802 else ""
    print(f"FAIL CUDA: cuInit returned {r} ({name(r)}){hint}"); sys.exit()
v, n = ctypes.c_int(), ctypes.c_int()
cu.cuDriverGetVersion(ctypes.byref(v))
cu.cuDeviceGetCount(ctypes.byref(n))
devs = []
for i in range(n.value):
    d = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(d), i)
    devs.append(d.value)
missing = []
for i in devs:
    for j in devs:
        c = ctypes.c_int()
        if i != j and (cu.cuDeviceCanAccessPeer(ctypes.byref(c), i, j) or not c.value):
            missing.append(f"{i}->{j}")
api = f"{v.value // 1000}.{v.value % 1000 // 10}"
print(("PASS" if v.value >= 13000 else "FAIL") + f" CUDA: initialized; driver API {api} (CUDA 13.0 needs 13.0); {n.value} devices")
pairs = len(devs) * (len(devs) - 1)
if pairs:
    print(("PASS" if not missing else "WARN") + f" CUDA peer access: {pairs - len(missing)} of {pairs} GPU pairs"
          + (f" (missing {' '.join(missing[:12])})" if missing else ""))
EOF
)
  while IFS= read -r line; do [ -n "$line" ] && verdict "${line%% *}" "${line#* }"; done <<< "$out"
}
net_check() {
  local u bad=""
  for u in https://huggingface.co https://github.com https://pypi.org https://developer.download.nvidia.com; do
    curl -sS -o /dev/null -m 20 -I "$u" 2>/dev/null || bad+="$u "
  done
  if [ -z "$bad" ]; then verdict PASS "network: Hugging Face, GitHub, PyPI and NVIDIA reachable"; else verdict WARN "network: cannot reach $bad"; fi
}
checks() {
  local n cc drv x
  n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  if [ "$n" = "$NGPU" ]; then verdict PASS "nvidia-smi: $n GPUs"; else verdict WARN "nvidia-smi: $n GPUs, NGPU=$NGPU"; fi
  cc=$(q compute_cap | sort -u | paste -sd' ')
  verdict INFO "GPUs: $(q name | sort | uniq -c | awk '{c = $1; $1 = ""; sub(/^ /, ""); printf "%s%d x %s", (NR > 1 ? "; " : ""), c, $0}'); compute capability $cc; memory $(q memory.total | sort -u | paste -sd' ') MiB"
  [ "$cc" = 9.0 ] || verdict WARN "compute capability $cc: S1-S3's INT8 kernel was CUTLASS on SM90 (H100/H200)"
  drv=$(q driver_version | head -1)
  if ver_ge "$drv" "$MIN_DRIVER"; then verdict PASS "driver $drv (CUDA 13.0 needs >= $MIN_DRIVER)"; else verdict FAIL "driver $drv is older than CUDA 13.0's minimum $MIN_DRIVER"; fi
  field_check persistence_mode '^Enabled$' WARN "persistence mode on"
  field_check mig.mode.current '^(Disabled|.N/A.)$' WARN "MIG off"
  field_check ecc.mode.current '^(Enabled|.N/A.)$' WARN "ECC on"
  field_check ecc.errors.uncorrected.volatile.total '^(0|.N/A.)$' WARN "no uncorrected ECC error since boot"
  field_check remapped_rows.pending '^(No|.N/A.)$' WARN "no row remapping pending (pending needs a GPU reset: reboot)"
  field_check remapped_rows.failure '^(No|.N/A.)$' FAIL "no row-remapping failure (a failure means a bad GPU: ask for another node)"
  field_check clocks_event_reasons.hw_slowdown '^(Not Active|.N/A.)$' WARN "no hardware slowdown"
  x=$(paste -d' ' <(q pcie.link.width.current) <(q pcie.link.width.max) | awk '$1 < $2 {printf "GPU%d x%s of x%s; ", NR - 1, $1, $2}')
  if [ -z "$x" ]; then verdict PASS "PCIe links at full width"; else verdict WARN "PCIe links below full width: $x"; fi
  x=$(q memory.used | awk '$1 > 2048 {printf "GPU%d %s MiB; ", NR - 1, $1}')
  [ -z "$x" ] || verdict WARN "GPU memory already in use: $x"
  x=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  [ -z "$x" ] || verdict WARN "processes already on the GPUs: $(paste -sd';' <<< "$x")"
  nvlink_check
  [ "$nsw" -gt 0 ] && fabric_check
  cuda_probe
  if [ "$mode" = install ]; then x=$(root journalctl -k -b --no-pager 2>/dev/null | grep -c 'NVRM: Xid')
  else x=$(journalctl -k -b --no-pager 2>/dev/null | grep -c 'NVRM: Xid'); fi
  if [ "${x:-0}" = 0 ]; then verdict PASS "no NVIDIA Xid error in this boot's kernel log"; else verdict WARN "$x NVIDIA Xid error(s) in the kernel log (journalctl -k | grep Xid)"; fi
  net_check
  detail "nvidia-smi" nvidia-smi
  detail "nvidia-smi topo -m" nvidia-smi topo -m
  detail "nvidia-smi nvlink -s" nvidia-smi nvlink -s
}

# ===== 0. where we are
[ -r /etc/os-release ] && . /etc/os-release
mode=install
[ "${CHECK_ONLY:-0}" = 1 ] && mode=check
if [ "$DRY" != 1 ] && grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  mode=check; verdict INFO "WSL: the driver belongs to Windows; checks only"
elif [ "$DRY" != 1 ] && { [ -f /.dockerenv ] || [ -f /run/.containerenv ]; }; then
  mode=check; verdict INFO "container: the driver belongs to the host; checks only"
fi
log "bootstrap_node.sh: $mode$([ "$DRY" = 1 ] && echo ', DRY_RUN'); ${PRETTY_NAME:-unknown OS}, kernel $(uname -r)"
if [ "$mode" = install ]; then
  if [ "${ID:-}" != ubuntu ] || [ "$(uname -m)" != x86_64 ]; then
    verdict FAIL "written for Ubuntu 24.04 on x86_64; found ${PRETTY_NAME:-unknown} on $(uname -m)"; finish
  fi
  [ "${VERSION_ID:-}" = 24.04 ] || verdict WARN "written for Ubuntu 24.04; found $PRETTY_NAME (NVIDIA's repository for it is used)"
  if [ ${#SUDO[@]} -gt 0 ] && [ "$DRY" != 1 ]; then
    sudo -v || { verdict FAIL "the system steps need sudo"; finish; }
    ( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) >/dev/null 2>&1 &
  fi
fi
vid=${VERSION_ID:-}
REPO="ubuntu${vid//./}"

# ===== 1. inventory
log "1. inventory"
if [ "$mode" = install ] && ! command -v lspci >/dev/null; then aptq update >/dev/null; aptq install pciutils >/dev/null; fi
if [ -n "${TEST_PCI_FILE:-}" ]; then pci=$(cat "$TEST_PCI_FILE"); else pci=$(lspci -n -d 10de: 2>/dev/null); fi
ngpu_pci=$(awk '$2 ~ /^030[02]:?$/' <<< "$pci" | grep -c .)
nsw=$(awk '$2 ~ /^0680:?$/' <<< "$pci" | grep -c .)
ids=$(awk '$2 ~ /^030[02]:?$/ {print $3}' <<< "$pci" | sort | uniq -c | awk '{printf "%s%d x %s", (NR > 1 ? ", " : ""), $1, $2}')
verdict INFO "PCI: $ngpu_pci NVIDIA GPU(s)${ids:+ ($ids)}, $nsw NVSwitch(es)$([ "$nsw" -gt 0 ] && echo ': an NVSwitch board, so Fabric Manager is needed')"
[ "$mode" = install ] && [ "$ngpu_pci" != "$NGPU" ] && verdict WARN "$ngpu_pci GPUs on the PCI bus, NGPU=$NGPU"
ram=$(free -g | awk '/^Mem:/ {print $2}')
verdict INFO "CPU: $(nproc) cores ($(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)); RAM: $ram GB"
[ "$ram" -ge 512 ] || verdict WARN "RAM $ram GB: the plan assumes at least 512 GB"
detail "PCI (NVIDIA)" lspci -nn -d 10de:
detail "disks" lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
detail "filesystems" df -h -x tmpfs -x devtmpfs -x overlay -x squashfs -x efivarfs
sb=off
if [ -d /sys/firmware/efi ]; then
  if command -v mokutil >/dev/null; then
    mokutil --sb-state 2>/dev/null | grep -qi 'secureboot enabled' && sb=on
  else
    f=$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1)
    [ -n "$f" ] && [ "$(od -An -tu1 "$f" | awk '{print $NF}')" = 1 ] && sb=on
  fi
fi
verdict INFO "boot: $([ -d /sys/firmware/efi ] && echo UEFI || echo legacy BIOS), Secure Boot $sb"

# ===== 2. storage
log "2. storage"
if [ -n "${DATA_DISK:-}" ]; then
  if [ "$mode" != install ]; then verdict WARN "DATA_DISK ignored in $mode mode"
  elif prepare_disk "$(readlink -f "$DATA_DISK")"; then DATA_DIR=/data; fi
fi
[ -n "${DATA_DIR:-}" ] && [ "$fail" = 0 ] && move_home_dirs "$DATA_DIR"
free_gb=$(df --output=avail -BG "$S4/." 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -ge "$NEED_GB" ]; then verdict PASS "storage: $free_gb GB free beside ~/qurl (the plan needs about $NEED_GB GB)"
else verdict WARN "storage: ${free_gb:-0} GB free beside ~/qurl; the plan needs about $NEED_GB GB. Blank disks: $(blank_disks). Run this again with DATA_DISK=<one of them> (it is formatted) or DATA_DIR=<a large mounted filesystem>"; fi
x=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "${x:-0}" -ge 50 ] || verdict WARN "/ has ${x:-0} GB free: Ray writes its logs to /tmp"
x=$(df --output=size -BG /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${x:-0}" -ge 64 ] || verdict WARN "/dev/shm is ${x:-0} GB: Ray's object store lives there"

# ===== 3. packages; automatic upgrades off
if [ "$mode" = install ]; then
  log "3. packages; automatic upgrades off for the rental"
  command -v cloud-init >/dev/null && root timeout 600 cloud-init status --wait >/dev/null 2>&1
  root systemctl disable --now apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1
  root systemctl disable unattended-upgrades.service >/dev/null 2>&1
  printf 'APT::Periodic::Update-Package-Lists "0";\nAPT::Periodic::Unattended-Upgrade "0";\n' | root tee /etc/apt/apt.conf.d/99qurl-no-auto-upgrades >/dev/null
  aptq update || verdict WARN "apt-get update reported errors"
  aptq install --no-install-recommends "linux-headers-$(uname -r)" \
    || { verdict FAIL "no kernel headers for $(uname -r): DKMS cannot build the driver. Install a stock kernel (sudo apt install linux-generic), reboot into it, and run this again"; finish; }
  aptq install --no-install-recommends build-essential git curl wget ca-certificates pciutils mokutil rsync zstd tmux htop jq \
    || { verdict FAIL "the system packages did not install (apt's messages above)"; finish; }
  aptq install --no-install-recommends nvtop >/dev/null 2>&1
fi

# ===== 4. NVIDIA driver (nvidia-smi) and Fabric Manager
drv=""
gpu_ok && drv=$(q driver_version | head -1)
if [ "$mode" = install ]; then
  log "4. NVIDIA driver"
  if [ -n "$drv" ] && ver_ge "$drv" "$MIN_DRIVER"; then
    verdict PASS "driver $drv loaded (CUDA 13.0 needs >= $MIN_DRIVER); not changed"
  elif [ -n "$drv" ]; then
    if [ "${UPGRADE_DRIVER:-0}" = 1 ]; then purge_driver && install_driver && reboot=1
    else verdict FAIL "driver $drv is older than CUDA 13.0's minimum $MIN_DRIVER. Run this again with UPGRADE_DRIVER=1 to replace it with branch $BRANCH (one reboot)"; fi
  elif [ "${PRETEND_NO_DRIVER:-0}" != 1 ] && dpkg -s "nvidia-headless-$BRANCH-open" >/dev/null 2>&1; then
    load_driver || reboot=1   # installed by an earlier run, not loaded yet
  else
    install_driver && { load_driver || reboot=1; }
  fi
  [ "$reboot" = 0 ] && [ "$fail" = 0 ] && gpu_ok && setup_fm
fi

# ===== 5. CUDA 13.0 toolkit (nvcc), also before a reboot so that the second run is short
if [ "$mode" = install ] && [ "$fail" = 0 ]; then
  log "5. CUDA 13.0 toolkit"
  if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    { ensure_repo && aptq install --no-install-recommends cuda-toolkit-13-0; } \
      || verdict WARN "the CUDA 13.0 toolkit did not install: setup_s4.sh will skip the flash-attn build (the runs then use SDPA)"
  fi
  [ -e /etc/profile.d/qurl-cuda.sh ] || echo "export PATH=$CUDA_DIR/bin:\$PATH" | root tee /etc/profile.d/qurl-cuda.sh >/dev/null
fi
if [ -x "$CUDA_DIR/bin/nvcc" ]; then
  verdict PASS "CUDA toolkit $("$CUDA_DIR/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p') at $CUDA_DIR (nvcc for the flash-attn build)"
elif [ "$mode" = install ] && [ "$DRY" != 1 ]; then
  verdict WARN "no nvcc at $CUDA_DIR: the runs will use SDPA"
fi

# ===== 6. GPU state
if [ "$mode" = install ] && [ "$reboot" = 0 ] && [ "$fail" = 0 ] && gpu_ok; then
  log "6. GPU state"
  root systemctl enable --now nvidia-persistenced >/dev/null 2>&1
  q persistence_mode | grep -q Disabled && root nvidia-smi -pm 1 >/dev/null
  if q mig.mode.current | grep -q Enabled; then
    root nvidia-smi -mig 0; reboot=1; verdict WARN "MIG was on: now off, from the next reboot"
  fi
  hold_driver
  install_monitor
fi

# ===== 7. checks
if [ "$reboot" = 0 ] && gpu_ok; then
  log "7. checks"
  checks
elif [ "$reboot" = 0 ] && [ "$fail" = 0 ] && [ "$DRY" != 1 ]; then
  verdict FAIL "nvidia-smi does not work (no NVIDIA driver loaded)"
fi
finish
