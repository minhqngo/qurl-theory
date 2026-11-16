#!/usr/bin/env bash
# S4 environment on the rented 8-GPU node, all in user space (no root, no docker), after bootstrap_node.sh has
# installed the driver, Fabric Manager and the CUDA 13.0 toolkit. About 1-3 h: the flash-attn build, two model
# downloads, and four UAQ checkpoints written on the CPU (7B in FP32 is 30 GB each).
#   bash ~/qurl/s4/setup_s4.sh 2>&1 | tee ~/qurl/s4/setup.log
# Once the venv exists (about 10 min in), check_node.py loads every GPU (about 2 min): a node that fails it stops here,
# before the long steps, so it can go back to the provider early.
#   FLASH_ATTN=skip  skip the flash-attn build (the runs then use SDPA: correct, slower; README "Attention")
#   CKPT_ROOT=/big/disk/s4ckpt  where run checkpoints go (default ~/qurl/s4/ckpt); needs about 800 GB free
#   SIZES="1.5b"     models and UAQ checkpoints to prepare (default "1.5b 7b"; 1.5b alone for a shake-out)
# Creates:
#   ~/qurl/verl            VeRL v0.9.1 source, installed editable into the venv; no local patches (S3's patch was for
#                          the PPO critic, which GRPO does not have)
#   ~/venvs/verl           uv.lock's "vllm + fsdp" stack, as in S3: vLLM 0.24.0, torch 2.11.0+cu130, transformers
#                          5.9.0; plus flash-attn 2.8.3 built from source for SM90 when nvcc 13.x is available (the
#                          lock's prebuilt wheel has been removed from VeRL's wheelhouse)
#   ~/qurl/models/qwen2.5-math-{7b,1.5b}-uaq{1.5,2}   UAQ checkpoints (W_QKV / s, gamma * s) in FP32
#   ~/data/s4/             prepare_data.py: DAPO-Math-17k (17,391 distinct problems), AIME 2024, MATH-500
#   ~/qurl/s4/env.txt      versions, GPUs, attention implementation
# Re-running is safe: finished steps are skipped.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VERL_TAG=v0.9.1
SRC="$HOME/qurl/verl"
VENV="$HOME/venvs/verl"
S4="$HOME/qurl/s4"
MODELS="$HOME/qurl/models"
DATA="$HOME/data/s4"
CKPT_ROOT=${CKPT_ROOT:-$S4/ckpt}
SIZES=${SIZES:-1.5b 7b}
hub_id() { case "$1" in 7b) echo Qwen/Qwen2.5-Math-7B ;; 1.5b) echo Qwen/Qwen2.5-Math-1.5B ;; esac; }

# 0. The machine.
nvidia-smi -L >/dev/null 2>&1 || { echo "no working NVIDIA driver: run bootstrap_node.sh first (README, Day 1)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
echo "CPU cores $(nproc) | RAM $(free -g | awk '/Mem:/{print $2}') GB"
mkdir -p "$CKPT_ROOT" "$MODELS" "$DATA"
avail=$(df --output=avail -BG "$CKPT_ROOT" | tail -1 | tr -dc 0-9)
echo "free space for checkpoints at $CKPT_ROOT: ${avail} GB"
[ "$avail" -ge 800 ] || echo "WARNING: the plan needs about 800 GB for checkpoints and merged models"

# 1. uv (Astral's installer, into ~/.local/bin) and git.
command -v git >/dev/null || { echo "git is required (uv builds two git dependencies)" >&2; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. VeRL at the release tag, and the locked environment minus flash-attn (S3's environment).
[ -d "$SRC/.git" ] || git clone --depth 1 --branch "$VERL_TAG" https://github.com/verl-project/verl.git "$SRC"
test "$(git -C "$SRC" describe --tags --exact-match)" = "$VERL_TAG"
(cd "$SRC" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --all-packages --extra vllm --extra fsdp --python 3.12 \
  --no-install-package flash-attn)
PY="$VENV/bin/python"

# 2a. NCCL without its InfiniBand/RoCE transport. Every S4 run is on one node, where NCCL talks over NVLink. On the
# rental, NCCL 2.28.9's IB transport picked the Intel RoCE NIC (irdma0) and segfaulted at the first collective, on any
# GPUs and whatever else was switched off (nccl_diag.sh, nccl_diag2.sh). NCCL reads ~/.nccl.conf in every process,
# Ray's and vLLM's workers included.
if ! grep -q '^NCCL_IB_DISABLE=1' "$HOME/.nccl.conf" 2>/dev/null; then
  echo "NCCL_IB_DISABLE=1" >> "$HOME/.nccl.conf"
  echo "NCCL: InfiniBand transport off for this user (~/.nccl.conf)"
fi

# 2b. The GPUs under load (check_node.py: memory, matmuls, host copies, NCCL), skipped once it has passed.
NODE="$S4/results/node"
mkdir -p "$NODE"
if [ ! -s "$NODE/check_node.txt" ] || grep -q '^FAIL' "$NODE/check_node.txt"; then
  rc=0
  "$PY" "$S4/check_node.py" --out "$NODE/check_node.txt" || rc=$?
  if [ "$rc" = 2 ]; then
    echo "the node FAILED check_node.py (results/node/check_node.txt): fix it or ask the provider for another node" >&2
    exit 1
  fi
  [ "$rc" = 0 ] || echo "check_node.py: warnings (results/node/check_node.txt); setup continues"
fi

# 3. flash-attn 2.8.3 from source, SM90 only (the version the lock pins for torch 2.11 + CUDA 13).
if [ "${FLASH_ATTN:-build}" = build ] && ! "$PY" -c "import flash_attn" 2>/dev/null; then
  NVCC=$(command -v nvcc || ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "$NVCC" ] && "$NVCC" --version | grep -q "release 13\."; then
    export CUDA_HOME
    CUDA_HOME=$(dirname "$(dirname "$NVCC")")
    ram=$(free -g | awk '/Mem:/{print $2}')
    jobs=$((ram / 8)); half=$(($(nproc) / 2))
    [ "$jobs" -gt "$half" ] && jobs=$half
    [ "$jobs" -lt 1 ] && jobs=1
    echo "building flash-attn 2.8.3 with $NVCC, $jobs parallel jobs (can take 30-60 min)"
    uv pip install --python "$PY" ninja packaging psutil setuptools wheel
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=$jobs NVCC_THREADS=2 \
      timeout 3h uv pip install --python "$PY" --no-build-isolation "flash-attn==2.8.3" \
      || echo "WARNING: the flash-attn build failed; the runs will use SDPA"
  else
    echo "no nvcc 13.x found: skipping the flash-attn build; the runs will use SDPA (bootstrap_node.sh installs the"
    echo "  CUDA 13.0 toolkit at /usr/local/cuda-13.0: run it again, then rerun this)"
  fi
fi

# 4. Model weights.
for size in $SIZES; do
  "$PY" -c "from huggingface_hub import snapshot_download; print('$(hub_id "$size")', '->', snapshot_download('$(hub_id "$size")'))"
done

# 5. UAQ checkpoints on the CPU (make_uaq_checkpoint.py checks the logits: bit-identical at s = 2, FP32 rounding at 1.5).
for size in $SIZES; do
  M=$(hub_id "$size")
  for s in 1.5 2; do
    out="$MODELS/qwen2.5-math-$size-uaq$s"
    [ -f "$out/uaq.json" ] && { echo "UAQ checkpoint exists: $out"; continue; }
    CUDA_VISIBLE_DEVICES='' "$PY" "$S4/make_uaq_checkpoint.py" --model "$M" --s "$s" --out "$out"
  done
done

# 6. Data.
"$PY" -I "$S4/prepare_data.py" --out "$DATA"

# 7. The record.
"$PY" - <<'EOF' | tee "$S4/env.txt"
import importlib.util, torch, vllm, transformers, verl, ray
fa = importlib.util.find_spec("flash_attn") is not None
print("verl", verl.__version__, "| vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| transformers", transformers.__version__, "| ray", ray.__version__)
print("flash_attn installed:", fa, "-> trainer attention:", "flash_attention_2" if fa else "sdpa (packed via transformers' mask)")
print("devices:", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
EOF
git -C "$SRC" rev-parse HEAD | sed 's/^/verl commit /' | tee -a "$S4/env.txt"
echo "~/.nccl.conf: $(tr '\n' ' ' < "$HOME/.nccl.conf")" | tee -a "$S4/env.txt"
{ cat "$DATA/data.json"; echo; } >> "$S4/env.txt"   # data.json has no final newline; run.sh appends to env.txt
echo "== setup done $(date -Is): next, bash ~/qurl/s4/gates.sh"
