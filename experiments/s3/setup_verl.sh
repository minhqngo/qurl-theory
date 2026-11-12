#!/usr/bin/env bash
# S3 environment (P1 note, section 7): VeRL v0.9.1 with its own locked stack, in a venv separate from S1/S2's.
#   bash ~/qurl/s3/setup_verl.sh
# Creates (all user space; no root, nvcc or docker; uv installs prebuilt wheels only):
#   ~/qurl/verl                 VeRL source at tag v0.9.1, installed editable into the venv, with one local patch
#                               (step 2b: the critic loader honours the configured attention implementation)
#   ~/venvs/verl                uv.lock's "vllm + fsdp" stack: vLLM 0.24.0, torch 2.11.0+cu130, transformers 5.9.0
#                               (S1/S2's ~/venvs/qurl with vLLM 0.33 is left untouched). flash-attn is skipped: the
#                               v0.9.1 lock points at a wheelhouse release that has since been removed (HTTP 404; the
#                               wheelhouse now only has torch 2.13 builds). Training uses PyTorch SDPA attention instead
#                               (pilot.sh sets it); vLLM ships its own attention kernels and does not need the package.
#   ~/data/gsm8k/{train,test}.parquet   VeRL's GSM8K preprocessing (7,473 train / 1,319 test prompts)
#   the Qwen2.5-0.5B-Instruct weights in the Hugging Face cache
# Re-running is safe: finished steps are skipped or repeated idempotently.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VERL_TAG=v0.9.1
SRC="$HOME/qurl/verl"
VENV="$HOME/venvs/verl"
DATA="$HOME/data/gsm8k"
MODEL=Qwen/Qwen2.5-0.5B-Instruct
OUT="$HOME/qurl/s3"
mkdir -p "$OUT"

command -v uv >/dev/null || { echo "uv not found: run qurl/h100/setup.sh first" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found: uv needs it to build mbridge from its git source" >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader

# 1. Source at the release tag.
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VERL_TAG" https://github.com/verl-project/verl.git "$SRC"
fi
test "$(git -C "$SRC" describe --tags --exact-match)" = "$VERL_TAG"

# 2. The locked environment, exactly as uv.lock pins it for the vllm + fsdp extras, minus flash-attn (see above).
cd "$SRC"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --all-packages --extra vllm --extra fsdp --python 3.12 \
  --no-install-package flash-attn
PY="$VENV/bin/python"

# 2b. One local patch to the (editable) VeRL source, idempotent. v0.9.1 loads the critic's value-head model with
#     attn_implementation hard-coded to "flash_attention_2" (verl/utils/model.py, load_valuehead_model) and ignores
#     critic.model.override_config. Without flash-attn that fails, so the loader is made to use the attention the
#     critic's config asks for (pilot.sh asks for sdpa). `git diff` in env.txt records the change.
"$PY" - "$SRC/verl/utils/model.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'attn_implementation="flash_attention_2",'
new = 'attn_implementation=getattr(model_config, "_attn_implementation", None) or "flash_attention_2",'
head, sep, tail = s.partition("def load_valuehead_model(")
body, nxt, rest = tail.partition("\n_architecture_to_auto_class")
if body.count(new) == 2:
    print("verl patch: already applied")
else:
    assert body.count(old) == 2, f"verl patch: expected 2 hard-coded attention settings, found {body.count(old)}"
    open(p, "w").write(head + sep + body.replace(old, new) + nxt + rest)
    print("verl patch: the critic loader now uses the configured attention implementation")
EOF

# 3. Versions and a CUDA smoke check (written to env.txt for the record).
"$PY" - <<'EOF' | tee "$OUT/env.txt"
import importlib.util, torch, vllm, transformers, verl, ray
print("verl", verl.__version__, "| vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| transformers", transformers.__version__, "| ray", ray.__version__,
      "| flash_attn installed:", importlib.util.find_spec("flash_attn") is not None)
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool(torch.isfinite(x @ x).all()))
EOF
git -C "$SRC" rev-parse HEAD | sed 's/^/verl commit /' | tee -a "$OUT/env.txt"
git -C "$SRC" diff --stat | sed 's/^/local patch: /' | tee -a "$OUT/env.txt"

# 4. Data: VeRL's own GSM8K preprocessing (prompt + 'Let's think step by step and output the final answer after "####".').
mkdir -p "$DATA"
if [ ! -s "$DATA/train.parquet" ] || [ ! -s "$DATA/test.parquet" ]; then
  "$PY" "$SRC/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA"
fi
"$PY" - "$DATA" <<'EOF' | tee -a "$OUT/env.txt"
import sys, pandas as pd
d = sys.argv[1]
tr, te = pd.read_parquet(f"{d}/train.parquet"), pd.read_parquet(f"{d}/test.parquet")
print("gsm8k train", len(tr), "| test", len(te))
print("example prompt:", tr.iloc[0]["prompt"][0]["content"][:160].replace("\n", " "))
EOF

# 5. Model weights (already cached if S1 ran on this machine).
"$PY" -c "from huggingface_hub import snapshot_download; print('model at', snapshot_download('$MODEL'))"
echo "== setup done $(date -Is)"
