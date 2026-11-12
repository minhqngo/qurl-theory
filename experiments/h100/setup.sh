#!/usr/bin/env bash
# One-time environment setup on the H100 server (no root, no nvcc, no docker needed).
#   bash ~/qurl/h100/setup.sh
# Creates:
#   ~/.local/bin/uv            user-space uv
#   ~/venvs/qurl               uv-managed Python 3.12 (ships Python.h for Triton) + vLLM (prebuilt wheels)
#   ~/venvs/llmc               llm-compressor, used only to write INT8 W8A8 checkpoints
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

if [ ! -x "$HOME/venvs/qurl/bin/python" ]; then
  uv venv --managed-python --python 3.12 "$HOME/venvs/qurl"
fi
VIRTUAL_ENV="$HOME/venvs/qurl" uv pip install --python "$HOME/venvs/qurl/bin/python" vllm --torch-backend=auto

if [ ! -x "$HOME/venvs/llmc/bin/python" ]; then
  uv venv --managed-python --python 3.12 "$HOME/venvs/llmc"
fi
uv pip install --python "$HOME/venvs/llmc/bin/python" llmcompressor

"$HOME/venvs/qurl/bin/python" - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
EOF
"$HOME/venvs/llmc/bin/python" -c "import llmcompressor, torch; print('llmcompressor', llmcompressor.__version__, '| torch', torch.__version__)"
