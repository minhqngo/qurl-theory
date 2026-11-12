#!/usr/bin/env bash
# Noise floor for MXFP8: rescore the original checkpoint in the same order and compare.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HOME/venvs/qurl/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
ORIG=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B-Instruct'))")
python "$HERE/e0_logprobs.py" score "$ORIG" "$HERE/results/responses.json" "$HERE/results/repeat_mxfp8_orig.json" \
  --quant mxfp8 > "$HERE/results_repeat.log" 2>&1
python - "$HERE/results" <<'EOF'
import json, sys
import numpy as np
r = sys.argv[1]
L = lambda p: np.concatenate([np.array(x) for x in json.load(open(f"{r}/{p}"))["logprobs"]])
a = L("score_mxfp8_orig.json")
for name, f in [("same-order repeat", "repeat_mxfp8_orig.json"), ("s=2", "score_mxfp8_s2.json"),
                ("s=1.5", "score_mxfp8_s1.5.json")]:
    d = np.abs(a - L(f))
    print(f"mxfp8 orig vs {name:<18}: mean|d| {d.mean():.2e}, identical {100 * np.mean(d == 0):.2f}%")
EOF
