#!/usr/bin/env bash
# Revision experiments: check the server before the queue starts. It changes nothing; paste its output back.
#   bash ~/qurl/rev/preflight.sh
set -uo pipefail
REV="$HOME/qurl/rev"
S3="$HOME/qurl/s3"
ok() { printf '  %-4s %s\n' OK "$*"; }
no() { printf '  %-4s %s\n' MISS "$*"; }
chk() { if [ -e "$1" ]; then ok "$2"; else no "$2 ($1)"; fi; }

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$apps" ]; then no "GPU busy: $apps"; else ok "no process on the GPU"; fi
pgrep -af '[p]ilot.sh|[m]ain_ppo|[q]ueue.sh' | sed 's/^/  running: /'

echo "== Disk (X3 needs about 35 GB at its peak, X6 about 25 GB; keep 100 GB free)"
df -h "$HOME" | sed 's/^/  /'

echo "== Code"
for f in "$S3/arm.sh" "$S3/pilot.sh" "$S3/progress.sh" "$S3/qurl_verl/int8_rollout.py" "$S3/qurl_verl/r5_optim.py" \
         "$S3/qurl_verl/acr.py" "$S3/qurl_verl/int8_w8a8_config.json" "$HOME/qurl/s2/s2_train.py" \
         "$HOME/qurl/e0/e0_logprobs.py" "$HOME/qurl/e0/make_uaq_checkpoints.py" "$HOME/qurl/e0/make_int8_w8a8.py" \
         "$HOME/qurl/e0/compare_int8_codes.py" "$HOME/qurl/e0/analyze_e0.py" "$HOME/qurl/h100/analyze_s1.py" \
         "$HOME/qurl/h100/s1_run.sh"; do
  chk "$f" "$f"
done
S2_SHA=86d57c5f28c5de3cb7fff30a137f193ee930ef65ebda664dc2ae5e3346f59f71   # S2's s2_train.py, which X4 imports
if [ "$(sha256sum "$HOME/qurl/s2/s2_train.py" 2>/dev/null | cut -c1-64)" = "$S2_SHA" ]; then ok "s2_train.py is S2's version"
else no "s2_train.py differs from S2's version (sha256 $S2_SHA): copy experiments/s2/s2_train.py"; fi
for f in score_gsm8k.py merge_fp32.py common.sh queue.sh x1_base.sh x2_e2dumps.sh x3_run.sh x5_run.sh \
         x4_run.sh x6_run.sh x7_run.sh registration.sha256; do
  chk "$REV/$f" "rev/$f"
done
if [ -f "$REV/registration.sha256" ]; then
  (cd "$REV" && sha256sum --quiet -c registration.sha256) && ok "frozen analyses match registration.sha256" \
    || no "frozen analyses differ from registration.sha256"
fi

echo "== Python environments"
for v in verl qurl llmc; do
  if [ -x "$HOME/venvs/$v/bin/python" ]; then
    "$HOME/venvs/$v/bin/python" - "$v" <<'EOF' 2>/dev/null | sed 's/^/  OK   /' || echo "  MISS venv $v imports"
import importlib, sys
mods = {"verl": ["verl", "vllm", "torch", "transformers"], "qurl": ["vllm", "torch", "transformers"],
        "llmc": ["llmcompressor", "torch"]}[sys.argv[1]]
out = []
for m in mods:
    out.append(f"{m} {getattr(importlib.import_module(m), '__version__', '?')}")
print(f"venv {sys.argv[1]}: " + ", ".join(out))
EOF
  else no "venv ~/venvs/$v"; fi
done

echo "== Data and models"
chk "$HOME/data/gsm8k/train.parquet" "GSM8K train parquet"
chk "$HOME/data/gsm8k/test.parquet" "GSM8K test parquet"
chk "$HOME/qurl/models/qwen0.5b-instruct-uaq1.5/uaq.json" "S3's UAQ checkpoint (s = 1.5) for R4"
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-Math-1.5B Qwen/Qwen2.5-Math-7B; do
  if "$HOME/venvs/verl/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$m', local_files_only=True)" \
       > /dev/null 2>&1; then ok "model cached: $m"; else no "model not in the HF cache: $m"; fi
done

echo "== Earlier results reused"
n=0
for d in "$HOME"/qurl/e2/results/*/dumps/full; do
  [ -d "$d" ] || continue
  syncs=$(ls "$d" | sed -n -E 's/^sync([0-9]{4})_b[0-9]+\.pt$/\1/p' | sort -u | tr '\n' ' ')
  ok "E2 dumps $d: syncs $syncs"
  n=$((n + 1))
done
[ $n -gt 0 ] || no "E2 full dumps (~/qurl/e2/results/*/dumps/full): X2 will be skipped"
for t in qwen0.5b-instruct math1.5b math7b; do   # X7 regenerates the 0.5B responses if missing; X6 needs the others
  chk "$HOME/qurl/h100/results/$t/responses.json" "S1 responses: $t"
  n=$(ls "$HOME/qurl/h100/results/$t"/score_*.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok "S1 score files: $t ($n)" || no "S1 score files: $t (the on-server X6/X7 reports need them; the laptop's copies suffice)"
done
for d in "$HOME"/e0_work/*/; do [ -d "$d" ] && ok "S1 work dir: $d ($(ls "$d" | tr '\n' ' '))"; done
chk "$S3/results/r5_check/compare.txt" "S3's R5 exactness check"
