#!/usr/bin/env bash
# Progress of S3 runs, one line per run: steps done, last step time, training score, rollout-trainer mismatch and the
# latest greedy validation accuracies.
#   bash ~/qurl/s3/progress.sh <tag> [<tag> ...]
#   e.g. bash ~/qurl/s3/progress.sh t1_r1_lr1e-5 t1_r3_lr1e-5
if pgrep -f '[p]ilot.sh' >/dev/null; then echo "running: $(pgrep -af '[p]ilot.sh' | grep -o 'pilot.sh [^ ]*' | head -1)"; else echo "no run in progress"; fi
for TAG in "$@"; do
  M="$HOME/qurl/s3/results/$TAG/metrics.jsonl"
  if [ ! -f "$M" ]; then echo "$TAG: not started"; continue; fi
  "$HOME/venvs/verl/bin/python" - "$M" "$TAG" <<'EOF'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
tag = sys.argv[2]
if not rows:
    print(f"{tag}: starting")
    sys.exit()
d = rows[-1]["data"]
val = [(r["step"], r["data"]["val-core/openai/gsm8k/acc/mean@1"]) for r in rows
       if "val-core/openai/gsm8k/acc/mean@1" in r["data"]]
print(f"{tag}: step {rows[-1]['step']}, last step {d.get('timing_s/step', float('nan')):.1f} s, "
      f"score {d.get('critic/score/mean', float('nan')):.3f}, mismatch k3 {d.get('rollout_corr/k3_kl', float('nan')):.2e}; "
      f"validation: " + (", ".join(f"step {s} {100 * v:.1f}%" for s, v in val[-4:]) or "none yet"))
EOF
done
