#!/usr/bin/env bash
# S4 shake-out on S3's single H100, before the rental (README "Before the rental"). Not a gate, and nothing from it is
# reported: it runs every day-1 step that one GPU can run, so configuration errors surface before 8 GPUs are paid for.
#   nohup bash ~/qurl/s4/shakeout.sh > ~/qurl/s4/shakeout.log 2>&1 < /dev/null &      (about 1-1.5 h; needs the GPU)
# Parts. After the setup, each runs even if an earlier one failed, so one pass finds as many problems as it can:
#   0  setup_s4.sh for 1.5B only, without the flash-attn build (SIZES=1.5b FLASH_ATTN=skip; log ~/qurl/s4/setup.log).
#      On S3's H100 it re-syncs S3's venv to the same lock, so the venv stays as it is; ~/qurl/verl keeps S3's
#      critic-loader patch, which GRPO never calls. Its 800 GB disk warning is expected here: this needs about 40 GB.
#   1  the machine: check_node.py on one GPU (run by the setup), and the frozen hashes of the analysis plan and its
#      addendum (as in G1)
#   2  packed attention: gate G2, unchanged
#   3  R5's exactness at 1.5B: gate G3, on one GPU
#   4  a 2-step 1.5B R3 run at a small batch (32 prompts x 4 responses, mini-batches of 8 prompts), otherwise with S4's
#      settings: validation on AIME 2024 x 32 before step 1 and after step 2, the INT8 test dump, a checkpoint at
#      step 2, then finish.sh: the merge and a short evaluation (eval.py --limit 8)
#   5  gate G4's checks that need no 20-step window: the INT8 codes after the first sync equal an independent
#      quantization (check_int8_dump.py); every sync re-quantized the same number of weights; and the flip log has
#      exactly the first sync plus one per step (syncs 1-3), no sync flipping 20% or more, and no flip at sync 2,
#      because the warm-up gives step 1 a learning rate of 0. An extra sync (say, for a validation) would shift the
#      analysis plan's flip window by one step in the full runs. And the tie log, which the registered M1 variant
#      without the ties needs: about 0.2% of the codes are ties at sync 1, none of them flips at sync 2, and step 2's
#      update flips about half of them.
# Writes ~/qurl/s4/results/shakeout/summary.txt, ending "shake-out: N of 9 checks passed". Rerunning repeats parts 1-5
# (the setup skips its finished steps). Its results must not go to the local experiments/s4/results/ (README).
set -uo pipefail
S4="$HOME/qurl/s4"
PY="$HOME/venvs/verl/bin/python"
OUT="$S4/results/shakeout"
TAG=shake_1.5b_R3
RUN="$S4/results/$TAG"
BASE=Qwen/Qwen2.5-Math-1.5B
export NGPU=1
export PYTHONPATH="$S4${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
: > "$OUT/summary.txt"
passed=0
total=0
log() { echo "[$(date -Is)] $*" | tee -a "$OUT/summary.txt"; }
show() { sed 's/^/    /' | tee -a "$OUT/summary.txt"; }
check() {  # check <exit status> <what was checked>
  total=$((total + 1))
  if [ "$1" = 0 ]; then passed=$((passed + 1)); log "PASS $2"; else log "FAIL $2"; fi
}

log "== 0 setup_s4.sh, 1.5B only, no flash-attn build (log: ~/qurl/s4/setup.log)"
SIZES=1.5b FLASH_ATTN=skip bash "$S4/setup_s4.sh" > "$S4/setup.log" 2>&1
rc=$?
tail -n 3 "$S4/setup.log" | show
check "$rc" "setup_s4.sh finished"
[ "$rc" = 0 ] || { log "shake-out: stopped; nothing else can run without the setup"; exit 1; }

log "== 1 the machine"
show < "$S4/env.txt"
NODE="$S4/results/node/check_node.txt"
grep -E '^(INFO|PASS|WARN|FAIL)' "$NODE" 2>/dev/null | show
[ -s "$NODE" ] && ! grep -q '^FAIL' "$NODE"
check $? "check_node.py on one GPU: no failure (results/node/check_node.txt)"
(cd "$S4" && sha256sum -c s4_analysis.sha256 s4_m1_ties.sha256 > /dev/null 2>&1)
check $? "the analysis plan: s4_analysis.py and its addendum s4_m1_ties.py match their frozen SHA-256"

log "== 2 packed attention (gate G2)"
{
  "$PY" "$S4/test_packed_attn.py" --model "$BASE" --attn sdpa --dtype float32
  "$PY" "$S4/test_packed_attn.py" --model "$BASE" --attn sdpa flash_attention_2 --dtype bfloat16 --timing-tokens 16384
} > "$OUT/packed_attn.txt" 2>&1
grep -E "OK|WRONG|not available|packed forward|^PASS|^FAIL" "$OUT/packed_attn.txt" | show
[ "$(grep -c '^PASS' "$OUT/packed_attn.txt")" = 2 ]
check $? "G2: VeRL's packed forward gives the same log-probs as one sequence at a time (packed_attn.txt)"

log "== 3 R5's exactness at 1.5B, one GPU (gate G3)"
bash "$S4/r5_check_mp.sh" 1.5b > "$OUT/r5_check.txt" 2>&1
rc=$?
grep -E "grad norms|BIT-IDENTICAL|DIFFERENT|G3 |failed" "$OUT/r5_check.txt" | show
check "$rc" "G3: R5 and an R4 repeat bit-identical to R4, R5 with ordinary clipping different (r5_check.txt)"

log "== 4 a 2-step 1.5B R3 run (results/$TAG)"
rm -rf "$RUN" "${CKPT_ROOT:-$S4/ckpt}/$TAG"
SEED=1 DUMP=1 VAL_EVERY=2 SAVE_EVERY=2 EVAL_ARGS="--limit 8 --k-aime 2 --k-math 1" \
  bash "$S4/run.sh" 1.5b R3 "$TAG" 2 data.train_batch_size=32 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 < /dev/null > "$OUT/run.txt" 2>&1
rc=$?
[ -s "$RUN/summary.txt" ] && show < "$RUN/summary.txt"
grep -E "^(aime24|math500)" "$OUT/run.txt" | show
[ "$rc" = 0 ] && grep -q "^val-core/" "$RUN/summary.txt" && [ -s "$RUN/eval.json" ]
check $? "the run: 2 steps, validation, a checkpoint, the merge and a short evaluation (run.txt, $TAG/train.log)"

log "== 5 the INT8 rollout (gate G4's checks without its 20-step window)"
"$PY" "$S4/check_int8_dump.py" "$RUN/int8_dump" --model "$BASE" > "$OUT/int8_dump.txt" 2>&1
tail -n 5 "$OUT/int8_dump.txt" | show
grep -q "^PASS" "$OUT/int8_dump.txt"
check $? "the INT8 codes after the first sync equal an independent quantization (int8_dump.txt)"
"$PY" - "$RUN" <<'EOF' | show
import json
import os
import re
import sys

run = sys.argv[1]
QKV, OTHER = ("q_proj", "k_proj", "v_proj"), ("o_proj", "gate_proj", "up_proj", "down_proj")
problems = []
path = os.path.join(run, "summary.txt")
m = re.search(r"INT8 refit: (\d+) sync lines.*?weights per sync (\d+)-(\d+)", open(path).read()) if os.path.exists(path) else None
if m is None:
    problems.append("no 'INT8 refit:' line in summary.txt")
else:
    print(m.group(0))
    if m.group(2) != m.group(3):
        problems.append("the number of re-quantized weights differs between syncs")
path = os.path.join(run, "e3_int8.jsonl")
recs = [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []


def share(r, projs):
    """Code-weighted flip share over projs (as s4_analysis.pooled_flips); None at the attempt's first sync."""
    if any(x is None for p in projs for x in r["flip"][p]):
        return None
    num = sum(r["codes_per_layer"][p] * sum(r["flip"][p]) / len(r["flip"][p]) for p in projs)
    return num / sum(r["codes_per_layer"][p] for p in projs)


for r in recs:
    q, o = share(r, QKV), share(r, OTHER)
    if q is None or o is None:
        print(f"sync {r['sync']}: the attempt's first sync (no history yet)")
        continue
    print(f"sync {r['sync']} (the update of step {r['start_step'] + r['sync'] - 1}): code flips q/k/v {q:.4%}, "
          f"other projections {o:.4%}")
    if max(q, o) >= 0.2:
        problems.append(f"sync {r['sync']} flips 20% or more")
    if r["sync"] == 2 and max(q, o) > 0:
        problems.append("sync 2 flips codes, but step 1's learning rate is 0")
syncs = [r["sync"] for r in recs]
if syncs != [1, 2, 3]:
    problems.append(f"syncs logged: {syncs}; expected [1, 2, 3], the first sync and one per step")
for p in problems:
    print("problem:", p)
sys.exit(1 if problems else 0)
EOF
check $? "the INT8 refit and its flip log: one sync per step, the same weights each time, none at step 1 (lr 0)"
"$PY" - "$RUN" <<'EOF' | show
import json
import os
import sys

run = sys.argv[1]
QKV = ("q_proj", "k_proj", "v_proj")
path = os.path.join(run, "e3_int8.jsonl")
by = {r["sync"]: r for r in (json.loads(x) for x in open(path) if x.strip())} if os.path.exists(path) else {}
problems = []
if any(r.get("ties") is None or r.get("flip_ties") is None for r in by.values()) or sorted(by) != [1, 2, 3]:
    print("no tie log (int8_rollout.py's 'ties' and 'flip_ties'), or not syncs 1-3")
    sys.exit(1)


def tie_share(projs):
    r = by[1]
    return sum(r["codes_per_layer"][p] * sum(r["ties"][p]) for p in projs) / \
        sum(r["codes_per_layer"][p] * len(r["ties"][p]) for p in projs)


def tie_flips(sync, projs):
    ties = flips = 0.0
    for p in projs:
        c = by[1]["codes_per_layer"][p]
        for t, g in zip(by[1]["ties"][p], by[sync]["flip_ties"][p]):
            ties, flips = ties + t * c, flips + (g or 0.0) * t * c
    return flips / ties


tq, to = tie_share(QKV), tie_share(tuple(by[1]["ties"]))
print(f"ties at sync 1: {tq:.3%} of the q/k/v codes, {to:.3%} of all codes")
f2, f3 = tie_flips(2, QKV), tie_flips(3, QKV)
print(f"share of the q/k/v ties that flipped: {f2:.3f} at sync 2 (step 1, lr 0), {f3:.3f} at sync 3 (step 2)")
r3 = by[3]
pr = r3["pred"]
for p in QKV:   # without the ties, step 2's flips on the tracked tensors against Proposition 4 (E2: 0.91 at pair 1-2)
    for j, i in enumerate(pr["layers"]):
        n = r3["codes_per_layer"][p]
        t, g, f = by[1]["ties"][p][i], r3["flip_ties"][p][i] or 0.0, r3["flip"][p][i]
        print(f"  layer {i:2d} {p}: flips without the ties {(f * n - g * t * n) / (n - t * n):.4%}, "
              f"Proposition 4 {pr[p][j]:.4%}")
if any(g is not None for p in by[1]["flip_ties"] for g in by[1]["flip_ties"][p]):
    problems.append("sync 1 logs flips of the ties")
if not 0.0005 < tq < 0.005:
    problems.append("the q/k/v tie share is not about 0.2%")
if f2 != 0.0:
    problems.append("ties flipped at sync 2, where the learning rate is 0")
if not 0.3 < f3 < 0.7:
    problems.append("step 2 did not flip about half of the ties")
for p in problems:
    print("problem:", p)
sys.exit(1 if problems else 0)
EOF
check $? "the tie log for the registered M1 variant: about 0.2% ties, none flipped at step 1, about half at step 2"

log "shake-out: $passed of $total checks passed"
[ "$passed" = "$total" ]
