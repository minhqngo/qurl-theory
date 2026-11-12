#!/usr/bin/env bash
# Laptop smoke test of the S4 loop (not reported; one 12 GB GPU under WSL, so nothing here tests multi-GPU code).
#   bash ~/qurl/s4/laptop_smoke.sh <arm> [steps]      e.g. R3 (INT8 + TIS + ACR + E3 log + test dump), R4, R5
# Runs run.sh at tiny scale with Qwen2.5-0.5B-Instruct as a stand-in: 8 prompts x 4 responses per step, responses
# <= 512 tokens, checkpoints every step, then finish.sh (merge + eval.py on 2 problems per benchmark).
# Training data: S3's GSM8K file (~/data/gsm8k/train.parquet; the DAPO reward manager scores it with VeRL's GSM8K rule),
# because the 0.5B model gets no DAPO problem right within a short cap: all rewards equal, zero advantages, zero
# gradients, and neither the update path nor the code-flip log would be exercised.
# Laptop-only settings (as in S3's smoke tests): BF16 master weights for the actor (FP32 + Adam does not fit next to
# vLLM in 12 GB), eager vLLM at 30% memory, one reward and one agent-loop worker, and vLLM's Triton INT8 kernel
# (CUTLASS has no INT8 build for this SM120 GPU). The VeRL install on the laptop also carries S3's WSL patch
# (CUDA IPC reported unsupported, so weights go through shared memory).
# Resume test: RESUME=1 deletes summary.txt / eval.json of a finished smoke run and continues it one more step from
# its last checkpoint (metrics.jsonl must be kept as metrics.part1.jsonl, the E3 log appended with a new attempt).
# Flip-log test: TAG=smoke_R3_lr BASE_LR=1e-4 SMOKE_BATCH=16 SMOKE_N=8 bash laptop_smoke.sh R3 3. At lr 1e-6 the
# laptop's BF16 master weights absorb almost no update (BF16 cannot represent it), so codes barely flip; at 1e-4 they
# do, and the logged flip shares can be compared with Proposition 4's prediction from the same weights. The larger
# batch makes a correct GSM8K answer (hence a non-zero gradient) likely in every step.
set -euo pipefail
ARM=$1
STEPS=${2:-2}
TAG=${TAG:-smoke_$ARM}
S4="$HOME/qurl/s4"
if [ "${RESUME:-0}" = 1 ]; then
  rm -f "$S4/results/$TAG/summary.txt" "$S4/results/$TAG/eval.json"
else
  rm -rf "$S4/results/$TAG" "$S4/ckpt/$TAG"
fi
export VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel
NGPU=1 VAL_BEFORE=${VAL_BEFORE:-True} MAX_PROMPT=512 MAX_RESPONSE=512 OVERLONG=128 CONTEXT=32768 VAL_EVERY=1 SAVE_EVERY=1 \
TOK_TRAIN=2048 TOK_LOGPROB=4096 GPU_MEM=0.3 DUMP=1 KEEP_CKPT=1 \
TRAIN_FILE="$HOME/data/gsm8k/train.parquet" VAL_FILE="$HOME/data/s4/aime_smoke.parquet" \
EVAL_ARGS="--limit 2 --k-aime 2 --k-math 2 --max-prompt 512" \
bash "$S4/run.sh" 0.5b "$ARM" "$TAG" "$STEPS" \
  data.train_batch_size="${SMOKE_BATCH:-8}" \
  actor_rollout_ref.rollout.n="${SMOKE_N:-4}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  reward.num_workers=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
