#!/usr/bin/env bash
# S3 run driver (P1 note, section 7): VeRL PPO on GSM8K with Qwen2.5-0.5B-Instruct, i.e. the QuRL paper's GSM8K setup,
# which is VeRL's quickstart: batch 256, prompts and responses up to 512 tokens, PPO with a critic (GAE), 15 epochs =
# 435 steps. The regularizers follow VeRL's documented PPO baseline for this model and task (56.7% GSM8K test
# accuracy): KL penalty 0.001 in the reward and entropy bonus 0.001. Both were VeRL defaults when that baseline was
# run; v0.9.1 turns both off unless set, which is why they are set explicitly here.
# Arms (environment variable ARM, default R1):
#   R1  BF16 rollout
#   R3  QuRL: INT8 W8A8 rollout re-quantized at every weight sync (qurl_verl/int8_rollout.py, S1's quantizer config),
#       token-level TIS with cap C = QURL_TIS_C (default 2.0) and the ACR clip bound (qurl_verl/acr.py)
#   (R2, R4, R5, R6 come later.)
#   [ARM=R3] bash ~/qurl/s3/pilot.sh <tag> [steps] [extra Hydra overrides ...]
#   e.g. bash ~/qurl/s3/pilot.sh r1_pilot 10                  # 10-step timing pilot, validation at step 10
#        ARM=R3 bash ~/qurl/s3/pilot.sh r3_pilot 10
#        bash ~/qurl/s3/pilot.sh r1_pilot_lr1e-5 10 actor_rollout_ref.actor.optim.lr=1e-5
# Outputs in ~/qurl/s3/results/<tag>/: metrics.jsonl (one line per step: all VeRL metrics, incl. timing_s/*),
# train.log (console), env.txt (versions, GPU, arm, the exact command), summary.txt. INT8 arms add int8_refit.txt
# (the refit's per-sync log: tensors re-quantized, synced dtype, vLLM's INT8 kernel) and int8_dump/ (layer 0's fused
# QKV codes and scales after the first two syncs, for an offline exactness check).
set -euo pipefail
TAG=$1; shift
STEPS=${1:-10}; [ $# -gt 0 ] && shift
ARM=${ARM:-R1}
S3="$HOME/qurl/s3"
RES="$S3/results/$TAG"
PY="$HOME/venvs/verl/bin/python"
DATA="$HOME/data/gsm8k"
MODEL=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p "$RES"
if [ -s "$RES/metrics.jsonl" ]; then echo "results/$TAG already has metrics.jsonl; use a new tag" >&2; exit 1; fi

ARGS=(
  # data (VeRL quickstart = QuRL's GSM8K setup)
  data.train_files="$DATA/train.parquet"
  data.val_files="$DATA/test.parquet"
  data.train_batch_size=256
  data.max_prompt_length=320   # every GSM8K prompt is <= 256 tokens with the chat template (train max 256, test
                               # 232), so nothing is truncated (VeRL's agent loop would left-truncate a longer one
                               # with a warning). Prompts are padded to this length; pilot 2 showed no measurable
                               # saving over 512, but nothing is lost either
  data.max_response_length=512
  # actor: AdamW (VeRL defaults: betas 0.9/0.999, weight decay 0.01, grad clip 1.0, constant lr), FP32 master weights
  # model: PyTorch SDPA attention (no flash-attn package, see setup_verl.sh), so padded batches without packing
  actor_rollout_ref.model.path=$MODEL
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.entropy_coeff=0.001
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8     # padded 832-token sequences x 152k-vocab FP32
                                                             # logits plus their entropy: about 1 GB per sequence
  # rollout: vLLM (BF16 for R1; INT8 W8A8 for R3, see below), temperature 1; validation is greedy (VeRL default)
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
  # critic
  critic.model.path=$MODEL
  +critic.model.override_config.attn_implementation=sdpa
  critic.model.use_remove_padding=False
  critic.optim.lr=1e-5
  critic.ppo_micro_batch_size_per_gpu=16
  algorithm.use_kl_in_reward=True
  algorithm.kl_ctrl.kl_coef=0.001
  # trainer: one GPU; validate at the last step only; no checkpoints in pilots
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.total_epochs=15
  trainer.total_training_steps=$STEPS
  trainer.val_before_train=False
  trainer.test_freq=$STEPS
  trainer.save_freq=-1
  trainer.logger='["console","file"]'
  trainer.project_name=qurl_s3
  trainer.experiment_name=$TAG
  ray_kwargs.ray_init.runtime_env.py_executable=$PY
)

case "$ARM" in
  R1) ;;
  R3)
    # The plugin must be importable in every Ray worker and in vLLM's worker processes; they inherit this environment.
    export PYTHONPATH="$S3${PYTHONPATH:+:$PYTHONPATH}"
    export QURL_TIS_C=${QURL_TIS_C:-2.0}
    export QURL_INT8_DUMP_DIR="$RES/int8_dump"
    HF_JSON=$("$PY" -c 'import json, sys; print(json.dumps({"quantization_config": json.load(open(sys.argv[1]))}, separators=(",", ":")))' \
      "$S3/qurl_verl/int8_w8a8_config.json")
    ARGS+=(
      # QuRL's objective: TIS weight min(rho, C) and ACR's upper bound, applied by the qurl_acr loss, which needs
      # VeRL's token-level rho = pi_old / mu untruncated (hence the huge threshold)
      actor_rollout_ref.model.external_lib=qurl_verl
      actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
      algorithm.rollout_correction.rollout_is=token
      algorithm.rollout_correction.rollout_is_threshold=1e30
      # INT8 W8A8 rollout: vLLM builds compressed-tensors INT8 layers (S1's config); the worker extension re-quantizes
      # the synced weights into them at every weight sync
      +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='${HF_JSON}'"
      +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl_verl.int8_rollout.QuRLWorkerExtension
    )
    ;;
  *) echo "ARM=$ARM is not supported yet (R1, R3)" >&2; exit 1 ;;
esac

{
  echo "== S3 $TAG: arm $ARM, $STEPS steps $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
  cat "$S3/env.txt" 2>/dev/null || true
  [ "$ARM" = R1 ] || echo "QURL_TIS_C=$QURL_TIS_C"
  printf '%s\n' "command: $PY -m verl.trainer.main_ppo" "${ARGS[@]}" "$@"
} | tee "$RES/env.txt"

cd "$RES"                                   # Ray/Hydra outputs land here
export VERL_FILE_LOGGER_PATH="$RES/metrics.jsonl"
export PYTHONUNBUFFERED=1
"$PY" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" > "$RES/train.log" 2>&1
echo "== done $TAG $(date -Is)"
if [ "$ARM" != R1 ]; then
  grep -o "\[qurl int8 refit\].*" "$RES/train.log" > "$RES/int8_refit.txt" || true
fi
"$PY" "$S3/summarize_pilot.py" "$RES" | tee "$RES/summary.txt"
