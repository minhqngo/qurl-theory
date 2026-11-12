#!/usr/bin/env bash
# S4 run driver: QuRL's DAPO setup (paper Sec. 5.1, Table 4) on one 8-GPU node, any arm, resumable.
#   [SEED=1] [QURL_S=1.5] [BASE_LR=1e-6] [NGPU=8] bash ~/qurl/s4/run.sh <size> <arm> <tag> [steps] [Hydra overrides ...]
#   size: 7b (Qwen2.5-Math-7B, the paper's) | 1.5b (Qwen2.5-Math-1.5B) | 0.5b (Qwen2.5-0.5B-Instruct, laptop tests)
#   arm:  R3  QuRL: INT8 W8A8 rollout re-quantized at every sync (one vLLM replica per GPU), token TIS (C = 2), ACR
#         R4  R3 with the UAQ checkpoint (W_QKV / s, gamma * s)
#         R5  R3 without UAQ, with Theorem 2(b)-(c)'s per-group optimizer settings and clipping rule (c)
#         R6  R3 with the global learning rate x s (the paper's control)
#         R1, R2  BF16 rollout without / with UAQ (not in the 10-day plan; kept for completeness)
#   e.g. SEED=1 bash ~/qurl/s4/run.sh 7b R4 s4_7b_R4_s1
# The recipe is VeRL's DAPO recipe for Qwen2.5-Math-7B (verl-recipe dapo/test_dapo_7b_math.sh), whose stated settings
# match everything the paper states for this experiment: 512 prompts x 16 responses per step, 200 steps, lr 1e-6, clip
# 0.2 / 0.28, no KL. The paper leaves the rest unstated; from the recipe: prompts <= 2,048 and responses <= 8,192
# tokens (context raised from 4,096 to 32,768), overlong penalty over the last 4,096 tokens, mini-batches of 32
# prompts (16 updates per step), 10 warm-up steps, weight decay 0.1, clip_ratio_c 10, token-mean loss, no entropy bonus,
# no dynamic sampling. Validation: AIME 2024 x 32 at temperature 1, top-p 0.7 (the mean is Avg@32) every VAL_EVERY steps.
# Seed k sets data.seed=k (prompt order) and rollout.seed=42+k; the arms of a seed share both.
# Every arm trains with FSDP2 through qurl_verl's optimizer path (QURL_OPT), so the arms run the same code.
# Resumable: checkpoints every SAVE_EVERY steps under $CKPT_ROOT/<tag>; rerunning the same command resumes from the
# latest one (VeRL's resume_mode=auto). Earlier metrics.jsonl / train.log are kept as metrics.partN.jsonl / train.partN.log.
# After training: int8_refit.txt, summary.txt (summarize.py), then finish.sh (merge the final checkpoint, evaluate it
# with eval.py into eval.json, delete the FSDP checkpoints) unless NO_FINISH=1.
# Mechanism runs (queue.sh, gates.sh): 25 steps, NO_FINISH=1 VAL_EVERY=-1 VAL_BEFORE=False SAVE_EVERY=-1, i.e. no
# validation, no checkpoints, no final evaluation; only the training metrics and the per-sync code-flip log count.
# Outputs in ~/qurl/s4/results/<tag>/: env.txt, arm.txt, metrics*.jsonl, train*.log, e3_int8.jsonl (INT8 arms: per-sync
# code flips, scale changes and quantization error), int8_refit.txt, summary.txt, eval.json.
set -euo pipefail
SIZE=$1; ARM=$2; TAG=$3; shift 3
STEPS=${1:-200}; [ $# -gt 0 ] && shift
S4="$HOME/qurl/s4"
RES="$S4/results/$TAG"
CKPT="${CKPT_ROOT:-$S4/ckpt}/$TAG"
PY="$HOME/venvs/verl/bin/python"
DATA="${DATA_DIR:-$HOME/data/s4}"
MODELS="$HOME/qurl/models"
SEED=${SEED:-1}
export QURL_S=${QURL_S:-1.5}
BASE_LR=${BASE_LR:-1e-6}
NGPU=${NGPU:-8}
MAXP=${MAX_PROMPT:-2048}
MAXR=${MAX_RESPONSE:-8192}
OVERLONG=${OVERLONG:-4096}
CTX=${CONTEXT:-32768}
VAL_EVERY=${VAL_EVERY:-20}
SAVE_EVERY=${SAVE_EVERY:-20}
TRAIN_FILE=${TRAIN_FILE:-$DATA/dapo17k.parquet}
VAL_FILE=${VAL_FILE:-$DATA/aime24_x32.parquet}

case "$SIZE" in
  7b)   BASE=Qwen/Qwen2.5-Math-7B;        UAQ_MODEL="$MODELS/qwen2.5-math-7b-uaq$QURL_S" ;;
  1.5b) BASE=Qwen/Qwen2.5-Math-1.5B;      UAQ_MODEL="$MODELS/qwen2.5-math-1.5b-uaq$QURL_S" ;;
  0.5b) BASE=Qwen/Qwen2.5-0.5B-Instruct;  UAQ_MODEL="$MODELS/qwen0.5b-instruct-uaq$QURL_S" ;;
  *) echo "size must be 7b, 1.5b or 0.5b" >&2; exit 1 ;;
esac

# Attention in the trainer. flash-attn's varlen kernel if the package imports; otherwise PyTorch SDPA, which handles
# VeRL's packed batches correctly through transformers' block-diagonal mask (test_packed_attn.py, gate G2) but spends
# attention compute on the whole packed row, so it gets smaller per-GPU token budgets. Budgets are tokens per GPU per
# micro-batch (dynamic batching) and must be >= MAXP + MAXR.
if "$PY" -c "import flash_attn" 2>/dev/null; then ATTN=flash_attention_2; else ATTN=sdpa; fi
case "$SIZE/$ATTN" in
  7b/flash_attention_2)   TT=16384; TL=32768 ;;
  7b/sdpa)                TT=10240; TL=20480 ;;
  1.5b/flash_attention_2) TT=32768; TL=65536 ;;
  1.5b/sdpa)              TT=16384; TL=32768 ;;
  0.5b/*)                 TT=4096;  TL=8192 ;;
esac
TOK_TRAIN=${TOK_TRAIN:-$TT}
TOK_LOGPROB=${TOK_LOGPROB:-$TL}
[ "$TOK_TRAIN" -ge $((MAXP + MAXR)) ] || { echo "TOK_TRAIN $TOK_TRAIN < MAX_PROMPT + MAX_RESPONSE" >&2; exit 1; }

export PYTHONPATH="$S4${PYTHONPATH:+:$PYTHONPATH}"   # S4's qurl_verl (multi-GPU version), not S3's
export QURL_OPT=plain
LR=$BASE_LR
MODEL=$BASE
case "$ARM" in
  R1|R3) ;;
  R2|R4)
    [ -f "$UAQ_MODEL/uaq.json" ] || { echo "no UAQ checkpoint at $UAQ_MODEL (setup_s4.sh writes it)" >&2; exit 1; }
    MODEL=$UAQ_MODEL ;;
  R5) export QURL_OPT=r5 ;;
  R6) LR=$("$PY" -c "print(float('$QURL_S') * float('$BASE_LR'))") ;;
  *) echo "unknown arm $ARM (R1-R6)" >&2; exit 1 ;;
esac
case "$ARM" in R1|R2) INT8=0 ;; *) INT8=1 ;; esac

mkdir -p "$RES" "$CKPT"
if [ -s "$RES/summary.txt" ]; then echo "$TAG: training already finished (summary.txt); use finish.sh for the rest" >&2; exit 0; fi
# Resume: keep the earlier attempt's metrics and log (VeRL's file logger truncates metrics.jsonl when it opens it).
if [ -s "$RES/metrics.jsonl" ]; then
  n=1; while [ -e "$RES/metrics.part$n.jsonl" ]; do n=$((n + 1)); done
  mv "$RES/metrics.jsonl" "$RES/metrics.part$n.jsonl"
  [ -e "$RES/train.log" ] && mv "$RES/train.log" "$RES/train.part$n.log"
  echo "resuming $TAG: earlier attempt kept as metrics.part$n.jsonl; latest checkpoint:" \
       "$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
fi
export QURL_RUN_ATTEMPT=$(date +%s)
# The step this attempt starts from (0, or the checkpoint it resumes); the INT8 refit logs it with every sync, so
# sync k of an attempt is the weight change of step QURL_START_STEP + k - 1.
export QURL_START_STEP=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)

ARGS=(
  # algorithm: GRPO with DAPO's clip-higher, token-mean loss, no KL anywhere, no entropy bonus
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  # data
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.train_batch_size=512
  data.max_prompt_length=$MAXP
  data.max_response_length=$MAXR
  data.truncation=left
  data.filter_overlong_prompts=False
  data.shuffle=True
  data.seed=$SEED
  # reward: DAPO's rule-based score (+1 / -1) with the soft overlong penalty
  reward_model.reward_manager=dapo
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
  +reward_model.reward_kwargs.max_resp_len=$MAXR
  # model
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.external_lib=qurl_verl
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  +actor_rollout_ref.model.override_config.max_position_embeddings=$CTX
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN
  # actor: AdamW (betas 0.9 / 0.999), FP32 master weights, FSDP2 over all GPUs, constant lr after 10 warm-up steps
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=$LR
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOK_TRAIN
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  # rollout: vLLM, one replica per GPU (tensor parallel size 1, required by the INT8 refit), temperature 1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=16
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.6}
  actor_rollout_ref.rollout.max_model_len=$((MAXP + MAXR))
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAXP + MAXR))
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOK_LOGPROB
  actor_rollout_ref.rollout.seed=$((42 + SEED))
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7
  actor_rollout_ref.rollout.val_kwargs.top_k=-1
  # trainer
  trainer.n_gpus_per_node=$NGPU
  trainer.nnodes=1
  trainer.total_epochs=100
  trainer.total_training_steps=$STEPS
  trainer.val_before_train=${VAL_BEFORE:-True}
  trainer.test_freq=$VAL_EVERY
  trainer.save_freq=$SAVE_EVERY
  trainer.max_actor_ckpt_to_keep=1
  trainer.resume_mode=auto
  trainer.default_local_dir="$CKPT"
  trainer.log_val_generations=0
  trainer.logger='["console","file"]'
  trainer.project_name=qurl_s4
  trainer.experiment_name=$TAG
  ray_kwargs.ray_init.runtime_env.py_executable=$PY
)

if [ "$INT8" = 1 ]; then
  export QURL_TIS_C=${QURL_TIS_C:-2.0}
  [ "${E3:-1}" = 1 ] && export QURL_E3_PATH="$RES/e3_int8.jsonl"
  [ "${DUMP:-0}" = 1 ] && export QURL_INT8_DUMP_DIR="$RES/int8_dump"
  HF_JSON=$("$PY" -c 'import json, sys; print(json.dumps({"quantization_config": json.load(open(sys.argv[1])), "max_position_embeddings": int(sys.argv[2])}, separators=(",", ":")))' \
    "$S4/qurl_verl/int8_w8a8_config.json" "$CTX")
  ARGS+=(
    # QuRL's objective: TIS weight min(rho, C) and ACR's upper bound, applied by the qurl_acr loss, which needs VeRL's
    # token-level rho = pi_old / mu untruncated (hence the huge threshold)
    actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=1e30
    # INT8 W8A8 rollout (S1's quantizer config); the worker extension re-quantizes the synced weights at every sync.
    # hf_overrides replaces VeRL's own, so it also carries the raised context length.
    +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='${HF_JSON}'"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl_verl.int8_rollout.QuRLWorkerExtension
  )
else
  HF_JSON=$("$PY" -c 'import json, sys; print(json.dumps({"max_position_embeddings": int(sys.argv[1])}, separators=(",", ":")))' "$CTX")
  ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='${HF_JSON}'")
fi

{
  echo "== S4 $TAG: size $SIZE, arm $ARM, seed $SEED, $STEPS steps, $NGPU GPUs, attention $ATTN $(date -Is)"
  echo "model $MODEL | QURL_OPT=$QURL_OPT | QURL_S=$QURL_S | lr $LR (base $BASE_LR) | INT8 rollout $INT8" \
       "| QURL_TIS_C=${QURL_TIS_C:-n/a} | token budgets train $TOK_TRAIN / log-prob $TOK_LOGPROB | attempt $QURL_RUN_ATTEMPT"
  [ "$ARM" = R2 ] || [ "$ARM" = R4 ] && cat "$UAQ_MODEL/uaq.json"
  true
} | tee -a "$RES/arm.txt"
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  cat "$S4/env.txt" 2>/dev/null || true
  md5sum "$S4"/run.sh "$S4"/qurl_verl/*.py "$S4"/qurl_verl/*.json 2>/dev/null | sed 's/^/code: /'
  printf '%s\n' "command: $PY -m verl.trainer.main_ppo" "${ARGS[@]}" "$@"
} > "$RES/env.txt"

cd "$RES"                                   # Ray/Hydra outputs land here
export VERL_FILE_LOGGER_PATH="$RES/metrics.jsonl"
export PYTHONUNBUFFERED=1
set +e
"$PY" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" > "$RES/train.log" 2>&1
rc=$?
set -e
echo "== training exit $rc $TAG $(date -Is)" | tee -a "$RES/arm.txt"
[ "$INT8" = 1 ] && { cat "$RES"/train.part*.log "$RES/train.log" 2>/dev/null | grep -o "\[qurl int8 refit\].*" > "$RES/int8_refit.txt" || true; }
[ $rc -eq 0 ] || exit $rc
"$PY" "$S4/summarize.py" "$RES" --full-steps "$STEPS" | tee "$RES/summary.txt"
[ "${NO_FINISH:-0}" = 1 ] || bash "$S4/finish.sh" "$SIZE" "$TAG"
