#!/usr/bin/env bash
# S3's UAQ experiment (P1 note, section 7): arms R1-R6 on QuRL's GSM8K setup, through pilot.sh.
#   [QURL_S=1.5] [BASE_LR=1e-6] bash ~/qurl/s3/arm.sh <arm> <tag> [steps] [extra Hydra overrides ...]
#   e.g. bash ~/qurl/s3/arm.sh R5 r5_s1 435 trainer.test_freq=29 data.seed=1
# Every arm trains the actor with FSDP2 and the qurl_verl optimizer path (QURL_OPT), so the arms run identical code
# and differ only in what the arm changes:
#   R1  BF16 rollout
#   R2  BF16 rollout, UAQ checkpoint (W_QKV/s, gamma*s; make_uaq_checkpoint.py)
#   R3  QuRL: INT8 W8A8 rollout re-quantized at every sync, token-level TIS (C = QURL_TIS_C, default 2), ACR
#   R4  R3 with the UAQ checkpoint
#   R5  R3 without UAQ, with Theorem 2(b)-(c)'s per-group optimizer settings and clipping rule (c) for s
#   R6  R3 with the global learning rate x s (the paper's control)
# s = QURL_S (default 1.5, the paper's). The base actor learning rate is BASE_LR (default 1e-6); R6 uses s*BASE_LR,
# so change it through BASE_LR, not through an lr override.
# pilot.sh's own ARM=R1/R3 runs (the Table 1 check) use VeRL's default optimizer and FSDP1 instead.
set -euo pipefail
ARM=$1; TAG=$2; shift 2
STEPS=${1:-10}; [ $# -gt 0 ] && shift
S3="$HOME/qurl/s3"
PY="$HOME/venvs/verl/bin/python"
export QURL_S=${QURL_S:-1.5}
BASE_LR=${BASE_LR:-1e-6}
UAQ_MODEL=${UAQ_MODEL:-$HOME/qurl/models/qwen0.5b-instruct-uaq$QURL_S}
export PYTHONPATH="$S3${PYTHONPATH:+:$PYTHONPATH}"
export QURL_OPT=plain

EXTRA=(actor_rollout_ref.model.external_lib=qurl_verl actor_rollout_ref.actor.strategy=fsdp2
       actor_rollout_ref.actor.optim.lr=$BASE_LR)
case "$ARM" in
  R1|R3) ;;
  R2|R4)
    [ -f "$UAQ_MODEL/uaq.json" ] || { echo "no UAQ checkpoint at $UAQ_MODEL: run make_uaq_checkpoint.py --s $QURL_S" >&2; exit 1; }
    EXTRA+=(actor_rollout_ref.model.path="$UAQ_MODEL") ;;
  R5) export QURL_OPT=r5 ;;
  R6) EXTRA+=(actor_rollout_ref.actor.optim.lr=$("$PY" -c "print(float('$QURL_S') * float('$BASE_LR'))")) ;;
  *) echo "unknown arm $ARM (R1-R6)" >&2; exit 1 ;;
esac
case "$ARM" in R1|R2) ROLLOUT=R1 ;; *) ROLLOUT=R3 ;; esac

mkdir -p "$S3/results/$TAG"
{
  echo "arm $ARM (rollout: pilot.sh ARM=$ROLLOUT) | QURL_OPT=$QURL_OPT | QURL_S=$QURL_S | BASE_LR=$BASE_LR"
  [ "$ARM" = R2 ] || [ "$ARM" = R4 ] && cat "$UAQ_MODEL/uaq.json"
  printf 'arm overrides: %s\n' "${EXTRA[*]}"
} > "$S3/results/$TAG/arm.txt" || true
ARM=$ROLLOUT bash "$S3/pilot.sh" "$TAG" "$STEPS" "${EXTRA[@]}" "$@"
