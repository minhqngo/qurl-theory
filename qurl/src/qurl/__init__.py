"""QuRL for VeRL: the INT8 W8A8 rollout refit, the TIS/ACR loss, UAQ checkpoints and the R5 optimizer.

QuRL (Li et al., ICLR 2026) runs RL with an INT8 W8A8 rollout model in vLLM, re-quantized from the trainer's weights at
every weight sync, corrects the rollout-trainer mismatch with token-level truncated importance sampling (TIS) and an
adaptive clipping range (ACR), and proposes Update-Aware Quantization (UAQ). The theory note
(latex/p1_uaq_and_update_tracking.tex) shows that UAQ leaves the INT8 rollout unchanged (Theorem 1) and equals a
per-layer learning-rate, Adam-epsilon and weight-decay change under AdamW (Theorem 2), which arm R5 implements.

Framework-free modules (torch only; qurl.uaq's command line also needs transformers):
  qurl.quant      per-row absmax INT8 quantizer, and the weights on a rounding boundary (ties)
  qurl.refit      the INT8 refit of vLLM's layers at every weight sync, with its completeness checks
  qurl.synclog    the refit's per-sync log: code flips, row-scale changes, quantization error, ties
  qurl.loss       QuRL's TIS weight and ACR clip bound, as per-token PPO losses
  qurl.r5         Theorem 2(b)-(c): per-group optimizer settings and clipping rule (c) (arm R5)
  qurl.uaq        UAQ checkpoints, with the exact s = 2 probe
  qurl.overrides  the Hydra overrides that wire the package into VeRL
VeRL integration (VeRL v0.9.1, vLLM 0.24): qurl.verl, the trainer plugin (`actor_rollout_ref.model.external_lib`),
and qurl.verl.rollout.QuRLWorkerExtension, vLLM's worker extension.
"""
__version__ = "0.1.0"
