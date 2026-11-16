"""QuRL for VeRL: the training algorithm of "QuRL: Efficient Reinforcement Learning with Quantized Rollout".

QuRL (Li et al., ICLR 2026, arXiv:2602.13953) accelerates RL by rolling out with a quantized copy of the actor:
  - the old actor's weights and activations are quantized for rollout (here INT8 W8A8 in vLLM), re-quantized from the
    trainer's full-precision weights at every weight sync;
  - the policy loss is the decoupled PPO objective with the quantized actor as the behavior policy and the
    full-precision old actor as the proximal policy, weighted by truncated importance sampling (TIS) and with the
    adaptive clipping range (ACR);
  - Update-Aware Quantization (UAQ) rescales W_{Q,K,V} by 1/s and the input-layernorm gain by s once, before training.

Modules (torch only; qurl.uaq's command line also needs transformers):
  qurl.quant      per-row absmax INT8 quantizer of the rollout weights
  qurl.refit      the INT8 refit of vLLM's layers at every weight sync, with its checks
  qurl.loss       the TIS weight and ACR clip bound, as per-token PPO losses
  qurl.uaq        UAQ checkpoints, with an exact s = 2 probe
  qurl.overrides  the Hydra overrides that wire the package into VeRL
VeRL integration (VeRL v0.9.1, vLLM 0.24): qurl.verl, the trainer plugin (`actor_rollout_ref.model.external_lib`),
and qurl.verl.rollout.QuRLWorkerExtension, vLLM's worker extension.
"""
__version__ = "0.2.0"
