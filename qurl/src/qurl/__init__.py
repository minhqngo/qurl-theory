"""QuRL in plain PyTorch: the training algorithm of "QuRL: Efficient Reinforcement Learning with Quantized Rollout".

QuRL (Li et al., ICLR 2026, arXiv:2602.13953) trains with rollouts from a quantized copy of the actor:
  - the old actor's weights and activations are quantized for rollout (here INT8 W8A8, simulated), re-quantized from
    the trainer's full-precision weights at every RL step;
  - the policy loss is the decoupled PPO objective with the quantized actor as the behavior policy and the
    full-precision old actor as the proximal policy, weighted by truncated importance sampling (TIS) and with the
    adaptive clipping range (ACR);
  - Update-Aware Quantization (UAQ) rescales W_{Q,K,V} by 1/s and the input-layernorm gain by s once, before training.

Modules (torch; the loop, the samplers and the UAQ command line also need transformers):
  qurl.quant      per-row absmax INT8 quantizer
  qurl.fakequant  INT8 W8A8 fake quantization of a model's linear layers: the quantized rollout
  qurl.sampling   batched sampling that records the behavior policy's log-probabilities; the training forward
  qurl.loss       the TIS weight and ACR clip bound, as per-token PPO losses
  qurl.uaq        UAQ, with an exact s = 2 probe; command line for UAQ checkpoints
  qurl.gsm8k      GSM8K problems, prompt and exact-match reward
  qurl.train      the train + validation loop (GRPO, decoupled PPO with TIS + ACR, optional UAQ): python -m qurl.train
"""
__version__ = "0.3.0"
