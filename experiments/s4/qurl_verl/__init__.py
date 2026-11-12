"""QuRL components for VeRL v0.9.1, multi-GPU version for S4 (experiments/s4/README.md).

  acr           loss_mode=qurl_acr: TIS weight min(rho, C) and ACR's upper clip bound (QuRL Eq. 9). Unchanged from S3.
                Loaded into the training workers through `actor_rollout_ref.model.external_lib=qurl_verl`
                (~/qurl/s4 on PYTHONPATH).
  int8_rollout  QuRLWorkerExtension: INT8 W8A8 rollout re-quantized at every weight sync, one replica per GPU
                (tensor parallelism refused), with optional per-sync flip / error logging. vLLM imports it by name in
                its worker processes (engine_kwargs.vllm.worker_extension_cls), so it is not imported here.
  r5_optim      QURL_OPT=r5|plain: Theorem 2(b)-(c) per-group optimizer settings and clipping rule (c) for the FSDP2
                actor, with the clip norm reduced over all GPUs.
"""
from . import acr  # noqa: F401  (registers the policy loss)
from . import r5_optim  # noqa: F401  (patches VeRL's FSDP actor optimizer when QURL_OPT is set)
