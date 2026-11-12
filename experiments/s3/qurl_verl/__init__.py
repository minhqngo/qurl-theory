"""QuRL components for VeRL v0.9.1 (S3, P1 note section 7).

  acr           loss_mode=qurl_acr: TIS weight min(rho, C) and ACR's upper clip bound (QuRL Eq. 9). Loaded into the
                training workers through `actor_rollout_ref.model.external_lib=qurl_verl` (~/qurl/s3 on PYTHONPATH).
  int8_rollout  QuRLWorkerExtension: INT8 W8A8 rollout re-quantized at every weight sync. vLLM imports it by name
                in its worker processes (engine_kwargs.vllm.worker_extension_cls), so it is not imported here.
  r5_optim      QURL_OPT=r5|plain: Theorem 2(b)-(c) per-group optimizer settings and clipping rule (c) for the FSDP
                actor (R5), or the same code path with plain settings (the arms R5 is compared with).
"""
from . import acr  # noqa: F401  (registers the policy loss)
from . import r5_optim  # noqa: F401  (patches VeRL's FSDP actor optimizer when QURL_OPT is set)
