"""QuRL's VeRL plugin (VeRL v0.9.1): actor_rollout_ref.model.external_lib=qurl.verl.

VeRL imports the plugin in its training workers. Importing it
  - registers the TIS + ACR policy loss as loss_mode=qurl_acr (qurl.verl.loss), and
  - when QURL_OPT is set, patches VeRL's FSDP2 actor optimizer with Theorem 2(b)-(c)'s per-group settings and clipping
    rule (c) (qurl.verl.optim; arm R5 with QURL_OPT=r5, the same code path with plain settings for QURL_OPT=plain).
The INT8 rollout is qurl.verl.rollout.QuRLWorkerExtension, which vLLM imports by name in its worker processes
(engine_kwargs.vllm.worker_extension_cls); importing it runs this file first, as for any submodule.
"""
from . import loss  # noqa: F401  (registers loss_mode=qurl_acr)
from . import optim

optim.apply_if_enabled()   # patches VeRL's FSDP actor optimizer when QURL_OPT is set
