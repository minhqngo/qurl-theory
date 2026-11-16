"""QuRL's VeRL plugin (VeRL v0.9.1): actor_rollout_ref.model.external_lib=qurl.verl.

VeRL imports the plugin in its training workers; importing it registers QuRL's policy loss, TIS with the adaptive
clipping range, as loss_mode=qurl_acr (qurl.verl.loss). The quantized rollout is qurl.verl.rollout.QuRLWorkerExtension,
which vLLM imports by name in its worker processes (engine_kwargs.vllm.worker_extension_cls); importing it runs this
file first, as for any submodule.
"""
from . import loss  # noqa: F401  (registers loss_mode=qurl_acr)
