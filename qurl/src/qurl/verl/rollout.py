"""QuRL's INT8 W8A8 rollout in VeRL v0.9.1 + vLLM 0.24: a colocated vLLM worker extension with the INT8 refit.

Hydra overrides (qurl.overrides builds them):
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=<{quantization_config: ..., max_position_embeddings: ...}>
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl.verl.rollout.QuRLWorkerExtension
With tensor_model_parallel_size=1, VeRL starts one vLLM replica per GPU and sends each the full weights, so the refit
(qurl.refit.Int8Refit) runs unchanged in every replica. The refit's optional outputs come from the environment
(qurl.refit.RefitConfig.from_env, read at the first sync): QURL_E3_PATH (per-sync log), QURL_INT8_DUMP_DIR (test
dump), QURL_RUN_ATTEMPT and QURL_START_STEP.
"""
from __future__ import annotations

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

from ..refit import Int8Refit, RefitConfig


def tp_world_size() -> int:
    """vLLM's tensor-parallel world size (1 when vLLM's distributed state is not initialized)."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:   # noqa: BLE001  (not initialized: a single-process engine)
        return 1


class QuRLWorkerExtension(vLLMColocateWorkerExtension):
    """VeRL's colocated vLLM worker extension with an INT8 W8A8 refit path."""

    def update_weights_from_ipc(self, *args, **kwargs):
        # After the last bucket, VeRL re-runs vLLM's process_weights_after_loading on the whole model (meant for
        # unquantized models, where it is harmless). For the INT8 model it is wrong twice over: it would transpose
        # the int8 weights a second time, and it fails on the attention layers' KV-scale attributes, which the first
        # pass already consumed. Nothing needs re-processing here: the refit writes codes and scales in the kernel's
        # layout, and every other synced tensor (embeddings, norms, biases) is loaded in place. So the call is made a
        # no-op for the duration of the sync.
        import vllm.model_executor.model_loader.utils as loader_utils

        original = loader_utils.process_weights_after_loading
        loader_utils.process_weights_after_loading = lambda *a, **k: None
        try:
            return super().update_weights_from_ipc(*args, **kwargs)
        finally:
            loader_utils.process_weights_after_loading = original

    def _update_weights(self, weights, peft_config, base_sync_done):
        modules = dict(self.model_runner.model.named_modules())
        refit = getattr(self, "_qurl_refit", None)
        if refit is None:   # per-replica state, created at the first sync
            refit = Int8Refit(modules, tp_world_size=tp_world_size(), config=RefitConfig.from_env())
            self._qurl_refit = refit
        rest = refit.load(weights, modules)
        if rest:
            super()._update_weights(rest, peft_config=peft_config, base_sync_done=base_sync_done)
