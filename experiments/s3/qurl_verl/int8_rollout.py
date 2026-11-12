"""QuRL's INT8 W8A8 rollout for VeRL v0.9.1 + vLLM 0.24, re-quantized at every weight sync (arms R3-R6).

DRAFT: not yet run. Test plan: laptop smoke run with the overrides below, then compare the refitted INT8 codes
against an offline quantization of the synced weights (they must be identical).

vLLM side. The rollout engine is launched as an INT8 W8A8 compressed-tensors model, with the same quantizer as
S1's llm-compressor checkpoints (int8_w8a8_config.json: channel-wise symmetric weights, dynamic per-token symmetric
activations, lm_head kept in BF16). VeRL's dummy load format means every linear layer starts with garbage codes,
which the first weight sync overwrites. Hydra overrides (pilot.sh adds them for the INT8 arms):
  +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=<{quantization_config: int8_w8a8_config.json}>
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl_verl.int8_rollout.QuRLWorkerExtension

Weight sync. In vLLM 0.24, a CompressedTensorsW8A8Int8 layer holds an int8 `weight` and a float32 `weight_scale`
[out, 1]; after process_weights_after_loading both the CUTLASS (H100) and the Triton (fallback) kernel keep `weight`
as a transposed [in, out] view of the [out, in] codes. QuRLWorkerExtension._update_weights quantizes each synced
q/k/v/o/gate/up/down weight row by row (scale = row absmax / 127, codes = round(W / scale) clamped to +-127: the
quantizer of Theorem 1 and S2) and writes codes and scales into that shard's rows of the live layer. Per-row scales
make the fused shards (q/k/v in qkv_proj, gate/up in gate_up_proj) independent, so each tensor can be handled as it
arrives. Everything else (embeddings, norms, biases) goes through VeRL's ordinary load path.
"""
import json
import os
import re

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

QMAX = 127.0
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "int8_w8a8_config.json")
_LINEAR = re.compile(r"^(?P<prefix>.*layers\.\d+\.)(?P<block>self_attn|mlp)\."
                     r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
_FUSED = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
          "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
          "o_proj": ("o_proj", None), "down_proj": ("down_proj", None)}


def quantization_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


@torch.no_grad()
def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row absmax INT8: returns int8 codes [rows, in] and float32 scales [rows, 1].

    The scale is absmax * fl32(1/127), written as a multiplication on purpose: PyTorch divides a CUDA tensor by a
    Python scalar through its reciprocal but a CPU tensor by true division, and the two differ by one ulp in ~5% of
    rows. Either way the quantizer is absmax-based and exactly homogeneous under power-of-two rescaling (Theorem 1).
    """
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(w / scale).clamp_(-QMAX, QMAX).to(torch.int8)
    return codes, scale


def _is_int8_layer(module) -> bool:
    w = getattr(module, "weight", None)
    return w is not None and w.dtype == torch.int8 and getattr(module, "weight_scale", None) is not None


DUMP_DIR = os.environ.get("QURL_INT8_DUMP_DIR")   # test hook: dump layer 0's fused QKV codes at the first syncs
DUMP_MODULE_SUFFIX = "layers.0.self_attn.qkv_proj"
DUMP_SYNCS = 2


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
        model = self.model_runner.model
        modules = dict(model.named_modules())
        rest = []
        n_int8 = 0
        info = ""
        for name, tensor in weights:
            m = _LINEAR.match(name)
            if m is None:
                rest.append((name, tensor))
                continue
            if name.endswith("layers.0.self_attn.q_proj.weight"):   # every sync carries this tensor exactly once
                self._qurl_sync = getattr(self, "_qurl_sync", 0) + 1
                self._qurl_dumped_shards = set()
            fused, shard = _FUSED[m["proj"]]
            module_name = m["prefix"] + m["block"] + "." + fused
            layer = modules.get(module_name)
            if layer is None or not _is_int8_layer(layer):
                # Loading BF16 values into int8 parameters would silently corrupt the rollout model: refuse.
                raise RuntimeError(f"[qurl int8 refit] no INT8 layer {module_name!r} for synced weight {name!r}; "
                                   f"is vLLM running the compressed-tensors INT8 W8A8 config?")
            codes_t = layer.weight.data            # [in, out] view of the [out, in] codes (CUTLASS and Triton)
            out_total = sum(layer.logical_widths)
            assert codes_t.shape == (tensor.shape[1], out_total), (name, tuple(codes_t.shape), tuple(tensor.shape))
            r0 = sum(layer.logical_widths[:shard]) if shard is not None else 0
            r1 = r0 + tensor.shape[0]
            if not info:   # what the rollout actually runs: the synced dtype and vLLM's INT8 GEMM kernel
                scheme = getattr(layer, "scheme", None)          # compressed-tensors attaches the scheme to the layer
                kernel = type(getattr(scheme, "kernel", None)).__name__
                info = f" (synced weights arrive as {tensor.dtype}; INT8 kernel {kernel})"
            codes, scale = quantize_rows(tensor.to(codes_t.device))
            codes_t.t()[r0:r1].copy_(codes)
            layer.weight_scale.data[r0:r1].copy_(scale)
            n_int8 += 1
            if DUMP_DIR and module_name.endswith(DUMP_MODULE_SUFFIX) and getattr(self, "_qurl_sync", 0) <= DUMP_SYNCS:
                self._qurl_dumped_shards.add(shard)
                if len(self._qurl_dumped_shards) == len(layer.logical_widths):
                    os.makedirs(DUMP_DIR, exist_ok=True)
                    torch.save({"codes": codes_t.t().cpu().clone(), "scale": layer.weight_scale.data.cpu().clone(),
                                "logical_widths": list(layer.logical_widths), "module": module_name},
                               os.path.join(DUMP_DIR, f"sync{self._qurl_sync}_qkv_layer0.pt"))
        # One line per received bucket; over a full sync the INT8 counts must add up to 7 per decoder layer.
        print(f"[qurl int8 refit] sync {getattr(self, '_qurl_sync', 0)}: {n_int8} linear weights re-quantized "
              f"to INT8, {len(rest)} other tensors via VeRL's load path{info}", flush=True)
        if rest:
            super()._update_weights(rest, peft_config=peft_config, base_sync_done=base_sync_done)
