"""QuRL's quantized rollout: vLLM's INT8 W8A8 layers, re-quantized from the trainer's weights at every weight sync.

In QuRL, the old actor is quantized for rollout, theta_hat_old = Q(theta_old), while the trainer keeps full-precision
weights (paper, Sec. 4): after every training step, the new weights are synced to the rollout engine and quantized
again.

vLLM side. The rollout engine is launched as an INT8 W8A8 compressed-tensors model with the config in
int8_w8a8_config.json (quantization_config()): channel-wise symmetric INT8 weights, dynamic per-token symmetric INT8
activations, lm_head kept in BF16; the KV cache is not quantized (as in the paper). With VeRL's dummy load format every
linear layer starts with garbage codes, which the first weight sync overwrites.

Weight sync. In vLLM 0.24, a compressed-tensors W8A8 INT8 layer holds an int8 `weight` and a float32 `weight_scale`
[out, 1]; after process_weights_after_loading, both the CUTLASS (H100) and the Triton (fallback) kernel keep `weight`
as a transposed [in, out] view of the [out, in] codes. Int8Refit.load quantizes each synced q/k/v/o/gate/up/down weight
row by row (qurl.quant.quantize_rows) and writes codes and scales into that shard's rows of the live layer. Per-row
scales make the fused shards (q/k/v in qkv_proj, gate/up in gate_up_proj) independent, so each tensor is handled as it
arrives. Every other tensor (embeddings, norms, biases) is handed back, for VeRL's ordinary load path.

Checks, all of which raise:
  - tensor parallelism: a row-parallel layer (o_proj, down_proj) would hold a slice of every row, and a per-row scale
    needs the whole row. Run the rollout with tensor_model_parallel_size=1: VeRL then starts one vLLM replica per GPU
    and sends each the full weights, so the refit runs unchanged in every replica;
  - completeness: every sync must re-quantize all 7 linear weights of every layer (a weight arriving a second time
    before its sync is complete means the previous sync was not);
  - a synced linear weight without an INT8 layer to receive it (loading BF16 values into int8 parameters would
    silently corrupt the rollout model).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .quant import quantize_rows

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "int8_w8a8_config.json")

LINEAR_WEIGHT = re.compile(r"^(?P<prefix>.*layers\.(?P<layer>\d+)\.)(?P<block>self_attn|mlp)\."
                           r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
"""The trainer-side (Hugging Face) names of the linear weights the refit re-quantizes."""

FUSED = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
         "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
         "o_proj": ("o_proj", None), "down_proj": ("down_proj", None)}
"""Trainer projection -> (vLLM module that holds it, shard index within the fused module, or None)."""

WEIGHTS_PER_LAYER = 7


def quantization_config() -> dict:
    """vLLM's compressed-tensors INT8 W8A8 config (hf_overrides' quantization_config)."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def is_int8_layer(module) -> bool:
    """A vLLM INT8 linear layer: an int8 `weight` with a `weight_scale`."""
    w = getattr(module, "weight", None)
    return w is not None and w.dtype == torch.int8 and getattr(module, "weight_scale", None) is not None


@dataclass(frozen=True)
class LinearTarget:
    """Where a synced linear weight goes: vLLM's (fused) module, the decoder layer, the projection and its shard."""
    module: str
    layer: int
    proj: str
    shard: int | None


def match_linear(name: str) -> LinearTarget | None:
    """The target of a synced weight named `name`, or None if the refit does not handle it."""
    m = LINEAR_WEIGHT.match(name)
    if m is None:
        return None
    fused, shard = FUSED[m["proj"]]
    return LinearTarget(module=m["prefix"] + m["block"] + "." + fused, layer=int(m["layer"]), proj=m["proj"],
                        shard=shard)


class Int8Refit:
    """One rollout replica's refit state, created at its first weight sync.

    `modules` maps vLLM's module names to modules (dict(model.named_modules())). The INT8 layers fix the number of
    linear weights a sync must carry (7 per decoder layer). `tp_world_size` is vLLM's tensor-parallel size; anything
    but 1 is refused.
    """

    def __init__(self, modules: Mapping[str, torch.nn.Module], *, tp_world_size: int = 1):
        if tp_world_size != 1:
            raise RuntimeError(f"[qurl int8 refit] tensor parallelism ({tp_world_size}) is not supported: per-row "
                               f"scales of row-parallel layers need whole rows; run the rollout with "
                               f"tensor_model_parallel_size=1")
        int8 = {n: m for n, m in modules.items() if is_int8_layer(m)}
        self.expected = sum(len(m.logical_widths) for m in int8.values())
        self.layers = sum(1 for n in int8 if n.endswith("self_attn.qkv_proj"))
        if self.expected != WEIGHTS_PER_LAYER * self.layers:
            raise RuntimeError(f"[qurl int8 refit] expected {WEIGHTS_PER_LAYER} INT8 weights per layer, found "
                               f"{self.expected} for {self.layers} layers")
        self.sync = 0
        self.count = 0
        self.info = ""
        self._seen: set[str] = set()

    def load(self, weights: Iterable[tuple[str, torch.Tensor]],
             modules: Mapping[str, torch.nn.Module]) -> list[tuple[str, torch.Tensor]]:
        """Re-quantize the linear weights among `weights` into the INT8 layers of `modules`; return the others."""
        rest = []
        for name, tensor in weights:
            target = match_linear(name)
            if target is None:
                rest.append((name, tensor))
                continue
            # A sync starts with the first linear weight after a complete one (whatever the bucket order). A weight
            # seen twice before its sync is complete means the previous sync was incomplete: refuse.
            if self.sync == 0 or self.count == self.expected:
                self._start_sync()
            elif name in self._seen:
                raise RuntimeError(f"[qurl int8 refit] sync {self.sync} re-quantized {self.count} of "
                                   f"{self.expected} linear weights before {name!r} arrived again")
            self._seen.add(name)
            layer = modules.get(target.module)
            if layer is None or not is_int8_layer(layer):
                # Loading BF16 values into int8 parameters would silently corrupt the rollout model: refuse.
                raise RuntimeError(f"[qurl int8 refit] no INT8 layer {target.module!r} for synced weight {name!r}; "
                                   f"is vLLM running the compressed-tensors INT8 W8A8 config?")
            self._write(name, target, layer, tensor)
            if self.count == self.expected:
                self._finish_sync()
        return rest

    def _start_sync(self) -> None:
        self.sync += 1
        self.count = 0
        self._seen = set()

    def _finish_sync(self) -> None:
        """All weights of this sync arrived: print the replica's line."""
        print(f"[qurl int8 refit] pid {os.getpid()} sync {self.sync}: {self.count} linear weights "
              f"re-quantized to INT8{self.info}", flush=True)

    def _write(self, name: str, target: LinearTarget, layer: torch.nn.Module, tensor: torch.Tensor) -> None:
        codes_t = layer.weight.data            # [in, out] view of the [out, in] codes (CUTLASS and Triton)
        out_total = sum(layer.logical_widths)
        assert codes_t.shape == (tensor.shape[1], out_total), (name, tuple(codes_t.shape), tuple(tensor.shape))
        r0 = sum(layer.logical_widths[:target.shard]) if target.shard is not None else 0
        r1 = r0 + tensor.shape[0]
        if not self.info:   # what the rollout actually runs: the synced dtype and vLLM's INT8 GEMM kernel
            scheme = getattr(layer, "scheme", None)          # compressed-tensors attaches the scheme to the layer
            kernel = type(getattr(scheme, "kernel", None)).__name__
            self.info = f" (synced weights arrive as {tensor.dtype}; INT8 kernel {kernel})"
        codes, scale = quantize_rows(tensor.to(codes_t.device))
        codes_t.t()[r0:r1].copy_(codes)
        layer.weight_scale.data[r0:r1].copy_(scale)
        self.count += 1
