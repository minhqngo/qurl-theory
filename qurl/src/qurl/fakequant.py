"""INT8 W8A8 fake quantization of a Hugging Face model: QuRL's quantized rollout without vLLM.

While int8_rollout(model) is active, every linear layer of the model except lm_head computes
    y = (Q(x) @ Q(W)^T) + b,
with Q the per-row absmax INT8 quantizer of qurl.quant: applied to W it gives channel-wise weight scales, fixed when
the context is entered; applied to x, per token, it gives dynamic token-wise activation scales. This is the scheme of
the paper (Sec. 5) and of vLLM's compressed-tensors INT8 W8A8 kernels. The integer product is computed as an FP32
matrix product of the codes (exact up to FP32 accumulation), then scaled by the token and channel scales. lm_head
stays in full precision, as in the vLLM config (ignore: [lm_head]); embeddings, norms and attention are not quantized,
and neither is the KV cache.

Entering the context quantizes the current weights, so entering it at every rollout re-quantizes the trained weights:
QuRL's theta_hat_old = Q(theta_old) at every RL step. The codes are kept as int8 (a quarter of the FP32 weights'
memory) and converted per call.
"""
from __future__ import annotations

import types

import torch

from .quant import quantize_rows


def _int8_forward(self, x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    codes_x, scale_x = quantize_rows(x.reshape(-1, shape[-1]))      # dynamic, per token
    y = (codes_x.float() @ self._qurl_codes.float().t()) * scale_x * self._qurl_scale
    if self.bias is not None:
        y = y + self.bias.float()
    return y.view(*shape[:-1], -1).to(x.dtype)


class int8_rollout:  # noqa: N801  (used like a function: `with int8_rollout(model):`)
    """Context manager: the model's linear layers (all but those named in `skip`) run INT8 W8A8 while active."""

    def __init__(self, model: torch.nn.Module, skip: tuple[str, ...] = ("lm_head",)):
        self.layers = [m for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] not in skip]
        if not self.layers:
            raise ValueError("int8_rollout: the model has no linear layers to quantize")

    @torch.no_grad()
    def __enter__(self) -> int8_rollout:
        for m in self.layers:
            if "forward" in m.__dict__:
                raise RuntimeError("int8_rollout: a layer is already patched (nested int8_rollout?)")
            codes, scale = quantize_rows(m.weight.detach())
            m._qurl_codes = codes                     # [out, in] int8
            m._qurl_scale = scale.t()                 # [1, out] float32
            m.forward = types.MethodType(_int8_forward, m)
        return self

    def __exit__(self, *exc) -> bool:
        for m in self.layers:
            del m.forward, m._qurl_codes, m._qurl_scale
        return False
