"""Per-row absmax INT8 quantization of QuRL's rollout weights.

QuRL quantizes the old actor's weights with channel-wise scales and its activations with token-wise scales (paper,
Sec. 5); vLLM's INT8 W8A8 kernel quantizes the activations dynamically, per token, and quantize_rows produces the
weights it multiplies them with: symmetric, one scale per output row (channel), scale = absmax / 127, codes =
round-half-even(W / scale) clamped to +-127.

The quantizer is homogeneous: rescaling a row by c > 0 rescales its scale by c and leaves its codes unchanged, exactly
when c is a power of two and up to FP32 rounding otherwise. The theory note (latex/p1_uaq_and_update_tracking.tex,
Theorem 1) works out what this means for UAQ.
"""
from __future__ import annotations

import torch

QMAX = 127.0
"""Largest INT8 code magnitude: codes lie in [-127, 127]."""


@torch.no_grad()
def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row absmax INT8: returns int8 codes [rows, in] and float32 scales [rows, 1].

    The scale is absmax * fl32(1/127), written as a multiplication on purpose: PyTorch divides a CUDA tensor by a
    Python scalar through its reciprocal but a CPU tensor by true division, and the two differ by one ulp in ~5% of
    rows; the multiplication gives the same scales on both devices. The codes are W / scale in FP32, rounded half to
    even and clamped to +-127. A row of zeros gets scale 1 (and zero codes).
    """
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(w / scale).clamp_(-QMAX, QMAX).to(torch.int8)
    return codes, scale
