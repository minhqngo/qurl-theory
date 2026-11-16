"""Per-row absmax INT8 quantization, as in QuRL's W8A8 rollout, and the weights that sit on a rounding boundary.

The quantizer is the one of Theorem 1 in the theory note (latex/p1_uaq_and_update_tracking.tex, section 3.1):
symmetric, one scale per output row, scale = absmax / 127, codes = round-half-even(W / scale) clamped to +-127. It is
homogeneous: rescaling a row by c > 0 rescales its scale by c and leaves its codes unchanged, exactly when c is a power
of two and up to FP32 rounding otherwise. UAQ's W / s therefore leaves the INT8 rollout unchanged.

Ties. A BF16 checkpoint stores weights on a lattice, and about 0.1-0.3% of them land exactly on a rounding boundary
(W / scale = k + 1/2; Remark "BF16 storage" in the note). Any update moves such a weight off the boundary, so about
half of them flip at the first update, whatever its size, and they keep re-crossing afterwards. tie_indices finds them,
to TIE_TOL grid steps, so that the refit's log can count their flips separately.
"""
from __future__ import annotations

import torch

QMAX = 127.0
"""Largest INT8 code magnitude: codes lie in [-127, 127]."""

TIE_TOL = 1e-4
"""Grid steps: a weight this close to a rounding boundary at the run's start is a tie."""


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


@torch.no_grad()
def tie_indices(w: torch.Tensor, scale: torch.Tensor, tol: float = TIE_TOL) -> torch.Tensor:
    """Flat (row-major) indices of the weights within `tol` grid steps of a rounding boundary.

    `scale` is the row scale [rows, 1] from quantize_rows(w); W / scale is computed in FP32 as there, so an exact tie
    of a BF16 tensor (W / scale = k + 1/2) is found whatever the FP32 rounding of its scale. Returns an int64 tensor on
    w's device.
    """
    q = w.float() / scale
    q.sub_(torch.round(q)).abs_()
    return torch.nonzero(q.view(-1) >= 0.5 - tol).view(-1)
