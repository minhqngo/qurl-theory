"""Theorem 2(b)-(c) without UAQ (arm R5): per-group AdamW settings and clipping rule (c), on any number of ranks.

Theorem 2 of the theory note (latex/p1_uaq_and_update_tracking.tex, section 3.2): training the UAQ model
(W' = W / s, gamma' = s * gamma) with learning rate eta, Adam constant eps and decoupled weight decay wd is training the
original model with, per parameter group,
    q/k/v_proj.weight (W):    lr s*eta,  Adam eps eps/s,  weight decay wd/s
    input_layernorm.weight:   lr eta/s,  Adam eps s*eps,  weight decay s*wd
    everything else:          lr eta,    eps,             wd
and (c) the global clip norm computed in UAQ coordinates, on (s*g_W, g_gamma/s, g_rest). param_groups and clip_factors
build these; s = 1 gives the plain settings and the ordinary norm through the same code path. A learning-rate schedule
multiplies every group's rate by the same factor, so the ratios between groups, and Theorem 2, hold at every step.

Exactness at s = 2: every rescaling by a power of two is exact in floating point, so R4 (the UAQ checkpoint, plain
settings) and R5 (the original checkpoint, these settings) give bit-identical weights after mapping R4's back
(to_original_coordinates), on any number of ranks.

Several ranks: each rank sums the squared, factor-weighted norms of its local gradient shards in float64; the per-rank
sums are all-gathered over the shard group and added in rank order. Every rank gets the same total, bit for bit, and so
do R4 and R5 at s = 2. Every gradient must then be a DTensor sharded on exactly one mesh dimension (FSDP2's Shard(0));
any other mesh dimension must be Replicate (HSDP) and is not summed over. Anything else raises instead of clipping
with a wrong norm.

Environment (read in VeRL's training workers through qurl.verl; see qurl.verl.optim):
  QURL_OPT=r5      these settings for s = QURL_S (default 1.5)
  QURL_OPT=plain   the same code path with s = 1 (the arms R5 is compared with, so every arm clips identically)
  QURL_CLIP=ordinary  R5's groups with the ordinary norm (the control R5o)
  unset            VeRL's own optimizer and clip
"""
from __future__ import annotations

import os
import re

import torch

_W = re.compile(r"self_attn\.(q|k|v)_proj\.weight$")
_GAMMA = re.compile(r"input_layernorm\.weight$")


def settings() -> tuple[str, float] | None:
    """(mode, s) from QURL_OPT and QURL_S, or None when the plugin optimizer is off."""
    mode = os.environ.get("QURL_OPT")
    if not mode:
        return None
    if mode not in ("plain", "r5"):
        raise ValueError(f"QURL_OPT must be 'plain' or 'r5', got {mode!r}")
    return mode, (float(os.environ.get("QURL_S", "1.5")) if mode == "r5" else 1.0)


def group_of(name: str) -> str:
    """The parameter group of a parameter name: "W" (q/k/v_proj.weight), "gamma" (input_layernorm.weight), "rest"."""
    if _W.search(name):
        return "W"
    if _GAMMA.search(name):
        return "gamma"
    return "rest"


def param_groups(named, lr: float, eps: float, wd: float, s: float):
    """Theorem 2(b)'s optimizer groups for scale s, and the number of tensors per group.

    `named` is a sequence of (name, parameter). "rest" comes first: VeRL logs param_groups[0]'s rate as actor/lr.
    """
    by = {"W": [], "gamma": [], "rest": []}
    for n, p in named:
        by[group_of(n)].append(p)
    if not by["W"] or not by["gamma"]:
        raise RuntimeError("QURL_OPT: no q/k/v_proj or input_layernorm parameters found")
    groups = [
        dict(params=by["rest"], lr=lr, eps=eps, weight_decay=wd),
        dict(params=by["W"], lr=s * lr, eps=eps / s, weight_decay=wd / s),
        dict(params=by["gamma"], lr=lr / s, eps=s * eps, weight_decay=s * wd),
    ]
    return groups, {k: len(v) for k, v in by.items()}


def clip_factors(named, s: float):
    """(parameter, factor) in parameter order: the clip norm is that of (s*g_W, g_gamma/s, g_rest); s = 1: ordinary."""
    return [(p, s if group_of(n) == "W" else 1.0 / s if group_of(n) == "gamma" else 1.0) for n, p in named]


def to_original_coordinates(name: str, tensor: torch.Tensor, s: float) -> torch.Tensor:
    """Map a parameter of the UAQ model back to the original coordinates: W = s * W', gamma = gamma' / s, rest as is."""
    g = group_of(name)
    return tensor * s if g == "W" else tensor / s if g == "gamma" else tensor


def _shard_group(g):
    """The process group a gradient is sharded over, or None for a plain (unsharded) tensor."""
    from torch.distributed.tensor import DTensor

    if not isinstance(g, DTensor):
        return None
    shard = [d for d, pl in enumerate(g.placements) if pl.is_shard()]
    odd = [pl for pl in g.placements if not (pl.is_shard() or pl.is_replicate())]
    if len(shard) != 1 or odd:
        raise RuntimeError(f"QURL_OPT: gradient placements {g.placements} are not supported; expected one Shard "
                           f"dimension (FSDP2) and Replicate on any other mesh dimension")
    return g.device_mesh.get_group(mesh_dim=shard[0])


def total_norm(clip):
    """Global 2-norm of (f * g) over all (parameter, f) in clip, identical on every rank; None if no gradients.

    Each rank adds (f * ||local shard||)^2 over its tensors in float64; the per-rank sums are all-gathered over the
    shard group and added in rank order (a fixed order, so the result does not depend on the collective's algorithm).
    """
    import torch.distributed as dist

    multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    sq, group = None, None
    for p, f in clip:
        g = p.grad
        if g is None:
            continue
        grp = _shard_group(g)
        if grp is None:
            if multi:
                raise RuntimeError("QURL_OPT: a gradient that is not an FSDP2 DTensor on several ranks; its norm "
                                   "would be counted once per rank")
            local = g
        else:
            if group is None:
                group = grp
            elif grp is not group:
                raise RuntimeError("QURL_OPT: gradients are sharded over different process groups")
            local = g.to_local()
        n = torch.linalg.vector_norm(local.float()).double() * f
        sq = n * n if sq is None else sq + n * n
    if sq is None:
        return None
    sq = sq.reshape(1)
    if group is not None and dist.get_world_size(group) > 1:
        parts = [torch.empty_like(sq) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, sq, group=group)
        sq = parts[0]
        for x in parts[1:]:
            sq = sq + x
    return torch.sqrt(sq[0])


def clip_and_step(optimizer, clip, max_norm: float) -> float:
    """Clip every gradient by min(1, max_norm / (total + 1e-6)), then step; skip the step if the norm is not finite.

    Returns the norm before clipping, as VeRL's optimizer_step does.
    """
    total = total_norm(clip)
    if total is None:
        optimizer.step()
        return 0.0
    if not torch.isfinite(total):
        print(f"WARN: grad_norm is not finite: {total}")
        optimizer.zero_grad()
        return float(total)
    coef = float(torch.clamp(max_norm / (total + 1e-6), max=1.0))
    for p, _ in clip:
        if p.grad is not None:
            p.grad.mul_(coef)   # the same coefficient on every rank and, at s = 2, in R4 and R5
    optimizer.step()
    return float(total)
