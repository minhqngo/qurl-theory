"""Per-group optimizer settings and clipping rule (c) for VeRL v0.9.1's FSDP2 actor, on one or more GPUs (S4).

S4 version of experiments/s3/qurl_verl/r5_optim.py (arm R5, P1 note Theorem 2(b)-(c)). What changed for several GPUs:
  - S3's version took the clip norm from each rank's local gradient shards and refused to run on more than one rank.
    Here each rank sums the squared, factor-weighted norms of its local shards in float64; the per-rank sums are
    all-gathered over the FSDP shard group and added in rank order. Every rank gets the same total, bit for bit, and
    R4 and R5 get the same total at s = 2 (a power of two rescales gradients, norms and squares exactly).
  - Every gradient must be a DTensor sharded on exactly one mesh dimension (FSDP2's Shard(0)); any other mesh
    dimension must be Replicate (HSDP) and is not summed over. Anything else raises instead of clipping with a wrong
    norm.
  - The arithmetic lives in plain functions (param_groups, clip_factors, total_norm, clip_and_step), which
    test_clip_mp.py runs on the CPU with several processes.
Unchanged from S3: the environment variables, the groups and their settings, FSDP2 only, the actor only.

Environment variables, read in the training workers (qurl_verl is imported there via
actor_rollout_ref.model.external_lib=qurl_verl):
  QURL_OPT=r5     Theorem 2(b)-(c) settings for s = QURL_S (default 1.5), without UAQ:
                    q/k/v_proj.weight (W):   lr s*eta,  Adam eps eps/s,  weight decay wd/s
                    input_layernorm.weight:  lr eta/s,  Adam eps s*eps,  weight decay s*wd
                    everything else:         lr eta,    eps,             wd
                  and the global clip norm computed in UAQ coordinates, i.e. on (s*g_W, g_gamma/s, g_rest).
  QURL_OPT=plain  the same code path with s = 1: plain settings, ordinary global-norm clipping computed here. Used by
                  the arms R5 is compared with (R3, R4, R6), so that every arm runs identical clipping arithmetic.
  QURL_CLIP=ordinary  R5's groups with the ordinary norm (control R5o; r5_check_mp.py only).
  unset           VeRL's own optimizer and clip.
A learning-rate schedule (the DAPO recipe warms up over 10 steps) multiplies every group's rate by the same factor,
so the ratios between groups, and Theorem 2, hold at every step.

Exactness at s = 2: with QURL_OPT=plain on the UAQ checkpoint (R4) and QURL_OPT=r5 on the original checkpoint (R5),
actor steps on the same batch must give bit-identical weights after mapping R4's back (W = s*W', gamma = gamma'/s),
on any number of ranks. r5_check_mp.py tests exactly that.
"""
import os
import re

import torch

_W = re.compile(r"self_attn\.(q|k|v)_proj\.weight$")
_GAMMA = re.compile(r"input_layernorm\.weight$")


def settings():
    """(mode, s) from the environment, or None when the plugin optimizer is off."""
    mode = os.environ.get("QURL_OPT")
    if not mode:
        return None
    if mode not in ("plain", "r5"):
        raise ValueError(f"QURL_OPT must be 'plain' or 'r5', got {mode!r}")
    return mode, (float(os.environ.get("QURL_S", "1.5")) if mode == "r5" else 1.0)


def group_of(name: str) -> str:
    if _W.search(name):
        return "W"
    if _GAMMA.search(name):
        return "gamma"
    return "rest"


def param_groups(named, lr: float, eps: float, wd: float, s: float):
    """Theorem 2(b)'s optimizer groups for scale s ("rest" first: VeRL logs param_groups[0]'s rate as actor/lr)."""
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
    """Clip every gradient by min(1, max_norm / (total + 1e-6)), then step; skip the step if the norm is not finite."""
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


def _is_actor(engine) -> bool:
    return (getattr(engine.model_config, "model_type", None) == "language_model"
            and not engine.engine_config.forward_only)


def _rank0() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _patch():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.workers.config.optimizer import build_optimizer
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine, "_qurl_patched", False):
        return
    original_build = FSDPEngine._build_optimizer
    original_step = FSDPEngine.optimizer_step

    def _build_optimizer(self, module):
        st = settings()
        if st is None or not _is_actor(self):
            return original_build(self, module)
        mode, s = st
        if isinstance(module, FSDP):
            raise RuntimeError("QURL_OPT needs the FSDP2 actor (actor_rollout_ref.actor.strategy=fsdp2)")
        cfg = self.optimizer_config
        eps = float((cfg.override_optimizer_config or {}).get("eps", 1e-8))
        named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        groups, sizes = param_groups(named, cfg.lr, eps, cfg.weight_decay, s)
        # QURL_CLIP=ordinary drops rule (c) (control arm R5o): per-group settings, but the ordinary norm.
        cs = 1.0 if os.environ.get("QURL_CLIP") == "ordinary" else s
        self._qurl_clip = clip_factors(named, cs)
        optimizer = build_optimizer(groups, cfg)
        if _rank0():
            import torch.distributed as dist

            ws = dist.get_world_size() if dist.is_initialized() else 1
            print(f"[qurl optimizer] mode {mode}, s {s}, {ws} rank(s): W {sizes['W']} tensors (lr {s * cfg.lr:.3g}, "
                  f"eps {eps / s:.3g}, wd {cfg.weight_decay / s:.3g}); gamma {sizes['gamma']} (lr {cfg.lr / s:.3g}, "
                  f"eps {s * eps:.3g}, wd {s * cfg.weight_decay:.3g}); rest {sizes['rest']} (lr {cfg.lr:.3g}, "
                  f"eps {eps:.3g}, wd {cfg.weight_decay:.3g}); clip {cfg.clip_grad} on the "
                  f"{'UAQ-coordinate' if cs != 1 else 'ordinary'} global norm, reduced over the FSDP shard group",
                  flush=True)
        return optimizer

    def optimizer_step(self):
        clip = getattr(self, "_qurl_clip", None)
        if settings() is None or clip is None:
            return original_step(self)
        return clip_and_step(self.optimizer, clip, self.optimizer_config.clip_grad)

    FSDPEngine._build_optimizer = _build_optimizer
    FSDPEngine.optimizer_step = optimizer_step
    FSDPEngine._qurl_patched = True


if settings() is not None:
    _patch()
