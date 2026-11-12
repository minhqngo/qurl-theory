"""Per-group optimizer settings and clipping rule (c) for VeRL v0.9.1's FSDP actor (arm R5, P1 note Theorem 2(b)-(c)).

Environment variables, read in the training workers (qurl_verl is imported there via
actor_rollout_ref.model.external_lib=qurl_verl):
  QURL_OPT=r5     Theorem 2(b)-(c) settings for s = QURL_S (default 1.5), without UAQ:
                    q/k/v_proj.weight (W):   lr s*eta,  Adam eps eps/s,  weight decay wd/s
                    input_layernorm.weight:  lr eta/s,  Adam eps s*eps,  weight decay s*wd
                    everything else:         lr eta,    eps,             wd
                  and the global clip norm computed in UAQ coordinates, i.e. on (s*g_W, g_gamma/s, g_rest).
  QURL_OPT=plain  the same code path with s = 1: plain settings, ordinary global-norm clipping computed here. Used by
                  the arms R5 is compared with (R3, R4, R6), so that every arm runs identical clipping arithmetic.
  unset           VeRL's own optimizer and FSDP clip (R1/R3 in the Table 1 check).
Only the actor is affected: an engine whose model_config.model_type is "language_model" and which has an optimizer.
The critic ("value_model") and the forward-only reference keep VeRL's code.

Per-parameter groups need the original parameters, so the actor must run FSDP2 (actor_rollout_ref.actor.strategy=fsdp2,
which keeps them as DTensors). FSDP1 would need use_orig_params=True, which fails under bf16 mixed precision in
PyTorch 2.11 (a gradient-dtype check in its sharded gradient views). The clip assumes one data-parallel rank (S3 runs
on one GPU), so per-parameter gradient norms are global norms; it refuses to run otherwise.

Exactness at s = 2 (powers of two make every rescaling exact): with QURL_OPT=plain on the UAQ checkpoint (R4) and
QURL_OPT=r5 on the original checkpoint (R5), one actor step on the same batch must give bit-identical weights after
mapping R4's back (W = s*W', gamma = gamma'/s). r5_check.py tests exactly that.
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


def _local(t):
    from torch.distributed.tensor import DTensor

    return t.to_local() if isinstance(t, DTensor) else t


def _is_actor(engine) -> bool:
    return (getattr(engine.model_config, "model_type", None) == "language_model"
            and not engine.engine_config.forward_only)


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
        if isinstance(module, FSDP) and not self.engine_config.use_orig_params:
            raise RuntimeError("QURL_OPT needs actor_rollout_ref.actor.fsdp_config.use_orig_params=True "
                               "(per-parameter optimizer groups under FSDP1)")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("QURL_OPT's clip assumes a single data-parallel rank")
        cfg = self.optimizer_config
        eps = float((cfg.override_optimizer_config or {}).get("eps", 1e-8))
        named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        by = {"W": [], "gamma": [], "rest": []}
        for n, p in named:
            by[group_of(n)].append(p)
        groups = [   # "rest" first: VeRL logs param_groups[0]'s learning rate as actor/lr
            dict(params=by["rest"], lr=cfg.lr, eps=eps, weight_decay=cfg.weight_decay),
            dict(params=by["W"], lr=s * cfg.lr, eps=eps / s, weight_decay=cfg.weight_decay / s),
            dict(params=by["gamma"], lr=cfg.lr / s, eps=s * eps, weight_decay=s * cfg.weight_decay),
        ]
        assert by["W"] and by["gamma"], "QURL_OPT: no q/k/v_proj or input_layernorm parameters found"
        # Clip factors in parameter order: the norm of (s*g_W, g_gamma/s, g_rest); s = 1 gives the ordinary norm.
        # QURL_CLIP=ordinary drops rule (c) (control arm R5o): per-group settings, but the ordinary norm.
        cs = 1.0 if os.environ.get("QURL_CLIP") == "ordinary" else s
        self._qurl_clip = [(p, cs if group_of(n) == "W" else 1.0 / cs if group_of(n) == "gamma" else 1.0)
                           for n, p in named]
        optimizer = build_optimizer(groups, cfg)
        print(f"[qurl optimizer] mode {mode}, s {s}: W {len(by['W'])} tensors (lr {s * cfg.lr:.3g}, eps {eps / s:.3g}, "
              f"wd {cfg.weight_decay / s:.3g}); gamma {len(by['gamma'])} (lr {cfg.lr / s:.3g}, eps {s * eps:.3g}, "
              f"wd {s * cfg.weight_decay:.3g}); rest {len(by['rest'])} (lr {cfg.lr:.3g}, eps {eps:.3g}, "
              f"wd {cfg.weight_decay:.3g}); clip {cfg.clip_grad} on the {'UAQ-coordinate' if cs != 1 else 'ordinary'} "
              f"global norm", flush=True)
        return optimizer

    def optimizer_step(self):
        clip = getattr(self, "_qurl_clip", None)
        if settings() is None or clip is None:
            return original_step(self)
        grads = [(p.grad, f) for p, f in clip if p.grad is not None]
        # FSDP2 keeps parameters and gradients as DTensors; on one rank the local shard is the whole tensor.
        norms = torch.stack([torch.linalg.vector_norm(_local(g).float()) * f for g, f in grads])
        total = torch.linalg.vector_norm(norms)
        if not torch.isfinite(total):
            print(f"WARN: grad_norm is not finite: {total}")
            self.optimizer.zero_grad()
            return total.item()
        coef = float(torch.clamp(self.optimizer_config.clip_grad / (total + 1e-6), max=1.0))
        for g, _ in grads:
            g.mul_(coef)   # the FP32 coefficient round-trips exactly through a Python float
        self.optimizer.step()
        return total.item()

    FSDPEngine._build_optimizer = _build_optimizer
    FSDPEngine.optimizer_step = optimizer_step
    FSDPEngine._qurl_patched = True


if settings() is not None:
    _patch()
