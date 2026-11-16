"""R5's optimizer in VeRL v0.9.1: patches the FSDP2 actor engine's optimizer construction and step.

With QURL_OPT set (qurl.r5.settings), the actor engine's optimizer gets Theorem 2(b)'s three parameter groups and its
step clips on the global norm in UAQ coordinates (rule (c), all ranks reduced; qurl.r5.clip_and_step). QURL_OPT=plain
runs the same code with s = 1, QURL_CLIP=ordinary keeps R5's groups with the ordinary norm (control R5o).

Only the actor is affected: an engine whose model_config.model_type is "language_model" and which is not forward-only.
The critic and the forward-only reference keep VeRL's code. Per-parameter groups need the original parameters, so the
actor must run FSDP2 (actor_rollout_ref.actor.strategy=fsdp2, which keeps them as DTensors): FSDP1 would need
use_orig_params=True, which fails under bf16 mixed precision in PyTorch 2.11.
"""
from __future__ import annotations

import os

from ..r5 import clip_and_step, clip_factors, param_groups, settings


def _is_actor(engine) -> bool:
    return (getattr(engine.model_config, "model_type", None) == "language_model"
            and not engine.engine_config.forward_only)


def _rank0() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def patch() -> None:
    """Patch FSDPEngine._build_optimizer and FSDPEngine.optimizer_step (once per process)."""
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


def apply_if_enabled() -> None:
    """Patch VeRL's actor optimizer if QURL_OPT is set (raises on an invalid QURL_OPT)."""
    if settings() is not None:
        patch()
