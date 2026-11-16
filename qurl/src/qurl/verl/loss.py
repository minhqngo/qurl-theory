"""QuRL's TIS + ACR objective as a VeRL policy loss: actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr.

The per-token arithmetic is qurl.loss.tis_acr_terms (paper, Eqs. 5 and 9); this module reads the clip settings from
VeRL's actor config, aggregates the losses with VeRL's agg_loss and reports VeRL's PPO metrics plus three of its own:
  qurl/tis_trunc_frac   share of tokens whose TIS weight was truncated (rho > C)
  qurl/acr_kept_frac    share of tokens ACR keeps that plain PPO clipping would drop
  qurl/rho_max          largest rho in the micro-batch
VeRL wiring:
  algorithm.rollout_correction.rollout_is=token           # VeRL computes token-level rho = pi_old / mu ...
  algorithm.rollout_correction.rollout_is_threshold=1e30  # ... and passes it untruncated; this loss truncates at C
  actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
  QURL_TIS_C=<C> in the environment (default 2.0)
"""
from __future__ import annotations

import os

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss

from ..loss import tis_acr_terms

TIS_C_DEFAULT = 2.0


def tis_c() -> float:
    """The TIS cap C, from QURL_TIS_C (default 2.0, the experiments' value; the paper does not state its C)."""
    return float(os.environ.get("QURL_TIS_C", str(TIS_C_DEFAULT)))


@register_policy_loss("qurl_acr")  # type: ignore[arg-type]
def compute_policy_loss_qurl_acr(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
):
    """VeRL's policy-loss interface; rollout_is_weights must carry VeRL's untruncated token-level rho."""
    assert config is not None
    if rollout_is_weights is None:
        raise ValueError("loss_mode=qurl_acr needs algorithm.rollout_correction.rollout_is=token "
                         "(with rollout_is_threshold large enough to pass rho untruncated)")
    c = tis_c()
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_c = config.get("clip_ratio_c", 3.0)
    assert clip_c > 1.0

    t = tis_acr_terms(old_log_prob, log_prob, advantages, rollout_is_weights, clip_ratio_low=clip_low,
                      clip_ratio_high=clip_high, clip_ratio_c=clip_c, tis_c=c)
    ppo_kl = verl_F.masked_mean(-t.neg_kl, response_mask)
    pg_clipfrac = verl_F.masked_mean(t.clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(t.clipped_lower * (advantages < 0).float(), response_mask)
    pg_loss = agg_loss(loss_mat=t.losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                       **config.global_batch_info)

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "qurl/tis_trunc_frac": verl_F.masked_mean((t.rho > c).float(), response_mask).item(),
        "qurl/acr_kept_frac": verl_F.masked_mean(t.acr_kept.float(), response_mask).item(),
        "qurl/rho_max": (t.rho * response_mask).max().item(),
    }
    return pg_loss, metrics
