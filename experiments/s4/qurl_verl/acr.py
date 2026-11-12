"""QuRL's adaptive clipping range (ACR) as a VeRL policy loss: loss_mode=qurl_acr.

QuRL (arXiv 2602.13953, Eq. 9) weights the PPO objective by the TIS weight min(rho, C), with rho = pi_old / mu
(pi_old: the FSDP actor's proximal probability; mu: the quantized rollout's), and raises the upper clip bound of a
token whose weight was truncated to (1 + eps_high) / min(1, C / rho) = (1 + eps_high) * max(1, rho / C).
Everything else is VeRL's vanilla PPO loss (core_algos.compute_policy_loss_vanilla), including the dual clip at
clip_ratio_c for negative advantages and the loss aggregation.

VeRL wiring for the QuRL arms:
  algorithm.rollout_correction.rollout_is=token          # VeRL computes token-level rho = pi_old / mu ...
  algorithm.rollout_correction.rollout_is_threshold=1e30 # ... and passes it untruncated; this loss truncates at C
  actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
  QURL_TIS_C=<C> in the environment (default 2.0)
"""
import os

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


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
    assert config is not None
    if rollout_is_weights is None:
        raise ValueError("loss_mode=qurl_acr needs algorithm.rollout_correction.rollout_is=token "
                         "(with rollout_is_threshold large enough to pass rho untruncated)")
    tis_c = float(os.environ.get("QURL_TIS_C", "2.0"))
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_c = config.get("clip_ratio_c", 3.0)
    assert clip_c > 1.0

    rho = rollout_is_weights.detach()                         # pi_old / mu per token (0 on padding)
    w = torch.clamp(rho, max=tis_c)                           # TIS weight
    upper = (1.0 + clip_high) * torch.clamp(rho / tis_c, min=1.0)   # ACR upper bound

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = verl_F.masked_mean(-neg_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.minimum(ratio.clamp(min=1.0 - clip_low), upper)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_c                         # dual clip, as in VeRL's vanilla loss
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1) * w
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                       **config.global_batch_info)

    # Tokens ACR keeps that plain PPO clipping would drop: A > 0 and 1 + eps_high <= ratio < upper.
    acr_kept = (advantages > 0) & (ratio >= 1.0 + clip_high) & (ratio < upper)
    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "qurl/tis_trunc_frac": verl_F.masked_mean((rho > tis_c).float(), response_mask).item(),
        "qurl/acr_kept_frac": verl_F.masked_mean(acr_kept.float(), response_mask).item(),
        "qurl/rho_max": (rho * response_mask).max().item(),
    }
    return pg_loss, metrics
