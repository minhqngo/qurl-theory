"""QuRL's policy loss: token-level TIS with the adaptive clipping range (ACR), as per-token PPO losses.

QuRL trains with the decoupled PPO objective (paper, Eq. 4): the behavior policy mu, which sampled the tokens, is the
quantized old actor (the INT8 rollout), and the proximal policy pi_old, which the ratio is clipped against, is the
full-precision old actor. It weights the objective by the truncated importance-sampling (TIS) weight min(rho, C),
with rho = pi_old / mu (Eq. 5), and ACR raises the upper clip bound of a token whose weight was truncated to
(1 + eps_high) / r with r = min(1, C / rho), that is (1 + eps_high) * max(1, rho / C) (Eq. 9). Everything else is
VeRL's vanilla PPO loss (core_algos.compute_policy_loss_vanilla), including the decoupled clip range (eps_low,
eps_high) and the dual clip at clip_ratio_c for negative advantages. With no truncated token (rho <= C everywhere) the
losses are the vanilla losses times rho, bit for bit: TIS alone.

This module holds the per-token arithmetic only; qurl.verl.loss aggregates it with VeRL's agg_loss and registers it as
VeRL's loss_mode=qurl_acr.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class TisAcrTerms(NamedTuple):
    """Per-token terms of QuRL's TIS + ACR objective; every tensor has the shape of the log-probabilities."""
    losses: torch.Tensor          # per-token loss: the dual-clipped PPO loss with ACR's upper bound, times tis_weight
    neg_kl: torch.Tensor          # log_prob - old_log_prob, clamped to [-20, 20]
    ratio: torch.Tensor           # exp(neg_kl) = pi / pi_old
    rho: torch.Tensor             # pi_old / mu (detached)
    tis_weight: torch.Tensor      # min(rho, C)
    upper: torch.Tensor           # ACR's upper clip bound on the ratio: (1 + eps_high) * max(1, rho / C)
    clipped: torch.Tensor         # bool: the clip bound was active (the clipped loss exceeds the unclipped one)
    clipped_lower: torch.Tensor   # bool: the dual clip's bound -A * c is below the clipped loss (VeRL's lower clip)
    acr_kept: torch.Tensor        # bool: tokens ACR keeps that plain PPO clipping drops (A > 0, 1 + eps_high <= ratio
    #                               < upper)


def tis_acr_terms(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor,
                  rho: torch.Tensor, *, clip_ratio_low: float, clip_ratio_high: float, clip_ratio_c: float,
                  tis_c: float) -> TisAcrTerms:
    """QuRL's per-token TIS + ACR terms; gradients flow through `log_prob` only.

    rho is pi_old / mu per token, untruncated (VeRL passes it as rollout_is_weights, 0 on padding). clip_ratio_low and
    clip_ratio_high are PPO's eps_low and eps_high, clip_ratio_c the dual-clip constant (> 1) and tis_c the TIS cap C.
    """
    rho = rho.detach()                                           # pi_old / mu per token (0 on padding)
    w = torch.clamp(rho, max=tis_c)                              # TIS weight
    upper = (1.0 + clip_ratio_high) * torch.clamp(rho / tis_c, min=1.0)   # ACR upper bound

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.minimum(ratio.clamp(min=1.0 - clip_ratio_low), upper)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c                      # dual clip, as in VeRL's vanilla loss
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1) * w
    # Tokens ACR keeps that plain PPO clipping would drop: A > 0 and 1 + eps_high <= ratio < upper.
    acr_kept = (advantages > 0) & (ratio >= 1.0 + clip_ratio_high) & (ratio < upper)
    return TisAcrTerms(losses=losses, neg_kl=neg_kl, ratio=ratio, rho=rho, tis_weight=w, upper=upper,
                       clipped=torch.gt(pg_losses2, pg_losses1), clipped_lower=torch.gt(clip_pg_losses1, pg_losses3),
                       acr_kept=acr_kept)
