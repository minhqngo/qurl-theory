"""qurl.loss: QuRL's TIS weight and ACR clip bound (the paper's Eqs. 5 and 9)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qurl.loss import tis_acr_terms
from qurl_test_support import load_original_acr

C = 2.0
CLIP = dict(clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0)


def random_batch(seed, dtype=torch.float32, b=6, t=19, max_rho=None):
    """Log-probabilities, advantages, untruncated rho (0 on padding) and a response mask with padding."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, t)
    for i, n in enumerate(torch.randint(t // 2, t + 1, (b,), generator=g).tolist()):
        mask[i, n:] = 0
    old = -torch.rand(b, t, generator=g) * 3 - 0.05
    new = old + torch.randn(b, t, generator=g) * 0.3
    adv = (torch.randn(b, 1, generator=g) + 0.2 * torch.randn(b, t, generator=g)) * mask
    rho = torch.exp(torch.randn(b, t, generator=g) * 0.7)
    if max_rho is not None:
        rho = rho.clamp(max=max_rho)
    rho = rho * mask
    return [x.to(dtype) for x in (old, new, adv, rho, mask)]


def reference_losses(old, new, adv, rho, low, high, c_dual, cap):
    """Eq. 9 per token, written independently in float64 numpy: -min(rho, C) * PPO's dual-clipped objective, with the
    upper clip bound raised to (1 + eps_high) / min(1, C / rho)."""
    old, new, adv, rho = (x.double().numpy() for x in (old, new, adv, rho))
    r = np.exp(np.clip(new - old, -20.0, 20.0))
    with np.errstate(divide="ignore"):
        upper = (1 + high) / np.minimum(1.0, cap / rho)
    obj = np.minimum(r * adv, np.clip(r, 1 - low, upper) * adv)
    obj = np.where(adv < 0, np.maximum(obj, c_dual * adv), obj)
    return -np.minimum(rho, cap) * obj


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_float64_reference(seed):
    old, new, adv, rho, mask = random_batch(seed, torch.float64, b=16, t=33)
    t = tis_acr_terms(old, new, adv, rho, tis_c=C, **CLIP)
    ref = reference_losses(old, new, adv, rho, 0.2, 0.28, 10.0, C)
    assert (rho > C).any() and t.acr_kept.any() and t.clipped.any()   # truncation, ACR and clipping all bind
    np.testing.assert_allclose(t.losses.numpy(), ref, rtol=1e-12, atol=1e-15)
    ratio = t.ratio.numpy()
    upper = t.upper.numpy()
    np.testing.assert_allclose(upper, 1.28 * np.maximum(1.0, rho.numpy() / C), rtol=1e-15)
    assert np.array_equal(t.acr_kept.numpy(), (adv.numpy() > 0) & (ratio >= 1.28) & (ratio < upper))
    assert torch.equal(t.tis_weight, rho.clamp(max=C))


def test_without_truncation_it_is_tis_on_ppo():
    """rho <= C everywhere: the TIS weight is rho and ACR's bound is exactly 1 + eps_high, i.e. plain PPO with the
    dual clip, weighted by rho."""
    old, new, adv, rho, mask = random_batch(3, max_rho=1.9)
    t = tis_acr_terms(old, new, adv, rho, tis_c=C, **CLIP)
    assert torch.equal(t.tis_weight, rho) and torch.equal(t.upper, torch.full_like(rho, 1.28))
    ratio = torch.exp(torch.clamp(new - old, -20.0, 20.0))
    clipped = torch.maximum(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.28))
    ppo = torch.where(adv < 0, torch.min(-adv * 10.0, clipped), clipped)
    assert torch.equal(t.losses, ppo * rho) and not t.acr_kept.any()


def test_gradient_flows_through_log_prob_only():
    old, new, adv, rho, mask = random_batch(3)
    new.requires_grad_(True)
    rho.requires_grad_(True)
    t = tis_acr_terms(old, new, adv, rho, tis_c=C, **CLIP)
    (t.losses * mask).sum().backward()
    assert new.grad is not None and new.grad.abs().sum() > 0 and rho.grad is None


class ActorConfig(SimpleNamespace):
    """What the original loss reads from VeRL's actor config."""

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.mark.originals
@pytest.mark.parametrize("cap", [None, "1.5"])
def test_matches_original(cap, monkeypatch):
    """The experiments' loss (VeRL's token-mean aggregation restated) and the token-mean of tis_acr_terms: the same
    loss and the same gradients, bit for bit."""
    original = load_original_acr(monkeypatch)
    if cap is not None:
        monkeypatch.setenv("QURL_TIS_C", cap)
    cfg = ActorConfig(clip_ratio=0.2, global_batch_info={}, **CLIP)
    for seed in range(3):
        old, new, adv, rho, mask = random_batch(seed)
        theirs_in = new.clone().requires_grad_(True)
        theirs, _ = original.compute_policy_loss_qurl_acr(old, theirs_in, adv, mask, loss_agg_mode="token-mean",
                                                          config=cfg, rollout_is_weights=rho)
        theirs.backward()
        ours_in = new.clone().requires_grad_(True)
        t = tis_acr_terms(old, ours_in, adv, rho, tis_c=float(cap or C), **CLIP)
        ours = (t.losses * mask).sum() / mask.sum() * 1
        ours.backward()
        assert torch.equal(ours.detach(), theirs.detach()) and torch.equal(ours_in.grad, theirs_in.grad)
