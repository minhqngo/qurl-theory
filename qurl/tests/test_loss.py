"""qurl.loss and qurl.verl.loss: QuRL's TIS weight and ACR clip bound (QuRL Eq. 9)."""
import numpy as np
import pytest
import torch

from qurl.loss import tis_acr_terms
from qurl_test_support import load_original_acr

C = 2.0
CLIP = dict(clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0)
AGG_MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]


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
    """QuRL Eq. 9 per token, written independently in float64 numpy: -min(rho, C) * PPO's dual-clipped objective,
    with the upper clip bound raised to (1 + eps_high) / min(1, C / rho)."""
    old, new, adv, rho = (x.double().numpy() for x in (old, new, adv, rho))
    r = np.exp(np.clip(new - old, -20.0, 20.0))
    with np.errstate(divide="ignore"):
        upper = (1 + high) / np.minimum(1.0, cap / rho)
    obj = np.minimum(r * adv, np.clip(r, 1 - low, upper) * adv)
    obj = np.where(adv < 0, np.maximum(obj, c_dual * adv), obj)
    return -np.minimum(rho, cap) * obj


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_core_matches_float64_reference(seed):
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


def test_gradient_flows_through_log_prob_only():
    old, new, adv, rho, mask = random_batch(3)
    new.requires_grad_(True)
    rho.requires_grad_(True)
    t = tis_acr_terms(old, new, adv, rho, tis_c=C, **CLIP)
    (t.losses * mask).sum().backward()
    assert new.grad is not None and new.grad.abs().sum() > 0 and rho.grad is None


def actor_config(**global_batch_info):
    from verl.workers.config import ActorConfig

    cfg = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, clip_ratio=0.2, **CLIP)
    cfg.global_batch_info.update(global_batch_info)
    return cfg


def run_loss(fn, batch, mode, cfg):
    old, new, adv, rho, mask = batch
    leaf = new.clone().requires_grad_(True)
    loss, metrics = fn(old, leaf, adv, mask, loss_agg_mode=mode, config=cfg, rollout_is_weights=rho)
    loss.backward()
    return loss.detach(), metrics, leaf.grad


@pytest.mark.verl
@pytest.mark.originals
@pytest.mark.parametrize("mode", AGG_MODES)
@pytest.mark.parametrize("cap", [None, "1.5"])
def test_verl_loss_matches_original(mode, cap, monkeypatch):
    from qurl.verl.loss import compute_policy_loss_qurl_acr

    original = load_original_acr(monkeypatch)
    if cap is not None:
        monkeypatch.setenv("QURL_TIS_C", cap)
    for seed in range(3):
        batch = random_batch(seed)
        for cfg in (actor_config(), actor_config(dp_size=1, batch_num_tokens=int(batch[4].sum()) + 7,
                                                 global_batch_size=9, loss_scale_factor=None)):
            ours = run_loss(compute_policy_loss_qurl_acr, batch, mode, cfg)
            theirs = run_loss(original.compute_policy_loss_qurl_acr, batch, mode, cfg)
            assert torch.equal(ours[0], theirs[0]) and ours[1] == theirs[1] and torch.equal(ours[2], theirs[2])


@pytest.mark.verl
@pytest.mark.parametrize("mode", AGG_MODES)
def test_without_truncation_it_is_vanilla_ppo_with_tis_weights(mode):
    """rho <= C everywhere: ACR's bound is 1 + eps_high and the TIS weight is rho, i.e. VeRL's vanilla loss."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    from qurl.verl.loss import compute_policy_loss_qurl_acr

    for seed in range(3):
        batch = random_batch(seed, max_rho=1.9)
        ours = run_loss(compute_policy_loss_qurl_acr, batch, mode, actor_config())
        vanilla = run_loss(compute_policy_loss_vanilla, batch, mode, actor_config())
        assert torch.equal(ours[0], vanilla[0]) and torch.equal(ours[2], vanilla[2])
        assert all(ours[1][k] == v for k, v in vanilla[1].items())
        assert ours[1]["qurl/tis_trunc_frac"] == 0.0


@pytest.mark.verl
def test_registered_and_needs_rho():
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    from qurl.verl.loss import compute_policy_loss_qurl_acr

    assert get_policy_loss_fn("qurl_acr") is compute_policy_loss_qurl_acr
    old, new, adv, _, mask = random_batch(0)
    with pytest.raises(ValueError, match="rollout_is=token"):
        compute_policy_loss_qurl_acr(old, new, adv, mask, config=actor_config())
