"""qurl.quant.tie_indices: the weights on a rounding boundary, in a BF16 tensor and after UAQ's FP32 rescaling."""
import numpy as np
import pytest
import torch

from qurl.quant import TIE_TOL, quantize_rows, tie_indices
from qurl.uaq import rescale
from qurl_test_support import (bf16_with_ties, load_original_int8_rollout, np_quantize_rows, np_tie_mask,
                               tiny_qwen2)


def ties_of(w: torch.Tensor) -> torch.Tensor:
    return tie_indices(w, quantize_rows(w)[1])


def tensors():
    g = torch.Generator().manual_seed(0)
    return [bf16_with_ties((64, 96), g), bf16_with_ties((32, 64), g), bf16_with_ties((96, 64), g)]


def test_planted_bf16_ties_are_found():
    g = torch.Generator().manual_seed(3)
    w = bf16_with_ties((64, 96), g)
    found = set(ties_of(w).tolist())
    x = w.double()
    # kind 1: +-absmax / 2 in every 4th row
    for r in range(0, w.shape[0], 4):
        half = x[r].abs().max() / 2
        cols = torch.nonzero(x[r].abs() == half).view(-1).tolist()
        assert len(cols) >= 2
        assert all(r * w.shape[1] + c in found for c in cols)
    # kind 2: row 1 has scale exactly 2^-13, and every exact half-integer quotient is a tie
    _, scale = quantize_rows(w)
    assert float(scale[1]) == 2.0 ** -13
    q = x[1] / 2.0 ** -13
    exact = torch.nonzero((q - torch.floor(q)) == 0.5).view(-1).tolist()
    assert len(exact) > 5
    assert all(w.shape[1] + c in found for c in exact)


def test_ties_match_independent_recomputation():
    g = torch.Generator().manual_seed(4)
    for w in [*tensors(), torch.randn(64, 80, generator=g) * 0.02]:
        assert np.array_equal(ties_of(w).numpy(), np.flatnonzero(np_tie_mask(w)))


@pytest.mark.originals
def test_ties_match_original(monkeypatch):
    original = load_original_int8_rollout(monkeypatch)
    for w in tensors():
        _, scale = quantize_rows(w)
        assert torch.equal(tie_indices(w, scale), original.tie_indices(w, scale))


@pytest.mark.parametrize("s", [1.5, 2.0])
def test_uaq_fp32_rescaling_keeps_the_tie_set(s):
    """The UAQ checkpoint stores fl32(W / s): the same ties as the BF16 checkpoint, moved by well under 1e-4 steps."""
    for w in tensors():
        ties = ties_of(w)
        assert ties.numel() > 0
        w_uaq = w.float() / s
        assert torch.equal(ties_of(w_uaq), ties)
        _, scale = np_quantize_rows(w)
        _, scale_uaq = np_quantize_rows(w_uaq)
        q = (w.float().numpy() / scale).ravel()[ties.numpy()]
        q_uaq = (w_uaq.numpy() / scale_uaq).ravel()[ties.numpy()]
        moved = np.abs(q_uaq.astype(np.float64) - q).max()
        assert moved < 0.5 * TIE_TOL and (moved == 0) == (s == 2.0)


def test_uaq_checkpoint_of_a_bf16_model_keeps_the_tie_sets():
    """Through qurl.uaq.rescale: a BF16 model loaded in FP32 and rescaled at s = 1.5 has the same q/k/v ties."""
    model = tiny_qwen2(seed=5, layers=3)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():   # BF16 weights with planted ties, held in FP32 (from_pretrained(..., dtype=float32))
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.weight.copy_(bf16_with_ties(tuple(proj.weight.shape), g).float())

    def qkv_ties():
        return [ties_of(p.weight.detach()) for layer in model.model.layers
                for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)]

    before = qkv_ties()
    rescale(model, 1.5)
    after = qkv_ties()
    assert sum(t.numel() for t in before) > 0
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_a_small_update_flips_about_half_of_the_ties():
    """Why ties matter: a tiny update (1e-3 grid steps) flips about half of them, and almost no other weight."""
    g = torch.Generator().manual_seed(6)
    w = bf16_with_ties((256, 256), g)
    codes, scale = quantize_rows(w)
    w2 = w.float() + torch.randn(w.shape, generator=g) * 1e-3 * scale
    flipped = (quantize_rows(w2)[0] != codes).view(-1)
    ties = ties_of(w)
    others = torch.ones_like(flipped)
    others[ties] = False
    assert 0.35 < float(flipped[ties].float().mean()) < 0.65
    assert float(flipped[others].float().mean()) < 0.01
