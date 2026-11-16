"""qurl.quant.quantize_rows: the per-row absmax INT8 quantizer of the refit (Theorem 1's quantizer)."""
import numpy as np
import pytest
import torch

from qurl.quant import QMAX, quantize_rows
from qurl_test_support import load_original_int8_rollout, np_quantize_rows


def weight_cases():
    g = torch.Generator().manual_seed(0)
    zero_row = torch.randn(8, 16, generator=g) * 0.02
    zero_row[3] = 0.0
    return {
        "fp32": torch.randn(64, 96, generator=g) * 0.02,
        "bf16": (torch.randn(64, 96, generator=g) * 0.02).to(torch.bfloat16),
        "fp16": (torch.randn(32, 40, generator=g) * 0.02).to(torch.float16),
        "heavy-tailed": torch.randn(48, 80, generator=g) ** 3 * 0.01,
        "zero row": zero_row,
    }


CASES = weight_cases()


@pytest.mark.parametrize("case", CASES)
def test_matches_independent_quantizer(case):
    w = CASES[case]
    codes, scale = quantize_rows(w)
    ref_codes, ref_scale = np_quantize_rows(w)
    assert codes.dtype == torch.int8 and scale.dtype == torch.float32
    assert codes.shape == w.shape and scale.shape == (w.shape[0], 1)
    assert np.array_equal(codes.numpy(), ref_codes)
    assert np.array_equal(scale.numpy(), ref_scale)


@pytest.mark.originals
@pytest.mark.parametrize("case", CASES)
def test_matches_original(case, monkeypatch):
    original = load_original_int8_rollout(monkeypatch)
    w = CASES[case]
    codes, scale = quantize_rows(w)
    o_codes, o_scale = original.quantize_rows(w)
    assert torch.equal(codes, o_codes) and torch.equal(scale, o_scale)


def test_range_and_row_extremes():
    w = CASES["heavy-tailed"]
    codes, _ = quantize_rows(w)
    assert int(codes.abs().max()) <= QMAX
    top = w.abs().argmax(dim=1)
    assert torch.equal(codes.gather(1, top[:, None]).abs().squeeze(1), torch.full((w.shape[0],), 127, dtype=torch.int8))
    zcodes, zscale = quantize_rows(CASES["zero row"])
    assert float(zscale[3]) == 1.0 and int(zcodes[3].abs().max()) == 0


def test_power_of_two_scale_and_round_half_even():
    """A row with absmax 254 * 2^-14 has scale exactly 2^-13; its half-integer quotients round to even."""
    k = torch.tensor([254.0, 1.0, 3.0, 5.0, -1.0, -3.0, 7.0, 2.0, -5.0])
    codes, scale = quantize_rows((k * 2.0 ** -14)[None])
    assert float(scale) == 2.0 ** -13
    assert codes[0].tolist() == [127, 0, 2, 2, 0, -2, 4, 1, -2]


@pytest.mark.parametrize("k", [-3, -1, 1, 4])
def test_homogeneous_under_powers_of_two(k):
    """Theorem 1 with exact arithmetic: W * 2^k has the same codes and 2^k times the scales (a zero row keeps 1)."""
    for w in CASES.values():
        codes, scale = quantize_rows(w)
        c2, s2 = quantize_rows(w.float() * 2.0 ** k)
        want = torch.where(w.float().abs().amax(dim=1, keepdim=True) > 0, scale * 2.0 ** k, scale)
        assert torch.equal(codes, c2) and torch.equal(s2, want)


def test_homogeneous_up_to_rounding_at_s_1_5():
    """At s = 1.5 the FP32 rounding of W / s can move only codes that sit within a hair of a rounding boundary."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 512, generator=g) * 0.02
    codes, scale = quantize_rows(w)
    c2, _ = quantize_rows(w / 1.5)
    moved = codes != c2
    frac = (w / scale - torch.round(w / scale)).abs()
    assert moved.float().mean() < 1e-3
    assert bool((frac[moved] > 0.5 - 1e-4).all())


@pytest.mark.gpu
def test_cpu_and_cuda_identical():
    """The scale is a multiplication by fl32(1/127), so CPU and CUDA give identical scales and codes."""
    g = torch.Generator().manual_seed(2)
    for w in [*CASES.values(), torch.randn(1024, 768, generator=g) * 0.02]:
        c_cpu, s_cpu = quantize_rows(w)
        c_gpu, s_gpu = quantize_rows(w.cuda())
        assert torch.equal(c_cpu, c_gpu.cpu()) and torch.equal(s_cpu, s_gpu.cpu())
