"""qurl.fakequant: INT8 W8A8 fake quantization of a model's linear layers (QuRL's rollout without vLLM)."""
import numpy as np
import pytest
import torch

from qurl import uaq
from qurl.fakequant import int8_rollout
from qurl.quant import quantize_rows
from qurl_test_support import np_quantize_rows, tiny_qwen2


def reference_linear(x, w, b):
    """INT8 W8A8, independently in float64 numpy: per-token absmax codes of x times per-row absmax codes of W."""
    xc, xs = np_quantize_rows(x.reshape(-1, x.shape[-1]))
    wc, ws = np_quantize_rows(w)
    y = (xc.astype(np.float64) @ wc.astype(np.float64).T) * xs.astype(np.float64) * ws.astype(np.float64).T
    if b is not None:
        y = y + b.detach().double().numpy()
    return y.reshape(*x.shape[:-1], -1)


def test_linear_layer_matches_reference():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(64, 48)
    x = torch.randn(3, 5, 64, generator=g) * torch.linspace(0.1, 4.0, 64)   # uneven channels, as activations have
    holder = torch.nn.Sequential(lin)
    with torch.no_grad(), int8_rollout(holder):
        y = lin(x)
    assert y.shape == (3, 5, 48) and y.dtype == torch.float32
    np.testing.assert_allclose(y.numpy(), reference_linear(x, lin.weight, lin.bias), rtol=1e-6, atol=1e-6)


def prompt_ids(n=12):
    return torch.randint(0, 96, (2, n), generator=torch.Generator().manual_seed(1))


def test_every_linear_but_lm_head_and_restored_after():
    model = tiny_qwen2(seed=0).eval()
    ids = prompt_ids()
    with torch.no_grad():
        fp = model(input_ids=ids).logits
        with int8_rollout(model) as ctx:
            names = {n for n, m in model.named_modules() if "forward" in m.__dict__}
            q = model(input_ids=ids).logits
        after = model(input_ids=ids).logits
    assert len(ctx.layers) == 7 * 2 and "lm_head" not in names
    assert all(n.endswith(("_proj",)) for n in names)
    assert not torch.equal(fp, q) and (q - fp).abs().max() < 0.05 * fp.abs().max()
    assert torch.equal(after, fp)
    assert not any("forward" in m.__dict__ or hasattr(m, "_qurl_codes") for m in model.modules())


def test_re_entering_requantizes_the_current_weights():
    model = tiny_qwen2(seed=0).eval()
    proj = model.model.layers[0].self_attn.q_proj
    with int8_rollout(model):
        before = proj._qurl_codes.clone()
    with torch.no_grad():
        proj.weight.mul_(-1.0)
    with int8_rollout(model):
        codes, scale = quantize_rows(proj.weight)
        assert torch.equal(proj._qurl_codes, codes) and torch.equal(proj._qurl_scale, scale.t())
    assert torch.equal(codes, -before)


def test_uaq_at_s_2_leaves_the_int8_rollout_bit_identical():
    """UAQ doubles the activations entering q/k/v and halves their weights: the INT8 codes are unchanged and the
    scales exactly compensate, so the quantized model's logits are bit-identical."""
    model = tiny_qwen2(seed=0).eval()
    ids = prompt_ids()
    with torch.no_grad():
        with int8_rollout(model):
            before = model(input_ids=ids).logits
        uaq.rescale(model, 2.0)
        with int8_rollout(model):
            after = model(input_ids=ids).logits
    assert torch.equal(before, after)


def test_nesting_is_refused():
    model = tiny_qwen2(seed=0)
    with int8_rollout(model):
        with pytest.raises(RuntimeError, match="already patched"):
            int8_rollout(model).__enter__()
