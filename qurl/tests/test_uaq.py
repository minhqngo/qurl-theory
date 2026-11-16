"""qurl.uaq: the UAQ rescaling, its exact s = 2 probe, and the FP32 checkpoint."""
import json
import sys

import pytest
import torch

from qurl import uaq
from qurl_test_support import load_original_uaq, tiny_qwen2, write_tiny_checkpoint


def prompt(seed=0, n=24):
    return torch.randint(0, 96, (1, n), generator=torch.Generator().manual_seed(seed))


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_rescaling_by_2_preserves_logits_bit_for_bit():
    model = tiny_qwen2(seed=0).eval()
    ids = prompt()
    before = snapshot(model)
    l0 = uaq.logits(model, ids)
    assert uaq.rescale(model, 2.0) == (3 * 2, 2)
    assert torch.equal(uaq.logits(model, ids), l0)
    for n, p in model.named_parameters():
        if n.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            assert torch.equal(p, before[n] / 2)
        elif n.endswith("input_layernorm.weight"):
            assert torch.equal(p, before[n] * 2)
        else:   # the q/k/v biases included
            assert torch.equal(p, before[n])


def test_check_at_s_2_and_1_5():
    ids = prompt()
    rec2 = uaq.check(tiny_qwen2(seed=0).eval(), ids, 2.0)
    assert rec2["probe_s2_bit_identical"] and rec2["bit_identical"] and rec2["logit_max_abs_diff"] == 0.0
    assert (rec2["qkv_tensors"], rec2["norm_gains"]) == (6, 2)
    rec15 = uaq.check(tiny_qwen2(seed=0).eval(), ids, 1.5)
    assert rec15["probe_s2_bit_identical"] and rec15["logit_rel_diff"] < 1e-5


@pytest.mark.originals
@pytest.mark.parametrize("s", [1.5, 2.0])
def test_check_matches_original(s):
    original = load_original_uaq()
    ids = prompt(seed=1)
    ours, theirs = tiny_qwen2(seed=2).eval(), tiny_qwen2(seed=2).eval()
    assert uaq.check(ours, ids, s) == original.check(theirs, ids, s)
    a, b = snapshot(ours), snapshot(theirs)
    assert all(torch.equal(a[n], b[n]) for n in a)


def wrong_selection(kind):
    def tensors(model):
        ws, gs = [], []
        for layer in model.model.layers:
            a = layer.self_attn
            ws += {"V left out": [a.q_proj.weight, a.k_proj.weight],
                   "O instead of V": [a.q_proj.weight, a.k_proj.weight, a.o_proj.weight]}.get(
                kind, [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            if kind == "post-attention norm":
                gs.append(layer.post_attention_layernorm.weight)
            elif kind != "no gains":
                gs.append(layer.input_layernorm.weight)
        return ws, gs
    return tensors


@pytest.mark.parametrize("kind", ["V left out", "O instead of V", "post-attention norm", "no gains"])
def test_wrong_rescalings_fail_the_probe_and_leave_the_weights(kind, monkeypatch):
    monkeypatch.setattr(uaq, "uaq_tensors", wrong_selection(kind))
    model = tiny_qwen2(seed=0).eval()
    before = snapshot(model)
    with pytest.raises(uaq.UAQCheckError, match="rescaling by 2 changed the logits"):
        uaq.check(model, prompt(), 1.5)
    after = snapshot(model)
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_checkpoint_is_fp32_and_rescaled_once(tmp_path):
    from safetensors.torch import load_file

    src, out = tmp_path / "src", tmp_path / "uaq"
    write_tiny_checkpoint(src, seed=3)
    uaq.main(["--model", str(src), "--s", "1.5", "--out", str(out)])
    base, new = load_file(src / "model.safetensors"), load_file(out / "model.safetensors")
    assert set(base) == set(new) and all(t.dtype == torch.float32 for t in new.values())
    for n, t in base.items():
        if n.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            want = t.float() / 1.5
        elif n.endswith("input_layernorm.weight"):
            want = t.float() * 1.5
        else:
            want = t.float()
        assert torch.equal(new[n], want), n
    info = json.loads((out / "uaq.json").read_text())
    assert info["s"] == 1.5 and info["dtype"] == "float32" and info["probe_s2_bit_identical"]
    assert info["source"] == str(src) and (info["qkv_tensors"], info["norm_gains"]) == (6, 2)


@pytest.mark.originals
def test_checkpoint_byte_identical_to_original(tmp_path, monkeypatch):
    src, ours, theirs = tmp_path / "src", tmp_path / "ours", tmp_path / "theirs"
    write_tiny_checkpoint(src, seed=4)
    uaq.main(["--model", str(src), "--s", "1.5", "--out", str(ours)])
    original = load_original_uaq()
    monkeypatch.setattr(sys, "argv", ["make_uaq_checkpoint.py", "--model", str(src), "--s", "1.5", "--out",
                                      str(theirs)])
    original.main()
    files = sorted(p.name for p in ours.iterdir())
    assert files == sorted(p.name for p in theirs.iterdir()) and "model.safetensors" in files
    for name in files:
        assert (ours / name).read_bytes() == (theirs / name).read_bytes(), name


def test_command_line_exits_with_the_message(tmp_path, monkeypatch):
    src = tmp_path / "src"
    write_tiny_checkpoint(src, seed=5)
    monkeypatch.setattr(uaq, "uaq_tensors", wrong_selection("no gains"))
    with pytest.raises(SystemExit, match="rescaling by 2 changed the logits"):
        uaq.main(["--model", str(src), "--s", "1.5", "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
