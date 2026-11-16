"""qurl.refit: the INT8 refit of the rollout model at every weight sync, on stand-in vLLM INT8 layers."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qurl.refit import Int8Refit, match_linear
from qurl_test_support import (Dims, live_rows, load_original_int8_rollout, np_quantize_rows, other_tensors,
                               stand_in_model, sync_buckets, trajectory)

DIMS = Dims()
LAYERS = 3


def run(refit, model, traj, device, first_seed=1):
    """Every sync of `traj` through the refit, bucket by bucket; returns the names handed back, per sync."""
    rest = []
    for k, weights in enumerate(traj, first_seed):
        handed = []
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k, device=device):
            handed += [n for n, _ in refit.load(bucket, dict(model.named_modules()))]
        rest.append(handed)
    return rest


def check_live_codes(model, weights):
    """The live INT8 layers hold an independent quantization of the synced weights."""
    for (i, p), w in weights.items():
        codes, scale = live_rows(model, i, p)
        ref_codes, ref_scale = np_quantize_rows(w)
        assert np.array_equal(codes.cpu().numpy(), ref_codes) and np.array_equal(scale.cpu().numpy(), ref_scale)


def test_match_linear():
    t = match_linear("model.layers.12.self_attn.k_proj.weight")
    assert (t.module, t.layer, t.proj, t.shard) == ("model.layers.12.self_attn.qkv_proj", 12, "k_proj", 1)
    t = match_linear("model.layers.0.mlp.down_proj.weight")
    assert (t.module, t.shard) == ("model.layers.0.mlp.down_proj", None)
    for name in ("model.layers.0.self_attn.q_proj.bias", "model.layers.0.input_layernorm.weight", "lm_head.weight",
                 "model.embed_tokens.weight"):
        assert match_linear(name) is None


def test_refit_requantizes_every_weight_at_every_sync(device, capsys):
    """BF16 starting weights (with exact ties), then FP32 weights after three updates: after every sync the live
    layers hold an independent quantization of the synced weights, and the other tensors are handed back."""
    model = stand_in_model(DIMS, LAYERS, device)
    refit = Int8Refit(dict(model.named_modules()))
    traj = trajectory(DIMS, LAYERS)
    for k, weights in enumerate(traj, 1):
        (rest,) = run(refit, model, [weights], device, first_seed=k)
        check_live_codes(model, weights)
        assert sorted(rest) == sorted(other_tensors(DIMS, LAYERS))
        assert (refit.sync, refit.count) == (k, 21)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"[qurl int8 refit] pid {os.getpid()} sync {k}: 21 linear weights re-quantized to INT8 "
                     f"(synced weights arrive as torch.bfloat16; INT8 kernel StandInKernel)" for k in (1, 2, 3)]


def test_refuses_tensor_parallelism():
    with pytest.raises(RuntimeError, match=r"tensor parallelism \(2\) is not supported"):
        Int8Refit(dict(stand_in_model(DIMS, LAYERS, "cpu").named_modules()), tp_world_size=2)


def test_refuses_a_model_without_7_int8_weights_per_layer():
    model = stand_in_model(DIMS, LAYERS, "cpu")
    model.model.layers[1].mlp.gate_up_proj = torch.nn.Linear(DIMS.hidden, 2 * DIMS.inter)
    with pytest.raises(RuntimeError, match="expected 7 INT8 weights per layer, found 19 for 3 layers"):
        Int8Refit(dict(model.named_modules()))


def test_refuses_an_incomplete_sync():
    model = stand_in_model(DIMS, LAYERS, "cpu")
    refit = Int8Refit(dict(model.named_modules()))
    traj = trajectory(DIMS, LAYERS)
    run(refit, model, traj[:1], "cpu")
    missing = dict(traj[1])
    del missing[(1, "down_proj")]
    run(refit, model, [missing], "cpu", first_seed=2)
    assert (refit.sync, refit.count) == (2, 20)
    first = ("model.layers.0.self_attn.q_proj.weight", traj[2][(0, "q_proj")])
    with pytest.raises(RuntimeError, match=r"sync 2 re-quantized 20 of 21 linear weights before "
                                           r"'model.layers.0.self_attn.q_proj.weight' arrived again"):
        refit.load([first], dict(model.named_modules()))


def test_refuses_a_weight_without_an_int8_layer():
    model = stand_in_model(DIMS, LAYERS, "cpu")
    refit = Int8Refit(dict(model.named_modules()))
    with pytest.raises(RuntimeError, match="no INT8 layer 'model.layers.7.self_attn.qkv_proj'"):
        refit.load([("model.layers.7.self_attn.q_proj.weight", torch.zeros(DIMS.q, DIMS.hidden))],
                   dict(model.named_modules()))
    with pytest.raises(AssertionError):   # a weight whose shape does not fit the layer
        refit.load([("model.layers.0.self_attn.q_proj.weight", torch.zeros(DIMS.q, DIMS.hidden + 1))],
                   dict(model.named_modules()))


def int8_state(model):
    return {n: (m.codes().cpu().clone(), m.weight_scale.data.cpu().clone()) for n, m in model.named_modules()
            if hasattr(m, "logical_widths")}


@pytest.mark.originals
def test_refit_matches_original(device, monkeypatch, capsys):
    """The same syncs through the experiment code's worker extension and through Int8Refit: the same live codes and
    scales after every sync, the same tensors handed back to VeRL, the same printed lines."""
    original = load_original_int8_rollout(monkeypatch)
    ext = original.QuRLWorkerExtension()
    ext.model_runner = SimpleNamespace(model=stand_in_model(DIMS, LAYERS, device))
    model = stand_in_model(DIMS, LAYERS, device)
    refit = Int8Refit(dict(model.named_modules()))
    for k, weights in enumerate(trajectory(DIMS, LAYERS), 1):
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k, device=device):
            before = len(getattr(ext, "loaded", []))
            ext._update_weights(bucket, peft_config=None, base_sync_done=True)
            rest = refit.load(bucket, dict(model.named_modules()))
            assert [n for n, _ in rest] == getattr(ext, "loaded", [])[before:]
        ours, theirs = int8_state(model), int8_state(ext.model_runner.model)
        assert all(torch.equal(ours[n][0], theirs[n][0]) and torch.equal(ours[n][1], theirs[n][1]) for n in ours)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6 and lines[0::2] == lines[1::2]
