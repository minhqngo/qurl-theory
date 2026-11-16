"""qurl.refit and qurl.synclog: the INT8 refit on stand-in vLLM INT8 layers, its checks, its dump and its log."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qurl.refit import Int8Refit, RefitConfig, match_linear
from qurl.synclog import PROJS, QKV
from qurl_test_support import (Dims, linear_shapes, live_rows, load_original_int8_rollout, np_quantize_rows,
                               np_tie_mask, other_tensors, stand_in_model, sync_buckets, trajectory)

DIMS = Dims()
LAYERS = 3   # tracked layers 0, 1, 2


def run(refit, model, traj, device, first_seed=1):
    """Every sync of `traj` through the refit, bucket by bucket; returns the names handed back, per sync."""
    rest = []
    for k, weights in enumerate(traj, first_seed):
        handed = []
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k, device=device):
            handed += [n for n, _ in refit.load(bucket, dict(model.named_modules()))]
        rest.append(handed)
    return rest


def expected_record(traj, k, start_step=0):
    """Sync k's log fields (k = 1, 2, ...), recomputed with numpy from the synced weights."""
    cur = {key: np_quantize_rows(w) for key, w in traj[k - 1].items()}
    prev = {key: np_quantize_rows(w) for key, w in traj[k - 2].items()} if k > 1 else None
    ties = {key: np.flatnonzero(np_tie_mask(w)) for key, w in traj[0].items()} if start_step == 0 else {}
    exp = {f: {p: [] for p in PROJS} for f in ("flip", "scale_changed", "ties", "flip_ties", "qerr")}
    pred = {p: [] for p in QKV}
    obs = {p: [] for p in QKV}
    for p in PROJS:
        for i in range(LAYERS):
            codes, scale = cur[(i, p)]
            x = traj[k - 1][(i, p)].float().numpy()
            res = (x - codes.astype(np.float32) * scale).astype(np.float64)
            exp["qerr"][p].append(float(np.sqrt((res ** 2).sum() / (x.astype(np.float64) ** 2).sum())))
            t = ties.get((i, p))
            exp["ties"][p].append(t.size / codes.size if t is not None else None)
            if prev is None:
                for f in ("flip", "scale_changed", "flip_ties"):
                    exp[f][p].append(None)
                continue
            changed = codes != prev[(i, p)][0]
            exp["flip"][p].append(np.count_nonzero(changed) / changed.size)
            exp["scale_changed"][p].append(np.count_nonzero(scale != prev[(i, p)][1]) / scale.size)
            exp["flip_ties"][p].append(np.count_nonzero(changed.ravel()[t]) / t.size if t is not None and t.size
                                       else None)
            if p in QKV:   # all layers are tracked here
                x0 = traj[k - 2][(i, p)].float().numpy().astype(np.float64)
                pred[p].append(float(np.minimum(np.abs(x - x0) / scale, 1.0).mean()))
                obs[p].append(np.count_nonzero(changed) / changed.size)
    return exp, pred, obs


def check_record(rec, traj, k, start_step=0, attempt="0"):
    exp, pred, obs = expected_record(traj, k, start_step)
    assert (rec["sync"], rec["attempt"], rec["start_step"], rec["layers"]) == (k, attempt, start_step, LAYERS)
    assert rec["codes_per_layer"] == {p: r * c for p, (r, c) in linear_shapes(DIMS).items()}
    for field in ("flip", "scale_changed", "ties", "flip_ties"):   # shares of integer counts: exact
        assert rec[field] == exp[field], field
    for p in PROJS:
        np.testing.assert_allclose(rec["qerr"][p], exp["qerr"][p], rtol=1e-12)
    assert rec["pred"]["layers"] == list(range(LAYERS))
    for p in QKV:
        if k == 1:
            assert rec["pred"][p] == [None] * LAYERS and rec["pred"]["observed_all_rows"][p] == [None] * LAYERS
        else:   # Proposition 4's prediction (an FP32 mean), and the flip share recomputed from the previous weights
            np.testing.assert_allclose(rec["pred"][p], pred[p], rtol=1e-5)
            np.testing.assert_allclose(rec["pred"]["observed_all_rows"][p], obs[p], rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(rec["pred"]["observed_all_rows"][p], rec["flip"][p], rtol=1e-6, atol=1e-9)


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


def test_refit_writes_every_weight(device, capsys):
    model = stand_in_model(DIMS, LAYERS, device)
    refit = Int8Refit(dict(model.named_modules()))
    traj = trajectory(DIMS, LAYERS)
    rest = run(refit, model, traj[:1], device)
    check_live_codes(model, traj[0])
    assert sorted(rest[0]) == sorted(other_tensors(DIMS, LAYERS)) and refit.log is None and refit.dump_dir is None
    for k, weights in enumerate(traj[1:], 2):
        run(refit, model, [weights], device, first_seed=k)
        check_live_codes(model, weights)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"[qurl int8 refit] pid {os.getpid()} sync {k}: 21 linear weights re-quantized to INT8 "
                     f"(synced weights arrive as torch.bfloat16; INT8 kernel StandInKernel)" for k in (1, 2, 3)]


def test_log_equals_independent_recomputation(device, tmp_path):
    path = tmp_path / "log" / "e3.jsonl"
    model = stand_in_model(DIMS, LAYERS, device)
    refit = Int8Refit(dict(model.named_modules()), config=RefitConfig(log_path=str(path), attempt="7"))
    traj = trajectory(DIMS, LAYERS)
    run(refit, model, traj, device)
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(recs) == 3
    for k, rec in enumerate(recs, 1):
        check_record(rec, traj, k, attempt="7")
    # the planted ties are there, and the first (tiny) update flips about half of them
    n_ties = sum(t * (r * c) for p, (r, c) in linear_shapes(DIMS).items() for t in recs[0]["ties"][p])
    assert n_ties > 100
    assert 0.3 < np.mean([x for p in PROJS for x in recs[1]["flip_ties"][p]]) < 0.7
    assert (tmp_path / "log" / "e3.jsonl.7.claim").read_text() == f"{os.getpid()}\n"


def test_resumed_attempt_logs_no_ties(tmp_path):
    path = tmp_path / "e3.jsonl"
    model = stand_in_model(DIMS, LAYERS, "cpu")
    refit = Int8Refit(dict(model.named_modules()),
                      config=RefitConfig(log_path=str(path), attempt="8", start_step=20))
    traj = trajectory(DIMS, LAYERS)
    run(refit, model, traj, "cpu")
    for k, line in enumerate(path.read_text().splitlines(), 1):
        rec = json.loads(line)
        check_record(rec, traj, k, start_step=20, attempt="8")
        assert all(x is None for p in PROJS for x in rec["ties"][p] + rec["flip_ties"][p])


def test_one_replica_claims_the_log_and_the_dump(tmp_path):
    cfg = RefitConfig(log_path=str(tmp_path / "e3.jsonl"), dump_dir=str(tmp_path / "dump"), attempt="1")
    first, second = (Int8Refit(dict(stand_in_model(DIMS, LAYERS, "cpu").named_modules()), config=cfg)
                     for _ in range(2))
    assert first.log is not None and first.dump_dir is not None
    assert second.log is None and second.dump_dir is None
    resumed = Int8Refit(dict(stand_in_model(DIMS, LAYERS, "cpu").named_modules()),
                        config=RefitConfig(log_path=cfg.log_path, attempt="2"))
    assert resumed.log is not None   # a new attempt claims the log afresh


def test_dump_matches_independent_quantization(device, tmp_path):
    dump = tmp_path / "dump"
    model = stand_in_model(DIMS, LAYERS, device)
    refit = Int8Refit(dict(model.named_modules()), config=RefitConfig(dump_dir=str(dump), attempt="1"))
    traj = trajectory(DIMS, LAYERS)
    run(refit, model, traj, device)
    assert sorted(os.listdir(dump)) == ["claim.1", "sync1_qkv_layer0.pt", "sync2_qkv_layer0.pt"]
    for k in (1, 2):
        d = torch.load(dump / f"sync{k}_qkv_layer0.pt")
        codes, scale = np_quantize_rows(torch.cat([traj[k - 1][(0, p)].float() for p in QKV]))
        assert np.array_equal(d["codes"].numpy(), codes) and np.array_equal(d["scale"].numpy(), scale)
        assert d["logical_widths"] == [DIMS.q, DIMS.kv, DIMS.kv]
        assert d["module"] == "model.layers.0.self_attn.qkv_proj"
    later = Int8Refit(dict(stand_in_model(DIMS, LAYERS, "cpu").named_modules()),
                      config=RefitConfig(dump_dir=str(dump), attempt="2"))
    assert later.dump_dir is None   # the dump is of the run's starting weights: a resumed attempt does not redo it


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


def original_extension(original, device):
    ext = original.QuRLWorkerExtension()
    ext.model_runner = SimpleNamespace(model=stand_in_model(DIMS, LAYERS, device))
    return ext


def int8_state(model):
    return {n: (m.codes().cpu().clone(), m.weight_scale.data.cpu().clone()) for n, m in model.named_modules()
            if hasattr(m, "logical_widths")}


@pytest.mark.originals
def test_refit_matches_original(tmp_path, monkeypatch, capsys):
    """Without the log (the original's log needs CUDA): same live codes and scales, same tensors handed back, same
    printed lines, same dump."""
    monkeypatch.setenv("QURL_INT8_DUMP_DIR", str(tmp_path / "theirs"))
    monkeypatch.setenv("QURL_RUN_ATTEMPT", "3")
    original = load_original_int8_rollout(monkeypatch)
    ext = original_extension(original, "cpu")
    model = stand_in_model(DIMS, LAYERS, "cpu")
    refit = Int8Refit(dict(model.named_modules()), config=RefitConfig(dump_dir=str(tmp_path / "ours"), attempt="3"))
    for k, weights in enumerate(trajectory(DIMS, LAYERS), 1):
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k):
            before = len(getattr(ext, "loaded", []))
            ext._update_weights(bucket, peft_config=None, base_sync_done=True)
            rest = refit.load(bucket, dict(model.named_modules()))
            assert [n for n, _ in rest] == getattr(ext, "loaded", [])[before:]
        ours, theirs = int8_state(model), int8_state(ext.model_runner.model)
        assert all(torch.equal(ours[n][0], theirs[n][0]) and torch.equal(ours[n][1], theirs[n][1]) for n in ours)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6 and lines[0::2] == lines[1::2]
    for name in ("sync1_qkv_layer0.pt", "sync2_qkv_layer0.pt"):
        a, b = torch.load(tmp_path / "ours" / name), torch.load(tmp_path / "theirs" / name)
        assert a.keys() == b.keys() and torch.equal(a["codes"], b["codes"]) and torch.equal(a["scale"], b["scale"])
        assert (a["logical_widths"], a["module"]) == (b["logical_widths"], b["module"])


@pytest.mark.gpu
@pytest.mark.originals
@pytest.mark.parametrize("start_step", [0, 20])
def test_log_matches_original_on_cuda(start_step, tmp_path, monkeypatch):
    """The original's per-sync log (CUDA only) and ours, on the same syncs: identical records but for the time."""
    monkeypatch.setenv("QURL_E3_PATH", str(tmp_path / "theirs" / "e3.jsonl"))
    monkeypatch.setenv("QURL_RUN_ATTEMPT", "5")
    monkeypatch.setenv("QURL_START_STEP", str(start_step))
    original = load_original_int8_rollout(monkeypatch)
    ext = original_extension(original, "cuda")
    model = stand_in_model(DIMS, LAYERS, "cuda")
    ours_path = tmp_path / "ours" / "e3.jsonl"
    refit = Int8Refit(dict(model.named_modules()),
                      config=RefitConfig(log_path=str(ours_path), attempt="5", start_step=start_step))
    for k, weights in enumerate(trajectory(DIMS, LAYERS), 1):
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k, device="cuda"):
            ext._update_weights(bucket, peft_config=None, base_sync_done=True)
            refit.load(bucket, dict(model.named_modules()))

    def records(path):
        return [{k: v for k, v in json.loads(line).items() if k != "time"} for line in path.read_text().splitlines()]

    ours, theirs = records(ours_path), records(tmp_path / "theirs" / "e3.jsonl")
    assert len(ours) == 3 and ours == theirs
