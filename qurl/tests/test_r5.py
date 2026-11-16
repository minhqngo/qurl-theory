"""qurl.r5 and qurl.verl.optim: Theorem 2(b)-(c)'s optimizer (arm R5), bit-identical to UAQ (arm R4) at s = 2."""
import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

from qurl import r5
from qurl_test_support import load_original_r5, r5_rank_worker, tiny_qwen2, to_uaq


def test_settings_from_the_environment(monkeypatch):
    assert r5.settings() is None
    monkeypatch.setenv("QURL_S", "2")
    monkeypatch.setenv("QURL_OPT", "plain")
    assert r5.settings() == ("plain", 1.0)
    monkeypatch.setenv("QURL_OPT", "r5")
    assert r5.settings() == ("r5", 2.0)
    monkeypatch.delenv("QURL_S")
    assert r5.settings() == ("r5", 1.5)
    monkeypatch.setenv("QURL_OPT", "R5")
    with pytest.raises(ValueError, match="QURL_OPT must be"):
        r5.settings()


def test_group_of():
    groups = {"model.layers.3.self_attn.q_proj.weight": "W", "model.layers.3.self_attn.k_proj.weight": "W",
              "_fsdp_wrapped_module.model.layers.0.self_attn.v_proj.weight": "W",
              "model.layers.3.self_attn.q_proj.bias": "rest", "model.layers.3.self_attn.o_proj.weight": "rest",
              "model.layers.3.input_layernorm.weight": "gamma",
              "model.layers.3.post_attention_layernorm.weight": "rest", "model.norm.weight": "rest",
              "lm_head.weight": "rest"}
    assert {n: r5.group_of(n) for n in groups} == groups


def test_param_groups_and_clip_factors():
    named = list(tiny_qwen2().named_parameters())
    groups, sizes = r5.param_groups(named, 1e-6, 1e-8, 0.1, 1.5)
    assert sizes == {"W": 6, "gamma": 2, "rest": len(named) - 8}
    rest, w, gamma = ((g["lr"], g["eps"], g["weight_decay"]) for g in groups)
    assert rest == (1e-6, 1e-8, 0.1)
    assert w == (1.5 * 1e-6, 1e-8 / 1.5, 0.1 / 1.5)
    assert gamma == (1e-6 / 1.5, 1.5 * 1e-8, 1.5 * 0.1)
    factors = r5.clip_factors(named, 1.5)
    assert [p for p, _ in factors] == [p for _, p in named]
    assert [f for _, f in factors] == [{"W": 1.5, "gamma": 1 / 1.5, "rest": 1.0}[r5.group_of(n)] for n, _ in named]
    with pytest.raises(RuntimeError, match="no q/k/v_proj or input_layernorm"):
        r5.param_groups([(n, p) for n, p in named if "layernorm" not in n], 1e-6, 1e-8, 0.1, 1.5)


def batch(step, rank=0):
    g = torch.Generator().manual_seed(1000 * rank + step)
    ids = torch.randint(0, 96, (3, 17), generator=g)
    return ids[:, :-1], ids[:, 1:]


def loss_of(model, step):
    x, y = batch(step)
    logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def test_total_norm_and_clip_and_step():
    model = tiny_qwen2(seed=1)
    named = list(model.named_parameters())
    assert r5.total_norm(r5.clip_factors(named, 1.0)) is None   # no gradients yet
    loss_of(model, 0).backward()
    for s in (1.0, 1.5, 2.0):
        got = r5.total_norm(r5.clip_factors(named, s))
        ref = torch.sqrt(sum((f * p.grad.double().norm()) ** 2 for p, f in r5.clip_factors(named, s)))
        assert got.dtype == torch.float64 and abs(float(got) - float(ref)) / float(ref) < 1e-6
    grads = [p.grad.clone() for _, p in named]
    weights = [p.detach().clone() for _, p in named]
    opt = torch.optim.SGD([p for _, p in named], lr=0.0)
    total = r5.clip_and_step(opt, r5.clip_factors(named, 1.0), 0.01)
    coef = float(torch.clamp(0.01 / (torch.tensor(total, dtype=torch.float64) + 1e-6), max=1.0))
    assert coef < 1 and all(torch.equal(p.grad, g * coef) for (_, p), g in zip(named, grads))
    # a non-finite norm: no step, and the gradients are dropped
    named[0][1].grad[0] = float("inf")
    opt = torch.optim.SGD([p for _, p in named], lr=1.0)
    assert r5.clip_and_step(opt, r5.clip_factors(named, 1.0), 0.01) == float("inf")
    assert all(p.grad is None for _, p in named)
    assert all(torch.equal(p, w) for (_, p), w in zip(named, weights))


def train(arm, impl=r5, s=2.0, steps=3, lr=1e-2, eps=1e-8, wd=0.1, clip=0.05):
    """K AdamW steps of one arm on the same batches; returns the weights in original coordinates and the norms.

    R4: UAQ weights (W / s, gamma * s), plain settings, ordinary clip. R5: original weights, Theorem 2(b) settings,
    clip rule (c). R5o: R5's settings with the ordinary clip. R4r: R4 again (determinism).
    """
    model = tiny_qwen2(seed=0)
    if arm in ("R4", "R4r"):
        to_uaq(model, s)
    named = list(model.named_parameters())
    groups, _ = impl.param_groups(named, lr, eps, wd, s if arm in ("R5", "R5o") else 1.0)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    clipf = impl.clip_factors(named, s if arm == "R5" else 1.0)
    norms = []
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_of(model, k).backward()
        norms.append(impl.clip_and_step(opt, clipf, clip))
    back = s if arm in ("R4", "R4r") else 1.0
    return {n: r5.to_original_coordinates(n, p.detach(), back) for n, p in named}, norms


def test_r5_equals_r4_bit_for_bit_at_s_2():
    ref, ref_norms = train("R4")
    assert all(x > 0.05 for x in ref_norms)   # clipping binds at every step
    for arm, identical in (("R4r", True), ("R5", True), ("R5o", False)):
        params, norms = train(arm)
        assert all(torch.equal(ref[n], params[n]) for n in ref) == identical, arm
        assert (norms == ref_norms) == identical, arm


@pytest.mark.originals
@pytest.mark.parametrize("arm", ["R4", "R5", "R5o"])
@pytest.mark.parametrize("s", [1.5, 2.0])
def test_matches_original(arm, s, monkeypatch):
    original = load_original_r5(monkeypatch)
    ours, ours_norms = train(arm, s=s)
    theirs, their_norms = train(arm, impl=original, s=s)
    assert ours_norms == their_norms and all(torch.equal(ours[n], theirs[n]) for n in ours)


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 3])
def test_several_ranks_fsdp2(world, tmp_path):
    """FSDP2 on the CPU (gloo): the clip norm over sharded gradients (3 ranks: uneven shards) matches a float64
    reference identically on every rank, and R5 equals R4 bit for bit at s = 2."""
    import torch.multiprocessing as mp

    result = tmp_path / "result.json"
    mp.spawn(r5_rank_worker, args=(world, str(tmp_path / "init"), 3, str(result)), nprocs=world, join=True)
    res = json.loads(result.read_text())
    assert res["norms_ok"], res["norm_checks"]
    assert res["arms"] == {"R4r": {"bit_identical": True, "norms_equal": True},
                           "R5": {"bit_identical": True, "norms_equal": True},
                           "R5o": {"bit_identical": False, "norms_equal": False}}
    assert res["clip_bound"]


@contextlib.contextmanager
def patched_engine(patch):
    """VeRL's FSDPEngine with `patch` applied; the class is restored afterwards."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    names = ("_build_optimizer", "optimizer_step", "_qurl_patched")
    saved = {n: FSDPEngine.__dict__[n] for n in names if n in FSDPEngine.__dict__}
    assert not saved.get("_qurl_patched", False), "FSDPEngine was patched outside the test"
    try:
        patch()
        yield FSDPEngine._build_optimizer, FSDPEngine.optimizer_step
    finally:
        for n in names:
            if n in saved:
                setattr(FSDPEngine, n, saved[n])
            elif n in FSDPEngine.__dict__:
                delattr(FSDPEngine, n)


def engine_run(patch, steps=3):
    """Build the actor optimizer and take K steps through VeRL's FSDPEngine methods as patched, on a stand-in
    engine (a plain module: the patched methods only need its configs)."""
    from verl.workers.config import FSDPOptimizerConfig

    with patched_engine(patch) as (build, step):
        model = tiny_qwen2(seed=0)
        engine = SimpleNamespace(model_config=SimpleNamespace(model_type="language_model"),
                                 engine_config=SimpleNamespace(forward_only=False),
                                 optimizer_config=FSDPOptimizerConfig(lr=1e-2, weight_decay=0.1, clip_grad=0.05,
                                                                      lr_scheduler_type="constant"))
        engine.optimizer = build(engine, model)
        norms = []
        for k in range(steps):
            engine.optimizer.zero_grad(set_to_none=True)
            loss_of(model, k).backward()
            norms.append(step(engine))
        groups = [(g["lr"], g["eps"], g["weight_decay"], len(g["params"])) for g in engine.optimizer.param_groups]
        return {n: p.detach().clone() for n, p in model.named_parameters()}, norms, groups


@pytest.mark.verl
@pytest.mark.originals
@pytest.mark.parametrize("env", [{"QURL_OPT": "r5", "QURL_S": "2"}, {"QURL_OPT": "r5", "QURL_S": "1.5"},
                                 {"QURL_OPT": "plain"}, {"QURL_OPT": "r5", "QURL_S": "1.5", "QURL_CLIP": "ordinary"}])
def test_verl_engine_patch_matches_original(env, monkeypatch, capsys):
    from qurl.verl import optim   # imported before QURL_OPT is set: importing qurl.verl must not patch VeRL here

    original = load_original_r5(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ours = engine_run(optim.patch)
    out_ours = capsys.readouterr().out
    theirs = engine_run(original._patch)
    out_theirs = capsys.readouterr().out
    assert ours[1] == theirs[1] and ours[2] == theirs[2]
    assert all(torch.equal(ours[0][n], theirs[0][n]) for n in ours[0])
    assert out_ours == out_theirs and out_ours.startswith(f"[qurl optimizer] mode {env['QURL_OPT']}")
    s = float(env.get("QURL_S", "1")) if env["QURL_OPT"] == "r5" else 1.0
    assert ours[2][1][:3] == (s * 1e-2, 1e-8 / s, 0.1 / s)
