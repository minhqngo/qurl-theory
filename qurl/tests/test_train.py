"""qurl.train: the QuRL train + validation loop, on a tiny random Qwen2 and a toy task (CPU, offline)."""
import json

import pytest
import torch
from safetensors.torch import load_file

from qurl.gsm8k import Problem
from qurl.train import Task, TrainConfig, check_config, grpo_advantages, parse_args, train
from qurl_test_support import write_tiny_checkpoint

TIMING = ("rollout_s", "update_s", "val_s")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    write_tiny_checkpoint(path, seed=0)
    return path


def toy_task():
    """A reward the tiny random model can earn: the parity of its response's length in characters, which varies
    within a group, so the advantages and gradients are not zero."""
    return Task(train=[Problem(f"clips {i}", i) for i in range(20)], val=[Problem(f"friends {i}", i) for i in range(5)],
                prompt=lambda q: q, reward=lambda text, gold: float(len(text) % 2))


def config(checkpoint, out, **kw):
    base = dict(model=str(checkpoint), out=str(out), steps=3, prompts=4, group=2, max_new=6, minibatches=2,
                micro_batch=2, lr=1e-3, val_every=2, val_size=4, val_batch=3, val_max_new=4, val_policy="both",
                device="cpu", attn="eager")
    base.update(kw)
    return TrainConfig(**base)


def weights(path):
    return {k: v.float() for k, v in load_file(path / "model.safetensors").items()}


def test_trains_validates_and_saves(checkpoint, tmp_path):
    records = train(config(checkpoint, tmp_path / "run"), toy_task())
    kinds = [(r["kind"], r["step"]) for r in records]
    assert kinds == [("val", 0), ("train", 1), ("train", 2), ("val", 2), ("train", 3), ("val", 3)]
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == records
    step = records[1]
    for key in ("reward", "response_len", "loss", "grad_norm", "pg_clipfrac", "pg_clipfrac_lower", "ppo_kl",
                "acr_kept_frac", "tis_trunc_frac", "rho_max", "kl_behav_prox", "sample"):
        assert key in step, key
    assert {"val/fp/acc", "val/int8/acc", "val/fp/response_len", "val/int8/response_len"} <= set(records[0])
    assert json.loads((tmp_path / "run" / "config.json").read_text())["steps"] == 3
    final = tmp_path / "run" / "final"
    assert json.loads((final / "qurl.json").read_text()) == {"step": 3, "uaq_s": None}
    train_steps = [r for r in records if r["kind"] == "train"]
    assert any(0 < r["reward"] < 1 for r in train_steps) and any(r["grad_norm"] > 0 for r in train_steps)
    before, after = weights(checkpoint), weights(final)
    assert any(not torch.equal(before[k], after[k]) for k in before)   # the policy was updated


def test_zero_learning_rate_keeps_every_ratio_at_one(checkpoint, tmp_path):
    """pi_old is computed on the update's own micro-batches, so before any change the ratio is exactly 1."""
    records = train(config(checkpoint, tmp_path / "run", lr=0.0, weight_decay=0.0, val_before_train=False,
                           val_every=0), toy_task())
    steps = [r for r in records if r["kind"] == "train"]
    assert all(r["ppo_kl"] == 0.0 and r["pg_clipfrac"] == 0.0 for r in steps)
    assert any(r["grad_norm"] > 0 for r in steps)   # gradients flowed; the step size was zero
    before, after = weights(checkpoint), weights(tmp_path / "run" / "final")
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_runs_are_reproducible(checkpoint, tmp_path):
    def run(name):
        recs = train(config(checkpoint, tmp_path / name, steps=2, val_every=0), toy_task())
        return [{k: v for k, v in r.items() if k not in TIMING} for r in recs], weights(tmp_path / name / "final")

    (a, wa), (b, wb) = run("a"), run("b")
    assert a == b and all(torch.equal(wa[k], wb[k]) for k in wa)


def test_gradient_checkpointing_changes_memory_not_results(checkpoint, tmp_path):
    def run(name, ckpt):
        recs = train(config(checkpoint, tmp_path / name, steps=2, val_every=0, grad_ckpt=ckpt), toy_task())
        return [{k: v for k, v in r.items() if k not in TIMING} for r in recs], weights(tmp_path / name / "final")

    (a, wa), (b, wb) = run("plain", False), run("ckpt", True)
    assert a == b and all(torch.equal(wa[k], wb[k]) for k in wa)


def test_uaq_at_s_2_leaves_the_first_rollout_unchanged(checkpoint, tmp_path):
    """At s = 2 the rescaling is exact: the first INT8 rollout (tokens and mu) and pi_old are the same as without
    UAQ, so are the rollout statistics of step 1; the updates then differ (UAQ changes the effective learning rates)."""
    kw = dict(steps=1, val_before_train=False, val_every=0, val_policy="int8")
    plain = train(config(checkpoint, tmp_path / "plain", **kw), toy_task())
    with_uaq = train(config(checkpoint, tmp_path / "uaq", uaq_s=2.0, **kw), toy_task())
    assert with_uaq[0]["kind"] == "uaq" and with_uaq[0]["probe_s2_bit_identical"] and with_uaq[0]["bit_identical"]
    p, u = plain[0], with_uaq[1]
    for key in ("reward", "response_len", "kl_behav_prox", "tis_trunc_frac", "rho_max", "sample"):
        assert p[key] == u[key], key
    assert json.loads((tmp_path / "uaq" / "final" / "qurl.json").read_text())["uaq_s"] == 2.0


def test_full_precision_rollout_has_no_mismatch(checkpoint, tmp_path):
    records = train(config(checkpoint, tmp_path / "run", quantize_rollout=False, steps=1, val_before_train=False,
                           val_policy="fp"), toy_task())
    step = records[0]
    assert abs(step["kl_behav_prox"]) < 1e-5 and step["tis_trunc_frac"] == 0.0 and step["rho_max"] < 1.001


def test_grpo_advantages():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    a = grpo_advantages(r, group=4)
    torch.testing.assert_close(a[:4], (r[:4] - 0.75) / (r[:4].std() + 1e-6))
    torch.testing.assert_close(a[4:], (r[4:] - 0.25) / (r[4:].std() + 1e-6))
    assert torch.equal(grpo_advantages(torch.ones(4), group=2), torch.zeros(4))   # equal rewards: no signal
    torch.testing.assert_close(grpo_advantages(r, group=4, std=False)[:4], r[:4] - 0.75)


def test_config_checks_and_command_line():
    with pytest.raises(ValueError, match="multiple of minibatches x micro_batch"):
        check_config(TrainConfig(prompts=3, group=2, minibatches=2, micro_batch=2))
    with pytest.raises(ValueError, match="group must be at least 2"):
        check_config(TrainConfig(group=1))
    cfg = parse_args(["--steps", "7", "--uaq-s", "1.5", "--no-quantize-rollout", "--clip-high", "0.28"])
    assert (cfg.steps, cfg.uaq_s, cfg.quantize_rollout, cfg.clip_high) == (7, 1.5, False, 0.28)
    assert parse_args([]) == TrainConfig()
