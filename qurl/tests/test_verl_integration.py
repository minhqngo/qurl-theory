"""qurl.verl in VeRL v0.9.1: the plugin import, the optimizer patch switch and the vLLM worker extension."""
from types import SimpleNamespace

import numpy as np
import pytest

from qurl_test_support import (Dims, live_rows, np_quantize_rows, other_tensors, stand_in_model, sync_buckets,
                               trajectory)

pytestmark = pytest.mark.verl
DIMS, LAYERS = Dims(), 3


def test_plugin_registers_the_loss():
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    import qurl.verl

    assert POLICY_LOSS_REGISTRY["qurl_acr"] is qurl.verl.loss.compute_policy_loss_qurl_acr


def test_optimizer_patch_follows_qurl_opt(monkeypatch):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    from qurl.verl import optim

    for name in ("_build_optimizer", "optimizer_step", "_qurl_patched"):   # restored after the test
        monkeypatch.setattr(FSDPEngine, name, getattr(FSDPEngine, name, False), raising=False)
    original_step = FSDPEngine.optimizer_step
    optim.apply_if_enabled()
    assert not FSDPEngine._qurl_patched and FSDPEngine.optimizer_step is original_step
    monkeypatch.setenv("QURL_OPT", "bogus")
    with pytest.raises(ValueError, match="QURL_OPT must be"):
        optim.apply_if_enabled()
    monkeypatch.setenv("QURL_OPT", "plain")
    optim.apply_if_enabled()
    assert FSDPEngine._qurl_patched and FSDPEngine.optimizer_step is not original_step
    patched = FSDPEngine.optimizer_step
    optim.apply_if_enabled()   # once per process
    assert FSDPEngine.optimizer_step is patched


def extension(model):
    from qurl.verl.rollout import QuRLWorkerExtension

    ext = object.__new__(QuRLWorkerExtension)   # vLLM mixes the extension into its worker; only model_runner is used
    ext.model_runner = SimpleNamespace(model=model)
    return ext


def test_worker_extension_refits_and_hands_back_the_rest(tmp_path, monkeypatch):
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    from qurl.verl.rollout import QuRLWorkerExtension

    assert issubclass(QuRLWorkerExtension, vLLMColocateWorkerExtension)
    handed = []
    monkeypatch.setattr(vLLMColocateWorkerExtension, "_update_weights",
                        lambda self, weights, peft_config, base_sync_done: handed.extend(n for n, _ in weights))
    monkeypatch.setenv("QURL_E3_PATH", str(tmp_path / "e3.jsonl"))
    monkeypatch.setenv("QURL_RUN_ATTEMPT", "4")
    model = stand_in_model(DIMS, LAYERS, "cpu")
    ext = extension(model)
    traj = trajectory(DIMS, LAYERS, syncs=2)
    for k, weights in enumerate(traj, 1):
        for bucket in sync_buckets(weights, DIMS, LAYERS, seed=k):
            ext._update_weights(bucket, peft_config=None, base_sync_done=True)
    assert ext._qurl_refit.sync == 2 and ext._qurl_refit.count == 21
    assert sorted(handed) == sorted(2 * list(other_tensors(DIMS, LAYERS)))
    for (i, p), w in traj[-1].items():
        codes, _ = live_rows(model, i, p)
        assert np.array_equal(codes.numpy(), np_quantize_rows(w)[0])
    recs = (tmp_path / "e3.jsonl").read_text().splitlines()
    assert len(recs) == 2 and '"attempt": "4"' in recs[0]


def test_update_weights_from_ipc_skips_process_weights_after_loading(monkeypatch):
    import vllm.model_executor.model_loader.utils as loader_utils
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    original = loader_utils.process_weights_after_loading
    seen = {}

    def fake_sync(self, *args, **kwargs):
        seen["during"] = loader_utils.process_weights_after_loading
        seen["call"] = loader_utils.process_weights_after_loading("model", "config", "device")
        seen["args"] = (args, kwargs)
        if kwargs.get("fail"):
            raise RuntimeError("sync failed")
        return "done"

    monkeypatch.setattr(vLLMColocateWorkerExtension, "update_weights_from_ipc", fake_sync)
    ext = extension(None)
    assert ext.update_weights_from_ipc(peft_config=None, base_sync_done=True) == "done"
    assert seen["during"] is not original and seen["call"] is None
    assert seen["args"] == ((), {"peft_config": None, "base_sync_done": True})
    assert loader_utils.process_weights_after_loading is original
    with pytest.raises(RuntimeError, match="sync failed"):
        ext.update_weights_from_ipc(fail=True)
    assert loader_utils.process_weights_after_loading is original
