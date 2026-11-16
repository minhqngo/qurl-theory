"""qurl.overrides: the Hydra overrides that wire the package into VeRL, and the INT8 W8A8 config they carry."""
import importlib
import json
from pathlib import Path

import pytest

from qurl import overrides, refit
from qurl_test_support import ORIGINAL_PKG


@pytest.mark.originals
def test_config_is_the_experiments_config():
    assert Path(refit.CONFIG_PATH).read_bytes() == (ORIGINAL_PKG / "int8_w8a8_config.json").read_bytes()


def test_hf_overrides_json():
    cfg = refit.quantization_config()
    assert cfg["quant_method"] == "compressed-tensors" and cfg["ignore"] == ["lm_head"]
    group = cfg["config_groups"]["group_0"]
    assert (group["weights"]["strategy"], group["input_activations"]["strategy"]) == ("channel", "token")
    assert group["input_activations"]["dynamic"] and cfg["kv_cache_scheme"] is None
    # what the experiments' run script passed: the config and the context length, as compact JSON
    want = json.dumps({"quantization_config": cfg, "max_position_embeddings": 32768}, separators=(",", ":"))
    assert overrides.hf_overrides_json(32768) == want
    assert overrides.hf_overrides() == {"quantization_config": cfg}


def test_hydra_overrides():
    args = overrides.hydra_overrides(8192)
    assert args == [
        "actor_rollout_ref.model.external_lib=qurl.verl",
        "actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=1e30",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{overrides.hf_overrides_json(8192)}'",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl.verl.rollout.QuRLWorkerExtension",
    ]


def test_command_line(capsys):
    overrides.main(["--hf-overrides", "--max-position-embeddings", "8192"])
    assert capsys.readouterr().out == overrides.hf_overrides_json(8192) + "\n"
    overrides.main(["--max-position-embeddings", "8192"])
    assert capsys.readouterr().out.splitlines() == overrides.hydra_overrides(8192)


@pytest.mark.verl
def test_hydra_parses_them_and_the_named_code_resolves():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    args = overrides.hydra_overrides(8192)
    parsed = {o.key_or_group: o.value() for o in OverridesParser.create().parse_overrides(args)}
    assert json.loads(parsed["actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"]) == \
        overrides.hf_overrides(8192)
    module, cls = parsed["actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls"].rsplit(".", 1)
    assert getattr(importlib.import_module(module), cls).__name__ == "QuRLWorkerExtension"
    importlib.import_module(parsed["actor_rollout_ref.model.external_lib"])
