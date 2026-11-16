"""The Hydra overrides that wire this package into VeRL's trainer (python -m verl.trainer.main_ppo).

    python -m qurl.overrides --max-position-embeddings 32768                  # every override, one per line
    python -m qurl.overrides --hf-overrides --max-position-embeddings 32768   # only vLLM's hf_overrides JSON

In bash, `mapfile -t QURL_ARGS < <(python -m qurl.overrides ...)` and then pass "${QURL_ARGS[@]}" to the trainer. The
TIS cap C is not an override but the environment variable QURL_TIS_C (default 2.0).
"""
from __future__ import annotations

import argparse
import json

from .refit import quantization_config

PLUGIN = "qurl.verl"
WORKER_EXTENSION = "qurl.verl.rollout.QuRLWorkerExtension"
LOSS_MODE = "qurl_acr"


def hf_overrides(max_position_embeddings: int | None = None) -> dict:
    """vLLM's hf_overrides: the INT8 W8A8 quantization config and, if given, the context length.

    It replaces VeRL's own hf_overrides, so it carries the context length as well.
    """
    out = {"quantization_config": quantization_config()}
    if max_position_embeddings is not None:
        out["max_position_embeddings"] = int(max_position_embeddings)
    return out


def hf_overrides_json(max_position_embeddings: int | None = None) -> str:
    """hf_overrides as compact JSON, for the Hydra override's quoted value."""
    return json.dumps(hf_overrides(max_position_embeddings), separators=(",", ":"))


def hydra_overrides(max_position_embeddings: int | None = None) -> list[str]:
    """The trainer overrides for QuRL: the plugin; the TIS + ACR loss on VeRL's untruncated token-level
    rho = pi_old / mu; and the INT8 W8A8 vLLM engine, one replica per GPU, with the refit."""
    return [
        f"actor_rollout_ref.model.external_lib={PLUGIN}",          # registers the qurl_acr loss
        f"actor_rollout_ref.actor.policy_loss.loss_mode={LOSS_MODE}",
        "algorithm.rollout_correction.rollout_is=token",           # VeRL computes token-level rho = pi_old / mu ...
        "algorithm.rollout_correction.rollout_is_threshold=1e30",  # ... and passes it untruncated
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",   # the INT8 refit needs whole rows
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{hf_overrides_json(max_position_embeddings)}'",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls={WORKER_EXTENSION}",
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Print the Hydra overrides that wire qurl into VeRL, one per line.")
    ap.add_argument("--max-position-embeddings", type=int, default=None, help="context length for vLLM")
    ap.add_argument("--hf-overrides", action="store_true", help="print only vLLM's hf_overrides JSON")
    a = ap.parse_args(argv)
    if a.hf_overrides:
        print(hf_overrides_json(a.max_position_embeddings))
    else:
        print("\n".join(hydra_overrides(a.max_position_embeddings)))


if __name__ == "__main__":
    main()
