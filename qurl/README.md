# qurl: QuRL for VeRL

This package implements the training algorithm of [QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953) (Li et al., ICLR 2026) for VeRL v0.9.1 and vLLM 0.24. QuRL speeds up RL by generating rollouts with a quantized copy of the actor while the trainer keeps full-precision weights.

The package was refactored from the code this project's experiments ran (`qurl_verl` and `make_uaq_checkpoint.py`) without redesigning it. The arithmetic is unchanged, and the tests show it bit for bit where it matters.

## The algorithm and where it lives

| QuRL (paper) | Here |
|---|---|
| **Quantized rollout** (Sec. 4, 5): the old actor is quantized for rollout, θ̂_old = Q(θ_old), at every RL step. Weights have channel-wise scales and activations token-wise scales; rollout uses vLLM's 8-bit kernels; the KV cache is not quantized. | `qurl.quant`: per-row absmax INT8 weights.<br>`qurl.refit`: re-quantizes vLLM's INT8 W8A8 layers from the synced weights at every weight sync.<br>`qurl.verl.rollout`: the vLLM worker extension that runs the refit. |
| **Decoupled PPO** (Eq. 4): the behavior policy μ, which sampled the tokens, is the quantized old actor; the proximal policy π_old, which clips the ratio, is the full-precision old actor. | VeRL computes the token-level ρ = π_old / μ (`algorithm.rollout_correction.rollout_is=token`) and passes it untruncated to the loss. |
| **TIS** (Eq. 5): weight each token by min(ρ, C). | `qurl.loss.tis_acr_terms`; registered in VeRL as `loss_mode=qurl_acr` (`qurl.verl.loss`). |
| **ACR** (Eq. 9): the upper clip bound becomes (1 + ε_high) / r, with r = min(1, C / ρ). This only raises it for tokens whose TIS weight was truncated. | Same loss. Everything else is VeRL's vanilla PPO loss, including the decoupled clip range and the dual clip for negative advantages. |
| **UAQ** (Sec. 4.3): a one-time invariant scaling before training. W_{Q,K,V} are divided by s, and the input-layernorm gain, which absorbs the activation scale, is multiplied by s; s = 1.5. | `qurl.uaq`: writes the rescaled checkpoint in FP32 and checks it exactly; command line `python -m qurl.uaq`. |

The helper `qurl.overrides` prints the Hydra overrides below.

**Not implemented:** the paper's FP8 rollout. The paper evaluates both INT8 and FP8; VeRL 0.9.1's own FP8 rollout uses block-wise scales, not the paper's channel-wise and token-wise ones.

This project's theory note, [`latex/p1_uaq_and_update_tracking.tex`](../latex/p1_uaq_and_update_tracking.tex), analyses what the quantized rollout and UAQ do to training.

## Installation

Install the package into the environment VeRL runs in (the `py_executable` of Ray's runtime environment). Ray's training workers and vLLM's worker processes then import it.

```bash
pip install -e ./qurl            # torch only
pip install -e "./qurl[uaq]"     # + transformers, for UAQ checkpoints
pip install -e "./qurl[test]"    # + pytest and what the tests use
```

`qurl.verl` needs VeRL 0.9.1 and vLLM 0.24: it subclasses VeRL's vLLM worker extension and relies on vLLM 0.24's INT8 layer layout. The package was tested with Python 3.12, torch 2.11, transformers 5.9, VeRL 0.9.1 and vLLM 0.24. Setting `PYTHONPATH=<repo>/qurl/src` instead of installing also works.

## Running QuRL in VeRL

Add these overrides to `python -m verl.trainer.main_ppo`:

```text
actor_rollout_ref.model.external_lib=qurl.verl            # registers loss_mode=qurl_acr
actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
algorithm.rollout_correction.rollout_is=token             # VeRL computes the token-level rho = pi_old / mu ...
algorithm.rollout_correction.rollout_is_threshold=1e30    # ... and passes it untruncated; the loss truncates at C
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1    # one vLLM replica per GPU: the refit needs whole rows
+actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{"quantization_config":{...},"max_position_embeddings":32768}'
+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl.verl.rollout.QuRLWorkerExtension
```

`hf_overrides` carries [`src/qurl/int8_w8a8_config.json`](src/qurl/int8_w8a8_config.json): channel-wise symmetric INT8 weights, dynamic per-token symmetric INT8 activations, `lm_head` kept in BF16, no KV-cache scheme. It replaces VeRL's own `hf_overrides`, so it also carries the context length.

`python -m qurl.overrides --max-position-embeddings 32768` prints these overrides with the JSON filled in. Add `--hf-overrides` to print only the JSON.

The TIS cap C comes from the environment variable `QURL_TIS_C`, read in the training workers. It defaults to 2.0, the value this project's experiments used; the paper does not state its C.

### The paper's configurations

| Configuration | Model path | Loss |
|---|---|---|
| QuRL without UAQ | the base model | `qurl_acr` (the overrides above) |
| QuRL with UAQ | the UAQ checkpoint | `qurl_acr` |
| TIS only (the FlashRL baseline) | the base model | VeRL's vanilla loss with `rollout_is=token` and `rollout_is_threshold=C`, on the same INT8 rollout |

Write the UAQ checkpoint once, before training; only the actor uses it, while the critic and the KL reference keep their usual model paths:

```bash
python -m qurl.uaq --model Qwen/Qwen2.5-Math-7B --s 1.5 --out ~/models/qwen2.5-math-7b-uaq1.5
```

### Example: the paper's DAPO setup

The paper's DAPO setup trains Qwen2.5-Math-7B for 200 steps: 512 prompts × 16 responses per step, lr 1e-6, ε_low 0.2, ε_high 0.28, no KL loss. Only the QuRL-specific parts are shown:

```bash
export QURL_TIS_C=2.0
mapfile -t QURL_ARGS < <(python -m qurl.overrides --max-position-embeddings 32768)
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$HOME/models/qwen2.5-math-7b-uaq1.5 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  ... \
  "${QURL_ARGS[@]}"
```

## The INT8 refit

vLLM starts the rollout model as a compressed-tensors INT8 W8A8 model with VeRL's dummy weights. At every weight sync, `QuRLWorkerExtension._update_weights` passes each q/k/v/o/gate/up/down weight to `Int8Refit`. The refit quantizes the weight row by row and writes the codes and scales into that shard's rows of the fused vLLM layer (`qkv_proj`, `gate_up_proj`, `o_proj`, `down_proj`).

In vLLM 0.24, both the CUTLASS and the Triton kernel hold the codes as a transposed [in, out] view. Per-row scales make the fused shards independent, so each tensor is handled as it arrives. Every other tensor (embeddings, norms, biases) goes down VeRL's ordinary load path.

During the sync, the extension makes vLLM's `process_weights_after_loading` a no-op: rerunning it would transpose the INT8 weights a second time. Each replica prints one line per sync. A real one, from a 28-layer model:

```text
[qurl int8 refit] pid 3776088 sync 1: 196 linear weights re-quantized to INT8 (synced weights arrive as torch.float32; INT8 kernel CutlassInt8ScaledMMLinearKernel)
```

The refit raises on:

- tensor parallelism above 1 (a row-parallel layer would hold a slice of every row, and a per-row scale needs the whole row);
- a sync that does not re-quantize all 7 linear weights of every layer;
- a synced linear weight without an INT8 layer to receive it.

It expects Qwen2/Llama-style module names.

## UAQ checkpoints

`python -m qurl.uaq --model <hub id or directory> --s 1.5 --out <dir>` writes:

- the rescaled model, in FP32, so that W / s is rounded once (BF16 storage would round it again);
- its tokenizer;
- `uaq.json`, recording s, the source model, the number of rescaled tensors and the checks.

The q/k/v biases stay as they are, which keeps the model's function unchanged. The command line checks the rescaling in FP32 on a fixed prompt:

- **Exact probe:** rescaled by 2, where every product and quotient is exact, the logits must be bit-identical, and undoing the probe must restore every weight. This catches a rescaling of the wrong tensors.
- **Comparison at s:** at a power of two the logits must be bit-identical; otherwise they may differ only by FP32 rounding.

## Tests

Run the tests from `qurl/`:

```bash
python -m pytest
```

Tests whose requirements are missing are skipped. Everything else runs on the CPU. The markers:

- `gpu`: needs CUDA.
- `verl`: needs VeRL and, for some tests, vLLM.
- `originals`: compares with the experiment code, which the tests load read-only from this repository's `experiments/` directory.

| File | What it shows |
|---|---|
| `test_quant.py` | The quantizer equals the original and an independent numpy quantizer.<br>It rounds half to even, and a power-of-two absmax gives an exact power-of-two scale.<br>It is exactly homogeneous under powers of two, and at s = 1.5 only near-ties move.<br>CPU and CUDA give identical codes and scales (GPU). |
| `test_refit.py` | On stand-in vLLM INT8 layers (CPU and CUDA), at every sync the live codes and scales equal an independent quantization of the synced weights. These are BF16 weights with exact rounding ties, then FP32 updates. The other tensors are handed back.<br>The refit equals the original worker extension: codes, scales, handed-back tensors and printed lines.<br>Its checks raise. |
| `test_loss.py` | `tis_acr_terms` matches an independent float64 implementation of Eq. 9.<br>The VeRL loss is bit-identical to the original (loss, metrics and gradients) for every aggregation mode and C.<br>Without truncation, it is bit-identical to VeRL's vanilla PPO loss with TIS weights. |
| `test_uaq.py` | On a tiny random Qwen2, rescaling by 2 preserves the logits bit for bit.<br>`check` equals the original's record.<br>Four wrong rescalings fail the probe and leave the weights untouched.<br>The command line writes an FP32 checkpoint rescaled once, byte-identical to the original script's. |
| `test_overrides.py` | The packaged config is byte-identical to the experiments' and quantizes as the paper does (channel-wise weights, dynamic per-token activations, no KV cache).<br>Hydra parses the overrides, and the named classes import. |
| `test_verl_integration.py` | Importing `qurl.verl` registers the loss and leaves VeRL's optimizer alone.<br>The worker extension refits and hands the other tensors to VeRL.<br>It no-ops `process_weights_after_loading` during a sync and restores it. |

## Differences from the experiment code

- **Left out:** the experiments' own instruments are not part of QuRL and are not in the package:
  - the optimizer of arm R5 (Theorem 2 of the theory note);
  - the refit's per-sync log of code flips, ties and quantization error;
  - its test dump;
  - the experiment arms.

  They stay in `experiments/`, and the package's first version (commit `7d433a9`) had them.
- **Names:** the plugin is `qurl.verl` (was `qurl_verl`), and the worker extension is `qurl.verl.rollout.QuRLWorkerExtension` (was `qurl_verl.int8_rollout.QuRLWorkerExtension`). The loss name `qurl_acr`, `QURL_TIS_C` and the refit's printed line are unchanged.
- **Structure:** the arithmetic lives in framework-free modules, under a thin VeRL layer, and the refit's state is an `Int8Refit` object.
- **Errors:** `qurl.uaq.check` raises `UAQCheckError` instead of `SystemExit`; the command line exits with the same message.
