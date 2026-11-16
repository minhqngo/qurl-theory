# qurl: QuRL for VeRL

This package is our QuRL implementation for VeRL v0.9.1 and vLLM 0.24 ([QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953), Li et al., ICLR 2026). It contains:

- the INT8 W8A8 rollout, re-quantized from the trainer's weights at every weight sync, with its per-sync log of code flips and ties;
- QuRL's policy loss: token-level TIS with an adaptive clipping range (ACR);
- UAQ checkpoints;
- the optimizer of arm R5 (Theorem 2 without UAQ).

It refactors the experiment code (`qurl_verl` and `make_uaq_checkpoint.py`) without redesigning it. The arithmetic is unchanged, and the tests show it bit for bit where it matters.

The maths is in the theory note, [`latex/p1_uaq_and_update_tracking.tex`](../latex/p1_uaq_and_update_tracking.tex):

- **Theorem 1** (§3.1): UAQ leaves the INT8 rollout unchanged, because per-row absmax quantization is homogeneous.
- **Theorem 2** (§3.2): under AdamW, UAQ equals a per-layer change of learning rate, Adam ε and weight decay, plus clipping rule (c).
- **Corollary 3**: the whole pipeline, rollouts and TIS/ACR included, is therefore identical.
- **Proposition 4** (§3.3): re-quantization captures an update; a code flips with probability |d|/Δ.
- **Remark "BF16 storage"** (§3.3): where the ties come from.

## Contents

| Module | What it implements | Theory |
|---|---|---|
| `qurl.quant` | `quantize_rows`: per-row absmax INT8. The scale is absmax · fl32(1/127), written as a multiplication so that CPU and CUDA agree; codes are round-half-even of W / scale, clamped to ±127.<br>`tie_indices`: the weights within 1e-4 grid steps of a rounding boundary. | Theorem 1; Remark "BF16 storage" |
| `qurl.refit` | `Int8Refit`: writes the re-quantized codes and scales into vLLM's live INT8 layers at every sync. It refuses tensor parallelism, an incomplete sync, and a weight without an INT8 layer.<br>`RefitConfig`, `quantization_config()`. | |
| `qurl.synclog` | `SyncLog`: one JSON line per sync, against the replica's own copy of all previous codes: code flips, row-scale changes, quantization error, ties and their flips, and Proposition 4's predicted flip share. | Proposition 4 |
| `qurl.loss` | `tis_acr_terms`: per-token TIS weight min(ρ, C) and ACR's upper clip bound (1 + ε_high) · max(1, ρ / C), on VeRL's vanilla PPO loss with dual clip. | QuRL Eq. 9; Corollary 3 |
| `qurl.r5` | `param_groups`, `clip_factors`, `total_norm`, `clip_and_step`, `to_original_coordinates`: Theorem 2(b)'s three AdamW groups and clipping rule (c), with the clip norm reduced over any number of FSDP2 ranks. | Theorem 2 |
| `qurl.uaq` | `rescale`, `check` (with the exact s = 2 probe), `make_checkpoint`; command line `python -m qurl.uaq`. | Theorems 1 and 2 |
| `qurl.overrides` | The Hydra overrides below; command line `python -m qurl.overrides`. | |
| `qurl.verl` | The VeRL plugin (`external_lib`). It registers `loss_mode=qurl_acr` and, when `QURL_OPT` is set, patches the FSDP2 actor's optimizer (`qurl.verl.optim`). | |
| `qurl.verl.rollout` | `QuRLWorkerExtension`: VeRL's colocated vLLM worker extension, with `Int8Refit` on the weight-sync path. | |

`qurl.quant`, `qurl.refit`, `qurl.synclog`, `qurl.loss`, `qurl.r5` and `qurl.uaq` need only torch, plus transformers for the UAQ command line. Only `qurl.verl` imports VeRL and vLLM.

## Installation

Install the package into the environment VeRL runs in (the `py_executable` of Ray's runtime environment). Ray's training workers and vLLM's worker processes then import it.

```bash
pip install -e ./qurl            # torch only
pip install -e "./qurl[uaq]"     # + transformers, for UAQ checkpoints
pip install -e "./qurl[test]"    # + pytest and what the tests use
```

`qurl.verl` needs VeRL 0.9.1 and vLLM 0.24: it subclasses and patches their internals, and relies on vLLM 0.24's INT8 layer layout. The package was tested with Python 3.12, torch 2.11, transformers 5.9, VeRL 0.9.1 and vLLM 0.24. Setting `PYTHONPATH=<repo>/qurl/src` instead of installing also works.

## Wiring into VeRL

### Every arm

These overrides of `python -m verl.trainer.main_ppo` apply to every arm:

```text
actor_rollout_ref.model.external_lib=qurl.verl            # the loss; R5's optimizer when QURL_OPT is set
actor_rollout_ref.actor.strategy=fsdp2                    # per-parameter optimizer groups need FSDP2
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1    # one vLLM replica per GPU: the refit needs whole rows
```

### INT8 rollout with TIS and ACR (arms R3 to R6)

```text
actor_rollout_ref.actor.policy_loss.loss_mode=qurl_acr
algorithm.rollout_correction.rollout_is=token             # VeRL computes the token-level rho = pi_old / mu ...
algorithm.rollout_correction.rollout_is_threshold=1e30    # ... and passes it untruncated; the loss truncates at C
+actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{"quantization_config":{...},"max_position_embeddings":32768}'
+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl.verl.rollout.QuRLWorkerExtension
```

`hf_overrides` carries `src/qurl/int8_w8a8_config.json`: channel-wise symmetric INT8 weights, dynamic per-token symmetric INT8 activations, with `lm_head` kept in BF16. It replaces VeRL's own `hf_overrides`, so it also carries the context length. `python -m qurl.overrides --max-position-embeddings 32768` prints all of these overrides with the JSON filled in. Add `--bf16` for the BF16 arms, or `--hf-overrides` to print only the JSON.

### Environment variables

The workers read these variables, so set them before launching.

| Variable | Read by | Meaning |
|---|---|---|
| `QURL_OPT` | trainer | `r5`: Theorem 2(b)–(c) settings for s = `QURL_S`.<br>`plain`: the same code path with s = 1.<br>Unset: VeRL's own optimizer and clip. |
| `QURL_S` | trainer | s for `QURL_OPT=r5` (default 1.5). |
| `QURL_CLIP` | trainer | `ordinary`: R5's groups with the ordinary clip norm (the control arm R5o). |
| `QURL_TIS_C` | trainer | The TIS cap C (default 2.0). |
| `QURL_E3_PATH` | vLLM workers | Path of the per-sync log (JSON lines). One replica writes it: the first to create `<path>.<attempt>.claim`. |
| `QURL_RUN_ATTEMPT` | vLLM workers | An id per launch, so that a resumed run claims the log afresh. |
| `QURL_START_STEP` | vLLM workers | The step the attempt starts from: 0, or the checkpoint step a resumed attempt loads. Ties are only detected at step 0. |
| `QURL_INT8_DUMP_DIR` | vLLM workers | Test dump of layer 0's fused QKV codes at the first two syncs of the run's first attempt. |

### Arms

Every arm trains through the plugin's optimizer path (`QURL_OPT` set), so all arms run identical clipping arithmetic.

| Arm | Model | Rollout and loss | `QURL_OPT` | Learning rate |
|---|---|---|---|---|
| R1 | base | BF16, PPO | `plain` | η |
| R2 | UAQ checkpoint | BF16, PPO | `plain` | η |
| R3 | base | INT8 + TIS + ACR | `plain` | η |
| R4 | UAQ checkpoint | INT8 + TIS + ACR | `plain` | η |
| R5 | base | INT8 + TIS + ACR | `r5` (`QURL_S=s`) | η |
| R6 | base | INT8 + TIS + ACR | `plain` | s · η |

### Example: arm R5, QuRL's DAPO setup on Qwen2.5-Math-7B

Only the QuRL-specific parts are shown:

```bash
export QURL_OPT=r5 QURL_S=1.5 QURL_TIS_C=2.0
export QURL_E3_PATH="$RES/e3_int8.jsonl" QURL_RUN_ATTEMPT=$(date +%s) QURL_START_STEP=0
mapfile -t QURL_ARGS < <(python -m qurl.overrides --max-position-embeddings 32768)
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  ... \
  "${QURL_ARGS[@]}"
```

For R4, write the UAQ checkpoint once, then point `actor_rollout_ref.model.path` at it with `QURL_OPT=plain`:

```bash
python -m qurl.uaq --model Qwen/Qwen2.5-Math-7B --s 1.5 --out ~/models/qwen2.5-math-7b-uaq1.5
```

Only the actor is rescaled: the critic and the KL reference keep their usual model paths.

## The INT8 refit and its log

vLLM starts the rollout model as a compressed-tensors INT8 W8A8 model with VeRL's dummy weights.

**What a sync does.** At every weight sync, `QuRLWorkerExtension._update_weights` passes each q/k/v/o/gate/up/down weight to `Int8Refit`. The refit quantizes the weight row by row and writes the codes and scales into that shard's rows of the fused vLLM layer (`qkv_proj`, `gate_up_proj`, `o_proj`, `down_proj`). In vLLM 0.24, both the CUTLASS and the Triton kernel hold the codes as a transposed [in, out] view. Per-row scales make the fused shards independent. Every other tensor (embeddings, norms, biases) goes down VeRL's ordinary load path. During the sync, the extension makes vLLM's `process_weights_after_loading` a no-op: rerunning it would transpose the INT8 weights a second time.

**Checks.** The refit raises in three cases:

- tensor parallelism above 1;
- a sync that does not re-quantize all 7 linear weights of every layer;
- a synced linear weight without an INT8 layer to receive it.

**The log.** With `QURL_E3_PATH` set, one replica appends a JSON line per sync:

| Field | Per layer and projection (lists over layers) |
|---|---|
| `flip` | Share of INT8 codes changed since the previous sync (`null` at the attempt's first sync). |
| `scale_changed` | Share of rows whose scale changed. |
| `qerr` | Relative quantization error ‖W − Q(W)‖ / ‖W‖. |
| `ties` | Share of codes that were ties at the first sync. This is only logged for attempts that start at step 0. The ties are a BF16 checkpoint's exact ties; in R4 they are the same weights, which UAQ's FP32 rescaling moves by under 5e-5 grid steps. |
| `flip_ties` | Share of those ties that flipped since the previous sync. |
| `pred` | For the q/k/v weights of the first, middle and last layer: Proposition 4's prediction mean(min(1, \|W_k − W_(k−1)\| / Δ)), and `observed_all_rows`, the flip share recomputed from the previous weights. `observed_all_rows` must equal `flip`, which checks the history. |

The record also carries `attempt`, `start_step`, `sync`, `time`, `layers` and `codes_per_layer`.

**Where the history lives.** The history is the replica's own copy of every previous code, in pinned CPU memory (6.5 GB for 7B). vLLM's live weights cannot serve as history, because VeRL's sleep mode discards them between rollouts. A row subset cannot serve either: strided rows alias with the attention heads, and random rows are noisy for small tensors.

## Tests

Run the tests from `qurl/`:

```bash
python -m pytest                    # everything; skips what the machine cannot run
python -m pytest -m "not slow"      # without the multi-process FSDP2 test
```

The test markers:

- `gpu`: needs CUDA.
- `verl`: needs VeRL and, for some tests, vLLM.
- `originals`: compares with the experiment code, which the tests load read-only from this repository's `experiments/` directory.
- `slow`: spawns 2 and 3 processes.

Tests whose requirements are missing are skipped. Everything else runs on the CPU.

| File | What it shows |
|---|---|
| `test_quant.py` | The quantizer equals the original and an independent numpy quantizer.<br>Codes round half to even; a power-of-two absmax gives an exact power-of-two scale.<br>The quantizer is exactly homogeneous under powers of two; at s = 1.5 only near-ties move.<br>CPU and CUDA give identical codes and scales (GPU). |
| `test_ties.py` | Planted exact ties of a BF16 tensor (both kinds a BF16 checkpoint has) are found.<br>`tie_indices` equals the original and an independent recomputation.<br>UAQ's FP32 rescaling at s = 1.5 and 2 keeps the same tie set, for single tensors and through `qurl.uaq.rescale` on a tiny Qwen2.<br>A tiny update flips about half of the ties. |
| `test_uaq.py` | On a tiny random Qwen2 with random gains and biases, rescaling by 2 preserves the logits bit for bit.<br>`check` equals the original's record.<br>Four wrong rescalings fail the probe and leave the weights untouched.<br>The command line writes an FP32 checkpoint rescaled once, byte-identical to the original script's. |
| `test_loss.py` | `tis_acr_terms` matches a float64 reference of Eq. 9.<br>The VeRL loss is bit-identical to the original (loss, metrics and gradients) for every aggregation mode and C.<br>Without truncation, it is bit-identical to VeRL's vanilla PPO loss with TIS weights. |
| `test_r5.py` | Theorem 2(b)'s groups and clip factors.<br>`total_norm` equals a float64 reference.<br>R5 equals R4 bit for bit at s = 2 after three clipped AdamW steps; an R4 repeat is identical and R5o differs.<br>The functions and the patched VeRL engine methods are bit-identical to the original.<br>Under FSDP2 with 2 and 3 CPU ranks (gloo), the clip norm is identical on every rank, and R5 equals R4 bit for bit (slow). |
| `test_refit.py` | On stand-in vLLM INT8 layers (CPU and CUDA):<br>- the live codes equal an independent quantization, with the original's printed line;<br>- every log field (flips, scale changes, ties, flips of the ties, error, Proposition 4) equals an independent numpy recomputation;<br>- a resumed attempt logs no ties;<br>- one replica claims the log and the dump, and the dump equals an independent quantization;<br>- the three checks raise.<br>The refit equals the original: codes, scales, handed-back tensors and dump on the CPU, and identical log records on CUDA (GPU). |
| `test_overrides.py` | The packaged config is byte-identical to the experiments'.<br>The `hf_overrides` JSON is the run script's.<br>Hydra parses the overrides, and the named classes import. |
| `test_verl_integration.py` | Importing `qurl.verl` registers the loss; `QURL_OPT` switches the optimizer patch.<br>The worker extension refits, logs and hands the other tensors to VeRL.<br>The extension no-ops `process_weights_after_loading` during a sync and restores it. |

## Differences from the experiment code

- **Names.** The plugin is `qurl.verl` (was `qurl_verl`), and the worker extension is `qurl.verl.rollout.QuRLWorkerExtension` (was `qurl_verl.int8_rollout.QuRLWorkerExtension`). The loss name `qurl_acr`, the environment variables, the printed lines and the log format are unchanged.
- **Structure.** The arithmetic lives in framework-free modules, under a thin VeRL layer. The refit's state moved from `_qurl_*` attributes of the extension into `Int8Refit` and `SyncLog`.
- **Environment.** The refit reads its environment variables at its first sync instead of at import. The values are the same, because the variables are set before the workers start.
- **CPU support.** The per-sync log also runs on the CPU: it pins memory and synchronizes the CUDA stream only on CUDA. On CUDA it runs the same operations as before, and its records equal the original's.
- **Errors.** `qurl.uaq.check` raises `UAQCheckError` instead of `SystemExit`; the command line exits with the same message.
- **New helpers.** `qurl.overrides`, `qurl.r5.to_original_coordinates` and `qurl.uaq.make_checkpoint` are new.
