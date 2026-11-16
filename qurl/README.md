# qurl: QuRL in PyTorch and VeRL

This package implements the training algorithm of [QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953) (Li et al., ICLR 2026). QuRL speeds up RL by generating rollouts with a quantized copy of the actor while the trainer keeps full-precision weights. There are two ways to run it:

- **`python -m qurl.train`:** a self-contained train + validation loop on GSM8K in plain PyTorch, without VeRL or vLLM. Its INT8 rollout is simulated (fake quantization), so it is faithful to the algorithm but brings none of INT8's speedup.
- **VeRL v0.9.1 with vLLM 0.24:** a plugin and a vLLM worker extension that run the rollout on vLLM's real INT8 kernels, at the paper's scale.

The VeRL path, the loss and UAQ were refactored from the code this project's experiments ran (`qurl_verl` and `make_uaq_checkpoint.py`) without redesigning it. The arithmetic is unchanged, and the tests show it bit for bit where it matters. The PyTorch loop builds on the experiments' standalone loop (`experiments/s2/s2_train.py`).

## The algorithm and where it lives

| QuRL (paper) | PyTorch loop | VeRL |
|---|---|---|
| **Quantized rollout** (Sec. 4, 5): the old actor is quantized for rollout, θ̂_old = Q(θ_old), at every RL step. Weights have channel-wise scales and activations token-wise scales; the KV cache is not quantized. | `qurl.fakequant`: INT8 W8A8 fake quantization of every linear layer but `lm_head`, re-quantized at every rollout, with `qurl.quant`'s per-row absmax quantizer. | `qurl.refit`: re-quantizes vLLM's INT8 W8A8 layers from the synced weights at every weight sync.<br>`qurl.verl.rollout`: the vLLM worker extension that runs the refit. |
| **Decoupled PPO** (Eq. 4): the behavior policy μ, which sampled the tokens, is the quantized old actor; the proximal policy π_old, which clips the ratio, is the full-precision old actor. | `qurl.sampling.generate` records μ while sampling; `qurl.train.ppo_update` computes π_old and ρ = π_old / μ. | VeRL computes the token-level ρ (`algorithm.rollout_correction.rollout_is=token`) and passes it untruncated to the loss. |
| **TIS** (Eq. 5): weight each token by min(ρ, C). | `qurl.loss.tis_acr_terms`, token-mean over each minibatch. | The same function, registered as `loss_mode=qurl_acr` (`qurl.verl.loss`). |
| **ACR** (Eq. 9): the upper clip bound becomes (1 + ε_high) / r, with r = min(1, C / ρ). This only raises it for tokens whose TIS weight was truncated. | Same loss. | Same loss. Everything else is VeRL's vanilla PPO loss, including the decoupled clip range and the dual clip for negative advantages. |
| **UAQ** (Sec. 4.3): a one-time invariant scaling before training. W_{Q,K,V} are divided by s, and the input-layernorm gain, which absorbs the activation scale, is multiplied by s; s = 1.5. | `--uaq-s 1.5`: `qurl.uaq.check` rescales the loaded model and checks it exactly. | `python -m qurl.uaq` writes the rescaled checkpoint in FP32. |

The helper `qurl.overrides` prints the Hydra overrides for the VeRL path.

**Not implemented:** the paper's FP8 rollout. The paper evaluates both INT8 and FP8; VeRL 0.9.1's own FP8 rollout uses block-wise scales, not the paper's channel-wise and token-wise ones.

This project's theory note, [`latex/p1_uaq_and_update_tracking.tex`](../latex/p1_uaq_and_update_tracking.tex), analyses what the quantized rollout and UAQ do to training.

## Installation

For the VeRL path, install the package into the environment VeRL runs in (the `py_executable` of Ray's runtime environment); Ray's training workers and vLLM's worker processes then import it.

```bash
pip install -e ./qurl            # torch only
pip install -e "./qurl[train]"   # + transformers, for the PyTorch loop and UAQ checkpoints
pip install -e "./qurl[test]"    # + pytest and what the tests use
```

`qurl.verl` needs VeRL 0.9.1 and vLLM 0.24: it subclasses VeRL's vLLM worker extension and relies on vLLM 0.24's INT8 layer layout. The package was tested with Python 3.12, torch 2.11, transformers 5.9, VeRL 0.9.1 and vLLM 0.24. Setting `PYTHONPATH=<repo>/qurl/src` instead of installing also works.

## Training without VeRL: `python -m qurl.train`

`qurl.train` runs QuRL on GSM8K on one device, with Hugging Face transformers. Each RL step:

1. **Rollout:** the actor, fake-quantized to INT8 W8A8 by `qurl.fakequant` (re-quantized from the current weights), samples `--group` responses to each of `--prompts` training problems at temperature 1, recording each token's log-probability μ.
2. **Advantages:** the reward is 1 when the final answer (the last `\boxed{}`, else the last number) is right, else 0. The GRPO advantage is the reward minus its group's mean, divided by the group's standard deviation.
3. **Proximal policy:** the full-precision actor's log-probabilities π_old of the same tokens, computed on the update's micro-batches, give ρ = π_old / μ.
4. **Update:** `--ppo-epochs` passes of `--minibatches` AdamW steps, each on the token-mean of the TIS + ACR loss, with global-norm gradient clipping.

The training forward runs in BF16 autocast over FP32 master weights, as VeRL's does. Validation runs before training, every `--val-every` steps and at the end: greedy decoding of `--val-size` GSM8K test problems by the full-precision actor (`--val-policy fp`, the default), its INT8 copy (`int8`), or both.

```bash
python -m qurl.train --out runs/qurl --steps 100                         # QuRL without UAQ
python -m qurl.train --out runs/qurl_uaq --steps 100 --uaq-s 1.5         # QuRL with UAQ
python -m qurl.train --out runs/rl --steps 100 --no-quantize-rollout     # full-precision rollout (baseline)
```

The defaults:

- **Model:** Qwen2.5-0.5B-Instruct. The paper trains this model on GSM8K, but with PPO and a critic; this loop uses GRPO, which needs no critic.
- **Batch:** 16 problems × 8 responses per step, responses up to 512 tokens.
- **Optimizer:** lr 1e-6, ε 0.2, C = 2, two AdamW steps per RL step.

`python -m qurl.train --help` lists every option. GSM8K is downloaded on first use from [the official repository](https://github.com/openai/grade-school-math) to `~/.cache/qurl/gsm8k` (`--data-dir` to change).

The outputs, in `--out`:

- `config.json`;
- `metrics.jsonl`, with one record per step and one per validation:
  - per step: reward, response length, loss, gradient norm, clip fractions, TIS truncation and ACR shares, the largest ρ, and the paper's behaviour-to-proximal KL, E_μ[log μ − log π_old];
  - per validation: accuracy and mean response length for each policy;
- `final/`: the trained model in Hugging Face format, in UAQ coordinates if UAQ was applied (the same function).

**Cost.** On a laptop's 12 GB RTX PRO 3000, a 3-step smoke run took 347 s. Its settings: `--prompts 8 --group 4 --max-new 256 --micro-batch 4`, UAQ at 1.5, and validation on 32 problems with both policies.

- **Time:** about 29 s per rollout, 18–30 s per update, and 50 s per validation.
- **Memory:** the peak, 12.7 GiB, exceeded the card, so the driver spilled into system memory. The 0.5B model's FP32 weights, gradients and AdamW states alone take about 8 GB. `--grad-ckpt` and a smaller `--micro-batch` reduce the activation memory on top of that.

The default settings (16 × 8 responses of up to 512 tokens) are sized for a larger GPU.

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
| `test_fakequant.py` | A fake-quantized linear layer equals an independent float64 INT8 W8A8 computation.<br>Every linear layer but `lm_head` is patched, and restored afterwards; re-entering re-quantizes the current weights.<br>UAQ at s = 2 leaves the INT8 rollout's logits bit-identical. |
| `test_sampling.py` | The μ recorded while sampling with the KV cache equals a full forward pass, with and without INT8 and at another temperature.<br>Seeded sampling is reproducible; greedy decoding takes the argmax.<br>Stop tokens end responses; left padding does not change a greedy response. |
| `test_gsm8k.py` | Answer extraction (boxed, last number, commas, dollar signs), the reward, and parsing the official files (offline). |
| `test_train.py` | On a tiny random Qwen2 with a toy task (CPU, offline):<br>- the loop trains, validates at the configured steps with both policies, logs every metric and saves the model;<br>- with a zero learning rate every ratio π / π_old is exactly 1, because π_old is computed on the update's own micro-batches;<br>- runs are reproducible;<br>- UAQ at s = 2 leaves the first INT8 rollout and π_old unchanged;<br>- the full-precision rollout has no behaviour-proximal mismatch.<br>Also: GRPO advantages, configuration checks and the command line. |

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
- **The PyTorch loop is new.** It follows the experiments' standalone loop (`experiments/s2/s2_train.py`): fake-quantized INT8 rollout, float64 inverse-CDF sampling, GRPO, and decoupled PPO with TIS and ACR. It is built from the package's own pieces, and differs from that loop in four ways:
  - **Quantizer:** `qurl.quant`'s, as in the refit (scale = absmax · fl32(1/127)), for weights and activations alike.
  - **Loss:** `qurl.loss.tis_acr_terms`, VeRL's PPO loss with the dual clip.
  - **Advantages:** GRPO divides by the group's standard deviation, VeRL's default.
  - **Validation:** added.

  The experiment's arms and fingerprints are not part of it.
