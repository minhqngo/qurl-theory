# qurl: QuRL in PyTorch

This package implements the training algorithm of [QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953) (Li et al., ICLR 2026) in plain PyTorch, with a train + validation loop on GSM8K (`python -m qurl.train`). QuRL speeds up RL by generating rollouts with a quantized copy of the actor while the trainer keeps full-precision weights.

Here the INT8 rollout is simulated (fake quantization): the loop is faithful to the algorithm but brings none of INT8's speedup, and it runs on one device.

The quantizer, the TIS + ACR arithmetic and UAQ come unchanged from the code this project's experiments ran (`qurl_verl` and `make_uaq_checkpoint.py`); the tests show it bit for bit. The loop builds on the experiments' standalone loop (`experiments/s2/s2_train.py`).

## The algorithm and where it lives

| QuRL (paper) | Here |
|---|---|
| **Quantized rollout** (Sec. 4, 5): the old actor is quantized for rollout, θ̂_old = Q(θ_old), at every RL step. Weights have channel-wise scales and activations token-wise scales; the KV cache is not quantized. | `qurl.fakequant`: INT8 W8A8 fake quantization of every linear layer but `lm_head`, re-quantized at every rollout, with `qurl.quant`'s per-row absmax quantizer. |
| **Decoupled PPO** (Eq. 4): the behavior policy μ, which sampled the tokens, is the quantized old actor; the proximal policy π_old, which clips the ratio, is the full-precision old actor. | `qurl.sampling.generate` records μ while sampling.<br>`qurl.train.ppo_update` computes π_old and ρ = π_old / μ. |
| **TIS** (Eq. 5): weight each token by min(ρ, C). | `qurl.loss.tis_acr_terms`, as a token-mean over each minibatch. |
| **ACR** (Eq. 9): the upper clip bound becomes (1 + ε_high) / r, with r = min(1, C / ρ). This only raises it for tokens whose TIS weight was truncated. | The same loss, on PPO's dual-clipped objective. |
| **UAQ** (Sec. 4.3): a one-time invariant scaling before training. W_{Q,K,V} are divided by s, and the input-layernorm gain, which absorbs the activation scale, is multiplied by s; s = 1.5. | `--uaq-s 1.5`: `qurl.uaq.check` rescales the loaded model and checks it exactly.<br>`python -m qurl.uaq` writes a UAQ checkpoint. |

**Not implemented:**

- the paper's FP8 rollout (it evaluates both INT8 and FP8);
- real INT8 kernels, hence no speedup;
- training on several GPUs.

The project's earlier VeRL integration, a VeRL plugin and a vLLM INT8 refit that ran QuRL on vLLM's INT8 kernels, is not part of the package. It stays in `experiments/` and in this repository's history (up to commit `4bb8d85`).

This project's theory note, [`latex/p1_uaq_and_update_tracking.tex`](../latex/p1_uaq_and_update_tracking.tex), analyses what the quantized rollout and UAQ do to training.

## Installation

```bash
pip install -e ./qurl            # torch only: the quantizer and the loss
pip install -e "./qurl[train]"   # + transformers: the training loop and UAQ
pip install -e "./qurl[test]"    # + pytest and what the tests use
```

The package was tested with Python 3.12, torch 2.11 and transformers 5.9. Setting `PYTHONPATH=<repo>/qurl/src` instead of installing also works.

## Training: `python -m qurl.train`

`qurl.train` runs QuRL on GSM8K on one device, with Hugging Face transformers. Each RL step:

1. **Rollout:** the actor, fake-quantized to INT8 W8A8 by `qurl.fakequant` (re-quantized from the current weights), samples `--group` responses to each of `--prompts` training problems at temperature 1, recording each token's log-probability μ.
2. **Advantages:** the reward is 1 when the final answer (the last `\boxed{}`, else the last number) is right, else 0. The GRPO advantage is the reward minus its group's mean, divided by the group's standard deviation.
3. **Proximal policy:** the full-precision actor's log-probabilities π_old of the same tokens, computed on the update's micro-batches, give ρ = π_old / μ.
4. **Update:** `--ppo-epochs` passes of `--minibatches` AdamW steps, each on the token-mean of the TIS + ACR loss, with global-norm gradient clipping.

The training forward runs in BF16 autocast over FP32 master weights. Validation runs before training, every `--val-every` steps and at the end: greedy decoding of `--val-size` GSM8K test problems by the full-precision actor (`--val-policy fp`, the default), its INT8 copy (`int8`), or both.

```bash
python -m qurl.train --out runs/qurl --steps 100                         # QuRL without UAQ
python -m qurl.train --out runs/qurl_uaq --steps 100 --uaq-s 1.5         # QuRL with UAQ
python -m qurl.train --out runs/rl --steps 100 --no-quantize-rollout     # full-precision rollout (baseline)
```

The defaults:

- **Model:** Qwen2.5-0.5B-Instruct. The paper trains this model on GSM8K, but with PPO and a critic; this loop uses GRPO, which needs no critic.
- **Batch:** 16 problems × 8 responses per step, responses up to 512 tokens.
- **Optimizer:** lr 1e-6, ε 0.2, C = 2, two AdamW steps per RL step. The paper does not state its C; 2 is the value this project's experiments used.

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

## UAQ checkpoints

`python -m qurl.uaq --model <hub id or directory> --s 1.5 --out <dir>` writes, for training with another framework:

- the rescaled model, in FP32, so that W / s is rounded once (BF16 storage would round it again);
- its tokenizer;
- `uaq.json`, recording s, the source model, the number of rescaled tensors and the checks.

The q/k/v biases stay as they are, which keeps the model's function unchanged. The rescaling is checked in FP32 on a fixed prompt; `--uaq-s` in `qurl.train` runs the same checks:

- **Exact probe:** rescaled by 2, where every product and quotient is exact, the logits must be bit-identical, and undoing the probe must restore every weight. This catches a rescaling of the wrong tensors.
- **Comparison at s:** at a power of two the logits must be bit-identical; otherwise they may differ only by FP32 rounding.

## Tests

Run the tests from `qurl/`:

```bash
python -m pytest
```

Everything runs on the CPU, except one test that needs CUDA (marker `gpu`). Tests marked `originals` compare with the experiment code, which they load read-only from this repository's `experiments/` directory; the few VeRL names that code imports are replaced by stand-ins, so neither VeRL nor vLLM is needed. Tests whose requirements are missing are skipped.

| File | What it shows |
|---|---|
| `test_quant.py` | The quantizer equals the original and an independent numpy quantizer, including on BF16 weights with exact rounding ties.<br>It rounds half to even, and a power-of-two absmax gives an exact power-of-two scale.<br>It is exactly homogeneous under powers of two, and at s = 1.5 only near-ties move.<br>CPU and CUDA give identical codes and scales (GPU). |
| `test_loss.py` | `tis_acr_terms` matches an independent float64 implementation of Eq. 9.<br>Without truncation, it is PPO's dual-clipped loss times ρ, bit for bit.<br>Its token-mean equals the original loss, and so do the gradients, bit for bit. |
| `test_uaq.py` | On a tiny random Qwen2, rescaling by 2 preserves the logits bit for bit.<br>`check` equals the original's record.<br>Four wrong rescalings fail the probe and leave the weights untouched.<br>The command line writes an FP32 checkpoint rescaled once, byte-identical to the original script's. |
| `test_fakequant.py` | A fake-quantized linear layer equals an independent float64 INT8 W8A8 computation.<br>Every linear layer but `lm_head` is patched, and restored afterwards; re-entering re-quantizes the current weights.<br>UAQ at s = 2 leaves the INT8 rollout's logits bit-identical. |
| `test_sampling.py` | The μ recorded while sampling with the KV cache equals a full forward pass, with and without INT8 and at another temperature.<br>Seeded sampling is reproducible; greedy decoding takes the argmax.<br>Stop tokens end responses; left padding does not change a greedy response. |
| `test_gsm8k.py` | Answer extraction (boxed, last number, commas, dollar signs), the reward, and parsing the official files (offline). |
| `test_train.py` | On a tiny random Qwen2 with a toy task (CPU, offline):<br>- the loop trains, validates at the configured steps with both policies, logs every metric and saves the model;<br>- with a zero learning rate every ratio π / π_old is exactly 1, because π_old is computed on the update's own micro-batches;<br>- runs are reproducible, and gradient checkpointing does not change them;<br>- UAQ at s = 2 leaves the first INT8 rollout and π_old unchanged;<br>- the full-precision rollout has no behaviour-proximal mismatch.<br>Also: GRPO advantages, configuration checks and the command line. |

## Differences from the experiment code

- **Left out:**
  - The VeRL integration: the plugin that registered the loss in VeRL, the vLLM worker extension that re-quantized vLLM's INT8 layers at every weight sync, and the Hydra overrides. It was in the package up to commit `4bb8d85`.
  - The experiments' own instruments, which are not part of QuRL: the optimizer of arm R5 (Theorem 2 of the theory note), the per-sync log of code flips, ties and quantization error, the test dump and the experiment arms. The package's first version, commit `7d433a9`, had them.

  All of it stays in `experiments/`.
- **The loss** keeps the experiments' per-token arithmetic; the loop aggregates it as a token-mean, which is what the experiments ran.
- **Errors:** `qurl.uaq.check` raises `UAQCheckError` instead of `SystemExit`; the command line exits with the same message.
- **The loop is new.** It follows the experiments' standalone loop (`experiments/s2/s2_train.py`): fake-quantized INT8 rollout, float64 inverse-CDF sampling, GRPO, and decoupled PPO with TIS and ACR. It is built from the package's own pieces, and differs from that loop in four ways:
  - **Quantizer:** `qurl.quant`'s (scale = absmax · fl32(1/127)), for weights and activations alike.
  - **Loss:** `qurl.loss.tis_acr_terms`, PPO's loss with the dual clip, as the experiments ran it.
  - **Advantages:** GRPO divides by the group's standard deviation.
  - **Validation:** added.

  The experiment's arms and fingerprints are not part of it.
