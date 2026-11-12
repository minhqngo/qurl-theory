# S2: exact test of Theorem 2(b)–(c) (P1 experiments report §2.3 and §3.3)

UAQ at s = 2 (Q/K/V weights ÷ 2, input-norm gain × 2) trained with plain AdamW should be **bit-identical**, step by step, to training without UAQ at the per-layer settings of Theorem 2(b)–(c):
- learning rate 2η on W_QKV and η/2 on γ;
- Adam ε/2 and 2ε;
- weight decay λ/2 and 2λ;
- global-norm clipping computed in UAQ coordinates.

At s = 2 every rescaling is exact in floating point, so the prediction is exact equality, not closeness.

## Design
- **Loop.** One QuRL-style loop on GSM8K with Qwen2.5-0.5B-Instruct (S3's model). It uses GRPO advantages and decoupled PPO with TIS weights and the ACR bound, makes 2 epochs × 2 minibatches of updates per rollout batch, and uses AdamW with global-norm clipping.
- **Rollouts.** INT8 W8A8 fake quantization in PyTorch (fp32): per-channel absmax weights and dynamic per-token absmax activations, with the lm_head unquantized. This is the quantizer class of Theorem 1, the same scheme as S1's llm-compressor checkpoints. Sampling is by inverse CDF in float64, from per-step seeded uniforms.
- **Why not vLLM.** vLLM has no online INT8 W8A8 re-quantization after weight updates, and S1 showed its default engine is not repeatable at small model sizes. vLLM's Theorem 1 behaviour was already tested in S1.
- **Determinism.** The training forward pass runs in BF16 autocast with FP32 master weights. `torch.use_deterministic_algorithms(True)` is on, attention is eager, and TF32 is off.
- **Fingerprints.** Each step logs SHA-256 fingerprints of the sampled tokens, the rollout and proximal log-probabilities, and the parameters mapped back to original coordinates (per group). The final parameters are saved as snapshots.

| Arm | Setting | Prediction vs. R4 |
|---|---|---|
| R4 | UAQ, plain AdamW and clipping | reference |
| R4_repeat | identical to R4 | bit-identical (pipeline is deterministic) |
| R5 | no UAQ, Theorem 2(b)–(c) | **bit-identical at every step** |
| R5o | as R5, ordinary clipping | drifts once clipping binds (rule (c) needed) |
| R5x | no UAQ, learning rates matched only | drifts (ε, weight decay, clipping unmatched) |
| R3 | no UAQ, plain settings | differs |

**Two learning rates.** Decoupled weight decay multiplies the weights by 1 − ηλ. In FP32 this is exactly 1 whenever ηλ < 2⁻²⁵ ≈ 3·10⁻⁸, so at η = 1e-6 and λ = 0.01 weight decay does nothing at all. The lr = 1e-6 run is the regime of interest. The lr = 1e-5 run makes weight decay active and lets ratios move enough for PPO and ACR to bind.

## Run (on the H100, from the project root)
```bash
ssh h100 "mkdir -p qurl/s2"
scp experiments/s2/s2_train.py experiments/s2/s2_compare.py experiments/s2/s2_run.sh h100:qurl/s2/
ssh h100 "nohup bash -c 'bash qurl/s2/s2_run.sh lr1e-6 --lr 1e-6; bash qurl/s2/s2_run.sh lr1e-5 --lr 1e-5' > qurl/s2/s2.log 2>&1 &"
```
It needs no new installs: it uses `~/venvs/qurl` (torch and transformers come with vLLM). Each configuration runs 6 arms × 20 rollout steps. Re-running skips finished arms. Results go to `qurl/s2/results/<config>/`, and `compare.txt` holds the verdict.

Copy the results back without the parameter snapshots. Each `.pt` file is 2 GB; they stay on the server, where `compare.txt` has already used them:
```bash
ssh h100 "cd qurl/s2 && tar czf s2_results.tgz --exclude='*.pt' results"
scp h100:qurl/s2/s2_results.tgz experiments/s2/
tar xzf experiments/s2/s2_results.tgz -C experiments/s2
```

## Results (H100 NVL, 2026-11-09)
Raw output is in `results/lr1e-6/` and `results/lr1e-5/`: per-arm `log.jsonl` and `.log`, plus `compare.txt`. The write-up is in the P1 experiments report, §3.3.
- **R5 matches R4.** R5 and R4_repeat were bit-identical to R4 at all 20 steps under both learning rates. That covers the tokens, μ, π_old and all three parameter groups. Their saved weights after steps 1 and 20 equal R4's mapped weights in every element.
- **Controls diverge.** R5o, R5x and R3 diverged in the weights at step 1 and in the tokens at step 2.
- **What was exercised.**
  - Clipping bound on every update with a nonzero gradient: 70/70 at lr 1e-6 and 44/44 at lr 1e-5.
  - TIS truncated 0.2–1.3% of tokens per batch.
  - ACR's raised bound kept 26 and 7 token terms.
  - Weight decay was active only at lr 1e-5.
- **lr 1e-5 is a stress test.** The reward fell from 0.31 to about 0.06 in every arm, so this run is a numerical stress test, not a training result.
- **Timing.** Each arm took about 7.7 minutes; both configurations took 1.5 hours.

## Smoke test (laptop, not reported)
Tiny random Qwen2 model with the real tokenizer and a toy reward, 4 steps, CUDA in deterministic mode:
- R4_repeat and R5 were bit-identical to R4 at every step: tokens, rollout and proximal log-probabilities, and all parameter groups.
- R5x and R3 diverged at step 1.

Qwen2.5-0.5B-Instruct ran in 12.8 GB with R4 ≡ R5 ≡ R5o. Rewards were 0 there, so no gradients flowed and that run only checked the plumbing.
