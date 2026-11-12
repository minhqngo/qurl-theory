# S3: QuRL's GSM8K PPO setup in VeRL (P1 experiments report §2.4–2.5 and §3.4–3.5)

S3 tests H1–H3 (R4 ≡ R5 in distribution; UAQ's effect is not quantization-specific; R6 ≠ R4) on Qwen2.5-0.5B-Instruct, in the GSM8K PPO configuration of the QuRL paper (Sec. 5.1, Table 1).

## The configuration
The paper's GSM8K setup is VeRL's quickstart:
- PPO with a critic (GAE);
- batch 256;
- prompts and responses up to 512 tokens;
- 15 epochs of the 7,473 training problems, i.e. 435 steps;
- greedy validation on the 1,319 test problems;
- KL penalty 0.001 in the reward and entropy bonus 0.001.

The last two follow [VeRL's documented PPO baseline](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log) for this model and task: 56.7% GSM8K test accuracy at actor lr 1e-6 and critic lr 1e-5.
- Both were VeRL defaults when that baseline was run. v0.9.1 turns both off unless set, so `pilot.sh` sets them explicitly.
- The QuRL paper does not state either.
- R1 at 1e-6 is therefore VeRL's baseline configuration and should land near 56.7%.

**Learning rate (decided 2026-11-10).**
- The paper ran this setup at 1e-5 and switched UAQ off for it: "Update-Aware Quantization is disabled for this experiment due to the relatively high learning rate (10^-5), which already provides sufficient weight update magnitude."
- The paper's UAQ results (DAPO, DeepScaleR) are all at 1e-6, which is also VeRL's quickstart value.
- **The UAQ arms (R1–R6 × 3 seeds) run at 1e-6**, which is `pilot.sh`'s default.
- **R1 and R3 also run at 1e-5, one seed each to start, to reproduce the paper's Table 1:** 55.35% GSM8K test accuracy for BF16 and 53.55% for QuRL INT8. This checks our INT8 rollout, TIS and ACR before the UAQ arms rely on them.
- **Result (2026-11-10): not reproduced.** Both runs collapsed at 1e-5 (0.0% and 2.9%); see "Table 1 check" below.

## Environment: `setup_verl.sh`
- **What it installs.** VeRL v0.9.1 (Sep 2026, the latest release) with its own `uv.lock` stack in `~/venvs/verl`: vLLM 0.24.0, torch 2.11.0+cu130, transformers 5.9.0.
- **Separate from S1/S2.** `~/venvs/qurl` (vLLM 0.33, used by S1/S2) is not touched. S1's Theorem 1 result was measured on vLLM 0.33's CUTLASS INT8 kernel; S3's rollouts run on vLLM 0.24.
- **Requirements.** No root, nvcc or docker: every native package is a prebuilt wheel. uv needs `git` to build two pure-Python git dependencies.
- **flash-attn is not installed.** The v0.9.1 lockfile points at a wheel release that has been removed from VeRL's wheelhouse (HTTP 404; only torch 2.13 builds remain). Actor and critic therefore use PyTorch's SDPA attention, without sequence packing (`use_remove_padding=False`). vLLM has its own attention kernels and is unaffected.
- **One local patch to VeRL.** v0.9.1 loads the critic's value-head model with `attn_implementation="flash_attention_2"` hard-coded (`verl/utils/model.py`, `load_valuehead_model`), so the critic ignores its configured attention and fails without flash-attn. The laptop smoke test found this. `setup_verl.sh` patches the two lines to use the attention set in the critic's config, idempotently, and `env.txt` records the `git diff --stat`.
- **Data and model.** The script also writes VeRL's GSM8K parquet files to `~/data/gsm8k/` and fetches the model into the Hugging Face cache.

```bash
ssh h100 "mkdir -p qurl/s3"
scp experiments/s3/setup_verl.sh experiments/s3/pilot.sh experiments/s3/summarize_pilot.py h100:qurl/s3/
ssh h100 "nohup bash qurl/s3/setup_verl.sh > qurl/s3/setup.log 2>&1 < /dev/null &"
ssh h100 "tail -5 qurl/s3/setup.log"
```
Setup is finished when the log ends with `== setup done`. `qurl/s3/env.txt` records the versions.

## R1 timing pilot: `pilot.sh`
```bash
ssh h100 "nohup bash qurl/s3/pilot.sh r1_pilot 10 > qurl/s3/pilot_r1.log 2>&1 < /dev/null &"
ssh h100 "cat qurl/s3/results/r1_pilot/summary.txt"
```
The pilot runs 10 PPO steps and validates once at step 10.
- **Arms:** chosen by the `ARM` environment variable.
  - `R1` (default): BF16 rollout.
  - `R3`: QuRL, meaning the INT8 W8A8 rollout re-quantized at every weight sync, token-level TIS with cap C = `QURL_TIS_C` (default 2), and the ACR clip bound. The paper does not state C; 2 is the value VeRL's FP8 recipe uses.
  - R2, R4, R5 and R6 come later.
- **INT8 outputs:** the INT8 arm also writes `int8_refit.txt` (one line per sync: tensors re-quantized, the dtype the synced weights arrive in, vLLM's INT8 kernel) and `int8_dump/` (layer 0's fused QKV codes and scales after the first two syncs, for the offline exactness check).
- **Summary:** `summary.txt` also reports the rollout–trainer mismatch and, for R3, the TIS and ACR activity.

INT8 pilot (R3) on the H100:
```bash
scp experiments/s3/pilot.sh experiments/s3/summarize_pilot.py h100:qurl/s3/
scp -r experiments/s3/qurl_verl h100:qurl/s3/
ssh h100 "ARM=R3 nohup bash qurl/s3/pilot.sh r3_pilot 10 > qurl/s3/pilot_r3.log 2>&1 < /dev/null &"
```
- `metrics.jsonl` has one line per step with all of VeRL's metrics, including the per-stage `timing_s/*`.
- `summary.txt` gives the per-stage time and projects a 435-step run with one validation per epoch.
- Extra arguments after the step count are passed to Hydra as overrides, e.g. `actor_rollout_ref.actor.optim.lr=1e-5`.
- **Prompt budget:** `data.max_prompt_length=320`.
  - With the chat template, every GSM8K prompt fits in 256 tokens (train max 256, test max 232), so nothing is truncated.
  - VeRL's agent loop would left-truncate a longer prompt, with a warning, rather than fail.
  - The agent loop pads every prompt to this budget, but pilot 2 showed no measurable saving over the quickstart's 512. The two pilots also differ in micro-batch sizes and in the entropy term, so the effect is not isolated.
- **Detaching on the H100.** Every `nohup … &` command needs `< /dev/null`. Without it the background job keeps the SSH connection's stdin open, and `ssh` hangs until the job ends. The job survives Ctrl+C either way.
- **Micro-batches:** 8 for the actor update, 16 for the log-probability passes and the critic.
  - Without sequence packing, every sequence is padded to 832 tokens (320 + 512).
  - The actor's 152k-vocabulary logits are kept in FP32 together with their entropy, about 1.2 GB per sequence.
  - The laptop smoke test ran out of memory at larger sizes, though on a 12 GB card.
  - Micro-batch size changes speed and memory, not the update itself, so the pilot's per-stage times decide whether to raise them.

**First H100 pilot (2026-11-10, `results/r1_pilot/`).** It ran an earlier `pilot.sh`: no KL penalty or entropy bonus, 512-token prompt budget, micro-batches 16/32.
- **Timing:** 34.5 s per step (median). The actor update takes 17.3 s, the proximal log-probabilities 7.4 s, generation 4.1 s and the critic update 2.9 s. Validation takes 18 s. That projects to 4.2 h per 435-step run.
- **Mismatch:** the BF16 rollout–trainer mismatch was 5–6e-4 (k3 KL per token).
- **Reward:** greedy validation accuracy at step 10 was 0%. This is expected: VeRL's own baseline was at 1.3% at step 9 and passed 45% only by step 29, once the "####" answer format is learned.
- **Shutdown:** a "DataLoader worker … killed" error appears after the final metrics. It is a shutdown artifact.

**Second H100 pilot: the final configuration (2026-11-10, `results/r1_pilot2/` and `results/r1_pilot2_nockpt/`).**
- **Timing:** 44.8 s per step (median), which projects to **5.5 h per 435-step run**.
  - The added reference-model pass takes 7.5 s.
  - The actor update, now with the entropy term at micro-batch 8, takes 20.5 s.
  - Proximal log-probabilities take 7.5 s and generation 4.1 s.
- **Budget:** about 4.6 days for the 20 planned runs.
- **Gradient checkpointing:** turning it off for actor and critic gave 45.2 s per step, no faster. It stays on.
- **Run-to-run variation:** the no-checkpointing run's greedy validation at step 10 was 4.8% against 0%. Checkpointing does not change the math, so this is run-to-run variation; VeRL's vLLM rollouts are not bit-reproducible.
- **Memory:** neither run ran out of memory.

**INT8 pilot, R3 (2026-11-10, `results/r3_pilot/`).** The final configuration with QuRL's rollout: INT8 W8A8, TIS at C = 2, ACR.
- **Rollout path:**
  - `int8_refit.txt`: vLLM ran `CutlassInt8ScaledMMLinearKernel`.
  - The synced weights arrive as **FP32**, so the INT8 codes come from the FP32 master weights directly.
  - All 168 linear weights were re-quantized at each of the 11 syncs.
  - `check_int8_dump.py`: after sync 1, the codes and scales are bit-identical to an independent quantization.
- **Cost:** 44.6 s per step, the same as R1. INT8 generation is no faster than BF16 at 0.5B (4.1 s). About 5.5 h per run for every arm.
- **Mismatch:** the rollout–trainer KL (k3) is 1.1e-2 at step 1, matching S1's step-0 INT8 W8A8 value for this model (9.5e-3). It rises to 2.7e-2 by step 3 and then stays near 2.0e-2, about 38× R1's flat 5e-4. The rise is not yet attributed.
- **TIS and ACR:** TIS truncated 0.2% of tokens at step 1 and 0.7–0.9% afterwards; the largest ρ in a batch reached 122. ACR changed the clipping decision for 1–7e-5 of tokens.
- **Code flips** (`int8_flip_rate.py`): 0.21% of layer 0's Q/K/V codes flipped in the first PPO step. Proposition 4's |d|/Δ, with |d| estimated per row from the move of the absmax element, predicts 0.20%, or 0.24% with the grid's own shift.

## Table 1 check: R1 and R3 at the paper's lr 1e-5
- **Runs:** full 435-step runs, one seed each, run back to back (about 11 h), validating every epoch (every 29 steps).
- **Targets:** the paper's Table 1 reports 55.35% GSM8K test accuracy for BF16 RL and 53.55% for QuRL INT8.
```bash
ssh h100 "nohup bash -c 'bash qurl/s3/pilot.sh t1_r1_lr1e-5 435 actor_rollout_ref.actor.optim.lr=1e-5 trainer.test_freq=29; ARM=R3 bash qurl/s3/pilot.sh t1_r3_lr1e-5 435 actor_rollout_ref.actor.optim.lr=1e-5 trainer.test_freq=29' > qurl/s3/table1.log 2>&1 < /dev/null &"
ssh h100 "bash qurl/s3/progress.sh t1_r1_lr1e-5 t1_r3_lr1e-5"
```
`progress.sh` prints, per run, the step reached, the last step time, the training score, the mismatch KL and the latest validation accuracies. The runs save no checkpoints, so a crash means restarting that run under a new tag.

**Results (H100, 2026-11-10, `results/t1_r1_lr1e-5/` and `results/t1_r3_lr1e-5/`): not reproduced.** The P1 experiments report §3.4 has the analysis.

| Run | GSM8K test accuracy at step 435 | Best of 15 validations | Paper (Table 1) | Wall clock |
|---|---|---|---|---|
| R1, BF16 | 0.0% | 0.0% | 55.35% | 2.9 h |
| R3, QuRL INT8 | 2.9% | 3.4% (step 58) | 53.55% | 4.9 h |

- **R1:** 7 correct training answers in 435 steps, all but one in steps 1–4. Responses shrank from about 310 tokens to 2–3, and the KL from the reference model reached about 5 nats per token.
- **R3:** a 2–3% training score from step 30, with ~9-token answers. From step 290, nearly every response ran to the 512-token cap at an entropy of 0.001.
- **Cause: the learning rate.** The 10-step pilots (`r1_pilot2`, `r3_pilot`) ran the same configuration at 1e-6. After 10 steps the BF16 policy's KL from the reference was 2.1e-3 at 1e-6 against 0.15 at 1e-5, and only the 1e-6 run kept producing correct answers. At step 1 the critic's fresh value head is far off: values from −20 to +33 for returns in [0, 1]. So the early advantages are mostly noise, and at 1e-5 the actor follows that noise ten times as fast.
- **The paper's run:** it states lr 1e-5 and nothing about the critic's lr, warmup, KL penalty or entropy bonus. Either its actor ran at 1e-6, with 1e-5 the critic's rate as in VeRL's baseline, or it used a stabilizer it does not state.
- **For the UAQ experiment:** the check of the INT8 path moves to R1 vs. R3 at 1e-6, three seeds each. The first queued run (`uaq_R3_s1`) should validate near VeRL's 45.8% at step 29. If it stays near 0, run `uaq_R1_s1` next to tell a configuration problem from an INT8 problem.
- **Mismatch vs. entropy:** in R3, k3 tracked the policy's entropy over the run (Spearman 0.94): 7.2e-3 (median) at entropy > 0.3, 2.6e-5 at entropy < 0.01. Compare the arms' mismatch at matched entropy.
- **Shutdown:** R3's log ends with the same harmless "DataLoader worker … killed" error as the pilots.

## QuRL components: what VeRL v0.9.1 provides and what S3 adds
| Component | Needed by | VeRL v0.9.1 | Plan |
|---|---|---|---|
| BF16 vLLM rollout | R1, R2 | yes | — |
| INT8 W8A8 rollout, re-quantized every step (per-channel weights, per-token activations) | R3–R6 | no; only FP8 blockwise is refitted on weight sync (`rollout.quantization=fp8`) | **done:** `qurl_verl/int8_rollout.py` refits vLLM's compressed-tensors INT8 W8A8 layers; verified on the laptop |
| TIS, token level, cap C | R3–R6 | yes: `algorithm.rollout_correction.rollout_is=token`, `rollout_is_threshold=C` | — |
| ACR (upper clip (1+ε)·max(1, ρ/C)) | R3–R6 | no | **done:** `qurl_verl/acr.py`, a custom policy loss that receives the untruncated ρ and applies TIS and ACR itself; unit-tested, and run on the laptop |
| UAQ checkpoint (W_QKV/s, γ·s) | R2, R4 | — | **done:** `make_uaq_checkpoint.py` writes it in FP32, to be used as the actor's model path |
| R5 optimizer: per-group learning rate, ε and weight decay, and rule (c) clipping | R5 | no (one optimizer group; global-norm clip) | **done:** `qurl_verl/r5_optim.py` on an FSDP2 actor; bit-exact against R4 at s = 2 in VeRL's own engine (`r5_check.py`) |
| Global learning rate × 1.5 | R6 | yes | — |

**Written so far:**
- **`make_uaq_checkpoint.py`** writes the UAQ checkpoint for R2/R4 in FP32. It leaves the q/k/v biases unchanged and checks the result itself: FP32 logits on the CPU must be bit-identical at s = 2 and agree to FP32 rounding otherwise. On the laptop it gave bit-identical logits at s = 2 and a relative difference of 4.4e-6 at s = 1.5.
  ```bash
  ssh h100 "~/venvs/qurl/bin/python qurl/s3/make_uaq_checkpoint.py --s 1.5 --out ~/qurl/models/qwen0.5b-instruct-uaq1.5"
  ```
- **`qurl_verl/`** is the VeRL plugin. Both modules have run together in a 2-step laptop VeRL job (the R3 configuration: INT8 rollout, TIS at C = 2, ACR), but not yet on the H100.
  - `acr.py`: the ACR policy loss (`loss_mode=qurl_acr`), imported into the training workers via `actor_rollout_ref.model.external_lib=qurl_verl`.
    - It passes a CPU unit test. With no truncated token it is bit-identical to VeRL's vanilla PPO loss with TIS weights; with truncation it matches a float64 reference of QuRL's Eq. 9 to 6e-10.
    - It logs `actor/qurl/tis_trunc_frac`, `actor/qurl/acr_kept_frac` and `actor/qurl/rho_max`.
  - `int8_rollout.py`: `QuRLWorkerExtension`, the INT8 refit described below. vLLM loads it in its worker processes via `engine_kwargs.vllm.worker_extension_cls`.
  - `int8_w8a8_config.json`: the compressed-tensors config of S1's llm-compressor INT8 checkpoints, so vLLM builds the same INT8 W8A8 layers as in S1.

  - `r5_optim.py`: the optimizer path behind every arm of the UAQ experiment. It patches the actor engine only; the critic and the forward-only reference keep VeRL's code.
    - `QURL_OPT=r5` gives Theorem 2(b)–(c)'s per-group settings for s = `QURL_S`: lr sη on Q/K/V weights and η/s on input-norm gains, Adam ε/s and sε, weight decay λ/s and sλ. The global clip norm is computed in UAQ coordinates, on (s·g_W, g_γ/s, g_rest).
    - `QURL_OPT=plain` runs the same code with s = 1. It is used for the arms R5 is compared with, so every arm runs identical clipping arithmetic.
    - `QURL_CLIP=ordinary` keeps R5's groups but drops rule (c): the R5o control.
    - It needs the actor on **FSDP2** (`actor_rollout_ref.actor.strategy=fsdp2`). Under FSDP1, per-parameter groups need `use_orig_params=True`, which crashes with bf16 mixed precision in PyTorch 2.11 (a gradient-dtype check).
    - One data-parallel rank only.

**R5's exactness check (`r5_check.py`, laptop, 2026-11-10).** The check runs three actor updates on one fixed GSM8K batch through VeRL's FSDP2 actor engine: bf16 mixed precision, PPO loss with the entropy bonus, lr 1e-5, weight decay 0.01, clip 0.1, strict deterministic algorithms. There is no rollout, so every run sees identical data. At s = 2:
- **R4 repeat:** bit-identical to R4 in all 494M weights, so the engine is deterministic.
- **R5:** bit-identical to R4, with R4's weights mapped back (W = 2W', γ = γ'/2), in all 494M weights. Its pre-clip gradient norms are also identical (3.66, 3.18, 5.96).
- **R5o** (R5 with ordinary clipping): its gradient norms differ (2.66, 2.33, 3.83), and so do 96–100% of its weights.
- **What was exercised:** clipping bound at every update; weight decay was active (lr·wd = 1e-7).

`r5_check.sh` repeats the check on the H100. Run it once the GPU is free, i.e. after the Table 1 check.

**H100 result (2026-11-10, `results/r5_check/`), run by the queue: passed.**
- R5 and the R4 repeat are bit-identical to R4 in all 494M parameters, with identical gradient norms (3.859, 2.573, 3.110).
- R5o differs: 99.98% of W's elements after three updates, with gradient norms 2.791, 1.884, 2.272.
- The check took 1.6 minutes. The `params.pt` snapshots (2 GB each) stay on the server.

## UAQ experiment: `arm.sh`
`[QURL_S=1.5] [BASE_LR=1e-6] bash qurl/s3/arm.sh <R1..R6> <tag> [steps] [overrides]` wraps `pilot.sh`.
- **Shared code path:** every arm trains the actor with FSDP2 through `r5_optim.py`, so the arms differ only in what each arm changes.
- **R2/R4:** use the UAQ checkpoint at `~/qurl/models/qwen0.5b-instruct-uaq<s>`, written by `make_uaq_checkpoint.py`.
- **R6:** uses lr s·`BASE_LR`.
- **Why a wrapper:** `pilot.sh` itself is left untouched while the Table 1 chain is still running it on the H100. Overwriting a shell script that a running bash is reading can break it, and the R3 half of that chain starts hours after the R1 half.
- **Laptop test of the full loop:** a 2-step run of `arm.sh R5` (FSDP2 actor, R5 settings at s = 1.5, INT8 rollout, TIS, ACR) completed. The plugin printed R5's settings; clipping bound at both steps (pre-clip norms 10.4 and 9.4 against 1.0); all 168 weights were re-quantized at each of three syncs; the INT8 mismatch was k3 = 0.012.
- **FSDP2 on the H100** (from the queue's first run): as fast as FSDP1 at matched response length (46.5 s per step at steps 2–10, against 44.6–46.3 s in the FSDP1 pilots), with the actor peaking at 36 GB of GPU memory.

### The 18 runs: `uaq_queue.sh`
```bash
ssh h100 "nohup bash qurl/s3/uaq_queue.sh > qurl/s3/uaq_queue.log 2>&1 < /dev/null &"
ssh h100 "tail -5 qurl/s3/uaq_queue.log; bash qurl/s3/uaq_queue.sh --progress"
```
- **What it runs:** R1–R6 × seeds 1–3 at actor lr 1e-6 and s = 1.5, 435 steps each, validating every epoch. At about 5.5 h per run, that is roughly 4 days back to back. The first run took 4.5 h, so about 3.4 days.
- **Order:** the core arms first, R3, R4 and R5 for seeds 1–3 (H1, 9 runs), then R1, R2 and R6 for seeds 1–3 (H2, H3).
- **Seeds:** seed k sets `data.seed=k` (prompt order) and `rollout.seed=42+k` (sampling). The six arms of one seed share both, so the comparisons between arms are paired by prompt order.
  - Without `data.seed`, VeRL's unseeded generator gives every run the same prompt order. That is the case for the pilots and the Table 1 runs.
- **Waits:** it can be started at once. It waits while any other `pilot.sh` runs or the GPU is busy, so it begins when the Table 1 chain ends.
- **Prerequisites:** it writes the s = 1.5 UAQ checkpoint if missing and runs `r5_check.sh` on the H100. It refuses to start the runs unless R5 and the R4 repeat are bit-identical to R4.
- **Resumable:** finished runs (those with a `summary.txt`) are skipped. An unfinished run's folder is moved aside to `<tag>.incomplete.<time>` and the run starts over.
- **Stops on failure:** it stops at the first failed run, so a systematic problem cannot burn through the rest. `CONTINUE_ON_FAIL=1` overrides this.
- **Options:** `--list` shows the plan and the status of each run. `DRY_RUN=1` prints the commands without running them.
- **Tested on the laptop:** a dry run checked the skip, move-aside and seed logic, and `r5_check.sh` ran whole there and passed.
- **Copying results back.** Bundle finished runs on the server without the `.pt` snapshots (the `params.pt` files and `int8_dump/`), then unpack the archive locally into `experiments/s3/`:
```bash
ssh h100 "tar czf uaq_first.tgz --exclude='*.pt' -C qurl/s3 uaq_queue.log results/r5_check results/uaq_R3_s1"
scp h100:uaq_first.tgz experiments/s3/
```

**Results so far (P1 experiments report §3.5).** The queue started its first run at 13:09 UTC on 2026-11-10, after the gate passed. The nine H1 runs (R3, R4 and R5 for seeds 1–3) were done at 06:53 UTC on 2026-11-12, when R1 seed 1 started.

| Run | Arm | Seed | GSM8K test accuracy at step 435 | Best (step) | Run time |
|---|---|---|---|---|---|
| `uaq_R3_s1` | R3, QuRL INT8 | 1 | 50.1% | 52.5% (232) | 4.5 h |
| `uaq_R4_s1` | R4, QuRL + UAQ | 1 | 52.2% | 52.2% (435) | 4.5 h |
| `uaq_R5_s1` | R5, Theorem 2 settings | 1 | 46.6% (0% until step 377) | 46.6% (435) | 4.9 h |
| `uaq_R3_s2` | R3 | 2 | 49.6% | 53.8% (377) | 4.8 h |
| `uaq_R4_s2` | R4 | 2 | 50.2% | 50.4% (261) | 4.5 h |
| `uaq_R5_s2` | R5 | 2 | 50.4% | 50.4% (203, 435) | 4.7 h |
| `uaq_R3_s3` | R3 | 3 | 51.9% (5.5–8.3% until step 116) | 52.6% (232) | 4.3 h |
| `uaq_R4_s3` | R4 | 3 | 51.2% | 53.3% (290) | 4.8 h |
| `uaq_R5_s3` | R5 | 3 | 48.4% (0% at steps 29 and 58; 33.6% at step 319) | 48.4% (435) | 4.7 h |

Late accuracy (the H1 primary endpoint, mean of the last five validations, from `h1_analysis.py`): R3 50.2 / 51.4 / 51.0, R4 49.0 / 49.0 / 51.2 and R5 11.8 / 49.9 / 44.5 for seeds 1–3. The H1 test withholds its verdict until seeds 4–5 are in, and `h23_analysis.py` withholds H2 and H3 until R1, R2 and R6 finish. The R4 − R3 sensitivity contrast (seeds 1–3) is complete.

**Seed 3 of R4 and R5 (H100, finished 2026-11-12, copied back the same day as `uaq_s3.tgz`).**
- **Both runs are clean:** each re-quantized all 168 linear weights at every one of the 436 syncs on the CUTLASS kernel and ran with `data.seed=3` and `rollout.seed=45`. Each logged the intended optimizer settings (R4: `mode plain` on the UAQ checkpoint; R5: `mode r5, s 1.5`) and ended with the harmless DataLoader shutdown error.
- **R5, seed 3, stalled early, for about 80 steps.** It had 6, 3, 4 and 1 correct training answers at steps 1–4, then none at steps 5–7, 9–80 and 82. Validation was 0.0% at steps 29 and 58, 25.5% at step 87 and 46.4% at step 116. The training score first reached 0.3 at step 92.
  - **A quiet stall, unlike the other two:** over steps 1–87 the policy stayed close to the reference model (KL 0.006–0.047, window medians). The entropy fell from 0.55 to 0.38, the responses stayed long (median 243–304 tokens), and the mismatch stayed low (k3 1.7–2.6e-2).
  - **The other two stalls:** R5 seed 1's entropy rose to 0.93 by step 10 and its KL from the reference was 1.1 at step 119. R3 seed 3 gave 7-token answers, with KL above 1 and the highest mismatch of any run (0.195).
  - **What the three share:** the few correct answers of the first steps disappeared within ten steps, and none came back for a long stretch. The task reward is zero throughout such a stretch, so the drift in the meantime is not driven by the task, and each run drifted differently.
- **R5, seed 3, also dipped once late:** validation fell to 33.6% at step 319, between 45.1% (step 290) and 46.3% (step 348).
  - **No sign in training:** the training score stayed at 0.51–0.61 over steps 300–335.
  - **During the run's highest mismatch:** k3 was 0.06–0.13 per step over steps 315–331, about 1.5× the run's usual level.
  - **Effect on H1:** over the other four late validations, R4 − R5 is 3.7 points; with step 319 included, the registered late accuracy gives 6.7.
  - **Across all nine runs (exploratory):** how far a validation falls below its two neighbours correlates weakly with the mismatch over the 10 steps before it (Spearman +0.22 over 95 validations after learning; +0.01 to +0.73 per run). The two largest dips came at 1.5× and 2.9× their run's median mismatch: 12.1 points here, and 4.9 points in R3 seed 1 at step 145.
- **R4, seed 3, wobbled but did not stall.**
  - **The wobble:** no correct answer at steps 4–6, and the highest early entropy of any run (0.96 at step 11). The mismatch's 29-step median reached 0.128 (steps 26–54), second only to R3 seed 3's stall.
  - **The recovery came at once:** 15 correct answers at step 10 and a training score of 0.3 from step 22. Validation was 42.4–45.6% through step 116 and 46.8–53.3% afterwards.
- **Seed 3 was hard for all three arms.** Early reward (mean training score over steps 1–58) was R3 2.9, R4 34.1 and R5 0.1. In seeds 1–2 it was 34.4–43.7 for every run except R5 seed 1 (0.1).
- **Early stalls after seeds 1–3 (descriptive):** three of the nine runs were below 10% at the first validation (step 29): R5 seed 1 (0.0%), R3 seed 3 (5.5%) and R5 seed 3 (0.0%). The other six were at 41.4–46.4%.
  - **By arm:** R3 1 of 3, R4 0 of 3, R5 2 of 3. If every run stalled with the same chance, a 2–1–0 split would be the most likely pattern: 54 of the 84 equally likely ways to place 3 stalls among 9 runs. So the split says nothing about the arms.
  - **Side hypothesis** (the 1.5× Q/K/V rate shared by R4 and R5 causes stalls): R4 and R5 stalled in 2 of 6 runs, R3 in 1 of 3. Still not supported.
  - **No collapse:** all three recovered. The lowest late accuracy is R5 seed 1's 11.8%, above the 10% threshold.
- **H1's registered verdict will almost surely be "inconclusive".** Per seed, R4 − R5 is +37.2, −0.9 and +6.7 points of late accuracy.
  - **Supported:** impossible. Whatever seeds 4 and 5 give, the far end of the 90% CI stays at least 23.1 points from 0, against the 2-point margin.
  - **Contradicted:** only if R4 − R5 lands between about 31 and 35 points in both seeds 4 and 5. The best case, 33.3 twice, gives a 95% CI of [0.02, 43.8].
  - **Other outcomes fall short:** two clean pairs (0 each) give a 95% CI of [−11.6, +28.8]. Two full R5 stalls (40–49 each) give lower ends of −0.3 to −1.3.
  - **Inconclusive:** every other outcome. The frozen rules are unchanged. Seeds 4–5 still run as registered, since they sharpen the estimate, the secondary endpoints and the stall count.
- **UAQ's own effect, R4 − R3 (the registered sensitivity contrast, seeds 1–3), is complete:**
  - **Mismatch:** R4's mean k3 is 1.036× R3's (per seed 1.063, 1.134, 0.923; 90% CI [0.867, 1.238]). That lies inside the 0.80–1.25 bounds, so UAQ changed the training mismatch by less than the registered margin.
  - **Accuracy:** late −1.12 points (90% CI [−3.30, +1.05]), final +0.66 [−1.64, +2.96] and reward AUC +4.15 [−9.35, +17.65]. All three are inconclusive at the ±2-point margin.
  - **Entropy:** 0.017, 0.037 and 0.045 higher with UAQ (descriptive).
- **Mismatch in seed 3:**
  - **R4:** k3 was 1.02e-2 at step 1, with medians of 7.44e-2 over steps 60–175 and 3.95e-2 over steps 400–435. That is the same rise and partial fall as the other runs without a stall.
  - **R5:** 1.14e-2 at step 1, then 3.12e-2 and 3.75e-2 over the same stretches. It stayed low during the stall, as in R5 seed 1.
  - **TIS and ACR:** TIS truncated 0.21–1.20% (R4) and 0.25–1.80% (R5) of tokens per step, and the largest ratios were 6,431 and 2,239. ACR kept 0.011% and 0.008% of tokens on average.
  - **Correlations:** k3 against entropy, Spearman −0.38 (R4) and −0.28 (R5); against the KL from the reference, −0.12 and +0.83. R5's +0.83 comes from the stall, when both were low.

**Seed 2 and R3 seed 3 (H100, finished 2026-11-11, copied back 2026-11-12 as `uaq_s2.tgz`).**
- **All four runs are clean:** each re-quantized all 168 linear weights at every one of the 436 syncs on the CUTLASS kernel, logged the intended optimizer settings (R5: `mode r5, s 1.5`, lr 1.5e-6 on W and 6.67e-7 on γ), and ended with the harmless DataLoader shutdown error.
- **R5, seed 2, learned normally.** Two zero-score steps (5 and 10), 44.7% at step 29, then 47.5–50.4% at every validation. Seed 1's 370-step stall did not recur.
- **R3, seed 3, stalled too, for about 125 steps.** No correct training answer at steps 2–22 (nor at 25).
  - **Steps 30–125:** the median response was 7 tokens long, the training score 1–14%, and the KL from the reference model above 1 at every step but one (peak 3.0 at step 93). Validation stayed at 5.5–8.3% through step 116.
  - **Recovery:** between steps 125 and 135 the policy returned to long answers (KL from the reference 2.1 → 0.10, training score 8% → 42%). Validation reached 47.2% at step 145 and 49.2–52.6% from then on.
  - **During the stall the mismatch rose:** k3's 29-step median peaked at 0.195 (steps 40–68), against at most 0.098 in the other runs. TIS truncated a median 2.1% of tokens per step over steps 30–125 (at most 3.4%), against at most 1.6% in every other run.
  - **What it shows:** an early stall is not specific to R5 or to rule (c). R3 runs VeRL's ordinary optimizer settings (`mode plain`) on the plain checkpoint. This supports the reading of R5 seed 1 as a possible outcome of the configuration at 1e-6. Its late accuracy is unaffected (51.0%).
- **Early learning varies a lot between seeds of one arm:** step-29 validation is 41.4 / 46.4 / 5.5% for R3's three seeds, so the early-reward secondary endpoint will be noisy.
- **H1's registered test cannot return "supported" any more.** Seed 1's difference R4 − R5 is 37.2 points of late accuracy; seed 2's is −0.9. For any results of seeds 3–5, the farther end of the 90% CI stays at least 20.4 points from 0, against the 2-point margin. "Contradicted" (95% CI excluding 0) needs R4 − R5 to average at least 16 points over seeds 3–5. With per-seed differences of either about 0 or a stall's ~40 points, only R5 stalling in all three remaining seeds (and R4 not) gets there. Otherwise the verdict is "inconclusive". This was computed from the frozen rules and changes nothing in them; the comparison without seed 1 is exploratory only, since 11.8% is above the 10% collapse threshold.

- **R4, seed 1:** learned faster than R3 at first: 49 correct answers per batch by step 10, against R3's 14. It then ran 47.0–52.2% over the last five validations.
- **R5, seed 1: stalled for 370 steps, then learned; no implementation fault found.**
  - **What happened:** the policy's entropy rose from 0.54 to 0.93 by step 10, and no training answer was correct from step 4 to step 373.
  - **The recovery:** the KL penalty pulled the policy back toward the reference model (per-token KL 1.1 at step 119, 0.03–0.05 by steps 300–390). Correct answers returned at step 374 and steadily from step 395. Validation went 0.0% (through step 377) → 12.4% (406) → 46.6% (435).
  - **Registered endpoints:** late accuracy 11.8%, final accuracy 46.6%, reward AUC 2.9, early reward 0.1. Late accuracy is just above the 10% collapse threshold fixed for the exploratory analyses, so by that rule the run does not count as collapsed. The rule stays as fixed.
  - **Same failure as Table 1:** this is the early-drift failure of the 1e-5 runs.
  - **Settings confirmed:** the log shows the intended settings (`[qurl optimizer] mode r5, s 1.5`), with no non-finite gradients. VeRL's constant scheduler keeps each group's rate.
  - **Not paired at step 1:** R3 and R5 start from identical weights with the same vLLM seed, yet sampled different responses at step 1. The pairing by seed fixes the prompt order only.
  - **Reading:** a long stall looks like a possible outcome at 1e-6, not an R5-specific fault. The note's "Reading" has the side hypothesis about the 1.5× Q/K/V learning rate.
- **R3, seed 1:**
  - **Learning:** 41.4% at step 29 (VeRL's BF16 log: 45.8%), and 50.0–50.9% over the last five validations.
  - **Speed:** 4.5 h, not 5.5 h. Responses shrink from about 310 tokens to 135, so the median step time drops to 35.8 s, which puts the 18 runs at about 3.4 days.
  - **INT8 path:** all 168 linear weights were re-quantized at each of the 436 syncs, on the CUTLASS kernel.
  - **Mismatch:** k3 went from 1.1e-2 at step 1 to medians of about 8e-2 over steps 60–175, then fell to 3.6e-2. It did not track entropy (Spearman 0.13) or the KL from the reference model (−0.22). TIS truncated 0.24–1.4% of tokens; the largest ρ reached 2,173.
  - **Validation dips:** 44.5% at step 116 and 43.0% at step 145, during the highest-mismatch stretches. Across all validations, accuracy and mismatch barely correlate (−0.15).

### Seeds 4–5 of R4 and R5 for H1: `uaq_seeds45.sh`
Decided on 2026-11-11, as part of the pre-registered H1 test: five pairs of R4 and R5 instead of three.
```bash
scp experiments/s3/uaq_seeds45.sh h100:qurl/s3/
ssh h100 "nohup bash qurl/s3/uaq_seeds45.sh > qurl/s3/uaq_seeds45.log 2>&1 < /dev/null &"
ssh h100 "tail -3 qurl/s3/uaq_seeds45.log; bash qurl/s3/uaq_seeds45.sh --list"
```
- **Runs:** R4 and R5 with seed 4, then with seed 5 (`data.seed` 4/5, sampling seeds 46/47). That is four runs, about 18 h, with the same `arm.sh` call as the main queue.
- **A separate file on purpose:** `uaq_queue.sh` is running on the server and must not be overwritten. Copying this script touches nothing else.
- **Starts after the main queue:**
  - It can be started at once. It waits until no bash process is running `uaq_queue.sh`, no `pilot.sh` runs and the GPU is free.
  - It then checks that all 18 main runs have a `summary.txt`. If any is missing it stops without starting anything; `FORCE=1` overrides this.
  - It also checks that R5's exactness check passed and that the UAQ checkpoint exists.
- **Like the main queue:** it is resumable, stops at the first failed run, and supports `--list`, `--progress` and `DRY_RUN=1`.
- **Tested in WSL** with `DRY_RUN=1`:
  - It refused to start while the main runs were missing.
  - With `FORCE=1`, it moved an unfinished run aside, skipped a finished one, and issued the four `arm.sh` calls with the right seeds.
  - It waited for a dummy process named like the main queue and started after it ended.
  - The wait matches only a bash process running the script. A first version also matched any command line that mentioned `uaq_queue.sh`, and hung in this test.

### H1 analysis: `h1_analysis.py` (pre-registered 2026-11-11)
The test of H1 (R4 ≡ R5 in distribution) was fixed before any R5 result; the P1 experiments report §2.5 has the rules and the reasons.
- **Primary endpoint:** late accuracy, the mean GSM8K test accuracy over the last five validations.
- **Test:** paired by seed, R4 − R5. H1 is supported if the 90% CI lies within ±2.0 points, contradicted if the 95% CI excludes 0, and inconclusive otherwise.
- **Secondary endpoints:** final accuracy, reward AUC, and the mismatch ratio with bounds 0.80–1.25.
- **Sensitivity:** R4 − R3, UAQ's own effect.
- **Frozen:** SHA-256 `5ac6da470e152a7b1680921bdcb2c1281d09cc52b63ce015f144c9c05f368f6f`. A change after registration is recorded in the note, and both versions' results are reported.
- **Verdicts:** only once all registered pairs are complete. Seeds 4–5 were added on 2026-11-11, so the H1 test runs on seeds 1–5 (`--seeds 1 2 3 4 5`), and the R4 − R3 contrast on seeds 1–3.
- **Exploratory, added after R5 seed 1 collapsed (before any other R4/R5 seed):** collapsed runs per arm, where collapsed means late accuracy below 10%, and the registered test repeated on the pairs where neither run collapsed. These are reported as exploratory and do not replace the registered verdict.
- **Tested:** `--self-test` checks the t quantiles and the decision logic. End-to-end runs on synthetic results with known answers, including missing and truncated runs, gave the expected verdicts.
```bash
python h1_analysis.py --seeds 1 2 3 4 5  # reads results/uaq_*/metrics.jsonl, writes results/h1_analysis.txt
python h1_analysis.py --self-test
```

### H2 and H3 analyses: `h23_analysis.py` (pre-registered 2026-11-11)
Fixed before any R1, R2 or R6 run; the P1 experiments report §2.5 has the rules and the reasons.
- **H2** (UAQ's effect is not quantization-specific): the interaction (R4 − R3) − (R2 − R1) per seed. Supported if the 90% CI lies within ±2.0 points, contradicted if the 95% CI excludes 0. For the mismatch it is the ratio of ratios (R4/R3)/(R2/R1), bounds 0.80–1.25.
- **H3** (R6 ≠ R4): R6 − R4 per seed. Supported if the 95% CI excludes 0, contradicted if the 90% CI lies within ±2.0 points.
- **Endpoints:** those of H1, plus early reward, the mean training score over steps 1–58 (secondary). Seeds 1–3.
- **Also printed:** the simple effects R4 − R3, R2 − R1 and R6 − R3 (no verdict), and the exploratory analyses for H1–H3: collapsed runs per arm (late accuracy below 10%) and each test without collapsed seeds.
- **Shared arithmetic:** it imports the statistics of the frozen `h1_analysis.py`.
- **Frozen:** SHA-256 `5ed1a3b280b340550b009e0e54c5d95ee2e00a00f2bddd74b7fa612af1dfb3e4`.
- **Tested:** on synthetic runs with known answers, including a collapsed run. On the real results so far it withholds every verdict.
```bash
python h23_analysis.py                   # writes results/h23_analysis.txt
python h23_analysis.py --self-test
```

**What the laptop INT8 run verified:**
- **Refit coverage:** each of the three syncs re-quantized all 168 linear weights (7 per layer).
- **Exact codes:** after sync 1, layer 0's fused QKV codes and scales are bit-identical to an independent quantization of the checkpoint (0 of 1,032,192 codes differ).
- **Tracking:** after the first PPO step, 0.0015% (Q), 0.0035% (K) and 0.028% (V) of those codes changed, with no row scale changing. The rollout follows the updates through re-quantization.
- **Mismatch:** the rollout–trainer mismatch was k3 ≈ 0.013 per token, about 25× the H100's BF16 floor and close to S1's INT8 W8A8 value for this model (9.5e-3).
- **TIS:** it truncated 0.1–0.5% of tokens at C = 2.
- **ACR:** it kept no extra tokens in this 16-prompt run.

**INT8 refit design.** In vLLM 0.24, a compressed-tensors INT8 W8A8 layer (`CompressedTensorsW8A8Int8`, channel strategy, dynamic symmetric per-token activations) holds an int8 `weight` [out, in] and a float32 `weight_scale` [out, 1].
- Its kernel may re-layout `weight` after loading, for example by transposing it for CUTLASS.
- Each output row has its own scale, so q/k/v (fused `qkv_proj`) and gate/up (`gate_up_proj`) shards quantize independently.
- On every weight sync, each synced linear weight is quantized row by row as it arrives: scale = row absmax · fl32(1/127), codes = round(W / scale) clamped to ±127. The codes and scales are then written into that shard's rows of the live int8 `weight` and `weight_scale`, in the kernel's layout.
  - The scale is written as a multiplication on purpose. PyTorch divides a GPU tensor by a scalar through its reciprocal and a CPU tensor by true division, and the two differ by one ulp in about 5% of rows. Either form is an absmax quantizer, exactly homogeneous under power-of-two rescaling (Theorem 1).
- Everything else (embeddings, norms, biases, lm_head) goes through vLLM's ordinary `load_weights`.
- After the last bucket, VeRL re-runs vLLM's `process_weights_after_loading` on the whole model. That suits an unquantized model, but here it would transpose the int8 weights a second time, and it crashes on the attention layers' KV-scale attributes. The extension therefore makes that call a no-op during a sync.
- Any synced Q/K/V/O/MLP weight that does not map to an INT8 layer raises an error instead of being loaded as BF16 into int8 parameters.
- vLLM is launched with a matching `quantization_config` override (compressed-tensors, W8A8 int8, lm_head ignored) and VeRL's dummy load format.
- `QURL_INT8_DUMP_DIR` (test hook) saves layer 0's fused QKV codes and scales after the first two syncs.

## Laptop smoke test (not reported)
Laptop runs only check that the scripts work; see the P1 experiments report §2 for what ran where. The test ran `pilot.sh` for 2 steps at a tiny scale (batch 16, 128-token responses, FSDP state offloaded to the CPU, 12 GB GPU), using `smoke_pilot.sh`, which lives in the scratchpad.
- **Found and fixed for the H100:**
  - the dead flash-attn wheel URL in VeRL's lockfile;
  - the critic's hard-coded flash attention (patched in `setup_verl.sh`).
- **Laptop only.**
  - WSL2 has no CUDA IPC, which VeRL uses to hand the trainer's weights to vLLM, and VeRL's CUDA platform always reports IPC as available.
  - On the laptop, `verl/plugin/platform/platform_cuda.py` was therefore patched to report IPC unsupported under WSL. That switches VeRL to its shared-memory transfer.
  - Native Linux, as on the H100, keeps VeRL's default CUDA-IPC path; this patch is not part of `setup_verl.sh`.
  - VeRL's default 8 reward workers and 8 agent-loop workers, each a full Python process with torch loaded, filled the laptop's 31 GB of RAM together with the CPU-offloaded FSDP state, and Ray killed a worker. The laptop runs therefore use one of each. The H100 keeps the defaults and needs no CPU offload.
  - FP32 master weights and Adam state for both actor and critic do not fit the laptop's 12 GB GPU during the optimizer step, so the laptop runs keep them in BF16. The H100 runs keep FP32 master weights, VeRL's default.
  - With the entropy bonus switched on, the laptop runs also need actor micro-batch 1, log-probability micro-batch 2 and VeRL's entropy checkpointing.
  - With an FSDP2 actor, CPU offload held about 18 GB in one process. Ray's OOM killer then stopped a worker after step 1, and generation hung. The laptop `arm.sh` runs therefore keep the actor on the GPU in BF16 (about 4 GB) and offload only the critic and the reference.
  - vLLM 0.24 picks its CUTLASS INT8 kernel on any CUDA GPU, but that kernel has no build for the laptop's SM120 ("Int8 not supported"). The laptop INT8 run therefore sets `VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel` and uses vLLM's Triton INT8 kernel. The H100 (SM90) runs CUTLASS, as in S1.
