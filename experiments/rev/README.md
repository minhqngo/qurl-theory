# Revision experiments X1–X7 (after the blind reviews of the P1 paper, November 2026)

The P1 paper had two blind reviews: a fresh agent's and a ChatGPT agent's. They agreed on the main weaknesses. These experiments answer the ones that the single H100 can answer before **28 November 2026**, when the server's lease ends.

All of them run on the single H100 NVL (alias `h100`), one at a time, from one resumable queue. The design, decision rules and frozen analyses are registered in the P1 experiments report (`latex/p1_experiments.tex`, "Revision experiments X1–X7"), before any of these data exist. The five analyses' SHA-256 hashes are in `registration.sha256`, and the queue refuses to start if a frozen analysis has changed.

| Job | Question (reviewer concern) | Runs | Time on the H100 |
|---|---|---|---|
| **X1** `x1_base` | INT8 scoring cost at step 0, and Theorem 1 on GSM8K accuracy: the UAQ checkpoint against the base model | 4 scorings | ~10 min |
| **X7** `x7_s1_vllm024` | Does S1's INT8 identity hold on vLLM 0.24, the version of the RL runs? (S1 used 0.33) | S1's INT8 part at 0.5B | ~10 min (15 if S1's checkpoints must be rebuilt) |
| **X2** `x2_e2dumps` | INT8 scoring cost along E2's INT8-trained run, using its recorded FP32 weights at no training cost | 14 syncs × 2 formats | ~1 h |
| **X3** `x3/rev_x3_*` | **The 4.9-point INT8 gap:** cost of INT8 *scoring* or of INT8 *training*? (both reviewers) | R1 and R3, seeds 1–3, full 435 steps; 15 checkpoints each scored in BF16 and INT8 | ~6 h per run, ~37 h in all |
| **X5** `x5/rev_x5_*` | **R5's early stalls:** do UAQ (R4) and its matched settings (R5) stall equally often? (both reviewers) | R4 and R5 for seeds 6–35, R3 for seeds 6–15; 29 steps each | ~25 min per run, ~29 h in all |
| **X4** `x4_rounding` | Does the s = 1.5 rounding act like seed noise, the hypothesis behind H1? (ChatGPT reviewer) | S2's deterministic loop at s = 1.5, 7 arms of 50 batches | ~2.4 h |
| **X6** `x6_fp8null` | Are S1's FP8 ratios of 0.94–0.98 just BF16 re-rounding noise? | 10 UAQ checkpoints at s = 1.45–1.55 (not 1.5) per model, at 1.5B and 7B, scored on S1's responses | ~2.5 h (2 h without INT8) |

The queue needs **about 72 hours, three days**, if nothing has to be repeated. Started on 20 or 21 November, it ends on 23 or 24 November, which leaves about four days before the lease ends on 28 November to repeat anything that fails.

## Shared pieces
- **`score_gsm8k.py`**: scores a checkpoint on GSM8K's 1,319 test problems, the way VeRL validates S3's runs:
  - the same prompts under the chat template;
  - greedy decoding, up to 512 tokens;
  - VeRL's own GSM8K reward rule;
  - S3's stack (`~/venvs/verl`: VeRL 0.9.1, vLLM 0.24).

  It has two formats:
  - **`bf16`**: vLLM serves the weights in BF16, like the R1 and R2 rollout engine.
  - **`int8`**: every decoder linear weight is quantized by **the rollout's own quantizer** (`qurl_verl.int8_rollout.quantize_rows`, on the GPU, from the FP32 values) into a compressed-tensors INT8 W8A8 checkpoint with S3's config, served by vLLM's CUTLASS INT8 kernel. This is the model that the R3–R6 rollout engine holds after a weight sync.
- **`merge_fp32.py`**: writes a trained actor as a Hugging Face model **in FP32**.
  - VeRL 0.9.1's own merger casts to BF16. BF16 storage moves weights by up to half an INT8 grid step (E2's BF16 lattice), so an INT8 model built from it would not be the one the rollout ran.
  - It reads VeRL's FSDP checkpoints (X3) and E2's recorded syncs (X2), and checks the key set against the base model.
- **`common.sh`**: paths, venv and the `score` helper. The helper skips existing results and accepts a complete JSON when vLLM aborts at exit.

## Before starting: copy and check
From the laptop, in PowerShell, at the repository root:
```powershell
scp -r experiments/rev h100:qurl/
```
Copy the whole folder only this first time. Later fixes go up file by file (`scp experiments/rev/<file> h100:qurl/rev/`), so that the local `copies/` folder is never uploaded. If preflight reports that `s2_train.py` differs from S2's version, copy `experiments/s2/s2_train.py` to `h100:qurl/s2/`.

Then on the server:
```bash
bash ~/qurl/rev/preflight.sh
```
`preflight.sh` changes nothing. It checks:
- the GPU and disk (keep 100 GB free);
- S3's, S2's and S1's code and venvs;
- the GSM8K data and S3's UAQ checkpoint;
- the cached models: 0.5B, Math-1.5B and Math-7B;
- S1's responses and score files for 0.5B, 1.5B and 7B (X7 regenerates the 0.5B responses if they are gone; X6 cannot run without the other two);
- whether E2's dumps and S1's work directories still exist;
- that the frozen analyses match `registration.sha256`.

Paste its output back before starting the queue.

## Start, watch, resume
```bash
nohup bash ~/qurl/rev/queue.sh > ~/qurl/rev/queue.log 2>&1 < /dev/null &
```
```bash
bash ~/qurl/rev/queue.sh --list
```
```bash
bash ~/qurl/rev/queue.sh --progress
```
- **Order:** X1, X7, X2, then X3 (seeds paired: R1 s1, R3 s1, R1 s2, ...), X5 (by seed: R4, R5, R3), X4 and X6.
- **Resuming:** every job is resumable and skipped once done, so the queue can be stopped (kill it and its child) and started again.
- **Failures:** a failed job is logged and skipped, and the queue lists failed jobs at the end. Start it again to retry them.
- **X3 resumes in stages.** An X3 run that already finished training goes straight to merging and scoring. An unfinished one is moved aside and starts over, because model-only checkpoints cannot resume an optimizer.
- **The GPU must be free.** Don't run anything else on it meanwhile. The queue waits for the GPU at start.

## Copy back
From the laptop, in PowerShell. The results are about 0.6 GB, mostly X6's per-token score files; X3's final FP32 weights (`~/qurl/rev/ckpt/*/hf_step435`, 2 GB each) stay on the server.
```powershell
ssh h100 "tar czf rev_results.tgz --exclude='x4_rounding/*/steps' -C qurl/rev results queue.log"
```
The copy leaves out X4's per-step arrays (about 0.8 GB). X4's analysis runs on the server and its report comes back; copy `results/x4_rounding/*/steps` separately only if you want to re-run it.
```powershell
scp h100:rev_results.tgz experiments/rev/
```
Unpack each copy into a new folder, `experiments/rev/copies/<date>/`, never into `results/`. Run the analyses with `python -I`:
```bash
python -I experiments/rev/x3_analysis.py --results <copy>/results --out <copy>/x3_analysis.txt
```
```bash
python -I experiments/rev/x5_analysis.py --results <copy>/results --s3 experiments/s3/results --out <copy>/x5_analysis.txt
```
```bash
python -I experiments/rev/x6_analysis.py --results <copy>/results --out <copy>/x6_analysis.txt
```
```bash
python -I experiments/rev/x7_analysis.py --results <copy>/results --out <copy>/x7_analysis.txt
```
- **X6 and X7** read S1's per-token files from `experiments/h100/results/` by default (`--s1` changes it). Both use only the standard library.
- **X4's verdict** comes back in `results/x4_rounding/x4_report.txt`, written on the server, because the copy leaves out its per-step arrays. To re-run it, copy the arrays as well. It needs numpy, so on the laptop it runs in WSL's `~/venvs/qurl`.
- **Every analysis** also has `--self-test`.

## The jobs in detail

### X1: step 0 (`x1_base.sh`)
Scores the base model and S3's UAQ checkpoint (s = 1.5, FP32) in both formats.
- The base model rarely uses the strict "#### answer" format before training. S3's first training batches had about 2% correct answers, so its scores are low in both formats.
- Its main use is the scorer check at step 0 against VeRL's own step-0 validation in X3, and Theorem 1 at the level of accuracy: by Theorem 1 the UAQ checkpoint's INT8 score equals the base model's.

### X2: E2's recorded weights (`x2_e2dumps.sh`)
E2's run recorded every synced FP32 tensor at 14 syncs (1, 2, 3, 5, 9, 17, 33, 65, 66, 129, 257, 258, 435, 436). Each sync is rebuilt (`merge_fp32.py e2`) and scored in both formats. If the dumps are gone, the job records `SKIPPED`.

### X3: the INT8 gap (`x3_run.sh <R1|R3> <seed>`)
S3's own call (`arm.sh`, as in `uaq_queue.sh`) with these additions:
- `trainer.val_before_train=True`, so VeRL validates at step 0 itself;
- `trainer.save_freq=29` with model-only checkpoints. The actor is kept at every validation step; the critic keeps one;
- `trainer.resume_mode=disable`.

After training, each checkpoint is merged in FP32, scored in both formats, and deleted except step 435. **Peak disk is about 35 GB per run.**

The registered analysis (`x3_analysis.py`) decomposes S3's gap G into two parts:
- **T**, the training effect: both arms' weights scored in BF16;
- **C3**, the INT8 scoring cost on R3's weights.

It also tests the scoring cost on R1's weights (**C1**) and their difference (**D**). Its first step is a **gate**: the scorer must reproduce VeRL's own validation in each run's format.

### X5: early stalls (`x5_run.sh <R3|R4|R5> <seed>`)
S3's own call for 29 steps, validated at step 29.
- **Stall:** validation below 10% at step 29, S3's description. In S3 the four stalled runs were at 0.0–5.5% and every other run at 33.9% or more.
- **Primary test:** the registered analysis (`x5_analysis.py`) runs an exact McNemar test of R4 against R5, over 30 seed pairs.
- **Power:** 0.88 if the rates are 40% and 5%, as S3 suggested with R5 at 2 of 5 and R4 at 0 of 5.

### X4: rounding at s = 1.5 (`x4_run.sh`)
S2's deterministic loop at s = 1.5: `x4_train.py` imports S2's `s2_train.py` unchanged and adds the arms' settings. Each arm uses lr 1e-6 and 50 rollout batches of 8 prompts × 4 samples. The arms, in run order:
- **R4** (UAQ) and **R5** (no UAQ, Theorem 2's matched settings): the question;
- **R5p** and **R4p**: R5 and R4 with every W_QKV and γ element moved by ±1 ulp at step 0, the ulp-noise controls;
- **R4_repeat**: must be bit-identical to R4 (determinism check);
- **R5_seed1** (sampling seed 1, same prompts) and **R5_seed1_full** (seed 1 for sampling and data order): seed-change references.

The registered analysis (`x4_analysis.py`) compares R4 with R5 from step 5 on, on four rule metrics:
- the relative distances of the master weights (W, γ and the rest);
- the FP32 policy's log-probabilities on a fixed probe set.

Its verdicts:
- **ULP-NOISE-LIKE** if R4–R5 never leaves the band [lo/2, 2·hi] spanned by the noise pairs;
- otherwise **LARGER-** or **SMALLER-THAN-ULP-NOISE**, or **MIXED**;
- **INVALID**, for example if R4_repeat is not bit-identical to R4;
- **INCOMPLETE** if arms are missing.

Secondary, against the seed change: **SEED-BOUNDED** or not, and **SATURATED** or not. Disk is about 120 MB per arm.

### X6: a null distribution for S1's FP8 ratios (`x6_run.sh`)
At s = 1.5, S1 found FP8 ratios of 0.94–0.98 at 1.5B and 7B. Their bootstrap intervals exclude 1, but the bootstrap covers the responses, not the BF16 re-rounding of the stored checkpoint.

X6 builds UAQ checkpoints at ten nearby values, s = 1.45, 1.46, …, 1.55 (not 1.5), with S1's `make_uaq_checkpoints.py`. By Theorem 1, each is the same quantized model up to BF16 re-rounding, so the ten are draws of the re-rounding noise.
- **Scoring:** each draw is scored on S1's 512 responses exactly as S1 did (S1's `e0_logprobs.py`, vLLM 0.33 in `~/venvs/qurl`): BF16, FP8 per-tensor, FP8 per-channel and INT8 W8A8. INT8 is optional; `X6_INT8=0` leaves it out.
- **Reproduction check:** X6 also re-scores the original and S1's s = 1.5 checkpoint, to check that S1's environment is reproduced.
- **Disk:** one draw at a time, deleted once scored. The peak is about 25 GB at 7B, and the job stops rather than leave under 10 GB free.
- **Needs:** S1's responses for `math1.5b` and `math7b` in `~/qurl/h100/results/`.

The registered analysis (`x6_analysis.py`) judges S1's ratio at s = 1.5 against the draws, per model, quantizer and metric. The primary cells are FP8 per-tensor and per-channel at 1.5B and 7B, on mean |Δlog p|.
- **Consistent with re-rounding noise** if the ratio lies within the draws' range.
- **Systematic** if it lies outside the range and outside the draws' 95% prediction interval.
- **Borderline** otherwise.
- **No verdict** with fewer than 8 usable draws.

Under the null hypothesis the range part is exact (a 2/11 chance of falling outside), and "systematic" has level 5%. A secondary t-interval of the draws' mean asks whether re-rounding near 1.5 shifts the ratios at all. If S1's environment is not reproduced, the verdict uses X6's own re-score of s = 1.5 instead.

s = 1.5 is special in BF16: W/1.5 is stored exactly for a third of the weights and off by 1/3 ulp for the rest. A "systematic" verdict may point at that, not at a failure of Theorem 1.

### X7: S1's INT8 identity on vLLM 0.24 (`x7_run.sh`)
On vLLM 0.33, S1 found that UAQ at s = 2 gives bit-identical INT8 W8A8 log-probabilities. Every RL run used vLLM 0.24.

X7 repeats S1's INT8 and BF16 scoring of the 0.5B model (S3's) on vLLM 0.24 (`~/venvs/verl`). It uses S1's own scripts, checkpoints and responses; missing pieces are rebuilt with S1's scripts.

The registered readings (`x7_analysis.py`):
- **Identity at s = 2:** reproduced only if INT8 at s = 2 and the original give bit-identical log-probabilities on every response token.
- **s = 1.5:** the ratios with their 95% CIs; "no s-fold reduction" if both CIs lie above 2/3 (mean |Δlog p|) and 4/9 (KL).
- **Kernel:** CUTLASS only if every INT8 run selected `CutlassInt8ScaledMMLinearKernel`.

The job also runs S1's own `analyze_e0.py` and `analyze_s1.py` on its files.
