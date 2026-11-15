# E2: tracking and persistence on real checkpoints (P1, S5)

E2 tests P1's tracking and persistence results (theory note, Section 3.3–3.4) on real weights instead of float64
simulations:
- Assumption U (uniform rounding phase);
- Proposition 3 (bounded lag);
- Proposition 4 (one-step capture: a code flips with probability |d|/Δ);
- Proposition 5 (layer-level tracking, measured in policy space as the change in log μ against the change in log π);
- Proposition 6 (the rounding error is stationary in size but persistent in time);
- Proposition 7 (rollout-quantization schemes, simulated on the real master-weight path).

The P1 experiments report has the design, the predictions and the decision rules.

## Data: one instrumented S3 run
`e2_run.sh` runs arm R3 of S3 (QuRL INT8 W8A8, TIS, ACR; lr 1e-6; seed 1; 435 steps). It uses the S3 queue's own call (`arm.sh R3 ... trainer.test_freq=29 data.seed=1 actor_rollout_ref.rollout.seed=43`) and adds one Hydra override, which swaps in E2's worker extension.

`e2_dump.E2WorkerExtension` subclasses S3's `QuRLWorkerExtension`, which stays unchanged, and taps the synced FP32 master weights. Sync k carries the weights after step k − 1. It saves:
- **Full FP32 state** at syncs 1, 2, 3, 5, 9, 17, 33, 65, 66, 129, 257, 258, 435 and 436. These are θ₀, θ₁, θ₂, θ₄, θ₈, …, θ₂₅₆ and θ₄₃₅, with the one-step pairs (θ₀, θ₁), (θ₁, θ₂), (θ₆₄, θ₆₅), (θ₂₅₆, θ₂₅₇) and (θ₄₃₄, θ₄₃₅). That is about 2 GB each, about 28 GB in total.
- **Row samples at every sync:** 16 fixed rows of each linear weight in layers 0, 6, 12, 18 and 23, about 1.4 GB in total.
- **Layer 0's live INT8 codes** at each full-dump sync, to check that the dumps are what the rollout quantizer saw.

Files:
- `e2_dump.py`: the worker extension.
- `e2_run.sh`: the run driver. It checks disk space and that the GPU is free.
- `e2_check_dumps.py`: checks a dump directory.
- `test_e2_dump.py`: laptop test with S3's real refit code and a fake vLLM model.
- `e2_static.py`: grid statistics of base checkpoints. These are Assumption U, Lemma 1's σ/Δ and ν = η/Δ (the link to 7B).
- `bf16_lattice_sim.py`: exploratory simulation, written after the static results (see Results).
- `bf16_update_sim.py`: exploratory simulation, written after the dynamic results (see Results).

## Analyses (pre-registered 2026-11-15; P1 experiments report §2.6)
The predictions and decision rules were fixed, and the scripts frozen, before any E2 data were analysed. `e2_dynamic.py` and `e2_policy.py` print PASS or FAIL against the rules. `e2_static.py` only measures; the static rules (U1, U2, Lemma 1) are applied to its table. Each script has a `--self-test` with known answers.

| Script | What it measures | SHA-256 |
|---|---|---|
| `e2_static.py` | base checkpoints (0.5B, 1.5B, 7B): U1, U2, Lemma 1, ν = η/Δ | `0db4ecece22b552caa49cecc83dbac474f8a929db7748d8bb7121af194cb40c4` |
| `e2_dynamic.py` | E2 run, weight space: U over training, P3, P4a–c, P6a–c, P7a–b | `9ec9dca1f693962b3f7ffe1a2cdd3a2dc930c469de0ba857656451f7f2556fb3` |
| `e2_policy.py` | E2 run, policy space: P5 and P5′, the slope of Δlog μ on Δlog π over 128 GSM8K probe responses | `af8f517022cda4b496af6425a4558480d6975750a2871a50a0d3695c7c2ec49c` |

Self-tests (laptop, 2026-11-15):
- `e2_dynamic.py`: on Gaussian weights, all 49 rules pass; on weights placed on the grid, U1 and P4 fail (F/P = 0).
- `e2_policy.py`: a gradient step gives slope 1.03 (r 0.89); frozen rollout weights give 0.00.
- `e2_policy.py` also ran end to end on the real 0.5B model, with 2 prompts and synthetic dumps, as a test of the code path only.

## Results
Status (2026-11-15): E2 is complete.
- **Static part:** U1 and Lemma 1 hold; U2 fails in 1 of 21 cells.
- **Weight space:** 87 of 88 checks pass. The one failure is P4c at the first pair, the first update from the BF16 checkpoint.
- **Policy space:** P5′ (not frozen) holds in all 11 checks; P5 (slope within ±10% of 1) fails in 2 of 11.
- **Dump check:** passed at all 14 full-dump syncs.

### Static: base checkpoints (H100, 2026-11-15)
Output: `results/static.txt` and `results/static.json`. There are 21 cells: 7 projection types × 3 checkpoints. The 0.5B numbers were seen beforehand in the laptop test and agree with it to the digits recorded there, so only the 1.5B and 7B cells are fresh tests.

| Rule | 0.5B | 1.5B | 7B | Verdict |
|---|---|---|---|---|
| U1: share of \|φ\| ≥ 0.45 over 0.10, in [0.95, 1.05] | 1.003–1.011 | 0.999–1.010 | 1.002–1.004 | **pass**, 21 of 21 |
| U2: largest of 40 bin deviations at \|c\| ≤ 15, ≤ 0.05 | 0.026–0.045 | 0.025–0.052 | 0.015–0.029 | **fail**: 1.5B up_proj 0.0516; 20 of 21 pass |
| Lemma 1: rows with σ/Δ < 1, below 1% | none | none | none | **pass** |

- **U1 holds** for every projection type of every checkpoint, with the exact ties counted as the rule says. Without them the ratio is 0.985–0.993: the ties sit on the boundary itself, and the band next to it holds slightly less than uniform.
- **U2 fails narrowly, in one cell:** Qwen2.5-Math-1.5B's up_proj, at 0.0516. The other 20 cells lie between 0.015 and 0.046. gate_proj and up_proj are the highest in 0.5B and 1.5B. By the registered reading, a failure of U limits where the propositions apply rather than refuting them.
- **Lemma 1 holds by a wide margin.** No row of any checkpoint has σ/Δ < 1. Median σ/Δ is 28–36. The lowest 1% quantile is 3.9, in 7B's gate_proj, whose outlier rows reach max/rms 32.5 at the 99% quantile.
- **ν = η/Δ at η = 1e-6** (reported without a rule):

  | | 0.5B | 1.5B | 7B |
  |---|---|---|---|
  | median over rows | 1.78e-3 | 8.50e-4 | 1.26e-3 |
  | 10–90% over rows | 1.21e-3 – 2.27e-3 | 6.57e-4 – 1.02e-3 | 9.63e-4 – 1.51e-3 |
  | median by decoder layer | 1.67e-3 – 1.87e-3 | 8.18e-4 – 9.36e-4 | 1.22e-3 – 1.49e-3 |

  At the same learning rate, the S3 runs (0.5B) sit at 1.4× the 7B median and a 1.5B run would sit at 0.67×. Matching the 7B median at 1.5B needs η ≈ 1.5e-6. In 7B, layers 0 and 1 have the highest ν (1.42e-3 and 1.49e-3).
- **Also measured:** exact ties are 0.18% (0.5B), 0.15% (1.5B) and 0.12% (7B) of elements. Bin deviations grow with code size: 0.07–0.13 for codes 16–63, and 0.31–0.42 for codes 64–126, as expected from BF16 storage.

**Exploratory, not registered: the deviations are those of BF16 storage.** `bf16_lattice_sim.py` is a CPU simulation, run on the laptop; its output is in `results/bf16_lattice_sim.txt`. It passes synthetic Gaussian rows through `e2_static.py`'s own statistics, once in FP32 and once rounded to BF16:

| Rows (24M elements per run) | U1 | Exact ties | Deviation, \|c\| ≤ 15 | 16–63 | 64–126 |
|---|---|---|---|---|---|
| Synthetic, FP32 | 0.9995 | 0.0002% | 0.005 (counting noise) | 0.004 | 0.011 |
| Synthetic, BF16; row scales spread over 0.75–3 octaves | 1.006–1.009 | 0.15–0.20% | 0.023–0.032 | 0.073–0.089 | 0.31–0.34 |
| Synthetic, BF16; all rows on one scale | 1.014–1.015 | 0.21–0.23% | 0.070–0.074 | 0.122 | 0.37–0.38 |
| Real checkpoints (21 cells) | 0.999–1.011 | 0.07–0.26% | 0.015–0.052 | 0.070–0.127 | 0.31–0.42 |

- Rounding to BF16 alone produces deviations of the real size in every column. Without it, the phases are uniform up to counting noise.
- Student-t rows (5 degrees of freedom) and rows of length 896 or 3584 give the same picture (\|c\| ≤ 15: 0.024–0.027).
- The simulation does not predict which cell is highest. Real row scales are not log-uniform, and the single failing cell lies between the spread and equal-scale cases.
- The FP32 master weights leave the BF16 lattice as training moves them. So U2 may improve over the E2 run; the dynamic analysis reports U1 and U2 at every full-dump sync. This is an expectation, not a rule.

### The E2 run (H100, 2026-11-15)
`e2_r3_s1` ran all 435 steps in 4.5 h. It saved 14 full dumps and 436 row-sample files (35 GB, kept on the H100). Its folder, `results/e2_r3_s1/`, holds:
- the run's own files (`summary.txt`, `metrics.jsonl`, `env.txt` and the logs of the dump and the INT8 refit);
- the two analyses (`dynamic.*`, `policy.*`);
- the probe set, `policy_probe.pt`. It is not in git, since the repository ignores `*.pt`; copies are kept on the H100 and in the local results folder.

`results/e2_smoke/` has the 3-step smoke test.
- **It behaved like S3's R3 seed 1**, the same call without the dump. Late accuracy (mean of steps 319–435) is 50.6% against 50.2%, and mean k3 is 5.9e-2 against 5.5e-2.
- **Two single-step k3 spikes** (0.55 at step 42, 0.60 at step 92) have no counterpart in S3's run, whose largest step is 0.15.
- **The dump check passes at all 14 full-dump syncs** (`e2_check_dumps.py` on the H100, after the run; its last lines were pasted here, as the output is not in the archive).
  - Every sync has 291 tensors (168 linear, all FP32) with the same names as sync 1.
  - Layer 0's live INT8 Q/K/V codes and scales equal `quantize_rows` of the dump, so the dumps are what the rollout quantized.
  - The row samples equal the matching rows of the full dumps.
- **The weight grids barely move.** At sync 436 no linear weight is more than 4.8e-4 from its start (the check's last line). A row's absmax moves no more than its largest element. So no grid moved by more than 0.7% of the median 0.5B row's absmax, or 1.3% for 99% of rows (from the static ν quantiles).

### Weight space: `e2_dynamic.py` (H100 CPU)
Output: `results/e2_r3_s1/dynamic.txt` and `dynamic.json`. Each full dump has 357.5M elements in the 168 linear weights, without each row's absmax element. The one-step pairs are 1-2, 2-3, 65-66, 257-258 and 435-436, listed in that order below. 87 of 88 checks pass.

| Rule | Measured | Verdict |
|---|---|---|
| U1 at each of the 14 full-dump syncs, in [0.95, 1.05] | 1.0018–1.0082 | **pass**, 14 of 14 |
| U2 at each sync, ≤ 0.05 | 0.036 at sync 1, falling to 0.001 at sync 436 | **pass**, 14 of 14 |
| P3: max \|e_k − e_0\| / max(Δ_k, Δ_0) ≤ 1, at 13 syncs | 0.999997 at most | **pass** (exact) |
| P4a: F/P with all elements, pairs from sync 65 on, in [0.90, 1.10] | 1.011, 1.002, 1.002 | **pass**, 3 of 3 |
| P4a′: F/P without the sync-1 ties, every pair, in [0.90, 1.10] | 0.912, 0.962, 0.987, 0.991, 0.994 | **pass**, 5 of 5 |
| P4b: calibration slope, in [0.90, 1.10] | 0.930, 0.969, 0.987, 0.990, 0.995 | **pass**, 5 of 5 |
| P4c: capture slope, in [0.95, 1.05] | 0.931, 0.969, 0.987, 0.990, 0.994 | **fail** at pair 1-2; 4 of 5 pass |
| P6a: Var(φ_k) / Var(φ_0), in [0.98, 1.02], at 13 syncs | 1.0000–1.0007 | **pass**, 13 of 13 |
| P6b: largest \|Corr − (1 − 6a(1 − a))\| over a-bins, ≤ 0.05 | 0.020 at most (sync 17) | **pass**, 13 of 13 |
| P6c: the same along k, row samples (436 syncs) | 0.0031 | **pass** |
| P7a: error feedback, \|Σe\| ≤ Δ_K / 2 | 1.000000 at most | **pass** (exact) |
| P7b: stochastic rounding, rms partial sum / √(K/6) at K = 436, in [0.90, 1.10] | 1.0004 | **pass** |

- **U holds at every sync, and the bulk grows more uniform during training.**
  - Sync 1 is the BF16 checkpoint in FP32: 0.18% exact ties and U2 0.036, as in the static analysis of 0.5B.
  - One step later the ties are gone (2e-6, FP32 round-off).
  - U2 then falls steadily as the master weights leave the BF16 lattice, as expected from the static results: 0.031, 0.029, 0.022, 0.015, 0.0095, 0.0046, 0.0026 at sync 65 and 0.0010 at sync 436.
  - 12·Var(φ) is 1.0002–1.0008 throughout.
- **P4 fails at the first pair only.** The capture slope there is 0.931, against [0.95, 1.05].
  - U1 held at sync 1 (1.0071), so by the registered reading this counts against Proposition 4 as stated.
  - The flip share (0.912) and the calibration slope (0.930) stay inside their wider bands.
  - The shortfall shrinks as the weights leave the lattice: capture 0.931, 0.969, 0.987, 0.990, 0.994. The exploratory simulation below traces it to BF16 storage.
- **From step 64 on, flips match the prediction within about 1%.** With all elements F/P is 1.002–1.011. Without the sync-1 ties it is 0.987–0.994, because the former ties still sit near the boundary and flip more often than the others.
- **Update sizes** (reported without a rule).
  - P is 3.6e-3 at the first step, 2.2e-3 at the second and 1.1–1.2e-3 from step 64 on. Late in training, one PPO step (four Adam updates) flips about 0.12% of the INT8 codes.
  - The moving grid adds 2–3% to the prediction (P against the fixed-grid prediction).
  - Phase and update are uncorrelated (|corr| ≤ 0.0006).
- **Proposition 6 holds: the rounding error keeps its size and persists.**
  - Var(φ) stays within 0.07% of its start.
  - Corr(φ_k, φ_0) follows 1 − 6a(1 − a) within 0.02 in every a-bin at every full dump. At sync 436 it runs from 0.87 to −0.50 across the bins, as predicted.
  - Along the 436 syncs of the row samples it stays within 0.0031 of the prediction. After 435 steps the rounding error still correlates 0.691 with its start (prediction 0.692).
- **Proposition 7 holds on the real path.** At K = 436 the rms partial sum of the errors, in units of Δ₀, is:
  - 115 for RTN, 0.91 × K/√12, so RTN's errors add up almost as if fully persistent;
  - 8.53 for fresh stochastic rounding, against √(K/6) = 8.52;
  - 0.289 for error feedback, which stays at about 1/√12 = 0.289 for every K.

**Exploratory, not registered: the first update starts from BF16 storage.** `bf16_update_sim.py` is a CPU simulation, run on the laptop; its output is in `results/bf16_update_sim.txt`. It passes synthetic rows through `e2_dynamic.py`'s own statistics, stored either in BF16, as the checkpoint is, or in FP32, with the same random draws otherwise. Each row's ν matches the 0.5B checkpoint. The updates are Adam on pure-noise gradients, four per sync. The resulting P is 6–15% above the run's: 3.8e-3, 2.3e-3 and 1.3e-3 at the three pairs.

| Without the sync-1 ties | F/P′ 1-2 | Capture 1-2 | F/P′ 2-3 | Capture 2-3 | F/P′ 65-66 | Capture 65-66 |
|---|---|---|---|---|---|---|
| E2 run | 0.912 | 0.931 | 0.962 | 0.969 | 0.987 | 0.987 |
| Synthetic, BF16 (2 seeds) | 0.913–0.914 | 0.936–0.938 | 0.961–0.963 | 0.968–0.973 | 0.995 | 0.989–0.996 |
| Synthetic, FP32 | 0.999 | 1.000 | 1.005 | 1.008 | 0.992 | 0.993 |

Counting noise (1/√flips) is 0.001–0.002 in the run. In the simulation it is 0.004 at the first two pairs and 0.006 at pair 65-66.
- **Rounding to BF16 alone reproduces the shortfall of the first two pairs**, to within 0.008. Without it, flips match the prediction within twice the counting noise. With all elements the simulation gives F/P 1.20–1.21 at pair 1-2, against the run's 1.17; it has slightly more ties (0.22% against 0.18%).
- **At pair 65-66 the simulation is too small** to resolve the run's shortfall of about 1%.
- **The shortfall is uneven across codes.** In the simulation, at pair 1-2, F/P′ is 0.95–0.96 for |c| ≤ 15, about 0.85 for 16–63 and 1.14–1.15 for 64–126; with FP32 storage it is 0.995–1.006 in all three. The run's analysis does not split by code.
- **Reading.** One update moves q by about 0.004 grid steps, 14 times less than the width of U1's band (0.05).
  - At that scale BF16 storage is visible. The weights next to a rounding boundary are not spread continuously: some sit exactly on it (the ties, which the rule leaves out), and the rest at discrete distances that are often larger than one update.
  - So fewer non-tie weights lie within one update of a boundary than U assumes. U fails at the scale of the first update even though U1 holds; the registered U1 test is too coarse to see it.
  - Training removes the lattice. By the second update the shortfall is halved, and from step 64 on it is about 1%.

### Policy space: `e2_policy.py` (H100 GPU)
Output: `results/e2_r3_s1/policy.txt` and `policy.json`. The probe set is 128 GSM8K training prompts with one response each from θ₀, 30,155 tokens in all (`policy_probe.pt`). 20 of 22 checks pass.

| Rule | Measured | Verdict |
|---|---|---|
| P5: slope in [0.90, 1.10], cumulative pairs (1, s′) with s′ ≥ 9 | 0.8985–1.0115, median 0.931 | **fail**: 1-258 at 0.8985 ± 0.0011; 9 of 10 pass |
| P5: slope in [0.90, 1.10], pooled one-step pairs | 1.169 ± 0.010 | **fail** |
| P5′: slope > 0.5 (not frozen), the same 11 slopes | 0.8985–1.169 | **pass**, 11 of 11 |

- **The rollout is not frozen.** P5′ holds everywhere, so QuRL's claim (i) gets no support. From sync 1 to sync 436 the INT8 model's log-probabilities follow the trainer's with slope 0.934 (r 0.983); a frozen rollout would give 0.
- **Cumulatively it follows at about 93%.**
  - From sync 33 on, r is 0.95–0.98. Six of the eight slopes lie between 0.915 and 0.934; 1-258 is at 0.8985 and 1-435 at 1.011.
  - So first-order tracking holds within the registered ±10% in 9 of 10 pairs, with a consistent shortfall of about 7% that the theory does not predict. 1-258 misses the band by 0.0015.
- **The standard errors are too small.** They treat the 30,155 tokens as independent, but the tokens of one response are not. The registered rules use the slope alone.
- **One-step changes are dominated by the rollout's own error, more so late in training.**
  - The spread of Δlog μ over the probe tokens is 1.7× that of Δlog π at pair 1-2, 2.6× at 2-3 and 65-66, 4.9× at 257-258 and 8.2× at 435-436.
  - r falls from 0.58 at the first pair to 0.13–0.33 at the last two.
  - The one-step slopes are 1.00, 0.91, 1.03, 0.64 and 2.71.
  - The pooled failure comes from the last pair, which carries 30% of the within-pair covariance and 13% of the variance. Without it the pooled slope would be 0.927; this is a post hoc calculation.
- **The mismatch grows while the weight error does not.**
  - Mean |log μ − log π| over the probe tokens rises 3.5-fold, from 0.057 at sync 1 to 0.20 at sync 435. Meanwhile the weight error keeps its size in grid units (P6a).
  - The probe contexts are fixed, and the weight grids moved by at most 1.3% (see the dump check above). So the rise must come from growing sensitivity or activation error (theory note, Section 3.4). Attributing it is E3, left to P4.
  - The probe responses come from θ₀ and become less likely as training proceeds (mean log π falls from −0.48 to −1.35), which may add to the rise.

## Tests (laptop, WSL VeRL venv, 2026-11-15)
- `test_e2_dump.py`: all checks pass.
  - Full dumps appear at the requested syncs only and equal the synced tensors.
  - The live INT8 codes equal `quantize_rows` of the dump.
  - The row samples are the deterministic subset and equal the synced rows.
  - Non-linear tensors still reach VeRL's load path.
- `e2_check_dumps.py` passes on the test output.
- A `++` Hydra override placed after `pilot.sh`'s `+...worker_extension_cls` replaces it (Hydra 1.3.3): the composed value is `e2_dump.E2WorkerExtension`.
- `e2_static.py --self-test` passes: Gaussian rows are uniform, on-grid rows are flagged, and ν = η/Δ.
- The script also ran on Qwen2.5-0.5B-Instruct as a test only (81 s); the reported numbers come from the H100. That test run showed:
  - phases near the rounding boundary have 1.007× the uniform mass;
  - bins deviate by up to 35% for |code| ≥ 64, from BF16 storage;
  - 0.18% of elements are exact ties;
  - median ν is 1.8e-3.

## H100 commands
Copy the folder:
```bash
scp -r experiments/e2 h100:qurl/
```
Smoke test (3 steps, full dumps at syncs 1–4, about 8 GB, a few minutes), then the check:
```bash
ssh -n h100 "nohup bash qurl/e2/e2_run.sh e2_smoke 3 > qurl/e2/e2_smoke.log 2>&1 < /dev/null &"
```
```bash
ssh h100 "tail -3 qurl/e2/e2_smoke.log; ~/venvs/verl/bin/python qurl/e2/e2_check_dumps.py qurl/e2/results/e2_smoke/dumps"
```
The E2 run (435 steps, about 5 h, about 30 GB):
```bash
ssh -n h100 "nohup bash qurl/e2/e2_run.sh e2_r3_s1 > qurl/e2/e2_r3_s1.log 2>&1 < /dev/null &"
```
```bash
ssh h100 "tail -2 qurl/e2/e2_r3_s1.log; bash qurl/s3/progress.sh e2_r3_s1"
```
Copy the analysis scripts and check their hashes against the table above:
```bash
scp experiments/e2/e2_static.py experiments/e2/e2_dynamic.py experiments/e2/e2_policy.py experiments/e2/e2_check_dumps.py h100:qurl/e2/
```
```bash
ssh h100 "cd qurl/e2 && sha256sum e2_static.py e2_dynamic.py e2_policy.py"
```
Static analysis. It runs on the CPU only, with 8 threads so it can share the machine with the run; 7B takes the longest.
```bash
ssh -n h100 "cd qurl/e2 && mkdir -p results && OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 nohup ~/venvs/verl/bin/python e2_static.py Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-Math-1.5B Qwen/Qwen2.5-Math-7B --out results/static.txt --json results/static.json > results/static.log 2>&1 < /dev/null &"
```
It is done when no `e2_static.py` process is left and `results/static.txt` exists. Copy it back while the run continues:
```bash
scp h100:qurl/e2/results/static.txt h100:qurl/e2/results/static.json h100:qurl/e2/results/static.log experiments/e2/results/
```
After the E2 run has finished: check the dumps, then run the weight-space analysis (CPU) and the policy-space analysis (GPU).
```bash
ssh h100 "cd qurl/e2 && ~/venvs/verl/bin/python e2_check_dumps.py results/e2_r3_s1/dumps | tail -3"
```
```bash
ssh -n h100 "cd qurl/e2 && nohup ~/venvs/verl/bin/python e2_dynamic.py results/e2_r3_s1/dumps --out results/e2_r3_s1/dynamic.txt --json results/e2_r3_s1/dynamic.json > results/e2_r3_s1/dynamic.log 2>&1 < /dev/null &"
```
```bash
ssh -n h100 "cd qurl/e2 && HF_HUB_OFFLINE=1 nohup ~/venvs/verl/bin/python e2_policy.py results/e2_r3_s1/dumps --out results/e2_r3_s1/policy.txt --json results/e2_r3_s1/policy.json > results/e2_r3_s1/policy.log 2>&1 < /dev/null &"
```
Copy back everything except the dumps:
```bash
ssh h100 "cd qurl/e2 && tar czf ~/e2_results.tgz --exclude='dumps' results"
```
```bash
scp h100:e2_results.tgz experiments/e2/
```
