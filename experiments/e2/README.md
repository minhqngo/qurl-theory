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
Status (2026-11-15): the static analysis is done. The E2 run `e2_r3_s1` is in progress (step 197 of 435 at the last check, no stall); the weight-space and policy-space analyses run after it.

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
