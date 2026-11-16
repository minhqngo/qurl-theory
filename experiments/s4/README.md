# S4: QuRL's DAPO experiment at 7B, with mechanism runs at 7B and 1.5B, on a rented 8-GPU node (5 days)

S4 tests UAQ where the QuRL paper made its claim: the DAPO setup with Qwen2.5-Math-7B (paper Sec. 5.1, Table 4). Qwen2.5-Math-1.5B is used only for cheap mechanism runs.
S3 runs the seed-based test of H1 (R4 ≡ R5) on GSM8K PPO at 0.5B, with five seed pairs. S4 adds the paper's scale and tests the mechanism directly.

**Status (2026-11-11): designed and coded; nothing has run on GPUs at scale.**
- The multi-GPU code passed its CPU tests, and the whole loop passed laptop smoke tests (below). On the laptop the flip log reproduces Proposition 4's prediction within a few percent.
- The analysis plan was frozen on 2026-11-11, before any S4 run: `s4_analysis.py`, SHA-256 `b61d9ee9c9112c43eec3d1dc1ea45290688895a9f18b8c14fc0972cdcd06d213` (also in `s4_analysis.sha256`).
- The plan is sized for a 5-day (120-hour) rental.
- The rental's image is plain Ubuntu 24.04 (`ubuntu-24.04-x86_64-standard-uefi`), with no NVIDIA driver.
  - `bootstrap_node.sh` installs the driver (and with it `nvidia-smi`), Fabric Manager and the CUDA 13.0 toolkit, and checks the node.
  - `check_node.py` then tests the node under load.
  - Both are tested on the laptop, except the installation itself, which only a real node can run.

## What S4 tests
- **M1, the mechanism (registered predictions).** The INT8 refit logs, at every weight sync, the share of INT8 codes that changed.
  - Theorem 2 with Proposition 4 predicts that over steps 1–20, R4 (UAQ) and R5 (Theorem 2's per-layer settings, no UAQ) both change about 1.5× as many Q/K/V codes as R3, and the same number elsewhere.
  - R6 (global lr × 1.5) should change 1.5× as many codes everywhere.
  - Each share counts every INT8 code of every layer (tens to hundreds of millions per sync), so one seed suffices.
  - The decisive comparison is R5 against R4. R5 never touches the quantizer, so if it matches R4, UAQ's effect is a learning-rate effect, not better rollout fidelity.
- **M2, the paper's mechanism claim (registered).** Does UAQ reduce the rollout–trainer mismatch (k3), as QuRL says? It is measured over steps 1–20 at both sizes, and over the whole run at 7B when the full R3 run fits. S1 found no reduction at step 0 for 7B INT8.
- **H1 at the paper's scale (descriptive).** One full 7B R4/R5 pair at the paper's exact setting.
  - A single pair supports no seed-based test (S3 has that), so differences are reported with evaluation-noise intervals.
  - The full R3 and R6 runs, when they fit, give UAQ's effect and the paper's Table 4 comparison.

## The setup
The paper states 200 steps, lr 1e-6, 512 prompts × 16 responses, clip 0.2/0.28 and no KL (and s = 1.5, INT8 and FP8 rollouts). VeRL's DAPO recipe for this model ([`dapo/test_dapo_7b_math.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/test_dapo_7b_math.sh)) uses exactly these training values, so S4 takes the paper's unstated settings from it. S4 runs the INT8 rollout only.

| | Value | Source |
|---|---|---|
| Model | Qwen2.5-Math-7B; Qwen2.5-Math-1.5B (S1's model) for mechanism runs | paper |
| Data | DAPO-Math-17k: the published file has 1,791,700 rows but 17,398 distinct problems; 7 with conflicting answers dropped → 17,391 | paper; deduplication ours |
| Algorithm | GRPO, clip 0.2 / 0.28, token-mean loss, no KL, no entropy bonus | paper + recipe |
| Batch | 512 prompts × 16 responses per step; mini-batches of 32 prompts (16 updates per step) | paper + recipe |
| Optimizer | AdamW, lr 1e-6 after 10 warm-up steps, weight decay 0.1, grad clip 1.0, FP32 master weights | paper + recipe |
| Lengths | prompts ≤ 2,048, responses ≤ 8,192 tokens; context raised from 4,096 to 32,768; overlong penalty over the last 4,096 tokens | recipe |
| Not used | dynamic sampling (not in the recipe for this model; the paper does not mention it) | recipe |
| Rollout | vLLM 0.24, one replica per GPU, temperature 1; INT8 W8A8 re-quantized at every sync, token TIS at C = 2, ACR | paper (C from S3) |
| Full runs | 200 steps; validation on AIME 2024 × 32 (temperature 1, top-p 0.7) every 20 steps, with the INT8 rollout engine; checkpoints every 20 steps; final evaluation | recipe; frequency ours |
| Mechanism runs | the same training for 25 steps; no validation, checkpoints or final evaluation | ours |
| Final evaluation | `eval.py` on the final checkpoint in BF16: AIME 2024 Avg@32 and greedy (the paper's metric); MATH-500 Avg@4 and greedy | paper + ours |

- **Arms:** R3 (QuRL INT8), R4 (R3 + UAQ), R5 (R3 with Theorem 2's settings), R6 (R3 with lr × 1.5), as in S3. All seed 1: seed 1 sets the prompt order (`data.seed=1`) and the sampling seed (43).
- **Answers** are scored with VeRL's DAPO rule: the `Answer:` line, falling back to `\boxed{}`. This is the training reward's rule.

## The 5 days
- **Budget:** 120 h × 8 GPUs = 960 GPU-hours.
  - Day 1 takes about 8.5–11.5 h:
    - the node bootstrap, about 0.5 h with one reboot;
    - the environment;
    - the gates, including the two R3 mechanism runs, which are also the timing pilots.
  - That leaves about 110 h for the queue, of which the last 6 h are a reserve.
- **Run lengths are estimates until day 1.** My estimate is 22–36 h per full 7B run on 8 H100s, 3–5 h per 7B mechanism run, and 1–2.5 h per 1.5B mechanism run. The gates measure them.

| When | What | Command |
|---|---|---|
| Before | copy the code; freeze the analysis plan; optionally shake out the config on S3's H100 | see "Before the rental" |
| Day 1 (8.5–11.5 h) | the node: NVIDIA driver (`nvidia-smi`), Fabric Manager, CUDA 13.0 toolkit, storage, checks, one reboot; then the environment, flash-attn build, models, UAQ checkpoints and data; gates G1–G6, including the R3 mechanism runs | `bootstrap_node.sh`, `setup_s4.sh`, `gates.sh` |
| Then (~104 h) | the queue, one run at a time on all 8 GPUs; copy results back daily | `queue.sh` |
| Last 6 h | last evaluation, `s4_analysis.py`, copy everything back | — |

### Run order
| Block | Runs | Purpose |
|---|---|---|
| G4 (day 1) | mechanism R3 at 1.5B and 7B | M1/M2 baselines; gates and timing pilots |
| M | mechanism R4, R5, R6 at 1.5B; R6 at 7B | M1 and early M2 at 1.5B; M1's R6 prediction at 7B |
| A | full 7B R4, R5 | H1 at the paper's scale; their steps 1–20 give M1 at 7B |
| B | full 7B R3 | UAQ's effect; M2 over the whole run; Table 4 |
| C | full 7B R6 | the paper's control |
| D | full 1.5B R4, R5 | H1 at 1.5B |

What fits, from the queue's own dry run (`queue.sh` with `DRY_RUN=1` simulates its clock). Day 1 includes the 0.5 h node bootstrap; adding it changed no scenario's outcome.

| Scenario | Full 7B run | 7B / 1.5B mechanism run | Day 1 | Completes |
|---|---|---|---|---|
| Fast | 22 h | 3.0 / 1.2 h | 8.5 h | M, A, B, C |
| Central | 28 h | 3.9 / 1.8 h | 10.0 h | M, A, B |
| Slow | 36 h | 5.0 / 2.3 h | 11.6 h | M, A |

- **Every scenario** completes M1 and early M2 at both sizes and the 7B R4/R5 pair. The fast and central scenarios add the full 7B R3, so also M2 over the whole run and UAQ's effect.
- **Time cut-off:** with the rental's end time given, the queue starts a run only if it is projected to end 6 h before it. An R4/R5 pair starts only if both fit.
- **Rerun the dry run on day 1** with the measured hours:
  ```bash
  FORCE=1 DRY_RUN=1 END="<rental end>" bash ~/qurl/s4/queue.sh
  ```
- **Contingency, fixed now:**
  - If day 1 projects more than 45 h per full 7B run, the pair would not fit. All S4 runs then use responses ≤ 4,096 tokens with a 2,048-token overlong buffer (`MAX_RESPONSE=4096 OVERLONG=2048`).
  - The two R3 mechanism runs are then repeated at that setting, so every comparison uses one setting.
  - This is decided on timing alone, before any queued run.

## Before the rental
- **Machine:**
  - 8× H100 80 GB, SXM with NVLink preferred. Not Blackwell: S1–S3's INT8 kernel is CUTLASS on SM90, and the laptop's Blackwell GPU has no CUTLASS INT8 build.
  - Image `ubuntu-24.04-x86_64-standard-uefi`: plain Ubuntu 24.04 with no NVIDIA driver. `bootstrap_node.sh` installs it.
  - **Ask for Secure Boot off.** The driver's kernel modules are built on the node and unsigned, so they would not load under Secure Boot. The script stops if Secure Boot is on.
  - sudo, and internet access to Hugging Face, GitHub, PyPI and NVIDIA's apt repository.
  - At least 1 TB free on local NVMe and at least 512 GB of RAM. A separate blank NVMe is fine: the script formats and mounts it when asked (`DATA_DISK`).
- **Copy the code:**
  ```bash
  ssh <rental> "mkdir -p qurl"
  scp -r experiments/s4 <rental>:qurl/s4
  ```
- **The analysis plan is frozen** (2026-11-11, before any S4 run; `gates.sh`'s R3 mechanism runs are already runs of the analysis).
  - `s4_analysis.sha256` holds the hash, and `gates.sh` checks it.
  - Any later change is recorded with both results, as in S3.
  - Its addendum, M1 without the BF16 ties, was registered on 2026-11-16, also before any rental run. It is frozen the same way (`s4_m1_ties.sha256`).
- **Recommended with only 5 days: a shake-out on S3's H100** (`shakeout.sh`, about 1–1.5 h). It runs every day-1 step that one GPU can run, so configuration errors surface before 8 GPUs are paid for. It is not a gate, and nothing from it is reported.
  - The setup, for 1.5B only and without the flash-attn build. On S3's H100 this re-syncs S3's venv to the same lock, so the venv stays as it is.
  - Gates G2 and G3 (at 1.5B, on one GPU).
  - A 2-step 1.5B R3 run at a small batch (32 prompts × 4 responses), otherwise with S4's settings: validation before and after, the INT8 test dump, a checkpoint, the merge and a short evaluation. On the rental, the merge and evaluation first run after a full 7B run.
  - G4's checks that need no 20-step window. The flip log must also hold exactly the first sync and one per step, with no flip at step 1, whose learning rate is 0. S3 and E2 never validated before training, and S4's full runs do: an extra sync there would shift their flip window by one step.
  - The tie log for the addendum: about 0.2% of the codes are ties, none of them flips at step 1, and about half flip at step 2.
  ```bash
  scp -r experiments/s4 h100:qurl/
  ssh -n h100 "nohup bash ~/qurl/s4/shakeout.sh > ~/qurl/s4/shakeout.log 2>&1 < /dev/null &"
  ```
  - Results: `~/qurl/s4/results/shakeout/summary.txt`. Keep them out of the local `experiments/s4/results/`, which goes to the rental with the code: a copied `results/node/check_node.txt` would make `setup_s4.sh` skip the rental's own node check.
  - **First run (2026-11-16): stopped in the setup.**
    - At s = 1.5, the UAQ check's FP32 logits differed by 1.6e-5 relative to the largest logit, on the H100's CPU. `make_uaq_checkpoint.py` had kept S3's limit of 1e-5, which was set from the 0.5B model.
    - Laptop tests show it is rounding: rescaled by 2 (exact), the model gives bit-identical logits, and the same s = 1.5 comparison gives 6.7e-6 on the laptop's CPU.
    - The script now checks the rescaling exactly instead (laptop tests below) and writes the same checkpoint.
  - **Second run (2026-11-16), with the fixed script: 8 of 8 checks passed.** Results in `experiments/s4/shakeout/run2/` (not reported).
    - The UAQ probe gave bit-identical logits on the H100's CPU as well.
    - Validation before training added no sync. The flip log holds syncs 1–3 for 2 steps, so in the full runs syncs 2–21 are the updates of steps 1–20, as the analysis plan assumes.
    - Step 1 (lr 0) changed no code.
    - Step 2 (lr 1e-7) changed 0.0996% of the q/k/v codes. On the tracked tensors that is 8.6 times Proposition 4's prediction (0.0956% against 0.0111%).
    - The excess, 0.084%, is close to half of the checkpoint's exact ties (E2: 0.15%), which flip with probability ½ under any update. By E2's data they keep re-crossing afterwards, in every arm alike. Hence the addendum to the analysis plan (M1 without the ties) and the tie log in `int8_rollout.py`.
    - Found and fixed: `env.txt` ended without a newline, so `run.sh`'s first `code:` line ran into the data record.
    - Seen, not changed: VeRL could not bind its trainer workers to their GPUs' CPUs, because this server has no `libnuma.so` (package `libnuma-dev`). On one GPU this does not matter.
  - **Third run (2026-11-16), with the tie log: 9 of 9 checks passed.** Results in `experiments/s4/shakeout/run3/` (not reported).
    - Ties at sync 1: 0.200% of the q/k/v codes and 0.199% of all codes, as the laptop's CPU test found for this checkpoint. None flipped at step 1, and 0.496 of them flipped at step 2.
    - Step 2's flips were almost all the ties': 0.0991% of the q/k/v codes, against 0.00055% for all the other codes (484 codes).
    - On the tracked tensors, the non-tie flips were 4% of Proposition 4's prediction. BF16 storage leaves a gap around each rounding boundary, and only the ties sit on the boundary. Updates of about 1e-4 grid steps cannot cross the gap.
    - This lattice effect raises the addendum's ratios during the warm-up. Hence the corrected expectation and the simulation in the addendum.
    - The other parts ran as in the second run, and the `env.txt` fix works.

## Day 1: node, setup and gates
```bash
ssh <rental>
tmux new -s s4      # survives a dropped connection (tmux attach -t s4); if missing: sudo apt-get install -y tmux
bash ~/qurl/s4/bootstrap_node.sh 2>&1 | tee -a ~/bootstrap.log
#   "REBOOT NEEDED" (exit 3): sudo reboot, reconnect, run the same command again
#   a storage warning: run it again with DATA_DISK=/dev/<a blank disk it lists> (that disk is formatted)
bash ~/qurl/s4/setup_s4.sh 2>&1 | tee ~/qurl/s4/setup.log
nohup bash ~/qurl/s4/gates.sh > ~/qurl/s4/gates.log 2>&1 < /dev/null &
cat ~/qurl/s4/results/gates/summary.txt
```
- **Node bootstrap** (`bootstrap_node.sh`), about 0.5 h with one reboot. It uses sudo. Every step checks its own state first, so running it again is safe.
  - **Driver:** from NVIDIA's apt repository, with branch 580 pinned (the oldest branch for CUDA 13.0, with long-term support). Headless, open kernel modules. It brings `nvidia-smi` and the persistence daemon.
    - A driver ≥ 580.65.06 that is already there is kept.
    - An older one is replaced only with `UPGRADE_DRIVER=1`.
  - **Fabric Manager**, on an NVSwitch board (HGX), at exactly the driver's version. Without it, CUDA fails with error 802.
  - **CUDA 13.0 toolkit** at `/usr/local/cuda-13.0`, for the flash-attn build.
  - **For the rental:**
    - Automatic upgrades are off and the driver packages are held. An unattended library update under a running job would break it.
    - Persistence mode is on and MIG is off.
    - Every GPU's state is logged every 60 s to `/var/log/qurl-gpu-monitor.csv`.
  - **Storage:** about 1 TB is needed beside `~/qurl`.
    - `DATA_DISK=` formats a blank disk and mounts it at `/data`; `DATA_DIR=` uses a filesystem that is already mounted.
    - `~/qurl`, `~/venvs`, `~/data` and `~/.cache` then move there behind symlinks, so no path changes.
  - **Checks, without PyTorch:**
    - GPU count and model, and the driver;
    - ECC errors and row remapping;
    - NVLink links, PCIe width and the fabric state;
    - CUDA initialization and peer access for every GPU pair;
    - kernel Xid errors;
    - disk, RAM, `/dev/shm` and the network.
  - **Report:** `results/node/bootstrap.txt`. Exit 0 means ready; 1 means warnings; 2 means failed; 3 means a reboot is needed.
- **Node under load** (`check_node.py`, about 2 min). `setup_s4.sh` runs it as soon as the venv exists. All GPUs work at once:
  - memory filled and read back;
  - BF16 and INT8 matmul throughput;
  - pinned host–GPU copies;
  - NCCL all-reduce, all-gather and reduce-scatter, for correctness and bus bandwidth;
  - clocks, temperature and power sampled meanwhile.
  - A failure stops the setup, so a bad node can go back to the provider within the first hour. Report: `results/node/check_node.txt`.
- **Setup**, in user space, about 2–3 h:
  - VeRL v0.9.1's locked stack, as in S3.
  - flash-attn 2.8.3 built from source for SM90 when `nvcc` 13 is present. The lock's prebuilt wheel no longer exists.
  - Both models, the UAQ checkpoints at s = 1.5 and 2 (FP32, checked on the CPU), and the data.
- **Gates**, about 6–8 h. `queue.sh` refuses to start until G2–G4 pass.
  - **G1, machine:** GPU type and count, attention implementation, the node checks' warnings and failures, and whether `s4_analysis.py` and its addendum `s4_m1_ties.py` still match their frozen hashes.
    - Recorded only, with one exception: a `check_node.py` failure stops the gates.
    - A hash mismatch is a loud warning.
  - **G2, packed attention.** VeRL's padding-free forward must give the same log-probs as one sequence at a time (`test_packed_attn.py`).
    - Without flash-attn, SDPA does this correctly only through transformers' block-diagonal mask. The test checks that.
    - Laptop: float32 mean error 8e-6, the same as the padded path.
    - A wrong call (with a KV cache) gives errors around 1, so the test can fail.
  - **G3, R5's exactness on all GPUs** at s = 2, 1.5B and 7B (`r5_check_mp.sh`). R5 and an R4 repeat must be bit-identical to R4; R5 with ordinary clipping must differ. Each GPU hashes its own weight shards.
  - **G4, the R3 mechanism runs** at 1.5B and 7B. These are runs of the analysis plan. Checks:
    - the INT8 codes after the first sync equal an independent quantization (`check_int8_dump.py`);
    - every replica re-quantizes all weights at every sync;
    - the flip log covers syncs 2–21, and its shares are plausible (below 20% per step; about 1% is expected);
    - the tie log, which the addendum needs, is recorded, with a warning if it is missing or implausible.
  - **G5, timing → `hours.txt`, which the queue reads.**
    - Full runs: 200 × the measured median step time, × 1.25 for response growth, + 2 h.
    - Mechanism runs: 1.1 × the measured wall time.
  - **G6, base models:** `eval.py` on both base models, for step 0.

## After day 1: the queue
```bash
END="<rental end, e.g. 2026-11-30 08:00 UTC>" nohup bash ~/qurl/s4/queue.sh > ~/qurl/s4/queue.log 2>&1 < /dev/null &
bash ~/qurl/s4/queue.sh --progress
bash ~/qurl/s4/queue.sh --list
```
- **Per run:** `run.sh` trains and writes `summary.txt`. For a full run, `finish.sh` then merges the final checkpoint, evaluates it (`eval.json`) and deletes the FSDP checkpoints; the merged model stays.
- **Failures:**
  - A failed full run resumes from its last checkpoint; a mechanism run restarts.
  - A second failure marks the run `FAILED` and the queue moves on, so the GPUs are not left idle at night.
  - **A hang counts as a failure** (`watchdog.sh`). A run that writes no file under its results or checkpoint directory for 60 min is stopped: its whole process tree, then Ray, then any process of the venv still on a GPU. The queue's retry then resumes it.
    - The limit rises to 3× the run's longest step so far if that is longer.
    - GPU utilization is not the signal, because a hung NCCL collective spins at 100%.
    - Every quiet phase writes more often than that: steps print metrics, validation and evaluation print vLLM's progress, checkpoints and the merge write files.
    - A stop is recorded in `results/<tag>/WATCHDOG`. `WATCH_MIN=0` turns it off.
  - `STOP_ON_FAIL=1` stops the queue instead.
- **Copy back daily.** Results are small; checkpoints and merged models stay on the node.
  ```bash
  ssh <rental> "cp /var/log/qurl-gpu-monitor.csv qurl/s4/results/node/; tar czf s4_results.tgz --exclude='*.pt' -C qurl/s4 results queue.log gates.log setup.log env.txt"
  scp <rental>:s4_results.tgz experiments/s4/
  ```
- **At the end:** `python s4_analysis.py` writes `results/s4_analysis.txt`, and `python s4_m1_ties.py` writes `results/s4_m1_ties.txt` (the addendum). The merged models are lost with the node unless uploaded.

## Analysis plan (frozen 2026-11-11)
`s4_analysis.py` implements this; `--self-test` checks its arithmetic. It was frozen before any S4 run, with SHA-256 `b61d9ee9c9112c43eec3d1dc1ea45290688895a9f18b8c14fc0972cdcd06d213`.
- **Runs that count:**
  - A full run counts when all 200 steps are logged and `eval.json` exists.
  - A mechanism run counts when its flip window is complete.
  - No run is excluded for its outcome.
- **Flip window:** syncs 2–21, the weight changes of steps 1–20. They come from the first attempt that starts at step 0. A run resumed from a checkpoint logs its starting step, so its later steps never enter the window.
- **M1, each size.**
  - Source per arm: the full run if its window is complete, else the mechanism run. Their first 20 steps have identical settings.
  - Code-flip ratios, pooled over layers and weighted by code counts:
    - q/k/v: R4/R3, R5/R3 and R6/R3 predicted 1.5, "consistent" within [1.35, 1.65].
    - Other projections: R4/R3 and R5/R3 predicted 1, within [0.90, 1.10]; R6/R3 predicted 1.5.
    - R4/R5: predicted 1 for both groups, within [0.95, 1.05].
- **M2, each size:** k3 ratio R4/R3 over steps 1–20 (M1's sources) and over steps 1–200 (full runs, when both exist). "Reduced" below 0.80, "not meaningfully changed" within 0.80–1.25, "increased" above 1.25.
- **Descriptive:**
  - Full-run differences R4 − R5 (H1), R4 − R3 (UAQ's effect) and R6 − R3 (the paper's control) for every endpoint: MATH-500 Avg@4, AIME 2024 Avg@32, the greedy versions, and the training score over steps 1–200 and 1–20.
  - For MATH-500 and AIME, also a paired bootstrap over problems. That is an interval for evaluation noise only: one seed per arm cannot show seed-to-seed variation.
  - The paper's Table 4; the base models.
- **Exploratory, fixed now:** a full run "stalled" if its mean training score over steps 181–200 is below that over steps 1–10.

### Addendum (registered 2026-11-16, before any run on the rental): M1 without the BF16 ties
Exploratory. `s4_m1_ties.py` implements it, imports `s4_analysis.py` unchanged, and is frozen with SHA-256 `90333f9980b3eeed8b4e96c0305213af8edb5fd5fc64abb074dd238bd02a4888` (`s4_m1_ties.sha256`; G1 checks it). The same day, after the third shake-out run and still before any rental run, its expected effect was corrected (below). The first version, `1830cb9c…d657`, had the same rule.
- **Why.** About 0.15% of a BF16 checkpoint's weights sit exactly on an INT8 rounding boundary (E2: 0.15% at 1.5B, 0.12% at 7B).
  - Any update flips about half of them, whatever its size. Afterwards they re-cross the boundary more often than other weights: in E2, 11% of them at the next update and still about 1.6% per update at update 65.
  - The ties are the same weights in every arm, and how often they cross does not depend on the update size. So they add about the same number of flips to every arm, which pulls M1's ratios predicted at 1.5 toward 1.
  - The shake-out measured the first crossing at 1.5B ("Before the rental").
- **Rule.** M1 exactly as `s4_analysis.py` computes it (the same runs, sources, attempt, window, pooling and bands), with each sync's flips counted only outside the run's ties.
  - The ties are the weights within 1e-4 grid steps of a rounding boundary at the attempt's first sync. `int8_rollout.py` logs them (fields `ties` and `flip_ties`; the other fields are unchanged).
  - In R3, R5 and R6 that is the BF16 checkpoint's exact ties and a few weights next to them.
  - In R4 it is the same weights: UAQ's FP32 rescaling moves them by at most about 5e-5 grid steps. At 1.5B it moves them by at most 7.6e-6, and the two sets are identical (laptop tests).
- **Expected.** The ratios predicted at 1 come out about as in M1: R4/R5, and R4/R3 and R5/R3 for the other projections. The ratios predicted at 1.5 come out above M1's, for two reasons: R4/R3, R5/R3 and R6/R3 for q/k/v, and R6/R3 for the other projections.
  - Removing the ties raises a ratio of 1.5 by about a third of the ties' share of R3's flips, which the script reports. The first version said by the whole share.
  - Until training moves the weights off the BF16 lattice, the lattice holds back small updates more than 1.5 times larger ones, which raises the ratio during the warm-up. The third shake-out run showed the effect: step 2's update flipped almost no weight outside the ties.
- **Size, simulated.** `sim_bf16_window.py` (exploratory; output in `sim_bf16_window.txt`) uses synthetic BF16 rows, Adam on noise gradients, 16 updates per step and S4's warm-up. For a true 1.5:
  - M1 reads 1.43 with 1.5B-like rows and 1.46 with 7B-like rows. The variant reads 1.54 at both. With FP32 storage both read 1.50–1.51.
  - The variant's ratio per step starts at 2.4–2.6 (step 2), falls to about 1.55 by step 8, and is 1.52 over steps 11–20.
  - R3 flipped 3.4% (1.5B-like) and 4.9% (7B-like) of its codes over the window. G4 measures the real sum on day 1.
- **Reading.** M1's registered verdicts stand. M1 and the variant bracket the BF16 effects: in the simulation, 1.43–1.46 and 1.54 for a true 1.5, both inside M1's band. A shortfall from 1.5 that M1 shows and the variant does not is the ties'.

## Code
| File | What |
|---|---|
| `qurl_verl/r5_optim.py` | R5's optimizer (Theorem 2(b)–(c)), now on any number of GPUs. Each GPU sums its shards' squared norms in float64; the sums are all-gathered and added in rank order, so the clip norm is bit-identical on every GPU and between R4 and R5 at s = 2 |
| `qurl_verl/int8_rollout.py` | The INT8 refit (unchanged quantizer), one vLLM replica per GPU. It refuses tensor parallelism and checks that every sync re-quantizes every weight. Optional per-sync log: code flips and row-scale changes against its own history (all previous codes, in pinned CPU memory: 6.5 GB for 7B), quantization error, the attempt's starting step, and for the q/k/v tensors of three layers Proposition 4's prediction next to the flip share recomputed from the previous weights. One replica writes. Since 2026-11-16 it also logs the ties (the weights within 1e-4 grid steps of a rounding boundary at the run's start) and their flips, for the addendum |
| `qurl_verl/acr.py` | QuRL's TIS + ACR loss, unchanged from S3 |
| `run.sh`, `finish.sh`, `queue.sh`, `watchdog.sh` | one run (full or mechanism; resumable); merge + evaluation; the 5-day queue; its hang watchdog |
| `bootstrap_node.sh`, `check_node.py` | the node: NVIDIA driver (`nvidia-smi`), Fabric Manager, CUDA 13.0 toolkit, storage, checks; then the node under load |
| `setup_s4.sh`, `gates.sh` | day-1 environment; gates G1–G6 |
| `shakeout.sh` | the shake-out on S3's H100 before the rental (not a gate) |
| `prepare_data.py`, `eval.py` | data; final evaluation, one vLLM per GPU, vLLM's native sampler (as in the training rollout) |
| `r5_check_mp.py/.sh`, `check_int8_dump.py`, `test_packed_attn.py` | gates G3, G4, G2 |
| `summarize.py`, `s4_analysis.py`, `s4_analysis.sha256` | per-run summary and time projection; the registered analysis and its frozen hash |
| `s4_m1_ties.py`, `s4_m1_ties.sha256` | the addendum (M1 without the BF16 ties) and its frozen hash |
| `sim_bf16_window.py`, `sim_bf16_window.txt` | exploratory CPU simulation of BF16 storage in M1's window, with S4's warm-up (laptop; imports E2's frozen `e2_dynamic.py`) |
| `test_clip_mp.py`, `laptop_smoke.sh` | laptop tests (below) |
| `make_uaq_checkpoint.py` | S3's, with an exact check of the rescaling in place of S3's fixed tolerance; the checkpoint is byte-identical to the one S3's script writes |

## Laptop tests (not reported)
- **`test_clip_mp.py`** runs R5's clip arithmetic under FSDP2 on 1, 2, 3 (uneven shards), 4 and 8 CPU processes, with a tiny Qwen2 model in float32. It passed in every case:
  - the cross-GPU clip norm matches a float64 reference to float32 rounding, identically on every rank;
  - after 3 AdamW steps with clipping binding, R5 is bit-identical to R4 at s = 2;
  - an R4 repeat is bit-identical to R4;
  - R5 with ordinary clipping differs.
- **`test_packed_attn.py`** with Qwen2.5-0.5B-Instruct on the laptop GPU: see G2 above.
- **`make_uaq_checkpoint.py`** with Qwen2.5-Math-1.5B on the laptop's CPU, with the same torch and transformers as the H100:
  - Rescaled by 2, the model gives bit-identical FP32 logits. Undoing it restores every weight bit for bit.
  - At s = 1.5 the logits differ by 6.7e-6 relative to the largest logit. The H100's CPU gave 1.6e-5.
  - Four wrong rescalings move the logits by 50–90% of the largest logit at s = 1.5: V left out, O instead of V, the post-attention norm instead of the input norm, and no gains. The probe stops all four and leaves the weights as they were.
  - The saved checkpoints at s = 1.5 and 2 equal the original weights rescaled once, bit for bit in every tensor. At s = 1.5 the file is byte-identical to the one S3's script writes.
- **The tie log and `s4_m1_ties.py`** (2026-11-16):
  - **Tie sets, on Qwen2.5-Math-1.5B's weights** (CPU, with `int8_rollout.py`'s own quantizer):
    - 0.16% of the q/k/v weights are exact ties, and 0.20% lie within 1e-4 grid steps of a boundary (other projections: 0.14% and 0.20%).
    - R4's set, from UAQ's FP32 checkpoint, is the same weights. It contains all 140,883 exact ties, which UAQ moves by at most 7.6e-6 grid steps.
    - R4's starting codes differ from R3's only at ties (42,440 weights).
  - **The refit on the laptop GPU:** `_update_weights` on stand-in INT8 layers, with layers 0–1 of the 1.5B model and two FP32 updates of about 1e-3 grid steps.
    - Every logged share (flips, ties, flips of the ties) equals an independent recomputation on the CPU exactly.
    - The first update flipped exactly half of the ties (0.500).
    - A resumed attempt logs no ties.
  - **`s4_m1_ties.py --self-test`:** with planted ties, M1 comes out diluted and the variant recovers the planted ratios exactly. Without ties the variant equals M1, and a run without the tie log gives no verdict.
  - **Gate scripts:** the shake-out's stub test (six cases, one without the tie log) and G4's snippet on synthetic logs gave the expected results.
- **`prepare_data.py`** produced the counts above.
- **`s4_analysis.py`:**
  - `--self-test` passes.
  - Four synthetic result sets with planted answers gave the planted verdicts:
    - predictions holding;
    - R5's flips off by 1.2× and the mismatch reduced (both flagged; a stalled run found);
    - a 7B run whose first attempt crashed before step 20 (the restarted attempt used), and one resumed from step 20 (the resumed attempt ignored);
    - the slow scenario (M1 from mechanism and full runs mixed, M2 over the whole run withheld).
- **`queue.sh` dry runs** produced the scenario table above.
- **`watchdog.sh`**, with a 1-minute limit:
  - A fake run that wrote once, started a child and hung was stopped after 65 s, child included. It returned non-zero and left a `WATCHDOG` record.
  - A run that kept writing for 108 s was left alone and returned its own exit status.
  - The limit came out as 60 min with no metrics, 60 with a 10-min longest step, and 76 with a 25-min one.
- **`bootstrap_node.sh` and `check_node.py`.** The laptop cannot install a driver, so the installation itself waits for the node.
  - **NVIDIA's apt repository**, read directly for branch 580 on Ubuntu 24.04:
    - `nvidia-smi` ships in `libnvidia-compute-580`, which the headless open driver pulls in;
    - Fabric Manager is the unversioned `nvidia-fabricmanager`;
    - `nvidia-driver-pinning-580` pins both to 580.\*;
    - the repository carries its own `dkms`.
  - **Bootstrap, check-only, on the laptop GPU (WSL):** every check ran; CUDA initialized with driver API 13.0.
  - **Bootstrap, dry run** on a pretend HGX board (8 GPUs, 4 NVSwitches, no driver): it planned the expected installations and ended with "reboot needed".
  - **Storage:** a disk in use was refused as `DATA_DISK`. `DATA_DIR` moved the four directories behind symlinks, and a second run changed nothing.
  - **`check_node.py`:**
    - It ran on the laptop GPU.
    - A crashed worker gave exit 2 and named the error.
    - Synthetic 8 × H100 results gave the expected verdicts: a healthy node passes; a slow, throttled GPU and NCCL at PCIe speed give warnings; an NCCL failure fails.
  - **shellcheck** reports no warnings for the bootstrap, setup and gate scripts.
- **`laptop_smoke.sh`** runs the S4 loop on one GPU at tiny scale, with Qwen2.5-0.5B-Instruct on GSM8K. It ran R3, a resume of R3, R4 and R5: training, the flip log, merge and evaluation. It cannot test multi-GPU behaviour; G3 and G4 do that on day 1.
  - **Flip log.** R3 ran at lr 1e-4 (batch 16 × 8, 4 steps), because the laptop's BF16 master weights absorb lr 1e-6 updates.
    - Step 1 changed no code: the warm-up's first learning rate is 0.
    - Later steps changed up to 1.4–3.4% of a tensor's codes.
    - In all 36 tracked tensor-syncs, the logged flip share equals the share recomputed from the previous weights (largest difference 2e-9).
    - Proposition 4's prediction, mean(min(1, |ΔW|/Δ)), matches the observed share within a few percent: 1.700% vs 1.706%, 2.717% vs 2.716%, 3.53% vs 3.43%. The capture law holds on real weights.
  - **INT8 refit:** all 168 weights re-quantized at every sync.
    - R3's and R5's codes after the first sync are identical to an independent quantization (0 of 1,032,192 differ).
    - R4's differ in 4.1%, as expected on the laptop only. The laptop keeps BF16 master weights, which re-round the FP32 UAQ checkpoint (S1 measured ~4% from BF16 re-rounding); the rental keeps FP32 masters.
  - **R5's optimizer** printed Theorem 2's settings: Q/K/V lr 1.5e-6, ε 6.67e-9, decay 0.0667; layer-norm gains lr 6.67e-7, ε 1.5e-8, decay 0.15; clip norm in UAQ coordinates.
  - **Resume:** continued from the step-2 checkpoint, kept the earlier metrics, and started a new flip-log attempt.
  - **Fixed along the way:**
    - The flip log first compared new codes with vLLM's live INT8 weights. VeRL puts vLLM to sleep at level 2 between rollouts, which discards them, so unchanged weights showed 98% "flips". The refit now keeps its own history of all codes, and G4 checks plausibility.
    - A history of every 16th row was biased: for q_proj it always picks the same 4 of the 64 dimensions of every head, and those flipped 2–4× more often than the whole tensor. A random row subset was unbiased but noisy for small tensors, so the history now keeps all codes.
    - `eval.py` failed at startup: vLLM's top-p sampling used FlashInfer, which compiles a kernel and needs `nvcc`. It now uses vLLM's native sampler, as the training rollout does.
    - A resumed run's validation-only record of the checkpointed step replaced that step's training metrics in the summaries. Metrics are now merged per metric.
    - The test dump is now written only by a run's first attempt.
