# H100 runs (P1 experiments report)

All reported GPU results in the P1 experiments report (`../../latex/p1_experiments.tex`) come from this server: one H100 NVL 94 GB, SSH alias `h100`. The scripts are run by hand from the project root; results are copied back to `results/<tag>/` and analysed locally.

## One-time setup
```bash
ssh h100 "mkdir -p qurl"
scp -r experiments/e0 experiments/h100 h100:qurl/
ssh h100 "bash qurl/h100/setup.sh"
```
`setup.sh` installs a user-space uv, `~/venvs/qurl` (vLLM on a uv-managed Python 3.12) and `~/venvs/llmc` (llm-compressor). It needs no root, nvcc or docker.

## S1: E0 (step-0 identity test) per model
```bash
ssh h100 "nohup bash qurl/h100/s1_run.sh <hf_model_id> <tag> > qurl/h100/s1_<tag>.log 2>&1 &"
```
| Model | Tag | Role |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `qwen0.5b-instruct` | S3's model (QuRL's GSM8K PPO setup) |
| `Qwen/Qwen2.5-Math-1.5B` | `math1.5b` | S4's model |
| `Qwen/Qwen2.5-Math-7B` | `math7b` | paper scale (QuRL's DAPO setting) |

- **What it runs.** The script builds UAQ checkpoints at s = 1.5 and s = 2, samples 512 GSM8K responses from the BF16 model, and teacher-forces them under BF16, FP8 per-tensor, FP8 per-channel (vLLM online) and INT8 W8A8 (llm-compressor checkpoints). Every quantizer, BF16 included, also gets a shuffled-batch run (the batch-invariance noise floor).
- **Kernels.** INT8 runs on `CutlassInt8ScaledMMLinearKernel`, QuRL's kernel on Hopper; `kernels.txt` records what vLLM selected.
- **Resumable.** Finished score files are skipped and results are written atomically, so re-running the same command continues an interrupted run. vLLM can abort during interpreter teardown after a result is saved; that no longer stops the script.
- **One run at a time.** Each vLLM instance takes 60% of GPU memory, so check that `nvidia-smi` shows no compute process before starting another.

## FP8 determinism repeat
```bash
ssh h100 "nohup bash qurl/h100/repeat_fp8.sh <hf_model_id> <tag> > qurl/h100/repeat_<tag>.log 2>&1 &"
```
Run this after `s1_run.sh` with the same arguments. It re-scores the original and s = 2 checkpoints under both FP8 modes in the same batch order (`score_<mode>_{orig,s2}_repeat.json`), then refreshes `summary.txt`.
- **Repeats identical to the first runs:** vLLM's FP8 path is run-to-run deterministic, so any s = 2 vs. original difference is systematic.
- **Repeats different:** the FP8 path is not deterministic.

`repeat_fp8_kernels.txt` lists each run's kernel choice and torch.compile cache hits and misses.

## Analysis (local)
```bash
scp -r h100:qurl/h100/results experiments/h100/
python -P experiments/h100/analyze_s1.py experiments/h100/results/<tag> | tee experiments/h100/results/<tag>/s1_analysis.txt
```
`summary.txt` (written on the server by `e0/analyze_e0.py`) has the identity checks: shuffled-batch noise floor, same-order repeats if present, and s = 2 and s = 1.5 against the original. `analyze_s1.py` compares each checkpoint's mismatch with the BF16 policy trained from it, with paired sequence-bootstrap CIs. Mean |Δlog p| is the primary metric. The k1 and k3 KL estimates come with an estimator check: the mean of r − 1 should be 0. That check is sensitive to rare tokens with μ ≫ π, so treat the KL values as secondary.

## `results/<tag>/`
`responses.json`, `score_<quant>_<variant>.json`, `checkpoints.jsonl` (codes changed by UAQ), `int8_codes.txt` (stored INT8 codes and scales), `kernels.txt`, `summary.txt`, `s1_analysis.txt`, `run.log`.
