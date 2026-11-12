# E0: step-0 identity test for UAQ

This tests Theorem 1 of the P1 theory note (`../../latex/p1_uaq_and_update_tracking.tex`) on real vLLM kernels; the design is in Section 2.2 of the P1 experiments report (`../../latex/p1_experiments.tex`). The UAQ'd checkpoints (Q/K/V weights ÷ s, input-norm gain × s) should give the same quantized rollout policy as the original.

**Status (2026-11-09).** Reported E0 results come from the H100 (stage S1, see `../h100/`), which reuses the Python scripts in this folder. The laptop run documented below, whose output is in `results/`, is a superseded preliminary run kept for provenance. It is not used as evidence: SM120 has no CUTLASS INT8 kernel, so INT8 ran on a Triton fallback; the laptop's BF16 path is not run-to-run deterministic; and two vLLM online modes quantized nothing.

## Environment
Everything runs inside WSL (Ubuntu 24.04):
- venv at `~/venvs/qurl`, built on a uv-managed Python 3.12, which ships the `Python.h` headers Triton needs;
- `uv pip install vllm --torch-backend=auto` (installed vllm 0.33.0 and torch 2.14.0+cu132);
- `VLLM_USE_FLASHINFER_SAMPLER=0`, because there is no CUDA toolkit (`nvcc`) in WSL for FlashInfer's compiler.

## Run
```bash
bash run_e0.sh Qwen/Qwen2.5-0.5B-Instruct fp8_per_tensor fp8_per_channel mxfp8   # about 30 min on the laptop GPU
bash repeat_mxfp8.sh                                                           # MXFP8 same-order noise floor
bash run_e0_int8.sh                                                            # INT8 W8A8 (QuRL's scheme)
```

`run_e0_int8.sh` uses a second, CPU-only venv `~/venvs/llmc` with llm-compressor 0.15 to build compressed-tensors INT8 W8A8 checkpoints (`make_int8_w8a8.py`) and checks their stored codes (`compare_int8_codes.py`). It then scores them with vLLM's Triton INT8 kernel, because vLLM's CUTLASS INT8 GEMM does not support SM120 (`VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel`).

## Files
- `make_uaq_checkpoints.py`: builds the s = 1.5 and s = 2 checkpoints (in `~/e0_work`) and counts weight codes changed by BF16 re-rounding.
- `e0_logprobs.py`: `generate` samples the fixed responses; `score` teacher-forces them and records per-token log-probabilities.
- `analyze_e0.py`: writes `results/summary.txt`.
- `results/`: the responses, per-configuration log-probabilities, `checkpoints.jsonl`, and `summary.txt`.
- `results_run.log`, `results_repeat.log`: full vLLM logs. These show which kernel was selected and how many layers were quantized.

This laptop run is superseded by the H100 run (S1); its results are in §3.2 of the P1 experiments report.
