"""Revision experiments X1-X3: score a policy on GSM8K's test set in BF16 or in INT8 W8A8, as VeRL validates S3's runs.

  python score_gsm8k.py --model <hf dir> --format bf16|int8 --out <json>
                        [--data ~/data/gsm8k/test.parquet] [--max-tokens 512] [--tokenizer <dir or id>]
                        [--work <dir>] [--keep-int8] [--limit N] [--gpu-mem 0.35]

What it reproduces: S3's validation in VeRL 0.9.1 (metric val-core/openai/gsm8k/acc/mean@1).
  prompts   The test parquet's chat messages under the tokenizer's chat template with the generation prompt, as
            VeRL's RLHFDataset renders them. Every GSM8K test prompt fits S3's 320-token limit, so none is
            truncated.
  decoding  Greedy (temperature 0), at most 512 new tokens, one response per prompt, vLLM's native sampler.
  reward    VeRL's default reward dispatch (verl.utils.reward_score.default_compute_score). For data_source
            openai/gsm8k this is the GSM8K rule (strict "#### <answer>" format), which S3's reward manager applies.
Formats:
  bf16  vLLM serves the checkpoint in BF16. This is how S3's BF16 rollout engine (R1, R2) holds the synced FP32 master
        weights.
  int8  Every decoder linear weight is quantized by the rollout's own quantizer,
        qurl_verl.int8_rollout.quantize_rows (per row, scale = absmax * fl32(1/127) and codes = round(W / scale)
        clamped to +-127), from the checkpoint's stored values: FP32 for trained checkpoints (merge_fp32.py), BF16
        for the base model, whose BF16 values VeRL's FP32 master copy holds exactly. It runs on the GPU, as in
        the rollout.
        Embeddings, norms, biases and the tied output head are not quantized; vLLM casts them to BF16, as VeRL's
        load path does.
        The codes and their FP32 scales go into a compressed-tensors INT8 W8A8 checkpoint with S3's config
        (qurl_verl/int8_w8a8_config.json: channel-wise weights, dynamic per-token INT8 activations, lm_head in
        BF16), which vLLM serves with its CUTLASS INT8 kernel on the H100. This is the model that S3's INT8
        rollout engine (R3-R6) holds after a weight sync.
The INT8 checkpoint is written under --work (default: <out>.int8) and deleted afterwards unless --keep-int8. The JSON
is written atomically before vLLM shuts down, so it is complete even if vLLM aborts at interpreter exit.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

S3 = os.path.expanduser(os.environ.get("QURL_S3_DIR", "~/qurl/s3"))
COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")


def load_tensors(model_dir):
    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no *.safetensors in {model_dir}")
    out = {}
    for f in files:
        out.update(load_file(f))
    return out


def write_int8(src, dst):
    """Quantize src's decoder linear weights with the rollout's quantizer into a compressed-tensors INT8 checkpoint."""
    import torch
    from safetensors.torch import save_file

    if S3 not in sys.path:
        sys.path.insert(0, S3)
    from qurl_verl.int8_rollout import _LINEAR, quantization_config, quantize_rows

    os.makedirs(dst, exist_ok=True)
    sd = load_tensors(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    if cfg.get("tie_word_embeddings") and "lm_head.weight" in sd:
        del sd["lm_head.weight"]   # tied to the embedding; vLLM ties it again
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out, n_q, src_dtypes = {}, 0, set()
    for name, t in sd.items():
        if _LINEAR.match(name):
            src_dtypes.add(str(t.dtype))
            codes, scale = quantize_rows(t.to(dev))
            out[name] = codes.cpu().contiguous()
            out[name + "_scale"] = scale.cpu().contiguous()
            n_q += 1
        else:
            out[name] = t.contiguous()
    n_layers = cfg["num_hidden_layers"]
    if n_q != 7 * n_layers:
        raise SystemExit(f"expected {7 * n_layers} decoder linear weights, quantized {n_q}")
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    cfg["quantization_config"] = quantization_config()
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in COPY_FILES[1:]:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return {"n_quantized": n_q, "source_dtypes": sorted(src_dtypes), "device": dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model dir (safetensors + config)")
    ap.add_argument("--format", required=True, choices=["bf16", "int8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="~/data/gsm8k/test.parquet")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tokenizer", default=None, help="default: --model")
    ap.add_argument("--work", default=None, help="where to write the INT8 checkpoint (default <out>.int8)")
    ap.add_argument("--keep-int8", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (tests)")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--save-texts", action="store_true", help="also store every response text (debugging)")
    a = ap.parse_args()
    t0 = time.time()

    import pandas as pd
    from transformers import AutoTokenizer

    model_dir = os.path.expanduser(a.model)
    tok_dir = os.path.expanduser(a.tokenizer or a.model)
    df = pd.read_parquet(os.path.expanduser(a.data))
    if a.limit:
        df = df.iloc[:a.limit]
    tok = AutoTokenizer.from_pretrained(tok_dir)
    prompts = [tok.apply_chat_template([dict(m) for m in p], add_generation_prompt=True, tokenize=False)
               for p in df["prompt"]]
    gts = [str(rm["ground_truth"]) for rm in df["reward_model"]]
    sources = list(df["data_source"]) if "data_source" in df.columns else ["openai/gsm8k"] * len(df)

    serve_dir, int8_info = model_dir, None
    if a.format == "int8":
        serve_dir = a.work or a.out + ".int8"
        shutil.rmtree(serve_dir, ignore_errors=True)
        int8_info = write_int8(model_dir, serve_dir)
        print(f"INT8 checkpoint: {serve_dir} ({int8_info})", flush=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    kw = dict(model=serve_dir, tokenizer=tok_dir, dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem,
              max_model_len=1024)
    if a.format == "int8":
        kw["quantization"] = "compressed-tensors"
    llm = LLM(**kw)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens, n=1))

    from verl.utils.reward_score import default_compute_score
    import verl

    per, lens, trunc = [], [], 0
    for o, gt, src in zip(outs, gts, sources):
        c = o.outputs[0]
        s = default_compute_score(src, c.text, gt)
        if isinstance(s, dict):
            s = s.get("acc", s.get("score"))
        per.append(float(s))
        lens.append(len(c.token_ids))
        trunc += c.finish_reason == "length"
    n = len(per)
    res = {"model": model_dir, "format": a.format, "n_problems": n, "accuracy": sum(per) / n,
           "per_problem": per, "mean_response_tokens": sum(lens) / n, "truncated_share": trunc / n,
           "max_tokens": a.max_tokens, "data": os.path.expanduser(a.data), "int8": int8_info,
           "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                        "transformers": transformers.__version__, "verl": getattr(verl, "__version__", "?")},
           "time_s": time.time() - t0}
    if a.save_texts:
        res["texts"] = [o.outputs[0].text for o in outs]
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, a.out)
    print(f"{a.format} accuracy {100 * res['accuracy']:.2f}% on {n} problems; mean response "
          f"{res['mean_response_tokens']:.0f} tokens, {100 * res['truncated_share']:.1f}% cut at {a.max_tokens}; "
          f"{res['time_s']:.0f} s -> {a.out}", flush=True)
    if a.format == "int8" and not a.keep_int8:
        shutil.rmtree(serve_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
