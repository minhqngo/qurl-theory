"""S4 offline evaluation of a trained policy in BF16 with vLLM: the paper's protocol (QuRL Sec. 5.1) plus MATH-500.

  python eval.py --model <hf dir or id> --out eval.json [--data ~/data/s4] [--gpus 8] [--max-tokens 8192]
                 [--context 32768] [--k-aime 32] [--k-math 4] [--limit N] [--step N]

  AIME 2024  Avg@32: 32 samples per problem at temperature 1.0, top-p 0.7 (the paper's DAPO evaluation); and Avg@1
             with greedy decoding
  MATH-500   Avg@4 with the same sampling, and greedy (secondary: 500 problems, so much less evaluation noise)
Prompts are the training prompts' format (prepare_data.py) under the model's chat template, as VeRL's rollout applies
it. Answers are scored with VeRL's DAPO rule (verl.utils.reward_score.math_dapo.compute_score: the "Answer:" line,
falling back to \\boxed{}), which is also the training reward's rule.
The policy is evaluated in BF16 (the trained weights, not their INT8 rollout copy). A UAQ model (R4) is in UAQ
coordinates, the same function. One vLLM engine per GPU in its own process; requests are split round-robin and each
has its own sampling seed. eval.json keeps per-problem accuracies for paired comparisons between runs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.abspath(__file__)
BENCH = {"aime24": ("aime24.parquet", "k_aime"), "math500": ("math500.parquet", "k_math")}


def build_requests(a):
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    reqs, meta = [], {}
    for b, (bench, (fname, kattr)) in enumerate(BENCH.items()):
        if bench not in a.benchmarks:
            continue
        df = pd.read_parquet(os.path.join(os.path.expanduser(a.data), fname))
        if a.limit:
            df = df.iloc[:a.limit]
        meta[bench] = {"n_problems": len(df), "k": getattr(a, kattr)}
        for i, (p, rm) in enumerate(zip(df["prompt"], df["reward_model"])):
            prompt = tok.apply_chat_template([dict(m) for m in p], add_generation_prompt=True, tokenize=False)
            gt = str(rm["ground_truth"])
            reqs.append(dict(id=f"{bench}/{i}/sample", bench=bench, problem=i, mode="sample", prompt=prompt, gt=gt,
                             n=getattr(a, kattr), temperature=1.0, top_p=0.7, seed=100000 * (b + 1) + i))
            reqs.append(dict(id=f"{bench}/{i}/greedy", bench=bench, problem=i, mode="greedy", prompt=prompt, gt=gt,
                             n=1, temperature=0.0, top_p=1.0, seed=0))
    return reqs, meta


def worker(a):
    """One GPU: generate for this shard's requests and write the texts."""
    from vllm import LLM, SamplingParams

    reqs = json.load(open(a.inp))
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, gpu_memory_utilization=0.85, seed=0,
              hf_overrides={"max_position_embeddings": a.context})
    params = [SamplingParams(n=r["n"], temperature=r["temperature"], top_p=r["top_p"], max_tokens=a.max_tokens,
                             seed=r["seed"] if r["temperature"] > 0 else None) for r in reqs]
    outs = llm.generate([r["prompt"] for r in reqs], params)
    res = [{"id": r["id"], "texts": [o.text for o in out.outputs], "lens": [len(o.token_ids) for o in out.outputs],
            "finish": [o.finish_reason for o in out.outputs]} for r, out in zip(reqs, outs)]
    json.dump(res, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--data", default="~/data/s4")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--k-aime", type=int, default=32)
    ap.add_argument("--k-math", type=int, default=4)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCH))
    ap.add_argument("--limit", type=int, default=0, help="first N problems of each benchmark (tests)")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inp", help=argparse.SUPPRESS)
    ap.add_argument("--max-len", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    assert a.out, "--out is required"

    t0 = time.time()
    reqs, meta = build_requests(a)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",")[:a.gpus] if visible else [str(i) for i in range(a.gpus)]
    tmp = a.out + ".work"   # requests, outputs and worker logs; removed after a successful evaluation
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    procs = []
    for g, dev in enumerate(devices):
        shard = reqs[g::len(devices)]
        inp = os.path.join(tmp, f"req{g}.json")
        json.dump(shard, open(inp, "w"))
        cmd = [sys.executable, HERE, "--worker", "--model", a.model, "--inp", inp, "--out", os.path.join(tmp, f"out{g}.json"),
               "--max-tokens", str(a.max_tokens), "--context", str(a.context), "--max-len", str(a.max_prompt + a.max_tokens)]
        log = open(os.path.join(tmp, f"worker{g}.log"), "w")
        # vLLM's native top-k/top-p sampler, as in the training rollout (VeRL asks for processed log-probs, which
        # rules out FlashInfer's sampler); FlashInfer's would also JIT-compile at startup and needs nvcc.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, VLLM_USE_FLASHINFER_SAMPLER="0")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, g))
    failed = []
    for p, log, g in procs:
        if p.wait() != 0:
            failed.append(g)
        log.close()
    if failed:
        for g in failed:
            print(f"--- worker {g} log tail ---\n" + "".join(open(os.path.join(tmp, f"worker{g}.log")).readlines()[-30:]))
        raise SystemExit(f"eval workers failed on GPU slots {failed} (logs in {tmp})")
    texts = {}
    for g in range(len(devices)):
        for r in json.load(open(os.path.join(tmp, f"out{g}.json"))):
            texts[r["id"]] = r

    from verl.utils.reward_score import math_dapo

    out = {"model": a.model, "step": a.step, "max_tokens": a.max_tokens, "context": a.context, "gpus": len(devices),
           "scorer": "verl.utils.reward_score.math_dapo.compute_score", "results": {}}
    for bench, m in meta.items():
        per_s, per_g, lens, trunc = [], [], [], 0
        for i in range(m["n_problems"]):
            rs, rg = texts[f"{bench}/{i}/sample"], texts[f"{bench}/{i}/greedy"]
            gt = next(r["gt"] for r in reqs if r["id"] == f"{bench}/{i}/greedy")
            acc = [bool(math_dapo.compute_score(t, gt)["acc"]) for t in rs["texts"]]
            per_s.append(sum(acc) / len(acc))
            per_g.append(float(math_dapo.compute_score(rg["texts"][0], gt)["acc"]))
            lens += rs["lens"]
            trunc += sum(f == "length" for f in rs["finish"])
        n = len(per_s)
        out["results"][bench] = {
            f"avg@{m['k']}": sum(per_s) / n, "greedy": sum(per_g) / n, "n_problems": n, "k": m["k"],
            "mean_response_tokens": sum(lens) / len(lens), "truncated_share": trunc / len(lens),
            "per_problem_sampled": per_s, "per_problem_greedy": per_g}
        print(f"{bench}: Avg@{m['k']} {100 * sum(per_s) / n:.2f}% | greedy {100 * sum(per_g) / n:.2f}% "
              f"| {n} problems | mean response {sum(lens) / len(lens):.0f} tokens, {100 * trunc / len(lens):.1f}% "
              f"cut at {a.max_tokens}")
    out["time_s"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out} ({out['time_s']:.0f} s)")


if __name__ == "__main__":
    main()
