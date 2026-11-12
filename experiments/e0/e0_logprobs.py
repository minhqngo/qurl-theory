"""E0 steps 2-3: generate fixed responses once, then score them (teacher forcing) under each configuration.

  python e0_logprobs.py generate <model_dir> <out.json> [--n 512]
  python e0_logprobs.py score <model_dir> <responses.json> <out.json> [--quant none|fp8|<other vLLM method>]
                        [--shuffle] [--seed S]

`score` records log pi(o_t | c_t) of every response token, in the exact order of responses.json.
--shuffle submits the sequences in a different order (and batch composition) to measure the
kernel/batching noise floor. Each run is a separate process so GPU memory is released cleanly.
"""
import argparse
import json
import os
import random
import sys
import urllib.request

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

GSM8K_TEST = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
              "grade_school_math/data/test.jsonl")


def save_json(obj, path):
    """Write atomically (temp file + rename), so a crash never leaves a truncated result behind."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def make_llm(model_dir, quant, seed):
    kwargs = dict(model=model_dir, dtype="bfloat16", max_model_len=1024, seed=seed,
                  gpu_memory_utilization=0.6, enable_prefix_caching=False)
    if quant != "none":
        kwargs["quantization"] = quant
    return LLM(**kwargs)


def generate(args):
    lines = urllib.request.urlopen(GSM8K_TEST).read().decode().splitlines()[: args.n]
    questions = [json.loads(l)["question"] for l in lines]
    llm = make_llm(args.model, "none", 0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step."}],
                                       tokenize=False, add_generation_prompt=True) for q in questions]
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=512, seed=0))
    data = [{"prompt_ids": list(o.prompt_token_ids), "response_ids": list(o.outputs[0].token_ids)}
            for o in outs]
    save_json(data, args.out)
    print(f"saved {len(data)} responses, {sum(len(d['response_ids']) for d in data)} response tokens")


def score(args):
    data = json.load(open(args.responses))
    order = list(range(len(data)))
    if args.shuffle:
        random.Random(123).shuffle(order)
    llm = make_llm(args.model, args.quant, args.seed)
    prompts = [TokensPrompt(prompt_token_ids=data[i]["prompt_ids"] + data[i]["response_ids"]) for i in order]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
    result = [None] * len(data)
    for i, o in zip(order, outs):
        p, ids = len(data[i]["prompt_ids"]), prompts[order.index(i)]["prompt_token_ids"]
        lps = []
        for pos in range(p, len(ids)):
            entry = o.prompt_logprobs[pos]
            lps.append(entry[ids[pos]].logprob)
        result[i] = lps
    quant_cfg = getattr(llm.llm_engine.model_config, "quantization", None)
    save_json({"model": args.model, "quant": args.quant, "vllm_quant": quant_cfg, "shuffle": args.shuffle,
               "logprobs": result}, args.out)
    print(f"scored {len(result)} sequences with quant={args.quant} (vLLM reports {quant_cfg})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("model"); g.add_argument("out"); g.add_argument("--n", type=int, default=512)
    s = sub.add_parser("score")
    s.add_argument("model"); s.add_argument("responses"); s.add_argument("out")
    s.add_argument("--quant", default="none"); s.add_argument("--shuffle", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else score(a)
    # vLLM's engine teardown at interpreter exit can abort ("PyEval_SaveThread ... GIL is released") after
    # the result is already on disk; skip finalization so a finished run exits 0.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
