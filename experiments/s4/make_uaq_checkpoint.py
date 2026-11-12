"""S3: write a UAQ checkpoint for arms R2 and R4 (P1 note, section 7): W_{Q,K,V} / s and input-layernorm gain * s.

Usage: python make_uaq_checkpoint.py --out ~/qurl/models/qwen0.5b-instruct-uaq1.5 [--model Qwen/Qwen2.5-0.5B-Instruct]
                                     [--s 1.5]

- The q/k/v biases stay as they are: (s * gamma * n(x)) (W / s)^T + b = (gamma * n(x)) W^T + b.
- The checkpoint is stored in FP32, so W / s is rounded once, in FP32. BF16 storage would re-round it (S1: about 4% of
  INT8 codes change at s = 1.5 that way). VeRL's FSDP actor keeps FP32 master weights, so it starts from exactly these.
- Check: the original and the rescaled model are run in FP32 on the CPU on a fixed GSM8K-style prompt. The logits must
  agree to FP32 rounding, and exactly when s is a power of two (every rescaling is then exact).
- Writes uaq.json next to the weights: s, source model, number of rescaled tensors, and the logit check.
Only the actor is rescaled; the critic and the KL reference keep their usual model paths.
"""
import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
          "How many clips did Natalia sell altogether in April and May? "
          "Let's think step by step and output the final answer after \"####\".")


def rescale(model, s):
    n_w = n_g = 0
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.weight.div_(s)
                n_w += 1
            layer.input_layernorm.weight.mul_(s)
            n_g += 1
    return n_w, n_g


@torch.no_grad()
def logits(model, ids):
    return model(input_ids=ids).logits.double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--s", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"]
    ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    l0 = logits(ref, ids)
    n_w, n_g = rescale(ref, a.s)
    l1 = logits(ref, ids)

    diff = (l1 - l0).abs()
    rel = float(diff.max() / l0.abs().max())
    exact = bool(torch.equal(l0, l1))
    pow2 = a.s > 0 and math.log2(a.s).is_integer()
    print(f"rescaled {n_w} Q/K/V weight tensors by 1/{a.s} and {n_g} input-layernorm gains by {a.s}")
    print(f"FP32 logits on {ids.shape[1]} prompt tokens: max |diff| {float(diff.max()):.3e} "
          f"(relative to max |logit|: {rel:.3e}); bit-identical: {exact}")
    if pow2 and not exact:
        raise SystemExit("s is a power of two, so the logits must be bit-identical")
    if rel > 1e-5:
        raise SystemExit(f"relative logit difference {rel:.2e} exceeds FP32 rounding: wrong layers rescaled?")

    os.makedirs(out, exist_ok=True)
    ref.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    with open(os.path.join(out, "uaq.json"), "w") as f:
        json.dump({"source": a.model, "s": a.s, "qkv_tensors": n_w, "norm_gains": n_g, "dtype": "float32",
                   "logit_max_abs_diff": float(diff.max()), "logit_rel_diff": rel, "bit_identical": exact,
                   "torch": torch.__version__}, f, indent=1)
    print("saved to", out)


if __name__ == "__main__":
    main()
