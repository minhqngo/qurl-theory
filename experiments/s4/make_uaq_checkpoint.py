"""Write a UAQ checkpoint (P1 note, section 7): W_{Q,K,V} / s and input-layernorm gain * s. S3's script (arms R2, R4),
with an exact check of the rescaling in place of S3's fixed tolerance, which Qwen2.5-Math-1.5B exceeded on the H100's
CPU by rounding alone. The checkpoint is the same, byte for byte.

Usage: python make_uaq_checkpoint.py --model Qwen/Qwen2.5-Math-1.5B --s 1.5 --out ~/qurl/models/qwen2.5-math-1.5b-uaq1.5

- The q/k/v biases stay as they are: (s * gamma * n(x)) (W / s)^T + b = (gamma * n(x)) W^T + b.
- The checkpoint is stored in FP32, so W / s is rounded once, in FP32. BF16 storage would re-round it (S1: about 4% of
  INT8 codes change at s = 1.5 that way). VeRL's FSDP actor keeps FP32 master weights, so it starts from exactly these.
- Checks, in FP32 on the CPU, on a fixed GSM8K-style prompt:
  1. The rescaling itself, for every s and without a tolerance: rescaled by 2, where every product and quotient is
     exact in floating point, the model must give bit-identical logits. Any rescaling that changes the model's
     function changes them (a wrong tensor, say). The probe is then undone, and the weights must again be
     bit-for-bit the original ones.
  2. The model rescaled by s, against the original. When s is a power of two, the logits must be bit-identical.
     Otherwise they differ only by the FP32 rounding of W / s and gamma * s, which depends on the CPU. That difference
     is recorded in uaq.json. Relative to the largest logit at s = 1.5, it was 4.4e-6 for Qwen2.5-0.5B-Instruct on the
     laptop, and 6.7e-6 (laptop) and 1.6e-5 (H100) for Qwen2.5-Math-1.5B; wrong rescalings gave 0.5-0.9. Above 1e-3
     the script stops: a backstop, not a tolerance, since check 1 tests the rescaling.
  S3's version had only check 2, with a limit of 1e-5 set from the 0.5B model.
- Writes uaq.json next to the weights: s, source model, number of rescaled tensors, and both checks.
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


def prompt_ids(tok):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"]


def uaq_tensors(model):
    """What UAQ rescales in each layer: the Q, K and V weights (divided by s) and the input-layernorm gain (times s)."""
    ws, gs = [], []
    for layer in model.model.layers:
        ws += [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight]
        gs.append(layer.input_layernorm.weight)
    return ws, gs


def rescale(model, s):
    ws, gs = uaq_tensors(model)
    with torch.no_grad():
        for w in ws:
            w.div_(s)
        for g in gs:
            g.mul_(s)
    return len(ws), len(gs)


@torch.no_grad()
def logits(model, ids):
    return model(input_ids=ids).logits.double()


def check(model, ids, s):
    """Rescales model by s, with checks 1 and 2 (module docstring); exits if one fails. Returns their record."""
    l0 = logits(model, ids)
    ws, gs = uaq_tensors(model)
    with torch.no_grad():
        orig = [t.clone() for t in ws + gs]
    rescale(model, 2)
    probe = bool(torch.equal(logits(model, ids), l0))
    rescale(model, 0.5)
    undone = all(torch.equal(t, o) for t, o in zip(ws + gs, orig))
    del orig
    print(f"probe, rescaled by 2: FP32 logits bit-identical: {probe}; undone, weights bit-identical: {undone}")
    if not probe:
        raise SystemExit("rescaling by 2 changed the logits: wrong tensors rescaled, or in the wrong direction?")
    if not undone:
        raise SystemExit("undoing the probe did not restore the weights bit for bit")

    n_w, n_g = rescale(model, s)
    l1 = logits(model, ids)
    diff = (l1 - l0).abs()
    rel = float(diff.max() / l0.abs().max())
    exact = bool(torch.equal(l0, l1))
    pow2 = s > 0 and math.log2(s).is_integer()
    print(f"rescaled {n_w} Q/K/V weight tensors by 1/{s} and {n_g} input-layernorm gains by {s}")
    print(f"FP32 logits on {ids.shape[1]} prompt tokens: max |diff| {float(diff.max()):.3e} "
          f"(relative to max |logit|: {rel:.3e}); bit-identical: {exact}")
    if pow2 and not exact:
        raise SystemExit("s is a power of two, so the logits must be bit-identical")
    if not (math.isfinite(rel) and rel <= 1e-3):
        raise SystemExit(f"relative logit difference {rel:.2e}: far beyond FP32 rounding, although the rescaling "
                         "passed the exact probe")
    return {"qkv_tensors": n_w, "norm_gains": n_g, "probe_s2_bit_identical": probe,
            "logit_max_abs_diff": float(diff.max()), "logit_rel_diff": rel, "bit_identical": exact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--s", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = prompt_ids(tok)
    ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    rec = check(ref, ids, a.s)

    os.makedirs(out, exist_ok=True)
    ref.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    with open(os.path.join(out, "uaq.json"), "w") as f:
        json.dump({"source": a.model, "s": a.s, **rec, "dtype": "float32", "torch": torch.__version__}, f, indent=1)
    print("saved to", out)


if __name__ == "__main__":
    main()
