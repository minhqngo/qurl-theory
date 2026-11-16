"""UAQ checkpoints: W_{Q,K,V} / s and the input-layernorm gain * s, stored in FP32, with an exact check.

QuRL's Update-Aware Quantization (UAQ; paper, Sec. 4.3) is a one-time weight adjustment before RL training: invariant
scaling WX = (W / s)(sX) of the Q, K and V projections, with the activation scale absorbed into the preceding RMSNorm
gain (input_layernorm), and s = 1.5. The paper chooses s > 1 to shrink the weight quantization error and enlarge the
weight updates relative to it. The q/k/v biases stay as they are: (s * gamma * n(x)) (W / s)^T + b = (gamma * n(x))
W^T + b, so the model's function is unchanged. The theory note (latex/p1_uaq_and_update_tracking.tex, Section 3)
analyses what the rescaling does to the INT8 rollout and to training.

The checkpoint is stored in FP32, so W / s is rounded once, in FP32. BF16 storage would re-round it (about 4% of the
INT8 codes change at s = 1.5 that way). VeRL's FSDP actor keeps FP32 master weights, so it starts from exactly these.

check() runs two checks, in FP32 on the CPU, on a fixed GSM8K-style prompt:
  1. The rescaling itself, for every s and without a tolerance: rescaled by 2, where every product and quotient is
     exact in floating point, the model must give bit-identical logits; any rescaling that changes the model's function
     changes them (a wrong tensor, say). The probe is then undone, and the weights must again be bit-for-bit the
     original ones.
  2. The model rescaled by s, against the original. When s is a power of two, the logits must be bit-identical.
     Otherwise they differ only by the FP32 rounding of W / s and gamma * s, which depends on the CPU (relative to the
     largest logit at s = 1.5: 4e-6 to 2e-5 on Qwen2.5 models; wrong rescalings gave 0.5-0.9). Above 1e-3 the check
     stops: a backstop, not a tolerance, since check 1 tests the rescaling.

Command line: the checkpoint, its tokenizer, and uaq.json (s, the source model, the number of rescaled tensors and both
checks). Only the actor is rescaled; the critic and the KL reference keep their usual model paths.
    python -m qurl.uaq --model Qwen/Qwen2.5-Math-1.5B --s 1.5 --out ~/models/qwen2.5-math-1.5b-uaq1.5
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
          "How many clips did Natalia sell altogether in April and May? "
          "Let's think step by step and output the final answer after \"####\".")


class UAQCheckError(RuntimeError):
    """A UAQ check failed: the rescaling changed the model's function."""


def prompt_ids(tok) -> torch.Tensor:
    """The check prompt's token ids [1, T], through the tokenizer's chat template."""
    return tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"]


def uaq_tensors(model):
    """What UAQ rescales in each layer: the Q, K and V weights (divided by s) and the input-layernorm gain (times s).

    `model` is a Hugging Face decoder (Qwen2, Llama, ...) with model.model.layers. Returns (weights, gains).
    """
    ws, gs = [], []
    for layer in model.model.layers:
        ws += [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight]
        gs.append(layer.input_layernorm.weight)
    return ws, gs


def rescale(model, s: float) -> tuple[int, int]:
    """Apply UAQ with scale s in place: W_{Q,K,V} / s and input-layernorm gain * s. Returns the tensor counts."""
    ws, gs = uaq_tensors(model)
    with torch.no_grad():
        for w in ws:
            w.div_(s)
        for g in gs:
            g.mul_(s)
    return len(ws), len(gs)


@torch.no_grad()
def logits(model, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids).logits.double()


def check(model, ids: torch.Tensor, s: float) -> dict:
    """Rescale `model` by s in place, with checks 1 and 2 (module docstring). Returns their record.

    Raises UAQCheckError if a check fails; a failed probe (check 1) leaves the weights as they were.
    """
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
        raise UAQCheckError("rescaling by 2 changed the logits: wrong tensors rescaled, or in the wrong direction?")
    if not undone:
        raise UAQCheckError("undoing the probe did not restore the weights bit for bit")

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
        raise UAQCheckError("s is a power of two, so the logits must be bit-identical")
    if not (math.isfinite(rel) and rel <= 1e-3):
        raise UAQCheckError(f"relative logit difference {rel:.2e}: far beyond FP32 rounding, although the rescaling "
                            "passed the exact probe")
    return {"qkv_tensors": n_w, "norm_gains": n_g, "probe_s2_bit_identical": probe,
            "logit_max_abs_diff": float(diff.max()), "logit_rel_diff": rel, "bit_identical": exact}


def make_checkpoint(model: str, s: float, out: str) -> dict:
    """Write the UAQ checkpoint of `model` (hub id or directory) at scale s to `out`; returns uaq.json's record."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = os.path.expanduser(out)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model)
    ids = prompt_ids(tok)
    ref = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    rec = check(ref, ids, s)

    os.makedirs(out, exist_ok=True)
    ref.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    info = {"source": model, "s": s, **rec, "dtype": "float32", "torch": torch.__version__}
    with open(os.path.join(out, "uaq.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("saved to", out)
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a UAQ checkpoint (W_QKV / s, input-layernorm gain * s, FP32).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="hub id or directory of the source model")
    ap.add_argument("--s", type=float, default=1.5, help="UAQ scale (QuRL: 1.5)")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)
    try:
        make_checkpoint(a.model, a.s, a.out)
    except UAQCheckError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
