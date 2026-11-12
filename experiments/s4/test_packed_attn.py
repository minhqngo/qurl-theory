"""Gate G2 (S4): is VeRL's packed ("remove padding") forward correct with this attention implementation?

  python test_packed_attn.py --model <hf id or dir> [--attn sdpa flash_attention_2] [--dtype float32|bfloat16]

With actor_rollout_ref.model.use_remove_padding=True, VeRL concatenates a micro-batch's sequences into one row
[1, total_tokens] and passes only input_ids and position_ids (restarting at 0 for each sequence), no attention mask.
flash-attn's varlen kernel keeps the sequences apart. With SDPA, transformers (5.x) has to detect the packing from
the position ids and build a block-diagonal causal mask; if it does not, tokens attend across sequences.
For each attention implementation this compares, on 4 random sequences of different lengths:
  packed   one row, position ids restarting per sequence, no attention mask (VeRL's path)
  single   each sequence on its own (the reference)
  padded   a right-padded batch with an attention mask (VeRL's use_remove_padding=False path)
It reports the mean and the largest |difference| in token log-probabilities against `single`, and judges on the
mean: cross-sequence attention gives mean differences of order 1, correct packing rounding-level ones. In float32
the criterion is absolute (mean < 1e-3; rounding is ~1e-5), and this is the definitive test. In bfloat16 even the
correct padded path differs by ~1e-1 on average (and by ~1 on single tokens), so the packed path must be within
2x the padded path's mean difference (same dtype and kernel). flash_attention_2 runs in bfloat16 only.
Exit status 0 only if every packed comparison passes.
Laptop result (Qwen2.5-0.5B-Instruct, transformers 5.9): with VeRL's exact call, SDPA packs correctly (float32
max 9e-5, as the padded path); called with the default use_cache=True, transformers skips the packing detection and
tokens attend across sequences (max 7.3).
"""
import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM


def logprobs(logits, ids):
    return torch.log_softmax(logits.float(), dim=-1)[..., :-1, :].gather(-1, ids[..., 1:, None]).squeeze(-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", nargs="+", default=["sdpa"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[37, 211, 64, 150])
    ap.add_argument("--timing-tokens", type=int, default=0, help="also time a packed forward of this many tokens")
    ap.add_argument("--negative-control", action="store_true",
                    help="make the packed call with use_cache=True, which disables the packing detection: must FAIL")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randint(100, 5000, (n,), generator=g) for n in a.lengths]
    ok = True
    for attn in a.attn:
        try:
            model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation=attn).cuda().eval()
        except Exception as e:   # noqa: BLE001  (e.g. flash-attn not installed)
            print(f"{attn}: not available ({type(e).__name__}: {str(e)[:120]})")
            continue
        dev = next(model.parameters()).device
        single = [logprobs(model(input_ids=s[None].to(dev), use_cache=False).logits, s[None].to(dev))[0]
                  for s in seqs]

        # Exactly VeRL's call (FSDPEngine.forward_step with use_remove_padding): no attention mask, use_cache=False.
        # transformers only looks for packed sequences when there is neither a mask nor a KV cache.
        ids = torch.cat(seqs)[None].to(dev)
        pos = torch.cat([torch.arange(len(s)) for s in seqs])[None].to(dev)
        out = model(input_ids=ids, attention_mask=None, position_ids=pos, use_cache=a.negative_control)
        lp = torch.log_softmax(out.logits.float(), dim=-1)[0]
        packed, o = [], 0
        for s in seqs:
            packed.append(lp[o:o + len(s) - 1].gather(-1, ids[0, o + 1:o + len(s), None]).squeeze(-1))
            o += len(s)

        L = max(a.lengths)
        pad = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :len(s)], mask[i, :len(s)] = s, 1
        lpp = logprobs(model(input_ids=pad.to(dev), attention_mask=mask.to(dev), use_cache=False).logits, pad.to(dev))
        padded = [lpp[i, :len(s) - 1] for i, s in enumerate(seqs)]

        def diff(xs):
            d = torch.cat([(p - r).abs() for p, r in zip(xs, single)])
            return float(d.max()), float(d.mean())

        (mx_k, mn_k), (mx_d, mn_d) = diff(packed), diff(padded)
        # float32: absolute (rounding is ~1e-5). bfloat16: relative to the padded path in the same dtype and kernel,
        # whose mean difference is pure rounding (~1e-2 to 1e-1); cross-sequence attention is of order 1.
        good = mn_k < 1e-3 if dtype == torch.float32 else mn_k <= 2 * mn_d + 1e-3
        ok &= good
        print(f"{attn} ({a.dtype}): |log-prob difference| vs single sequences, mean / max: packed {mn_k:.2e} / "
              f"{mx_k:.2e}, padded {mn_d:.2e} / {mx_d:.2e} -> packed path "
              f"{'OK' if good else 'WRONG (cross-sequence attention?)'}")
        if a.timing_tokens:
            n = a.timing_tokens
            chunk = 1200
            ids = torch.randint(100, 5000, (1, n), device=dev)
            pos = (torch.arange(n, device=dev) % chunk)[None]
            for _ in range(2):
                torch.cuda.synchronize()
                t = time.time()
                model(input_ids=ids, attention_mask=None, position_ids=pos, use_cache=False)
                torch.cuda.synchronize()
            print(f"  packed forward of {n} tokens ({chunk}-token sequences): {time.time() - t:.3f} s, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        del model
        torch.cuda.empty_cache()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
