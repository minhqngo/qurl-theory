"""Compare INT8 weight codes and scales of the W8A8 checkpoints for the UAQ'd Q/K/V layers.

Usage: python compare_int8_codes.py <work_dir>
Theorem 1 predicts: same codes, and scales divided by exactly s, for s = 2. For s = 1.5, BF16 storage of
W/s re-rounds a few percent of codes.
"""
import glob
import os
import sys

import torch
from safetensors.torch import load_file


def load_dir(d):
    t = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        t.update(load_file(f))
    return t


work = sys.argv[1]
orig = load_dir(os.path.join(work, "int8_w8a8_orig"))
qkv = [k for k in orig if any(k.endswith(f"self_attn.{p}_proj.weight") for p in "qkv")]
others = [k for k in orig if k.endswith("_proj.weight") and k not in qkv]
print(f"Q/K/V weight tensors: {len(qkv)}, dtype {orig[qkv[0]].dtype}; other quantized tensors: {len(others)}")
for name, s in [("s2", 2.0), ("s1.5", 1.5)]:
    other = load_dir(os.path.join(work, f"int8_w8a8_{name}"))
    n = changed = 0
    scale_err = 0.0
    for k in qkv:
        a, b = orig[k], other[k]
        changed += int((a != b).sum()); n += a.numel()
        sa, sb = orig[k + "_scale"].float(), other[k + "_scale"].float()
        scale_err = max(scale_err, float(((sa / s - sb).abs() / sa.abs()).max()))
    rest_same = all(torch.equal(orig[k], other[k]) for k in others)
    print(f"{name}: Q/K/V INT8 codes changed {100 * changed / n:.3f}%; max relative error of scale vs scale/s "
          f"{scale_err:.2e}; untouched layers identical: {rest_same}")
