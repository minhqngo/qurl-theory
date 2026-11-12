# Usage: python check_int8_dump.py results/<tag>/int8_dump   (needs torch + safetensors and the Qwen2.5-0.5B-Instruct HF cache)
# Offline check of the INT8 refit (any INT8 arm's int8_dump/). Independent of qurl_verl: re-implements per-row absmax INT8
# with the same arithmetic (scale = absmax * fl32(1/127)).
#  1. sync 1 (initial weights): vLLM's layer-0 fused QKV codes and scales must equal an independent quantization of
#     the checkpoint's q/k/v weights, exactly.
#  2. sync 2 (after the first PPO step): report how many codes and scales the re-quantization changed.
import glob
import os
import sys

import torch
from safetensors import safe_open

dump = sys.argv[1]
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/"))[0]
with safe_open(os.path.join(snap, "model.safetensors"), "pt") as f:
    parts = [f.get_tensor(f"model.layers.0.self_attn.{p}_proj.weight") for p in "qkv"]
W = torch.cat(parts).float()                                   # checkpoint is BF16; exact in FP32
amax = W.abs().amax(dim=1, keepdim=True)
scale_ref = amax * (1.0 / 127.0)
codes_ref = torch.round(W / scale_ref).clamp(-127, 127).to(torch.int8)

s1 = torch.load(os.path.join(dump, "sync1_qkv_layer0.pt"))
print("module:", s1["module"], "| logical widths:", s1["logical_widths"], "| codes", tuple(s1["codes"].shape),
      s1["codes"].dtype, "| scales", tuple(s1["scale"].shape), s1["scale"].dtype)
print("sync 1 vs independent quantization of the checkpoint: codes identical:",
      torch.equal(s1["codes"], codes_ref), "| scales identical:", torch.equal(s1["scale"].float(), scale_ref),
      "| mismatched codes:", int((s1["codes"] != codes_ref).sum()))

p2 = os.path.join(dump, "sync2_qkv_layer0.pt")
if os.path.exists(p2):
    s2 = torch.load(p2)
    bounds = [0]
    for w in s1["logical_widths"]:
        bounds.append(bounds[-1] + w)
    for name, a, b in zip("qkv", bounds[:-1], bounds[1:]):
        dc = (s2["codes"][a:b] != s1["codes"][a:b]).float().mean().item()
        ds = (s2["scale"][a:b] != s1["scale"][a:b]).float().mean().item()
        print(f"sync 2 vs sync 1, {name}_proj: {100 * dc:.4f}% of codes changed, {100 * ds:.2f}% of row scales changed")
else:
    print("no sync-2 dump (run had a single sync?)")
