"""Gate G4 (S4): the INT8 refit's codes after the first sync equal an independent quantization of the starting weights.

  python check_int8_dump.py <results>/<tag>/int8_dump --model <checkpoint the run started from: hub id or directory>

S4 version of experiments/s3/check_int8_dump.py (which was tied to Qwen2.5-0.5B-Instruct's cache): any Qwen2-type
checkpoint, single-file or sharded safetensors, BF16 or FP32. Independent of qurl_verl: it re-implements per-row
absmax INT8 with the same arithmetic (scale = absmax * fl32(1/127)).
  1. sync 1 (the starting weights): vLLM's layer-0 fused QKV codes and scales must equal an independent quantization
     of the checkpoint's q/k/v weights, exactly. Exit status 0 only then.
  2. sync 2 (after the first training step): how many codes and row scales the re-quantization changed.
The dump comes from one replica (the first to claim it); every replica receives the same weights.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open


def load(model, names):
    path = model if os.path.isdir(os.path.expanduser(model)) else None
    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors"])
    path = os.path.expanduser(path)
    index = os.path.join(path, "model.safetensors.index.json")
    files = json.load(open(index))["weight_map"] if os.path.exists(index) else {n: "model.safetensors" for n in names}
    out = {}
    for n in names:
        with safe_open(os.path.join(path, files[n]), "pt") as f:
            out[n] = f.get_tensor(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    names = [f"model.layers.0.self_attn.{p}_proj.weight" for p in "qkv"]
    w = load(a.model, names)
    W = torch.cat([w[n] for n in names]).float()              # BF16 or FP32 checkpoint; exact in FP32
    scale_ref = W.abs().amax(dim=1, keepdim=True) * (1.0 / 127.0)
    codes_ref = torch.round(W / scale_ref).clamp(-127, 127).to(torch.int8)
    print(f"checkpoint {a.model}: layer-0 q/k/v stored as {w[names[0]].dtype}")

    s1 = torch.load(os.path.join(a.dump, "sync1_qkv_layer0.pt"))
    same_c = torch.equal(s1["codes"], codes_ref)
    same_s = torch.equal(s1["scale"].float(), scale_ref)
    print("module:", s1["module"], "| logical widths:", s1["logical_widths"], "| codes", tuple(s1["codes"].shape),
          "| scales", tuple(s1["scale"].shape))
    print(f"sync 1 vs independent quantization: codes identical: {same_c} | scales identical: {same_s} | "
          f"mismatched codes: {int((s1['codes'] != codes_ref).sum())} of {codes_ref.numel()}")
    p2 = os.path.join(a.dump, "sync2_qkv_layer0.pt")
    if os.path.exists(p2):
        s2 = torch.load(p2)
        bounds = [0]
        for width in s1["logical_widths"]:
            bounds.append(bounds[-1] + width)
        for name, lo, hi in zip("qkv", bounds[:-1], bounds[1:]):
            dc = (s2["codes"][lo:hi] != s1["codes"][lo:hi]).float().mean().item()
            ds = (s2["scale"][lo:hi] != s1["scale"][lo:hi]).float().mean().item()
            print(f"sync 2 vs sync 1, {name}_proj: {100 * dc:.4f}% of codes changed, {100 * ds:.2f}% of row scales changed")
    print("PASS" if same_c and same_s else "FAIL")
    raise SystemExit(0 if same_c and same_s else 1)


if __name__ == "__main__":
    main()
