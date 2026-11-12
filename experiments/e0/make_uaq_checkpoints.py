"""E0 step 1: build UAQ checkpoints (W_{Q,K,V} / s, input_layernorm gain * s) for Qwen2-style models.

Usage: python make_uaq_checkpoints.py <hf_model_id> <out_root> <s> [<s> ...]

The output directory <out_root>/uaq_s<s> is a copy of the original checkpoint with only those tensors
changed, saved in the original dtype (BF16), mirroring what QuRL's UAQ does before RL starts.
Q/K/V biases are left unchanged: (W/s)(sX) + b = WX + b.
Also prints how many INT8 per-channel and FP8 per-tensor weight codes change because the BF16 storage
re-rounds W/s (Theorem 1 says: zero, in exact arithmetic).
"""
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def int8_codes(w):
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    return torch.round(w / scale).clamp(-127, 127)


def fp8_codes(w):
    scale = w.abs().amax() / 448.0
    return (w / scale).to(torch.float8_e4m3fn).view(torch.uint8)


def main():
    model_id, out_root, scales = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]
    src = snapshot_download(model_id)
    files = [f for f in os.listdir(src) if f.endswith(".safetensors")]
    for s in scales:
        dst = os.path.join(out_root, f"uaq_s{s:g}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.safetensors"))
        n_changed = {"int8": 0, "fp8": 0}
        n_total = 0
        for f in files:
            tensors = load_file(os.path.join(src, f))
            out = {}
            for name, t in tensors.items():
                if any(name.endswith(f"self_attn.{p}_proj.weight") for p in "qkv"):
                    new = (t.float() / s).to(t.dtype)
                    w0, w1 = t.float(), new.float()
                    n_changed["int8"] += int((int8_codes(w0) != int8_codes(w1)).sum())
                    n_changed["fp8"] += int((fp8_codes(w0) != fp8_codes(w1)).sum())
                    n_total += t.numel()
                elif name.endswith("input_layernorm.weight"):
                    new = (t.float() * s).to(t.dtype)
                else:
                    new = t
                out[name] = new.contiguous()
            save_file(out, os.path.join(dst, f), metadata={"format": "pt"})
        print(json.dumps({"s": s, "dir": dst, "qkv_weights": n_total,
                          "int8_per_channel_codes_changed_frac": n_changed["int8"] / n_total,
                          "fp8_per_tensor_codes_changed_frac": n_changed["fp8"] / n_total}))


if __name__ == "__main__":
    main()
