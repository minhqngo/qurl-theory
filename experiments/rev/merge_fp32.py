"""Revision experiments X2-X3: write a trained policy as a Hugging Face model in FP32 (the actor's master weights).

  python merge_fp32.py fsdp <ckpt>/global_step_N/actor <target_dir>
  python merge_fp32.py e2 <e2 dumps>/full <sync> <target_dir> [--base Qwen/Qwen2.5-0.5B-Instruct]

Why not VeRL's merger: verl.model_merger (v0.9.1) casts every shard to BF16 before saving. BF16 storage moves weights by
up to half an INT8 grid step for large codes (the BF16 lattice of E2), so an INT8 model built from it would not be
the one S3's rollout engine runs, which quantizes the FP32 master weights. This script keeps the stored dtype; with
--require-fp32 (used by X2 and X3) it refuses to write anything but FP32. VeRL's actor holds FP32 weights by default
(model_dtype: fp32), and S3's call keeps that default.
  fsdp  VeRL's FSDP checkpoint of the actor: model_world_size_<W>_rank_<r>.pt (DTensor shards, concatenated along
        their shard dimension) plus huggingface/ (config and tokenizer, copied).
  e2    one full dump of E2's run: sync<k>_b<b>.pt files with {name: FP32 tensor}, every tensor VeRL synced into the
        rollout engine (sync k holds the weights after step k-1). Config and tokenizer come from the base model.
In both modes the key set must equal the base architecture's (lm_head is dropped when the embedding is tied), and a
summary of dtypes is printed. Writes <target>/model.safetensors, config.json and the tokenizer files.
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import save_file

HF_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")


def from_fsdp(actor_dir):
    from torch.distributed.tensor import DTensor

    world = json.load(open(os.path.join(actor_dir, "fsdp_config.json")))["world_size"]
    shards = [torch.load(os.path.join(actor_dir, f"model_world_size_{world}_rank_{r}.pt"), map_location="cpu",
                         weights_only=False) for r in range(world)]
    out = {}
    for key in shards[0]:
        parts = [sd[key] for sd in shards]
        if isinstance(parts[0], DTensor):
            local = [p._local_tensor for p in parts]
            placement = parts[0].placements[-1]
            if world == 1 or placement.is_replicate():
                t = local[0]
            elif placement.is_shard():
                t = torch.cat(local, dim=placement.dim)
            else:
                raise SystemExit(f"unsupported placement {placement} for {key}")
        else:
            if world > 1 and not all(torch.equal(parts[0], p) for p in parts[1:]):
                raise SystemExit(f"non-DTensor {key} differs across ranks")
            t = parts[0]
        out[key] = t.detach().clone().contiguous()
    return out, os.path.join(actor_dir, "huggingface")


def from_e2(full_dir, sync, base):
    from huggingface_hub import snapshot_download

    files = sorted(glob.glob(os.path.join(full_dir, f"sync{sync:04d}_b*.pt")))
    if not files:
        raise SystemExit(f"no dump files for sync {sync} in {full_dir}")
    out = {}
    for f in files:
        part = torch.load(f, map_location="cpu", weights_only=True)
        dup = set(part) & set(out)
        if dup:
            raise SystemExit(f"tensors repeated across buckets: {sorted(dup)[:3]}")
        out.update(part)
    base_dir = base if os.path.isdir(base) else snapshot_download(base, local_files_only=True)
    return out, base_dir


def base_keys(hf_dir):
    from safetensors import safe_open

    keys = set()
    for f in glob.glob(os.path.join(hf_dir, "*.safetensors")):
        with safe_open(f, "pt") as g:
            keys |= set(g.keys())
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fsdp", "e2"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="architecture reference (and e2's config)")
    ap.add_argument("--require-fp32", action="store_true", help="fail unless every floating-point tensor is FP32")
    a = ap.parse_args()
    if a.mode == "fsdp":
        src, target = a.args
        sd, cfg_dir = from_fsdp(src)
    else:
        src, sync, target = a.args
        sd, cfg_dir = from_e2(src, int(sync), a.base)
    cfg = json.load(open(os.path.join(cfg_dir, "config.json")))
    if cfg.get("tie_word_embeddings") and "lm_head.weight" in sd:
        if not torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]):
            raise SystemExit("tie_word_embeddings is set but lm_head differs from the embedding")
        del sd["lm_head.weight"]
    from huggingface_hub import snapshot_download
    ref_dir = a.base if os.path.isdir(a.base) else snapshot_download(a.base, local_files_only=True)
    ref = base_keys(ref_dir) - {"lm_head.weight"} if cfg.get("tie_word_embeddings") else base_keys(ref_dir)
    missing, extra = sorted(ref - set(sd)), sorted(set(sd) - ref)
    if missing or extra:
        raise SystemExit(f"key mismatch with {a.base}: missing {missing[:5]} ({len(missing)}), "
                         f"extra {extra[:5]} ({len(extra)})")
    dtypes = {}
    for t in sd.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + t.numel()
    if a.require_fp32:
        bad = sorted(k for k, t in sd.items() if t.is_floating_point() and t.dtype != torch.float32)
        if bad:
            raise SystemExit(f"--require-fp32: {len(bad)} tensors are not FP32 (dtypes {dtypes}; e.g. {bad[:3]}); "
                             "nothing written")
    os.makedirs(target, exist_ok=True)
    save_file(sd, os.path.join(target, "model.safetensors"), metadata={"format": "pt"})
    for f in HF_FILES:
        if os.path.exists(os.path.join(cfg_dir, f)):
            shutil.copy(os.path.join(cfg_dir, f), os.path.join(target, f))
    n = sum(t.numel() for t in sd.values())
    print(f"wrote {target}: {len(sd)} tensors, {n / 1e6:.1f}M parameters, dtypes {dtypes}", flush=True)


if __name__ == "__main__":
    main()
