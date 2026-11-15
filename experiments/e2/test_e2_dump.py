"""Laptop test of e2_dump.E2WorkerExtension with S3's real INT8 refit code and a fake vLLM model (no GPU needed).
  python test_e2_dump.py        (from experiments/e2, in the VeRL venv; VeRL's own weight loader is patched out)
Three syncs of two buckets each; full dumps at syncs 1 and 3; row samples of layer 0 at every sync. Checks that the
dumps equal the synced FP32 tensors and that the live INT8 codes equal an independent quantization of them."""
import os
import sys
import tempfile
import types

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = tempfile.mkdtemp(prefix="e2_test_")
os.environ.update(QURL_E2_DIR=OUT, QURL_E2_FULL_SYNCS="1,3", QURL_E2_ROW_LAYERS="0", QURL_E2_ROWS="4")
sys.path[:0] = [HERE, os.path.join(HERE, "..", "s3")]

# VeRL's base extension loads the non-INT8 tensors into vLLM's model; here they are just recorded.
import verl.workers.rollout.vllm_rollout.utils as utils  # noqa: E402

received = []


def _record(self, weights, peft_config, base_sync_done):
    received.extend(name for name, _ in weights)


utils.vLLMColocateWorkerExtension._update_weights = _record

from qurl_verl.int8_rollout import quantize_rows  # noqa: E402
from e2_dump import E2WorkerExtension, row_index  # noqa: E402

H, KV, I, L = 16, 8, 32, 2


class FakeInt8(nn.Module):
    def __init__(self, widths, n_in):
        super().__init__()
        codes = torch.zeros(sum(widths), n_in, dtype=torch.int8)
        self.weight = nn.Parameter(codes.t(), requires_grad=False)       # [in, out] view, as after CUTLASS processing
        self.weight_scale = nn.Parameter(torch.zeros(sum(widths), 1), requires_grad=False)
        self.logical_widths = widths


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = FakeInt8([H, KV, KV], H)
        self.self_attn.o_proj = FakeInt8([H], H)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = FakeInt8([I, I], H)
        self.mlp.down_proj = FakeInt8([H], I)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(L)])


shapes = {"q_proj": (H, H), "k_proj": (KV, H), "v_proj": (KV, H), "o_proj": (H, H),
          "gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
g = torch.Generator().manual_seed(0)


def theta():
    t = {}
    for i in range(L):
        for p, s in shapes.items():
            blk = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            t[f"model.layers.{i}.{blk}.{p}.weight"] = torch.randn(*s, generator=g) * 0.02
        t[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(H)
    t["model.embed_tokens.weight"] = torch.randn(10, H, generator=g)
    t["model.norm.weight"] = torch.ones(H)
    return t


ext = E2WorkerExtension()
ext.model_runner = types.SimpleNamespace(model=Model())
syncs = []
for k in (1, 2, 3):
    th = theta()
    syncs.append(th)
    items = list(th.items())
    ext._update_weights(items[:-1], None, True)          # bucket 0: everything but the final norm
    ext._update_weights(items[-1:], None, True)          # bucket 1: the final norm

ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)


files = sorted(os.listdir(os.path.join(OUT, "full")))
check(files == ["sync0001_b00.pt", "sync0001_b01.pt", "sync0001_live_qkv0.pt",
                "sync0003_b00.pt", "sync0003_b01.pt", "sync0003_live_qkv0.pt"], f"full dumps at syncs 1 and 3 only: {files}")
for k in (1, 3):
    dump = {**torch.load(os.path.join(OUT, "full", f"sync{k:04d}_b00.pt")),
            **torch.load(os.path.join(OUT, "full", f"sync{k:04d}_b01.pt"))}
    th = syncs[k - 1]
    check(set(dump) == set(th) and all(torch.equal(dump[n], th[n]) for n in th), f"sync {k}: full dump equals the synced tensors")
    live = torch.load(os.path.join(OUT, "full", f"sync{k:04d}_live_qkv0.pt"))
    w = torch.cat([th[f"model.layers.0.self_attn.{p}.weight"] for p in ("q_proj", "k_proj", "v_proj")])
    codes, scale = quantize_rows(w)
    check(torch.equal(live["codes"], codes) and torch.equal(live["scale"], scale), f"sync {k}: live INT8 codes and scales equal an offline quantization")
index = torch.load(os.path.join(OUT, "rows", "index.pt"))
check(len(index) == 7 and all(n.startswith("model.layers.0.") for n in index), f"row index covers layer 0's 7 matrices ({len(index)})")
check(all(torch.equal(index[n], row_index(n, shapes[n.split('.')[-2]][0])) for n in index), "row index is the deterministic subset")
for k in (1, 2, 3):
    rows = torch.load(os.path.join(OUT, "rows", f"sync{k:04d}_b00.pt"))
    th = syncs[k - 1]
    check(all(torch.equal(rows[n], th[n][index[n]]) for n in index), f"sync {k}: row samples equal the synced rows")
check(not os.path.exists(os.path.join(OUT, "rows", "sync0001_b01.pt")), "no row file for a bucket without linear weights")
check(received.count("model.norm.weight") == 3, "non-linear tensors still reach VeRL's load path")
print("ALL PASS" if ok else "SOME FAILED", "| output in", OUT)
