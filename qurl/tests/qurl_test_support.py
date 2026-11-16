"""Shared test helpers: the original experiment code, a tiny Qwen2, stand-in vLLM INT8 layers, independent references.

The originals are loaded read-only by executing their source text into fresh modules: nothing is imported by path, so
no bytecode is written next to them.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
ORIGINALS = REPO / "experiments" / "s4"          # the experiment code this package was refactored from
ORIGINAL_PKG = ORIGINALS / "qurl_verl"


def have_originals() -> bool:
    return (ORIGINAL_PKG / "int8_rollout.py").is_file()


def load_source(path: Path, name: str) -> types.ModuleType:
    """Execute a source file as a fresh module named `name` (not registered in sys.modules)."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


# ---- VeRL stand-ins -------------------------------------------------------------------------------------------------
class StandInColocateExtension:
    """Stand-in for VeRL's vLLMColocateWorkerExtension: records what goes down VeRL's ordinary load path."""

    def _update_weights(self, weights, peft_config, base_sync_done):
        self.loaded = getattr(self, "loaded", []) + [name for name, _ in weights]


def load_original_int8_rollout(monkeypatch) -> types.ModuleType:
    """The original int8_rollout.py on the stand-in base class. It reads its QURL_* variables now, at load time."""
    stub = types.ModuleType("verl.workers.rollout.vllm_rollout.utils")
    stub.vLLMColocateWorkerExtension = StandInColocateExtension
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.utils", stub)
        return load_source(ORIGINAL_PKG / "int8_rollout.py", "original_int8_rollout")


def load_original_acr(monkeypatch) -> types.ModuleType:
    """The original acr.py with VeRL's real agg_loss and masked_mean, without touching VeRL's loss registry."""
    import verl.trainer.ppo.core_algos as core_algos

    stub = types.ModuleType("verl.trainer.ppo.core_algos")
    stub.agg_loss = core_algos.agg_loss
    stub.register_policy_loss = lambda name: (lambda fn: fn)
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "verl.trainer.ppo.core_algos", stub)
        return load_source(ORIGINAL_PKG / "acr.py", "original_acr")


def load_original_uaq() -> types.ModuleType:
    return load_source(ORIGINALS / "make_uaq_checkpoint.py", "original_make_uaq_checkpoint")


# ---- tiny models ----------------------------------------------------------------------------------------------------
def tiny_qwen2(seed: int = 0, layers: int = 2):
    """A tiny random Qwen2 in FP32 (eager attention). The input-layernorm gains and the q/k/v biases are random, so
    that UAQ's gain rescaling matters and a rescaling that forgets the biases' role changes the model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=96, hidden_size=48, intermediate_size=80, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False, attn_implementation="eager")
    m = Qwen2ForCausalLM(cfg).float()
    with torch.no_grad():
        for layer in m.model.layers:
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(0.0, 0.02)
    return m


def write_tiny_checkpoint(path: Path, seed: int = 0) -> None:
    """A tiny random Qwen2 checkpoint stored in BF16, as Qwen2.5 checkpoints are, with a word-level tokenizer and a
    chat template, so that qurl.uaq's command line runs offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from qurl.uaq import PROMPT

    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    for word, _ in pre_tokenizers.Whitespace().pre_tokenize_str(PROMPT):
        vocab.setdefault(word, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tok.save_pretrained(path)
    tiny_qwen2(seed).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)


# ---- independent references -----------------------------------------------------------------------------------------
def np_quantize_rows(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absmax INT8 in numpy float32: scale = absmax * fl32(1/127), codes = round-half-even(W / scale)."""
    x = w.detach().float().cpu().numpy()
    scale = (np.abs(x).max(axis=1, keepdims=True) * np.float32(1.0 / 127.0)).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    codes = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return codes, scale


def bf16_with_ties(shape: tuple[int, int], g: torch.Generator, std: float = 0.02) -> torch.Tensor:
    """A BF16 weight matrix with both kinds of exact rounding-boundary ties a BF16 checkpoint has, where two
    quantizer implementations most easily disagree (round half to even, and the rounding of the scale):

      - every 4th row: two weights at +-absmax / 2 (W / scale = +-63.5, up to the FP32 rounding of the scale);
      - row 1: absmax 254 * 2^-14, so that its scale is exactly 2^-13 and every odd multiple of 2^-14 is a tie.
    """
    rows, cols = shape
    w = (torch.randn(shape, generator=g) * std).to(torch.bfloat16)
    for r in range(0, rows, 4):
        a = w[r].float().abs().max()
        top = int(w[r].float().abs().argmax())
        j0, j1 = [j for j in torch.randperm(cols, generator=g).tolist() if j != top][:2]
        w[r, j0] = a / 2
        w[r, j1] = -a / 2
    a2 = 254 * 2.0 ** -14
    row = (torch.randn(cols, generator=g) * 0.004).clamp(-a2, a2).to(torch.bfloat16)
    row[0] = a2
    w[1] = row
    return w


# ---- stand-in vLLM model with INT8 layers ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dims:
    hidden: int = 64
    q: int = 64        # 4 heads x 16
    kv: int = 32       # 2 KV heads x 16
    inter: int = 96
    vocab: int = 50


PROJ_BLOCK = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
              "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def linear_shapes(dims: Dims) -> dict[str, tuple[int, int]]:
    """Trainer-side [out, in] shapes of a decoder layer's 7 linear weights."""
    return {"q_proj": (dims.q, dims.hidden), "k_proj": (dims.kv, dims.hidden), "v_proj": (dims.kv, dims.hidden),
            "o_proj": (dims.hidden, dims.q), "gate_proj": (dims.inter, dims.hidden),
            "up_proj": (dims.inter, dims.hidden), "down_proj": (dims.hidden, dims.inter)}


def weight_name(layer: int, proj: str) -> str:
    return f"model.layers.{layer}.{PROJ_BLOCK[proj]}.{proj}.weight"


def other_tensors(dims: Dims, layers: int) -> dict[str, torch.Tensor]:
    """Synced tensors the refit hands back to VeRL's load path: embeddings, norms, biases, lm_head."""
    out = {"model.embed_tokens.weight": torch.zeros(dims.vocab, dims.hidden),
           "model.norm.weight": torch.ones(dims.hidden), "lm_head.weight": torch.zeros(dims.vocab, dims.hidden)}
    for i in range(layers):
        out[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(dims.hidden)
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(dims.hidden)
        for p, n in (("q_proj", dims.q), ("k_proj", dims.kv), ("v_proj", dims.kv)):
            out[f"model.layers.{i}.self_attn.{p}.bias"] = torch.zeros(n)
    return out


class StandInKernel:
    """Named like one of vLLM's INT8 GEMM kernels: the refit's info line reports the kernel's class name."""


class StandInInt8Linear(torch.nn.Module):
    """What the refit sees of a vLLM 0.24 compressed-tensors W8A8 INT8 linear layer after loading.

    `weight` is an int8 [in, out] view of [out, in] codes (the CUTLASS and Triton kernels' layout), `weight_scale` a
    float32 [out, 1] row scale, `logical_widths` the output sizes of the fused shards and `scheme.kernel` the GEMM
    kernel. Codes and scales start as garbage, as with VeRL's dummy load format.
    """

    def __init__(self, in_features: int, logical_widths: list[int], g: torch.Generator, device):
        super().__init__()
        out = sum(logical_widths)
        codes = torch.randint(-127, 128, (out, in_features), generator=g, dtype=torch.int8)
        self.weight = torch.nn.Parameter(codes.to(device).t(), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.rand(out, 1, generator=g).to(device), requires_grad=False)
        self.logical_widths = list(logical_widths)
        self.scheme = SimpleNamespace(kernel=StandInKernel())

    def codes(self) -> torch.Tensor:
        """The [out, in] codes."""
        return self.weight.data.t()


def stand_in_model(dims: Dims, layers: int, device, seed: int = 0) -> torch.nn.Module:
    """A stand-in for vLLM's Qwen2 model: model.layers.<i>.{self_attn.qkv_proj, self_attn.o_proj, mlp.gate_up_proj,
    mlp.down_proj} as INT8 layers, plus modules that are not INT8 (embeddings, norms)."""
    g = torch.Generator().manual_seed(seed)
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.embed_tokens = torch.nn.Embedding(dims.vocab, dims.hidden)
    blocks = torch.nn.ModuleList()
    for _ in range(layers):
        b = torch.nn.Module()
        b.self_attn = torch.nn.Module()
        b.self_attn.qkv_proj = StandInInt8Linear(dims.hidden, [dims.q, dims.kv, dims.kv], g, device)
        b.self_attn.o_proj = StandInInt8Linear(dims.q, [dims.hidden], g, device)
        b.mlp = torch.nn.Module()
        b.mlp.gate_up_proj = StandInInt8Linear(dims.hidden, [dims.inter, dims.inter], g, device)
        b.mlp.down_proj = StandInInt8Linear(dims.inter, [dims.hidden], g, device)
        b.input_layernorm = torch.nn.LayerNorm(dims.hidden)
        blocks.append(b)
    root.model.layers = blocks
    root.model.norm = torch.nn.LayerNorm(dims.hidden)
    return root


FUSED_ROWS = {"q_proj": ("self_attn.qkv_proj", 0), "k_proj": ("self_attn.qkv_proj", 1),
              "v_proj": ("self_attn.qkv_proj", 2), "o_proj": ("self_attn.o_proj", 0),
              "gate_proj": ("mlp.gate_up_proj", 0), "up_proj": ("mlp.gate_up_proj", 1),
              "down_proj": ("mlp.down_proj", 0)}


def trajectory(dims: Dims, layers: int, syncs: int = 3, seed: int = 0) -> list[dict[tuple[int, str], torch.Tensor]]:
    """The synced linear weights of `syncs` weight syncs, keyed by (layer, projection): BF16 starting weights with
    planted ties (a BF16 checkpoint), then FP32 weights after updates of 0.002, 0.05 and 0.3 grid steps (RMS)."""
    g = torch.Generator().manual_seed(seed)
    out = [{(i, p): bf16_with_ties(shape, g) for i in range(layers) for p, shape in linear_shapes(dims).items()}]
    for size in (2e-3, 5e-2, 0.3)[:syncs - 1]:
        nxt = {}
        for key, w in out[-1].items():
            _, scale = np_quantize_rows(w)
            nxt[key] = w.float() + torch.randn(w.shape, generator=g) * size * torch.from_numpy(scale)
        out.append(nxt)
    return out


def sync_buckets(weights: dict[tuple[int, str], torch.Tensor], dims: Dims, layers: int, seed: int, device="cpu",
                 buckets: int = 3) -> list[list[tuple[str, torch.Tensor]]]:
    """One weight sync as VeRL sends it: every linear weight and the other tensors, shuffled, in a few buckets."""
    items = [(weight_name(i, p), w) for (i, p), w in weights.items()] + list(other_tensors(dims, layers).items())
    perm = torch.randperm(len(items), generator=torch.Generator().manual_seed(seed)).tolist()
    items = [(items[j][0], items[j][1].to(device)) for j in perm]
    n = len(items)
    return [items[j * n // buckets:(j + 1) * n // buckets] for j in range(buckets)]


def live_rows(model: torch.nn.Module, layer: int, proj: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The live INT8 codes [rows, in] and scales [rows, 1] of one projection inside its (fused) stand-in layer."""
    module, shard = FUSED_ROWS[proj]
    lin = model.get_submodule(f"model.layers.{layer}.{module}")
    r0 = sum(lin.logical_widths[:shard])
    r1 = r0 + lin.logical_widths[shard]
    return lin.codes()[r0:r1], lin.weight_scale.data[r0:r1]
