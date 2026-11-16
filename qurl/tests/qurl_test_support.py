"""Shared test helpers: the original experiment code, a tiny Qwen2 and its tokenizer, independent references.

The originals are loaded read-only by executing their source text into fresh modules: nothing is imported by path, so
no bytecode is written next to them. They were written for VeRL; the few VeRL names they import are replaced by
minimal stand-ins, so the comparisons need neither VeRL nor vLLM.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

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


# ---- stand-ins for the VeRL names the originals import --------------------------------------------------------------
def _stub_modules(monkeypatch, modules: dict[str, dict]) -> None:
    """Register stub modules (dotted name -> attributes) in sys.modules, each attached to its parent."""
    for name, attrs in modules.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
    for name in modules:
        parent, _, leaf = name.rpartition(".")
        if parent in modules:
            setattr(sys.modules[parent], leaf, sys.modules[name])


def load_original_int8_rollout(monkeypatch) -> types.ModuleType:
    """The original int8_rollout.py (for its quantizer), on a stand-in for VeRL's worker-extension base class."""
    base = type("vLLMColocateWorkerExtension", (), {})
    names = ["verl", "verl.workers", "verl.workers.rollout", "verl.workers.rollout.vllm_rollout"]
    with monkeypatch.context() as m:
        _stub_modules(m, {**{n: {} for n in names},
                          "verl.workers.rollout.vllm_rollout.utils": {"vLLMColocateWorkerExtension": base}})
        return load_source(ORIGINAL_PKG / "int8_rollout.py", "original_int8_rollout")


def masked_sum(values, mask, axis=None):
    """VeRL v0.9.1's verl.utils.torch_functional.masked_sum."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """VeRL v0.9.1's verl.utils.torch_functional.masked_mean."""
    return masked_sum(values, mask, axis) / (mask.sum(axis=axis) + 1e-8)


def token_mean_agg_loss(loss_mat, loss_mask, loss_agg_mode, dp_size=1, batch_num_tokens=None, **_):
    """VeRL v0.9.1's core_algos.agg_loss for loss_agg_mode="token-mean" on one rank."""
    assert loss_agg_mode == "token-mean"
    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum()
    return masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size


def load_original_acr(monkeypatch) -> types.ModuleType:
    """The original acr.py, with VeRL's masked_mean and token-mean agg_loss restated (stand-ins, no VeRL needed)."""
    names = ["verl", "verl.utils", "verl.trainer", "verl.trainer.ppo"]
    with monkeypatch.context() as m:
        _stub_modules(m, {**{n: {} for n in names},
                          "verl.utils.torch_functional": {"masked_mean": masked_mean},
                          "verl.trainer.ppo.core_algos": {"agg_loss": token_mean_agg_loss,
                                                          "register_policy_loss": lambda name: (lambda fn: fn)}})
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


def tiny_tokenizer():
    """A word-level tokenizer with as many tokens as tiny_qwen2's vocabulary (so every sampled id decodes): the special
    tokens <unk>, <pad> (pad) and <eos> (end of text), the words of qurl.uaq's prompt, then filler words; its chat
    template concatenates the messages' contents."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from qurl.uaq import PROMPT

    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    for word, _ in pre_tokenizers.Whitespace().pre_tokenize_str(PROMPT):
        vocab.setdefault(word, len(vocab))
    while len(vocab) < 96:
        vocab[f"w{len(vocab)}"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return tok


def write_tiny_checkpoint(path: Path, seed: int = 0) -> None:
    """A tiny random Qwen2 checkpoint stored in BF16, as Qwen2.5 checkpoints are, with tiny_tokenizer(), so that
    qurl.uaq's command line and the training loop run offline."""
    tiny_tokenizer().save_pretrained(path)
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
