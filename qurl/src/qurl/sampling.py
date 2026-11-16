"""Batched sampling with a KV cache that records the sampling policy's log-probabilities, and the training forward.

QuRL needs, for every sampled token, its log-probability under the behavior policy mu (the quantized actor at the
sampling temperature): generate() returns it with the tokens. Sampling is by inverse CDF in float64 on the CPU, from a
seeded torch.Generator, so a run is reproducible; temperature 0 is greedy decoding. sequence_logprobs() computes the
log-probabilities of the same tokens under the current (full-precision) policy, for the proximal policy pi_old and
for training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Rollout:
    """Prompts (left-padded) and their sampled responses; B = prompts x n rows, P prompt and L response columns."""
    input_ids: torch.Tensor        # [B, P + L] prompt + response
    attention_mask: torch.Tensor   # [B, P + L]
    prompt_len: int                # P
    response: torch.Tensor         # [B, L] sampled tokens; pad after a response's end
    logprobs: torch.Tensor         # [B, L] log-probability of each token under the sampling policy; 0 after the end
    mask: torch.Tensor             # [B, L] bool: the response's tokens, up to and including its end token


def stop_token_ids(tok) -> list[int]:
    """The tokens that end a response: the tokenizer's EOS and pad tokens, and Qwen's chat and text end tokens."""
    vocab = tok.get_vocab()
    ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    ids |= {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab}
    return sorted(ids)


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, n: int = 1, max_new: int, temperature: float = 1.0,
             generator: torch.Generator | None = None, stop_ids: list[int] | None = None) -> Rollout:
    """Sample n responses per prompt (rows grouped by prompt), at most max_new tokens each.

    The tokenizer must pad on the left. Sampling draws its uniforms from `generator` (a CPU generator).
    """
    if tok.padding_side != "left":
        raise ValueError("generate needs a tokenizer that pads on the left (tok.padding_side = 'left')")
    device = next(model.parameters()).device
    stop = torch.tensor(stop_ids if stop_ids is not None else stop_token_ids(tok))
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc.input_ids.repeat_interleave(n, 0)
    prompt_mask = enc.attention_mask.repeat_interleave(n, 0)
    batch, prompt_len = ids.shape
    am = prompt_mask.to(device)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids.to(device), attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    past, logits, last = out.past_key_values, out.logits[:, -1].float(), pos[:, -1:]
    tokens, logprobs, valid = [], [], []
    done = torch.zeros(batch, dtype=torch.bool)
    for _ in range(max_new):
        if temperature > 0:
            scaled = logits / temperature
            probs = torch.softmax(scaled.double().cpu(), dim=-1)
            u = torch.rand(batch, 1, generator=generator, dtype=torch.float64)
            nxt = torch.searchsorted(probs.cumsum(-1), u).squeeze(1).clamp_(max=probs.size(-1) - 1)
        else:
            scaled = logits
            nxt = logits.argmax(-1).cpu()
        valid.append(~done)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        tokens.append(nxt)
        logprobs.append(torch.log_softmax(scaled, dim=-1).gather(1, nxt.to(device)[:, None]).squeeze(1).cpu())
        done = done | torch.isin(nxt, stop)
        if bool(done.all()):
            break
        am = torch.cat([am, torch.ones(batch, 1, dtype=am.dtype, device=device)], dim=1)
        last = last + 1
        out = model(input_ids=nxt.to(device)[:, None], attention_mask=am, position_ids=last, past_key_values=past,
                    use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
    response, mask = torch.stack(tokens, 1), torch.stack(valid, 1)
    return Rollout(input_ids=torch.cat([ids, response], 1),
                   attention_mask=torch.cat([prompt_mask, torch.ones_like(response)], 1),
                   prompt_len=prompt_len, response=response, logprobs=torch.stack(logprobs, 1) * mask, mask=mask)


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    """Log-probabilities [B, L] of the response tokens (the last L columns of input_ids) under the model.

    On CUDA the forward runs in BF16 autocast over the FP32 (master) weights.
    """
    length = response.size(1)
    pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.is_cuda):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos,
                       logits_to_keep=length + 1).logits
    logits = logits[:, :-1].float()
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    return -F.cross_entropy(logits.reshape(-1, logits.size(-1)), response.reshape(-1).to(logits.device),
                            reduction="none").view(response.shape)
