"""QuRL training on GSM8K in plain PyTorch: a train + validation loop, without VeRL or vLLM.

One RL step (the paper's Sec. 4), on one device:
  1. Rollout. The old actor, quantized to INT8 W8A8 (qurl.fakequant; re-quantized from the current weights at every
     step), samples `group` responses to each of `prompts` training problems at temperature T and records each
     token's log-probability mu under it: the behavior policy.
  2. Rewards (exact match of the final answer) and GRPO advantages: the reward minus its group's mean, divided by the
     group's standard deviation (VeRL's default).
  3. The full-precision old actor's log-probabilities pi_old of the same tokens (the proximal policy), computed on the
     same micro-batches as training, and the token ratios rho = pi_old / mu.
  4. `ppo_epochs` passes over the batch, each in `minibatches` AdamW steps with global-norm gradient clipping, on the
     token-mean of QuRL's TIS + ACR loss (qurl.loss.tis_acr_terms): the decoupled PPO objective weighted by
     min(rho, C), with ACR's raised upper clip bound for truncated tokens.
Validation, before training, every `val_every` steps and at the end: greedy decoding of GSM8K test problems by the
full-precision actor and/or its INT8 copy (`val_policy`); accuracy and mean response length.
UAQ (`--uaq-s 1.5`): the model is rescaled once, before training, and checked exactly (qurl.uaq.check).

    python -m qurl.train --out runs/gsm8k_qurl [--steps 100 --uaq-s 1.5 ...]

Writes <out>/config.json, <out>/metrics.jsonl (one record per training step and per validation), and the model in
Hugging Face format to <out>/final (and to <out>/step<k> every `save_every` steps), with qurl.json (step, uaq_s).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from . import gsm8k, uaq
from .fakequant import int8_rollout
from .gsm8k import Problem
from .loss import tis_acr_terms
from .sampling import Rollout, generate, sequence_logprobs, stop_token_ids


def _opt(default, help_text):
    return field(default=default, metadata={"help": help_text})


@dataclass
class TrainConfig:
    """A run's settings; each field is also a command-line option (--field-name)."""
    model: str = _opt("Qwen/Qwen2.5-0.5B-Instruct", "hub id or directory of the starting model")
    out: str = _opt("runs/qurl_gsm8k", "output directory")
    data_dir: str | None = _opt(None, f"GSM8K jsonl directory (default {gsm8k.DEFAULT_DIR}; downloaded if missing)")
    steps: int = _opt(100, "RL steps (rollout batches)")
    prompts: int = _opt(16, "training problems per step")
    group: int = _opt(8, "responses per problem (the GRPO group)")
    max_new: int = _opt(512, "longest response, in tokens")
    temperature: float = _opt(1.0, "sampling temperature of the rollout")
    quantize_rollout: bool = _opt(True, "INT8 W8A8 rollout (--no-quantize-rollout: full-precision RL baseline)")
    uaq_s: float | None = _opt(None, "UAQ scale s (the paper: 1.5); default: no UAQ")
    tis_c: float = _opt(2.0, "TIS cap C")
    clip_low: float = _opt(0.2, "PPO epsilon_low")
    clip_high: float = _opt(0.2, "PPO epsilon_high (DAPO: 0.28)")
    clip_c: float = _opt(3.0, "dual-clip constant for negative advantages (VeRL's default)")
    grpo_std: bool = _opt(True, "divide GRPO advantages by the group's standard deviation")
    ppo_epochs: int = _opt(1, "passes over each rollout batch")
    minibatches: int = _opt(2, "optimizer steps per pass")
    micro_batch: int = _opt(4, "sequences per forward/backward (gradient accumulation within a minibatch)")
    lr: float = _opt(1e-6, "AdamW learning rate")
    weight_decay: float = _opt(0.01, "AdamW decoupled weight decay")
    adam_eps: float = _opt(1e-8, "AdamW epsilon")
    max_grad_norm: float = _opt(1.0, "global-norm gradient clipping threshold")
    val_every: int = _opt(10, "validate every this many steps (0: only before training and at the end)")
    val_size: int = _opt(256, "test problems per validation (0: all)")
    val_batch: int = _opt(32, "problems per validation batch")
    val_max_new: int = _opt(512, "longest validation response, in tokens")
    val_policy: str = _opt("fp", "policy to validate: fp, int8 or both")
    val_before_train: bool = _opt(True, "validate before the first step")
    save_every: int = _opt(0, "save the model every this many steps (0: only the final model)")
    seed: int = _opt(0, "seed of the problem order and of the sampling")
    device: str | None = _opt(None, "device (default: cuda if available, else cpu)")
    attn: str = _opt("sdpa", "attention implementation (sdpa or eager)")
    grad_ckpt: bool = _opt(False, "gradient checkpointing in the update (less memory, slower)")


@dataclass
class Task:
    """What the loop trains and validates on: problems, the user message for a question, and the reward."""
    train: Sequence[Problem]
    val: Sequence[Problem]
    prompt: Callable[[str], str] = gsm8k.prompt
    reward: Callable[[str, object], float] = gsm8k.reward


def gsm8k_task(data_dir: str | None = None) -> Task:
    """GSM8K: the train split to train on, the test split to validate on."""
    return Task(train=gsm8k.load_split("train", data_dir), val=gsm8k.load_split("test", data_dir))


def check_config(cfg: TrainConfig) -> None:
    batch = cfg.prompts * cfg.group
    if cfg.group < 2:
        raise ValueError("group must be at least 2 (GRPO compares responses within a group)")
    if cfg.temperature <= 0:
        raise ValueError("the rollout must sample (temperature > 0)")
    if batch % (cfg.minibatches * cfg.micro_batch):
        raise ValueError(f"prompts x group ({batch}) must be a multiple of minibatches x micro_batch "
                         f"({cfg.minibatches} x {cfg.micro_batch})")
    if cfg.val_policy not in ("fp", "int8", "both"):
        raise ValueError("val_policy must be fp, int8 or both")


def grpo_advantages(rewards: torch.Tensor, group: int, std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """GRPO: each reward minus its group's mean, divided by the group's (unbiased) standard deviation plus eps."""
    r = rewards.view(-1, group)
    a = r - r.mean(dim=1, keepdim=True)
    if std:
        a = a / (r.std(dim=1, keepdim=True) + eps)
    return a.reshape(-1)


def chat_prompts(tok, task: Task, questions: Sequence[str]) -> list[str]:
    return [tok.apply_chat_template([{"role": "user", "content": task.prompt(q)}], tokenize=False,
                                    add_generation_prompt=True) for q in questions]


def decode(tok, roll: Rollout) -> list[str]:
    return [tok.decode(roll.response[i][roll.mask[i]], skip_special_tokens=True) for i in range(len(roll.response))]


def rollout_context(model, quantized: bool):
    return int8_rollout(model) if quantized else contextlib.nullcontext()


def ppo_update(model, opt, roll: Rollout, advantages: torch.Tensor, cfg: TrainConfig) -> dict:
    """QuRL's policy update on one rollout batch (steps 3 and 4 of the module docstring); returns its statistics."""
    device = next(model.parameters()).device
    ids, am, resp = roll.input_ids.to(device), roll.attention_mask.to(device), roll.response.to(device)
    mask = roll.mask.to(device).float()
    mu = roll.logprobs.to(device)
    size = ids.size(0)
    micro = [slice(i, i + cfg.micro_batch) for i in range(0, size, cfg.micro_batch)]
    per_step = len(micro) // cfg.minibatches
    with torch.no_grad():   # the proximal policy: the full-precision old actor, on the micro-batches of the update
        lp_old = torch.cat([sequence_logprobs(model, ids[s], am[s], resp[s], cfg.temperature) for s in micro]) * mask
    rho = torch.exp(lp_old - mu) * mask      # pi_old / mu per token, 0 on padding
    adv = advantages.to(device)[:, None] * mask
    sums = {"clipped": 0.0, "clipped_lower": 0.0, "kl": 0.0, "acr": 0.0, "tokens": 0.0}
    losses, norms = [], []
    for _ in range(cfg.ppo_epochs):
        for k in range(cfg.minibatches):
            part = micro[k * per_step:(k + 1) * per_step]
            n_tok = max(sum(float(mask[s].sum()) for s in part), 1.0)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in part:
                lp = sequence_logprobs(model, ids[s], am[s], resp[s], cfg.temperature)
                t = tis_acr_terms(lp_old[s], lp, adv[s], rho[s], clip_ratio_low=cfg.clip_low,
                                  clip_ratio_high=cfg.clip_high, clip_ratio_c=cfg.clip_c, tis_c=cfg.tis_c)
                m = mask[s]
                loss = (t.losses * m).sum() / n_tok          # token-mean over the minibatch
                loss.backward()
                with torch.no_grad():
                    total += float(loss)
                    sums["clipped"] += float((t.clipped.float() * m).sum())
                    sums["clipped_lower"] += float((t.clipped_lower.float() * (adv[s] < 0).float() * m).sum())
                    sums["kl"] += float((-t.neg_kl * m).sum())
                    sums["acr"] += float((t.acr_kept.float() * m).sum())
                    sums["tokens"] += float(m.sum())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if torch.isfinite(norm):
                opt.step()
            else:                    # skip the step, as VeRL does
                opt.zero_grad(set_to_none=True)
            losses.append(total)
            norms.append(float(norm))
    n = max(sums["tokens"], 1.0)
    valid = mask.bool()
    return {"loss": sum(losses) / len(losses), "grad_norm": sum(norms) / len(norms),
            "pg_clipfrac": sums["clipped"] / n, "pg_clipfrac_lower": sums["clipped_lower"] / n,
            "ppo_kl": sums["kl"] / n, "acr_kept_frac": sums["acr"] / n,
            "tis_trunc_frac": float((rho[valid] > cfg.tis_c).float().mean()) if valid.any() else 0.0,
            "rho_max": float(rho.max()),
            # the paper's D_KL(pi_behav || pi_prox) = E_mu[log mu - log pi_old] (its Fig. 3a)
            "kl_behav_prox": float((mu - lp_old)[valid].mean()) if valid.any() else 0.0}


@torch.no_grad()
def validate(model, tok, task: Task, problems: Sequence[Problem], cfg: TrainConfig, quantized: bool,
             stop_ids: list[int]) -> dict:
    """Greedy decoding of `problems`: accuracy (mean reward) and mean response length."""
    correct, lengths = 0.0, []
    for i in range(0, len(problems), cfg.val_batch):
        chunk = problems[i:i + cfg.val_batch]
        with rollout_context(model, quantized):
            roll = generate(model, tok, chat_prompts(tok, task, [p.question for p in chunk]), max_new=cfg.val_max_new,
                            temperature=0.0, stop_ids=stop_ids)
        correct += sum(task.reward(text, p.answer) for text, p in zip(decode(tok, roll), chunk))
        lengths += roll.mask.sum(1).tolist()
    return {"acc": correct / len(problems), "response_len": sum(lengths) / len(lengths)}


def save(model, tok, path: str, step: int, cfg: TrainConfig) -> None:
    """The model (in UAQ coordinates if UAQ was applied; the function is the same) and its tokenizer."""
    model.save_pretrained(path)
    tok.save_pretrained(path)
    with open(os.path.join(path, "qurl.json"), "w") as f:
        json.dump({"step": step, "uaq_s": cfg.uaq_s}, f)


def train(cfg: TrainConfig, task: Task | None = None) -> list[dict]:
    """Run QuRL as configured; returns the metric records, which are also written to <out>/metrics.jsonl."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    check_config(cfg)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":    # FP32 matmuls stay FP32: the INT8 simulation multiplies integer codes in FP32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out = os.path.expanduser(cfg.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    task = task if task is not None else gsm8k_task(cfg.data_dir)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32, attn_implementation=cfg.attn).eval()
    records: list[dict] = []
    log = open(os.path.join(out, "metrics.jsonl"), "w")

    def emit(rec: dict) -> None:
        records.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "sample"}
        print(json.dumps(shown), flush=True)

    if cfg.uaq_s:   # rescales the model in place, on the CPU in FP32, with the exact s = 2 probe
        emit({"step": 0, "kind": "uaq", "s": cfg.uaq_s, **uaq.check(model, uaq.prompt_ids(tok), cfg.uaq_s)})
    model.to(device)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=cfg.adam_eps,
                            weight_decay=cfg.weight_decay)
    stop_ids = stop_token_ids(tok)
    order = torch.randperm(len(task.train), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_problems = list(task.val[:cfg.val_size] if cfg.val_size else task.val)
    policies = {"fp": ["fp"], "int8": ["int8"], "both": ["fp", "int8"]}[cfg.val_policy]

    def run_validation(step: int) -> None:
        t0 = time.time()
        rec: dict = {"step": step, "kind": "val", "problems": len(val_problems)}
        for policy in policies:
            res = validate(model, tok, task, val_problems, cfg, policy == "int8", stop_ids)
            rec[f"val/{policy}/acc"], rec[f"val/{policy}/response_len"] = res["acc"], res["response_len"]
        rec["val_s"] = time.time() - t0
        emit(rec)

    if cfg.val_before_train:
        run_validation(0)
    for step in range(1, cfg.steps + 1):
        t0 = time.time()
        problems = [task.train[order[((step - 1) * cfg.prompts + j) % len(order)]] for j in range(cfg.prompts)]
        gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + step)
        with rollout_context(model, cfg.quantize_rollout):   # theta_hat_old = Q(theta_old), re-quantized every step
            roll = generate(model, tok, chat_prompts(tok, task, [p.question for p in problems]), n=cfg.group,
                            max_new=cfg.max_new, temperature=cfg.temperature, generator=gen, stop_ids=stop_ids)
        t1 = time.time()
        texts = decode(tok, roll)
        rewards = torch.tensor([task.reward(text, problems[i // cfg.group].answer) for i, text in enumerate(texts)],
                               dtype=torch.float32)
        if cfg.grad_ckpt:
            model.train()    # Hugging Face checkpoints only in training mode (Qwen2 has no dropout)
        stats = ppo_update(model, opt, roll, grpo_advantages(rewards, cfg.group, cfg.grpo_std), cfg)
        model.eval()
        rec = {"step": step, "kind": "train", "reward": float(rewards.mean()),
               "response_len": float(roll.mask.sum(1).float().mean()), **stats,
               "rollout_s": t1 - t0, "update_s": time.time() - t1}
        if step == 1:
            rec["sample"] = texts[0][:500]
        emit(rec)
        if cfg.val_every and step % cfg.val_every == 0 and step < cfg.steps:
            run_validation(step)
        if cfg.save_every and step % cfg.save_every == 0 and step < cfg.steps:
            save(model, tok, os.path.join(out, f"step{step}"), step, cfg)
    if cfg.steps > 0:
        run_validation(cfg.steps)
    save(model, tok, os.path.join(out, "final"), cfg.steps, cfg)
    log.close()
    return records


_TYPES = {"int": int, "float": float, "str": str, "float | None": float, "str | None": str}


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="QuRL training on GSM8K (plain PyTorch: INT8 W8A8 fake-quantized "
                                             "rollout, GRPO, decoupled PPO with TIS + ACR, optional UAQ).")
    for f in dataclasses.fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default, help=f.metadata["help"])
        else:
            ap.add_argument(name, type=_TYPES[f.type], default=f.default, help=f.metadata["help"])
    return TrainConfig(**vars(ap.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
