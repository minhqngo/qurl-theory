"""S2 (P1 note, section 7): exact test of Theorem 2(b)-(c) / Corollary 3 on a real model, at s = 2.

One QuRL-style loop on GSM8K (default model Qwen2.5-0.5B-Instruct, S3's model):
  rollout  mu: INT8 W8A8 fake quantization in PyTorch, fp32: per-output-channel absmax weights, dynamic per-token
               absmax activations, round-to-nearest, codes in [-127, 127]; lm_head unquantized (as in the
               llm-compressor checkpoints of S1). Sampling at temperature 1 by inverse CDF in float64 on the CPU,
               with uniforms from a per-step seeded generator.
  train    pi: same model in BF16 autocast with FP32 master weights. GRPO advantages (reward minus group mean);
               decoupled PPO with TIS weight min(pi_old/mu, C) and the ACR upper bound (1+eps)/min(1, C/rho);
               `epochs` x `minibatches` updates per rollout batch; AdamW (single-tensor implementation);
               global-norm gradient clipping.
Arms (identical data and sampling seeds):
  R4   UAQ: W_QKV/s and input_layernorm*s; plain lr, Adam eps, weight decay; ordinary clipping.
  R5   no UAQ, Theorem 2(b)-(c): lr s*eta on W_QKV and eta/s on gamma; eps eps/s and s*eps; weight decay wd/s and
       s*wd; clip norm computed in UAQ coordinates, i.e. on (s*g_W, g_gamma/s, g_rest).
  R5o  as R5 but with ordinary clipping (norm of g_W, g_gamma, g_rest): isolates rule (c); drifts when clipping binds.
  R5x  no UAQ, learning rates matched but eps, weight decay and clipping not (should drift from R4).
  R3   no UAQ, plain hyperparameters (QuRL without UAQ; should differ from R4).
Prediction (s a power of two, so every rescaling is exact): R4 and R5 give bit-identical rollouts and bit-identical
parameters after mapping R4 back (W = s*W', gamma = gamma'/s), at every step.

Each run writes <out>/log.jsonl with, per step, SHA-256 fingerprints of the sampled tokens, the rollout and proximal
log-probabilities and the parameters mapped to original coordinates (per group: W, gamma, rest), plus activity
statistics (clipping, TIS truncation, ACR); and parameter snapshots at --snap-steps. s2_compare.py compares runs.

  python s2_train.py --arm R4 --out results/R4 [--model ... --steps 20 ...]
  python s2_train.py --arm R4 --out results/smoke_R4 --tiny --reward toy --steps 4 --max-new 24   # smoke test
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must precede CUDA init

import argparse
import hashlib
import json
import re
import time
import types
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

GSM8K_TRAIN = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
               "grade_school_math/data/train.jsonl")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
QMAX = 127.0
GROUPS = ("W", "gamma", "rest")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["R4", "R5", "R5o", "R5x", "R3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--s", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20, help="rollout batches")
    ap.add_argument("--prompts", type=int, default=8, help="prompts per rollout batch")
    ap.add_argument("--group", type=int, default=4, help="samples per prompt (GRPO group)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2, help="passes over each rollout batch")
    ap.add_argument("--minibatches", type=int, default=2, help="optimizer steps per pass")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    ap.add_argument("--wd", type=float, default=0.01, help="decoupled weight decay (all parameters)")
    ap.add_argument("--tau", type=float, default=0.1, help="global-norm clipping threshold")
    ap.add_argument("--tis-c", type=float, default=2.0, help="TIS truncation C")
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon (ACR upper bound uses it too)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap-steps", default="1,last", help="steps at which to save mapped parameters")
    ap.add_argument("--data", default=None, help="GSM8K train jsonl (downloaded if missing)")
    ap.add_argument("--tiny", action="store_true", help="tiny random model with the real tokenizer (smoke test)")
    ap.add_argument("--reward", default="gsm8k", choices=["gsm8k", "toy"])
    return ap.parse_args()


def setup_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False


def group_of(name):
    if re.search(r"\.self_attn\.(q|k|v)_proj\.weight$", name):
        return "W"
    if name.endswith("input_layernorm.weight"):
        return "gamma"
    return "rest"


def load_model(a, dev):
    if a.tiny:
        cfg = AutoConfig.from_pretrained(a.model)
        cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
        cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
        if getattr(cfg, "head_dim", None) is not None:
            cfg.head_dim = 16
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager")
    return model.float().to(dev).eval()   # FP32 master weights; dropout is 0 in Qwen2 anyway


# ----------------------------------------------------------------------------- rollout quantizer (Theorem 1 class)
def _qforward(self, x):
    xs = x.float()
    sx = xs.abs().amax(dim=-1, keepdim=True)
    sx = torch.where(sx > 0, sx / QMAX, torch.ones_like(sx))
    xq = torch.round(xs / sx).clamp_(-QMAX, QMAX)
    y = (xq @ self._q_codes_t) * sx * self._q_scale
    if self.bias is not None:
        y = y + self.bias
    return y.to(x.dtype)


class FakeQuantRollout:
    """INT8 W8A8 for every decoder Linear while active. Absmax scales make it homogeneous (Theorem 1), and at
    s = 2 every scale is exactly halved or doubled, so the codes and the dequantized outputs are bit-identical."""

    def __init__(self, model):
        self.mods = [m for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")]

    def __enter__(self):
        for m in self.mods:
            w = m.weight.detach()
            sw = w.abs().amax(dim=1)
            sw = torch.where(sw > 0, sw / QMAX, torch.ones_like(sw))
            m._q_codes_t = torch.round(w / sw[:, None]).clamp_(-QMAX, QMAX).t().contiguous()
            m._q_scale = sw
            m.forward = types.MethodType(_qforward, m)
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            del m.forward, m._q_codes_t, m._q_scale


@torch.no_grad()
def rollout(model, tok, prompts, G, max_new, gen, dev, eos_ids):
    enc = tok(prompts, return_tensors="pt", padding=True)               # left padding
    ids = enc.input_ids.repeat_interleave(G, 0)
    pam = enc.attention_mask.repeat_interleave(G, 0)
    B, P = ids.shape
    am = pam.to(dev)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    toks, mus, valid = [], [], []
    done = torch.zeros(B, dtype=torch.bool)
    with FakeQuantRollout(model):
        out = model(input_ids=ids.to(dev), attention_mask=am, position_ids=pos, use_cache=True)
        past, logits, last = out.past_key_values, out.logits[:, -1].float(), pos[:, -1:]
        for _ in range(max_new):
            lp = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits.double().cpu(), dim=-1)       # sample in float64 on the CPU
            u = torch.rand(B, 1, generator=gen, dtype=torch.float64)
            nxt = torch.searchsorted(probs.cumsum(-1), u).squeeze(1).clamp_(max=probs.size(-1) - 1)
            valid.append(~done)
            nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
            toks.append(nxt)
            mus.append(lp.gather(1, nxt.to(dev)[:, None]).squeeze(1).cpu())
            done = done | torch.isin(nxt, eos_ids)
            if bool(done.all()):
                break
            am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=dev)], 1)
            last = last + 1
            out = model(input_ids=nxt.to(dev)[:, None], attention_mask=am, position_ids=last,
                        past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1].float()
    resp, mu, mask = torch.stack(toks, 1), torch.stack(mus, 1), torch.stack(valid, 1)
    full_ids = torch.cat([ids, resp], 1)
    full_am = torch.cat([pam, torch.ones_like(resp)], 1)
    return dict(ids=full_ids, am=full_am, P=P, resp=resp, mu=mu * mask, mask=mask)


# ----------------------------------------------------------------------------- policy log-probabilities (training)
def seq_logprobs(model, ids, am, P, resp):
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
        logits = model(input_ids=ids, attention_mask=am, position_ids=pos).logits
    logits = logits[:, P - 1:-1].float()
    return -F.cross_entropy(logits.reshape(-1, logits.size(-1)), resp.reshape(-1),
                            reduction="none").view_as(resp)


# ----------------------------------------------------------------------------- data and rewards
def load_gsm8k(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(urllib.request.urlopen(GSM8K_TRAIN).read())
        os.replace(tmp, path)
    rows = [json.loads(line) for line in open(path)]
    return [(r["question"], float(r["answer"].split("####")[-1].replace(",", "").strip())) for r in rows]


def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = boxed[-1] if boxed else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def rewards(tok, roll, golds, kind):
    out = []
    for i in range(roll["resp"].size(0)):
        r_ids = roll["resp"][i][roll["mask"][i]]
        if kind == "toy":            # smoke test: varies within a group even for a random model
            out.append(float((r_ids % 2 == 0).float().mean()) if len(r_ids) else 0.0)
        else:
            pred = extract_answer(tok.decode(r_ids, skip_special_tokens=True))
            out.append(float(pred is not None and abs(pred - golds[i]) < 1e-6))
    return torch.tensor(out)


# ----------------------------------------------------------------------------- fingerprints
def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def mapped_params(named, groups, s, uaq):
    """Parameters in original (non-UAQ) coordinates: W = s*W', gamma = gamma'/s. Exact for s a power of two."""
    out = {}
    for n, p in named:
        t = p.detach()
        if uaq and groups[n] == "W":
            t = t * s
        elif uaq and groups[n] == "gamma":
            t = t / s
        out[n] = t
    return out


def param_hashes(mp, groups):
    h = {g: hashlib.sha256() for g in GROUPS + ("all",)}
    for n in sorted(mp):
        b = mp[n].float().cpu().numpy().tobytes()
        h[groups[n]].update(b)
        h["all"].update(b)
    return {g: x.hexdigest()[:16] for g, x in h.items()}


def main():
    a = parse_args()
    setup_determinism()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    eos_ids = torch.tensor(sorted({i for i in [tok.eos_token_id, tok.pad_token_id,
                                               tok.convert_tokens_to_ids("<|im_end|>")] if i is not None}))
    model = load_model(a, dev)
    named = list(model.named_parameters())
    groups = {n: group_of(n) for n, _ in named}
    assert {"W", "gamma", "rest"} <= set(groups.values()), "parameter groups not found"
    s, uaq = a.s, a.arm == "R4"
    if uaq:
        with torch.no_grad():
            for n, p in named:
                if groups[n] == "W":
                    p.div_(s)
                elif groups[n] == "gamma":
                    p.mul_(s)

    # Optimizer settings per group (Theorem 2(b)); R5x matches the learning rates only.
    def hp(g):
        if a.arm in ("R5", "R5o", "R5x") and g != "rest":
            k = s if g == "W" else 1.0 / s
            if a.arm in ("R5", "R5o"):
                return dict(lr=k * a.lr, eps=a.eps / k, weight_decay=a.wd / k)
            return dict(lr=k * a.lr, eps=a.eps, weight_decay=a.wd)
        return dict(lr=a.lr, eps=a.eps, weight_decay=a.wd)

    pgroups = [dict(params=[p for n, p in named if groups[n] == g], **hp(g)) for g in GROUPS]
    opt = torch.optim.AdamW(pgroups, betas=(0.9, 0.999), foreach=False, fused=False)
    # Clipping rule (c): R5 computes the global norm in UAQ coordinates. R4's own norm already is in them.
    clip_params = [p for n, p in named]
    clip_fac = [(s if groups[n] == "W" else 1.0 / s if groups[n] == "gamma" else 1.0) if a.arm == "R5" else 1.0
                for n, _ in named]

    data = load_gsm8k(a.data or os.path.join(os.path.dirname(os.path.abspath(a.out)), "gsm8k_train.jsonl"))
    order = np.random.default_rng(a.seed).permutation(len(data))
    snaps = {int(x) if x != "last" else a.steps for x in a.snap_steps.split(",") if x}
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    mp0 = mapped_params(named, groups, s, uaq)
    log.write(json.dumps({"step": 0, "arm": a.arm, "args": vars(a), "params": param_hashes(mp0, groups),
                          "torch": torch.__version__, "device": str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")}) + "\n")
    del mp0

    G, B = a.group, a.prompts * a.group
    for step in range(1, a.steps + 1):
        t0 = time.time()
        idx = order[(step - 1) * a.prompts: step * a.prompts]
        prompts = [tok.apply_chat_template([{"role": "user", "content": data[i][0] + SUFFIX}],
                                           tokenize=False, add_generation_prompt=True) for i in idx]
        golds = [data[i][1] for i in idx for _ in range(G)]
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + step)
        roll = rollout(model, tok, prompts, G, a.max_new, gen, dev, eos_ids)
        r = rewards(tok, roll, golds, a.reward)
        A = r - r.view(a.prompts, G).mean(1).repeat_interleave(G)       # GRPO advantage (group-mean baseline)

        ids, am, resp, mask = (roll[k].to(dev) for k in ("ids", "am", "resp", "mask"))
        mu = roll["mu"].to(dev)
        mbs = list(torch.arange(B).chunk(a.minibatches))
        with torch.no_grad():
            lp_old = torch.cat([seq_logprobs(model, ids[m], am[m], roll["P"], resp[m]) for m in mbs]) * mask
        rho = torch.exp(lp_old - mu)
        w = torch.clamp(rho, max=a.tis_c)                                 # TIS weight
        upper = (1 + a.clip_eps) / torch.clamp(a.tis_c / rho, max=1.0)  # ACR upper bound
        A_tok = A.to(dev)[:, None].expand_as(lp_old)
        stats = dict(norms=[], coefs=[], acr=0)
        for _ in range(a.epochs):
            for m in mbs:
                lp = seq_logprobs(model, ids[m], am[m], roll["P"], resp[m])
                R = torch.exp(lp - lp_old[m])
                Am = A_tok[m]
                active = torch.where(Am > 0, R < upper[m], torch.where(Am < 0, R > 1 - a.clip_eps,
                                                                       torch.zeros_like(R, dtype=torch.bool)))
                active = active & mask[m]
                stats["acr"] += int(((Am > 0) & (R >= 1 + a.clip_eps) & active).sum())
                coeff = (w[m] * Am * active).detach()
                loss = -(coeff * R).sum() / mask[m].sum().clamp(min=1)
                opt.zero_grad(set_to_none=False)
                loss.backward()
                for p in clip_params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                norms = torch.stack([torch.linalg.vector_norm(p.grad) * f for p, f in zip(clip_params, clip_fac)])
                total = torch.linalg.vector_norm(norms)
                coef = torch.clamp(a.tau / (total + 1e-6), max=1.0)
                for p in clip_params:
                    p.grad.mul_(coef)
                opt.step()
                stats["norms"].append(float(total))
                stats["coefs"].append(float(coef))
        mp = mapped_params(named, groups, s, uaq)
        n_tok = int(mask.sum())
        rec = {"step": step, "tok": sha(roll["resp"] * roll["mask"]), "mu": sha(roll["mu"].float()),
               "pi_old": sha(lp_old.float()), "params": param_hashes(mp, groups),
               "reward": float(r.mean()), "resp_tokens": n_tok,
               "tis_trunc_frac": float(((rho > a.tis_c) & mask).sum()) / max(n_tok, 1),
               "acr_kept": stats["acr"], "clip_frac": float(np.mean([c < 1.0 for c in stats["coefs"]])),
               "grad_norms": stats["norms"], "sec": round(time.time() - t0, 1)}
        if step == 1:
            rec["sample"] = tok.decode(roll["resp"][0][roll["mask"][0]], skip_special_tokens=True)[:400]
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step in snaps:
            torch.save({n: t.cpu() for n, t in mp.items()}, os.path.join(a.out, f"params_step{step}.pt"))
        del mp
        print(f"step {step}: reward {rec['reward']:.3f} tokens {n_tok} clip {rec['clip_frac']:.2f} "
              f"trunc {rec['tis_trunc_frac']:.4f} acr {stats['acr']} params {rec['params']['all']} "
              f"({rec['sec']} s)", flush=True)
    log.close()


if __name__ == "__main__":
    main()
