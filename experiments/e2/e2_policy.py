"""E2, policy space (P1 experiments report, E2 design): does the INT8 rollout policy mu follow the trainer's policy pi
from one checkpoint to the next? (Proposition 5, read in policy space.)

  python e2_policy.py <dumps dir> --model Qwen/Qwen2.5-0.5B-Instruct --data ~/data/gsm8k/train.parquet
                      [--prompts 128] [--max-new 256] [--out report.txt] [--json data.json]
  python e2_policy.py --self-test

Probe set: a fixed random subset of GSM8K training prompts (numpy seed 0), with the model's chat template, and one
response each sampled at temperature 1 (no top-k or top-p, no repetition penalty) from theta_0 (sync 1), at most
--max-new tokens (torch seed 0). Saved next to the dumps as policy_probe.pt and reused if present.
For every full-dump sync s and every probe response token t:
  log pi_s(t)   FP32 forward of the model with the dumped weights;
  log mu_s(t)   the same model under QuRL's W8A8 rollout quantizer, as deterministic FP32 fake quantization: each
                q/k/v/o/gate/up/down weight replaced by its INT8 codes times its row scale (the rollout's quantize_rows),
                and the input of each of these layers quantized per token (symmetric, absmax / 127, round half to
                even). Embeddings, norms and lm_head stay unquantized, as in the rollout engine.
For a pair (s, s'): d_pi = log pi_s' - log pi_s and d_mu = log mu_s' - log mu_s over the probe tokens; the OLS slope (with
its standard error) and intercept of d_mu on d_pi, Pearson r, and Var(d_mu - d_pi) / Var(d_pi). Pairs: consecutive full
dumps (one step), pooled over all one-step pairs, and (1, s') for every later full dump (cumulative).
Decision rules (pre-registered in the P1 experiments report), for every cumulative pair from s' = 9 on and for the
pooled one-step pairs: P5, first-order tracking, slope in [0.90, 1.10]; P5', the rollout is not frozen, slope > 0.5.
A rollout that did not follow the trainer would give a slope near 0.
"""
import argparse
import glob
import json
import math
import os
import sys

import torch

QMAX = 127.0
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CUM_FROM = 9


def quantize_rows(w):
    """The rollout's quantizer (qurl_verl.int8_rollout.quantize_rows): symmetric per-row absmax INT8."""
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(w / scale).clamp_(-QMAX, QMAX), scale


def _act_quant(module, args):
    x = args[0]
    s = x.abs().amax(dim=-1, keepdim=True) * (1.0 / QMAX)
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.round(x / s).clamp(-QMAX, QMAX) * s,) + tuple(args[1:])


class Policy:
    """An HF causal LM in FP32 whose weights can be set to a dumped state, with or without W8A8 fake quantization."""

    def __init__(self, model):
        self.model = model.float().eval()
        self.linears = {n: m for n, m in model.named_modules() if n.split(".")[-1] in PROJ and isinstance(m, torch.nn.Linear)}
        self.hooks = []

    def set(self, state, quantized):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.endswith("lm_head.weight")]
        if bad:
            raise RuntimeError(f"the dumped state misses model parameters: {bad[:5]}")
        if unexpected and not getattr(self, "_warned", False):
            print(f"note: dumped tensors not in the model's state dict, ignored: {list(unexpected)[:5]}", flush=True)
            self._warned = True
        if quantized:
            with torch.no_grad():
                for m in self.linears.values():
                    codes, scale = quantize_rows(m.weight.data)
                    m.weight.data.copy_(codes * scale)
                    self.hooks.append(m.register_forward_pre_hook(_act_quant))

    @torch.no_grad()
    def logprobs(self, seqs, starts, pad_id, bs=8):
        dev = next(self.model.parameters()).device
        out = []
        for i in range(0, len(seqs), bs):
            batch, st = seqs[i:i + bs], starts[i:i + bs]
            L = max(len(s) for s in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for j, s in enumerate(batch):
                ids[j, :len(s)] = s
                mask[j, :len(s)] = 1
            logits = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            for j, s in enumerate(batch):
                pos = torch.arange(st[j] - 1, len(s) - 1, device=dev)
                tgt = s[st[j]:].to(dev)
                out.append(lp[j, pos].gather(-1, tgt.view(-1, 1)).squeeze(-1).double().cpu())
        return torch.cat(out)


def pair_stat(x, y):
    n = x.numel()
    mx, my = x.mean(), y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    slope = cov / vx
    intercept = my - slope * mx
    res = y - intercept - slope * x
    se = math.sqrt(float((res ** 2).mean()) / (n * float(vx)))
    return {"tokens": n, "sd_dpi": float(vx.sqrt()), "slope": float(slope), "slope_se": se, "intercept": float(intercept),
            "r": float(cov / (vx * vy).sqrt()), "resid_share": float((((y - x) - (my - mx)) ** 2).mean() / vx)}


def full_syncs(d):
    return sorted({int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(d, "full", "sync*_b*.pt"))})


def load_state(d, k):
    st = {}
    for f in sorted(glob.glob(os.path.join(d, "full", f"sync{k:04d}_b*.pt"))):
        st.update(torch.load(f, map_location="cpu"))
    return st


def make_probe(pol, tok, data, n, max_new):
    import numpy as np
    import pandas as pd
    from transformers import GenerationConfig

    df = pd.read_parquet(data)
    rows = sorted(np.random.default_rng(0).choice(len(df), size=n, replace=False).tolist())
    dev = next(pol.model.parameters()).device
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                          max_new_tokens=max_new, eos_token_id=tok.eos_token_id,
                          pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    torch.manual_seed(0)
    seqs, starts = [], []
    for i in rows:
        msgs = [dict(m) for m in df.iloc[i]["prompt"]]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        p = tok(text, return_tensors="pt").input_ids
        out = pol.model.generate(input_ids=p.to(dev), attention_mask=torch.ones_like(p).to(dev), generation_config=gc)[0].cpu()
        resp = out[p.shape[1]:]
        eos = (resp == tok.eos_token_id).nonzero()
        if len(eos):
            resp = resp[:int(eos[0]) + 1]
        if len(resp):
            seqs.append(torch.cat([p[0], resp]))
            starts.append(p.shape[1])
    return {"rows": rows, "seqs": seqs, "starts": starts}


def analyse(pol, d, probe, pad_id):
    syncs = full_syncs(d)
    lp = {}
    for k in syncs:
        st = load_state(d, k)
        pol.set(st, quantized=False)
        pi = pol.logprobs(probe["seqs"], probe["starts"], pad_id)
        pol.set(st, quantized=True)
        mu = pol.logprobs(probe["seqs"], probe["starts"], pad_id)
        lp[k] = (pi, mu)
        print(f"sync {k}: {pi.numel()} probe tokens, mean log pi {float(pi.mean()):.4f}, mean log mu {float(mu.mean()):.4f},"
              f" mean |log mu - log pi| {float((mu - pi).abs().mean()):.4e}", flush=True)
    out = {"syncs": syncs, "one_step": {}, "cumulative": {}}
    xs, ys = [], []
    for a, b in zip(syncs, syncs[1:]):
        if b == a + 1:
            x, y = lp[b][0] - lp[a][0], lp[b][1] - lp[a][1]
            out["one_step"][f"{a}-{b}"] = pair_stat(x, y)
            xs.append(x); ys.append(y)
    if xs:
        out["one_step_pooled"] = pair_stat(torch.cat(xs), torch.cat(ys))
    for b in syncs[1:]:
        out["cumulative"][f"{syncs[0]}-{b}"] = pair_stat(lp[b][0] - lp[syncs[0]][0], lp[b][1] - lp[syncs[0]][1])
    out["mismatch"] = {k: float((v[1] - v[0]).abs().mean()) for k, v in lp.items()}
    return out


def rules(r):
    """P5: first-order tracking, slope in [0.90, 1.10]. P5': freezing rejected, slope > 0.5."""
    tested = [(f"cumulative {pk}", s) for pk, s in r["cumulative"].items() if int(pk.split("-")[1]) >= CUM_FROM]
    if "one_step_pooled" in r:
        tested.append(("one-step pairs pooled", r["one_step_pooled"]))
    res = []
    for name, s in tested:
        detail = f"{s['slope']:.4f} +- {s['slope_se']:.4f}"
        res.append((f"P5 {name}: slope in [0.90, 1.10]", 0.9 <= s["slope"] <= 1.1, detail))
        res.append((f"P5' {name}: slope > 0.5 (not frozen)", s["slope"] > 0.5, detail))
    return res


def report(r):
    lines = ["E2 policy-space analysis (e2_policy.py)", f"full dumps at syncs {r['syncs']}", "",
             f"{'pair':>9} {'tokens':>7} {'sd d_pi':>9} {'slope':>8} {'se':>7} {'intercept':>10} {'r':>6} {'resid share':>11}"]
    rows = list(r["one_step"].items())
    if "one_step_pooled" in r:
        rows.append(("1-step all", r["one_step_pooled"]))
    rows += list(r["cumulative"].items())
    for pk, s in rows:
        lines.append(f"{pk:>9} {s['tokens']:7d} {s['sd_dpi']:9.3e} {s['slope']:8.4f} {s['slope_se']:7.4f}"
                     f" {s['intercept']:10.2e} {s['r']:6.3f} {s['resid_share']:11.4f}")
    lines.append("mean |log mu - log pi| by sync: " + ", ".join(f"{k}: {v:.3e}" for k, v in r["mismatch"].items()))
    lines += ["", "Decision rules (pre-registered):"]
    for name, ok, detail in rules(r):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return "\n".join(lines)


def self_test():
    """Tiny random Qwen2 on the CPU. theta_1 = theta_0 + an Adam-like step that raises the probe log-likelihood:
    mu must follow pi (slope near 1). With the rollout weights kept at theta_0's codes, it must not (slope 0)."""
    import tempfile

    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    pol = Policy(model)
    seqs = [torch.randint(0, 256, (48,)) for _ in range(64)]
    starts = [24] * len(seqs)
    st0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.zero_grad()
    total = 0
    for s, b in zip(seqs, starts):
        lp = torch.log_softmax(model(input_ids=s[None]).logits.float(), -1)[0]
        total = total + lp[torch.arange(b - 1, len(s) - 1), s[b:]].sum()
    total.backward()
    st1 = {}
    for name, p in model.named_parameters():
        step = 0.0
        if name.endswith(".weight") and name.split(".")[-2] in PROJ and p.grad is not None:
            delta = p.detach().abs().amax(dim=1, keepdim=True) / QMAX        # update of 0.3 grid steps
            step = 0.3 * delta * torch.sign(p.grad)
        st1[name] = (p.detach() + step).clone()
    for k, v in st0.items():
        st1.setdefault(k, v)
    d = tempfile.mkdtemp(prefix="e2_pol_")
    os.makedirs(os.path.join(d, "full"))
    torch.save(st0, os.path.join(d, "full", "sync0001_b00.pt"))
    torch.save(st1, os.path.join(d, "full", "sync0002_b00.pt"))
    probe = {"seqs": seqs, "starts": starts}
    r = analyse(pol, d, probe, pad_id=0)
    s = r["one_step"]["1-2"]
    ok1 = 0.8 <= s["slope"] <= 1.2 and s["r"] > 0.3
    # Frozen control: mu at sync 2 computed with sync 1's weights for the quantized layers.
    frozen = dict(st1)
    for k in st1:
        if k.endswith(".weight") and k.split(".")[-2] in PROJ:
            frozen[k] = st0[k]
    pol.set(st1, False); x = pol.logprobs(seqs, starts, 0)
    pol.set(st0, False); x0 = pol.logprobs(seqs, starts, 0)
    pol.set(frozen, True); y = pol.logprobs(seqs, starts, 0)
    pol.set(st0, True); y0 = pol.logprobs(seqs, starts, 0)
    sf = pair_stat(x - x0, y - y0)
    ok2 = abs(sf["slope"]) < 0.2
    print(f"self-test: tracking slope {s['slope']:.3f} (r {s['r']:.3f}) in [0.8, 1.2]: {ok1}; frozen-weights control"
          f" slope {sf['slope']:.3f} near 0: {ok2}")
    print(report(r).splitlines()[3])
    print("SELF-TEST PASSED" if ok1 and ok2 else "SELF-TEST FAILED")
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="?")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=os.path.expanduser("~/data/gsm8k/train.parquet"))
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    pol = Policy(model)
    probe_path = os.path.join(os.path.dirname(os.path.abspath(a.dumps.rstrip("/"))), "policy_probe.pt")
    if os.path.exists(probe_path):
        probe = torch.load(probe_path)
    else:
        pol.set(load_state(a.dumps, full_syncs(a.dumps)[0]), quantized=False)
        probe = make_probe(pol, tok, a.data, a.prompts, a.max_new)
        torch.save(probe, probe_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    r = analyse(pol, a.dumps, probe, pad)
    text = report(r)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
