"""X4 (revision): does the rounding between UAQ (R4) and the matched settings (R5) at s = 1.5 act like ulp noise?

Theorem 2: training with W_QKV/s and the input-norm gains gamma*s (R4, UAQ) is, in exact arithmetic, training the
original model with lr x s on W_QKV and / s on gamma, Adam eps / s and x s, weight decay / s and x s, and the clipping
norm computed in UAQ coordinates (R5). S2 found R4 and R5 bit-identical at s = 2, where every rescaling is exact. At
s = 1.5, W/s and gamma*s are rounded in FP32, so the arms differ from step 0 and drift apart; the paper assumes that
the rounding acts like a change of seed. X4 compares the (R4, R5) divergence with the divergence caused by +-1 ulp
noise (R5p, R4p) and by a seed change (R5_seed1). x4_run.sh runs the arms; x4_analysis.py applies the rule fixed in
advance.

This script runs one arm of S2's loop: s2_train.py's helpers are imported unchanged from $X4_S2_DIR (default
~/qurl/s2). That loop runs Qwen2.5-0.5B-Instruct on GSM8K with:
  - INT8 W8A8 fake-quantized rollouts and seeded inverse-CDF sampling;
  - GRPO with TIS and ACR;
  - AdamW with global-norm clipping;
  - BF16 autocast, FP32 master weights and deterministic kernels.
The training step in main() is copied verbatim from s2_train.main. Every X4 addition is marked "X4:".

X4 additions:
  --seed, --data-seed  --seed seeds the sampling, as in S2. The data order has its own seed (default: --seed), so a
                       seed change can keep the prompts (R5_seed1: --seed 1 --data-seed 0).
  --perturb G          at step 0, after the UAQ rescaling, moves every FP32 master weight of groups G (comma list of W,
                       gamma, rest) to an adjacent float: +1 or -1 ulp, with a random sign seeded by --perturb-seed.
  probe set            --make-probe draws 16 GSM8K training questions, which are then left out of training, and
                       samples one response to each from the base model under mu. At every step each arm records the
                       log-probabilities of these fixed responses under three policies:
                         - pi in FP32;
                         - pi in BF16 autocast, the training precision;
                         - mu, the INT8 W8A8 rollout policy.
  parameter subsets    fixed random subsets of the master weights of each group (W, gamma, rest). They are seeded and
                       the same in every arm, and are saved raw (FP32, in the arm's own coordinates) at every step.
                       x4_analysis.py maps them to original coordinates in float64 (W = s W', gamma = gamma'/s for
                       UAQ arms).
  per-step record      steps/step_NNNN.npz holds the subsets and the probe log-probabilities; step 0 is the initial
                       state. For steps >= 1 it also holds:
                         - the prompts, the sampled tokens and their mask;
                         - the rollout log-probabilities mu and the proximal log-probabilities pi_old;
                         - the rewards and advantages.
                       log.jsonl is S2's log, with SHA-256 fingerprints of the raw parameters per group.
                       meta.json holds the settings.

  python x4_train.py --make-probe --probe results/x4_rounding/probe.npz [--data ...]
  python x4_train.py --arm R4 --out results/x4_rounding/R4 --probe results/x4_rounding/probe.npz [--steps 50 ...]
  python x4_train.py ... --tiny --reward toy --max-new 24 --prompts 2       # smoke test
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

S2_DIR = os.environ.get("X4_S2_DIR", os.path.expanduser("~/qurl/s2"))
if not os.path.exists(os.path.join(S2_DIR, "s2_train.py")):
    sys.exit(f"x4_train.py: no s2_train.py in {S2_DIR} (set X4_S2_DIR)")
sys.path.insert(0, S2_DIR)
import s2_train as s2  # noqa: E402  S2's loop, unchanged: rollout, seq_logprobs, FakeQuantRollout, rewards, ...
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FORMAT = 1            # on-disk format read by x4_analysis.py
GROUPS = s2.GROUPS    # ("W", "gamma", "rest")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["R4", "R5"], help="R4: UAQ; R5: no UAQ, Theorem 2(b)-(c) settings")
    ap.add_argument("--out", help="output directory of the arm")
    ap.add_argument("--probe", required=True, help="probe set .npz (written by --make-probe, read by the arms)")
    ap.add_argument("--make-probe", action="store_true", help="write the probe set and exit")
    ap.add_argument("--probe-n", type=int, default=16, help="probe questions (--make-probe)")
    ap.add_argument("--probe-seed", type=int, default=7, help="probe questions and sampling seed (--make-probe)")
    ap.add_argument("--probe-chunk", type=int, default=4, help="probe sequences per forward pass")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--s", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50, help="rollout batches")
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
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--data-seed", type=int, default=None, help="data-order seed (default: --seed)")
    ap.add_argument("--perturb", default="", help="groups to move by +-1 ulp at step 0, e.g. W,gamma")
    ap.add_argument("--perturb-seed", type=int, default=1)
    ap.add_argument("--subset", type=int, default=2 ** 18, help="recorded elements per parameter group")
    ap.add_argument("--subset-seed", type=int, default=20261118)
    ap.add_argument("--data", default=None, help="GSM8K train jsonl (downloaded if missing)")
    ap.add_argument("--tiny", action="store_true", help="tiny random model with the real tokenizer (smoke test)")
    ap.add_argument("--reward", default="gsm8k", choices=["gsm8k", "toy"])
    a = ap.parse_args(argv)
    if not a.make_probe and (a.arm is None or a.out is None):
        ap.error("--arm and --out are required unless --make-probe")
    if a.data_seed is None:
        a.data_seed = a.seed
    a.perturb_groups = sorted({g for g in a.perturb.split(",") if g})
    bad = set(a.perturb_groups) - set(GROUPS)
    if bad:
        ap.error(f"--perturb: unknown groups {sorted(bad)}")
    return a


# ----------------------------------------------------------------------------- X4: helpers
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_arrays(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        v = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
    return h.hexdigest()


def save_npz(path, arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def data_path(a):
    return a.data or os.path.join(os.path.dirname(os.path.abspath(a.out or a.probe)), "gsm8k_train.jsonl")


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)   # as in s2_train.main
    tok.padding_side = "left"
    eos_ids = torch.tensor(sorted({i for i in [tok.eos_token_id, tok.pad_token_id,
                                               tok.convert_tokens_to_ids("<|im_end|>")] if i is not None}))
    return tok, eos_ids


def chat_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question + s2.SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)


def seq_logprobs_fp32(model, ids, am, P, resp):
    """s2_train.seq_logprobs without BF16 autocast: the policy of the FP32 master weights."""
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=am, position_ids=pos).logits
    logits = logits[:, P - 1:-1].float()
    return -F.cross_entropy(logits.reshape(-1, logits.size(-1)), resp.reshape(-1),
                            reduction="none").view_as(resp)


def probe_indices(n_data, n, seed):
    return np.sort(np.random.default_rng(10_000 + seed).permutation(n_data)[:n]).astype(np.int64)


def make_probe(a, dev):
    """Probe set: one response per question, sampled under mu from the base model (original coordinates)."""
    tok, eos_ids = load_tokenizer(a.model)
    model = s2.load_model(a, dev)
    data = s2.load_gsm8k(data_path(a))
    pidx = probe_indices(len(data), a.probe_n, a.probe_seed)
    prompts = [chat_prompt(tok, data[i][0]) for i in pidx]
    gen = torch.Generator().manual_seed(1_000_000_007 + a.probe_seed)
    roll = s2.rollout(model, tok, prompts, 1, a.max_new, gen, dev, eos_ids)
    arrays = dict(prompt_idx=pidx, ids=roll["ids"].numpy().astype(np.int64), am=roll["am"].numpy().astype(np.int64),
                  P=np.array(roll["P"], dtype=np.int64), resp=roll["resp"].numpy().astype(np.int64),
                  mask=roll["mask"].numpy().astype(bool))
    os.makedirs(os.path.dirname(os.path.abspath(a.probe)) or ".", exist_ok=True)
    save_npz(a.probe, arrays)
    lens = arrays["mask"].sum(1)
    print(f"probe set {a.probe}: {len(pidx)} questions, response tokens {int(lens.sum())} "
          f"(min {int(lens.min())}, max {int(lens.max())}), sha256 {sha_arrays(arrays)[:16]}", flush=True)


def load_probe(path, dev):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    t = {k: torch.from_numpy(arr[k]).to(dev) for k in ("ids", "am", "resp", "mask")}
    return dict(t, P=int(arr["P"]), prompt_idx=arr["prompt_idx"], mask_np=arr["mask"].astype(bool),
                sha=sha_arrays(arr))


@torch.no_grad()
def probe_logprobs(model, pr, chunk):
    """Log-probabilities of the fixed probe responses: pi in FP32, pi in BF16 autocast (s2's seq_logprobs, the
    training precision) and mu (INT8 W8A8, teacher-forced in one pass)."""
    n = pr["ids"].size(0)
    chunks = [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    args = [(pr["ids"][c], pr["am"][c], pr["P"], pr["resp"][c]) for c in chunks]
    out = {"pi32": [seq_logprobs_fp32(model, *x) for x in args],
           "pi16": [s2.seq_logprobs(model, *x) for x in args]}
    with s2.FakeQuantRollout(model):
        out["mu"] = [seq_logprobs_fp32(model, *x) for x in args]
    return {k: (torch.cat(v) * pr["mask"]).float().cpu().numpy() for k, v in out.items()}


class Subsets:
    """Fixed random subset of each group's elements, in named_parameters order: the same in every arm."""

    def __init__(self, named, groups, k, seed, dev):
        rng = np.random.default_rng(seed)
        self.parts, self.info = {}, {}
        for g in GROUPS:
            plist = [(n, p.numel()) for n, p in named if groups[n] == g]
            sizes = np.array([m for _, m in plist], dtype=np.int64)
            N = int(sizes.sum())
            idx = (np.arange(N, dtype=np.int64) if k >= N
                   else np.unique(rng.integers(0, N, size=k, dtype=np.int64)))
            offs = np.concatenate([[0], np.cumsum(sizes)])
            parts = []
            for j, (n, _) in enumerate(plist):
                lo, hi = np.searchsorted(idx, [offs[j], offs[j + 1]])
                if hi > lo:
                    parts.append((n, torch.from_numpy(idx[lo:hi] - offs[j]).to(dev)))
            self.parts[g] = parts
            names = ";".join(f"{n}:{m}" for n, m in plist)
            self.info[g] = dict(N=N, K=int(idx.size), idx_sha=sha_arrays({"idx": idx})[:16],
                                names_sha=hashlib.sha256(names.encode()).hexdigest()[:16])

    def take(self, params):
        return {g: torch.cat([params[n].detach().reshape(-1).index_select(0, sel) for n, sel in parts]).cpu().numpy()
                for g, parts in self.parts.items()}


@torch.no_grad()
def perturb_ulp(named, groups, which, seed):
    """Move every element of the chosen groups to an adjacent FP32 value (+-1 ulp, random sign; CPU generator)."""
    gen = torch.Generator().manual_seed(seed)
    changed = {g: 0 for g in which}
    for n, p in named:
        if groups[n] not in which:
            continue
        up = torch.randint(0, 2, tuple(p.shape), generator=gen, dtype=torch.uint8).bool().to(p.device)
        target = torch.where(up, torch.full_like(p, float("inf")), torch.full_like(p, float("-inf")))
        new = torch.nextafter(p.detach(), target)
        changed[groups[n]] += int((new != p).sum())
        p.copy_(new)
    return changed


def raw_hashes(named, groups):
    h = {g: hashlib.sha256() for g in GROUPS + ("all",)}
    for n, p in sorted(named, key=lambda x: x[0]):
        b = p.detach().float().cpu().numpy().tobytes()
        h[groups[n]].update(b)
        h["all"].update(b)
    return {g: x.hexdigest()[:16] for g, x in h.items()}


def prepare_out(out):
    if os.path.exists(os.path.join(out, "done")):
        sys.exit(f"x4_train.py: {out} is marked done; remove it to rerun")
    os.makedirs(os.path.join(out, "steps"), exist_ok=True)
    stale = glob.glob(os.path.join(out, "steps", "step_*.npz")) + \
        [os.path.join(out, f) for f in ("log.jsonl", "meta.json") if os.path.exists(os.path.join(out, f))]
    for f in stale:   # a rerun after a crash starts from scratch
        os.remove(f)


# ----------------------------------------------------------------------------- arm
def main():
    a = parse_args()
    s2.setup_determinism()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.make_probe:
        return make_probe(a, dev)
    tag = os.path.basename(os.path.normpath(a.out))
    prepare_out(a.out)
    t_start = time.time()
    tok, eos_ids = load_tokenizer(a.model)
    model = s2.load_model(a, dev)
    named = list(model.named_parameters())
    groups = {n: s2.group_of(n) for n, _ in named}
    assert {"W", "gamma", "rest"} <= set(groups.values()), "parameter groups not found"
    s, uaq = a.s, a.arm == "R4"
    if uaq:
        with torch.no_grad():
            for n, p in named:
                if groups[n] == "W":
                    p.div_(s)
                elif groups[n] == "gamma":
                    p.mul_(s)
    # X4: +-1 ulp perturbation of the FP32 master weights at step 0 (after the UAQ rescaling).
    perturbed = perturb_ulp(named, groups, a.perturb_groups, a.perturb_seed) if a.perturb_groups else {}

    # Optimizer settings per group (Theorem 2(b)), as in s2_train.main for R4 and R5.
    def hp(g):
        if a.arm == "R5" and g != "rest":
            k = s if g == "W" else 1.0 / s
            return dict(lr=k * a.lr, eps=a.eps / k, weight_decay=a.wd / k)
        return dict(lr=a.lr, eps=a.eps, weight_decay=a.wd)

    pgroups = [dict(params=[p for n, p in named if groups[n] == g], **hp(g)) for g in GROUPS]
    opt = torch.optim.AdamW(pgroups, betas=(0.9, 0.999), foreach=False, fused=False)
    # Clipping rule (c): R5 computes the global norm in UAQ coordinates. R4's own norm already is in them.
    clip_params = [p for n, p in named]
    clip_fac = [(s if groups[n] == "W" else 1.0 / s if groups[n] == "gamma" else 1.0) if a.arm == "R5" else 1.0
                for n, _ in named]

    dpath = data_path(a)
    data = s2.load_gsm8k(dpath)
    # X4: fixed probe set, left out of training; data order from --data-seed.
    probe = load_probe(a.probe, dev)
    held_out = set(int(i) for i in probe["prompt_idx"])
    order = np.array([i for i in np.random.default_rng(a.data_seed).permutation(len(data)) if int(i) not in held_out])
    assert len(order) >= a.steps * a.prompts, "not enough GSM8K questions for --steps x --prompts"
    subsets = Subsets(named, groups, a.subset, a.subset_seed, dev)
    params = dict(named)
    meta = dict(format=FORMAT, tag=tag, arm=a.arm, uaq=uaq, s=s, seed=a.seed, data_seed=a.data_seed,
                perturb=a.perturb_groups, perturb_seed=a.perturb_seed, perturbed_elements=perturbed,
                args={k: v for k, v in vars(a).items() if k != "perturb_groups"},
                subsets=subsets.info, probe_sha=probe["sha"][:16], probe_tokens=int(probe["mask_np"].sum()),
                data_sha=sha_file(dpath)[:16], n_data=len(data), s2_train=os.path.abspath(s2.__file__),
                s2_train_sha=sha_file(s2.__file__), x4_train_sha=sha_file(os.path.abspath(__file__)),
                torch=torch.__version__, transformers=transformers.__version__,
                device=str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"),
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{tag}] X4 arm {a.arm} s={s} seed={a.seed} data_seed={a.data_seed} "
          f"perturb={','.join(a.perturb_groups) or '-'} {perturbed or ''} steps={a.steps} on {meta['device']} "
          f"(torch {torch.__version__}, s2_train {meta['s2_train_sha'][:16]})", flush=True)

    def record(step, extra):
        """X4: subsets and probe log-probabilities (every step) plus the rollout arrays (steps >= 1)."""
        arrays = {f"raw_{g}": v for g, v in subsets.take(params).items()}
        arrays.update({f"probe_{k}": v for k, v in probe_logprobs(model, probe, a.probe_chunk).items()})
        arrays["probe_mask"] = probe["mask_np"]
        arrays.update(extra)
        save_npz(os.path.join(a.out, "steps", f"step_{step:04d}.npz"), arrays)

    log = open(os.path.join(a.out, "log.jsonl"), "w")
    record(0, {})
    log.write(json.dumps({"step": 0, "arm": a.arm, "raw_params": raw_hashes(named, groups)}) + "\n")
    log.flush()
    print(f"[{tag}] step 0 recorded ({time.time() - t_start:.0f} s since start)", flush=True)

    G, B = a.group, a.prompts * a.group
    secs = []
    for step in range(1, a.steps + 1):
        t0 = time.time()
        idx = order[(step - 1) * a.prompts: step * a.prompts]
        prompts = [chat_prompt(tok, data[i][0]) for i in idx]
        golds = [data[i][1] for i in idx for _ in range(G)]
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + step)
        roll = s2.rollout(model, tok, prompts, G, a.max_new, gen, dev, eos_ids)
        r = s2.rewards(tok, roll, golds, a.reward)
        A = r - r.view(a.prompts, G).mean(1).repeat_interleave(G)       # GRPO advantage (group-mean baseline)

        # ---- training step: copied verbatim from s2_train.main ----
        ids, am, resp, mask = (roll[k].to(dev) for k in ("ids", "am", "resp", "mask"))
        mu = roll["mu"].to(dev)
        mbs = list(torch.arange(B).chunk(a.minibatches))
        with torch.no_grad():
            lp_old = torch.cat([s2.seq_logprobs(model, ids[m], am[m], roll["P"], resp[m]) for m in mbs]) * mask
        rho = torch.exp(lp_old - mu)
        w = torch.clamp(rho, max=a.tis_c)                                 # TIS weight
        upper = (1 + a.clip_eps) / torch.clamp(a.tis_c / rho, max=1.0)  # ACR upper bound
        A_tok = A.to(dev)[:, None].expand_as(lp_old)
        stats = dict(norms=[], coefs=[], acr=0)
        for _ in range(a.epochs):
            for m in mbs:
                lp = s2.seq_logprobs(model, ids[m], am[m], roll["P"], resp[m])
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
        # ---- end of the copied training step ----

        # X4: per-step record and S2's log line (raw instead of mapped parameter fingerprints).
        record(step, dict(prompt_idx=np.asarray(idx, dtype=np.int64),
                          resp=roll["resp"].numpy().astype(np.int32), mask=roll["mask"].numpy().astype(bool),
                          mu=roll["mu"].float().numpy(), lp_old=lp_old.float().cpu().numpy(),
                          reward=r.float().numpy(), adv=A.float().numpy()))
        hashes = raw_hashes(named, groups)
        n_tok = int(mask.sum())
        secs.append(time.time() - t0)
        rec = {"step": step, "tok": s2.sha(roll["resp"] * roll["mask"]), "mu": s2.sha(roll["mu"].float()),
               "pi_old": s2.sha(lp_old.float()), "raw_params": hashes,
               "reward": float(r.mean()), "resp_tokens": n_tok,
               "tis_trunc_frac": float(((rho > a.tis_c) & mask).sum()) / max(n_tok, 1),
               "acr_kept": stats["acr"], "clip_frac": float(np.mean([c < 1.0 for c in stats["coefs"]])),
               "grad_norms": stats["norms"], "sec": round(secs[-1], 1)}
        if step == 1:
            rec["sample"] = tok.decode(roll["resp"][0][roll["mask"][0]], skip_special_tokens=True)[:400]
        log.write(json.dumps(rec) + "\n")
        log.flush()
        eta = (a.steps - step) * float(np.mean(secs)) / 60
        print(f"[{tag}] step {step}/{a.steps}: reward {rec['reward']:.3f} tokens {n_tok} clip {rec['clip_frac']:.2f} "
              f"trunc {rec['tis_trunc_frac']:.4f} acr {stats['acr']} params {hashes['all']} "
              f"({rec['sec']} s, ETA {eta:.1f} min)", flush=True)
    log.close()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
    print(f"[{tag}] finished {a.steps} steps in {(time.time() - t_start) / 60:.1f} min "
          f"(peak GPU memory {peak:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
