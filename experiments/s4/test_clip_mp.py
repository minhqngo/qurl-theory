"""CPU test of qurl_verl/r5_optim.py's multi-rank arithmetic (S4), with several processes and the gloo backend.

  python test_clip_mp.py --nproc 2      (also run with 3: uneven FSDP2 shards; 1: the single-rank path)

Uses a tiny random Qwen2 model under FSDP2 (fully_shard on a CPU device mesh), float32 throughout, no VeRL.
  1. total_norm on FSDP2-sharded gradients equals a float64 reference computed from the full gradients to float32
     rounding (per-tensor norms are float32), for the factors of s = 1, 1.5 and 2, and is bit-identical on every rank.
  2. Theorem 2 at s = 2 across ranks: R4 (UAQ weights W/2 and 2*gamma, plain groups, ordinary clip) and R5 (original
     weights, Theorem 2(b) groups, clip rule (c)) take K AdamW steps on the same per-rank batches with clipping
     binding; mapped back (W = 2W', gamma = gamma'/2), R4's weights must equal R5's bit for bit. An R4 repeat must
     equal R4 (determinism), and R5o (R5's groups, ordinary clip) must differ.
Exit status 0 only if every check passes.
"""
import argparse
import importlib.util
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def load_r5():
    spec = importlib.util.spec_from_file_location("r5_optim_s4", os.path.join(HERE, "qurl_verl", "r5_optim.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_model(seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=96, hidden_size=48, intermediate_size=80, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False, attn_implementation="eager")
    m = Qwen2ForCausalLM(cfg).float()
    with torch.no_grad():   # random gains, so that gamma's group is exercised
        for layer in m.model.layers:
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
    return m


def to_uaq(model, s):
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.weight.div_(s)
            layer.input_layernorm.weight.mul_(s)


def shard(model, mesh):
    from torch.distributed.fsdp import fully_shard

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def batch(rank, step):
    g = torch.Generator().manual_seed(1000 * rank + step)
    ids = torch.randint(0, 96, (3, 17), generator=g)
    return ids[:, :-1], ids[:, 1:]


def loss_of(model, x, y):
    logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def full_params(model, r5, s_back):
    """Full (gathered) parameters, mapped back to original coordinates by s_back (1: unchanged)."""
    from torch.distributed.tensor import DTensor

    out = {}
    for n, p in model.named_parameters():
        t = (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        g = r5.group_of(n)
        out[n] = t * s_back if g == "W" else t / s_back if g == "gamma" else t
    return out


def run_arm(r5, mesh, rank, arm, steps, lr=1e-2, eps=1e-8, wd=0.1, clip=0.05):
    s = 2.0
    model = tiny_model(seed=0)
    if arm in ("R4", "R4r"):
        to_uaq(model, s)
    shard(model, mesh)
    named = [(n, p) for n, p in model.named_parameters()]
    gs = s if arm in ("R5", "R5o") else 1.0
    cs = s if arm == "R5" else 1.0
    groups, _ = r5.param_groups(named, lr, eps, wd, gs)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    clipf = r5.clip_factors(named, cs)
    norms = []
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_of(model, *batch(rank, k)).backward()
        norms.append(r5.clip_and_step(opt, clipf, clip))
    return full_params(model, r5, s if arm in ("R4", "R4r") else 1.0), norms


def check_norms(r5, mesh, rank, world):
    """Check 1: sharded total_norm against a float64 reference from full gradients."""
    from torch.distributed.tensor import DTensor

    ok = True
    model = shard(tiny_model(seed=1), mesh)
    loss_of(model, *batch(rank, 0)).backward()
    named = [(n, p) for n, p in model.named_parameters()]
    full = {n: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).double() for n, p in named}
    for s in (1.0, 1.5, 2.0):
        got = r5.total_norm(r5.clip_factors(named, s))
        ref = torch.sqrt(sum((f * full[n].norm()) ** 2 for (n, _), (_, f) in zip(named, r5.clip_factors(named, s))))
        allv = [torch.zeros(1, dtype=torch.float64) for _ in range(world)]
        dist.all_gather(allv, got.reshape(1))
        same = all(torch.equal(v, allv[0]) for v in allv)
        rel = abs(float(got) - float(ref)) / float(ref)
        if rank == 0:
            print(f"  total_norm s={s}: {float(got):.12e} vs reference {float(ref):.12e} (rel. diff {rel:.1e}); "
                  f"identical on all {world} ranks: {same}")
        # Per-tensor norms are float32 (as in S3 and in VeRL's own clip), so agreement is to float32 rounding; a
        # missing or double-counted shard would be off by far more than 1e-6.
        ok &= same and rel < 1e-6
    return ok


def worker(rank, world, init_file, steps, result_file):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,))
    r5 = load_r5()
    if rank == 0:
        print(f"== {world} rank(s), FSDP2 on CPU (gloo), tiny Qwen2, float32")
        print("check 1: clip norm over sharded gradients")
    ok = check_norms(r5, mesh, rank, world)
    if rank == 0:
        print(f"check 2: R4 vs R5 at s = 2 after {steps} AdamW steps with clipping")
    res = {arm: run_arm(r5, mesh, rank, arm, steps) for arm in ("R4", "R4r", "R5", "R5o")}
    if rank == 0:
        ref, nref = res["R4"]
        for arm, want in (("R4r", True), ("R5", True), ("R5o", False)):
            p, n = res[arm]
            same = all(torch.equal(ref[k], p[k]) for k in ref)
            print(f"  {arm} vs R4: weights bit-identical: {same} (expected {want}); grad norms equal: {n == nref}")
            print(f"    norms R4 {['%.6e' % x for x in nref]}\n    norms {arm} {['%.6e' % x for x in n]}")
            ok &= same == want and ((n == nref) == want)
        bound = all(x > 0.05 for x in nref)
        print(f"  clipping bound at every step (norm > 0.05): {bound}")
        ok &= bound
        with open(result_file, "w") as f:
            f.write("PASS" if ok else "FAIL")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    init_file = tempfile.mktemp(prefix="clip_mp_")
    result_file = tempfile.mktemp(prefix="clip_mp_result_")
    mp.spawn(worker, args=(a.nproc, init_file, a.steps, result_file), nprocs=a.nproc, join=True)
    verdict = open(result_file).read()
    print(verdict)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
