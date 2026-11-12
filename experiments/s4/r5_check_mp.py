"""Gate G3 (S4): R5 against R4 inside VeRL's FSDP2 actor engine on all GPUs (P1 note Theorem 2(b)-(c), at s = 2).

Multi-GPU version of experiments/s3/r5_check.py, launched with torchrun:
  torchrun --nproc_per_node 8 r5_check_mp.py run --arm R4 --model <UAQ s=2 checkpoint> --out <dir>
  torchrun --nproc_per_node 8 r5_check_mp.py run --arm R5 --model <base model>         --out <dir>
  python r5_check_mp.py compare <dir R4> <dir other>
No rollout: every run takes K actor updates through VeRL's FSDP2 engine (bf16 mixed precision, FP32 master weights,
the PPO loss, the qurl_verl optimizer patch with its cross-GPU clip norm), each GPU on its own fixed slice of a batch
(DAPO prompts with seeded random responses and advantages). At s = 2 every rescaling is exact, so
  R4: UAQ checkpoint (W_QKV/2, 2*gamma), QURL_OPT=plain (ordinary settings and clipping)
  R5: original checkpoint, QURL_OPT=r5 with QURL_S=2 (Theorem 2(b)-(c) settings, clip rule (c))
must give bit-identical weights after mapping R4's back (W = 2*W', gamma = gamma'/2), and identical gradient norms.
An R4 repeat checks determinism (the engine runs with strict deterministic algorithms and SDPA on padded batches);
R5 with ordinary clipping (R5o) must differ. Each GPU hashes (SHA-256) its own shard of every parameter, so a 7B check
needs no full-model gathers; rank 0 writes run.json with the hashes and the gradient norms.
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initializes

GROUP_HELP = "W = q/k/v_proj.weight, gamma = input_layernorm.weight, rest = everything else"


def build_batch(tok, data, rank, n, max_resp, seed):
    """Rank's slice: n DAPO prompts (left-padded) + seeded random responses (right-padded), VeRL's no-padding layout."""
    import pandas as pd
    import torch

    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    from verl.workers.utils.padding import left_right_2_no_padding

    df = pd.read_parquet(os.path.expanduser(data)).iloc[rank * n:(rank + 1) * n]
    p_ids = [tok(tok.apply_chat_template([dict(m) for m in p], add_generation_prompt=True, tokenize=False),
                 add_special_tokens=False).input_ids for p in df["prompt"]]
    g = torch.Generator().manual_seed(seed + rank)
    r_ids = [torch.randint(100, 5000, (int(torch.randint(max_resp // 2, max_resp, (1,), generator=g)),),
                           generator=g).tolist() + [tok.eos_token_id] for _ in range(n)]
    P, R, pad = max(map(len, p_ids)), max(map(len, r_ids)), tok.pad_token_id
    prompts = torch.tensor([[pad] * (P - len(x)) + x for x in p_ids])
    responses = torch.tensor([x + [pad] * (R - len(x)) for x in r_ids])
    p_mask = torch.tensor([[0] * (P - len(x)) + [1] * len(x) for x in p_ids])
    r_mask = torch.tensor([[1] * len(x) + [0] * (R - len(x)) for x in r_ids])
    attention_mask = torch.cat([p_mask, r_mask], dim=1)
    adv = torch.randn(n, 1, generator=g).expand(n, R).contiguous() * r_mask
    data = DataProto.from_single_dict(
        {"input_ids": torch.cat([prompts, responses], dim=1), "prompts": prompts, "attention_mask": attention_mask,
         "position_ids": compute_position_id_with_mask(attention_mask), "responses": responses,
         "response_mask": r_mask, "old_log_probs": torch.zeros(n, R), "advantages": adv},
        meta_info={"temperature": 1.0})
    return left_right_2_no_padding(data.to_tensordict())


def run(a):
    os.environ["QURL_OPT"] = {"R4": "plain", "R5": "r5", "R5o": "r5"}[a.arm]
    os.environ["QURL_S"] = str(a.s)
    if a.arm == "R5o":   # R5's per-group settings, but the ordinary clip norm: drops rule (c)
        os.environ["QURL_CLIP"] = "ordinary"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.distributed as dist
    from functools import partial
    from torch.distributed.tensor import DTensor
    from transformers import AutoTokenizer

    import qurl_verl  # noqa: F401  (optimizer patch, reads QURL_OPT / QURL_S)
    from qurl_verl import r5_optim
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine import EngineRegistry
    from verl.workers.utils.losses import ppo_loss

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    model_config = HFModelConfig(path=os.path.expanduser(a.model), load_tokenizer=False,
                                 override_config={"attn_implementation": "sdpa"}, use_remove_padding=False)
    engine_config = FSDPEngineConfig(forward_only=False, fsdp_size=-1, strategy="fsdp2",
                                     ulysses_sequence_parallel_size=1, use_remove_padding=False, use_dynamic_bsz=False,
                                     micro_batch_size_per_gpu=a.micro, max_token_len_per_gpu=16384,
                                     full_determinism=True, seed=42)
    optimizer_config = FSDPOptimizerConfig(lr=a.lr, weight_decay=a.wd, clip_grad=a.clip, lr_scheduler_type="constant")
    engine = EngineRegistry.new(model_type="language_model", backend="fsdp2", model_config=model_config,
                                engine_config=engine_config, optimizer_config=optimizer_config,
                                checkpoint_config=CheckpointConfig())
    engine.initialize()
    # VeRL's full_determinism uses warn_only=True, which leaves SDPA's memory-efficient backward non-deterministic;
    # strict mode makes PyTorch pick deterministic kernels (or fail loudly on an op that has none).
    torch.use_deterministic_algorithms(True, warn_only=False)
    actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, entropy_coeff=0.001)
    loss_fn = partial(ppo_loss, config=actor_config)
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.model))

    norms = []
    for _ in range(a.steps):
        batch = build_batch(tok, a.data, rank, a.n, a.max_resp, seed=0)
        tu.assign_non_tensor(batch, global_batch_size=a.n * world, use_remove_padding=False, use_dynamic_bsz=False,
                             max_token_len_per_gpu=16384, micro_batch_size_per_gpu=a.micro, use_fused_kernels=False,
                             calculate_entropy=True)
        with engine.train_mode(zero_grad_on_exit=True):
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(batch, loss_function=loss_fn, forward_only=False)
            norms.append(engine.optimizer_step())

    hashes = {}
    for n, p in engine.module.named_parameters():
        name = n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        t = (p.to_local() if isinstance(p, DTensor) else p).detach().float()
        if a.arm == "R4":   # map back to original coordinates; exact for s a power of two
            g = r5_optim.group_of(name)
            t = t * a.s if g == "W" else t / a.s if g == "gamma" else t
        hashes[name] = hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(hashes, gathered, dst=0)
    if rank == 0:
        os.makedirs(os.path.expanduser(a.out), exist_ok=True)
        with open(os.path.join(os.path.expanduser(a.out), "run.json"), "w") as f:
            json.dump({"arm": a.arm, "s": a.s, "model": a.model, "grad_norms": norms, "steps": a.steps, "lr": a.lr,
                       "wd": a.wd, "clip": a.clip, "n_per_rank": a.n, "world": world, "torch": torch.__version__,
                       "device": torch.cuda.get_device_name(0), "hashes": gathered}, f)
        print(f"{a.arm}: grad norms (before clipping) {norms}; hashed {len(hashes)} tensors on {world} ranks -> {a.out}")
    dist.barrier()
    dist.destroy_process_group()


def compare(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qurl_verl.r5_optim import group_of

    ra, rb = (json.load(open(os.path.join(os.path.expanduser(d), "run.json"))) for d in (a.ref, a.other))
    assert ra["world"] == rb["world"], "different numbers of ranks"
    names = list(ra["hashes"][0])
    print(f"{rb['arm']} vs {ra['arm']} after {ra['steps']} updates on {ra['world']} GPUs (s = {ra['s']}; {GROUP_HELP})")
    print(f"  grad norms {ra['arm']}: {ra['grad_norms']}\n  grad norms {rb['arm']}: {rb['grad_norms']}"
          f"\n  identical grad norms: {ra['grad_norms'] == rb['grad_norms']}")
    allsame = ra["grad_norms"] == rb["grad_norms"]
    for g in ("W", "gamma", "rest"):
        ns = [n for n in names if group_of(n) == g]
        same = sum(ha[n] == hb[n] for n in ns for ha, hb in zip(ra["hashes"], rb["hashes"]))
        tot = len(ns) * ra["world"]
        allsame &= same == tot
        print(f"  {g:<5}: {same} of {tot} parameter shards bit-identical")
    print("  BIT-IDENTICAL" if allsame else "  DIFFERENT")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=["R4", "R5", "R5o"])
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default="~/data/s4/dapo17k.parquet")
    r.add_argument("--s", type=float, default=2.0)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--n", type=int, default=2, help="sequences per GPU")
    r.add_argument("--max-resp", type=int, default=128)
    r.add_argument("--micro", type=int, default=1)
    r.add_argument("--lr", type=float, default=1e-5)
    r.add_argument("--wd", type=float, default=0.1)
    r.add_argument("--clip", type=float, default=0.1)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
