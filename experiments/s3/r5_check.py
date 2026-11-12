"""Exactness check of R5 against R4 inside VeRL's own FSDP actor engine (P1 note Theorem 2(b)-(c), at s = 2).

No rollout is involved: every run takes K actor updates on the same fixed GSM8K batch (prompts plus the reference
solutions as responses, seeded advantages), through VeRL's FSDP1 engine with mixed precision, the PPO loss with
entropy bonus, and the qurl_verl optimizer patch. At s = 2 every rescaling is exact, so
  R4: UAQ checkpoint (W_QKV/2, 2*gamma), QURL_OPT=plain (ordinary settings and clipping)
  R5: original checkpoint, QURL_OPT=r5 with QURL_S=2 (Theorem 2(b)-(c) settings, clip rule (c))
must give bit-identical weights after mapping R4's back (W = 2*W', gamma = gamma'/2), and identical gradient norms.
A repeat of R4 checks that the engine is deterministic; R5 with ordinary clipping (R5o) shows rule (c) matters.

  python r5_check.py run --arm R4  --model ~/qurl/models/qwen0.5b-instruct-uaq2 --out ~/qurl/s3/r5check/R4
  python r5_check.py run --arm R5  --model Qwen/Qwen2.5-0.5B-Instruct          --out ~/qurl/s3/r5check/R5
  python r5_check.py compare ~/qurl/s3/r5check/R4 ~/qurl/s3/r5check/R5
"""
import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initializes

GROUP_HELP = "W = q/k/v_proj.weight, gamma = input_layernorm.weight, rest = everything else"


def build_batch(tok, n, max_resp, seed):
    """Left-padded prompts + right-padded responses (VeRL's layout), as a no-padding TensorDict for the engine."""
    import pandas as pd
    import torch

    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask
    from verl.workers.utils.padding import left_right_2_no_padding

    df = pd.read_parquet(os.path.expanduser("~/data/gsm8k/train.parquet")).iloc[:n]
    p_ids = [tok(tok.apply_chat_template([dict(m) for m in p], add_generation_prompt=True, tokenize=False),
                 add_special_tokens=False).input_ids for p in df["prompt"]]
    r_ids = [tok(x["answer"], add_special_tokens=False).input_ids[:max_resp - 1] + [tok.eos_token_id]
             for x in df["extra_info"]]
    P, R, pad = max(map(len, p_ids)), max(map(len, r_ids)), tok.pad_token_id
    prompts = torch.tensor([[pad] * (P - len(x)) + x for x in p_ids])
    responses = torch.tensor([x + [pad] * (R - len(x)) for x in r_ids])
    p_mask = torch.tensor([[0] * (P - len(x)) + [1] * len(x) for x in p_ids])
    r_mask = torch.tensor([[1] * len(x) + [0] * (R - len(x)) for x in r_ids])
    attention_mask = torch.cat([p_mask, r_mask], dim=1)
    g = torch.Generator().manual_seed(seed)
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
    from transformers import AutoTokenizer

    import qurl_verl  # noqa: F401  (optimizer patch, reads QURL_OPT / QURL_S)
    from qurl_verl import r5_optim
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine import EngineRegistry
    from verl.workers.utils.losses import ppo_loss

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1)
    model_config = HFModelConfig(path=os.path.expanduser(a.model), load_tokenizer=False,
                                 override_config={"attn_implementation": "sdpa"}, use_remove_padding=False)
    engine_config = FSDPEngineConfig(forward_only=False, fsdp_size=1, strategy="fsdp2",
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
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    norms = []
    for _ in range(a.steps):
        batch = build_batch(tok, a.n, a.max_resp, seed=0)
        tu.assign_non_tensor(batch, global_batch_size=a.n, use_remove_padding=False, use_dynamic_bsz=False,
                             max_token_len_per_gpu=16384, micro_batch_size_per_gpu=a.micro, use_fused_kernels=False,
                             calculate_entropy=True)
        with engine.train_mode(zero_grad_on_exit=True):
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(batch, loss_function=loss_fn, forward_only=False)
            norms.append(engine.optimizer_step())

    from torch.distributed.tensor import DTensor
    if True:   # FSDP2: parameters are DTensors; on one rank full_tensor() is the whole parameter
        params = {}
        for n, p in engine.module.named_parameters():
            name = n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
            t = (p.full_tensor() if isinstance(p, DTensor) else p).detach().float().cpu().clone()
            if a.arm == "R4":   # map back to original coordinates; exact for s a power of two
                gname = r5_optim.group_of(name)
                t = t * a.s if gname == "W" else t / a.s if gname == "gamma" else t
            params[name] = t
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    torch.save(params, os.path.join(os.path.expanduser(a.out), "params.pt"))
    with open(os.path.join(os.path.expanduser(a.out), "run.json"), "w") as f:
        json.dump({"arm": a.arm, "s": a.s, "model": a.model, "grad_norms": norms, "steps": a.steps, "lr": a.lr,
                   "wd": a.wd, "clip": a.clip, "n": a.n, "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0)}, f, indent=1)
    print(f"{a.arm}: grad norms (before clipping) {norms}; saved {len(params)} tensors to {a.out}")
    dist.destroy_process_group()


def compare(a):
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qurl_verl.r5_optim import group_of

    A, B = (torch.load(os.path.join(os.path.expanduser(d), "params.pt")) for d in (a.ref, a.other))
    ra, rb = (json.load(open(os.path.join(os.path.expanduser(d), "run.json"))) for d in (a.ref, a.other))
    assert A.keys() == B.keys(), "different parameter sets"
    print(f"{rb['arm']} vs {ra['arm']} after {ra['steps']} updates (s = {ra['s']}; {GROUP_HELP})")
    print(f"  grad norms {ra['arm']}: {ra['grad_norms']}\n  grad norms {rb['arm']}: {rb['grad_norms']}"
          f"\n  identical grad norms: {ra['grad_norms'] == rb['grad_norms']}")
    allsame = True
    for g in ("W", "gamma", "rest"):
        names = [n for n in A if group_of(n) == g]
        same = sum(int((A[n] == B[n]).sum()) for n in names)
        tot = sum(A[n].numel() for n in names)
        mx = max(float((A[n] - B[n]).abs().max()) for n in names)
        allsame &= same == tot
        print(f"  {g:<5}: {100 * same / tot:.4f}% of {tot} elements identical; max |difference| {mx:.3e}")
    print("  BIT-IDENTICAL" if allsame else "  DIFFERENT")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=["R4", "R5", "R5o"])
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--s", type=float, default=2.0)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--max-resp", type=int, default=128)
    r.add_argument("--micro", type=int, default=2)
    r.add_argument("--lr", type=float, default=1e-5)
    r.add_argument("--wd", type=float, default=0.01)
    r.add_argument("--clip", type=float, default=0.1)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
