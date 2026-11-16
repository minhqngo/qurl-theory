"""Minimal NCCL check over the visible GPUs, run by torchrun (nccl_diag.sh): an all-reduce whose sum is checked, then
an all-gather and a reduce-scatter (FSDP's collectives), 256 MiB per GPU. Every rank prints one line when it is done.
    torchrun --nproc_per_node=8 nccl_test.py
"""
import datetime
import os

import torch
import torch.distributed as dist

rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=2))
x = torch.full((64 * 2**20,), float(rank + 1), device="cuda")
dist.all_reduce(x)
ok = bool(torch.all(x == world * (world + 1) / 2).item())
chunk = torch.ones(64 * 2**20 // world, device="cuda")
full = torch.empty(chunk.numel() * world, device="cuda")
dist.all_gather_into_tensor(full, chunk)
dist.reduce_scatter_tensor(chunk, full)
torch.cuda.synchronize()
dist.barrier()
print(f"rank {rank} (local GPU {local} of {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}): all-reduce correct {ok}",
      flush=True)
dist.destroy_process_group()
