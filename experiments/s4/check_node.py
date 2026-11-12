#!/usr/bin/env python3
"""S4 node check under load, with the venv's PyTorch. setup_s4.sh runs it as soon as the venv exists; gates.sh G1
records it. Every GPU works at once, so power and cooling are loaded as in the runs:
  - memory: nearly all free memory filled with a pattern, read back, inverted, read back
  - BF16 matmul throughput (8192^3, sustained over --seconds) and INT8 matmul throughput (torch._int_mm, cuBLASLt)
  - pinned host <-> GPU copy bandwidth (the INT8 refit keeps its code history in pinned host memory)
  - NCCL over all GPUs: an all-reduce's result, then the bus bandwidth of all-reduce, all-gather and reduce-scatter
    (FSDP's collectives) at 64 MiB and 1 GiB, computed as nccl-tests does
nvidia-smi is sampled every 2 s meanwhile: SM clock, temperature and power under load.
Exit 0: all good; 1: warnings (the runs work, but slower or worth a look); 2: failure (the runs cannot work).
  ~/venvs/verl/bin/python ~/qurl/s4/check_node.py --out ~/qurl/s4/results/node/check_node.txt
Absolute thresholds are for H100 SXM: BF16 at least 550 TFLOPS per GPU, all-reduce at least 150 GB/s (NVLink; it
reaches 350-480). Other GPUs are only compared with each other.
"""
import argparse
import datetime
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

N = 8192
GIB = 1 << 30


def sustained(fn, seconds, batch):
    """Calls of fn per second over the second half of about `seconds` of back-to-back batches (clocks settled)."""
    import torch
    times = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(times) < 4:
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / 1000)
    half = times[len(times) // 2:]
    return batch * len(half) / sum(half)


def memory_test(dev):
    """Fill all free memory but 2 GiB in 1 GiB blocks; read back the pattern, then its complement."""
    import torch
    free, total = torch.cuda.mem_get_info(dev)
    blocks, bad = [], 0
    for k in range(max(0, int((free - 2 * GIB) // GIB))):
        t = torch.empty(GIB // 4, dtype=torch.int32, device=dev)
        t.fill_(0x5A5A5A5A ^ (k * 40503 % 65536))
        blocks.append(t)
    torch.cuda.synchronize(dev)
    for k, t in enumerate(blocks):
        v = 0x5A5A5A5A ^ (k * 40503 % 65536)
        bad += int(torch.count_nonzero(t != v))
        t.bitwise_not_()
        bad += int(torch.count_nonzero(t != ~v))
    n, t = len(blocks), None
    blocks.clear()
    torch.cuda.empty_cache()
    return {"mem_free_gib": free / GIB, "mem_total_gib": total / GIB, "mem_tested_gib": n, "mem_bad": bad}


def worker(rank, world, port, tmp, seconds):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    res = {"rank": rank, "name": torch.cuda.get_device_name(dev)}

    def save():  # after every phase, so that a crash later still leaves the earlier results
        with open(os.path.join(tmp, f"rank{rank}.json"), "w") as f:
            json.dump(res, f)

    # Gloo for the barriers (all GPUs load together); NCCL is the thing under test, in its own group.
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=10))
    try:
        res.update(memory_test(dev))
    except Exception as e:  # noqa: BLE001 - reported, not raised
        res["mem_error"] = repr(e)[:300]
        torch.cuda.empty_cache()
    save()
    dist.barrier()
    a = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    c = torch.empty(N, N, device=dev, dtype=torch.bfloat16)
    res["bf16_tflops"] = 2 * N ** 3 * sustained(lambda: torch.mm(a, b, out=c), seconds, 20) / 1e12
    del a, b, c
    try:
        ai = torch.randint(-128, 128, (N, N), device=dev, dtype=torch.int8)
        bi = torch.randint(-128, 128, (N, N), device=dev, dtype=torch.int8).t()  # column-major, cuBLASLt's layout
        res["int8_tops"] = 2 * N ** 3 * sustained(lambda: torch._int_mm(ai, bi), seconds / 4, 20) / 1e12
        del ai, bi
    except Exception as e:  # noqa: BLE001
        res["int8_error"] = repr(e)[:200]
    save()
    dist.barrier()
    h = torch.empty(GIB, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(GIB, dtype=torch.uint8, device=dev)
    res["h2d_gbs"] = GIB * sustained(lambda: d.copy_(h, non_blocking=True), 3, 2) / 1e9
    res["d2h_gbs"] = GIB * sustained(lambda: h.copy_(d, non_blocking=True), 3, 2) / 1e9
    del h, d
    save()
    dist.barrier()
    try:
        g = dist.new_group(backend="nccl", timeout=datetime.timedelta(minutes=3))
        x = torch.full((1 << 20,), float(rank + 1), device=dev)
        dist.all_reduce(x, group=g)
        res["nccl_correct"] = bool(torch.all(x == world * (world + 1) / 2).item())
        bw = {}
        for size in (64 << 20, GIB):
            chunk = torch.zeros(size // 2 // world, device=dev, dtype=torch.bfloat16)
            full = torch.zeros(chunk.numel() * world, device=dev, dtype=torch.bfloat16)
            nbytes = full.numel() * 2
            ops = {"all-reduce": (lambda: dist.all_reduce(full, group=g), 2 * (world - 1) / world),
                   "all-gather": (lambda: dist.all_gather_into_tensor(full, chunk, group=g), (world - 1) / world),
                   "reduce-scatter": (lambda: dist.reduce_scatter_tensor(chunk, full, group=g), (world - 1) / world)}
            for name, (fn, factor) in ops.items():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                bw[f"{name} {size >> 20} MiB"] = nbytes * sustained(fn, 2, 5) * factor / 1e9
            del chunk, full
        res["nccl_busbw_gbs"] = bw
        dist.destroy_process_group(g)
    except Exception as e:  # noqa: BLE001
        res["nccl_error"] = repr(e)[:300]
    save()
    dist.destroy_process_group()


def num(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


class Sampler(threading.Thread):
    """nvidia-smi every 2 s: index, utilization, SM clock, temperature, power."""
    FIELDS = "index,utilization.gpu,clocks.sm,temperature.gpu,power.draw"

    def __init__(self):
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.rows = []

    def run(self):
        while not self.halt.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.rows += [[num(x) for x in line.split(",")] for line in out.strip().splitlines()]
            except (OSError, subprocess.SubprocessError):
                pass
            self.halt.wait(2)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def evaluate(res, rows, world, err):
    lines = []

    def v(level, text):
        lines.append(f"{level} {text}")

    if err:
        v("FAIL", f"a worker stopped: {err}")
    bad = [r for r in res if r.get("mem_bad", 1) != 0 or "mem_error" in r]
    if bad:
        v("FAIL", "memory test failed: " + "; ".join(
            f"GPU{r['rank']} " + (r.get("mem_error") or (f"{r['mem_bad']} wrong words" if "mem_bad" in r else "not run"))
            for r in bad))
    else:
        t = [r["mem_tested_gib"] for r in res]
        v("PASS", f"memory: {min(t)}-{max(t)} GiB per GPU written and read back (pattern and complement), no error")
    busy = [f"GPU{r['rank']} {r['mem_total_gib'] - r['mem_free_gib']:.1f} GiB" for r in res
            if "mem_free_gib" in r and r["mem_free_gib"] < 0.95 * r["mem_total_gib"]]
    if busy:
        v("WARN", "memory already in use before the test: " + ", ".join(busy))

    tf = [r.get("bf16_tflops") for r in res]
    if None in tf:
        v("FAIL", "BF16 matmul did not finish on GPU " + ", ".join(str(r["rank"]) for r in res if "bf16_tflops" not in r))
    else:
        med = statistics.median(tf)
        slow = [f"GPU{r['rank']} {r['bf16_tflops']:.0f}" for r in res if r["bf16_tflops"] < 0.9 * med]
        low = all("H100" in r["name"] and "HBM3" in r["name"] for r in res) and med < 550
        v("WARN" if slow or low else "PASS",
          f"BF16 matmul {N}^3, all GPUs at once: median {med:.0f} TFLOPS (range {min(tf):.0f}-{max(tf):.0f})"
          + (f"; below 90% of the median: {', '.join(slow)} (the slowest GPU paces every step)" if slow else "")
          + ("; an H100 SXM should reach about 550 or more" if low else ""))
    i8 = [r.get("int8_tops") for r in res]
    if None in i8:
        v("WARN", "INT8 matmul (torch._int_mm) failed: " + next((r["int8_error"] for r in res if "int8_error" in r), "?"))
    else:
        v("INFO", f"INT8 matmul {N}^3 (cuBLASLt): median {statistics.median(i8):.0f} TOPS "
                  f"(range {min(i8):.0f}-{max(i8):.0f}); vLLM's INT8 kernels are CUTLASS, not timed here")
    for key, label in (("h2d_gbs", "host to GPU"), ("d2h_gbs", "GPU to host")):
        vals = [r.get(key) for r in res]
        if None in vals:
            v("FAIL", f"pinned copies {label} did not finish")
            continue
        med = statistics.median(vals)
        low = [f"GPU{r['rank']} {r[key]:.1f}" for r in res if r[key] < max(10, 0.5 * med)]
        v("WARN" if low else "PASS", f"pinned copies {label}, all GPUs at once: median {med:.1f} GB/s "
                                     f"(range {min(vals):.1f}-{max(vals):.1f})" + (f"; low: {', '.join(low)}" if low else ""))

    # Samples at >= 90% utilization are the load phases. Utilization is averaged over the last interval but the clock is
    # instantaneous, so a sample just after a phase can pair high utilization with an idle clock: medians, not minima.
    load = {}
    for row in rows:
        if len(row) == 5 and not math.isnan(row[0]) and row[1] >= 90:
            load.setdefault(int(row[0]), []).append(row)
    if load:
        clk = {g: statistics.median(r[2] for r in rs) for g, rs in load.items()}
        temp = {g: max(r[3] for r in rs) for g, rs in load.items()}
        pw = [r[4] for rs in load.values() for r in rs if not math.isnan(r[4])]
        med = statistics.median(clk.values())
        slow = [f"GPU{g} {c:.0f} MHz" for g, c in sorted(clk.items()) if c < 0.85 * med]
        hot = [f"GPU{g} {t:.0f} C" for g, t in sorted(temp.items()) if t >= 87]
        v("WARN" if slow or hot else "PASS",
          f"under load (nvidia-smi): median SM clock {min(clk.values()):.0f}-{max(clk.values()):.0f} MHz across GPUs, "
          f"hottest {max(temp.values()):.0f} C" + (f", highest power {max(pw):.0f} W" if pw else "")
          + (f"; throttled: {', '.join(slow)}" if slow else "") + (f"; hot: {', '.join(hot)}" if hot else ""))
    else:
        v("INFO", "under load (nvidia-smi): no sample at >= 90% utilization")

    if any("nccl_error" in r for r in res) or not all("nccl_busbw_gbs" in r for r in res):
        v("FAIL", "NCCL did not finish: " + next((r["nccl_error"] for r in res if "nccl_error" in r),
                                                 "a worker stopped before it"))
    elif not all(r.get("nccl_correct") for r in res):
        v("FAIL", "NCCL all-reduce returned a wrong sum")
    elif world == 1:
        v("INFO", "NCCL: one GPU; all-reduce correct, bandwidth not measured")
    else:
        bw = res[0]["nccl_busbw_gbs"]
        ar = bw["all-reduce 1024 MiB"]
        v("WARN" if ar < 150 else "PASS",
          f"NCCL over {world} GPUs: all-reduce correct; bus bandwidth at 1 GiB: all-reduce {ar:.0f}, all-gather "
          f"{bw['all-gather 1024 MiB']:.0f}, reduce-scatter {bw['reduce-scatter 1024 MiB']:.0f} GB/s"
          + ("; below 150 GB/s, so NVLink looks unused (H100 SXM reaches 350-480)" if ar < 150 else ""))

    table = ["GPU  name                            memory GiB  BF16 TFLOPS  INT8 TOPS  H2D GB/s  D2H GB/s"]
    for r in res:
        f = lambda k, p=0: f"{r[k]:.{p}f}" if k in r else "-"  # noqa: E731
        table.append(f"{r['rank']:<4} {r.get('name', '?')[:31]:<31} {f('mem_tested_gib'):>10} {f('bf16_tflops'):>12} "
                     f"{f('int8_tops'):>10} {f('h2d_gbs', 1):>9} {f('d2h_gbs', 1):>9}")
    if load:
        table += ["", "nvidia-smi GPU  median SM MHz  hottest C  highest W  samples   (at >= 90% utilization)"]
        for g in sorted(load):
            p = [r[4] for r in load[g] if not math.isnan(r[4])]
            table.append(f"{g:<15} {clk[g]:>13.0f} {temp[g]:>10.0f} {(max(p) if p else math.nan):>10.0f} "
                         f"{len(load[g]):>8}")
    if world > 1 and "nccl_busbw_gbs" in res[0]:
        table += ["", "NCCL bus bandwidth, GB/s"] + [f"  {k:<26} {x:7.1f}" for k, x in res[0]["nccl_busbw_gbs"].items()]
    return lines, table


def main():
    ap = argparse.ArgumentParser(description="S4 node check under load (see the module docstring).")
    ap.add_argument("--out", required=True, help="report file")
    ap.add_argument("--seconds", type=float, default=20, help="BF16 load per GPU, seconds (default 20)")
    ap.add_argument("--gpus", type=int, default=0, help="number of GPUs (default: all)")
    a = ap.parse_args()
    import torch
    import torch.multiprocessing as mp
    t0 = time.time()
    world = a.gpus or torch.cuda.device_count()
    head = f"# check_node.py {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} on {socket.gethostname()}"
    if world < 1:
        lines, table = ["FAIL PyTorch sees no GPU"], []
    else:
        tmp = tempfile.mkdtemp(prefix="check_node_")
        sampler = Sampler()
        sampler.start()
        err = None
        try:
            mp.spawn(worker, args=(world, free_port(), tmp, a.seconds), nprocs=world, join=True)
        except Exception as e:  # noqa: BLE001 - a worker died; what it saved is still reported
            msg = str(e).strip().splitlines()  # a worker's traceback: the error is its last "...Error: ..." line
            errs = [m for m in msg if re.match(r"^[\w.]+(Error|Exception)\b", m)]
            err = (errs[-1] if errs else msg[-1] if msg else repr(e))[:300]
        sampler.halt.set()
        sampler.join(timeout=15)
        res = []
        for r in range(world):
            p = os.path.join(tmp, f"rank{r}.json")
            res.append(json.load(open(p)) if os.path.exists(p) else {"rank": r})
        lines, table = evaluate(res, sampler.rows, world, err)
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        lines.insert(0, f"INFO {world} GPU(s): {', '.join(sorted({r.get('name', '?') for r in res}))}; torch "
                        f"{torch.__version__}, CUDA {torch.version.cuda}, NCCL {nccl}; {time.time() - t0:.0f} s")
    status = 2 if any(x.startswith("FAIL") for x in lines) else 1 if any(x.startswith("WARN") for x in lines) else 0
    text = "\n".join([f"{head}: exit {status}", *lines, "", *table]) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    print(text, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
