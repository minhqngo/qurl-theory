"""E2, exploratory (written after the static results of 2026-11-15; not pre-registered).

Question: are the phase deviations that e2_static.py found on real checkpoints (U2 at |c| <= 15: 0.015-0.052;
codes 16-63: 0.07-0.13; codes 64-126: 0.31-0.42; exact ties 0.07-0.26% of elements) caused by BF16 storage?

Synthetic weight rows (Gaussian, or Student-t with 5 degrees of freedom) go through e2_static.Acc, the frozen
script's own statistics, once as FP32 and once rounded to BF16 as a checkpoint stores them. Each row's scale is
drawn log-uniformly over `spread` octaves: rows whose scales are spread out have absmax values with many different
BF16 mantissas, so their grids sit at many different offsets to the BF16 lattice; rows of equal scale share a few.

  python bf16_lattice_sim.py [--out results/bf16_lattice_sim.txt]

CPU only, about a minute. Deterministic for a given torch version (CPU generator, fixed seeds).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_static import Acc  # noqa: E402  (frozen script, imported read-only)

N_ELEM = 24_000_000
CHUNK = 4_000_000


def run(dist, n, spread, bf16, seed):
    torch.manual_seed(seed)
    acc = Acc()
    n_rows = N_ELEM // n
    per = max(1, CHUNK // n)
    t5 = torch.distributions.StudentT(torch.tensor(5.0))
    for start in range(0, n_rows, per):
        r = min(per, n_rows - start)
        x = torch.randn(r, n) if dist == "gauss" else t5.sample((r, n))
        w = x * (torch.exp2(torch.rand(r, 1) * spread) * 0.01)
        acc.add(w.to(torch.bfloat16) if bf16 else w, 1e-6)
    s = acc.summary()
    bc = s["by_code"]
    noise = (39 / bc["0-15"]["n"]) ** 0.5            # standard deviation of one bin density from counting alone
    return (f"{dist:5s} {n:5d} {spread:6.2f} {seed:4d} {'BF16' if bf16 else 'FP32':>7s} {s['boundary_ratio']:8.4f}"
            f" {s['ties']:9.2e} {bc['0-15']['max_bin_dev']:11.4f} {noise:9.1e} {bc['16-63']['max_bin_dev']:7.3f}"
            f" {bc['64-126']['max_bin_dev']:7.3f} {s['sigma_over_delta_q'][2]:8.1f}")


RUNS = [("gauss", 1536, 3.0, False, 0), ("gauss", 1536, 0.0, False, 0)] + \
       [("gauss", 1536, sp, True, sd) for sp in (3.0, 1.5, 0.75, 0.0) for sd in (0, 1)] + \
       [("t5", 1536, 3.0, True, 0), ("gauss", 896, 3.0, True, 0), ("gauss", 3584, 3.0, True, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = [f"E2 exploratory: BF16 storage and the rounding phase ({N_ELEM:,} elements per run; noise = SD of one"
             f" bin density from counting, |c|<=15)",
             f"{'dist':5s} {'n':>5s} {'spread':>6s} {'seed':>4s} {'storage':>7s} {'U1':>8s} {'ties':>9s}"
             f" {'dev |c|<=15':>11s} {'noise':>9s} {'16-63':>7s} {'64-126':>7s} {'s/D 50%':>8s}"]
    print("\n".join(lines), flush=True)
    for spec in RUNS:
        lines.append(run(*spec))
        print(lines[-1], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
