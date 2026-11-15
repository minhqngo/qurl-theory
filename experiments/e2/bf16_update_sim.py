"""E2, exploratory (written after the dynamic results of 2026-11-15; not pre-registered).

Question: is the shortfall of the first one-step pair caused by starting from weights stored in BF16? On the E2 run,
e2_dynamic.py found, without the sync-1 ties, a flip share over prediction (F/P) of 0.912 and a capture slope of 0.931
at pair 1-2, 0.962 and 0.969 at pair 2-3, and 0.988 and 0.987 at pair 65-66.

Synthetic weight rows go through e2_dynamic.py's own statistics (u_stats, tie_masks, pair_stats), once rounded to BF16
as the checkpoint stores them and once in FP32, with the same random draws otherwise.
  Rows      Gaussian, 896 columns (the 0.5B model's hidden size), scales log-uniform over 3 octaves.
  Step size each row's nu = eta / Delta is log-normal with median 1.78e-3 and 10-90% range 1.21-2.27e-3, the 0.5B
            checkpoint's values (static results).
  Updates   Adam (beta1 0.9, beta2 0.999) on pure-noise gradients, four updates per sync as in the run (batch 256,
            mini-batch 64). This moves an element by 2.00, 1.22 and 0.69 eta on average over sync intervals 1, 2 and
            65; the run's fixed-grid predictions over the mean nu give 1.90, 1.15 and 0.60.
For pair 1-2 it also splits F/P (without ties) by code size, which pair_stats does not report.

  python bf16_update_sim.py [--out results/bf16_update_sim.txt]

CPU only, about five minutes. Deterministic for a given torch version (CPU generators, fixed seeds).
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_dynamic import QMAX, grid, pair_stats, tie_masks, u_stats  # noqa: E402  (frozen script, imported read-only)

N_COLS, N_ROWS, N_MATS = 896, 4096, 6            # 22.0M elements per run
U_SYNCS = (1, 2, 3, 65, 66)
PAIRS = ((1, 2), (2, 3), (65, 66))
NU_MEDIAN, NU_SIGMA = 1.78e-3, math.log(2.27 / 1.21) / (2 * 1.2816)
CODE_RANGES = ((0, 15), (16, 63), (64, 126))


def by_code(a, b, ties0):
    """F / P without the sync-1 ties, split by the code at the first sync of the pair (pair_stats' arithmetic)."""
    f = {r: 0.0 for r in CODE_RANGES}
    p = {r: 0.0 for r in CODE_RANGES}
    for name in a:
        qa, ca, _ = grid(a[name])
        qb, cb, _ = grid(b[name])
        keep = (ca.abs() < QMAX) & (cb.abs() < QMAX) & ~ties0[name]
        flip = (cb != ca).double()
        pred = (qb.double() - qa.double()).abs().clamp(max=1.0)
        for lo, hi in CODE_RANGES:
            m = keep & (ca.abs() >= lo) & (ca.abs() <= hi)
            f[(lo, hi)] += float(flip[m].sum())
            p[(lo, hi)] += float(pred[m].sum())
    return {r: (f[r] / p[r] if p[r] > 0 else float("nan"), f[r]) for r in CODE_RANGES}


def run(bf16, seed):
    g = torch.Generator().manual_seed(seed)
    W, eta = {}, {}
    for i in range(N_MATS):
        x = torch.randn(N_ROWS, N_COLS, generator=g)
        w = x * (torch.exp2(torch.rand(N_ROWS, 1, generator=g) * 3.0) * 0.01)
        nu = NU_MEDIAN * torch.exp(NU_SIGMA * torch.randn(N_ROWS, 1, generator=g))
        w0 = w.to(torch.bfloat16).float() if bf16 else w
        W[f"m{i}"] = w0
        eta[f"m{i}"] = nu * (w0.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX))
    m = {k: torch.zeros_like(v) for k, v in W.items()}
    v = {k: torch.zeros_like(w) for k, w in W.items()}
    noise = torch.Generator().manual_seed(seed + 1000)
    snap = {1: {k: w.clone() for k, w in W.items()}}
    ties0 = tie_masks(snap[1])
    out = {"u": {1: u_stats(snap[1])}, "pairs": {}}
    t = 0
    for sync in range(2, max(U_SYNCS) + 1):
        for _ in range(4):
            t += 1
            for k in W:
                gk = torch.randn(W[k].shape, generator=noise)
                m[k].mul_(0.9).add_(gk, alpha=0.1)
                v[k].mul_(0.999).addcmul_(gk, gk, value=0.001)
                W[k] -= eta[k] * (m[k] / (1 - 0.9 ** t)) / ((v[k] / (1 - 0.999 ** t)).sqrt() + 1e-8)
        if sync in U_SYNCS:
            snap[sync] = {k: w.clone() for k, w in W.items()}
            out["u"][sync] = u_stats(snap[sync])
            for a, b in PAIRS:
                if b == sync:
                    out["pairs"][(a, b)] = pair_stats(snap[a], snap[b], ties0)
                    if (a, b) == (1, 2):
                        out["by_code"] = by_code(snap[a], snap[b], ties0)
            for k in [k for k in snap if k != 1 and k not in {a for a, _ in PAIRS if _ > sync}]:
                del snap[k]
    return out


def lines_for(label, r):
    out = []
    for k, s in r["u"].items():
        out.append(f"{label:7s} sync {k:4d}   U1 {s['near_ratio']:.4f}   dev |c|<=15 {s['max_dev_small']:.4f}"
                   f"   ties {s['ties']:.2e}")
    for (a, b), s in r["pairs"].items():
        n_flips = s["F_no_ties"] * s["elements"]
        out.append(f"{label:7s} pair {a:>3d}-{b:<3d} P {s['P']:.3e}   F/P {s['ratio_all']:.4f}   F/P no ties"
                   f" {s['ratio_no_ties']:.4f} (+- {1 / math.sqrt(n_flips):.4f})   cal slope {s['cal_slope']:.4f}"
                   f"   capture {s['capture_slope']:.4f}")
    out.append(f"{label:7s} pair 1-2 F/P no ties by code: " + ", ".join(
        f"|c| {lo}-{hi}: {ratio:.3f} ({int(nf)} flips)" for (lo, hi), (ratio, nf) in r["by_code"].items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    head = [f"E2 exploratory: first updates from BF16 storage ({N_MATS * N_ROWS * N_COLS / 1e6:.1f}M elements per run;"
            " Adam on noise gradients, 4 updates per sync; +- is 1/sqrt(flips))"]
    print(head[0], flush=True)
    lines = list(head)
    for label, bf16, seed in (("BF16", True, 0), ("FP32", False, 0), ("BF16 s1", True, 1)):
        block = lines_for(label, run(bf16, seed))
        print("\n".join(block), flush=True)
        lines += block
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
