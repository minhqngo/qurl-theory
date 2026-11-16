"""S4, exploratory (2026-11-16, after the third shake-out run; laptop CPU; not pre-registered): BF16 storage in M1's
flip window, with S4's warm-up.

Question: how do the BF16 checkpoint's ties and lattice move M1's q/k/v ratio of 1.5, and the ratio of the addendum's
variant without the ties (s4_m1_ties.py)? The third shake-out run found that step 2's update (about 1e-4 grid steps)
flipped about half of the ties but almost no other weight: 4% of Proposition 4's prediction on the tracked tensors.

Two arms on the same BF16-stored rows and the same noise gradients; one moves 1.5x as far in grid units, as R4, R5 and
R6 do against R3 for q/k/v. Rows as in E2's bf16_update_sim.py (Gaussian, scales log-uniform over 3 octaves, 1536
columns as at 1.5B); each row's nu = eta / Delta log-normal around E2's medians at eta = 1e-6 (1.5B: 1/1200, 7B: 1/790)
with E2's 0.5B spread. Adam (0.9, 0.999) on pure-noise gradients, 16 updates per step, lr 0 at step 1 and then a
10-step linear warm-up, as in S4. Ties: the weights within 1e-4 grid steps of a boundary at the start, as
int8_rollout.py. Per step (sync = step + 1): flip shares of all elements and outside the ties; M1's window is steps
1-20 (syncs 2-21). An FP32-stored control has no lattice.

  python sim_bf16_window.py [--out sim_bf16_window.txt]

CPU only, about 10 minutes with 16 threads. Deterministic for a given torch version (CPU generators, fixed seeds).
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2"))
from e2_dynamic import QMAX, grid  # noqa: E402  (E2's frozen script, imported read-only)

N_COLS, N_ROWS, N_MATS = 1536, 1024, 5          # 7.9M elements
STEPS, UPD = 20, 16
SIGMA = math.log(2.27 / 1.21) / (2 * 1.2816)
TIE_TOL = 1e-4


def lr(step):
    return 0.0 if step == 1 else min(1.0, (step - 1) / 10)


def run(nu_med, bf16, seed):
    g = torch.Generator().manual_seed(seed)
    w0s, etas = [], []
    for _ in range(N_MATS):
        w = torch.randn(N_ROWS, N_COLS, generator=g) * (torch.exp2(torch.rand(N_ROWS, 1, generator=g) * 3.0) * 0.01)
        nu = nu_med * torch.exp(SIGMA * torch.randn(N_ROWS, 1, generator=g))
        w0 = w.to(torch.bfloat16).float() if bf16 else w
        w0s.append(w0)
        etas.append(nu * (w0.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)))
    ties = []
    for w in w0s:
        q, c, _ = grid(w)
        ties.append((q - c).abs() >= 0.5 - TIE_TOL)
    arms = {gain: [w.clone() for w in w0s] for gain in (1.0, 1.5)}
    prev = {gain: [grid(w)[1] for w in w0s] for gain in arms}
    m = [torch.zeros_like(w) for w in w0s]
    v = [torch.zeros_like(w) for w in w0s]
    noise = torch.Generator().manual_seed(seed + 1000)
    out = {gain: [] for gain in arms}
    t = 0
    for step in range(1, STEPS + 1):
        for _ in range(UPD):
            t += 1
            for i in range(N_MATS):
                gr = torch.randn(w0s[i].shape, generator=noise)
                m[i].mul_(0.9).add_(gr, alpha=0.1)
                v[i].mul_(0.999).addcmul_(gr, gr, value=0.001)
                d = (m[i] / (1 - 0.9 ** t)) / ((v[i] / (1 - 0.999 ** t)).sqrt() + 1e-8) * (lr(step) * etas[i])
                for gain, ws in arms.items():
                    ws[i].sub_(d, alpha=gain)
        for gain, ws in arms.items():
            fa = fn = n = nn = 0.0
            for i in range(N_MATS):
                _, c, _ = grid(ws[i])
                ch = c != prev[gain][i]
                fa, fn = fa + float(ch.sum()), fn + float(ch[~ties[i]].sum())
                n, nn = n + ch.numel(), nn + float((~ties[i]).sum())
                prev[gain][i] = c
            out[gain].append((fa / n, fn / nn))
    tie_share = sum(float(x.sum()) for x in ties) / sum(x.numel() for x in ties)
    return out, tie_share


def ratio(out, k, steps):
    return sum(out[1.5][s - 1][k] for s in steps) / sum(out[1.0][s - 1][k] for s in steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = [f"S4 exploratory: BF16 storage in M1's window ({N_MATS * N_ROWS * N_COLS / 1e6:.1f}M elements; Adam on "
             f"noise gradients, {UPD} updates per step; lr 0 at step 1, then a 10-step warm-up; arms 1x and 1.5x in "
             f"grid units; ties within {TIE_TOL} grid steps)"]
    print(lines[0], flush=True)
    for nu_med, size in ((1 / 1200, "1.5B-like rows, nu 8.3e-4"), (1 / 790, "7B-like rows, nu 1.27e-3")):
        for bf16 in (True, False):
            out, ts = run(nu_med, bf16, 0)
            block = [f"== {size}, {'BF16' if bf16 else 'FP32'} storage: ties {ts:.3%}"]
            if bf16:
                block.append("   step  sync   1x all   1x no-ties  1.5x no-ties  ratio no-ties  ratio all")
                for s in (2, 3, 4, 5, 6, 8, 11, 15, 20):
                    x, y = out[1.0][s - 1], out[1.5][s - 1]
                    block.append(f"   {s:4d}  {s + 1:4d}  {x[0]:.4%}  {x[1]:.4%}     {y[1]:.4%}      {y[1] / x[1]:.3f}"
                                 f"          {y[0] / x[0]:.3f}")
            w = range(1, STEPS + 1)
            block.append(f"   window syncs 2-21: ratio all {ratio(out, 0, w):.3f} (M1), "
                         f"without ties {ratio(out, 1, w):.3f}"
                         f"; syncs 12-21 without ties {ratio(out, 1, range(11, 21)):.3f}; 1x flips over the window "
                         f"{sum(out[1.0][s - 1][0] for s in w):.2%}")
            print("\n".join(block), flush=True)
            lines += block
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
