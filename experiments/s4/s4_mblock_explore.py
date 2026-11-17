"""S4, exploratory (not registered): a closer look at the M1 and early-M2 data of the mechanism runs.

  python s4_mblock_explore.py [--results results] [--out results/s4_mblock_explore.txt]

1. M1 and its tie-free variant (the addendum) over steps 11-20 (syncs 12-21), besides the registered window (syncs
   2-21, which reproduces s4_analysis.py and s4_m1_ties.py), and the tie-free ratios step by step, to set beside
   sim_bf16_window.py. Also the ties' own flips per sync, which the addendum assumes are the same in every arm.
2. Two runs of one arm: the full 7B R4 run against its mechanism run, over the syncs and steps both have logged.
   Their first 20 steps have the same settings, so the difference is run-to-run variation.
3. The mismatch k3 step by step, and its ratios over steps 1-10 and 11-20.
It reads the mechanism runs (s4m_<size>_<arm>_s1) and imports s4_analysis.py and s4_m1_ties.py read-only.
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_analysis as A  # noqa: E402  (frozen, imported read-only)
import s4_m1_ties as T  # noqa: E402  (frozen, imported read-only)

GROUPS = (("q/k/v", A.QKV), ("other", A.OTHER))
LATE = range(12, 22)   # syncs of steps 11-20
RATIOS = (("R4", "R3", "q/k/v"), ("R5", "R3", "q/k/v"), ("R6", "R3", "q/k/v"), ("R6", "R3", "other"),
          ("R4", "R3", "other"), ("R5", "R3", "other"), ("R4", "R5", "q/k/v"), ("R4", "R5", "other"))


def mech(size, arm):
    return f"s4m_{size}_{arm}_s1"


def shares(recs, syncs):
    """{(group, with ties?): code-weighted share per sync, averaged over the syncs}, pooled as the frozen scripts do."""
    recs = [r for r in recs if r["sync"] in syncs]
    out = {}
    for g, projs in GROUPS:
        out[(g, True)] = A.pooled_flips(recs, projs)[0]
        out[(g, False)] = T.pooled_nontie(recs, projs)[0]
    return out


def tie_flips(recs, projs):
    """Flips among the ties per sync, as a share of all codes and of the ties, averaged over the window."""
    of_codes, of_ties = [], []
    for r in recs:
        if r["sync"] not in A.FLIP_SYNCS:
            continue
        flips = codes = ties = 0.0
        for p in projs:
            n = r["codes_per_layer"][p]
            for t, g in zip(r["ties"][p], r["flip_ties"][p]):
                flips += (g or 0.0) * t * n
                ties += t * n
                codes += n
        of_codes.append(flips / codes)
        of_ties.append(flips / ties)
    return statistics.fmean(of_codes), statistics.fmean(of_ties)


def ratio(a, b):
    return a / b if a is not None and b else None


def fmt(x, spec=".3f"):
    return "  -  " if x is None else format(x, spec)


def m1_part(results, size, out):
    recs = {arm: A.load_e3(results / mech(size, arm)) for arm in A.ARMS}
    if any(r is None for r in recs.values()):
        out.append(f"{size}: a mechanism run is missing")
        return
    for label, syncs in (("syncs 2-21, the registered window", A.FLIP_SYNCS), ("steps 11-20 (syncs 12-21)", LATE)):
        s = {arm: shares(recs[arm], syncs) for arm in A.ARMS}
        out.append(f"{size}, {label}: M1 | without the ties")
        for num, den, g in RATIOS:
            out.append(f"  {num}/{den} {g:5s}  {fmt(ratio(s[num][(g, True)], s[den][(g, True)]))} | "
                       f"{fmt(ratio(s[num][(g, False)], s[den][(g, False)]))}")
    out.append(f"{size}, the ties' flips per sync over syncs 2-21 (share of all codes; share of the ties):")
    for arm in A.ARMS:
        cells = [f"{g} {100 * c:.4f}%; {100 * t:.1f}%" for g, projs in GROUPS for c, t in [tie_flips(recs[arm], projs)]]
        out.append(f"  {arm}: " + " | ".join(cells))
    out.append(f"{size}, without the ties, step by step: R4/R3 R5/R3 R6/R3 q/k/v | R6/R3 other | R4/R5 q/k/v, other")
    for sync in A.FLIP_SYNCS:
        s = {arm: shares(recs[arm], [sync]) for arm in A.ARMS}
        cells = [ratio(s[num][(g, False)], s[den][(g, False)]) for num, den, g in
                 (RATIOS[0], RATIOS[1], RATIOS[2], RATIOS[3], RATIOS[6], RATIOS[7])]
        out.append(f"  step {sync - 1:2d}: " + " ".join(fmt(c) for c in cells)
                   + f"   (R3's q/k/v share {100 * s['R3'][('q/k/v', False)]:.4f}%)")


def k3_part(results, size, out):
    by = {arm: A.load_metrics(results / mech(size, arm)) for arm in A.ARMS}
    k = {arm: [by[arm].get(st, {}).get(A.K3) for st in A.EARLY_STEPS] for arm in A.ARMS}
    if any(v is None for arm in A.ARMS for v in k[arm]):
        out.append(f"{size}: k3 is missing for some step")
        return
    out.append(f"{size}, k3 x 1e3 per step (R3 R4 R5 R6) and R4/R3 R5/R3 R6/R3 R4/R5:")
    for i, st in enumerate(A.EARLY_STEPS):
        out.append(f"  step {st:2d}: " + " ".join(f"{1e3 * k[arm][i]:6.3f}" for arm in A.ARMS) + "   "
                   + " ".join(f"{k[a][i] / k[b][i]:.3f}" for a, b in (("R4", "R3"), ("R5", "R3"), ("R6", "R3"),
                                                                    ("R4", "R5"))))
    for lo, hi in ((1, 20), (1, 10), (11, 20)):
        m = {arm: statistics.fmean(k[arm][lo - 1:hi]) for arm in A.ARMS}
        out.append(f"  steps {lo}-{hi}: " + "  ".join(f"{a}/{b} {m[a] / m[b]:.3f}" for a, b in
                                                     (("R4", "R3"), ("R5", "R3"), ("R6", "R3"), ("R4", "R5"))))


def two_runs(results, out, full="s4_7b_R4_s1", mechanism="s4m_7b_R4_s1"):
    a, b = A.load_e3(results / full), A.load_e3(results / mechanism)
    if not a or not b:
        out.append(f"two runs of one arm: {full} or {mechanism} has no flip log yet")
        return
    syncs = sorted({r["sync"] for r in a} & {r["sync"] for r in b} & set(A.FLIP_SYNCS))
    if not syncs:
        out.append("two runs of one arm: no common sync in the window yet")
        return
    sa, sb = shares(a, syncs), shares(b, syncs)
    out.append(f"two runs of one arm: {full} / {mechanism}, syncs {syncs[0]}-{syncs[-1]} (M1 | without the ties)")
    for g, _ in GROUPS:
        with_ties, without = ratio(sa[(g, True)], sb[(g, True)]), ratio(sa[(g, False)], sb[(g, False)])
        out.append(f"  {g:5s}  {fmt(with_ties)} | {fmt(without)}")
    ka, kb = A.load_metrics(results / full), A.load_metrics(results / mechanism)
    steps = [st for st in A.EARLY_STEPS if A.K3 in ka.get(st, {}) and A.K3 in kb.get(st, {})]
    if steps:
        ma = statistics.fmean(ka[st][A.K3] for st in steps)
        mb = statistics.fmean(kb[st][A.K3] for st in steps)
        out.append(f"  k3 over steps {steps[0]}-{steps[-1]}: {1e3 * ma:.3f} / {1e3 * mb:.3f} x 1e-3 = {ma / mb:.3f}")


def report(results):
    out = ["S4, exploratory look at the mechanism runs (not registered; s4_mblock_explore.py)",
           f"results: {results}", ""]
    for size in A.SIZES:
        m1_part(results, size, out)
        out.append("")
        k3_part(results, size, out)
        out.append("")
    two_runs(results, out)
    return "\n".join(out) + "\n"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = Path(a.results)
    text = report(results)
    out = Path(a.out) if a.out else results / "s4_mblock_explore.txt"
    out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
