#!/usr/bin/env python3
"""S4: M1 without the BF16 ties. An exploratory variant of the registered M1, registered on 2026-11-16 before any run on
the rental (experiments/s4/README.md, "Analysis plan", addendum) and frozen with s4_m1_ties.sha256.

Why. In the BF16 checkpoints, 0.15% (1.5B) and 0.12% (7B) of the linear weights sit exactly on an INT8 rounding
boundary (E2). Any update flips about half of them, whatever its size, and afterwards they re-cross the boundary more
often than other weights (E2, at 0.5B: 11% of them at the next update, about 1.6% per update by update 65). The
shake-out measured the first crossing at 1.5B: step 2's update (lr 1e-7) flipped 0.0996% of the q/k/v codes, and on the
tracked tensors 8.6 times Proposition 4's prediction. The ties are the same weights in every arm, and how often they
cross does not depend on the update size, so they add about the same number of flips to every arm and pull M1's
predicted ratios of 1.5 toward 1.

What. M1 exactly as s4_analysis.py computes it (imported unchanged: the same runs, sources, attempt, window, pooling and
bands), with each sync's flips counted only outside the run's ties. The ties are the weights within 1e-4 grid steps of a
rounding boundary at the attempt's first sync, as int8_rollout.py logs them ("ties", "flip_ties"). In R3, R5 and R6
that is the BF16 checkpoint's exact ties and a few weights next to them. In R4 it is the same weights: UAQ's FP32
rescaling moves them by at most about 5e-5 grid steps (at 1.5B, by at most 7.6e-6, and the two sets are identical).
Reported next to M1, with M1's bands. It is exploratory: M1's registered verdicts stand. Also reported for each source:
the share of its flips over the window that were the ties' flips.

Expected (registered before the data; corrected the same day, still before any run on the rental). The ratios predicted
to be 1 (R4/R5; R4/R3 and R5/R3 for the other projections) come out about as in M1. The ratios predicted to be 1.5
(R4/R3, R5/R3 and R6/R3 for q/k/v; R6/R3 for the other projections) come out above M1's, for two reasons. Removing the
ties raises a ratio of 1.5 by about a third of the ties' share of R3's flips. And until training moves the weights off
the BF16 lattice, the lattice holds back small updates more than 1.5 times larger ones, which raises the ratio during
the warm-up. A simulation of the window (sim_bf16_window.py) gave, for a true 1.5, 1.43 (1.5B-like rows) and 1.46
(7B-like) in M1 and 1.54 here.

usage: python s4_m1_ties.py [--results DIR] [--out FILE]
       python s4_m1_ties.py --self-test
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_analysis as A  # noqa: E402  (the frozen analysis, imported read-only)


def pooled_nontie(recs, projs):
    """A.pooled_flips, counting each sync's flips outside the ties only: (value, syncs, the ties' share of the flips).

    Per sync: flipped codes outside the ties over codes outside the ties, summed over layers and projs; then the mean
    over the window's syncs. With no ties this is A.pooled_flips. (None, 0, None) if a sync lacks the tie log."""
    vals, tie_flips, all_flips = [], 0.0, 0.0
    for r in recs:
        if r["sync"] not in A.FLIP_SYNCS:
            continue
        if r.get("ties") is None or r.get("flip_ties") is None:
            return None, 0, None
        num = den = 0.0
        for p in projs:
            n = r["codes_per_layer"][p]
            for f, t, g in zip(r["flip"][p], r["ties"][p], r["flip_ties"][p]):
                if f is None or t is None:
                    return None, 0, None
                k = t * n                  # the layer's ties
                ft = (g or 0.0) * k        # their flips (g is None in a layer without ties)
                num += f * n - ft
                den += n - k
                tie_flips += ft
                all_flips += f * n
        vals.append(num / den)
    return (statistics.fmean(vals) if vals else None), len(vals), (tie_flips / all_flips if all_flips else None)


def m1_sources(results):
    """M1's sources at each size, chosen exactly as s4_analysis.report chooses them: {size: {arm: (values, tag)}}."""
    full, mech = {}, {}
    for size in A.SIZES:
        for arm in A.ARMS:
            v, status = A.load_full(Path(results) / f"s4_{size}_{arm}_s1")
            if status != "missing":
                full[(size, arm)] = v
            v, status = A.load_mech(Path(results) / f"s4m_{size}_{arm}_s1")
            if status != "missing":
                mech[(size, arm)] = v
    return {size: A.m1_sources(full, mech, size) for size in A.SIZES}


def m1_ties(src, results, size):
    out = [f"M1 without the ties, {size} (syncs {A.FLIP_SYNCS.start}-{A.FLIP_SYNCS.stop - 1}; M1's sources: "
           + (", ".join(f"{a} {t}" for a, (_, t) in src.items()) or "none") + ")"]
    nt = {}
    for arm, (v, tag) in src.items():
        recs = A.load_e3(Path(results) / tag)
        q, nq, sq = pooled_nontie(recs, A.QKV) if recs else (None, 0, None)
        o, no, so = pooled_nontie(recs, A.OTHER) if recs else (None, 0, None)
        if q is None or o is None or min(nq, no) < A.MIN_FLIP_SYNCS:
            out.append(f"  {arm}: no tie log over the window ({tag})")
            continue
        nt[arm] = {"flip_qkv": q, "flip_other": o}
        out.append(f"  {arm}: flip shares without the ties: q/k/v {q:.4%}, other {o:.4%} (M1: {v['flip_qkv']:.4%}, "
                   f"{v['flip_other']:.4%}); the ties' share of the flips: q/k/v {sq:.1%}, other {so:.1%}")
    if "R3" not in nt:
        return out + ["  no verdict: no R3 tie log"]
    m1 = {arm: v for arm, (v, _) in src.items()}
    for arm in ("R4", "R5", "R6"):
        if arm not in nt:
            out.append(f"  {arm}/R3: not available")
            continue
        for key, label, pred in (("flip_qkv", "q/k/v", A.S), ("flip_other", "other", A.S if arm == "R6" else 1.0)):
            out.append(f"  {arm}/R3 {label}  {A.ratio_verdict(nt[arm][key] / nt['R3'][key], pred)}"
                       f"  [M1: {m1[arm][key] / m1['R3'][key]:.3f}]")
    if "R4" in nt and "R5" in nt:
        for key, label in (("flip_qkv", "q/k/v"), ("flip_other", "other")):
            out.append(f"  R4/R5 {label}  {A.ratio_verdict(nt['R4'][key] / nt['R5'][key], 'R4/R5')}"
                       f"  [M1: {m1['R4'][key] / m1['R5'][key]:.3f}]")
    return out


def report(results):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sha_a = hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest()
    out = [f"S4, M1 without the ties (exploratory, registered 2026-11-16; s4_m1_ties.py sha256 {sha}; "
           f"s4_analysis.py sha256 {sha_a})", f"results: {results}", ""]
    src = m1_sources(results)
    for size in A.SIZES:
        out += m1_ties(src[size], results, size) + [""]
    return "\n".join(out)


def self_test():
    """Synthetic runs with planted ties: M1 is diluted toward 1, the variant recovers the planted ratios exactly."""
    layers, t = 2, 0.0015
    codes = {"q_proj": 4000, "k_proj": 1000, "v_proj": 1000, "o_proj": 4000, "gate_proj": 9000, "up_proj": 9000,
             "down_proj": 9000}

    def lr(sync):          # the update of step sync - 1: lr 0 at step 1, then a 10-step warm-up
        return min(1.0, (sync - 2) / 10)

    def tie_flip(sync):    # half of the ties at the first nonzero update, then E2's re-crossing
        return 0.0 if sync < 3 else 0.5 if sync == 3 else 0.157 / math.sqrt(sync - 2)

    def write_run(root, arm, ties=True):
        run = Path(root) / f"s4m_1.5b_{arm}_s1"
        run.mkdir(parents=True)
        recs = []
        for sync in range(1, 26):
            flip, ft = {}, {}
            for p in codes:
                gain = 1.5 if (arm == "R6" or (arm in ("R4", "R5") and p in A.QKV)) else 1.0
                base = 1e-3 * gain * lr(sync)                           # outside the ties
                g = [tie_flip(sync), 0.0 if p == "k_proj" else tie_flip(sync)]
                tl = [t, 0.0 if p == "k_proj" else t]                   # layer 1 of k_proj: no ties
                flip[p] = None if sync == 1 else [(1 - tl[i]) * base + tl[i] * g[i] for i in range(layers)]
                ft[p] = None if sync == 1 else [g[i] if tl[i] > 0 else None for i in range(layers)]
            rec = {"attempt": "1", "start_step": 0, "sync": sync, "layers": layers, "codes_per_layer": codes,
                   "flip": flip}
            if ties:
                rec["ties"] = {p: [t, 0.0 if p == "k_proj" else t] for p in codes}
                rec["flip_ties"] = ft
            recs.append(rec)
        (run / "e3_int8.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (run / "metrics.jsonl").write_text("".join(json.dumps({"step": s, "data": {A.K3: 1e-3}}) + "\n"
                                                   for s in range(1, 26)))

    with tempfile.TemporaryDirectory() as root:
        for arm in A.ARMS:
            write_run(root, arm)
        src = m1_sources(root)["1.5b"]
        assert set(src) == set(A.ARMS) and all(tag == f"s4m_1.5b_{a}_s1" for a, (_, tag) in src.items()), src
        nt = {a: {k: pooled_nontie(A.load_e3(Path(root) / tag), ps)[0] for k, ps in
                  (("flip_qkv", A.QKV), ("flip_other", A.OTHER))} for a, (_, tag) in src.items()}
        m1 = {a: v for a, (v, _) in src.items()}
        for arm, gq, go in (("R4", 1.5, 1.0), ("R5", 1.5, 1.0), ("R6", 1.5, 1.5)):
            assert abs(nt[arm]["flip_qkv"] / nt["R3"]["flip_qkv"] - gq) < 1e-12, (arm, nt[arm], nt["R3"])
            assert abs(nt[arm]["flip_other"] / nt["R3"]["flip_other"] - go) < 1e-12, (arm, nt[arm], nt["R3"])
            assert m1[arm]["flip_qkv"] / m1["R3"]["flip_qkv"] < 1.5 - 0.02    # M1: diluted toward 1
        assert abs(m1["R4"]["flip_qkv"] / m1["R5"]["flip_qkv"] - 1) < 1e-12
        text = "\n".join(m1_ties(src, root, "1.5b"))
        assert "R4/R3 q/k/v  1.500" in text and "R4/R5 q/k/v  1.000" in text and "NOT" not in text, text
        # Without ties the variant is M1 itself.
        r = A.load_e3(Path(root) / "s4m_1.5b_R3_s1")
        for rec in r:
            rec["ties"] = {p: [0.0] * layers for p in codes}
            rec["flip_ties"] = {p: [None] * layers for p in codes}
        for ps in (A.QKV, A.OTHER):
            assert abs(pooled_nontie(r, ps)[0] - A.pooled_flips(r, ps)[0]) < 1e-15
    with tempfile.TemporaryDirectory() as root:   # a run without the tie log: reported, no verdict from it
        write_run(root, "R3", ties=False)
        write_run(root, "R4")
        text = "\n".join(m1_ties(m1_sources(root)["1.5b"], root, "1.5b"))
        assert "R3: no tie log" in text and "no verdict: no R3 tie log" in text, text
    print("self-test passed: planted ties dilute M1 and the variant recovers the planted ratios; without ties the "
          "variant is M1; a run without the tie log gives no verdict")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    text = report(a.results) + "\n"
    sys.stdout.write(text)
    out = Path(a.out) if a.out else Path(a.results) / "s4_m1_ties.txt"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
