#!/usr/bin/env python3
"""Pre-registered analysis for S4, the 5-day plan (experiments/s4/README.md, "Analysis plan"). Freeze before gates.sh.

Runs (all seed 1):
  mechanism runs  results/s4m_<size>_<arm>_s1/   25 steps, no evaluation; R3 (gates.sh), R4, R5, R6 at 1.5B; R3, R6 at 7B
  full runs       results/s4_<size>_<arm>_s1/    200 steps and eval.json; 7B R4, R5, then R3, R6; then 1.5B R4, R5
  base models     results/base_<size>/eval.json
A full run counts only when complete (all 200 training steps logged and eval.json written); a mechanism run when its
code-flip window is complete. No run is excluded for its outcome. The seed-based H1 test is S3's (h1_analysis.py,
five pairs at 0.5B); S4 adds the paper's scale and the mechanism.

Quantities
  flip_qkv, flip_other  code-flip share of the INT8 rollout weights over syncs 2-21 (the updates of steps 1-20), pooled
                        over layers and weighted by code counts: q/k/v projections, and o/gate/up/down. Taken from the
                        first attempt that starts at step 0 and logged at least 10 syncs in the window.
  k3_early              mean rollout-trainer mismatch (rollout_corr/k3_kl) over steps 1-20
  k3                    mean mismatch over steps 1-200 (full runs)
  math, aime            MATH-500 Avg@4 and AIME 2024 Avg@32 of the final checkpoint (BF16, eval.py), in points;
                        math_g, aime_g greedy; auc, early: mean training score over steps 1-200 and 1-20, x 100

Registered analyses
  M1  Flip-share ratios at each size. Source per arm: the full run if its window is complete, else the mechanism run
      (the first 20 steps of both have identical settings). Theorem 2 with Proposition 4's small-step regime predicts
        q/k/v:  R4/R3 = R5/R3 = R6/R3 = s = 1.5, and R4/R5 = 1
        other:  R4/R3 = R5/R3 = 1, R6/R3 = 1.5, and R4/R5 = 1
      "Consistent" if a ratio lies in [1.35, 1.65] (prediction 1.5), [0.90, 1.10] (prediction 1 against R3), or
      [0.95, 1.05] (R4/R5). The decisive pair is R5 against R4: R5 never touches the quantizer.
  M2  UAQ and the mismatch, at each size: the k3 ratio R4/R3, early (k3_early, M1's sources) and over the whole run
      (k3, full runs, where both exist). "Reduced" below 0.80, "not meaningfully changed" in [0.80, 1.25], "increased"
      above 1.25. QuRL's mechanism predicts a reduction; Theorem 1 and S1 (7B INT8 W8A8 at step 0: 1.013-1.022) none.
Descriptive
  Full-run pairs at each size: R4 - R5 (H1 at the paper's scale), R4 - R3 (UAQ's effect), R6 - R3 (the paper's
  control), for every endpoint. For math and aime also a paired bootstrap over problems (10,000 resamples, seed 0):
  an interval for evaluation noise only, since one seed per arm cannot show seed-to-seed variation.
  The paper's Table 4 (AIME 2024 Avg@32, INT8: s = 1 30.63, s = 1.5 31.25, lr 1.5e-6 29.06); the base models.
Exploratory (fixed at registration; not tests)
  A full run "stalled" if its mean training score over steps 181-200 is below its mean over steps 1-10.

usage: python s4_analysis.py [--results DIR] [--out FILE]
       python s4_analysis.py --self-test
"""
import argparse
import glob
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

# ---- registered constants ----
N_STEPS = 200
FLIP_SYNCS = range(2, 22)
EARLY_STEPS = range(1, 21)
MIN_FLIP_SYNCS = 10
S = 1.5
RATIO_BOUNDS = (0.80, 1.25)
QKV = ("q_proj", "k_proj", "v_proj")
OTHER = ("o_proj", "gate_proj", "up_proj", "down_proj")
BANDS = {1.5: (1.35, 1.65), 1.0: (0.90, 1.10), "R4/R5": (0.95, 1.05)}
PAPER_TABLE4 = {"R3": 30.63, "R4": 31.25, "R6": 29.06}
BOOT_N, BOOT_SEED = 10000, 0
SCORE, K3 = "critic/score/mean", "rollout_corr/k3_kl"
SIZES, ARMS = ("7b", "1.5b"), ("R3", "R4", "R5", "R6")


# ---- data ----
def load_metrics(run):
    parts = sorted(glob.glob(str(run / "metrics.part*.jsonl")), key=lambda p: int(re.search(r"part(\d+)", p).group(1)))
    by = {}
    for path in parts + [str(run / "metrics.jsonl")]:
        if Path(path).is_file():
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    # per metric: a resumed attempt's re-run steps replace the earlier values, and its validation-only
                    # record of the checkpointed step leaves that step's training metrics in place
                    by.setdefault(int(r["step"]), {}).update(r["data"])
    return by


def pooled_flips(recs, projs):
    """Code-weighted flip share of projs over the window, averaged over syncs; (value, number of syncs)."""
    vals = []
    for r in recs:
        if r["sync"] not in FLIP_SYNCS:
            continue
        num = sum(r["codes_per_layer"][p] * statistics.fmean(r["flip"][p]) for p in projs)
        den = sum(r["codes_per_layer"][p] for p in projs)
        vals.append(num / den)
    return (statistics.fmean(vals) if vals else None), len(vals)


def load_e3(run):
    """Records of the first attempt that starts at step 0 and covers the window; else of the first step-0 attempt."""
    path = run / "e3_int8.jsonl"
    if not path.is_file():
        return None
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    attempts = []
    for r in recs:
        if r["attempt"] not in attempts:
            attempts.append(r["attempt"])
    step0 = [a for a in attempts if all(r.get("start_step", 0) == 0 for r in recs if r["attempt"] == a)]
    for a in step0:
        mine = [r for r in recs if r["attempt"] == a]
        if sum(r["sync"] in FLIP_SYNCS for r in mine) >= MIN_FLIP_SYNCS:
            return mine
    return [r for r in recs if r["attempt"] == step0[0]] if step0 else None


def mechanism(run, by):
    """flip_qkv, flip_other, k3_early, k3_1 of a run, or None where its window is incomplete."""
    recs = load_e3(run)
    v = {}
    for name, projs in (("flip_qkv", QKV), ("flip_other", OTHER)):
        val, n = pooled_flips(recs, projs) if recs else (None, 0)
        v[name] = val if n >= MIN_FLIP_SYNCS else None
    early = [by[s][K3] for s in EARLY_STEPS if s in by and K3 in by[s]]
    v["k3_early"] = statistics.fmean(early) if len(early) == len(EARLY_STEPS) else None
    v["k3_1"] = by.get(1, {}).get(K3)
    return v


def load_full(run):
    run = Path(run)
    if not run.is_dir():
        return None, "missing"
    by = load_metrics(run)
    done = [s for s in range(1, N_STEPS + 1) if s in by and SCORE in by[s]]
    mech = mechanism(run, by)
    if len(done) < N_STEPS:
        return {"partial": mech}, f"incomplete ({len(done)} of {N_STEPS} steps)"
    if not (run / "eval.json").is_file():
        return {"partial": mech}, "trained, not evaluated"
    res = json.load(open(run / "eval.json", encoding="utf-8"))["results"]
    score = [by[s][SCORE] for s in range(1, N_STEPS + 1)]
    k3 = [by[s][K3] for s in range(1, N_STEPS + 1) if K3 in by[s]]
    v = {"math": 100 * res["math500"]["avg@4"], "aime": 100 * res["aime24"]["avg@32"],
         "math_g": 100 * res["math500"]["greedy"], "aime_g": 100 * res["aime24"]["greedy"],
         "auc": 100 * statistics.fmean(score), "early": 100 * statistics.fmean(score[:20]),
         "k3": statistics.fmean(k3) if len(k3) == N_STEPS else None,
         "stalled": statistics.fmean(score[180:200]) < statistics.fmean(score[:10]),
         "pp_math": [100 * x for x in res["math500"]["per_problem_sampled"]],
         "pp_aime": [100 * x for x in res["aime24"]["per_problem_sampled"]], **mech}
    return v, "complete"


def load_mech(run):
    run = Path(run)
    if not run.is_dir():
        return None, "missing"
    v = mechanism(run, load_metrics(run))
    return v, "complete" if v["flip_qkv"] is not None and v["k3_early"] is not None else "incomplete window"


# ---- statistics ----
def boot_diff(a, b, n=BOOT_N, seed=BOOT_SEED):
    """95% percentile interval of mean(a) - mean(b) over problems resampled with replacement (paired)."""
    rng = random.Random(seed)
    k = len(a)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(statistics.fmean(d[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def ratio_verdict(r, pred):
    lo, hi = BANDS[pred]
    return f"{r:.3f} (predicted {pred if pred != 'R4/R5' else 1.0}; band {lo:.2f}-{hi:.2f}: " \
           f"{'consistent' if lo <= r <= hi else 'NOT consistent'})"


def mismatch_word(x):
    return "reduced" if x < RATIO_BOUNDS[0] else "increased" if x > RATIO_BOUNDS[1] else "not meaningfully changed"


# ---- report ----
def m1_sources(full, mech, size):
    """Per arm: (values, source tag) for M1 and early M2; the full run when its window is complete."""
    out = {}
    for arm in ARMS:
        f = full.get((size, arm))
        fm = (f if f and "flip_qkv" in f else (f or {}).get("partial")) if f else None
        if fm and fm.get("flip_qkv") is not None and fm.get("k3_early") is not None:
            out[arm] = (fm, f"s4_{size}_{arm}_s1")
        elif mech.get((size, arm)) and mech[(size, arm)]["flip_qkv"] is not None:
            out[arm] = (mech[(size, arm)], f"s4m_{size}_{arm}_s1")
    return out


def m1(src, size):
    out = [f"M1 flip-share ratios, {size} (syncs {FLIP_SYNCS.start}-{FLIP_SYNCS.stop - 1}; sources: "
           + (", ".join(f"{a} {t}" for a, (_, t) in src.items()) or "none") + ")"]
    if "R3" not in src:
        return out + ["  no verdict: no R3 source"]
    base = src["R3"][0]
    for arm in ("R4", "R5", "R6"):
        if arm not in src:
            out.append(f"  {arm}/R3: not available")
            continue
        r = src[arm][0]
        out.append(f"  {arm}/R3 q/k/v  {ratio_verdict(r['flip_qkv'] / base['flip_qkv'], S)}")
        out.append(f"  {arm}/R3 other  {ratio_verdict(r['flip_other'] / base['flip_other'], S if arm == 'R6' else 1.0)}")
    if "R4" in src and "R5" in src:
        r4, r5 = src["R4"][0], src["R5"][0]
        out.append(f"  R4/R5 q/k/v  {ratio_verdict(r4['flip_qkv'] / r5['flip_qkv'], 'R4/R5')}")
        out.append(f"  R4/R5 other  {ratio_verdict(r4['flip_other'] / r5['flip_other'], 'R4/R5')}")
    return out


def m2(src, full, size):
    out = []
    if "R3" in src and "R4" in src:
        a = src["R4"][0]["k3_early"] / src["R3"][0]["k3_early"]
        out.append(f"M2 mismatch, {size}, early (steps 1-20): k3 ratio R4/R3 {a:.3f} -> {mismatch_word(a)}")
    else:
        out.append(f"M2 mismatch, {size}, early: no verdict (no R3 or R4 source)")
    r3, r4 = full.get((size, "R3")), full.get((size, "R4"))
    if r3 and r4 and r3.get("k3") and r4.get("k3"):
        b = r4["k3"] / r3["k3"]
        out.append(f"M2 mismatch, {size}, whole run (steps 1-200): k3 ratio R4/R3 {b:.3f} -> {mismatch_word(b)}")
    else:
        out.append(f"M2 mismatch, {size}, whole run: no verdict (full R3 and R4 runs not both complete)")
    return out


def pairs(full, size):
    out = []
    for a, b, what in (("R4", "R5", "H1"), ("R4", "R3", "UAQ's effect"), ("R6", "R3", "the paper's control")):
        ra, rb = full.get((size, a)), full.get((size, b))
        if not (ra and rb and "math" in ra and "math" in rb):
            continue
        cells = [f"{k} {ra[k] - rb[k]:+.2f}" for k in ("math", "aime", "math_g", "aime_g", "auc", "early")]
        lo_m, hi_m = boot_diff(ra["pp_math"], rb["pp_math"])
        lo_a, hi_a = boot_diff(ra["pp_aime"], rb["pp_aime"])
        out.append(f"  {size} {a} - {b} ({what}): " + "; ".join(cells))
        out.append(f"    evaluation-noise 95% intervals (bootstrap over problems): math [{lo_m:+.2f}, {hi_m:+.2f}], "
                   f"aime [{lo_a:+.2f}, {hi_a:+.2f}]")
    return out


def report(results):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = [f"S4 analysis (pre-registered; s4_analysis.py sha256 {sha})", f"results: {results}", ""]
    full, mech = {}, {}
    out.append(f"{'run':20s} {'math':>7s} {'aime':>7s} {'math_g':>7s} {'aime_g':>7s} {'auc':>7s} {'early':>7s} "
               f"{'k3':>9s} {'k3_early':>9s} {'flip_qkv':>9s} {'flip_oth':>9s}  status")

    def row(tag, v, status):
        def f(k, fmt):
            return format(v[k], fmt) if v and v.get(k) is not None else ""
        cells = [f(k, ".2f").rjust(7) for k in ("math", "aime", "math_g", "aime_g", "auc", "early")]
        cells += [f(k, ".3e").rjust(9) for k in ("k3", "k3_early", "flip_qkv", "flip_other")]
        stalled = " (stalled)" if v and v.get("stalled") else ""
        return f"{tag:20s} " + " ".join(cells) + f"  {status}{stalled}"

    for size in SIZES:
        for arm in ARMS:
            v, status = load_full(Path(results) / f"s4_{size}_{arm}_s1")
            if status != "missing":
                full[(size, arm)] = v
                out.append(row(f"s4_{size}_{arm}_s1", v if "math" in v else v.get("partial"), status))
            v, status = load_mech(Path(results) / f"s4m_{size}_{arm}_s1")
            if status != "missing":
                mech[(size, arm)] = v
                out.append(row(f"s4m_{size}_{arm}_s1", v, status))
    for size in SIZES:
        p = Path(results) / f"base_{size}" / "eval.json"
        if p.is_file():
            r = json.load(open(p, encoding="utf-8"))["results"]
            out.append(f"base {size} (step 0): MATH-500 Avg@4 {100 * r['math500']['avg@4']:.2f}, AIME 2024 Avg@32 "
                       f"{100 * r['aime24']['avg@32']:.2f}")
    out.append("")
    complete_full = {k: v for k, v in full.items() if v and "math" in v}
    for size in SIZES:
        src = m1_sources(full, mech, size)
        out += m1(src, size) + m2(src, complete_full, size) + [""]
    out.append("Descriptive: full-run differences, seed 1 (points; k3 not shown)")
    for size in SIZES:
        out += pairs(complete_full, size)
    out.append("  the paper's Table 4, AIME 2024 Avg@32: " + ", ".join(f"{a} {v}" for a, v in PAPER_TABLE4.items()))
    return "\n".join(out) + "\n"


def self_test():
    # pooled flips: two projections with 100 and 300 codes per layer, two layers; syncs outside the window ignored
    rec = {"sync": 2, "codes_per_layer": {"a": 100, "b": 300}, "flip": {"a": [0.01, 0.03], "b": [0.0, 0.02]}}
    val, n = pooled_flips([rec, dict(rec, sync=1), dict(rec, sync=22)], ("a", "b"))
    assert n == 1 and abs(val - (100 * 0.02 + 300 * 0.01) / 400) < 1e-15, val
    assert "NOT" not in ratio_verdict(1.40, 1.5) and "NOT" in ratio_verdict(1.70, 1.5)
    assert "NOT" not in ratio_verdict(0.96, "R4/R5") and "NOT" in ratio_verdict(1.06, "R4/R5")
    assert mismatch_word(0.79) == "reduced" and mismatch_word(1.0).startswith("not") and mismatch_word(1.3) == "increased"
    # bootstrap: identical runs give [0, 0]; a constant shift gives exactly the shift; deterministic with the seed
    assert boot_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, 0.0)
    assert boot_diff([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == (1.0, 1.0)
    x, y = [random.Random(1).random() for _ in range(50)], [random.Random(2).random() for _ in range(50)]
    assert boot_diff(x, y) == boot_diff(x, y)
    print("self-test passed: pooled flip shares, ratio bands, mismatch words, paired bootstrap")


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
    text = report(a.results)
    sys.stdout.write(text)
    out = Path(a.out) if a.out else Path(a.results) / "s4_analysis.txt"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
