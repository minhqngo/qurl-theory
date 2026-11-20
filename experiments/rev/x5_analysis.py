"""Revision experiment X5, pre-registered analysis. Do UAQ (R4) and its matched optimizer settings (R5) stall early
equally often?

  python -I x5_analysis.py --results <copy of ~/qurl/rev/results> [--s3 <copy of S3's results>] [--out report.txt]
  python -I x5_analysis.py --self-test

Data: 29-step runs of S3's arms with S3's own call (x5_run.sh), validated by VeRL at step 29.
  - R4 and R5 for seeds 6-35, paired by seed. Each seed fixes the prompt order and the sampling seed.
  - R3 for seeds 6-15, as a reference.
Stall: S3's description, fixed before these runs: validation accuracy below 10% at step 29. In S3's 22 runs, the four
stalled runs validated at 0.0-5.5% at step 29 and every other run at 33.9% or more.

Primary test (the stall part of H1): over the seeds with both R4 and R5 complete, the exact McNemar test, i.e. a
two-sided binomial test with p = 1/2 on the discordant seeds.
  "R5 stalls more often"      p < 0.05 and more seeds where only R5 stalled than where only R4 did;
  "R4 stalls more often"      p < 0.05 and the reverse;
  "no detectable difference"  otherwise.
It is reported with each arm's stall rate (Clopper-Pearson 95% CI) and with the paired difference in rates and its
percentile-bootstrap 95% CI over seeds (10,000 resamples, seed 0).
Power, stated in advance, assuming independent stalls and 30 pairs:
  - 0.88 if the rates are 40% and 5% (S3 had R5 at 2 of 5 and R4 at 0 of 5);
  - 0.62 for 30% and 5%;
  - 0.25 for 20% and 5%.
Stalls that share a seed's prompt order make discordant seeds rarer, and the power lower. If fewer than the 30
registered pairs complete, the test runs on those available and the report says so.

Secondary (descriptive, no verdicts):
  - R3's stall rate on seeds 6-15, next to R4's and R5's on the same seeds;
  - among seeds where neither R4 nor R5 stalled, the paired difference R4 - R5 in step-29 accuracy and in early
    reward (critic/score/mean averaged over steps 1-29), each with its mean and 95% t-interval;
  - each run's longest stretch of steps without a correct training answer (critic/score/mean = 0);
  - S3's own runs (R4 and R5 seeds 1-5, R3 seeds 1-3), whose step-29 validations were seen before this registration:
    their stall counts pooled with these, labelled as such.

Fixed in writing before any of these runs, and frozen by SHA-256 (registration.sha256).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from math import comb

VAL_KEY = "val-core/openai/gsm8k/acc/mean@1"
SCORE_KEY = "critic/score/mean"
STALL_BELOW = 10.0
PAIR_SEEDS = list(range(6, 36))
R3_SEEDS = list(range(6, 16))
REGISTERED_PAIRS = 30
T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042, 2.040,
        2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021]


def run_summary(metrics_path):
    """Step-29 validation (points), stall flag, early reward (points), longest stretch without a correct answer."""
    if not os.path.exists(metrics_path):
        return None
    val, score = {}, {}
    for line in open(metrics_path):
        if not line.strip():
            continue
        r = json.loads(line)
        step, d = int(r["step"]), r.get("data", {})
        if VAL_KEY in d:
            val[step] = 100.0 * d[VAL_KEY]
        if SCORE_KEY in d:
            score[step] = d[SCORE_KEY]
    if 29 not in val or any(s not in score for s in range(1, 30)):
        return None
    longest = cur = 0
    for s in range(1, 30):
        cur = cur + 1 if score[s] == 0 else 0
        longest = max(longest, cur)
    return {"val29": val[29], "stall": val[29] < STALL_BELOW,
            "early_reward": 100.0 * sum(score[s] for s in range(1, 30)) / 29, "longest_zero": longest}


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def _binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(f):   # f increasing in p from negative to positive
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return lower, upper


def tci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T975[min(n - 1, 40)] * sd / math.sqrt(n)
    return m, (m - h, m + h)


def analyse(results, s3=None):
    lines = []
    out = lines.append
    runs = {}
    for arm, seeds in (("R4", PAIR_SEEDS), ("R5", PAIR_SEEDS), ("R3", R3_SEEDS)):
        for s in seeds:
            r = run_summary(os.path.join(results, "x5", f"rev_x5_{arm}_s{s}", "metrics.jsonl"))
            if r is not None:
                runs[(arm, s)] = r
    out("X5 analysis (pre-registered): early stalls of R4 (UAQ) and R5 (matched settings)")
    out(f"Stall: validation accuracy below {STALL_BELOW:.0f}% at step 29.")
    out("")
    out("Runs (val29 % | stall | early reward % | longest stretch without a correct answer):")
    for s in PAIR_SEEDS:
        row = []
        for arm in ("R4", "R5", "R3"):
            r = runs.get((arm, s))
            row.append(f"{arm} " + (f"{r['val29']:5.1f} {'STALL' if r['stall'] else '  -  '} {r['early_reward']:5.1f} "
                                    f"{r['longest_zero']:2d}" if r else "   (none)            "))
        out(f"  seed {s:2d}: " + " | ".join(row))
    out("")

    pairs = [s for s in PAIR_SEEDS if ("R4", s) in runs and ("R5", s) in runs]
    n = len(pairs)
    out(f"Primary test (exact McNemar, paired by seed): {n} complete pairs of {REGISTERED_PAIRS} registered"
        + ("" if n >= REGISTERED_PAIRS else " (fewer than registered)"))
    if n == 0:
        out("  no complete pairs")
        return "\n".join(lines), {"verdict": "no data", "b": 0, "c": 0, "n": 0}
    b = sum(runs[("R5", s)]["stall"] and not runs[("R4", s)]["stall"] for s in pairs)
    c = sum(runs[("R4", s)]["stall"] and not runs[("R5", s)]["stall"] for s in pairs)
    both = sum(runs[("R4", s)]["stall"] and runs[("R5", s)]["stall"] for s in pairs)
    p = mcnemar_p(b, c)
    if p < 0.05 and b > c:
        v = "R5 stalls more often"
    elif p < 0.05 and c > b:
        v = "R4 stalls more often"
    else:
        v = "no detectable difference"
    k4 = sum(runs[("R4", s)]["stall"] for s in pairs)
    k5 = sum(runs[("R5", s)]["stall"] for s in pairs)
    for arm, k in (("R4", k4), ("R5", k5)):
        lo, hi = clopper_pearson(k, n)
        out(f"  {arm}: stalled in {k} of {n} ({100 * k / n:.0f}%, 95% CI [{100 * lo:.0f}%, {100 * hi:.0f}%])")
    diffs = [int(runs[("R5", s)]["stall"]) - int(runs[("R4", s)]["stall"]) for s in pairs]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(10000))
    out(f"  both stalled: {both}; only R5: {b}; only R4: {c}; McNemar p = {p:.4f}")
    out(f"  stall rate R5 - R4: {100 * sum(diffs) / n:+.0f} points, bootstrap 95% CI "
        f"[{100 * boots[249]:+.0f}, {100 * boots[9749]:+.0f}]")
    out(f"  -> {v}")
    out("")

    out("Secondary (descriptive):")
    r3 = [s for s in R3_SEEDS if ("R3", s) in runs]
    if r3:
        k3 = sum(runs[("R3", s)]["stall"] for s in r3)
        same4 = [s for s in r3 if ("R4", s) in runs]
        same5 = [s for s in r3 if ("R5", s) in runs]
        out(f"  R3 on seeds 6-15: stalled in {k3} of {len(r3)}; on the same seeds R4 "
            f"{sum(runs[('R4', s)]['stall'] for s in same4)} of {len(same4)}, R5 "
            f"{sum(runs[('R5', s)]['stall'] for s in same5)} of {len(same5)}")
    clean = [s for s in pairs if not runs[("R4", s)]["stall"] and not runs[("R5", s)]["stall"]]
    for key, name in (("val29", "step-29 accuracy"), ("early_reward", "early reward")):
        xs = [runs[("R4", s)][key] - runs[("R5", s)][key] for s in clean]
        if len(xs) >= 2:
            m, ci = tci(xs)
            out(f"  R4 - R5, {name}, {len(xs)} seeds where neither stalled: {m:+.2f} points, 95% CI "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}]")
    if s3:
        prior = {}
        for arm, seeds in (("R4", range(1, 6)), ("R5", range(1, 6)), ("R3", range(1, 4))):
            for s in seeds:
                r = run_summary(os.path.join(s3, f"uaq_{arm}_s{s}", "metrics.jsonl"))
                if r is not None:
                    prior[(arm, s)] = r
        out("  S3's own runs (seen before this registration), step-29 stalls:")
        for arm in ("R4", "R5", "R3"):
            ks = [s for (a, s) in prior if a == arm]
            new = [s for (a, s) in runs if a == arm]
            kp = sum(prior[(arm, s)]["stall"] for s in ks)
            kn = sum(runs[(arm, s)]["stall"] for s in new)
            out(f"    {arm}: S3 {kp} of {len(ks)}; pooled with X5 {kp + kn} of {len(ks) + len(new)}")
    return "\n".join(lines), {"verdict": v, "b": b, "c": c, "n": n, "p": p}


def _write_run(root, arm, seed, val29, scores):
    d = os.path.join(root, "x5", f"rev_x5_{arm}_s{seed}")
    os.makedirs(d)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for step in range(1, 30):
            data = {SCORE_KEY: scores[step - 1]}
            if step == 29:
                data[VAL_KEY] = val29 / 100
            f.write(json.dumps({"step": step, "data": data}) + "\n")


def self_test():
    ok = True

    def scenario(stall4, stall5, stall3=()):
        root = tempfile.mkdtemp(prefix="x5_selftest_")
        try:
            for s in PAIR_SEEDS:
                for arm, st in (("R4", stall4), ("R5", stall5)):
                    if s in st:
                        _write_run(root, arm, s, 2.0, [0.01] * 3 + [0.0] * 26)
                    else:
                        _write_run(root, arm, s, 45.0, [0.02 * min(1, t / 10) + 0.01 for t in range(29)])
            for s in R3_SEEDS:
                _write_run(root, "R3", s, 1.0 if s in stall3 else 44.0, [0.0] * 29 if s in stall3 else [0.2] * 29)
            return analyse(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    cases = [("R5 stalls on 12 seeds, R4 on 1 other", set(range(6, 7)), set(range(10, 22)), "R5 stalls more often"),
             ("R4 stalls on 9 seeds, R5 on none", set(range(6, 15)), set(), "R4 stalls more often"),
             ("same 3 seeds stall in both, plus 2 each", {6, 7, 8, 9, 10}, {6, 7, 8, 30, 31}, "no detectable difference"),
             ("R5 on 7 seeds, R4 on 1 (p = 0.07)", {6}, set(range(20, 27)), "no detectable difference")]
    for name, s4, s5, want in cases:
        text, v = scenario(s4, s5, stall3={6, 7})
        good = v["verdict"] == want and v["n"] == 30
        ok &= good
        print(f"self-test: {name}: {v['verdict']} (b={v['b']}, c={v['c']}, p={v['p']:.4f}) "
              f"{'ok' if good else 'WRONG, want ' + want}")
    for k, nn, want in ((0, 30, (0.0, 0.1157)), (15, 30, (0.3130, 0.6870)), (2, 5, (0.0527, 0.8534))):
        lo, hi = clopper_pearson(k, nn)
        good = abs(lo - want[0]) < 5e-4 and abs(hi - want[1]) < 5e-4
        ok &= good
        print(f"  Clopper-Pearson({k}/{nn}) = [{lo:.4f}, {hi:.4f}] {'ok' if good else 'WRONG, want ' + str(want)}")
    good = abs(mcnemar_p(7, 1) - 0.0703125) < 1e-9 and mcnemar_p(0, 0) == 1.0 and abs(mcnemar_p(12, 1) - 28 / 8192) < 1e-12
    ok &= good
    print(f"  McNemar p values {'ok' if good else 'WRONG'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--s3", help="S3's results (experiments/s3/results) for the descriptive pooled counts")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.results:
        ap.error("--results is required")
    text, _ = analyse(a.results, a.s3)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
