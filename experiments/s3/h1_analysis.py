#!/usr/bin/env python3
"""Pre-registered H1 analysis for S3 (P1 note, section 7, "Pre-registered H1 analysis for S3"; fixed 2026-11-11).

H1: R4 (UAQ at s = 1.5, plain AdamW) and R5 (no UAQ, Theorem 2(b)-(c) settings) are equal in distribution.
The test is paired by seed (the arms of one seed share prompt order and sampling seed):
  d_k = Y(R4, seed k) - Y(R5, seed k),  t-intervals on n - 1 degrees of freedom.
  H1 supported   : the 90% CI of the mean difference lies inside the margin (two one-sided tests, alpha = 0.05)
  H1 contradicted: the 95% CI excludes 0
  inconclusive   : neither
Primary endpoint: late accuracy = mean GSM8K test accuracy over the last five validations (steps 319-435), margin
+-2.0 points. Secondary (supportive): final accuracy (step 435), reward AUC (mean training score over all steps),
both +-2.0 points; mismatch = mean k3 over all steps, as the ratio R4/R5 with bounds 0.80-1.25. The paired entropy
difference is reported next to the mismatch (descriptive). Sensitivity: the same analysis for R4 - R3 (UAQ's own
effect), on seeds 1-3 unless --sensitivity-seeds says otherwise.

Verdicts are printed only when every registered seed is complete for both arms of a contrast. Completed runs are
never excluded for their outcome. Any change to this file after registration must be recorded in the note.

usage: python h1_analysis.py [--results DIR] [--seeds 1 2 3] [--sensitivity-seeds 1 2 3] [--out FILE]
       python h1_analysis.py --self-test
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

# ---- registered constants (2026-11-11) ----
N_STEPS = 435
VAL_EVERY = 29
LATE_STEPS = (319, 348, 377, 406, 435)
MARGIN_POINTS = 2.0            # late accuracy, final accuracy, reward AUC (percentage points)
RATIO_BOUNDS = (0.80, 1.25)    # mismatch ratio R4/R5 (or R4/R3)
SEEDS = (1, 2, 3)
VAL_KEY = "val-core/openai/gsm8k/acc/mean@1"
STEP_KEYS = {"score": "critic/score/mean", "k3": "rollout_corr/k3_kl", "entropy": "actor/entropy"}


# ---- Student t quantiles without scipy (regularized incomplete beta, Lentz continued fraction) ----
def _betacf(a, b, x, itmax=500, eps=1e-15):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_ppf(q, df):
    """Quantile for q in (0.5, 1), by bisection."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- data ----
def load_run(run_dir):
    """Per-run endpoints, or (None, reason) if the run is missing or incomplete."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return None, "missing"
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs.setdefault(int(r["step"]), {}).update(r["data"])
    missing_steps = [s for s in range(1, N_STEPS + 1) if s not in recs]
    if missing_steps:
        return None, f"incomplete ({N_STEPS - len(missing_steps)} of {N_STEPS} steps)"
    val_steps = range(VAL_EVERY, N_STEPS + 1, VAL_EVERY)
    if any(VAL_KEY not in recs[s] for s in val_steps):
        return None, "incomplete (validations missing)"
    for name, key in STEP_KEYS.items():
        if any(key not in recs[s] for s in range(1, N_STEPS + 1)):
            return None, f"incomplete ({name} missing)"
    per_step = {name: [recs[s][key] for s in range(1, N_STEPS + 1)] for name, key in STEP_KEYS.items()}
    return {
        "late_acc": 100.0 * statistics.fmean(recs[s][VAL_KEY] for s in LATE_STEPS),
        "final_acc": 100.0 * recs[N_STEPS][VAL_KEY],
        "reward_auc": 100.0 * statistics.fmean(per_step["score"]),
        "mismatch": statistics.fmean(per_step["k3"]),
        "entropy": statistics.fmean(per_step["entropy"]),
    }, "complete"


# ---- statistics ----
def paired_ci(diffs):
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    t90, t95 = t_ppf(0.95, n - 1), t_ppf(0.975, n - 1)
    return mean, sd, (mean - t90 * se, mean + t90 * se), (mean - t95 * se, mean + t95 * se)


def classify(ci90, ci95, lo, hi):
    equivalent = lo < ci90[0] and ci90[1] < hi
    different = ci95[0] > 0.0 or ci95[1] < 0.0
    return equivalent, different


H1_LABELS = {(True, False): "H1 supported (equivalent within the margin)",
             (False, True): "H1 contradicted (difference detected)",
             (True, True): "equivalent within the margin, smaller difference detected",
             (False, False): "inconclusive"}
EFFECT_LABELS = {(True, False): "no effect beyond the margin",
                 (False, True): "effect detected",
                 (True, True): "effect detected, smaller than the margin",
                 (False, False): "inconclusive"}

ENDPOINTS = [  # name, key, ratio?, label
    ("late accuracy (primary)", "late_acc", False),
    ("final accuracy", "final_acc", False),
    ("reward AUC", "reward_auc", False),
    ("mismatch k3 (ratio)", "mismatch", True),
]


def contrast(runs, arm_a, arm_b, seeds, labels):
    lines = [f"{arm_a} - {arm_b}, paired by seed"]
    pairs = [k for k in seeds if runs.get((arm_a, k)) and runs.get((arm_b, k))]
    if len(pairs) < len(seeds) or len(pairs) < 2:
        lines.append(f"  no verdict: {len(pairs)} of {len(seeds)} registered pairs complete")
        return lines
    lines.append(f"  n = {len(pairs)} pairs, t on {len(pairs) - 1} df; 90% CI decides equivalence, 95% CI decides difference")
    for name, key, ratio in ENDPOINTS:
        a = [runs[(arm_a, k)][key] for k in pairs]
        b = [runs[(arm_b, k)][key] for k in pairs]
        if ratio:
            diffs = [math.log(x) - math.log(y) for x, y in zip(a, b)]
            lo, hi = math.log(RATIO_BOUNDS[0]), math.log(RATIO_BOUNDS[1])
            mean, sd, ci90, ci95 = paired_ci(diffs)
            verdict = labels[classify(ci90, ci95, lo, hi)]
            per = " ".join(f"{math.exp(d):.3f}" for d in diffs)
            lines.append(f"  {name:24s} bounds {RATIO_BOUNDS[0]:.2f}-{RATIO_BOUNDS[1]:.2f}; per seed {per};"
                         f" geometric mean ratio {math.exp(mean):.3f} (log SD {sd:.3f});"
                         f" 90% CI [{math.exp(ci90[0]):.3f}, {math.exp(ci90[1]):.3f}];"
                         f" 95% CI [{math.exp(ci95[0]):.3f}, {math.exp(ci95[1]):.3f}] -> {verdict}")
        else:
            diffs = [x - y for x, y in zip(a, b)]
            mean, sd, ci90, ci95 = paired_ci(diffs)
            verdict = labels[classify(ci90, ci95, -MARGIN_POINTS, MARGIN_POINTS)]
            per = " ".join(f"{d:+.2f}" for d in diffs)
            lines.append(f"  {name:24s} margin +-{MARGIN_POINTS:.1f} pts; per seed {per}; mean {mean:+.2f} (SD {sd:.2f});"
                         f" 90% CI [{ci90[0]:+.2f}, {ci90[1]:+.2f}]; 95% CI [{ci95[0]:+.2f}, {ci95[1]:+.2f}] -> {verdict}")
    ent = [runs[(arm_a, k)]["entropy"] - runs[(arm_b, k)]["entropy"] for k in pairs]
    lines.append(f"  {'mean entropy':24s} descriptive; per seed {' '.join(f'{d:+.3f}' for d in ent)};"
                 f" mean difference {statistics.fmean(ent):+.3f}")
    return lines


def report(results, seeds, sens_seeds):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = [f"H1 analysis for S3, pre-registered 2026-11-11 (P1 note, section 7); h1_analysis.py sha256 {sha}",
           f"results: {results}; seeds for H1: {' '.join(map(str, seeds))};"
           f" seeds for the sensitivity contrast: {' '.join(map(str, sens_seeds))}", ""]
    runs = {}
    out.append(f"{'run':14s} {'late acc':>9s} {'final acc':>9s} {'reward AUC':>10s} {'mean k3':>9s} {'entropy':>8s}  status")
    all_seeds = sorted(set(seeds) | set(sens_seeds))
    for arm in ("R3", "R4", "R5"):
        for k in all_seeds:
            tag = f"uaq_{arm}_s{k}"
            vals, status = load_run(Path(results) / tag)
            runs[(arm, k)] = vals
            if vals:
                out.append(f"{tag:14s} {vals['late_acc']:9.2f} {vals['final_acc']:9.2f} {vals['reward_auc']:10.2f}"
                           f" {vals['mismatch']:9.2e} {vals['entropy']:8.3f}  {status}")
            else:
                out.append(f"{tag:14s} {'':9s} {'':9s} {'':10s} {'':9s} {'':8s}  {status}")
    out.append("")
    out.append("late accuracy per arm over its complete seeds (descriptive):")
    for arm in ("R3", "R4", "R5"):
        xs = [runs[(arm, k)]["late_acc"] for k in all_seeds if runs.get((arm, k))]
        out.append(f"  {arm}: " + (f"mean {statistics.fmean(xs):.2f}, seed-to-seed SD {statistics.stdev(xs):.2f} pts over {len(xs)} seeds"
                                   if len(xs) > 1 else f"{len(xs)} complete seed(s)"))
    out.append("")
    out.append("H1 test: " + "\n".join(contrast(runs, "R4", "R5", seeds, H1_LABELS)))
    out.append("")
    out.append("Sensitivity, UAQ's own effect: " + "\n".join(contrast(runs, "R4", "R3", sens_seeds, EFFECT_LABELS)))
    return "\n".join(out) + "\n"


def self_test():
    # t quantiles against standard tables
    table = {(0.95, 2): 2.919986, (0.975, 2): 4.302653, (0.95, 4): 2.131847, (0.975, 4): 2.776445,
             (0.95, 1): 6.313752, (0.975, 9): 2.262157}
    for (q, df), ref in table.items():
        got = t_ppf(q, df)
        assert abs(got - ref) < 1e-5, (q, df, got, ref)
    # interval and verdict logic on a hand-computed case: d = (0.5, 1.0, 0.0)
    mean, sd, ci90, ci95 = paired_ci([0.5, 1.0, 0.0])
    half90 = 2.919986 * 0.5 / math.sqrt(3)
    assert abs(mean - 0.5) < 1e-12 and abs(sd - 0.5) < 1e-12
    assert abs(ci90[0] - (0.5 - half90)) < 1e-5 and abs(ci90[1] - (0.5 + half90)) < 1e-5
    assert H1_LABELS[classify(ci90, ci95, -2, 2)].startswith("H1 supported")
    assert H1_LABELS[classify(*paired_ci([3.0, 3.5, 2.5])[2:], -2, 2)].startswith("H1 contradicted")
    assert H1_LABELS[classify(*paired_ci([1.0, -3.0, 4.0])[2:], -2, 2)] == "inconclusive"
    assert H1_LABELS[classify(*paired_ci([0.5, 0.6, 0.55])[2:], -2, 2)].startswith("equivalent within")
    print("self-test passed: t quantiles match tables to 1e-5; interval and verdict logic as specified")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="seeds for the H1 test (R4 - R5)")
    ap.add_argument("--sensitivity-seeds", type=int, nargs="+", default=list(SEEDS), help="seeds for R4 - R3")
    ap.add_argument("--out", default=None, help="report file (default: <results>/h1_analysis.txt)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    text = report(a.results, a.seeds, a.sensitivity_seeds)
    sys.stdout.write(text)
    out = Path(a.out) if a.out else Path(a.results) / "h1_analysis.txt"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
