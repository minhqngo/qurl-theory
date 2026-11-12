#!/usr/bin/env python3
"""Pre-registered H2 and H3 analyses for S3 (P1 note, section 7, "Pre-registered H2 and H3 analyses for S3"; fixed
2026-11-11), and the exploratory collapse analyses for H1-H3 fixed the same day.

H2: UAQ's effect is not specific to quantization, (R2 - R1) ~ (R4 - R3). Per seed k the interaction
      c_k = [Y(R4,k) - Y(R3,k)] - [Y(R2,k) - Y(R1,k)]
    supported if the 90% CI of its mean lies inside the margin, contradicted if the 95% CI excludes 0, inconclusive
    otherwise (the rule H1 uses).
H3: R6 (global learning rate x1.5) differs from R4 (UAQ). Per seed k, d_k = Y(R6,k) - Y(R4,k);
    supported if the 95% CI excludes 0, contradicted if the 90% CI lies inside the margin, inconclusive otherwise.
Endpoints: late accuracy (primary); final accuracy, reward AUC, early reward (mean training score over steps 1-58,
the first two epochs) and mismatch (on a log scale: for H2 the ratio of ratios (R4/R3)/(R2/R1), for H3 the ratio
R6/R4). Margins as in h1_analysis.py: 2.0 points, ratio bounds 0.80-1.25. Seeds 1-3, t-intervals on 2 df.
Also printed, descriptive: the simple effects R4 - R3, R2 - R1 and R6 - R3.
Exploratory (labelled so; they do not replace the registered verdicts): runs whose late accuracy is below 10% count
as collapsed and are counted per arm; H1 (R4 - R5, seeds 1-5), H2 and H3 are repeated on the seeds where none of
the runs involved collapsed.

The statistics (loading, endpoints, t-intervals, decision logic) are imported from the frozen h1_analysis.py, so all
three hypotheses use identical arithmetic. Verdicts are printed only when every registered run of a contrast is
complete. Any change to this file after registration must be recorded in the note.

usage: python h23_analysis.py [--results DIR] [--out FILE]
       python h23_analysis.py --self-test
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_analysis as h1  # noqa: E402  (frozen 2026-11-11, sha256 5ac6da47...)

# ---- registered constants (2026-11-11) ----
EARLY_STEPS = range(1, 59)          # early reward: the first two epochs (29 steps each)
COLLAPSE_BELOW = 10.0               # exploratory: late accuracy below this (percentage points) counts as collapsed
SEEDS_H23 = (1, 2, 3)
SEEDS_H1 = (1, 2, 3, 4, 5)
ARMS = ("R1", "R2", "R3", "R4", "R5", "R6")
SEEDS_PER_ARM = {"R1": SEEDS_H23, "R2": SEEDS_H23, "R3": SEEDS_H23, "R4": SEEDS_H1, "R5": SEEDS_H1, "R6": SEEDS_H23}
ENDPOINTS = [  # label, key, compared on a log scale?
    ("late accuracy (primary)", "late_acc", False),
    ("final accuracy", "final_acc", False),
    ("reward AUC", "reward_auc", False),
    ("early reward (steps 1-58)", "early_reward", False),
    ("mismatch k3 (log scale)", "mismatch", True),
]
H1_COMBO = [("R4", 1), ("R5", -1)]
H2_COMBO = [("R4", 1), ("R3", -1), ("R2", -1), ("R1", 1)]
H3_COMBO = [("R6", 1), ("R4", -1)]


def equivalence_labels(name):
    return {(True, False): f"{name} supported (equivalent within the margin)",
            (False, True): f"{name} contradicted (difference detected)",
            (True, True): "equivalent within the margin, smaller difference detected",
            (False, False): "inconclusive"}


def difference_labels(name):
    return {(False, True): f"{name} supported (difference detected)",
            (True, False): f"{name} contradicted (equivalent within the margin)",
            (True, True): f"{name} supported, difference smaller than the margin",
            (False, False): "inconclusive"}


def load(results, arm, k):
    run_dir = Path(results) / f"uaq_{arm}_s{k}"
    vals, status = h1.load_run(run_dir)
    if vals is None:
        return None, status
    key = h1.STEP_KEYS["score"]
    scores = {}
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if key in r["data"]:
                    scores[int(r["step"])] = r["data"][key]
    return dict(vals, early_reward=100.0 * statistics.fmean(scores[s] for s in EARLY_STEPS)), status


def collapsed(vals):
    return vals["late_acc"] < COLLAPSE_BELOW


def combine(runs, combo, key, ratio, k):
    f = math.log if ratio else (lambda x: x)
    return sum(c * f(runs[(arm, k)][key]) for arm, c in combo)


def contrast(runs, title, combo, seeds, labels, exploratory=False):
    arms = [a for a, _ in combo]
    lines = [title]
    complete = [k for k in seeds if all(runs.get((a, k)) for a in arms)]
    if len(complete) < len(seeds):
        lines.append(f"  no verdict: {len(complete)} of {len(seeds)} registered seeds complete for {' '.join(arms)}")
        return lines
    use = complete
    if exploratory:
        use = [k for k in complete if not any(collapsed(runs[(a, k)]) for a in arms)]
        dropped = [k for k in complete if k not in use]
        lines.append(f"  EXPLORATORY: seeds without a collapsed run: {' '.join(map(str, use)) or 'none'}"
                     + (f" (dropped: {' '.join(map(str, dropped))})" if dropped else ""))
    if len(use) < 2:
        lines.append("  fewer than 2 seeds: no interval")
        return lines
    lines.append(f"  n = {len(use)} seeds, t on {len(use) - 1} df" + ("" if labels is None else
                 "; 90% CI decides equivalence, 95% CI decides difference"))
    for name, key, ratio in ENDPOINTS:
        vals = [combine(runs, combo, key, ratio, k) for k in use]
        mean, sd, ci90, ci95 = h1.paired_ci(vals)
        if ratio:
            lo, hi = math.log(h1.RATIO_BOUNDS[0]), math.log(h1.RATIO_BOUNDS[1])
            body = (f"bounds {h1.RATIO_BOUNDS[0]:.2f}-{h1.RATIO_BOUNDS[1]:.2f}; per seed "
                    + " ".join(f"{math.exp(v):.3f}" for v in vals)
                    + f"; ratio {math.exp(mean):.3f} (log SD {sd:.3f}); 90% CI [{math.exp(ci90[0]):.3f}, {math.exp(ci90[1]):.3f}];"
                    f" 95% CI [{math.exp(ci95[0]):.3f}, {math.exp(ci95[1]):.3f}]")
        else:
            lo, hi = -h1.MARGIN_POINTS, h1.MARGIN_POINTS
            body = (f"margin +-{h1.MARGIN_POINTS:.1f} pts; per seed " + " ".join(f"{v:+.2f}" for v in vals)
                    + f"; mean {mean:+.2f} (SD {sd:.2f}); 90% CI [{ci90[0]:+.2f}, {ci90[1]:+.2f}];"
                    f" 95% CI [{ci95[0]:+.2f}, {ci95[1]:+.2f}]")
        verdict = "" if labels is None else " -> " + labels[h1.classify(ci90, ci95, lo, hi)]
        lines.append(f"  {name:26s} {body}{verdict}")
    return lines


def report(results):
    shas = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("h23_analysis.py", "h1_analysis.py")}
    out = ["H2 and H3 analyses for S3, pre-registered 2026-11-11 (P1 note, section 7)",
           f"h23_analysis.py sha256 {shas['h23_analysis.py']}", f"h1_analysis.py  sha256 {shas['h1_analysis.py']}",
           f"results: {results}", ""]
    runs = {}
    out.append(f"{'run':14s} {'late acc':>9s} {'final acc':>9s} {'reward AUC':>10s} {'early rew':>9s} {'mean k3':>9s} {'entropy':>8s}  status")
    for arm in ARMS:
        for k in SEEDS_PER_ARM[arm]:
            vals, status = load(results, arm, k)
            runs[(arm, k)] = vals
            tag = f"uaq_{arm}_s{k}"
            if vals:
                flag = "  (collapsed)" if collapsed(vals) else ""
                out.append(f"{tag:14s} {vals['late_acc']:9.2f} {vals['final_acc']:9.2f} {vals['reward_auc']:10.2f}"
                           f" {vals['early_reward']:9.2f} {vals['mismatch']:9.2e} {vals['entropy']:8.3f}  {status}{flag}")
            else:
                out.append(f"{tag:14s} {'':9s} {'':9s} {'':10s} {'':9s} {'':9s} {'':8s}  {status}")
    out += ["", "REGISTERED", ""]
    out += contrast(runs, "H2 test: (R4 - R3) - (R2 - R1), paired by seed", H2_COMBO, SEEDS_H23, equivalence_labels("H2"))
    out.append("")
    out += contrast(runs, "H3 test: R6 - R4, paired by seed", H3_COMBO, SEEDS_H23, difference_labels("H3"))
    out += ["", "DESCRIPTIVE (no verdict)", ""]
    for title, combo in (("R4 - R3 (UAQ with INT8 rollouts)", [("R4", 1), ("R3", -1)]),
                         ("R2 - R1 (UAQ with BF16 rollouts)", [("R2", 1), ("R1", -1)]),
                         ("R6 - R3 (global learning rate x1.5)", [("R6", 1), ("R3", -1)])):
        out += contrast(runs, title, combo, SEEDS_H23, None)
        out.append("")
    out += ["EXPLORATORY (fixed 2026-11-11 after R5 seed 1 collapsed; does not replace the registered verdicts)", ""]
    out.append(f"collapsed runs per arm (late accuracy below {COLLAPSE_BELOW:.0f}%):")
    for arm in ARMS:
        done = [k for k in SEEDS_PER_ARM[arm] if runs.get((arm, k))]
        bad = [k for k in done if collapsed(runs[(arm, k)])]
        out.append(f"  {arm}: {len(bad)} of {len(done)} complete runs" + (f" (seeds {' '.join(map(str, bad))})" if bad else ""))
    out.append("")
    out += contrast(runs, "H1 without collapsed seeds: R4 - R5", H1_COMBO, SEEDS_H1, equivalence_labels("H1"), exploratory=True)
    out.append("")
    out += contrast(runs, "H2 without collapsed seeds", H2_COMBO, SEEDS_H23, equivalence_labels("H2"), exploratory=True)
    out.append("")
    out += contrast(runs, "H3 without collapsed seeds", H3_COMBO, SEEDS_H23, difference_labels("H3"), exploratory=True)
    return "\n".join(out) + "\n"


def self_test():
    h1.self_test()
    # interaction arithmetic: c = (R4 - R3) - (R2 - R1)
    runs = {("R1", 1): {"x": 55.0}, ("R2", 1): {"x": 57.0}, ("R3", 1): {"x": 50.0}, ("R4", 1): {"x": 53.0}}
    assert combine(runs, H2_COMBO, "x", False, 1) == 1.0
    runs = {("R1", 1): {"x": 1.0}, ("R2", 1): {"x": 2.0}, ("R3", 1): {"x": 4.0}, ("R4", 1): {"x": 2.0}}
    assert abs(math.exp(combine(runs, H2_COMBO, "x", True, 1)) - 0.25) < 1e-12   # (2/4)/(2/1)
    # H3 labels: a detected difference supports H3, equivalence contradicts it
    lab = difference_labels("H3")
    assert lab[h1.classify(*h1.paired_ci([-5.0, -4.5, -5.0])[2:], -2, 2)].startswith("H3 supported (difference")
    assert lab[h1.classify(*h1.paired_ci([0.1, -0.2, 0.0])[2:], -2, 2)].startswith("H3 contradicted")
    assert lab[h1.classify(*h1.paired_ci([1.0, -3.0, 4.0])[2:], -2, 2)] == "inconclusive"
    print("self-test passed: h1 statistics, H2 interaction arithmetic and H3 verdict labels as specified")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None, help="report file (default: <results>/h23_analysis.txt)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    text = report(a.results)
    sys.stdout.write(text)
    out = Path(a.out) if a.out else Path(a.results) / "h23_analysis.txt"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
