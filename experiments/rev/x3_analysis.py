"""Revision experiments X1-X3, pre-registered analysis. How much of S3's INT8 accuracy gap is the cost of scoring with
the INT8 policy, and how much is the cost of training with INT8 rollouts?

  python -I x3_analysis.py --results <copy of ~/qurl/rev/results> [--out report.txt]
  python -I x3_analysis.py --self-test

Data
  X3  S3's arms R1 (BF16 rollouts) and R3 (QuRL: INT8 rollouts, TIS, ACR), seeds 1-3, rerun with S3's call
      (x3_run.sh).
      - VeRL validates each run in its own rollout format at steps 0, 29, ..., 435
        (metrics.jsonl, val-core/openai/gsm8k/acc/mean@1).
      - score_gsm8k.py scores the actor's FP32 weights at steps 29, ..., 435 in BF16 and in INT8 W8A8
        (step<t>_<bf16|int8>.json).
  X1  The base model (step 0 of every run) and S3's UAQ checkpoint, in both formats (x1_base/).
  X2  E2's run (arm R3, seed 1) at its recorded syncs, in both formats (x2_e2dumps/), if the dumps still existed.

Quantities, in accuracy points. "Late" is the mean over steps 319, 348, 377, 406 and 435, S3's primary endpoint.
  A_f(arm, seed, t)  score_gsm8k.py's accuracy for that checkpoint in format f.
  T  = A_bf16,late(R1) - A_bf16,late(R3)  training effect: both arms' trained weights scored in BF16.
  C3 = A_bf16,late(R3) - A_int8,late(R3)  scoring cost on INT8-trained weights.
  C1 = A_bf16,late(R1) - A_int8,late(R1)  scoring cost on BF16-trained weights.
  D  = C3 - C1                            does INT8 training change how much INT8 scoring costs?
  G  = A_bf16,late(R1) - A_int8,late(R3)  the gap as S3 measured it; G = T + C3 exactly.
Each contrast is paired by seed: the mean with t-intervals on n - 1 degrees of freedom (90% and 95%).

Gate: is the scorer valid? This is checked first. For every X3 run and validation step t (0, 29, ..., 435):
  - compare the scorer's accuracy in the run's own format (R1: bf16, R3: int8; at t = 0, X1's base-model scores)
    with VeRL's own validation at t;
  - per format: mean signed difference within +-0.5 points, mean absolute difference at most 1.0 point, and no
    single difference above 3.0 points.
The scorer is validated if both formats pass. Otherwise every verdict is reported as "(scorer not validated)".

Decision rules for T, C3, C1 and D. The margin is +-2 points, the same as S3's H1-H3.
  positive       the 95% CI lies above 0 ("within the margin" is added if the 90% CI's upper end is at most +2);
  negative       the 95% CI lies below 0 (likewise);
  no effect      neither, and the 90% CI lies within [-2, +2];
  inconclusive   otherwise.
A contrast needs all three seeds with all five late steps scored in both formats; otherwise it is "incomplete".
Reading: if T is positive, INT8 rollouts cost accuracy even when the trained weights are scored in BF16. If C3 is
positive, part of S3's gap is the cost of INT8 scoring.

Descriptive, no verdicts:
  - G and the share C3 / G;
  - the same contrasts at step 435 alone;
  - the time courses of C1, C3 and T (means over seeds) next to VeRL's own validations;
  - X1: C0 = A_bf16 - A_int8 for the base model, the UAQ checkpoint's scores against the base model's (equal by
    Theorem 1, up to FP32 rounding of W / 1.5), and the share of problems scored alike;
  - X2: C at each sync of E2's run.

Fixed in writing before any of these data existed, and frozen by SHA-256 (registration.sha256).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile

STEPS = list(range(29, 436, 29))   # 29, 58, ..., 435
LATE = [319, 348, 377, 406, 435]
SEEDS = (1, 2, 3)
ARMS = ("R1", "R3")
OWN = {"R1": "bf16", "R3": "int8"}
VAL_KEY = "val-core/openai/gsm8k/acc/mean@1"
MARGIN = 2.0
GATE_BIAS, GATE_MEAN_ABS, GATE_MAX_ABS = 0.5, 1.0, 3.0
# two-sided t quantiles: (90%, 95%) for df = 1..10
T_Q = {1: (6.313752, 12.706205), 2: (2.919986, 4.302653), 3: (2.353363, 3.182446), 4: (2.131847, 2.776445),
       5: (2.015048, 2.570582), 6: (1.943180, 2.446912), 7: (1.894579, 2.364624), 8: (1.859548, 2.306004),
       9: (1.833113, 2.262157), 10: (1.812461, 2.228139)}


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def score(path):
    """Accuracy in points from a score_gsm8k.py JSON, or None."""
    d = jload(path)
    return None if d is None else 100.0 * d["accuracy"]


def verl_val(run_dir):
    """VeRL's own validation accuracy (points) by step, from the run's metrics.jsonl."""
    out = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("data", {}).get(VAL_KEY)
        if v is not None:
            out[int(r["step"])] = 100.0 * v
    return out


def tint(xs):
    """Mean and (90%, 95%) t-intervals of a list; intervals are None for fewer than two values."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    q90, q95 = T_Q[n - 1]
    h90, h95 = q90 * sd / math.sqrt(n), q95 * sd / math.sqrt(n)
    return m, (m - h90, m + h90), (m - h95, m + h95)


def verdict(xs, n_needed):
    if len(xs) < n_needed:
        return "incomplete"
    _, c90, c95 = tint(xs)
    within = -MARGIN <= c90[0] and c90[1] <= MARGIN
    if c95[0] > 0:
        return "positive" + (" (within the margin)" if c90[1] <= MARGIN else "")
    if c95[1] < 0:
        return "negative" + (" (within the margin)" if c90[0] >= -MARGIN else "")
    return "no effect beyond +-2 points" if within else "inconclusive"


def fmt_ci(xs):
    m, c90, c95 = tint(xs)
    if c90 is None:
        return f"{m:+.2f}"
    return f"{m:+.2f}, 90% CI [{c90[0]:+.2f}, {c90[1]:+.2f}], 95% CI [{c95[0]:+.2f}, {c95[1]:+.2f}]"


def analyse(results):
    lines = []
    out = lines.append
    x1 = os.path.join(results, "x1_base")
    base = {f: score(os.path.join(x1, f"orig_{f}.json")) for f in ("bf16", "int8")}
    uaq = {f: score(os.path.join(x1, f"uaq1.5_{f}.json")) for f in ("bf16", "int8")}

    # Per run: scores by step and format, and VeRL's validations.
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            d = os.path.join(results, "x3", f"rev_x3_{arm}_s{s}")
            acc = {(t, f): score(os.path.join(d, f"step{t}_{f}.json")) for t in STEPS for f in ("bf16", "int8")}
            for f in ("bf16", "int8"):
                acc[(0, f)] = base[f]
            runs[(arm, s)] = {"acc": acc, "val": verl_val(d), "exists": os.path.isdir(d)}

    # Gate.
    out("X1-X3 analysis (pre-registered): INT8 scoring cost and INT8 training effect")
    out("")
    out("Gate: score_gsm8k.py against VeRL's own validation, in each run's own format (points)")
    gate_ok = True
    for arm in ARMS:
        f = OWN[arm]
        diffs = []
        for s in SEEDS:
            r = runs[(arm, s)]
            for t in [0] + STEPS:
                ours, theirs = r["acc"].get((t, f)), r["val"].get(t)
                if ours is not None and theirs is not None:
                    diffs.append(ours - theirs)
        if not diffs:
            out(f"  {f} ({arm} runs): no pairs")
            gate_ok = False
            continue
        bias = sum(diffs) / len(diffs)
        mabs = sum(abs(x) for x in diffs) / len(diffs)
        mx = max(abs(x) for x in diffs)
        ok = abs(bias) <= GATE_BIAS and mabs <= GATE_MEAN_ABS and mx <= GATE_MAX_ABS
        gate_ok &= ok
        out(f"  {f} ({arm} runs): {len(diffs)} pairs; mean signed {bias:+.2f} (limit +-{GATE_BIAS}), mean absolute "
            f"{mabs:.2f} (limit {GATE_MEAN_ABS}), max absolute {mx:.2f} (limit {GATE_MAX_ABS}) -> "
            f"{'pass' if ok else 'FAIL'}")
    out(f"  -> scorer {'validated' if gate_ok else 'NOT validated'}")
    tag = "" if gate_ok else " (scorer not validated)"
    out("")

    # Late endpoints per seed.
    def late(arm, s, f):
        vals = [runs[(arm, s)]["acc"].get((t, f)) for t in LATE]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)

    per = {k: [] for k in ("T", "C3", "C1", "D", "G")}
    fin = {k: [] for k in ("T", "C3", "C1", "D", "G")}
    out("Late accuracy per seed (mean over steps 319-435), points:")
    out("  seed   R1 bf16   R1 int8   R3 bf16   R3 int8 |      T     C3     C1      D      G")
    for s in SEEDS:
        a = {(arm, f): late(arm, s, f) for arm in ARMS for f in ("bf16", "int8")}
        if any(v is None for v in a.values()):
            out(f"  {s}      incomplete")
            continue
        T = a[("R1", "bf16")] - a[("R3", "bf16")]
        C3 = a[("R3", "bf16")] - a[("R3", "int8")]
        C1 = a[("R1", "bf16")] - a[("R1", "int8")]
        G = a[("R1", "bf16")] - a[("R3", "int8")]
        for k, v in (("T", T), ("C3", C3), ("C1", C1), ("D", C3 - C1), ("G", G)):
            per[k].append(v)
        out(f"  {s}    {a[('R1', 'bf16')]:7.2f}   {a[('R1', 'int8')]:7.2f}   {a[('R3', 'bf16')]:7.2f}   "
            f"{a[('R3', 'int8')]:7.2f} | {T:+6.2f} {C3:+6.2f} {C1:+6.2f} {C3 - C1:+6.2f} {G:+6.2f}")
        f435 = {(arm, f): runs[(arm, s)]["acc"].get((435, f)) for arm in ARMS for f in ("bf16", "int8")}
        if all(v is not None for v in f435.values()):
            fin["T"].append(f435[("R1", "bf16")] - f435[("R3", "bf16")])
            fin["C3"].append(f435[("R3", "bf16")] - f435[("R3", "int8")])
            fin["C1"].append(f435[("R1", "bf16")] - f435[("R1", "int8")])
            fin["D"].append(fin["C3"][-1] - fin["C1"][-1])
            fin["G"].append(f435[("R1", "bf16")] - f435[("R3", "int8")])
    out("")
    out("Registered contrasts (late accuracy, paired by seed):")
    names = {"T": "T  training effect (both scored in BF16)", "C3": "C3 INT8 scoring cost, R3's weights",
             "C1": "C1 INT8 scoring cost, R1's weights", "D": "D  = C3 - C1"}
    for k in ("T", "C3", "C1", "D"):
        xs = per[k]
        res = fmt_ci(xs) if xs else "no complete seed"
        out(f"  {names[k]}: {res} -> {verdict(xs, len(SEEDS))}{tag}")
    if per["G"]:
        g = sum(per["G"]) / len(per["G"])
        c3 = sum(per["C3"]) / len(per["C3"])
        share = f"{100 * c3 / g:.0f}%" if abs(g) > 1e-9 else "n/a"
        out(f"  G  the gap as S3 measured it (descriptive): {fmt_ci(per['G'])}; share from INT8 scoring C3/G = "
            f"{share}")
    out("")
    out("Step 435 alone (descriptive):")
    for k in ("T", "C3", "C1", "D", "G"):
        if fin[k]:
            out(f"  {k}: {fmt_ci(fin[k])}")
    out("")

    # Time course.
    out("Time course, means over seeds (points; n = seeds available):")
    out("  step  n  R1 bf16  R1 int8  R3 bf16  R3 int8 |   C1    C3     T | VeRL R1  VeRL R3")
    for t in [0] + STEPS:
        rows = []
        for s in SEEDS:
            a = {(arm, f): runs[(arm, s)]["acc"].get((t, f)) for arm in ARMS for f in ("bf16", "int8")}
            if all(v is not None for v in a.values()):
                rows.append(a)
        vr1 = [runs[("R1", s)]["val"].get(t) for s in SEEDS]
        vr3 = [runs[("R3", s)]["val"].get(t) for s in SEEDS]
        vr1 = [v for v in vr1 if v is not None]
        vr3 = [v for v in vr3 if v is not None]
        if not rows:
            continue
        m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        out(f"  {t:4d}  {len(rows)}  {m[('R1', 'bf16')]:7.2f}  {m[('R1', 'int8')]:7.2f}  {m[('R3', 'bf16')]:7.2f}  "
            f"{m[('R3', 'int8')]:7.2f} | {m[('R1', 'bf16')] - m[('R1', 'int8')]:+5.2f} "
            f"{m[('R3', 'bf16')] - m[('R3', 'int8')]:+5.2f} {m[('R1', 'bf16')] - m[('R3', 'bf16')]:+5.2f} | "
            f"{(sum(vr1) / len(vr1)) if vr1 else float('nan'):7.2f}  {(sum(vr3) / len(vr3)) if vr3 else float('nan'):7.2f}")
    out("")

    # X1.
    out("X1, step 0 (descriptive):")
    if base["bf16"] is not None and base["int8"] is not None:
        out(f"  base model: BF16 {base['bf16']:.2f}, INT8 {base['int8']:.2f}, C0 = {base['bf16'] - base['int8']:+.2f}")
    else:
        out("  base model: missing")
    for f in ("bf16", "int8"):
        if uaq[f] is not None and base[f] is not None:
            a = jload(os.path.join(x1, f"orig_{f}.json"))["per_problem"]
            b = jload(os.path.join(x1, f"uaq1.5_{f}.json"))["per_problem"]
            same = sum(x == y for x, y in zip(a, b)) / len(a)
            out(f"  UAQ checkpoint ({f}): {uaq[f]:.2f} vs base {base[f]:.2f} ({uaq[f] - base[f]:+.2f}); "
                f"problems scored alike {100 * same:.1f}%")
    out("")

    # X2.
    x2 = os.path.join(results, "x2_e2dumps")
    out("X2, E2's run (R3, seed 1) at its recorded syncs (descriptive):")
    if os.path.exists(os.path.join(x2, "SKIPPED")):
        out("  skipped: " + open(os.path.join(x2, "SKIPPED")).read().strip())
    else:
        syncs = sorted({int(f[4:8]) for f in (os.listdir(x2) if os.path.isdir(x2) else [])
                        if f.startswith("sync") and f.endswith("_bf16.json")})
        for k in syncs:
            b = score(os.path.join(x2, f"sync{k:04d}_bf16.json"))
            i = score(os.path.join(x2, f"sync{k:04d}_int8.json"))
            if b is not None and i is not None:
                out(f"  sync {k:3d} (after step {k - 1:3d}): BF16 {b:6.2f}, INT8 {i:6.2f}, C {b - i:+.2f}")
        if not syncs:
            out("  no results")
    return "\n".join(lines), {"gate": gate_ok, **{k: verdict(per[k], len(SEEDS)) for k in ("T", "C3", "C1", "D")}}


def _write_score(path, acc, n=1319, rng=None):
    k = int(round(acc * n))
    per = [1.0] * k + [0.0] * (n - k)
    if rng is not None:
        rng.shuffle(per)
    with open(path, "w") as f:
        json.dump({"accuracy": k / n, "per_problem": per}, f)


def self_test():
    rng = random.Random(0)

    def build(root, bias_int8_val=0.0):
        os.makedirs(os.path.join(root, "x1_base"))
        for name, b, i in (("orig", 0.30, 0.28), ("uaq1.5", 0.30, 0.28)):
            _write_score(os.path.join(root, "x1_base", f"{name}_bf16.json"), b)
            _write_score(os.path.join(root, "x1_base", f"{name}_int8.json"), i)
        for s in SEEDS:
            noise = {t: rng.gauss(0, 0.002) for t in STEPS}
            for arm in ARMS:
                d = os.path.join(root, "x3", f"rev_x3_{arm}_s{s}")
                os.makedirs(d)
                recs = []
                for t in [0] + STEPS:
                    lvl = 0.30 + 0.25 * min(1, t / 200) + noise.get(t, 0)
                    bf = lvl - (0.03 if arm == "R3" else 0.0) * min(1, t / 100)   # T = 3 points late
                    it = bf - (0.02 if arm == "R3" else 0.01)                        # C3 = 2, C1 = 1
                    if t == 0:
                        bf, it = 0.30, 0.28
                    else:
                        _write_score(os.path.join(d, f"step{t}_bf16.json"), bf)
                        _write_score(os.path.join(d, f"step{t}_int8.json"), it)
                    own = bf if arm == "R1" else it + bias_int8_val
                    recs.append({"step": t, "data": {VAL_KEY: own + rng.gauss(0, 0.002)}})
                with open(os.path.join(d, "metrics.jsonl"), "w") as f:
                    f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        os.makedirs(os.path.join(root, "x2_e2dumps"))
        with open(os.path.join(root, "x2_e2dumps", "SKIPPED"), "w") as f:
            f.write("synthetic")

    ok = True
    for bias, want_gate in ((0.0, True), (0.02, False)):
        root = tempfile.mkdtemp(prefix="x3_selftest_")
        try:
            build(root, bias)
            text, v = analyse(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        checks = {"gate": v["gate"] == want_gate, "T positive": v["T"].startswith("positive"),
                  "C3 positive": v["C3"].startswith("positive"), "C1 positive": v["C1"].startswith("positive"),
                  "D positive": v["D"].startswith("positive")}
        print(f"self-test (INT8 validation bias {bias * 100:+.0f} points): " +
              ", ".join(f"{k} {'ok' if c else 'WRONG'}" for k, c in checks.items()))
        ok &= all(checks.values())
        if not all(checks.values()):
            print(text)
    # Verdict rules on fixed inputs.
    cases = [([3.1, 2.9, 3.0], "positive"), ([-3.1, -2.9, -3.0], "negative"), ([0.1, -0.1, 0.0], "no effect"),
             ([4.0, -4.0, 0.5], "inconclusive"), ([1.0, 1.1, 0.9], "positive (within the margin)"),
             ([1.0, 1.1], "incomplete")]
    for xs, want in cases:
        got = verdict(xs, 3)
        good = got.startswith(want)
        ok &= good
        print(f"  verdict({xs}) = {got} {'ok' if good else 'WRONG (want ' + want + ')'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.results:
        ap.error("--results is required")
    text, _ = analyse(a.results)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
