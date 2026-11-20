#!/usr/bin/env python3
"""X7: S1's INT8 W8A8 identity check on vLLM 0.24 (the RL runs' stack), next to S1's own run on vLLM 0.33.

  python -I x7_analysis.py --results <copy>/results [--s1 DIR] [--out FILE]   (or --x7 <copy>/results/x7_s1_vllm024)
  python -I x7_analysis.py --self-test
Defaults: --x7 <results>/x7_s1_vllm024, where <results> is --results or results/ next to this file (a copy of
~/qurl/rev/results); --s1 ../h100/results/qwen0.5b-instruct next to this file (S1's 0.5B results); report to
<x7>/x7_analysis.txt and stdout. Standard library only.

Reported for X7 (vLLM 0.24) and, recomputed from S1's files with the same code, for S1 (vLLM 0.33):
  - the INT8 W8A8 row of S1's table (paper, Table "UAQ on vLLM's quantized kernels"): the shuffled re-run and UAQ
    s = 2 against the original (mean |dlog p| over response tokens; share of identical tokens), and at s = 1.5 the
    ratios UAQ / original of mean |dlog p| and of k3 KL, with 95% paired sequence-bootstrap CIs. Each checkpoint is
    compared with its own BF16 policy, and k3 at s = 1.5 is importance-weighted by pi_1.5 / pi_orig per token: the
    arithmetic of analyze_s1.py;
  - the same identity columns for BF16;
  - the INT8 kernel and the vLLM version of every run (kernels.txt);
  - X7 against S1 run by run (same checkpoint, same tokens), if both scored the same responses;
  - whether the stored INT8 codes and scales match S1's (int8_codes.txt, S1's compare_int8_codes.py).
Readings, fixed with this script:
  identity at s = 2   reproduced if INT8 W8A8 at s = 2 and the original give bit-identical log-probabilities on
                      every response token;
  s = 1.5             the ratios with their CIs; "no s-fold reduction" if the CI's lower end lies above the
                      reduction an s-fold error decrease would give (2/3 for mean |dlog p|, 4/9 for KL);
  kernel              CUTLASS if every INT8 engine start selected CutlassInt8ScaledMMLinearKernel.
The bootstrap uses Python's random module (seed 0), not numpy's generator, so the CIs recomputed for S1 can differ from
the published ones in the third decimal; the point estimates are the same. x7_run.sh also runs S1's own
analyze_e0.py and analyze_s1.py on X7's files (summary.txt, s1_analysis.txt).
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("none_orig", "none_orig_shuffled", "none_s2", "none_s1.5",
         "int8-w8a8_orig", "int8-w8a8_orig_shuffled", "int8-w8a8_s2", "int8-w8a8_s1.5")
Q = "int8-w8a8"
S = 1.5
CUTLASS = "CutlassInt8ScaledMMLinearKernel"
BOOT, SEED = 2000, 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lp(path, lengths=None):
    """Per-sequence log-probs of an e0_logprobs.py score file: (list, None) or (None, reason)."""
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path) as f:
            lp = [[float(v) for v in seq] for seq in json.load(f)["logprobs"]]
    except (OSError, ValueError, TypeError, KeyError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if not all(math.isfinite(v) for seq in lp for v in seq):
        return None, "non-finite values"
    if lengths is not None and [len(s) for s in lp] != lengths:
        return None, "not aligned with the original's tokens"
    return lp, None


def compare(a, b):
    """mean |d|, share of identical tokens, max |d| between two aligned runs."""
    d = [abs(x - y) for sa, sb in zip(a, b) for x, y in zip(sa, sb)]
    if not d:
        return 0.0, 1.0, 0.0
    return math.fsum(d) / len(d), sum(x == 0.0 for x in d) / len(d), max(d)


def seq_sums(mu, pi, w=None):
    """Per-sequence sums of |log mu - log pi| (unweighted) and w * ((r - 1) - log r), r = mu / pi (analyze_s1.py)."""
    mad, k3 = [], []
    for i, (m, p) in enumerate(zip(mu, pi)):
        wi = w[i] if w is not None else None
        mad.append(math.fsum(abs(a - b) for a, b in zip(m, p)))
        k3.append(math.fsum((math.expm1(a - b) - (a - b)) * (wi[t] if wi is not None else 1.0)
                            for t, (a, b) in enumerate(zip(m, p))))
    return {"mad": mad, "k3": k3}


def quantile(sorted_vals, q):
    """Linear-interpolation quantile (numpy's default) of a sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def ratio_ci(num, den, n_boot=BOOT, seed=SEED):
    """sum(num) / sum(den) and its 95% percentile CI over a paired bootstrap of sequences (None if sum(den) = 0)."""
    if math.fsum(den) <= 0:
        return None
    r = math.fsum(num) / math.fsum(den)
    rng = random.Random(seed)
    n = len(num)
    vals = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        d = math.fsum(den[i] for i in idx)
        if d > 0:
            vals.append(math.fsum(num[i] for i in idx) / d)
    vals.sort()
    if len(vals) < 2:
        return r, None, None
    return r, quantile(vals, 0.025), quantile(vals, 0.975)


def table_row(d, n_boot, out):
    """The INT8 (and BF16) row of S1's table from one results folder; None if its BF16 original is unusable."""
    base, why = load_lp(os.path.join(d, "score_none_orig.json"))
    if base is None:
        out(f"  {d}: score_none_orig.json is {why}")
        return None
    lengths = [len(s) for s in base]
    runs = {"none_orig": base}
    for name in NAMES[1:]:
        lp, why = load_lp(os.path.join(d, f"score_{name}.json"), lengths)
        if lp is None:
            out(f"  {os.path.basename(d)}/score_{name}.json: {why}")
        runs[name] = lp
    row = {"n_seq": len(lengths), "n_tok": sum(lengths), "runs": runs}
    for q in ("none", Q):
        o = runs[f"{q}_orig"]
        for v in ("orig_shuffled", "s2"):
            x = runs[f"{q}_{v}"]
            row[f"{q}_{v}"] = compare(o, x) if o is not None and x is not None else None
    pi_o, pi_s, mu_o, mu_s = runs["none_orig"], runs["none_s1.5"], runs[f"{Q}_orig"], runs[f"{Q}_s1.5"]
    if mu_o is not None:
        m_o = seq_sums(mu_o, pi_o)
        row["level"] = {k: math.fsum(v) / row["n_tok"] for k, v in m_o.items()}
        if mu_s is not None and pi_s is not None:
            w = [[math.exp(a - b) for a, b in zip(s, o)] for s, o in zip(pi_s, pi_o)]
            m_s = seq_sums(mu_s, pi_s, w)
            row["ratio"] = {k: ratio_ci(m_s[k], m_o[k], n_boot) for k in ("mad", "k3")}
            flat = [x for s in w for x in s]
            row["is_mean"] = math.fsum(flat) / len(flat)
            row["bf16_s15"] = compare(pi_s, pi_o)[0]
    return row


def parse_kernels(path):
    """kernels.txt of x7_run.sh: score file -> (vLLM versions, selected kernels)."""
    runs, cur = {}, None
    if not os.path.isfile(path):
        return runs
    with open(path) as f:
        for line in f:
            m = re.match(r"== score (\S+)", line.strip())
            if m:
                cur = m.group(1)
                runs.setdefault(cur, (set(), set()))
                continue
            if cur is None:
                continue
            m = re.search(r"vLLM (v\S+)", line)
            if m:
                runs[cur][0].add(m.group(1))
            m = re.search(r"Selected (\w+) for (\w+)", line)
            if m:
                runs[cur][1].add((m.group(1), m.group(2)))
    return runs


def fmt_id(c):
    if c is None:
        return "-"
    m, ident, _ = c
    return f"{'0' if m == 0 else f'{m:.1e}'}; {100 * ident:.2f}%"


def fmt_ratio(r):
    if r is None:
        return "-"
    v, lo, hi = r
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{v:.3f}"


def run(x7, s1, n_boot, out_path):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("X7: S1's INT8 W8A8 check (Qwen2.5-0.5B-Instruct) on vLLM 0.24, the RL runs' stack, next to S1 on vLLM 0.33")
    try:
        out(f"x7_analysis.py SHA-256 {sha256_file(os.path.abspath(__file__))}")
    except OSError:
        pass
    out(f"X7 results: {x7}")
    out(f"S1 results: {s1}")
    man = {}
    mpath = os.path.join(x7, "manifest.txt")
    if os.path.isfile(mpath):
        with open(mpath) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(("gpu:", "scoring stack", "S1 stack", "responses", "checkpoint ", "INT8 kernel",
                                    "VLLM_DISABLED_KERNELS")):
                    out(f"  manifest: {line}")
                if line.startswith("responses sha256:"):
                    man["sha"] = line.split(":", 1)[1].strip()
    same_resp = None
    s1_resp = os.path.join(s1, "responses.json")
    if man.get("sha") and os.path.isfile(s1_resp):
        same_resp = man["sha"] == sha256_file(s1_resp)
        out("responses: " + ("S1's (SHA-256 match)" if same_resp else "NOT S1's (SHA-256 differs)"))

    out("\nloading X7 (vLLM 0.24):")
    rx = table_row(x7, n_boot, out) if os.path.isdir(x7) else None
    out("loading S1 (vLLM 0.33):")
    rs = table_row(s1, n_boot, out) if os.path.isdir(s1) else None
    if rx is None:
        out("no usable X7 results")

    out("\n== The INT8 row of S1's table (identity: mean |dlog p|; identical tokens. s = 1.5: ratio UAQ / original, "
        "95% CI)")
    hdr = "{:<22} {:<8} {:>18} {:>18} {:>22} {:>22}"
    out(hdr.format("run", "quant", "shuffled re-run", "UAQ s=2 vs orig", "s=1.5 mean|dlog p|", "s=1.5 k3 KL"))
    for label, r in (("X7, vLLM 0.24", rx), ("S1, vLLM 0.33", rs)):
        if r is None:
            out(hdr.format(label, "-", "-", "-", "-", "-"))
            continue
        ratio = r.get("ratio", {})
        out(hdr.format(label, "INT8", fmt_id(r.get(f"{Q}_orig_shuffled")), fmt_id(r.get(f"{Q}_s2")),
                       fmt_ratio(ratio.get("mad")), fmt_ratio(ratio.get("k3"))))
        out(hdr.format("", "BF16", fmt_id(r.get("none_orig_shuffled")), fmt_id(r.get("none_s2")), "", ""))
    for label, r in (("X7", rx), ("S1", rs)):
        if r and "level" in r:
            out(f"{label}: {r['n_seq']} responses, {r['n_tok']} tokens; INT8 original vs BF16 original: mean |dlog p| "
                f"{r['level']['mad']:.4e}, k3 {r['level']['k3']:.4e}"
                + (f"; BF16 s = 1.5 vs original {r['bf16_s15']:.3e}, IS weight mean {r['is_mean']:.4f}"
                   if "bf16_s15" in r else ""))

    out("\n== Kernels and vLLM version per run (kernels.txt)")
    kern = parse_kernels(os.path.join(x7, "kernels.txt"))
    n_int8 = n_cut = 0
    for name in sorted(kern):
        vers, sel = kern[name]
        int8 = sorted(k for k, layer in sel if layer == "CompressedTensorsW8A8Int8")
        if int8:
            n_int8 += 1
            n_cut += int8 == [CUTLASS]
        out(f"  {name:<36} {', '.join(sorted(vers)) or '-':<10} {', '.join(f'{k} for {l}' for k, l in sorted(sel)) or '-'}")
    if not kern:
        out("  kernels.txt missing or empty")
    s1k = os.path.join(s1, "kernels.txt")
    if os.path.isfile(s1k):
        with open(s1k) as f:
            txt = f.read()
        out(f"  S1 (kernels.txt): {CUTLASS} selected in {txt.count('Selected ' + CUTLASS)} INT8 engine starts")

    out("\n== X7 against S1, run by run (same checkpoint and tokens; mean |d|, identical tokens, max |d|)")
    if rx is None or rs is None:
        out("  not possible: one side is missing")
    elif same_resp is False or rx["n_seq"] != rs["n_seq"]:
        out("  not possible: X7 and S1 scored different responses")
    else:
        lengths = [len(s) for s in rs["runs"]["none_orig"]]
        for name in NAMES:
            a, b = rx["runs"].get(name), rs["runs"].get(name)
            if a is None or b is None or [len(s) for s in a] != lengths:
                out(f"  {name:<26} -")
                continue
            m, ident, mx = compare(a, b)
            out(f"  {name:<26} {m:.2e}  {100 * ident:6.2f}%  {mx:.2e}")

    out("\n== INT8 codes and scales (S1's compare_int8_codes.py)")
    codes = {}
    for label, d in (("X7", x7), ("S1", s1)):
        p = os.path.join(d, "int8_codes.txt")
        if os.path.isfile(p):
            with open(p) as f:
                codes[label] = [l.strip() for l in f if l.strip()]
            for l in codes[label]:
                out(f"  {label}: {l}")
    if len(codes) == 2:
        out("  X7's stored INT8 checkpoints " + ("match S1's code statistics line for line" if codes["X7"] == codes["S1"]
                                                   else "DIFFER from S1's code statistics"))

    out("\n== Readings")
    if rx is not None:
        c = rx.get(f"{Q}_s2")
        if c is None:
            out("identity at s = 2 on vLLM 0.24: not measured")
        elif c[1] == 1.0:
            out(f"identity at s = 2 on vLLM 0.24: reproduced, bit-identical on all {rx['n_tok']} response tokens")
        else:
            out(f"identity at s = 2 on vLLM 0.24: NOT reproduced: mean |d| {c[0]:.2e}, {100 * c[1]:.2f}% identical, "
                f"max {c[2]:.2e} (shuffled floor: {fmt_id(rx.get(f'{Q}_orig_shuffled'))})")
        if "ratio" in rx:
            ok = all(rx["ratio"][k] is not None and rx["ratio"][k][1] is not None and rx["ratio"][k][1] > lim
                     for k, lim in (("mad", 1 / S), ("k3", 1 / S ** 2)))
            out(f"s = 1.5 on vLLM 0.24: mean |dlog p| {fmt_ratio(rx['ratio']['mad'])}, k3 {fmt_ratio(rx['ratio']['k3'])}"
                + (f"; S1 on vLLM 0.33: {fmt_ratio(rs['ratio']['mad'])}, {fmt_ratio(rs['ratio']['k3'])}"
                   if rs and "ratio" in rs else "")
                + ("; no s-fold reduction (both CIs above 2/3 and 4/9)" if ok else
                   "; an s-fold reduction is not excluded"))
        out(f"INT8 kernel on vLLM 0.24: {CUTLASS} in {n_cut} of {n_int8} INT8 engine starts"
            + (" (CUTLASS, as in the RL runs)" if n_int8 and n_cut == n_int8 else " (NOT CUTLASS throughout)"))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nreport written to {out_path}")
    return {"x7": rx, "s1": rs, "n_int8": n_int8, "n_cut": n_cut, "lines": lines}


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)

    tmp = tempfile.mkdtemp(prefix="x7_selftest_")
    try:
        rng = random.Random(5)
        lens = [rng.randint(5, 30) for _ in range(40)]
        pi_o = [[-rng.uniform(0.01, 3.0) for _ in range(n)] for n in lens]
        e = [[rng.uniform(0.05, 0.15) for _ in range(n)] for n in lens]
        pi_s = [[v + rng.uniform(-0.004, 0.004) for v in s] for s in pi_o]
        r = 0.99
        mu_o = [[p + x for p, x in zip(ps, es)] for ps, es in zip(pi_o, e)]
        mu_s = [[p + r * x for p, x in zip(ps, es)] for ps, es in zip(pi_s, e)]
        shuf = [list(s) for s in mu_o]
        shuf[0][0] += 0.5                              # one token differs in the shuffled re-run

        def write(d, name, lp):
            with open(os.path.join(d, f"score_{name}.json"), "w") as f:
                json.dump({"logprobs": lp}, f)

        def make(d, s2_same=True, kernel=CUTLASS):
            os.makedirs(d)
            for name, lp in (("none_orig", pi_o), ("none_orig_shuffled", pi_o), ("none_s2", pi_o), ("none_s1.5", pi_s),
                             ("int8-w8a8_orig", mu_o), ("int8-w8a8_orig_shuffled", shuf),
                             ("int8-w8a8_s2", mu_o if s2_same else shuf), ("int8-w8a8_s1.5", mu_s)):
                write(d, name, lp)
            with open(os.path.join(d, "responses.json"), "w") as f:
                json.dump([{"prompt_ids": [1], "response_ids": [0] * n} for n in lens], f)
            with open(os.path.join(d, "manifest.txt"), "w") as f:
                f.write(f"responses sha256: {sha256_file(os.path.join(d, 'responses.json'))}\n")
            with open(os.path.join(d, "kernels.txt"), "w") as f:
                for name in ("score_int8-w8a8_orig.json", "score_int8-w8a8_s2.json", "score_none_orig.json"):
                    f.write(f"== score {name}\n  vLLM v0.24.0\n")
                    if "int8" in name:
                        f.write(f"  Selected {kernel} for CompressedTensorsW8A8Int8\n")
            with open(os.path.join(d, "int8_codes.txt"), "w") as f:
                f.write("s2: Q/K/V INT8 codes changed 0.000%\n")

        x7, s1 = os.path.join(tmp, "x7"), os.path.join(tmp, "s1")
        make(x7)
        make(s1, s2_same=False, kernel="TritonInt8ScaledMMLinearKernel")
        quiet = open(os.devnull, "w")
        old = sys.stdout
        sys.stdout = quiet
        try:
            res = run(x7, s1, 300, os.path.join(tmp, "report.txt"))
        finally:
            sys.stdout = old
        rx, rs = res["x7"], res["s1"]
        n_tok = sum(lens)
        check(rx[f"{Q}_s2"] == (0.0, 1.0, 0.0), "identity at s = 2 (X7)")
        check(abs(rx[f"{Q}_orig_shuffled"][0] - 0.5 / n_tok) < 1e-15 and
              abs(rx[f"{Q}_orig_shuffled"][1] - (n_tok - 1) / n_tok) < 1e-15, "shuffled floor")
        check(rs[f"{Q}_s2"][1] < 1.0, "S1's (synthetic) s = 2 differs")
        check(abs(rx["ratio"]["mad"][0] - r) < 1e-12, f"mad ratio {rx['ratio']['mad'][0]} vs {r}")
        k3 = (math.fsum(math.exp(a - b) * (math.expm1(r * x) - r * x)
                        for sa, sb, se in zip(pi_s, pi_o, e) for a, b, x in zip(sa, sb, se))
              / math.fsum(math.expm1(x) - x for se in e for x in se))
        check(abs(rx["ratio"]["k3"][0] - k3) < 1e-12, f"k3 ratio {rx['ratio']['k3'][0]} vs {k3}")
        lo, hi = rx["ratio"]["mad"][1:]
        check(lo <= r <= hi and hi - lo < 1e-9, "the mad CI is a point when every token has the same ratio")
        check(res["n_int8"] == 2 and res["n_cut"] == 2, "kernel count")
        txt = "\n".join(res["lines"])
        check("identity at s = 2 on vLLM 0.24: reproduced" in txt, "identity reading")
        check("no s-fold reduction" in txt, "s-fold reading")
        check("match S1's code statistics" in txt, "codes comparison")
        check("int8-w8a8_s2" in txt and "100.00%" in txt, "run-by-run comparison")
        check(os.path.isfile(os.path.join(tmp, "report.txt")), "report written")
        # Missing and misaligned files, a missing S1 folder.
        os.remove(os.path.join(x7, "score_int8-w8a8_s1.5.json"))
        lp, _ = load_lp(os.path.join(x7, "score_int8-w8a8_s2.json"))
        lp[0] = lp[0][:-1]
        write(x7, "int8-w8a8_s2", lp)
        sys.stdout = quiet
        try:
            res = run(x7, os.path.join(tmp, "nowhere"), 100, None)
        finally:
            sys.stdout = old
        check(res["s1"] is None and "ratio" not in res["x7"] and res["x7"][f"{Q}_s2"] is None, "missing files")
        check("identity at s = 2 on vLLM 0.24: not measured" in "\n".join(res["lines"]), "missing s = 2 reading")
        quiet.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("self-test " + ("passed" if ok else "FAILED") + ": identity and noise-floor columns, ratios against closed "
          "forms, bootstrap CI, kernel parsing, run-by-run comparison, missing and misaligned files")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"), help="a copy of ~/qurl/rev/results")
    ap.add_argument("--x7", help="X7's folder (default <results>/x7_s1_vllm024)")
    ap.add_argument("--s1", default=os.path.join(HERE, "..", "h100", "results", "qwen0.5b-instruct"))
    ap.add_argument("--out", help="report file (default <x7>/x7_analysis.txt; '-' for stdout only)")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.x7 = a.x7 or os.path.join(a.results, "x7_s1_vllm024")
    out_path = None if a.out == "-" else (a.out or (os.path.join(a.x7, "x7_analysis.txt") if os.path.isdir(a.x7) else None))
    run(os.path.abspath(a.x7), os.path.abspath(a.s1), a.boot, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
