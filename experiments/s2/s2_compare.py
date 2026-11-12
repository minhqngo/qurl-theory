"""S2: compare runs of s2_train.py bit-for-bit against a reference run (default R4).

Usage: python s2_compare.py <results_dir> [--ref R4]

For every other run in <results_dir> (subdirectories with log.jsonl) it reports, step by step, whether the sampled
tokens, the rollout log-probabilities (mu), the proximal log-probabilities (pi_old) and the parameters mapped to
original coordinates (per group W_QKV, gamma, rest) are bit-identical to the reference, and the first divergent
step. For parameter snapshots saved by both runs it reports the share of identical elements and the largest
difference per group. It also summarises how active clipping, TIS truncation and ACR were in the reference run,
since rule (c) and the clip only matter when they bind.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch


def load(d):
    return [json.loads(line) for line in open(os.path.join(d, "log.jsonl"))]


def group_of(name):
    if re.search(r"\.self_attn\.(q|k|v)_proj\.weight$", name):
        return "W"
    if name.endswith("input_layernorm.weight"):
        return "gamma"
    return "rest"


def first_diff(ref, run, key, sub=None):
    for a, b in zip(ref, run):
        va, vb = (a[key][sub], b[key][sub]) if sub else (a.get(key), b.get(key))
        if va != vb:
            return a["step"]
    return None


def snapshot_diff(pa, pb):
    A, B = torch.load(pa), torch.load(pb)
    out = {}
    for g in ("W", "gamma", "rest"):
        names = [n for n in A if group_of(n) == g]
        same = sum(int((A[n] == B[n]).sum()) for n in names)
        tot = sum(A[n].numel() for n in names)
        mx = max(float((A[n] - B[n]).abs().max()) for n in names)
        out[g] = (100.0 * same / tot, mx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", default="R4")
    a = ap.parse_args()
    runs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in sorted(glob.glob(os.path.join(a.root, "*", "log.jsonl")))}
    ref = load(runs[a.ref])
    steps = [r for r in ref if r["step"] > 0]
    print(f"reference {a.ref}: {len(steps)} rollout steps, args {ref[0]['args']['arm']} s={ref[0]['args']['s']} "
          f"lr={ref[0]['args']['lr']} tau={ref[0]['args']['tau']} on {ref[0].get('device')} (torch {ref[0].get('torch')})")
    clip = np.mean([r["clip_frac"] for r in steps])
    norms = [n for r in steps for n in r["grad_norms"]]
    print(f"  activity: clipping on {100 * clip:.0f}% of updates (median grad norm {np.median(norms):.3g}, "
          f"tau {ref[0]['args']['tau']}); TIS truncation on {100 * np.mean([r['tis_trunc_frac'] for r in steps]):.2f}% "
          f"of tokens; ACR kept {sum(r['acr_kept'] for r in steps)} token terms; "
          f"mean reward {np.mean([r['reward'] for r in steps]):.3f}\n")

    keys = [("tok", None), ("mu", None), ("pi_old", None)] + [("params", g) for g in ("W", "gamma", "rest", "all")]
    hdr = f"{'run':<12}" + "".join(f"{(k if not g else 'params.' + g):>16}" for k, g in keys)
    print("steps identical to the reference (first divergent step in brackets):")
    print(hdr)
    for name, d in runs.items():
        if name == a.ref:
            continue
        run = load(d)
        if run[0]["params"] != ref[0]["params"]:
            print(f"{name:<12} initial parameters differ from the reference: different model or mapping")
            continue
        n = min(len(run), len(ref)) - 1
        cells = []
        for k, g in keys:
            same = sum((r[k][g] if g else r[k]) == (q[k][g] if g else q[k]) for r, q in zip(ref[1:], run[1:]))
            fd = first_diff(ref[1:], run[1:], k, g)
            cells.append(f"{same}/{n}" + (f" [{fd}]" if fd else ""))
        print(f"{name:<12}" + "".join(f"{c:>16}" for c in cells))

    print("\nparameter snapshots vs. reference (share of identical elements; max |difference|):")
    for name, d in runs.items():
        if name == a.ref:
            continue
        for pa in sorted(glob.glob(os.path.join(runs[a.ref], "params_step*.pt"))):
            pb = os.path.join(d, os.path.basename(pa))
            if not os.path.exists(pb):
                continue
            res = snapshot_diff(pa, pb)
            step = re.search(r"step(\d+)", pa).group(1)
            print(f"  {name:<10} step {step:>3}: " + "; ".join(
                f"{g} {pct:.4f}% ({mx:.2e})" for g, (pct, mx) in res.items()))


if __name__ == "__main__":
    main()
