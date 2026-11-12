"""E0 step 4: compare per-token log-probs across configurations.

Usage: python analyze_e0.py <results_dir>

For each quantization method found (files score_<quant>_<variant>.json), compares against the original
checkpoint under the same quantization:
  - noise floor: original vs. original with shuffled batching
  - run-to-run determinism, if present: original vs. a same-order repeat, s=2 vs. its repeat (repeat_fp8.sh)
  - UAQ s=2 (exact in BF16) and s=1.5 vs. original
and reports the train-rollout mismatch of each quantized checkpoint against the BF16 original:
mean |log mu - log pi| and the sampled-token KL estimate E_pi[log pi - log mu] (tokens were sampled
from the BF16 model, so this estimates KL(pi || mu)).
"""
import glob
import json
import os
import sys

import numpy as np


def load(path):
    d = json.load(open(path))
    return np.concatenate([np.array(x, dtype=np.float64) for x in d["logprobs"]]), d.get("vllm_quant")


def cmp(a, b):
    d = np.abs(a - b)
    return {"mean_abs": d.mean(), "p99_abs": np.quantile(d, 0.99), "max_abs": d.max(),
            "frac_identical": np.mean(a == b)}


def main():
    root = sys.argv[1]
    runs = {os.path.basename(p)[6:-5]: load(p) for p in sorted(glob.glob(os.path.join(root, "score_*.json")))}
    print("runs:", {k: v[1] for k, v in runs.items()})
    bf16 = runs["none_orig"][0]
    fmt = "{:<34} {:>10} {:>10} {:>10} {:>12}"
    print("\n" + fmt.format("comparison (per response token)", "mean|d|", "p99|d|", "max|d|", "identical"))
    quants = sorted({k.split("_")[0] for k in runs})
    for q in quants:
        base = f"{q}_orig"
        pairs = [(base, f"{q}_orig_shuffled"), (base, f"{q}_orig_repeat"), (base, f"{q}_s2"), (base, f"{q}_s1.5"),
                 (f"{q}_s2", f"{q}_s2_repeat"), (f"{q}_orig_repeat", f"{q}_s2_repeat")]   # repeats: same batch order
        for a, b in pairs:
            if a in runs and b in runs:
                c = cmp(runs[a][0], runs[b][0])
                print(fmt.format(f"{a} vs {b}", f"{c['mean_abs']:.2e}", f"{c['p99_abs']:.2e}",
                                 f"{c['max_abs']:.2e}", f"{100 * c['frac_identical']:.2f}%"))
    print("\nmismatch vs BF16 original (pi):")
    print("{:<20} {:>14} {:>22}".format("rollout config", "mean|logmu-logpi|", "KL(pi||mu) estimate"))
    for k, (lp, _) in runs.items():
        if k.startswith("none"):
            continue
        print("{:<20} {:>14.4e} {:>22.4e}".format(k, np.abs(lp - bf16).mean(), (bf16 - lp).mean()))
    print(f"\nresponse tokens compared: {len(bf16)}")


if __name__ == "__main__":
    main()
