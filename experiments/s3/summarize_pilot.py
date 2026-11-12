"""S3: summarize a VeRL pilot run from its metrics.jsonl (written by pilot.sh) and extrapolate to a full run.

Usage: python summarize_pilot.py <results_dir> [--full-steps 435] [--val-every 29]

Prints the per-step timing breakdown (VeRL's timing_s/* metrics), reward and response length, the validation score,
and the projected wall-clock time of a full run: full_steps x the median step time (step 1 excluded: it includes
compilation and CUDA-graph capture), plus one validation every val_every steps (default: once per epoch).
Standard library only, so it runs with any Python.
"""
import argparse
import json
import os
import statistics

STAGES = ["gen", "reward", "old_log_prob", "ref", "values", "adv", "update_critic", "update_actor", "update_weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full-steps", type=int, default=435)
    ap.add_argument("--val-every", type=int, default=29)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.root, "metrics.jsonl")) if line.strip()]
    steps = sorted({r["step"] for r in rows})
    by = {}
    for r in rows:                                  # several log calls per step are merged
        by.setdefault(r["step"], {}).update(r["data"])

    present = [s for s in STAGES if any(f"timing_s/{s}" in by[k] for k in steps)]
    hdr = f"{'step':>4} {'total':>7} " + " ".join(f"{s[:12]:>12}" for s in present) + \
          f" {'score':>6} {'resp_len':>8} {'clipped':>7} {'tok/s':>7}"
    print(hdr)
    for k in steps:
        d = by[k]
        if "timing_s/step" not in d:
            continue
        print(f"{k:>4} {d['timing_s/step']:7.1f} " + " ".join(f"{d.get(f'timing_s/{s}', float('nan')):12.1f}" for s in present)
              + f" {d.get('critic/score/mean', float('nan')):6.3f} {d.get('response_length/mean', float('nan')):8.1f}"
              f" {d.get('response_length/clip_ratio', float('nan')):7.3f} {d.get('perf/throughput', float('nan')):7.0f}")

    t = [by[k]["timing_s/step"] for k in steps if "timing_s/step" in by[k]]
    warm = t[1:] if len(t) > 1 else t
    med = statistics.median(warm)
    val_t = [by[k]["timing_s/testing"] for k in steps if "timing_s/testing" in by[k]]
    val = [(k, key, v) for k in steps for key, v in by[k].items() if key.startswith("val-core/")]
    print(f"\nstep time: first {t[0]:.1f} s; median of the rest {med:.1f} s (min {min(warm):.1f}, max {max(warm):.1f})")
    for s in present:
        xs = [by[k][f"timing_s/{s}"] for k in steps[1:] if f"timing_s/{s}" in by[k]]
        if xs:
            print(f"  {s:<15} median {statistics.median(xs):7.1f} s ({100 * statistics.median(xs) / med:4.0f}% of a step)")
    if val_t:
        print(f"validation (greedy, test set): {val_t[-1]:.1f} s")
    for k, key, v in val:
        print(f"  step {k}: {key} = {v:.4f}")
    # Rollout-trainer mismatch (VeRL's rollout_corr/* metrics, logged for every arm) and QuRL's TIS/ACR activity
    # (qurl_verl/acr.py, INT8 arms only), averaged over steps.
    extra = [("rollout_corr/k3_kl", "mismatch KL (k3, per token)"), ("rollout_corr/kl", "mismatch KL (k1, per token)"),
             ("actor/qurl/tis_trunc_frac", "TIS: share of tokens with rho > C"),
             ("actor/qurl/acr_kept_frac", "ACR: share of tokens kept beyond 1 + eps"),
             ("rollout_corr/rollout_is_max", "largest rho in a batch")]
    shown = False
    for key, label in extra:
        xs = [by[k][key] for k in steps if key in by[k]]
        if xs:
            if not shown:
                print("\nrollout vs. trainer (mean over steps; min-max):")
                shown = True
            print(f"  {label:<42} {statistics.mean(xs):.3e}  ({min(xs):.3e}-{max(xs):.3e})")
    n_val = a.full_steps // a.val_every if a.val_every > 0 else 0
    full = a.full_steps * med + n_val * (val_t[-1] if val_t else 0.0)
    print(f"\nprojection for {a.full_steps} steps: {a.full_steps * med / 3600:.2f} h of training"
          + (f" + {n_val} validations = {full / 3600:.2f} h" if val_t else ""))


if __name__ == "__main__":
    main()
