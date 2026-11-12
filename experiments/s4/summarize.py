"""S4: summarize one run from its results folder (metrics, INT8 refit log, per-sync E3 log) and project its run time.

  python summarize.py <results_dir> [--full-steps 200] [--val-every 20] [--save-every 20]
  python summarize.py <results_dir> --brief      (one line: progress.sh)

Reads metrics.part1.jsonl, metrics.part2.jsonl, ... then metrics.jsonl (attempts of a resumed run; merged per metric,
the later attempt winning for a metric logged twice), e3_int8.jsonl and int8_refit.txt. Standard library only.
Prints: step-time breakdown (VeRL's timing_s/*) and the projection of a full run (median step time, step 1 excluded,
plus validations and checkpoint saves); training score (DAPO's +1/-1 rule plus the overlong penalty) and response
length; AIME 2024 validation (Avg@32 when the validation file has 32 copies per problem); rollout-trainer mismatch and
TIS/ACR activity; INT8 refit coverage; and the E2/E3 window of the analysis plan: mean code-flip share per projection
over syncs 2-21 (the updates of steps 1-20) of the first attempt, with Proposition 4's prediction for tracked layers.
"""
import argparse
import glob
import json
import os
import re
import statistics

FLIP_WINDOW = range(2, 22)   # syncs 2-21: the weight changes made by steps 1-20


def load_metrics(root):
    parts = sorted(glob.glob(os.path.join(root, "metrics.part*.jsonl")),
                   key=lambda p: int(re.search(r"part(\d+)", p).group(1)))
    by = {}
    for path in parts + [os.path.join(root, "metrics.jsonl")]:
        if not os.path.exists(path):
            continue
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                # Merged per metric: a later attempt's value wins for a metric it logged again (the steps it re-ran),
                # and the earlier attempt's other metrics stay (a resumed run re-logs only the validation of the
                # checkpointed step).
                by.setdefault(r["step"], {}).update(r["data"])
    return by


def series(by, key):
    return [(k, by[k][key]) for k in sorted(by) if key in by[k]]


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def load_e3(root):
    path = os.path.join(root, "e3_int8.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(line) for line in open(path) if line.strip()]
    step0 = [r for r in recs if r.get("start_step", 0) == 0]   # attempts that start at step 0 (not resumed)
    if not step0:
        return None
    first = step0[0]["attempt"]
    return [r for r in step0 if r["attempt"] == first]


def e3_window(recs):
    """Mean flip share, scale-change share and quantization error per projection (over layers) in FLIP_WINDOW."""
    win = [r for r in recs if r["sync"] in FLIP_WINDOW]
    out = {}
    for p in recs[0]["flip"]:
        fl = [x for r in win for x in r["flip"][p] if x is not None]
        sc = [x for r in win for x in r["scale_changed"][p] if x is not None]
        qe = [x for r in win for x in r["qerr"][p]]
        out[p] = (mean(fl), mean(sc), mean(qe))
    pred = {}
    for r in win:
        obs_all = r["pred"].get("observed_all_rows", {})
        for p in ("q_proj", "k_proj", "v_proj"):
            for k, (li, val) in enumerate(zip(r["pred"]["layers"], r["pred"][p])):
                if val is not None:
                    # the observed share on all rows (same rows as the prediction) when logged, else the row subset's
                    obs = obs_all[p][k] if obs_all.get(p) and obs_all[p][k] is not None else r["flip"][p][li]
                    pred.setdefault((li, p), []).append((val, obs))
    return len(win), out, pred


def brief(root, by, full):
    tag = os.path.basename(root.rstrip("/"))
    steps = sorted(k for k in by if "timing_s/step" in by[k])
    if not steps:
        return f"{tag:<22} no steps yet"
    last = steps[-1]
    d = by[last]
    vkey = next((k for s in sorted(by) for k in by[s] if k.startswith("val-core/")), None)
    val = series(by, vkey) if vkey else []
    k3 = d.get("rollout_corr/k3_kl", float("nan"))
    vs = f"val {100 * val[-1][1]:.1f}% @ {val[-1][0]}" if val else "val -"
    return (f"{tag:<22} step {last:>3}/{full} | {d['timing_s/step']:6.0f} s/step | score {d.get('critic/score/mean', float('nan')):+.3f}"
            f" | resp {d.get('response_length/mean', float('nan')):5.0f} tok | k3 {k3:.2e} | {vs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full-steps", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--hours", action="store_true", help="print only the projected hours of a full run")
    a = ap.parse_args()
    by = load_metrics(a.root)
    if a.brief:
        print(brief(a.root, by, a.full_steps))
        return
    if a.hours:
        t = [by[k]["timing_s/step"] for k in sorted(by) if "timing_s/step" in by[k]]
        val_t = [v for _, v in series(by, "timing_s/testing")]
        n_val = a.full_steps // a.val_every + 1 if a.val_every > 0 else 0
        med = statistics.median(t[1:] if len(t) > 1 else t)
        print(f"{(a.full_steps * med + n_val * (statistics.median(val_t) if val_t else 0)) / 3600:.2f}")
        return
    steps = sorted(k for k in by if "timing_s/step" in by[k])
    if not steps:
        print("no training steps logged")
        return
    stages = sorted({k[len("timing_s/"):] for s in steps for k in by[s] if k.startswith("timing_s/")} - {"step"})
    t = [by[k]["timing_s/step"] for k in steps]
    warm = t[1:] if len(t) > 1 else t
    med = statistics.median(warm)
    print(f"steps logged: {len(steps)} ({steps[0]}-{steps[-1]}); step time: first {t[0]:.0f} s, median of the rest "
          f"{med:.0f} s (min {min(warm):.0f}, max {max(warm):.0f})")
    for s in stages:
        xs = [by[k][f"timing_s/{s}"] for k in steps[1:] if f"timing_s/{s}" in by[k]]
        if xs and s not in ("testing", "save_checkpoint"):
            print(f"  {s:<22} median {statistics.median(xs):7.1f} s ({100 * statistics.median(xs) / med:4.0f}% of a step)")
    val_t = [v for _, v in series(by, "timing_s/testing")]
    save_t = [v for _, v in series(by, "timing_s/save_checkpoint")]
    n_val = a.full_steps // a.val_every + 1 if a.val_every > 0 else 0
    n_save = a.full_steps // a.save_every if a.save_every > 0 else 0
    total = a.full_steps * med + n_val * (statistics.median(val_t) if val_t else 0) + \
        n_save * (statistics.median(save_t) if save_t else 0)
    print(f"projection for {a.full_steps} steps: {a.full_steps * med / 3600:.1f} h of training + {n_val} validations"
          f" ({statistics.median(val_t) if val_t else float('nan'):.0f} s each) + {n_save} checkpoint saves"
          f" ({statistics.median(save_t) if save_t else float('nan'):.0f} s each) = {total / 3600:.1f} h")

    sc = series(by, "critic/score/mean")
    if sc:
        xs = [v for _, v in sc]
        early = [v for k, v in sc if k <= 20]
        late = [v for k, v in sc if k > a.full_steps - 20]
        print(f"\ntraining score (+1 correct / -1 wrong, plus overlong penalty): step {sc[0][0]} {xs[0]:+.3f}, "
              f"last {xs[-1]:+.3f}; mean over all steps {mean(xs):+.3f}, steps 1-20 {mean(early):+.3f}, "
              f"last 20 steps {mean(late):+.3f}")
    rl = series(by, "response_length/mean")
    if rl:
        cl = series(by, "response_length/clip_ratio")
        print(f"response length: step {rl[0][0]} {rl[0][1]:.0f}, last {rl[-1][1]:.0f} tokens; share cut at the cap: "
              f"last {cl[-1][1]:.3f}" if cl else "")
    for key in sorted({k for s in by for k in by[s] if k.startswith("val-core/")}):
        print(f"{key}: " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in series(by, key)))

    extra = [("rollout_corr/k3_kl", "mismatch KL (k3, per token)"), ("rollout_corr/kl", "mismatch KL (k1, per token)"),
             ("actor/qurl/tis_trunc_frac", "TIS: share of tokens with rho > C"),
             ("actor/qurl/acr_kept_frac", "ACR: share of tokens kept beyond 1 + eps"),
             ("actor/qurl/rho_max", "largest rho in a batch"), ("actor/grad_norm", "gradient norm (before clipping)")]
    shown = False
    for key, label in extra:
        xs = series(by, key)
        if xs:
            if not shown:
                print("\nper step (mean over steps; step 1; min-max):")
                shown = True
            v = [x for _, x in xs]
            print(f"  {label:<42} {mean(v):.3e}  ({v[0]:.3e}; {min(v):.3e}-{max(v):.3e})")

    refit = os.path.join(a.root, "int8_refit.txt")
    if os.path.exists(refit):
        lines = [ln for ln in open(refit) if "sync" in ln]
        counts = [int(m.group(1)) for m in (re.search(r": (\d+) linear weights", ln) for ln in lines) if m]
        pids = {m.group(1) for m in (re.search(r"pid (\d+)", ln) for ln in lines) if m}
        kern = sorted({m.group(1) for m in (re.search(r"INT8 kernel (\w+)", ln) for ln in lines) if m})
        dtypes = sorted({m.group(1) for m in (re.search(r"arrive as (\S+);", ln) for ln in lines) if m})
        if counts:
            print(f"\nINT8 refit: {len(lines)} sync lines from {len(pids)} vLLM worker process(es); weights per sync {min(counts)}-"
                  f"{max(counts)}; synced dtype {dtypes}; kernel {kern}")
    recs = load_e3(a.root)
    if recs:
        n, out, pred = e3_window(recs)
        print(f"\nE2/E3 window: syncs {FLIP_WINDOW.start}-{FLIP_WINDOW.stop - 1} of attempt {recs[0]['attempt']} "
              f"({n} syncs logged, {recs[0]['layers']} layers); mean over layers and syncs:")
        print(f"  {'projection':<10} {'code flips':>11} {'rows rescaled':>14} {'quant. error':>13}")
        for p, (fl, scl, qe) in out.items():
            print(f"  {p:<10} {100 * fl:10.4f}% {100 * scl:13.2f}% {qe:13.3e}")
        for (li, p), vals in sorted(pred.items()):
            pv, ov = mean([v for v, _ in vals]), mean([o for _, o in vals if o is not None])
            print(f"  layer {li:>2} {p}: Proposition 4 predicts {100 * pv:.4f}% flips, observed {100 * ov:.4f}%")


if __name__ == "__main__":
    main()
