"""S1 analysis: does UAQ (s = 1.5) reduce the step-0 rollout mismatch, with sequence-level uncertainty?

Usage: python analyze_s1.py <results_dir> [--boot 2000]

Reads the score_<quant>_<variant>.json files written by e0_logprobs.py (same layout as experiments/e0/results).
For each rollout quantizer q, the mismatch of mu = q(checkpoint) is measured against the BF16 policy that is
actually trained from that checkpoint:
  original checkpoint -> pi = none_orig;  UAQ s = 1.5 checkpoint -> pi = none_s1.5
(UAQ is applied to the trainable BF16 model too, and W/1.5 and 1.5*gamma are re-rounded in BF16).
Metrics, per response token:
  mad  mean |log mu - log pi|
  k1   KL(pi || mu) estimate  log pi - log mu
  k3   KL(pi || mu) estimate  (r - 1) - log r, r = mu / pi  (unbiased, non-negative, lower variance)
Tokens were sampled from none_orig, so k1/k3 against none_s1.5 are importance-weighted by pi_s1.5 / pi_orig
(per-token weights; exact in expectation because the full-vocabulary sum of pi_s1.5 is 1).
Uncertainty: 95% percentile CIs from a paired bootstrap over sequences (tokens within a response are not
independent). An s-fold error reduction, as QuRL's UAQ argument implies, would give mad ratio 1/s and KL
ratio 1/s^2.
Estimator check: k3 - k1 = r - 1, whose expectation is exactly 0 when tokens are drawn from the scored pi
(full-vocabulary sum of mu is 1). A sample mean of r - 1 whose CI excludes 0 means the sampling distribution
differs from the scored pi, and then neither sampled-token KL estimate is unbiased.
"""
import argparse
import glob
import json
import os

import numpy as np

S = 1.5


def load(path):
    d = json.load(open(path))
    return [np.asarray(x, dtype=np.float64) for x in d["logprobs"]]


def per_seq(values, weights=None):
    """Per-sequence sums of a per-token quantity (optionally weighted)."""
    if weights is None:
        return np.array([v.sum() for v in values])
    return np.array([(w * v).sum() for v, w in zip(values, weights)])


def metrics(mu, pi, w=None):
    """Per-sequence sums of mad, k1, k3 for rollout log-probs mu against policy log-probs pi."""
    d = [m - p for m, p in zip(mu, pi)]                       # log mu - log pi
    mad = per_seq([np.abs(x) for x in d])                    # unweighted distance on sampled tokens
    k1 = per_seq([-x for x in d], w)
    k3 = per_seq([np.expm1(x) - x for x in d], w)
    return {"mad": mad, "k1": k1, "k3": k3}


def boot_ratio(num, den, n_tok, idx):
    """Ratio of token-weighted means (num/den) on bootstrap resamples idx (B x n_seq)."""
    return (num[idx].sum(1) / n_tok[idx].sum(1)) / (den[idx].sum(1) / n_tok[idx].sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    runs = {os.path.basename(p)[6:-5]: load(p) for p in sorted(glob.glob(os.path.join(a.root, "score_*.json")))}
    pi_o, pi_s = runs["none_orig"], runs["none_s1.5"]
    n_seq = len(pi_o)
    n_tok = np.array([len(x) for x in pi_o], dtype=np.float64)
    for k, v in runs.items():
        assert len(v) == n_seq and all(len(x) == n for x, n in zip(v, n_tok)), f"misaligned run {k}"
    w_s = [np.exp(s - o) for s, o in zip(pi_s, pi_o)]          # importance weights pi_s1.5 / pi_orig
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_seq, size=(a.boot, n_seq))
    N = n_tok.sum()

    cat = np.concatenate
    nll = -cat(pi_o).mean()
    print(f"{os.path.basename(os.path.normpath(a.root))}: {n_seq} responses, {int(N)} response tokens, "
          f"mean NLL of sampled tokens under BF16 pi = {nll:.3f} nats")
    print(f"BF16 alone, none_s1.5 vs none_orig: mad {np.abs(cat(pi_s) - cat(pi_o)).mean():.3e}; "
          f"IS weight mean {cat(w_s).mean():.4f} (should be ~1)")
    print(f"reference for an s-fold error reduction: mad ratio {1 / S:.3f}, KL ratio {1 / S ** 2:.3f}\n")

    quants = sorted({k.rsplit("_", 1)[0] for k in runs if k.endswith("_orig") and not k.startswith("none")})
    hdr = "{:<16} {:<5} {:>11} {:>11} {:>8}  {:<17} {:>9}"
    print(hdr.format("rollout quant", "metric", "orig", "s=1.5", "ratio", "95% CI (seq boot)", "vs orig pi"))
    for q in quants:
        if f"{q}_s1.5" not in runs:
            continue
        m_o = metrics(runs[f"{q}_orig"], pi_o)
        if m_o["mad"].sum() / N < 1e-3:   # e.g. vLLM online modes that quantized 0 layers: really BF16
            print(f"{q:<16} skipped: mismatch to BF16 {m_o['mad'].sum() / N:.1e}, i.e. not actually quantized")
            continue
        m_s = metrics(runs[f"{q}_s1.5"], pi_s, w_s)               # matched pi (UAQ'd BF16 model)
        m_x = metrics(runs[f"{q}_s1.5"], pi_o)                    # as in summary.txt: against none_orig
        for key in ["mad", "k1", "k3"]:
            r = m_s[key].sum() / m_o[key].sum()
            lo, hi = np.quantile(boot_ratio(m_s[key], m_o[key], n_tok, idx), [0.025, 0.975])
            rx = m_x[key].sum() / m_o[key].sum()
            print(hdr.format(q if key == "mad" else "", key, f"{m_o[key].sum() / N:.4e}",
                             f"{m_s[key].sum() / N:.4e}", f"{r:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{rx:.3f}"))
        checks = []
        for name, m in [("orig", m_o), ("s1.5", m_s)]:
            cv = m["k3"] - m["k1"]                                   # per-sequence sums of (weighted) r - 1
            lo, hi = np.quantile(cv[idx].sum(1) / n_tok[idx].sum(1), [0.025, 0.975])
            checks.append(f"{name} {cv.sum() / N:+.1e} [{lo:+.1e}, {hi:+.1e}]")
        print(f"{'':<16} estimator check, mean(r-1) (0 if tokens ~ scored pi): " + "; ".join(checks))
        e_o = cat(runs[f"{q}_orig"]) - cat(pi_o)
        e_s = cat(runs[f"{q}_s1.5"]) - cat(pi_s)
        k3_o = np.sort(np.expm1(e_o) - e_o)[::-1]
        top = k3_o[: int(np.ceil(0.01 * len(k3_o)))].sum() / k3_o.sum()
        print(f"{'':<16} error corr(e_orig, e_s1.5) = {np.corrcoef(e_o, e_s)[0, 1]:.3f}; "
              f"|e| smaller after UAQ on {100 * np.mean(np.abs(e_s) < np.abs(e_o)):.1f}% of tokens, "
              f"larger on {100 * np.mean(np.abs(e_s) > np.abs(e_o)):.1f}%; "
              f"top 1% of tokens carry {100 * top:.0f}% of k3 (orig)")
    print("\nratio = s1.5 / orig with matched pi; 'vs orig pi' = same ratio with pi = none_orig for both "
          "(the summary.txt convention).")


if __name__ == "__main__":
    main()
