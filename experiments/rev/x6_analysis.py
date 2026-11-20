#!/usr/bin/env python3
"""X6, pre-registered analysis: are S1's FP8 changes at s = 1.5 more than BF16 re-rounding noise?

  python -I x6_analysis.py --results <copy>/results [--s1 DIR] [--out FILE]     (or --x6 <copy>/results/x6_fp8null)
  python -I x6_analysis.py --self-test
Defaults: --x6 <results>/x6_fp8null, where <results> is --results or results/ next to this file (a copy of
~/qurl/rev/results); --s1 ../h100/results next to this file (S1's results, one folder per model tag); report to
<x6>/x6_analysis.txt and stdout. Standard library only.

Background
  S1 scored 512 GSM8K responses per model under each rollout quantizer, for the original checkpoint and for UAQ
  checkpoints, each against the BF16 policy of its own checkpoint. At s = 1.5 the FP8 ratios (UAQ / original) were
  0.94-0.98 at 1.5B and 7B, with sequence-bootstrap CIs that exclude 1. Theorem 1 says every scalar s gives the same
  quantized model in exact arithmetic; the checkpoints are stored in BF16, so W/s and s*gamma are re-rounded. The
  bootstrap covers the variation over responses, not over re-rounding draws. X6 (x6_run.sh) builds UAQ checkpoints at
  ten nearby s with S1's make_uaq_checkpoints.py: different re-rounding draws of the same mathematical model. Each draw
  is scored on S1's responses in BF16 (its own policy), FP8 per-tensor and FP8 per-channel (vLLM 0.33 online
  quantization, as in S1) and, optionally, INT8 W8A8 (llm-compressor checkpoints). X6 also re-scores the original and
  s = 1.5 checkpoints, to check that S1's environment is reproduced.

Ratios, computed exactly as analyze_s1.py does
  R(s) = M(mu_s, pi_s, w_s) / M(mu_orig, pi_orig)
  M sums over all response tokens either |log mu - log pi| (mean |dlog p|, primary metric, unweighted) or
  w * ((r - 1) - log r) with r = mu / pi (k3 KL, secondary). The tokens were sampled from pi_orig, so k3 is weighted
  by w_s = pi_s / pi_orig per token (w = 1 for the original).
  The denominator and pi_orig are S1's (score_<q>_orig.json, score_none_orig.json), common to every ratio.
  R_obs is S1's own s = 1.5 ratio, from S1's score files: the value in the paper's table.

Decision rule (per model x quantizer x metric; fixed 2026-11-18, before any X6 data existed)
  The null draws are s = 1.45, 1.46, 1.47, 1.48, 1.49, 1.51, 1.52, 1.53, 1.54, 1.55. A verdict needs n >= 8 usable
  draws. With m and SD (n - 1 denominator) the mean and standard deviation of the null ratios R_1..R_n:
    consistent with re-rounding noise   min R_j <= R_obs <= max R_j
    systematic                          R_obs outside [min, max] AND outside the 95% prediction interval
                                        m +- t(0.975, n-1) * SD * sqrt(1 + 1/n)
    borderline                          R_obs outside [min, max] but inside that prediction interval
  Reported with every verdict: the rank of R_obs among the n + 1 values (1 = lowest ratio), the empirical two-sided
  p-value p_emp = min(1, 2 * min(1 + #{R_j <= R_obs}, 1 + #{R_j >= R_obs}) / (n + 1)), and the prediction-interval
  p-value p_PI = 2 * (1 - F_t(|R_obs - m| / (SD * sqrt(1 + 1/n)); n - 1)).
  Primary cells: FP8 per-tensor and FP8 per-channel at 1.5B and 7B on mean |dlog p| (4 cells). k3 and INT8 W8A8 are
  secondary, read with the same rule. No correction for multiplicity; the summary counts the verdicts.
Why this rule
  - Under the null hypothesis (exact invariance; BF16 re-rounding is the only difference between checkpoints), the
    ratio at s = 1.5 and the ratios at nearby s are draws of one re-rounding process on the same tokens, so the n + 1
    values are exchangeable. The range part is then exact without distributional assumptions:
    P(R_obs outside the range) = 2 / (n + 1), which is 0.18 at n = 10.
  - That rate is too high to call anything systematic on the range alone, so "systematic" also needs the normal
    prediction interval, which has level 5%. Each ratio averages about 2e5 tokens and each draw re-rounds about 1e8
    weights independently, so the draw-to-draw distribution should be close to normal. Under the null hypothesis,
    P(systematic) <= 0.05 and P(consistent) = 1 - 2 / (n + 1).
  - s only seeds the re-rounding; nearby s keep everything else (size of the rescaling, exponent ranges) as at
    s = 1.5. One difference is arithmetic: 1.5 = 3/2 is a simple ratio, so BF16 stores W/1.5 exactly for the third
    of values whose 8-bit significand is divisible by 3 and misses the rest by exactly 1/3 ulp, while at s = 1.47 the
    errors spread over [-1/2, 1/2] ulp (s*gamma likewise). An s = 1.5-specific effect of that kind is what
    "systematic" would point at; the per-draw re-rounding sizes below help to read it.
Location of the null (secondary; a different question)
  The mean of the null ratios with its 95% t-interval, and the number of draws below 1 with an exact two-sided sign
  test. The verdict asks whether s = 1.5 is special among re-rounding draws; it cannot see an effect that every draw
  shares. If the t-interval excludes 1, BF16-stored UAQ near s = 1.5 shifts the mismatch systematically, by about the
  null mean, whether or not s = 1.5 is special.
Reading
  consistent, t-interval covers 1     the s = 1.5 change is re-rounding noise, as Theorem 1 predicts
  consistent, t-interval excludes 1   a systematic effect of BF16 re-rounding near s = 1.5, not specific to 1.5
  systematic                          s = 1.5 differs from nearby s
  borderline                          inconclusive
Same environment
  If X6's re-scores of the original or s = 1.5 checkpoint (BF16 or the cell's quantizer) differ from S1's by more
  than S1's shuffled-batch noise floor of the same configuration (mean |dlog p|; the floor is 0 where S1's shuffled
  re-run was bit-identical), S1's environment was not reproduced. The cell's verdict is then taken from the
  same-environment comparison instead: X6's own s = 1.5 ratio against the null ratios over X6's own original. So it
  is when S1's s = 1.5 ratio cannot be computed. The report always shows both, and marks which one the verdict uses.
Data checks
  - A score file is used only if it is readable, all finite, and aligned token for token with S1's responses (same
    number of sequences, same lengths).
  - The responses' SHA-256 recorded by x6_run.sh must match S1's responses.json when both exist; otherwise none of
    that model's X6 files are used.
  - A draw whose quantized run is within 1e-3 of its own BF16 policy (mean |dlog p|, analyze_s1.py's guard) quantized
    nothing and is dropped. So is a quantizer whose original run fails that guard.
  - Missing or unusable draws only reduce n. Values of s outside the registered list are listed but not used.
Descriptive, no verdicts
  - The response-sampling SD of S1's ratio (paired sequence bootstrap, as in analyze_s1.py's CI) next to the
    re-rounding SD of the null.
  - Per draw, the size of the BF16 re-rounding (mean |log pi_s - log pi_orig| and the mean IS weight) and the share
    of Q/K/V weight codes it changes (make_uaq_checkpoints.py's checkpoints.jsonl).
Frozen by SHA-256 before the data existed (registration.sha256); --models, --draws and --min-draws exist for smoke
tests only, and the report marks any non-registered setting.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REG_MODELS = (("math1.5b", "Qwen/Qwen2.5-Math-1.5B"), ("math7b", "Qwen/Qwen2.5-Math-7B"))
REG_DRAWS = (1.45, 1.46, 1.47, 1.48, 1.49, 1.51, 1.52, 1.53, 1.54, 1.55)
S_OBS = 1.5
N_MIN = 8
QUANTS = (("fp8-per-tensor", "FP8 per-tensor", True),        # True: part of the primary question
          ("fp8-per-channel", "FP8 per-channel/token", True),
          ("int8-w8a8", "INT8 W8A8", False))
METRICS = (("mad", "mean |dlog p|"), ("k3", "k3 KL"))
PRIMARY_METRIC = "mad"
NOT_QUANTIZED = 1e-3      # analyze_s1.py: mismatch to BF16 below this means the mode quantized nothing
BOOT = 2000
BOOT_SEED = 0


# ---------------------------------------------------------------------------------------------------- statistics
def _betacf(a, b, x):
    """Continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """CDF of Student's t with df degrees of freedom."""
    if t == 0.0:
        return 0.5
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q, df):
    """Quantile of Student's t (bisection on t_cdf; 0 < q < 1)."""
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sign_test_p(k, n):
    """Exact two-sided sign test: k of n draws on one side, p = 0.5 each."""
    if n == 0:
        return 1.0
    j = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(j + 1)) / 2.0 ** n)


def judge(obs, null, n_min):
    """Statistics of the null ratios and the registered verdict for R_obs (obs may be None)."""
    n = len(null)
    r = {"n": n, "verdict": None}
    if n:
        r["mean"] = math.fsum(null) / n
        r["min"], r["max"] = min(null), max(null)
    if n >= 2:
        m = r["mean"]
        r["sd"] = math.sqrt(math.fsum((x - m) ** 2 for x in null) / (n - 1))
        tq = t_ppf(0.975, n - 1)
        r["mean_ci"] = (m - tq * r["sd"] / math.sqrt(n), m + tq * r["sd"] / math.sqrt(n))
        r["pi"] = (m - tq * r["sd"] * math.sqrt(1 + 1 / n), m + tq * r["sd"] * math.sqrt(1 + 1 / n))
        below, eq = sum(x < 1.0 for x in null), sum(x == 1.0 for x in null)
        r["below1"], r["n_sign"] = below, n - eq
        r["sign_p"] = sign_test_p(below, n - eq)
    if obs is None:
        r["verdict"] = "no s = 1.5 ratio"
        return r
    if n >= 1:
        r["rank"] = 1 + sum(x < obs for x in null)
        r["ties"] = sum(x == obs for x in null)
        r["p_emp"] = min(1.0, 2.0 * min(1 + sum(x <= obs for x in null), 1 + sum(x >= obs for x in null)) / (n + 1))
    if n >= 2:
        sd, m = r["sd"], r["mean"]
        if sd > 0:
            r["p_pi"] = 2.0 * (1.0 - t_cdf(abs(obs - m) / (sd * math.sqrt(1 + 1 / n)), n - 1))
        else:
            r["p_pi"] = 1.0 if obs == m else 0.0
    if n < n_min:
        r["verdict"] = f"no verdict (n = {n} < {n_min})"
        return r
    in_range = r["min"] <= obs <= r["max"]
    in_pi = r["pi"][0] <= obs <= r["pi"][1]
    if in_range:
        r["verdict"] = "consistent with re-rounding noise"
    elif in_pi:
        r["verdict"] = "borderline"
    else:
        r["verdict"] = "systematic"
    return r


# ---------------------------------------------------------------------------------------------------------- data
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
            d = json.load(f)
        lp = [[float(v) for v in seq] for seq in d["logprobs"]]
    except (OSError, ValueError, TypeError, KeyError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if not all(math.isfinite(v) for seq in lp for v in seq):
        return None, "non-finite values"
    if lengths is not None and [len(s) for s in lp] != lengths:
        return None, "not aligned with S1's responses"
    return lp, None


def is_weights(pi_s, pi_o):
    """Per-token importance weights pi_s / pi_orig."""
    return [[math.exp(a - b) for a, b in zip(s, o)] for s, o in zip(pi_s, pi_o)]


def seq_sums(mu, pi, w=None):
    """Per-sequence sums of |log mu - log pi| (unweighted) and of w * (expm1(d) - d), d = log mu - log pi."""
    mad, k3 = [], []
    for i, (m, p) in enumerate(zip(mu, pi)):
        sa, s3 = [], []
        wi = w[i] if w is not None else None
        for t, (a, b) in enumerate(zip(m, p)):
            d = a - b
            sa.append(abs(d))
            s3.append((math.expm1(d) - d) * (wi[t] if wi is not None else 1.0))
        mad.append(math.fsum(sa))
        k3.append(math.fsum(s3))
    return {"mad": mad, "k3": k3}


def total(sums):
    return {k: math.fsum(v) for k, v in sums.items()}


def compare(a, b):
    """Token-level comparison of two aligned runs: mean |d|, share identical, max |d|."""
    d = [abs(x - y) for sa, sb in zip(a, b) for x, y in zip(sa, sb)]
    if not d:
        return 0.0, 1.0, 0.0
    return math.fsum(d) / len(d), sum(x == 0.0 for x in d) / len(d), max(d)


def boot_sd(num, den, seed=BOOT_SEED, n_boot=BOOT):
    """SD and 95% percentile interval of sum(num)/sum(den) over a paired bootstrap of sequences."""
    rng = random.Random(seed)
    n = len(num)
    vals = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        sd = math.fsum(den[i] for i in idx)
        if sd > 0:
            vals.append(math.fsum(num[i] for i in idx) / sd)
    if len(vals) < 2:
        return None
    vals.sort()
    m = math.fsum(vals) / len(vals)
    sdv = math.sqrt(math.fsum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return sdv, vals[int(0.025 * (len(vals) - 1))], vals[int(math.ceil(0.975 * (len(vals) - 1)))]


def read_checkpoints(path):
    """s -> code-change fractions from make_uaq_checkpoints.py's JSON lines (last line wins)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                out[round(float(d["s"]), 6)] = (d.get("int8_per_channel_codes_changed_frac"),
                                                d.get("fp8_per_tensor_codes_changed_frac"))
            except (ValueError, KeyError, TypeError):
                continue
    return out


def draws_found(x6d):
    """Values of s with a BF16 draw score in the X6 folder of a model."""
    out = set()
    if os.path.isdir(x6d):
        for f in os.listdir(x6d):
            if f.startswith("score_none_s") and f.endswith(".json"):
                try:
                    out.add(round(float(f[len("score_none_s"):-5]), 6))
                except ValueError:
                    pass
    return out


def s_name(s):
    return f"s{s:g}"


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def fp(x):
    return "-" if x is None else (f"{x:.3f}" if x >= 0.001 else f"{x:.1e}")


# ------------------------------------------------------------------------------------------------------ analysis
def analyze_model(tag, label, x6_root, s1_root, draws, n_min, n_boot, out):
    """Analyse one model; returns summary rows (dicts) for the final table."""
    rows = []
    s1d, x6d = os.path.join(s1_root, tag), os.path.join(x6_root, tag)
    out(f"\n== {tag} ({label})")
    pi_o, why = load_lp(os.path.join(s1d, "score_none_orig.json"))
    if pi_o is None:
        out(f"S1's score_none_orig.json is {why} ({s1d}): nothing to compare with")
        return rows
    lengths = [len(s) for s in pi_o]
    n_tok = sum(lengths)
    out(f"S1: {len(lengths)} responses, {n_tok} response tokens")

    x6_ok = os.path.isdir(x6d)
    if not x6_ok:
        out(f"no X6 folder for this model ({x6d}): S1's ratios only")
    resp = os.path.join(s1d, "responses.json")
    sha_s1 = sha256_file(resp) if os.path.isfile(resp) else None
    sha_x6 = None
    p = os.path.join(x6d, "responses.sha256")
    if os.path.isfile(p):
        with open(p) as f:
            parts = f.read().split()
        sha_x6 = parts[0] if parts else None
    if sha_s1 and sha_x6:
        if sha_s1 == sha_x6:
            out("responses: X6 scored S1's responses (SHA-256 match)")
        else:
            out(f"responses: X6's SHA-256 {sha_x6[:12]} differs from S1's {sha_s1[:12]}: X6 data not used")
            x6_ok = False
    else:
        out("responses: SHA-256 not compared (missing on one side); token alignment is checked per file")

    warned = set()

    def get(d, name, use=True):
        if not use:
            return None
        lp, why = load_lp(os.path.join(d, f"score_{name}.json"), lengths)
        if lp is None and why != "missing" and (d, name) not in warned:
            warned.add((d, name))
            out(f"  skipped {os.path.basename(d)}/score_{name}.json: {why}")
        return lp

    # 1. References: S1's files (primary) and X6's re-scores of the same checkpoints (same environment as the draws).
    ref = {"S1": {"none_orig": pi_o, "none_s1.5": get(s1d, "none_s1.5")},
           "X6": {"none_orig": get(x6d, "none_orig", x6_ok), "none_s1.5": get(x6d, "none_s1.5", x6_ok)}}
    for r in ref.values():
        r["w_s1.5"] = (is_weights(r["none_s1.5"], r["none_orig"])
                       if r["none_s1.5"] is not None and r["none_orig"] is not None else None)
    sh = get(s1d, "none_orig_shuffled")
    floor = {"none": compare(pi_o, sh)[0] if sh is not None else 0.0}
    del sh
    qinfo = {}
    for q, qlabel, primary in QUANTS:
        a = get(s1d, f"{q}_orig")
        if a is None:
            continue
        mo = seq_sums(a, pi_o)
        if total(mo)["mad"] / n_tok < NOT_QUANTIZED:
            out(f"{q}: S1's original run is within {NOT_QUANTIZED} of BF16 (quantized nothing): skipped")
            continue
        b = get(s1d, f"{q}_s1.5")
        sh = get(s1d, f"{q}_orig_shuffled")
        floor[q] = compare(a, sh)[0] if sh is not None else 0.0
        xa, xb = get(x6d, f"{q}_orig", x6_ok), get(x6d, f"{q}_s1.5", x6_ok)
        ref["S1"][f"{q}_orig"], ref["S1"][f"{q}_s1.5"] = a, b
        ref["X6"][f"{q}_orig"], ref["X6"][f"{q}_s1.5"] = xa, xb
        info = {"label": qlabel, "primary": primary, "den": {"S1": mo}, "num": {}, "obs": {},
                "null": {"S1": {}, "X6": {}}}
        if b is not None and ref["S1"]["w_s1.5"] is not None:
            info["num"]["S1"] = seq_sums(b, ref["S1"]["none_s1.5"], ref["S1"]["w_s1.5"])
        if xa is not None and ref["X6"]["none_orig"] is not None:
            xmo = seq_sums(xa, ref["X6"]["none_orig"])
            if total(xmo)["mad"] / n_tok >= NOT_QUANTIZED:
                info["den"]["X6"] = xmo
                if xb is not None and ref["X6"]["w_s1.5"] is not None:
                    info["num"]["X6"] = seq_sums(xb, ref["X6"]["none_s1.5"], ref["X6"]["w_s1.5"])
            else:
                out(f"{q}: X6's re-score of the original is within {NOT_QUANTIZED} of BF16: no same-environment ratios")
        for env in ("S1", "X6"):
            if env in info["num"] and env in info["den"]:
                tn, td = total(info["num"][env]), total(info["den"][env])
                info["obs"][env] = {k: tn[k] / td[k] for k in tn}
        qinfo[q] = info

    # 2. Was S1's environment reproduced? X6's re-scores against S1's files, per configuration.
    repro = {}
    if x6_ok:
        out("reproduction of S1 by X6's re-scores (per token: mean |d|, share identical, max |d|; against S1's "
            "shuffled-batch floor):")
    for cfg in ["none"] + list(qinfo):
        res = None
        for v in ("orig", "s1.5"):
            a, b = ref["S1"].get(f"{cfg}_{v}"), ref["X6"].get(f"{cfg}_{v}")
            if a is None or b is None:
                if x6_ok:
                    out(f"  {cfg}_{v:<5} not checked (X6's re-score or S1's file is missing)")
                continue
            m, ident, mx = compare(a, b)
            ok = m <= floor[cfg]
            res = ok if res is None else (res and ok)
            out(f"  {cfg}_{v:<5} {m:.2e}, {100 * ident:.2f}%, {mx:.2e}; floor {floor[cfg]:.2e} -> "
                f"{'reproduced' if ok else 'NOT reproduced'}")
        repro[cfg] = res
    for env in ref.values():             # the raw quantized runs are no longer needed
        for key in [k for k in env if k not in ("none_orig", "none_s1.5", "w_s1.5")]:
            del env[key]

    found = draws_found(x6d) if x6_ok else set()
    extra = sorted(s for s in found if s not in {round(x, 6) for x in draws} and s not in (1.5, 2.0))
    if extra:
        out(f"draws found but not registered (not used): {', '.join(f'{s:g}' for s in extra)}")
    codes = read_checkpoints(os.path.join(x6d, "checkpoints.jsonl")) if x6_ok else {}
    codes_s1 = read_checkpoints(os.path.join(s1d, "checkpoints.jsonl"))

    # 3. Null draws, one at a time (each draw's BF16 policy is its own reference, as in S1).
    bf16_rows = []
    for s in draws:
        name = s_name(s)
        pi_j = get(x6d, f"none_{name}", x6_ok)
        if pi_j is None:
            continue
        w = {"S1": is_weights(pi_j, pi_o)}
        if ref["X6"]["none_orig"] is not None:
            w["X6"] = is_weights(pi_j, ref["X6"]["none_orig"])
        flat_w = [x for seq in w["S1"] for x in seq]
        bf16_rows.append((s, compare(pi_j, pi_o)[0], math.fsum(flat_w) / len(flat_w)))
        for q, info in qinfo.items():
            mu = get(x6d, f"{q}_{name}")
            if mu is None:
                continue
            for env in ("S1", "X6"):
                if env not in w or env not in info["den"]:
                    continue
                sums = seq_sums(mu, pi_j, w[env])
                if env == "S1" and math.fsum(sums["mad"]) / n_tok < NOT_QUANTIZED:
                    out(f"  dropped {q}_{name}: within {NOT_QUANTIZED} of its BF16 policy (quantized nothing)")
                    break
                tn, td = total(sums), total(info["den"][env])
                info["null"][env][s] = {k: tn[k] / td[k] for k in tn}
        del pi_j, w

    # 4. Descriptive: size of each draw's BF16 re-rounding and its code changes.
    out("BF16 re-rounding per draw: mean |log pi_s - log pi_orig|, mean IS weight; Q/K/V codes changed "
        "(INT8 per-channel, FP8 per-tensor):")
    if ref["S1"]["none_s1.5"] is not None:
        flat = [x for seq in ref["S1"]["w_s1.5"] for x in seq]
        c = codes_s1.get(1.5, (None, None))
        out(f"  s=1.5 (S1) {compare(ref['S1']['none_s1.5'], pi_o)[0]:.3e}, {math.fsum(flat) / len(flat):.4f}; "
            f"codes {fp(c[0])}, {fp(c[1])}")
    for s, m, wm in bf16_rows:
        c = codes.get(round(s, 6), (None, None))
        out(f"  s={s:<4g}      {m:.3e}, {wm:.4f}; codes {fp(c[0])}, {fp(c[1])}")
    if not bf16_rows:
        out("  (no usable null draws)")

    # 5. Per quantizer and metric: the null draws, their statistics, the verdicts.
    def null_lines(r, indent):
        if r["n"] == 0:
            return [f"{indent}no usable null draws"]
        lines = [f"{indent}null: n {r['n']}, mean {f4(r['mean'])}, SD {f4(r.get('sd'))}, min {f4(r['min'])}, "
                 f"max {f4(r['max'])}"]
        if r["n"] >= 2:
            lo, hi = r["mean_ci"]
            lines.append(f"{indent}location: mean 95% CI [{lo:.4f}, {hi:.4f}] "
                         f"({'excludes' if lo > 1 or hi < 1 else 'covers'} 1); below 1: {r['below1']} of "
                         f"{r['n_sign']} (sign test p {r['sign_p']:.3f})")
        return lines

    for q, info in qinfo.items():
        out(f"\n-- {tag} {info['label']} ({'primary question' if info['primary'] else 'secondary'})")
        rep = None if repro.get(q) is None or repro.get("none") is None else (repro[q] and repro["none"])
        null_s1 = info["null"]["S1"]
        out("  draw    " + "  ".join(f"{lab:>14}" for _, lab in METRICS))
        for s in sorted(null_s1):
            out(f"  s={s:<5g}" + "  ".join(f"{f4(null_s1[s][k]):>14}" for k, _ in METRICS))
        if "S1" in info["obs"]:
            out("  S1 1.5 " + "  ".join(f"{f4(info['obs']['S1'][k]):>14}" for k, _ in METRICS) + "   (R_obs)")
        if "X6" in info["obs"]:
            out("  X6 1.5 " + "  ".join(f"{f4(info['obs']['X6'][k]):>14}" for k, _ in METRICS)
                + "   (X6's re-score of s = 1.5, over X6's original)")
        bs = None
        if "S1" in info["num"]:
            bs = {k: boot_sd(info["num"]["S1"][k], info["den"]["S1"][k], n_boot=n_boot) for k, _ in METRICS}
        for k, klabel in METRICS:
            res = {}
            for env in ("S1", "X6"):
                if env == "X6" and "X6" not in info["den"]:
                    continue
                null = [v[k] for _, v in sorted(info["null"][env].items())]
                res[env] = judge(info["obs"].get(env, {}).get(k), null, n_min)
            use = "S1"
            if (rep is False or "S1" not in info["obs"]) and "X6" in res and res["X6"]["n"] >= n_min \
                    and "X6" in info["obs"]:
                use = "X6"
            out(f"  {klabel}{' (primary)' if info['primary'] and k == PRIMARY_METRIC else ''}:")
            for env, rr in res.items():
                what = ("S1's s = 1.5 ratio, null over S1's original" if env == "S1" else
                        "X6's s = 1.5 ratio, null over X6's original (same environment)")
                if env == "S1" or rr["n"] != res["S1"]["n"] or \
                        any(abs(rr[x] - res["S1"][x]) > 1e-9 for x in ("mean", "sd") if x in rr and x in res["S1"]):
                    if env == "X6":
                        out("    same environment:")
                    for line in null_lines(rr, "    "):
                        out(line)
                obs = info["obs"].get(env, {}).get(k)
                if obs is None or rr["n"] == 0:
                    out(f"    {what}: {rr['verdict']}")
                    continue
                pi_txt = f"95% PI [{rr['pi'][0]:.4f}, {rr['pi'][1]:.4f}], p_PI {rr['p_pi']:.3f}; " if "pi" in rr else ""
                ties = f" ({rr['ties']} tied)" if rr.get("ties") else ""
                out(f"    {what}: R_obs {obs:.4f}, rank {rr['rank']}{ties} of {rr['n'] + 1}, p_emp {rr['p_emp']:.3f}; "
                    f"{pi_txt}-> {rr['verdict']}{'   <- verdict' if env == use else ''}")
            r = res["S1"]
            if bs and bs.get(k) and r.get("sd") is not None:
                sdb, lo, hi = bs[k]
                out(f"    response-sampling SD of S1's ratio (sequence bootstrap) {sdb:.4f}, 95% CI "
                    f"[{lo:.4f}, {hi:.4f}]; re-rounding SD of the null {r['sd']:.4f}")
            if rep is False:
                out("    S1's environment NOT reproduced: the verdict is the same-environment one" if use == "X6" else
                    "    S1's environment NOT reproduced, and no same-environment verdict is possible")
            elif use == "X6":
                out("    S1's s = 1.5 ratio is unavailable: the verdict is the same-environment one")
            elif rep is None:
                out("    reproduction of S1 not checked (X6's re-scores of the original or s = 1.5 are missing)")
            final = res[use]
            loc = None
            if final.get("mean_ci"):
                loc = "excludes 1" if final["mean_ci"][0] > 1 or final["mean_ci"][1] < 1 else "covers 1"
            rows.append({"tag": tag, "q": q, "metric": k, "primary": info["primary"] and k == PRIMARY_METRIC,
                         "obs": info["obs"].get(use, {}).get(k), "use": use, "loc": loc, **final})
    return rows


def reading(row):
    v = row.get("verdict") or ""
    if v.startswith("consistent"):
        if row.get("loc") == "excludes 1":
            return "systematic effect of re-rounding near s = 1.5, not specific to 1.5"
        return "re-rounding noise" if row.get("loc") == "covers 1" else "-"
    if v == "systematic":
        return "s = 1.5 differs from nearby s"
    if v == "borderline":
        return "inconclusive"
    return "-"


def run(x6_root, s1_root, models, draws, n_min, n_boot, out_path):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("X6: a null distribution for S1's s = 1.5 ratios (BF16 re-rounding draws at nearby s)")
    try:
        me = sha256_file(os.path.abspath(__file__))
    except OSError:
        me = "?"
    out(f"x6_analysis.py SHA-256 {me}")
    out(f"X6 results: {x6_root}")
    out(f"S1 results: {s1_root}")
    nonreg = []
    if tuple(models) != REG_MODELS:
        nonreg.append(f"models {[m[0] for m in models]}")
    if tuple(round(s, 6) for s in draws) != REG_DRAWS:
        nonreg.append(f"draws {[f'{s:g}' for s in draws]}")
    if n_min != N_MIN:
        nonreg.append(f"min draws {n_min}")
    if nonreg:
        out("NON-REGISTERED SETTINGS (smoke test only): " + "; ".join(nonreg))
    out(f"draws s = {', '.join(f'{s:g}' for s in draws)}; a verdict needs n >= {n_min}; ratios = UAQ / original, "
        "each against its own BF16 policy (analyze_s1.py)")
    man = os.path.join(x6_root, "manifest.txt")
    if os.path.isfile(man):
        with open(man) as f:
            env = [l.rstrip() for l in f if l.startswith(("vllm", "torch", "gpu", "llmcompressor"))]
        for l in env[:6]:
            out(f"  manifest: {l}")
    rows = []
    for tag, label in models:
        rows += analyze_model(tag, label, x6_root, s1_root, draws, n_min, n_boot, out)

    out("\n== Summary (verdict per the registered rule; '*' = primary cell)")
    hdr = "{:<1} {:<10} {:<16} {:<6} {:>7} {:>7} {:>7} {:>7} {:>3} {:>5} {:>6} {:>6}  {:<34} {}"
    out(hdr.format("", "model", "quantizer", "metric", "R_obs", "mean", "min", "max", "n", "rank", "p_emp", "p_PI",
                   "verdict", "reading"))
    for r in rows:
        out(hdr.format("*" if r["primary"] else "", r["tag"], r["q"], r["metric"], f4(r.get("obs")),
                       f4(r.get("mean")), f4(r.get("min")), f4(r.get("max")), r["n"],
                       r.get("rank", "-"), f"{r['p_emp']:.3f}" if "p_emp" in r else "-",
                       f"{r['p_pi']:.3f}" if "p_pi" in r else "-",
                       (r["verdict"] or "-") + (" [X6 env]" if r["use"] == "X6" else ""), reading(r)))
    prim = [r for r in rows if r["primary"]]
    counts = {}
    for r in prim:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    out(f"primary cells ({len(prim)} of 4 expected): " +
        (", ".join(f"{v}: {c}" for v, c in sorted(counts.items())) if counts else "none"))
    if out_path:
        d = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(d, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nreport written to {out_path}")
    return rows


# ----------------------------------------------------------------------------------------------------- self-test
def _write_scores(path, lp):
    with open(path, "w") as f:
        json.dump({"model": "synthetic", "quant": "synthetic", "logprobs": lp}, f)


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)

    # 1. Student t against tables, sign test, empirical p, verdict logic.
    for df, q975 in ((1, 12.706205), (4, 2.776445), (7, 2.364624), (9, 2.262157), (10, 2.228139), (30, 2.042272)):
        check(abs(t_ppf(0.975, df) - q975) < 2e-6, f"t quantile df={df}: {t_ppf(0.975, df)} vs {q975}")
        check(abs(t_cdf(q975, df) - 0.975) < 1e-7, f"t cdf df={df}")
    check(abs(t_cdf(-1.0, 5) - (1 - t_cdf(1.0, 5))) < 1e-12, "t symmetry")
    check(abs(sign_test_p(0, 10) - 2 / 1024) < 1e-15, "sign test 0 of 10")
    check(abs(sign_test_p(5, 10) - 1.0) < 1e-15, "sign test 5 of 10")
    check(abs(sign_test_p(2, 10) - 2 * (1 + 10 + 45) / 1024) < 1e-15, "sign test 2 of 10")
    null = [1 + 0.01 * z for z in (-1.5, -1.0, -0.6, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.4)]
    r = judge(0.995, null, 8)
    check(r["verdict"].startswith("consistent") and r["rank"] == 4 and abs(r["p_emp"] - 8 / 11) < 1e-12,
          f"inside the range: {r}")      # 3 null values below 0.995: rank 4, p = 2 * (1 + 3) / 11
    check(r["below1"] == 4 and r["n_sign"] == 9, "sign test leaves out a ratio of exactly 1")
    r = judge(0.95, null, 8)
    check(r["verdict"] == "systematic" and r["rank"] == 1 and abs(r["p_emp"] - 2 / 11) < 1e-12 and r["p_pi"] < 0.05,
          f"far below: {r}")
    m = math.fsum(null) / 10
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in null) / 9)
    lo_pi = m - t_ppf(0.975, 9) * sd * math.sqrt(1.1)
    r = judge(0.5 * (lo_pi + min(null)), null, 8)
    check(r["verdict"] == "borderline" and r["p_pi"] > 0.05, f"between the PI and the range: {r}")
    r = judge(1.03, null[:7], 8)
    check(r["verdict"].startswith("no verdict"), "n < 8")
    r = judge(None, null, 8)
    check(r["verdict"] == "no s = 1.5 ratio" and r["n"] == 10, "no R_obs")
    r = judge(0.9, [0.95] * 10, 8)
    check(r["verdict"] == "systematic" and r["p_pi"] == 0.0, "degenerate null")
    r = judge(0.97, [0.96 + 0.001 * i for i in range(10)], 8)
    check(r["mean_ci"][1] < 1 and r["below1"] == 10 and r["sign_p"] < 0.01, "location below 1")
    # Exchangeability: under the null, P(outside the range) = 2 / (n + 1); P(systematic) <= 0.05 for normal draws.
    rng = random.Random(1)
    reps, n_out, n_sys = 4000, 0, 0
    for _ in range(reps):
        x = [rng.gauss(1.0, 0.02) for _ in range(11)]
        v = judge(x[0], x[1:], 8)["verdict"]
        n_out += not v.startswith("consistent")
        n_sys += v == "systematic"
    check(abs(n_out / reps - 2 / 11) < 0.03, f"range rule size {n_out / reps:.3f} vs {2 / 11:.3f}")
    check(n_sys / reps < 0.06, f"systematic rate under the null {n_sys / reps:.3f}")

    # 2. The ratio arithmetic equals analyze_s1.py's (if it and numpy can be imported here).
    rng = random.Random(7)
    lens = [3, 5, 4, 6, 2, 7]
    pi_o = [[-rng.uniform(0.01, 3.0) for _ in range(n)] for n in lens]
    pi_s = [[v + rng.uniform(-0.01, 0.01) for v in s] for s in pi_o]
    mu_o = [[v + rng.uniform(-0.2, 0.2) for v in s] for s in pi_o]
    mu_s = [[v + rng.uniform(-0.2, 0.2) for v in s] for s in pi_s]
    w = is_weights(pi_s, pi_o)
    mine = {k: math.fsum(seq_sums(mu_s, pi_s, w)[k]) / math.fsum(seq_sums(mu_o, pi_o)[k]) for k in ("mad", "k3")}
    flat = lambda a: [x for s in a for x in s]
    d_o = [a - b for a, b in zip(flat(mu_o), flat(pi_o))]
    d_s = [a - b for a, b in zip(flat(mu_s), flat(pi_s))]
    ww = flat(w)
    direct = {"mad": math.fsum(abs(x) for x in d_s) / math.fsum(abs(x) for x in d_o),
              "k3": math.fsum(c * (math.expm1(x) - x) for c, x in zip(ww, d_s)) /
                    math.fsum(math.expm1(x) - x for x in d_o)}
    for k in mine:
        check(abs(mine[k] - direct[k]) < 1e-12, f"ratio {k}: {mine[k]} vs direct {direct[k]}")
    a1 = os.path.join(HERE, "..", "h100", "analyze_s1.py")
    try:
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location("analyze_s1_ref", a1)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        npa = lambda a: [np.asarray(s, dtype=np.float64) for s in a]
        w_np = [np.exp(s - o) for s, o in zip(npa(pi_s), npa(pi_o))]
        m_s = ref.metrics(npa(mu_s), npa(pi_s), w_np)
        m_o = ref.metrics(npa(mu_o), npa(pi_o))
        for k in ("mad", "k3"):
            theirs = m_s[k].sum() / m_o[k].sum()
            check(abs(theirs - mine[k]) < 1e-12, f"analyze_s1.py {k}: {theirs} vs {mine[k]}")
        print("ratio arithmetic identical to analyze_s1.py's metrics() on synthetic data")
    except (ImportError, OSError, AttributeError) as e:
        print(f"note: analyze_s1.py comparison skipped ({type(e).__name__}: needs numpy and ../h100/analyze_s1.py)")

    # 3. End to end on synthetic result folders with known ratios.
    tmp = tempfile.mkdtemp(prefix="x6_selftest_")
    try:
        rng = random.Random(3)
        tag = "synth"
        lens = [rng.randint(5, 40) for _ in range(30)]
        n_tok = sum(lens)
        e = [[rng.uniform(0.05, 0.15) for _ in range(n)] for n in lens]       # per-token error size
        pi_o = [[-rng.uniform(0.01, 3.0) for _ in range(n)] for n in lens]

        def model_runs(r_mult, delta_scale, seed):
            """BF16 policy pi_s = pi_o + small perturbation; quantized mu = pi_s + r * e (mad ratio exactly r)."""
            g = random.Random(seed)
            pi_s = [[v + delta_scale * g.uniform(-1, 1) for v in s] for s in pi_o]
            mu = [[p + r_mult * x for p, x in zip(ps, es)] for ps, es in zip(pi_s, e)]
            return pi_s, mu

        def k3_ratio(pi_s, r_mult):
            num = math.fsum(math.exp(a - b) * (math.expm1(r_mult * x) - r_mult * x)
                            for sa, sb, se in zip(pi_s, pi_o, e) for a, b, x in zip(sa, sb, se))
            den = math.fsum(math.expm1(x) - x for se in e for x in se)
            return num / den

        null_r = [1 + 0.01 * z for z in (-1.5, -1.0, -0.6, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.4)]

        def build(root, r_obs, x6_obs=None, sha_ok=True, x6_refs=True, perturb_x6_orig=False):
            s1d, x6d = os.path.join(root, "s1", tag), os.path.join(root, "x6", tag)
            os.makedirs(s1d)
            os.makedirs(x6d)
            with open(os.path.join(s1d, "responses.json"), "w") as f:
                json.dump([{"prompt_ids": [1], "response_ids": [0] * n} for n in lens], f)
            sha = sha256_file(os.path.join(s1d, "responses.json"))
            with open(os.path.join(x6d, "responses.sha256"), "w") as f:
                f.write((sha if sha_ok else "0" * 64) + "  responses.json\n")
            mu_o = [[p + x for p, x in zip(ps, es)] for ps, es in zip(pi_o, e)]
            pi_s, mu_s = model_runs(r_obs, 0.004, 99)
            for q in ("fp8-per-tensor", "fp8-per-channel"):
                _write_scores(os.path.join(s1d, f"score_{q}_orig.json"), mu_o)
                _write_scores(os.path.join(s1d, f"score_{q}_orig_shuffled.json"), mu_o)
                _write_scores(os.path.join(s1d, f"score_{q}_s1.5.json"), mu_s)
            _write_scores(os.path.join(s1d, "score_none_orig.json"), pi_o)
            _write_scores(os.path.join(s1d, "score_none_s1.5.json"), pi_s)
            if x6_refs:
                xo = [[v + (1e-3 if perturb_x6_orig else 0.0) for v in s] for s in mu_o]
                for q in ("fp8-per-tensor", "fp8-per-channel"):
                    _write_scores(os.path.join(x6d, f"score_{q}_orig.json"), xo)
                    xs = mu_s if x6_obs is None else model_runs(x6_obs, 0.004, 99)[1]
                    _write_scores(os.path.join(x6d, f"score_{q}_s1.5.json"), xs)
                _write_scores(os.path.join(x6d, "score_none_orig.json"), pi_o)
                _write_scores(os.path.join(x6d, "score_none_s1.5.json"), pi_s)
            want = {}
            with open(os.path.join(x6d, "checkpoints.jsonl"), "w") as f:
                for j, (s, r_j) in enumerate(zip(REG_DRAWS, null_r)):
                    pj, mj = model_runs(r_j, 0.004, j)
                    _write_scores(os.path.join(x6d, f"score_none_{s_name(s)}.json"), pj)
                    for q in ("fp8-per-tensor", "fp8-per-channel"):
                        _write_scores(os.path.join(x6d, f"score_{q}_{s_name(s)}.json"), mj)
                    want[s] = (r_j, k3_ratio(pj, r_j))
                    f.write(json.dumps({"s": s, "int8_per_channel_codes_changed_frac": 0.04,
                                        "fp8_per_tensor_codes_changed_frac": 0.02}) + "\n")
            return s1d, x6d, want, k3_ratio(pi_s, r_obs)

        def go(root, **kw):
            return run(os.path.join(root, "x6"), os.path.join(root, "s1"), ((tag, "synthetic"),), REG_DRAWS, N_MIN,
                       200, None)

        def cell(rows, q="fp8-per-tensor", metric="mad"):
            return [r for r in rows if r["q"] == q and r["metric"] == metric][0]

        quiet = open(os.devnull, "w")
        old = sys.stdout
        try:
            # (a) R_obs inside the null: consistent; the ratios equal their closed forms.
            root = os.path.join(tmp, "a")
            s1d, x6d, want, k3_obs = build(root, 0.995)
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            c = cell(rows)
            check(c["verdict"].startswith("consistent") and abs(c["obs"] - 0.995) < 1e-12 and c["n"] == 10,
                  f"(a) consistent: {c}")
            check(abs(c["mean"] - math.fsum(null_r) / 10) < 1e-12, "(a) null mean = known ratios")
            ck = cell(rows, metric="k3")
            check(abs(ck["obs"] - k3_obs) < 1e-12, f"(a) k3 R_obs {ck['obs']} vs {k3_obs}")
            check(abs(ck["mean"] - math.fsum(v[1] for v in want.values()) / 10) < 1e-12, "(a) k3 null mean")
            check(c["use"] == "S1", "(a) reproduced: the S1-based verdict is used")
            # (b) far below the null: systematic.
            root = os.path.join(tmp, "b")
            build(root, 0.95)
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            check(cell(rows)["verdict"] == "systematic" and cell(rows)["rank"] == 1, "(b) systematic")
            # (c) missing, corrupt, misaligned and unquantized draws reduce n; fewer than 8 give no verdict.
            os.remove(os.path.join(x6d, "score_fp8-per-tensor_s1.45.json"))
            with open(os.path.join(x6d, "score_fp8-per-tensor_s1.46.json"), "w") as f:
                f.write('{"logprobs": [[0.1, ')
            lp, _ = load_lp(os.path.join(x6d, "score_fp8-per-tensor_s1.47.json"))
            lp[0] = lp[0][:-1]
            _write_scores(os.path.join(x6d, "score_fp8-per-tensor_s1.47.json"), lp)
            pj, _ = load_lp(os.path.join(x6d, "score_none_s1.48.json"))
            _write_scores(os.path.join(x6d, "score_fp8-per-tensor_s1.48.json"), pj)       # quantized nothing
            sys.stdout = quiet
            rows = go(os.path.join(tmp, "a"))
            sys.stdout = old
            c = cell(rows)
            check(c["n"] == 6 and c["verdict"].startswith("no verdict"), f"(c) n = 6, no verdict: {c}")
            check(cell(rows, q="fp8-per-channel")["n"] == 10, "(c) the other quantizer keeps its 10 draws")
            # (d) other responses than S1's: no X6 data used.
            root = os.path.join(tmp, "d")
            build(root, 0.995, sha_ok=False)
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            check(cell(rows)["n"] == 0 and abs(cell(rows)["obs"] - 0.995) < 1e-12, "(d) sha mismatch: S1 only")
            # (e) S1 not reproduced by X6's re-scores: the same-environment verdict is used.
            root = os.path.join(tmp, "e")
            build(root, 0.95, x6_obs=0.995, perturb_x6_orig=True)
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            c = cell(rows)
            check(c["use"] == "X6" and c["verdict"].startswith("consistent"), f"(e) same-environment fallback: {c}")
            # (f) no X6 re-scores: S1-based verdict, reproduction not checked.
            root = os.path.join(tmp, "f")
            build(root, 0.95, x6_refs=False)
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            check(cell(rows)["use"] == "S1" and cell(rows)["verdict"] == "systematic", "(f) without X6 re-scores")
            # (h) S1's s = 1.5 file missing: the same-environment verdict is used.
            root = os.path.join(tmp, "h")
            s1d_h, _, _, _ = build(root, 0.95, x6_obs=0.995)
            os.remove(os.path.join(s1d_h, "score_fp8-per-tensor_s1.5.json"))
            sys.stdout = quiet
            rows = go(root)
            sys.stdout = old
            c = cell(rows)
            check(c["use"] == "X6" and c["verdict"].startswith("consistent") and abs(c["obs"] - 0.995) < 1e-12,
                  f"(h) S1's ratio unavailable: {c}")
            # (g) no X6 folder at all, and no S1 folder at all.
            sys.stdout = quiet
            rows = run(os.path.join(tmp, "nowhere"), os.path.join(tmp, "a", "s1"), ((tag, "synthetic"),), REG_DRAWS,
                       N_MIN, 50, None)
            rows2 = run(os.path.join(tmp, "a", "x6"), os.path.join(tmp, "nowhere"), ((tag, "synthetic"),), REG_DRAWS,
                        N_MIN, 50, os.path.join(tmp, "report.txt"))
            sys.stdout = old
            check(cell(rows)["n"] == 0 and cell(rows)["verdict"].startswith("no verdict"), "(g) no X6 folder")
            check(rows2 == [] and os.path.isfile(os.path.join(tmp, "report.txt")), "(g) no S1 folder")
        finally:
            sys.stdout = old
            quiet.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("self-test " + ("passed" if ok else "FAILED") +
          ": t quantiles, sign test, verdict rule (incl. its size under the null), ratio arithmetic, and eight "
          "scenarios on synthetic result folders with known answers")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"), help="a copy of ~/qurl/rev/results")
    ap.add_argument("--x6", help="X6's folder (default <results>/x6_fp8null)")
    ap.add_argument("--s1", default=os.path.join(HERE, "..", "h100", "results"))
    ap.add_argument("--out", help="report file (default <x6>/x6_analysis.txt; '-' for stdout only)")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--models", nargs="+", help="smoke tests only: model tags instead of the registered ones")
    ap.add_argument("--draws", nargs="+", type=float, help="smoke tests only: draws instead of the registered ones")
    ap.add_argument("--min-draws", type=int, default=N_MIN, help="smoke tests only")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.x6 = a.x6 or os.path.join(a.results, "x6_fp8null")
    models = REG_MODELS if not a.models else tuple((m, dict(REG_MODELS).get(m, m)) for m in a.models)
    draws = REG_DRAWS if not a.draws else tuple(a.draws)
    out_path = None if a.out == "-" else (a.out or os.path.join(a.x6, "x6_analysis.txt"))
    if out_path and not a.out and not os.path.isdir(a.x6):
        out_path = None
    run(os.path.abspath(a.x6), os.path.abspath(a.s1), models, draws, a.min_draws, a.boot, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
