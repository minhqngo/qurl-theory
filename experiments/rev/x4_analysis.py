"""X4 analysis: does the s = 1.5 rounding between UAQ (R4) and the matched settings (R5) act like ulp noise?

  python -I x4_analysis.py <results_dir> [--report PATH] [--json PATH]
  python -I x4_analysis.py --results <results_dir> --out <report>     # the same, in the revision README's style
  python -I x4_analysis.py --self-test

Input: <results_dir>/<tag>/ for each arm tag below, as written by x4_train.py:
  - meta.json and log.jsonl;
  - steps/step_NNNN.npz, where step 0 is the initial state (after the UAQ rescaling and any perturbation) and step t >= 1
    is the state after rollout batch t and its updates.
The script needs numpy only. It reports missing or partial arms and leaves them out; the verdict then says what could
not be tested.
Output:
  - the report, on stdout and in <results_dir>/x4_report.txt; its last line is "VERDICT: ...";
  - every per-step series, in <results_dir>/x4_metrics.json.

Arms (tag: x4_train.py settings). All use s = 1.5 and lr 1e-6, and see the same prompts unless stated.
  R4             UAQ (W_QKV/s, gamma*s), plain AdamW and clipping
  R4_repeat      identical to R4: must be bit-identical (determinism check)
  R5             no UAQ, Theorem 2(b)-(c) settings
  R5p            R5 with every W_QKV and gamma element moved by +-1 ulp (random sign) at step 0: ulp noise
  R4p            R4 with every W'_QKV and gamma' element moved by +-1 ulp at step 0: ulp noise on the UAQ side
  R5_seed1       R5 with sampling seed 1, same prompts: a seed change
  R5_seed1_full  R5 with seed 1 for sampling and data order: a full seed change (different prompts)

Metrics, for arms A, B at step t:
  tokens  share of identical sampled tokens. Counted over the rollout rows whose prompt is the same in both arms: the
          positions where both arms sampled the same valid token, over the positions valid in either arm.
          seqs is the share of those rows that are identical.
  P_g     relative L2 distance ||a - b|| / ||b|| over the fixed random subset of group g. The groups are W (Q/K/V
          weights), gamma (input-norm gains) and rest. Both arms are mapped to original coordinates in float64:
          W = s W' and gamma = gamma'/s for UAQ arms. The map is exact for W and within 1e-16 for gamma.
          P_all estimates the whole model by weighting each group's sums by N_g/K_g (group size over subset size).
  Q       mean over the probe tokens of |log pi_A - log pi_B| on the fixed probe set:
            Q_pi32  pi in FP32;
            Q_pi16  pi in BF16 autocast, the training precision;
            Q_mu    the INT8 W8A8 rollout policy mu.

DECISION RULE (fixed on 2026-11-18, before any X4 data existed; this file is frozen by SHA-256)
  Pairs. D = (R4, R5) is the question. The noise pairs are (R5, R5p) and (R4, R4p), those present.
    S = (R5, R5_seed1) is the seed reference.
  Rule metrics: P_W, P_gamma, P_rest and Q_pi32.
  Steps: every recorded step t from 5 to T. T is the last step that R4, R5 and at least one noise pair reached.
  Band. At step t, for metric M, let lo be the smallest and hi the largest M over the noise pairs recorded at t.
    D is ABOVE at t if M_D > 2 hi, and BELOW at t if M_D < lo / 2.
  Verdict:
    ULP-NOISE-LIKE          no step ABOVE or BELOW, on any rule metric
    LARGER-THAN-ULP-NOISE   some step ABOVE, none BELOW
    SMALLER-THAN-ULP-NOISE  some step BELOW, none ABOVE
    MIXED                   both
    INVALID                 any of:
                              - R4_repeat is present but not bit-identical to R4 at some common step;
                              - R4 or R5 is misconfigured for its tag;
                              - R4 and R5 differ in s, model, data, probe set, parameter subsets or loop settings;
                              - a rule metric is not finite at an evaluated step.
                            Other arms that are misconfigured, or differ from R5 in those, are left out and reported.
    INCOMPLETE              R4 or R5 missing, both noise pairs missing, or T < 5
  Secondary, against the seed change S if present, on the same metrics and steps:
    SEED-BOUNDED if M_D <= 2 M_S at every step, else EXCEEDS-SEED;
    SATURATED if M_D >= M_S / 2 on every rule metric at T (or at the last step before T that S reached),
      else NOT-SATURATED.
  The verdict line also says whether determinism was verified (R4_repeat present and bit-identical) or unchecked.
  Why steps 1-4 are left out: the first token flips come at random times, and each curve is a single realization.
  Why Q_pi32 and not Q_pi16 or Q_mu:
    - Q_pi32 measures the policy of the FP32 master weights, which is where the arms' trajectories differ.
    - BF16 autocast and INT8 W8A8 round activations in every layer, so any ulp-level change of the master weights
      already lifts Q_pi16 and Q_mu to their rounding noise floors. In the laptop smoke test (0.5B, step 0) the
      floors were about 1e-2 (Q_pi16) and 4e-2 (Q_mu) per token, for R4 vs R5 and R5 vs R5p alike, while Q_pi32
      was 1.3e-6 for both.
    - Under BF16 autocast R4 also rounds W' = W/s and gamma' = s*gamma to BF16 on a grid that is not a rescaling
      of R5's.
    - Q_pi16 and Q_mu are reported but not used in the rule.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np

FORMAT = 1
GROUPS = ("W", "gamma", "rest")
T_MIN = 5
FACTOR = 2.0
RULE_METRICS = ("P_W", "P_gamma", "P_rest", "Q_pi32")
METRICS = ("tokens", "seqs", "P_W", "P_gamma", "P_rest", "P_all", "Q_pi32", "Q_pi16", "Q_mu", "reward")
ARMS = ("R4", "R5", "R5p", "R4_repeat", "R5_seed1", "R4p", "R5_seed1_full")
# tag: (x4_train --arm, perturbed groups, sampling seed differs from R5's, data seed differs from R5's)
EXPECTED = {"R4": ("R4", (), False, False), "R4_repeat": ("R4", (), False, False),
            "R5": ("R5", (), False, False), "R5p": ("R5", ("W", "gamma"), False, False),
            "R4p": ("R4", ("W", "gamma"), False, False), "R5_seed1": ("R5", (), True, False),
            "R5_seed1_full": ("R5", (), True, True)}
QUESTION = ("R4", "R5")
NOISE = (("R5", "R5p"), ("R4", "R4p"))
SEED = ("R5", "R5_seed1")
FULL = ("R5", "R5_seed1_full")
REPEAT = ("R4", "R4_repeat")
CROSS = (("R4", "R5p"), ("R4p", "R5"), ("R4p", "R5p"))
MAIN_PAIRS = (QUESTION,) + NOISE + (SEED, FULL, REPEAT)
PAIRS = MAIN_PAIRS + CROSS
LABEL = {QUESTION: "D", NOISE[0]: "N5", NOISE[1]: "N4", SEED: "S", FULL: "F", REPEAT: "rep",
         CROSS[0]: "R4|R5p", CROSS[1]: "R4p|R5", CROSS[2]: "R4p|R5p"}
COMMON = ("model", "tiny", "s", "prompts", "group", "max_new", "epochs", "minibatches", "lr", "eps", "wd", "tau",
          "tis_c", "clip_eps", "reward", "subset", "subset_seed")
WARN = ("device", "torch", "transformers", "s2_train_sha", "x4_train_sha", "cublas_workspace")
LOG_KEYS = ("tok", "mu", "pi_old", "raw_params", "reward")
NAN = float("nan")


def pname(pair):
    return f"{pair[0]}|{pair[1]}"


def fin(x):
    return x is not None and math.isfinite(x)


# ----------------------------------------------------------------------------- loading
class Arm:
    def __init__(self, tag, d):
        self.tag, self.dir = tag, d
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        if not isinstance(self.meta, dict):
            raise ValueError("meta.json is not an object")
        self.args = self.meta.get("args") if isinstance(self.meta.get("args"), dict) else {}
        self.log = {}
        lp = os.path.join(d, "log.jsonl")
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:      # e.g. a line cut by a crash
                        continue
                    if isinstance(rec, dict) and isinstance(rec.get("step"), int):
                        self.log[rec["step"]] = rec
        self.files = {}
        sd = os.path.join(d, "steps")
        if os.path.isdir(sd):
            for fn in os.listdir(sd):
                m = re.fullmatch(r"step_(\d+)\.npz", fn)
                if m:
                    self.files[int(m.group(1))] = os.path.join(sd, fn)
        self.steps = sorted(self.files)
        self.last = self.steps[-1] if self.steps else None
        self.done = os.path.exists(os.path.join(d, "done"))
        self.planned = self.args.get("steps")
        self.uaq = bool(self.meta.get("uaq"))
        self.s = float(self.meta.get("s", NAN))
        self.excluded = None

    def load(self, t):
        with np.load(self.files[t], allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def status(self):
        if not self.steps:
            return "no step files"
        gaps = sorted(set(range(self.last + 1)) - set(self.steps))
        s = f"steps 0..{self.last}" + (f" (missing {gaps})" if gaps else "")
        s += ", done" if self.done else f", NOT done (planned {self.planned})"
        return s


def discover(root):
    arms, notes = {}, []
    for tag in ARMS:
        d = os.path.join(root, tag)
        if not os.path.isdir(d):
            continue
        try:
            arms[tag] = Arm(tag, d)
        except Exception as e:  # noqa: BLE001  report and leave out any unreadable arm
            notes.append(f"{tag}: unreadable ({type(e).__name__}: {e}); left out")
    other = sorted(x for x in os.listdir(root)
                   if x not in ARMS and os.path.exists(os.path.join(root, x, "meta.json")))
    if other:
        notes.append(f"unknown arm directories ignored: {', '.join(other)}")
    return arms, notes


def check_config(arms, warnings=None):
    """Leave out arms that are misconfigured for their tag or inconsistent with R5 (else R4). Returns the reasons
    that make the verdict INVALID; differences in code versions or hardware go to `warnings` only."""
    invalid = []
    warnings = [] if warnings is None else warnings
    base = arms.get("R5") or arms.get("R4")
    for tag, arm in arms.items():
        m, why = arm.meta, []
        if m.get("format") != FORMAT:
            why.append(f"format {m.get('format')} (expected {FORMAT})")
        exp_arm, exp_pert, seed_diff, data_diff = EXPECTED[tag]
        if m.get("arm") != exp_arm:
            why.append(f"arm {m.get('arm')} (expected {exp_arm})")
        if sorted(m.get("perturb") or []) != sorted(exp_pert):
            why.append(f"perturb {m.get('perturb')} (expected {list(exp_pert)})")
        if base is not None:
            if (m.get("seed") != base.meta.get("seed")) != seed_diff:
                why.append(f"seed {m.get('seed')} vs {base.tag}'s {base.meta.get('seed')}")
            if (m.get("data_seed") != base.meta.get("data_seed")) != data_diff:
                why.append(f"data_seed {m.get('data_seed')} vs {base.tag}'s {base.meta.get('data_seed')}")
        if not arm.steps:
            why.append("no step files")
        if why:
            arm.excluded = "misconfigured: " + "; ".join(why)
    ref = next((arms[t] for t in ("R5", "R4") if t in arms and arms[t].excluded is None), None)
    if ref is not None:
        for tag, arm in arms.items():
            if arm is ref or arm.excluded is not None:
                continue
            diffs = [k for k in COMMON if arm.args.get(k) != ref.args.get(k)]
            diffs += [k for k in ("s", "probe_sha", "data_sha", "subsets") if arm.meta.get(k) != ref.meta.get(k)]
            if diffs:
                arm.excluded = f"differs from {ref.tag} in {', '.join(diffs)}"
            soft = [k for k in WARN if arm.meta.get(k) != ref.meta.get(k)]
            if soft:
                warnings.append(f"{tag} differs from {ref.tag} in {', '.join(soft)} (kept; check it is harmless)")
    for tag in QUESTION:
        if tag in arms and arms[tag].excluded is not None:
            invalid.append(f"{tag} {arms[tag].excluded}")
    return invalid


# ----------------------------------------------------------------------------- metrics
def mapped(arm, arrays):
    out = {}
    for g in GROUPS:
        v = arrays["raw_" + g].astype(np.float64)
        if arm.uaq and g == "W":
            v = v * arm.s
        elif arm.uaq and g == "gamma":
            v = v / arm.s
        out[g] = v
    return out


def rel_l2(a, b):
    num, den = float(np.sqrt(np.sum((a - b) ** 2))), float(np.sqrt(np.sum(b ** 2)))
    if den > 0:
        return num / den
    return 0.0 if num == 0 else float("inf")


def whole_model(ma, mb, sizes):
    num = sum(sizes[g][0] / sizes[g][1] * float(np.sum((ma[g] - mb[g]) ** 2)) for g in GROUPS)
    den = sum(sizes[g][0] / sizes[g][1] * float(np.sum(mb[g] ** 2)) for g in GROUPS)
    if den > 0:
        return math.sqrt(num / den)
    return 0.0 if num == 0 else float("inf")


def probe_dist(a, b, mask):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))[mask]
    return float(d.mean()) if d.size else NAN


def token_metrics(A, B):
    pa, pb = A["prompt_idx"], B["prompt_idx"]
    if pa.shape != pb.shape or A["resp"].shape[0] != B["resp"].shape[0] or pa.size == 0:
        return NAN, NAN
    G = A["resp"].shape[0] // pa.size
    rows = np.repeat(pa == pb, G)
    if not rows.any():
        return NAN, NAN
    L = max(A["resp"].shape[1], B["resp"].shape[1])

    def pad(x, fill):
        return np.pad(x, ((0, 0), (0, L - x.shape[1])), constant_values=fill)[rows]

    ra, rb = pad(A["resp"], -1), pad(B["resp"], -1)
    ma, mb = pad(A["mask"].astype(bool), False), pad(B["mask"].astype(bool), False)
    same, union = ma & mb & (ra == rb), ma | mb
    share = float(same.sum() / union.sum()) if union.sum() else NAN
    return share, float(np.mean(np.all(same == union, axis=1)))


def pair_metrics(A, B, ma, mb, sizes):
    same_shape = all(ma[g].shape == mb[g].shape for g in GROUPS)
    r = {f"P_{g}": rel_l2(ma[g], mb[g]) if same_shape else NAN for g in GROUPS}
    r["P_all"] = whole_model(ma, mb, sizes) if same_shape else NAN
    mask = A["probe_mask"].astype(bool)
    for k in ("pi32", "pi16", "mu"):
        a, b = A["probe_" + k], B["probe_" + k]
        r["Q_" + k] = probe_dist(a, b, mask) if a.shape == b.shape == mask.shape else NAN
    if "resp" in A and "resp" in B:
        r["tokens"], r["seqs"] = token_metrics(A, B)
        r["reward"] = abs(float(A["reward"].mean()) - float(B["reward"].mean()))
    return r


NEEDED = tuple(f"raw_{g}" for g in GROUPS) + ("probe_pi32", "probe_pi16", "probe_mu", "probe_mask")
NEEDED_ROLLOUT = ("prompt_idx", "resp", "mask", "reward")


def load_checked(arm, t, notes):
    """A step's arrays, or None (with a note) if the file is unreadable or lacks what the metrics need."""
    try:
        d = arm.load(t)
    except Exception as e:  # noqa: BLE001
        notes.append(f"{arm.tag} step {t}: unreadable ({type(e).__name__}); treated as missing")
        return None
    lack = [k for k in NEEDED + (NEEDED_ROLLOUT if t > 0 else ()) if k not in d]
    if lack:
        notes.append(f"{arm.tag} step {t}: lacks {lack}; treated as missing")
        return None
    return d


def ulp_check(base_raw, pert_raw):
    """Shares of elements exactly one FP32 ulp away (W, gamma) and unchanged (rest)."""
    out = {}
    for g in ("W", "gamma"):
        a, b = base_raw["raw_" + g].astype(np.float32), pert_raw["raw_" + g].astype(np.float32)
        one = (b == np.nextafter(a, np.float32(np.inf))) | (b == np.nextafter(a, np.float32(-np.inf)))
        out[g] = float(one.mean()) if one.size else NAN
    a, b = base_raw["raw_rest"], pert_raw["raw_rest"]
    out["rest_unchanged"] = float((a.view(np.uint32) == b.view(np.uint32)).mean()) if a.size else NAN
    return out


def arrays_equal(A, B):
    if set(A) != set(B):
        return "array names"
    for k in sorted(A):
        x, y = A[k], B[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return k
    return None


# ----------------------------------------------------------------------------- rule
def ratio(d, n):
    if n > 0:
        return d / n
    return 1.0 if d == 0 else float("inf")


def apply_rule(series, present, usable):
    """The pre-registered rule (module docstring). Returns a dict with the verdict and its evidence."""
    res = dict(primary=None, reasons=[], T=None, noise=[], metrics={}, seed=None)
    missing = [t for t in QUESTION if t not in usable]
    noise = [p for p in NOISE if pname(p) in series]
    res["noise"] = [pname(p) for p in noise]
    if missing or not noise:
        res["primary"] = "INCOMPLETE"
        res["reasons"].append(("missing " + ", ".join(missing)) if missing else "no noise pair (R5p and R4p missing)")
        return res
    lastp = {pname(p): max([t for t, x in enumerate(present[pname(p)]) if x], default=-1) for p in noise}
    T = min(usable["R4"].last, usable["R5"].last, max(lastp.values()))
    res["T"] = T
    if T < T_MIN:
        res["primary"] = "INCOMPLETE"
        res["reasons"].append(f"T = {T} < {T_MIN}")
        return res
    D = pname(QUESTION)
    for m in RULE_METRICS:
        ev = dict(n=0, above=[], below=[], skipped=[], nonfinite=[], max_hi=(None, None), min_lo=(None, None))
        for t in range(T_MIN, T + 1):
            nv = [series[pname(p)][m][t] for p in noise if present[pname(p)][t]]
            if not present[D][t] or not nv:
                ev["skipped"].append(t)
                continue
            d = series[D][m][t]
            if not fin(d) or not all(fin(x) for x in nv):
                ev["nonfinite"].append(t)
                continue
            lo, hi = min(nv), max(nv)
            ev["n"] += 1
            if d > FACTOR * hi:
                ev["above"].append(t)
            if d < lo / FACTOR:
                ev["below"].append(t)
            rh, rl = ratio(d, hi), ratio(d, lo)
            if ev["max_hi"][0] is None or rh > ev["max_hi"][0]:
                ev["max_hi"] = (rh, t)
            if ev["min_lo"][0] is None or rl < ev["min_lo"][0]:
                ev["min_lo"] = (rl, t)
        res["metrics"][m] = ev
    nonfin = {m: ev["nonfinite"] for m, ev in res["metrics"].items() if ev["nonfinite"]}
    above = any(ev["above"] for ev in res["metrics"].values())
    below = any(ev["below"] for ev in res["metrics"].values())
    if nonfin:
        res["primary"] = "INVALID"
        res["reasons"].append(f"non-finite rule metric at {nonfin}")
    elif not any(ev["n"] for ev in res["metrics"].values()):
        res["primary"] = "INCOMPLETE"
        res["reasons"].append("no evaluable step")
    elif above and below:
        res["primary"] = "MIXED"
    elif above:
        res["primary"] = "LARGER-THAN-ULP-NOISE"
    elif below:
        res["primary"] = "SMALLER-THAN-ULP-NOISE"
    else:
        res["primary"] = "ULP-NOISE-LIKE"
    S = pname(SEED)
    if S in series:
        both = [t for t in range(T_MIN, T + 1) if present[D][t] and present[S][t]]
        if both:
            exceed = {m: [t for t in both if series[D][m][t] > FACTOR * series[S][m][t]] for m in RULE_METRICS}
            Ts = both[-1]
            sat = {m: series[D][m][Ts] >= series[S][m][Ts] / FACTOR for m in RULE_METRICS}
            res["seed"] = dict(steps=len(both), Ts=Ts, exceed=exceed,
                               bounded=not any(exceed.values()), saturated=all(sat.values()),
                               ratio_at_Ts={m: ratio(series[D][m][Ts], series[S][m][Ts]) for m in RULE_METRICS})
    return res


# ----------------------------------------------------------------------------- report
def f_sci(x):
    if x is None or not isinstance(x, (int, float)) or math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf"
    return "0" if x == 0 else f"{x:.2e}"


def f_share(x):
    return "-" if x is None or math.isnan(x) else f"{x:.3f}"


def sha256_self():
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "?"


def analyze(root, report_path=None, json_path=None, echo=True):
    out = []
    p = out.append
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise SystemExit(f"x4_analysis.py: no such directory {root}")
    arms, notes = discover(root)
    invalid = check_config(arms, notes)
    usable = {t: a for t, a in arms.items() if a.excluded is None}
    p(f"X4 rounding-divergence test: report by x4_analysis.py (sha256 {sha256_self()[:16]})")
    p(f"results: {root}")
    p("")
    p("Arms:")
    for tag in ARMS:
        if tag not in arms:
            p(f"  {tag:<14} missing")
            continue
        a, m = arms[tag], arms[tag].meta
        desc = (f"arm {m.get('arm')} seed {m.get('seed')} data_seed {m.get('data_seed')} "
                f"perturb {','.join(m.get('perturb') or []) or '-'}")
        p(f"  {tag:<14} {a.status()}; {desc}" + (f"; LEFT OUT: {a.excluded}" if a.excluded else ""))
    for n in notes:
        p(f"  note: {n}")
    ref = usable.get("R5") or usable.get("R4") or next(iter(usable.values()), None)
    result = dict(verdict=None, primary=None, arms={t: dict(status=a.status(), excluded=a.excluded)
                                                    for t, a in arms.items()}, notes=notes)
    if ref is None:
        p("")
        result["primary"] = "INCOMPLETE"
        result["verdict"] = "VERDICT: INCOMPLETE (no usable arm); determinism: unchecked"
        p(result["verdict"])
        return finish(out, result, report_path, json_path, echo)
    ra, rm = ref.args, ref.meta
    p(f"Common settings (from {ref.tag}): model {ra.get('model')}{' (tiny)' if ra.get('tiny') else ''}, "
      f"s {rm.get('s')}, lr {ra.get('lr')}, {ra.get('prompts')} prompts x {ra.get('group')} samples, "
      f"max_new {ra.get('max_new')}, {ra.get('epochs')}x{ra.get('minibatches')} updates per batch, "
      f"tau {ra.get('tau')}, reward {ra.get('reward')}")
    p(f"  on {rm.get('device')} (torch {rm.get('torch')}, transformers {rm.get('transformers')}); "
      f"s2_train.py sha256 {str(rm.get('s2_train_sha'))[:16]}, x4_train.py sha256 {str(rm.get('x4_train_sha'))[:16]}")
    subs = rm.get("subsets") or {}
    sizes = {g: (float(subs[g]["N"]), float(subs[g]["K"])) for g in GROUPS if g in subs}
    p("  shared by every arm used: probe set (sha256 {}, {} tokens), data (sha256 {}), parameter subsets ({})".format(
        rm.get("probe_sha"), rm.get("probe_tokens"), rm.get("data_sha"),
        ", ".join(f"{g} {int(sizes[g][1])} of {int(sizes[g][0])}" for g in sizes)))
    if len(sizes) < len(GROUPS):
        invalid.append("parameter subset sizes missing from meta.json")
        sizes = {g: (1.0, 1.0) for g in GROUPS}
    p("")
    p("Activity per arm (means over steps): reward, share of updates clipped, TIS-truncated token share, "
      "ACR terms kept (total), minutes")
    for tag, a in usable.items():
        recs = [r for s, r in sorted(a.log.items()) if s > 0 and "reward" in r]
        if recs:
            p(f"  {tag:<14} reward {np.mean([r['reward'] for r in recs]):.3f}  "
              f"clipped {np.mean([r.get('clip_frac', NAN) for r in recs]):.2f}  "
              f"TIS {np.mean([r.get('tis_trunc_frac', NAN) for r in recs]):.4f}  "
              f"ACR {sum(int(r.get('acr_kept', 0)) for r in recs)}  "
              f"{sum(float(r.get('sec', 0)) for r in recs) / 60:.1f} min")

    # ---- per-step pass: pair metrics, movement, determinism, step-0 checks
    Tmax = max(a.last for a in usable.values())
    pairs = [pr for pr in PAIRS if pr[0] in usable and pr[1] in usable]
    series = {pname(pr): {m: [NAN] * (Tmax + 1) for m in METRICS} for pr in pairs}
    present = {pname(pr): [False] * (Tmax + 1) for pr in pairs}
    move = {t: {g: [NAN] * (Tmax + 1) for g in GROUPS + ("all",)} for t in usable}
    init, ulp, rep = {}, {}, dict(common=0, first=None, what=None)
    load_notes = []
    for t in range(Tmax + 1):
        data = {tag: load_checked(a, t, load_notes) for tag, a in usable.items() if t in a.files}
        data = {tag: d for tag, d in data.items() if d is not None}
        mp = {tag: mapped(usable[tag], d) for tag, d in data.items()}
        for pr in pairs:
            if pr[0] in data and pr[1] in data:
                for k, v in pair_metrics(data[pr[0]], data[pr[1]], mp[pr[0]], mp[pr[1]], sizes).items():
                    series[pname(pr)][k][t] = v
                present[pname(pr)][t] = True
        for tag in mp:
            if t == 0:
                init[tag] = mp[tag]
            if tag in init:
                for g in GROUPS:
                    move[tag][g][t] = rel_l2(mp[tag][g], init[tag][g])
                move[tag]["all"][t] = whole_model(mp[tag], init[tag], sizes)
        if t == 0:
            for base, pert in (("R5", "R5p"), ("R4", "R4p")):
                if base in data and pert in data:
                    ulp[pert] = ulp_check(data[base], data[pert])
        if "R4" in data and "R4_repeat" in data and rep["first"] is None:
            rep["common"] += 1
            what = arrays_equal(data["R4"], data["R4_repeat"])
            la, lb = usable["R4"].log.get(t), usable["R4_repeat"].log.get(t)
            if what is None and la is not None and lb is not None:
                what = next((f"log {k}" for k in LOG_KEYS if la.get(k) != lb.get(k)), None)
            if what is not None:
                rep["first"], rep["what"] = t, what
        del data, mp

    if "R4_repeat" in usable and "R4" in usable:
        if rep["first"] is not None:
            invalid.append(f"R4_repeat differs from R4 at step {rep['first']} ({rep['what']})")
            det = f"FAILED at step {rep['first']} ({rep['what']})"
        elif rep["common"] == 0:
            det = "unchecked (R4 and R4_repeat share no step)"
        else:
            det = f"verified (R4_repeat bit-identical to R4 at all {rep['common']} common steps)"
    else:
        det = "unchecked (no usable R4_repeat)"

    for n in load_notes:
        p(f"  note: {n}")
    notes.extend(load_notes)
    p("")
    p("1. Determinism, R4 vs R4_repeat (every recorded array and the log fingerprints of tokens, mu, pi_old and the raw "
      "parameters):")
    p(f"   {det}")

    p("")
    p("2. Step 0 (initial deviation, before any update):")
    p(f"   {'pair':<20}" + "".join(f"{m:>10}" for m in ("P_W", "P_gamma", "P_rest", "P_all", "Q_pi32", "Q_pi16", "Q_mu")))
    for pr in pairs:
        if present[pname(pr)][0]:
            p(f"   {pname(pr):<20}" + "".join(f"{f_sci(series[pname(pr)][m][0]):>10}"
                                             for m in ("P_W", "P_gamma", "P_rest", "P_all", "Q_pi32", "Q_pi16", "Q_mu")))
    for pert, c in ulp.items():
        p(f"   {pert} perturbation check vs {'R5' if pert == 'R5p' else 'R4'} (raw FP32): elements exactly one ulp "
          f"away: W {100 * c['W']:.3f}%, gamma {100 * c['gamma']:.3f}%; rest unchanged {100 * c['rest_unchanged']:.3f}%")
    D, N5 = pname(QUESTION), pname(NOISE[0])
    if D in series and N5 in series and present[D][0] and present[N5][0]:
        p("   initial size, (R4, R5) over (R5, R5p): " + ", ".join(
            f"{m} {f_sci(ratio(series[D][m][0], series[N5][m][0]))}" for m in ("P_W", "P_gamma", "Q_pi32", "Q_pi16", "Q_mu")))

    p("")
    p("3. Per-step curves. Pairs: " + ", ".join(f"{LABEL[pr]} = {pr[0]} vs {pr[1]}" for pr in MAIN_PAIRS
                                                if pname(pr) in series))
    cols = [pr for pr in MAIN_PAIRS if pname(pr) in series]
    tables = (("tokens", "share of identical sampled tokens (rows with the same prompt)", f_share),
              ("seqs", "share of identical sampled sequences", f_share),
              ("P_W", "P_W: relative L2 distance of W_QKV (rule metric)", f_sci),
              ("P_gamma", "P_gamma: relative L2 distance of the input-norm gains (rule metric)", f_sci),
              ("P_rest", "P_rest: relative L2 distance of the other parameters (rule metric)", f_sci),
              ("P_all", "P_all: whole-model estimate", f_sci),
              ("Q_pi32", "Q_pi32: mean |delta log pi| on the probe set, pi in FP32 (rule metric)", f_sci),
              ("Q_pi16", "Q_pi16: the same with pi in BF16 autocast (training precision)", f_sci),
              ("Q_mu", "Q_mu: the same under the INT8 W8A8 rollout policy mu", f_sci),
              ("reward", "|difference of mean rewards|", f_sci))
    for k, (m, title, fm) in enumerate(tables):
        p(f"3{chr(ord('a') + k)}. {title}")
        p("   step" + "".join(f"{LABEL[pr]:>10}" for pr in cols))
        for t in range(Tmax + 1):
            if any(present[pname(pr)][t] for pr in cols):
                p(f"   {t:>4}" + "".join(f"{fm(series[pname(pr)][m][t]) if present[pname(pr)][t] else '':>10}"
                                       for pr in cols))
    p(f"3{chr(ord('a') + len(tables))}. movement of each arm since its own step 0, P_all (and P_W) at selected steps")
    sel = sorted({t for t in (1, 2, 5, 10, 20, 30, 40, 50, Tmax) if t <= Tmax})
    p("   arm           " + "".join(f"{t:>20}" for t in sel))
    for tag in usable:
        p(f"   {tag:<14}" + "".join(f"{f_sci(move[tag]['all'][t]) + ' (' + f_sci(move[tag]['W'][t]) + ')':>20}"
                                    for t in sel))
    p("   first step at which the sampled tokens differ (share < 1) and fall below one half:")
    for pr in pairs:
        tok = series[pname(pr)]["tokens"]
        first = next((t for t in range(1, Tmax + 1) if present[pname(pr)][t] and fin(tok[t]) and tok[t] < 1), None)
        half = next((t for t in range(1, Tmax + 1) if present[pname(pr)][t] and fin(tok[t]) and tok[t] < 0.5), None)
        p(f"   {pname(pr):<20} {first if first is not None else '-':>6} {half if half is not None else '-':>6}")
    xs = [pr for pr in CROSS if pname(pr) in series]
    if xs:
        p("   cross pairs (information only), P_all / Q_pi32 at selected steps:")
        p("   pair                " + "".join(f"{t:>20}" for t in sel))
        for pr in xs:
            p(f"   {pname(pr):<20}" + "".join(
                f"{(f_sci(series[pname(pr)]['P_all'][t]) + '/' + f_sci(series[pname(pr)]['Q_pi32'][t])) if present[pname(pr)][t] else '':>20}"
                for t in sel))

    # ---- rule
    rule = apply_rule(series, present, usable)
    primary = "INVALID" if invalid else rule["primary"]
    p("")
    p(f"4. Rule (fixed in advance; see the docstring): D = R4 vs R5 against the noise band of "
      f"{', '.join(rule['noise']) or '-'}, steps {T_MIN} to T = {rule['T'] if rule['T'] is not None else '-'}, "
      f"factor {FACTOR:g}")
    if rule["metrics"]:
        p(f"   {'metric':<8}{'steps':>6}{'above':>7}{'below':>7}{'max D/hi (step)':>20}{'min D/lo (step)':>20}"
          f"  steps above / below")
        for m, ev in rule["metrics"].items():
            mh = f"{f_sci(ev['max_hi'][0])} ({ev['max_hi'][1]})" if ev["max_hi"][0] is not None else "-"
            ml = f"{f_sci(ev['min_lo'][0])} ({ev['min_lo'][1]})" if ev["min_lo"][0] is not None else "-"
            p(f"   {m:<8}{ev['n']:>6}{len(ev['above']):>7}{len(ev['below']):>7}{mh:>20}{ml:>20}"
              f"  {ev['above'] or '-'} / {ev['below'] or '-'}"
              + (f"; skipped {ev['skipped']}" if ev["skipped"] else "")
              + (f"; non-finite {ev['nonfinite']}" if ev["nonfinite"] else ""))
    for r in invalid + rule["reasons"]:
        p(f"   reason: {r}")
    p(f"   primary: {primary}")
    sd = rule.get("seed")
    if sd:
        p(f"   seed reference S = R5 vs R5_seed1, {sd['steps']} steps: "
          + ("SEED-BOUNDED (D <= 2 S at every step)" if sd["bounded"] else
             "EXCEEDS-SEED at " + "; ".join(f"{m} {v}" for m, v in sd["exceed"].items() if v)))
        p(f"   at step {sd['Ts']}: D/S = " + ", ".join(f"{m} {f_sci(v)}" for m, v in sd["ratio_at_Ts"].items())
          + (" -> SATURATED" if sd["saturated"] else " -> NOT-SATURATED"))
        for pr in NOISE:
            if pname(pr) in series and present[pname(pr)][sd["Ts"]]:
                p(f"   for comparison, {pname(pr)}/S at step {sd['Ts']}: " + ", ".join(
                    f"{m} {f_sci(ratio(series[pname(pr)][m][sd['Ts']], series[pname(SEED)][m][sd['Ts']]))}"
                    for m in RULE_METRICS))
        seed_txt = f"seed: {'SEED-BOUNDED' if sd['bounded'] else 'EXCEEDS-SEED'}, " \
                   f"{'SATURATED' if sd['saturated'] else 'NOT-SATURATED'} at step {sd['Ts']}"
    else:
        seed_txt = "seed: not evaluated" if pname(SEED) in series else "seed: no seed reference"
    partial =[t for t, a in usable.items() if not a.done or (a.planned is not None and a.last < a.planned)]
    result.update(primary=primary, T=rule["T"], rule=rule, seed=sd, determinism=det, invalid=invalid,
                  series={k: {m: [None if not fin(x) else x for x in v] for m, v in d.items()} for k, d in series.items()},
                  present=present, movement={t: {g: [None if not fin(x) else x for x in v] for g, v in d.items()}
                                            for t, d in move.items()},
                  ulp_check=ulp, partial=partial)
    result["verdict"] = (f"VERDICT: {primary}; {seed_txt}; determinism: {det.split(' (')[0]}"
                         + (f"; partial arms: {', '.join(partial)}" if partial else ""))
    p("")
    p(result["verdict"])
    return finish(out, result, report_path, json_path, echo)


def finish(out, result, report_path, json_path, echo):
    text = "\n".join(out) + "\n"
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if echo:
        sys.stdout.write(text)
    return result


# ----------------------------------------------------------------------------- self-test
def _st_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def _st_build(root, c, e, e4, sd, arms=ARMS, last=None, rep_break=None, meta_over=None, nan_at=None, gap=None,
              not_done=()):
    """Synthetic arms in x4_train.py's format with known distances (mapped coordinates, per group):
    R5 = theta; R5p = theta + e u; R4 = theta + c v; R4p = R4 + e4 w; R5_seed1 = theta + sd z; R5_seed1_full =
    theta + sd z2, with ||u_g|| = ... = ||theta_g||, so P(R4, R5) = c, P(R5, R5p) = e, P(R4, R4p) ~ e4, P(R5, S) = sd.
    Probe log-probs are shifted by the same amounts (pi16 by twice, mu by three times), so Q equals them exactly.
    Step 0 of R5p and R4p is an exact +-1 ulp perturbation instead."""
    T, s = len(c) - 1, 1.5
    last = dict(last or {})
    rng = np.random.default_rng(0)
    K = {"W": 600, "gamma": 48, "rest": 900}
    N = {"W": 2400, "gamma": 48, "rest": 9000}
    theta = {"W": 1.5 * rng.integers(1, 1024, K["W"]) * rng.choice([-1.0, 1.0], K["W"]) * 2.0 ** -12,
             "gamma": rng.integers(64, 512, K["gamma"]) * 2.0 ** -8,
             "rest": rng.normal(0, 0.02, K["rest"]).astype(np.float32).astype(np.float64)}

    def direction():
        out = {}
        for g in GROUPS:
            v = rng.normal(size=K[g])
            out[g] = v / np.linalg.norm(v) * np.linalg.norm(theta[g])
        return out

    u, v, w, z, z2 = (direction() for _ in range(5))
    PB, PL, B, L, G = 4, 7, 4, 6, 2
    pmask = np.ones((PB, PL), bool)
    pmask[1, 5:] = False
    pmask[3, 2:] = False
    L0 = -rng.uniform(0.1, 5, (PB, PL))
    resp5 = {t: rng.integers(0, 50, (B, L)).astype(np.int32) for t in range(1, T + 1)}
    mask5 = np.ones((B, L), bool)
    mask5[1, 4:] = False
    zero = [0.0] * (T + 1)
    spec = {"R5": (zero, None, None, 0.0), "R5p": (e, u, None, 1.0), "R4": (c, v, None, -1.0),
            "R4_repeat": (c, v, None, -1.0), "R4p": (c, v, e4, -1.0), "R5_seed1": (sd, z, None, 1.0),
            "R5_seed1_full": (sd, z2, None, -1.0)}
    sign = {g: rng.choice([np.inf, -np.inf], K[g]).astype(np.float32) for g in GROUPS}
    for tag in arms:
        coef, dirn, coef2, qs = spec[tag]
        arm, pert, seed_diff, data_diff = EXPECTED[tag]
        uaq = arm == "R4"
        d = os.path.join(root, tag)
        os.makedirs(os.path.join(d, "steps"))
        Tt = last.get(tag, T)
        meta = dict(format=FORMAT, tag=tag, arm=arm, uaq=uaq, s=s, seed=int(seed_diff), data_seed=int(data_diff),
                    perturb=list(pert), probe_sha="probe", data_sha="data", probe_tokens=int(pmask.sum()),
                    subsets={g: dict(N=N[g], K=K[g], idx_sha="i" + g, names_sha="n" + g) for g in GROUPS},
                    args=dict(model="synthetic", tiny=True, s=s, prompts=B // G, group=G, max_new=L, epochs=2,
                              minibatches=2, lr=1e-6, eps=1e-8, wd=0.01, tau=0.1, tis_c=2.0, clip_eps=0.2,
                              reward="toy", subset=K["rest"], subset_seed=1, steps=T),
                    device="synthetic", torch="-", transformers="-")
        meta.update((meta_over or {}).get(tag, {}))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
        logf = open(os.path.join(d, "log.jsonl"), "w")
        for t in range(Tt + 1):
            if gap and gap[0] == tag and gap[1] == t:
                continue
            mp = {g: theta[g] + (coef[t] * dirn[g] if dirn is not None else 0)
                  + (coef2[t] * w[g] if coef2 is not None else 0) for g in GROUPS}
            raw = {}
            for g in GROUPS:
                x = mp[g] / s if (uaq and g == "W") else mp[g] * s if (uaq and g == "gamma") else mp[g]
                raw[g] = x.astype(np.float32)
            if t == 0 and pert:   # exact +-1 ulp at step 0
                for g in pert:
                    raw[g] = np.nextafter(raw[g], sign[g])
            q = qs * (coef[t] + (coef2[t] if coef2 is not None else 0))
            arr = {f"raw_{g}": raw[g] for g in GROUPS}
            for k, mult in (("pi32", 1), ("pi16", 2), ("mu", 3)):
                arr["probe_" + k] = ((L0 + mult * q) * pmask).astype(np.float32)
            arr["probe_mask"] = pmask
            if nan_at and nan_at[0] == tag and nan_at[1] == t:
                arr["probe_pi32"][0, 0] = np.nan
            rec = {"step": t, "raw_params": {g: _st_sha(raw[g]) for g in GROUPS}}
            if t >= 1:
                resp, msk = resp5[t].copy(), mask5.copy()
                if tag in ("R4", "R4_repeat", "R4p") and t >= 2:
                    resp[0, 3:] += 1
                if tag in ("R5p", "R4p") and t >= 3:
                    resp[1, 1:] += 1
                if tag in ("R5_seed1", "R5_seed1_full"):
                    resp += 100
                pidx = np.arange(B // G, dtype=np.int64) + 1000 * t + (500 if tag == "R5_seed1_full" else 0)
                arr.update(prompt_idx=pidx, resp=resp, mask=msk, mu=np.zeros((B, L), np.float32),
                           lp_old=np.zeros((B, L), np.float32), reward=np.full(B, 0.25 + q, np.float32),
                           adv=np.zeros(B, np.float32))
                rec.update(tok=_st_sha(resp * msk), mu="m", pi_old="p", reward=0.25 + q, clip_frac=1.0,
                           tis_trunc_frac=0.0, acr_kept=0, sec=1.0)
            if rep_break and tag == "R4_repeat" and t == rep_break[0]:
                if rep_break[1] == "log":
                    rec["tok"] = "changed"
                else:
                    arr["raw_W"] = arr["raw_W"].copy()
                    arr["raw_W"][0] = np.nextafter(arr["raw_W"][0], np.float32(np.inf))
            np.savez(os.path.join(d, "steps", f"step_{t:04d}.npz"), **arr)
            logf.write(json.dumps(rec) + "\n")
        logf.close()
        if tag not in not_done:
            open(os.path.join(d, "done"), "w").close()


def self_test():
    T = 12
    e = [0.0] + [1e-3 * 1.25 ** t for t in range(1, T + 1)]

    def times(k, x):
        return [k * y for y in x]

    sd = [0.0] + [0.04] * T
    cases = []

    def case(name, expect, seed=None, det=None, check=None, **kw):
        cases.append((name, expect, seed, det, check, kw))

    def known(r):   # known-answer checks on the basic scenario
        S = r["series"]
        c = times(1.5, e)
        D, N5, N4, rep = S["R4|R5"], S["R5|R5p"], S["R4|R4p"], S["R4|R4_repeat"]
        assert D["P_W"][0] == 0 and D["P_gamma"][0] == 0 and D["P_rest"][0] == 0, "UAQ map not exact at step 0"
        for t in range(1, T + 1):
            for m in ("P_W", "P_gamma", "P_rest", "P_all"):
                # D's denominator is R5 = theta (exact); the noise pairs divide by the perturbed arm, which shifts
                # them by about e * cos(theta, u), up to ~1% for the 48-element gamma group.
                assert abs(D[m][t] / c[t] - 1) < 1e-3, (m, t, D[m][t], c[t])
                assert abs(N5[m][t] / e[t] - 1) < 2e-2, (m, t, N5[m][t], e[t])
                assert abs(N4[m][t] / (0.8 * e[t]) - 1) < 5e-2, (m, t, N4[m][t])
            assert abs(D["Q_pi32"][t] / c[t] - 1) < 1e-4 and abs(D["Q_pi16"][t] / (2 * c[t]) - 1) < 1e-4
            assert abs(D["Q_mu"][t] / (3 * c[t]) - 1) < 1e-4
            assert rep["P_all"][t] == 0 and rep["Q_pi32"][t] == 0 and rep["tokens"][t] == 1.0
            assert S["R5|R5_seed1_full"]["tokens"][t] is None, "different prompts must give no token share"
        assert D["tokens"][1] == 1.0 and D["seqs"][1] == 1.0
        assert abs(D["tokens"][2] - 19 / 22) < 1e-12 and D["seqs"][2] == 0.75, (D["tokens"][2], D["seqs"][2])
        assert abs(S["R4|R5p"]["tokens"][3] - 16 / 22) < 1e-12, S["R4|R5p"]["tokens"][3]
        assert abs(S["R4p|R5p"]["tokens"][3] - 19 / 22) < 1e-12, S["R4p|R5p"]["tokens"][3]
        assert S["R5|R5_seed1"]["tokens"][1] == 0.0
        for pert in ("R5p", "R4p"):
            u = r["ulp_check"][pert]
            assert u["W"] == 1.0 and u["gamma"] == 1.0 and u["rest_unchanged"] == 1.0, (pert, u)

    def above_from_8(r):
        got = r["rule"]["metrics"]["P_W"]["above"]
        assert got == list(range(8, T + 1)), got

    def all_steps_evaluated(r):
        ev = r["rule"]["metrics"]["P_W"]
        assert ev["n"] == T - T_MIN + 1 and "R4p" in r["partial"], (ev, r["partial"])

    def step6_skipped(r):
        ev = r["rule"]["metrics"]["P_W"]
        assert ev["skipped"] == [6] and ev["n"] == T - T_MIN, ev

    def r5p_left_out(r):
        assert r["arms"]["R5p"]["excluded"] and r["rule"]["noise"] == ["R4|R4p"], (r["arms"]["R5p"], r["rule"])

    def r5p_misconfigured(r):
        assert (r["arms"]["R5p"]["excluded"] or "").startswith("misconfigured"), r["arms"]["R5p"]

    def r5p_warned(r):
        assert r["arms"]["R5p"]["excluded"] is None and r["rule"]["noise"] == ["R5|R5p", "R4|R4p"], r["arms"]["R5p"]
        assert any(n.startswith("R5p differs from R5 in torch") for n in r["notes"]), r["notes"]

    case("noise-like, all arms; known answers", "ULP-NOISE-LIKE", seed=("SEED-BOUNDED", "SATURATED"), det="verified",
         check=known, c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd)
    case("larger from step 8", "LARGER-THAN-ULP-NOISE", check=above_from_8,
         c=[1.5 * x if t < 8 else 5 * x for t, x in enumerate(e)], e=e, e4=times(0.8, e), sd=sd)
    case("smaller", "SMALLER-THAN-ULP-NOISE", c=times(0.2, e), e=e, e4=times(0.8, e), sd=sd)
    case("mixed", "MIXED", c=[5 * x if t == 6 else 0.2 * x if t == 9 else 1.5 * x for t, x in enumerate(e)],
         e=e, e4=times(0.8, e), sd=sd)
    case("transient before step 5 ignored", "ULP-NOISE-LIKE",
         c=[100 * x if t < 5 else 1.5 * x for t, x in enumerate(e)], e=e, e4=times(0.8, e), sd=times(100, sd))
    case("envelope of two noise pairs", "ULP-NOISE-LIKE", c=times(2.5, e), e=e, e4=times(1.5, e), sd=sd)
    case("same, without R4p: one noise pair", "LARGER-THAN-ULP-NOISE", c=times(2.5, e), e=e, e4=times(1.5, e), sd=sd,
         arms=tuple(t for t in ARMS if t != "R4p"))
    case("R4_repeat differs in an array at step 7", "INVALID", det="FAILED", c=times(1.5, e), e=e,
         e4=times(0.8, e), sd=sd, rep_break=(7, "raw_W"))
    case("R4_repeat differs in the log at step 5", "INVALID", det="FAILED", c=times(1.5, e), e=e,
         e4=times(0.8, e), sd=sd, rep_break=(5, "log"))
    case("no R4_repeat", "ULP-NOISE-LIKE", det="unchecked", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         arms=tuple(t for t in ARMS if t != "R4_repeat"))
    case("no noise arms", "INCOMPLETE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         arms=("R4", "R5", "R4_repeat", "R5_seed1"))
    case("no R4", "INCOMPLETE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         arms=tuple(t for t in ARMS if t not in ("R4", "R4_repeat")))
    case("noise arms stop at step 4", "INCOMPLETE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         last={"R5p": 4, "R4p": 4}, not_done=("R5p", "R4p"))
    case("R4p partial (to 8), R5p gap at 6", "ULP-NOISE-LIKE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         last={"R4p": 8}, gap=("R5p", 6), not_done=("R4p",), check=all_steps_evaluated)
    case("only R5p, with a gap at 6", "ULP-NOISE-LIKE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         arms=("R4", "R5", "R5p"), gap=("R5p", 6), check=step6_skipped)
    case("exceeds the seed change", "ULP-NOISE-LIKE", seed=("EXCEEDS-SEED", "SATURATED"),
         c=times(1.5, e), e=e, e4=times(0.8, e), sd=[0.0] + [0.005] * T)
    case("seed change far away", "ULP-NOISE-LIKE", seed=("SEED-BOUNDED", "NOT-SATURATED"),
         c=times(1.5, e), e=e, e4=times(0.8, e), sd=[0.0] + [0.2] * T)
    case("R5p from another probe set: left out", "ULP-NOISE-LIKE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         meta_over={"R5p": {"probe_sha": "other"}}, check=r5p_left_out)
    case("R5p without its perturbation: left out", "ULP-NOISE-LIKE", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         meta_over={"R5p": {"perturb": []}}, check=r5p_misconfigured)
    case("R5p on another torch: kept, with a warning", "ULP-NOISE-LIKE", c=times(1.5, e), e=e, e4=times(0.8, e),
         sd=sd, meta_over={"R5p": {"torch": "other"}}, check=r5p_warned)
    case("R4 with another s", "INVALID", c=times(1.5, e), e=e, e4=times(0.8, e), sd=sd,
         meta_over={"R4": {"s": 2.0}, "R4_repeat": {"s": 2.0}})
    case("non-finite rule metric (R5's probe at step 6)", "INVALID", det="verified", c=times(1.5, e), e=e,
         e4=times(0.8, e), sd=sd, nan_at=("R5", 6))
    case("zero noise distance (D > 0)", "LARGER-THAN-ULP-NOISE", c=times(1.5, e), e=[0.0] * (T + 1),
         e4=[0.0] * (T + 1), sd=sd)
    case("R4 and R5 only (no noise, no repeat)", "INCOMPLETE", det="unchecked", c=times(1.5, e), e=e,
         e4=times(0.8, e), sd=sd, arms=("R4", "R5"))

    tmp = tempfile.mkdtemp(prefix="x4_selftest_")
    failed = 0
    try:
        for i, (name, expect, seed, det, check, kw) in enumerate(cases):
            root = os.path.join(tmp, f"case{i:02d}")
            os.makedirs(root)
            _st_build(root, **kw)
            try:
                r = analyze(root, report_path=os.path.join(root, "x4_report.txt"),
                            json_path=os.path.join(root, "x4_metrics.json"), echo=False)
                json.load(open(os.path.join(root, "x4_metrics.json")))
                assert r["primary"] == expect, f"primary {r['primary']}, expected {expect}"
                if seed:
                    sdr = r["seed"]
                    got = ("SEED-BOUNDED" if sdr["bounded"] else "EXCEEDS-SEED",
                           "SATURATED" if sdr["saturated"] else "NOT-SATURATED")
                    assert got == seed, f"seed {got}, expected {seed}"
                if det:
                    assert r["determinism"].startswith(det), f"determinism {r['determinism']}, expected {det}"
                assert r["verdict"].startswith("VERDICT: " + expect)
                with open(os.path.join(root, "x4_report.txt")) as f:
                    assert f.read().rstrip().splitlines()[-1] == r["verdict"]
                if check:
                    check(r)
                print(f"  pass  {name}: {r['verdict']}")
            except AssertionError as err:
                failed += 1
                print(f"  FAIL  {name}: {err}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"self-test: {len(cases) - failed} of {len(cases)} cases passed")
    return failed == 0


def main():
    ap = argparse.ArgumentParser(description="X4 analysis (rule in the module docstring)")
    ap.add_argument("root", nargs="?", help="results directory, e.g. ~/qurl/rev/results/x4_rounding")
    ap.add_argument("--results", default=None, help="results directory (instead of the positional argument)")
    ap.add_argument("--report", "--out", dest="report", default=None,
                    help="report path (default <results>/x4_report.txt)")
    ap.add_argument("--json", default=None, help="metrics path (default <results>/x4_metrics.json)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    root = a.root or a.results
    if not root:
        ap.error("results directory required")
    analyze(root, a.report or os.path.join(root, "x4_report.txt"), a.json or os.path.join(root, "x4_metrics.json"))


if __name__ == "__main__":
    main()
