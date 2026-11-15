"""E2, weight space (P1 experiments report, E2 design): Assumption U over training and Propositions 3, 4, 6 and 7
on the synced FP32 master weights of the E2 run.

  python e2_dynamic.py <dumps dir> [--out report.txt] [--json data.json]
  python e2_dynamic.py --self-test

Grid units throughout, with the rollout quantizer's arithmetic: Delta = row absmax * fl32(1/127), q = w / Delta,
code c = round(q), phase phi = q - c. Elements with |c| = 127 (each row's absmax) are left out. Every sync has its own
grid: when a row's absmax moves, its whole grid moves, and the exact change of q between syncs includes that.
Full dumps (all 168 linear weights at selected syncs; sync 1 holds theta_0):
  U    phases at every full-dump sync: share with |phi| >= 0.45 against 0.10, largest deviation of the 40 bin densities
       (all codes; codes |c| <= 15), exact ties.
  P3   bounded lag: max |e_k - e_0| / max(Delta_k, Delta_0) over all elements must be <= 1 (exact; a consistency check).
  P4   one-step capture for consecutive full dumps (k, k+1): flip share F against P = mean min(1, |u|), u = q_{k+1} - q_k,
       for all elements and without the elements that were exact ties at sync 1; observed against predicted flip
       rates in bins of |u| (weighted slope through the origin); capture slope sum(dc u) / sum(u^2); and, for
       comparison, the fixed-grid prediction mean min(1, |w_{k+1} - w_k| / Delta_k).
  P6   persistence against sync 1, without sync 1's exact ties: Corr(phi_k, phi_0) in 20 bins of a = frac(|q_k - q_0|)
       against the bin mean of 1 - 6a(1 - a) (bins with >= 10^4 elements); Var(phi_k) / Var(phi_0).
Row samples (every sync):
  P6c  Corr(phi_k, phi_0) at every sync k against mean(1 - 6a(1 - a)) from the measured displacements.
  P7   RTN, stochastic rounding (fresh dither, seeded) and error feedback on the recorded path: rms of the partial
       sums sum_{k<=K} e_k in units of the row's first Delta, at K = 1, 10, 100 and the last sync; EF's
       |sum| <= Delta_K / 2 (exact); SR's rms against sqrt(K / 6).
Decision rules (pre-registered in the P1 experiments report, E2 design) are applied in summary().
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import torch

QMAX = 127.0
LINEAR = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$")
NB, A_BINS, MIN_BIN = 40, 20, 10_000
U_EDGES = torch.tensor([10 ** (-5 + i / 4) for i in range(21)], dtype=torch.float64)   # |u| bins, 1e-5 .. 1
LATE_PAIR_FROM = 65            # P4 rule (a) applies to one-step pairs starting at this sync or later


def grid(w):
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = w / scale
    return q, torch.round(q), scale


def syncs_of(d, sub):
    return sorted({int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(d, sub, "sync*_b*.pt"))})


def load_linear(d, k):
    st = {}
    for f in sorted(glob.glob(os.path.join(d, "full", f"sync{k:04d}_b*.pt"))):
        st.update({n: t for n, t in torch.load(f, map_location="cpu").items() if LINEAR.search(n)})
    return st


# ----------------------------------------------------------------------------------------------- full dumps
def u_stats(st):
    h = torch.zeros(NB, dtype=torch.float64)
    hs = torch.zeros(NB, dtype=torch.float64)
    n = near = ties = 0
    s1 = s2 = 0.0
    for w in st.values():
        q, c, _ = grid(w)
        keep = c.abs() < QMAX
        p = (q - c)[keep].double()
        small = c.abs()[keep] <= 15
        h += torch.histc(p, NB, -0.5, 0.5)
        hs += torch.histc(p[small], NB, -0.5, 0.5)
        n += p.numel()
        near += int((p.abs() >= 0.45).sum())
        ties += int((p.abs() == 0.5).sum())
        s1 += float(p.sum())
        s2 += float((p * p).sum())
    dev = lambda hh: float((hh * NB / hh.sum() - 1).abs().max())
    return {"n": n, "near_ratio": near / n / 0.10, "max_dev_all": dev(h), "max_dev_small": dev(hs),
            "ties": ties / n, "var_phi": s2 / n - (s1 / n) ** 2}


def tie_masks(st):
    out = {}
    for name, w in st.items():
        q, c, _ = grid(w)
        out[name] = (q - c).abs() == 0.5
    return out


def pair_stats(a, b, ties0):
    acc = {k: 0.0 for k in ("n", "F", "P", "Pfix", "n2", "F2", "P2", "dcu", "uu", "su", "sp", "suu", "spp", "spu")}
    nb = len(U_EDGES) + 1
    cal = {k: torch.zeros(nb, dtype=torch.float64) for k in ("n", "f", "p")}
    for name in a:
        qa, ca, sa = grid(a[name])
        qb, cb, _ = grid(b[name])
        keep = (ca.abs() < QMAX) & (cb.abs() < QMAX)
        u = qb.double() - qa.double()
        flip = (cb != ca).double()
        pred = u.abs().clamp(max=1.0)
        pfix = ((b[name].double() - a[name].double()) / sa.double()).abs().clamp(max=1.0)
        acc["n"] += float(keep.sum())
        acc["F"] += float(flip[keep].sum())
        acc["P"] += float(pred[keep].sum())
        acc["Pfix"] += float(pfix[keep].sum())
        k2 = keep & ~ties0[name]
        uk, fk, pk = u[k2], flip[k2], pred[k2]
        acc["n2"] += float(k2.sum())
        acc["F2"] += float(fk.sum())
        acc["P2"] += float(pk.sum())
        acc["dcu"] += float(((cb - ca).double()[k2] * uk).sum())
        acc["uu"] += float((uk * uk).sum())
        phi = (qa - ca).double()[k2]
        acc["su"] += float(uk.sum()); acc["sp"] += float(phi.sum())
        acc["suu"] += float((uk * uk).sum()); acc["spp"] += float((phi * phi).sum()); acc["spu"] += float((phi * uk).sum())
        idx = torch.bucketize(uk.abs(), U_EDGES)
        cal["n"] += torch.bincount(idx, minlength=nb).double()
        cal["f"] += torch.bincount(idx, weights=fk, minlength=nb)
        cal["p"] += torch.bincount(idx, weights=pk, minlength=nb)
    m = cal["n"] >= MIN_BIN
    fr, pr = cal["f"][m] / cal["n"][m], cal["p"][m] / cal["n"][m]
    w = cal["n"][m]
    cal_slope = float((w * fr * pr).sum() / (w * pr * pr).sum()) if m.any() else float("nan")
    n2 = acc["n2"]
    cov = acc["spu"] / n2 - (acc["sp"] / n2) * (acc["su"] / n2)
    corr_phi_u = cov / math.sqrt(max(acc["spp"] / n2 - (acc["sp"] / n2) ** 2, 1e-300) *
                                 max(acc["suu"] / n2 - (acc["su"] / n2) ** 2, 1e-300))
    return {"F": acc["F"] / acc["n"], "P": acc["P"] / acc["n"], "ratio_all": acc["F"] / acc["P"],
            "Pfix": acc["Pfix"] / acc["n"], "F_no_ties": acc["F2"] / n2, "P_no_ties": acc["P2"] / n2,
            "ratio_no_ties": acc["F2"] / acc["P2"], "cal_slope": cal_slope, "cal_bins": int(m.sum()),
            "capture_slope": acc["dcu"] / acc["uu"], "corr_phi_u": corr_phi_u, "elements": int(acc["n"])}


def persistence_stats(a0, b, ties0):
    z = lambda: torch.zeros(A_BINS, dtype=torch.float64)
    S = {k: z() for k in ("n", "x", "y", "xx", "yy", "xy", "pred")}
    vx = vy = 0.0
    nn = 0
    lag = 0.0
    for name in a0:
        q0, c0, s0 = grid(a0[name])
        qk, ck, sk = grid(b[name])
        e0 = (c0 - q0).double() * s0.double()
        ek = (ck - qk).double() * sk.double()
        lag = max(lag, float(((ek - e0).abs() / torch.maximum(sk, s0).double()).max()))
        keep = (c0.abs() < QMAX) & (ck.abs() < QMAX) & ~ties0[name]
        x, y = (qk - ck).double()[keep], (q0 - c0).double()[keep]
        a = (qk.double() - q0.double()).abs()[keep]
        a = a - torch.floor(a)
        pred = 1 - 6 * a * (1 - a)
        idx = torch.clamp((a * A_BINS).long(), max=A_BINS - 1)
        for key, v in (("x", x), ("y", y), ("xx", x * x), ("yy", y * y), ("xy", x * y), ("pred", pred)):
            S[key] += torch.bincount(idx, weights=v, minlength=A_BINS)
        S["n"] += torch.bincount(idx, minlength=A_BINS).double()
        nn += x.numel()
        vx += float((x * x).sum()); vy += float((y * y).sum())
    n = S["n"].clamp(min=1)
    cov = S["xy"] / n - (S["x"] / n) * (S["y"] / n)
    corr = cov / torch.sqrt((S["xx"] / n - (S["x"] / n) ** 2) * (S["yy"] / n - (S["y"] / n) ** 2)).clamp(min=1e-300)
    pred = S["pred"] / n
    m = S["n"] >= MIN_BIN
    dev = float((corr[m] - pred[m]).abs().max()) if m.any() else float("nan")
    return {"bins_used": int(m.sum()), "max_abs_dev": dev, "corr": [float(v) for v in corr],
            "pred": [float(v) for v in pred], "n": [int(v) for v in S["n"]],
            "var_ratio": (vx / nn) / (vy / nn), "lag_ratio_max": lag}


# ----------------------------------------------------------------------------------------------- row samples
def load_rows(d):
    index = torch.load(os.path.join(d, "rows", "index.pt"))
    ks = syncs_of(d, "rows")
    series = {n: [] for n in index}
    for k in ks:
        part = {}
        for f in sorted(glob.glob(os.path.join(d, "rows", f"sync{k:04d}_b*.pt"))):
            part.update(torch.load(f, map_location="cpu"))
        for n in index:
            series[n].append(part[n])
    return ks, {n: torch.stack(v) for n, v in series.items()}       # [K, R, in]


def row_stats(ks, series, seed=0):
    K = len(ks)
    g = torch.Generator().manual_seed(seed)
    corr_num = torch.zeros(K, dtype=torch.float64); corr_x2 = torch.zeros(K, dtype=torch.float64)
    corr_y2 = 0.0
    pred_sum = torch.zeros(K, dtype=torch.float64)
    n_el = 0
    marks = sorted({m for m in (1, 10, 100) if m <= K} | {K})
    psum = {s: {m: 0.0 for m in marks} for s in ("rtn", "sr", "ef")}
    ef_ratio_max = 0.0
    sx = torch.zeros(K, dtype=torch.float64); sy = 0.0
    for name, W in series.items():
        W = W.double()
        scale = (W.abs().amax(dim=2, keepdim=True) * (1.0 / QMAX)).float().double()   # the quantizer's FP32 scale
        q = (W.float() / scale.float()).double()
        c = torch.round(q)
        keep = (c[0].abs() < QMAX)
        phi = (q - c)
        x0 = phi[0][keep]
        a = (q - q[0:1]).abs()
        a = a - torch.floor(a)
        for k in range(K):
            xk = phi[k][keep]
            corr_num[k] += float((xk * x0).sum())
            corr_x2[k] += float((xk * xk).sum())
            sx[k] += float(xk.sum())
            pred_sum[k] += float((1 - 6 * a[k] * (1 - a[k]))[keep].sum())
        corr_y2 += float((x0 * x0).sum())
        sy += float(x0.sum())
        n_el += int(keep.sum())
        d0 = scale[0]                                                   # [R, 1]
        S_rtn = torch.zeros_like(W[0]); S_sr = torch.zeros_like(W[0]); r = torch.zeros_like(W[0])
        for k in range(K):
            s = scale[k]
            e_rtn = torch.round(W[k] / s) * s - W[k]
            xi = torch.rand(W[k].shape, generator=g, dtype=torch.float64)
            e_sr = torch.floor(W[k] / s + xi) * s - W[k]
            v = W[k] + r
            qv = torch.clamp(torch.round(v / s), -QMAX, QMAX) * s
            r = v - qv
            e_ef = qv - W[k]
            S_rtn += e_rtn; S_sr += e_sr
            ef_ratio_max = max(ef_ratio_max, float((r.abs() / (s / 2)).max()))
            if (k + 1) in marks:
                psum["rtn"][k + 1] += float(((S_rtn / d0) ** 2).sum())
                psum["sr"][k + 1] += float(((S_sr / d0) ** 2).sum())
                psum["ef"][k + 1] += float(((r / d0) ** 2).sum())      # sum of EF errors = r_0 - r_K = -r_K
    n_all = sum(int(W[0].numel()) for W in series.values())
    rms = {s: {m: math.sqrt(v / n_all) for m, v in d.items()} for s, d in psum.items()}
    mx = sx / n_el; my = sy / n_el
    corr = (corr_num / n_el - mx * my) / torch.sqrt((corr_x2 / n_el - mx ** 2) * (corr_y2 / n_el - my ** 2))
    pred = pred_sum / n_el
    return {"syncs": len(ks), "elements": n_el, "corr": [float(v) for v in corr], "pred": [float(v) for v in pred],
            "max_abs_dev": float((corr - pred).abs().max()), "partial_sum_rms": rms,
            "sr_ratio_last": rms["sr"][K] / math.sqrt(K / 6.0), "ef_bound_max": ef_ratio_max}


# ----------------------------------------------------------------------------------------------- driver
def analyse(d):
    full = syncs_of(d, "full")
    first = load_linear(d, full[0])
    ties0 = tie_masks(first)
    out = {"full_syncs": full, "u": {}, "pairs": {}, "persistence": {}}
    prev, prev_k = first, full[0]
    out["u"][full[0]] = u_stats(first)
    for k in full[1:]:
        cur = load_linear(d, k)
        out["u"][k] = u_stats(cur)
        if k == prev_k + 1:
            out["pairs"][f"{prev_k}-{k}"] = pair_stats(prev, cur, ties0)
        out["persistence"][k] = persistence_stats(first, cur, ties0)
        prev, prev_k = cur, k
    ks, series = load_rows(d)
    out["rows"] = row_stats(ks, series)
    return out


def rules(r):
    """The pre-registered decision rules (P1 experiments report, E2 design). Returns (name, pass, detail) triples."""
    res = []
    for k, s in r["u"].items():
        res.append((f"U1 sync {k}: near-boundary share / 0.10 in [0.95, 1.05]", 0.95 <= s["near_ratio"] <= 1.05,
                    f"{s['near_ratio']:.4f}"))
        res.append((f"U2 sync {k}: codes |c|<=15, largest bin deviation <= 0.05", s["max_dev_small"] <= 0.05,
                    f"{s['max_dev_small']:.4f}"))
    for pk, s in r["pairs"].items():
        late = int(pk.split("-")[0]) >= LATE_PAIR_FROM
        if late:
            res.append((f"P4a pair {pk}: F / P in [0.90, 1.10] (all elements)", 0.9 <= s["ratio_all"] <= 1.1, f"{s['ratio_all']:.4f}"))
        res.append((f"P4a' pair {pk}: F / P in [0.90, 1.10] (without sync-1 ties)", 0.9 <= s["ratio_no_ties"] <= 1.1, f"{s['ratio_no_ties']:.4f}"))
        res.append((f"P4b pair {pk}: calibration slope in [0.90, 1.10]", 0.9 <= s["cal_slope"] <= 1.1, f"{s['cal_slope']:.4f}"))
        res.append((f"P4c pair {pk}: capture slope in [0.95, 1.05]", 0.95 <= s["capture_slope"] <= 1.05, f"{s['capture_slope']:.4f}"))
    for k, s in r["persistence"].items():
        res.append((f"P3 sync {k}: max |e_k - e_0| / max(Delta) <= 1", s["lag_ratio_max"] <= 1 + 1e-9, f"{s['lag_ratio_max']:.4f}"))
        res.append((f"P6a sync {k}: Var(phi_k) / Var(phi_0) in [0.98, 1.02]", 0.98 <= s["var_ratio"] <= 1.02, f"{s['var_ratio']:.4f}"))
        if s["bins_used"]:
            res.append((f"P6b sync {k}: max |Corr - (1-6a(1-a))| <= 0.05 over {s['bins_used']} bins", s["max_abs_dev"] <= 0.05,
                        f"{s['max_abs_dev']:.4f}"))
    rw = r["rows"]
    res.append(("P6c rows: max over syncs |Corr(phi_k, phi_0) - mean(1-6a(1-a))| <= 0.05", rw["max_abs_dev"] <= 0.05, f"{rw['max_abs_dev']:.4f}"))
    res.append(("P7a rows: error feedback |sum e| <= Delta_K / 2", rw["ef_bound_max"] <= 1 + 1e-9, f"{rw['ef_bound_max']:.6f}"))
    res.append(("P7b rows: SR rms partial sum / sqrt(K/6) in [0.90, 1.10]", 0.9 <= rw["sr_ratio_last"] <= 1.1, f"{rw['sr_ratio_last']:.4f}"))
    return res


def report(r):
    lines = ["E2 weight-space analysis (e2_dynamic.py)", f"full dumps at syncs {r['full_syncs']}", "",
             "Assumption U at each full-dump sync (codes |c| < 127):",
             f"{'sync':>5} {'elements':>11} {'near/0.10':>9} {'dev all':>8} {'dev |c|<=15':>11} {'ties':>9} {'Var(phi)*12':>11}"]
    for k, s in r["u"].items():
        lines.append(f"{k:5d} {s['n']:11d} {s['near_ratio']:9.4f} {s['max_dev_all']:8.4f} {s['max_dev_small']:11.4f}"
                     f" {s['ties']:9.2e} {12 * s['var_phi']:11.4f}")
    lines += ["", "Proposition 4, one-step pairs (flip share F, prediction P = mean min(1, |u|)):",
              f"{'pair':>9} {'F':>10} {'P':>10} {'F/P':>7} {'F/P no ties':>11} {'P fixed grid':>12} {'cal slope':>9}"
              f" {'capture':>8} {'corr(phi,u)':>11}"]
    for pk, s in r["pairs"].items():
        lines.append(f"{pk:>9} {s['F']:10.3e} {s['P']:10.3e} {s['ratio_all']:7.4f} {s['ratio_no_ties']:11.4f}"
                     f" {s['Pfix']:12.3e} {s['cal_slope']:9.4f} {s['capture_slope']:8.4f} {s['corr_phi_u']:11.4f}")
    lines += ["", "Propositions 3 and 6 against sync 1 (no sync-1 ties):",
              f"{'sync':>5} {'Var ratio':>9} {'bins':>5} {'max |dev|':>9} {'lag max':>8}   Corr / prediction by a-bin (bins with >= 1e4)"]
    for k, s in r["persistence"].items():
        bins = " ".join(f"{c:.2f}/{p:.2f}" for c, p, n in zip(s["corr"], s["pred"], s["n"]) if n >= MIN_BIN)
        lines.append(f"{k:5d} {s['var_ratio']:9.4f} {s['bins_used']:5d} {s['max_abs_dev']:9.4f} {s['lag_ratio_max']:8.4f}   {bins}")
    rw = r["rows"]
    lines += ["", f"Row samples: {rw['syncs']} syncs, {rw['elements']} elements",
              f"P6c: max |Corr(phi_k, phi_0) - prediction| over syncs {rw['max_abs_dev']:.4f}; Corr at syncs "
              + ", ".join(f"{k + 1}: {rw['corr'][k]:.3f}/{rw['pred'][k]:.3f}" for k in sorted({1, 9, 33, 129, rw['syncs'] - 1}) if k < rw['syncs']),
              "P7 rms partial sum / Delta_0 at K = " + ", ".join(
                  f"{m}: RTN {rw['partial_sum_rms']['rtn'][m]:.3f}, SR {rw['partial_sum_rms']['sr'][m]:.3f}, EF {rw['partial_sum_rms']['ef'][m]:.3f}"
                  for m in rw["partial_sum_rms"]["rtn"]),
              f"EF max |sum e| / (Delta_K/2) {rw['ef_bound_max']:.6f}; SR rms / sqrt(K/6) {rw['sr_ratio_last']:.4f}", "",
              "Decision rules (pre-registered):"]
    for name, ok, detail in rules(r):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- self-test
def _synthetic(d, on_grid=False, steps=40, seed=0):
    """Dumps in e2_dump's format for a toy model: continuous Gaussian weights (U holds) or weights on the grid."""
    g = torch.Generator().manual_seed(seed)
    shapes = {"q_proj": (256, 512), "k_proj": (64, 512), "v_proj": (64, 512), "o_proj": (512, 256),
              "gate_proj": (512, 256), "up_proj": (512, 256), "down_proj": (256, 512)}
    W = {}
    for p, s in shapes.items():
        blk = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        if on_grid:
            codes = torch.randint(-60, 61, s, generator=g).float()
            codes[:, 0] = 127.0
            W[f"model.layers.0.{blk}.{p}.weight"] = codes * 1e-4
        else:
            W[f"model.layers.0.{blk}.{p}.weight"] = torch.randn(*s, generator=g) * 0.02
    drift = {n: torch.randn(w.shape, generator=g) for n, w in W.items()}
    full = {1, 2, 3, 5, 9, 17, 33, 34}
    os.makedirs(os.path.join(d, "full")); os.makedirs(os.path.join(d, "rows"))
    index = {n: torch.randperm(w.shape[0], generator=g)[:16].sort().values for n, w in W.items()}
    torch.save(index, os.path.join(d, "rows", "index.pt"))
    eta = 1.3e-5 if not on_grid else 1e-7
    for k in range(1, steps + 1):
        if k in full:
            torch.save({n: w.clone() for n, w in W.items()}, os.path.join(d, "full", f"sync{k:04d}_b00.pt"))
        torch.save({n: w[index[n]].clone() for n, w in W.items()}, os.path.join(d, "rows", f"sync{k:04d}_b00.pt"))
        for n in W:                                 # Adam-like steps: magnitude eta, sign from a drifting gradient
            gk = drift[n] + 1.5 * torch.randn(W[n].shape, generator=g)
            W[n] = W[n] + eta * torch.sign(gk)


def self_test():
    import tempfile

    ok = True
    d = tempfile.mkdtemp(prefix="e2_dyn_")
    _synthetic(d)
    r = analyse(d)
    res = rules(r)
    bad = [x for x in res if not x[1]]
    print(f"self-test, continuous weights (U holds): {len(res) - len(bad)} of {len(res)} rules pass")
    for name, _, detail in bad:
        print(f"  unexpected FAIL {name}: {detail}")
    ok &= not bad
    d2 = tempfile.mkdtemp(prefix="e2_dyn_grid_")
    _synthetic(d2, on_grid=True)
    r2 = analyse(d2)
    p4 = [x for x in rules(r2) if x[0].startswith("P4a'")]
    u1 = [x for x in rules(r2) if x[0].startswith("U1")]
    flagged = all(not x[1] for x in p4) and all(not x[1] for x in u1)
    print(f"self-test, weights on the grid (U fails): P4a' and U1 flagged as FAIL: {flagged}"
          f" (F/P {[x[2] for x in p4][:2]}, near-boundary ratio {u1[0][2]})")
    ok &= flagged
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    r = analyse(a.dumps)
    text = report(r)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1, default=str)


if __name__ == "__main__":
    main()
