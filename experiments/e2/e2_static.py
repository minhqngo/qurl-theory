"""E2, static part (P1 experiments report, S5): the rollout quantizer's grid on real checkpoints, before any training.

  python e2_static.py <model id or local path> [...] [--eta 1e-6] [--out report.txt] [--json data.json]
  python e2_static.py --self-test

For every decoder linear weight (q, k, v, o, gate, up, down) and every row, with the rollout quantizer's own
arithmetic (qurl_verl.int8_rollout.quantize_rows: scale Delta = absmax * fl32(1/127), code = round(w / Delta)):
  phase      phi = w/Delta - round(w/Delta) in [-1/2, 1/2], the row's absmax element (|code| 127) excluded;
  sigma/Delta = rms(row) / Delta (Lemma 1's ratio; U is assured for sigma/Delta >= 1);
  nu         = eta / Delta, the update-to-grid ratio of an Adam step of size eta (the link to 7B).
Reported per projection type: phase uniformity (largest deviation of the 40 bin densities from 1, KS distance on
1000 bins, the share within 0.05 of a rounding boundary against 0.10 under U, exact ties), the same split by code
size (|c| <= 15, 16-63, 64-126: BF16 storage puts large codes on a coarse lattice), sigma/Delta and max/rms
quantiles, the share of rows with sigma/Delta < 1, and nu quantiles; per decoder layer the median nu.
Pre-registered criteria are in the P1 experiments report (E2 design); this script only measures.
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
NB, NKS = 40, 1000
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NAME = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
CODE_BINS = ((0, 15), (16, 63), (64, 126))
QS = (0.01, 0.1, 0.5, 0.9, 0.99)


class Acc:
    """Running phase histograms and per-row statistics for one group of matrices."""

    def __init__(self):
        self.h = torch.zeros(NB, dtype=torch.float64)
        self.hks = torch.zeros(NKS, dtype=torch.float64)
        self.hc = {cb: torch.zeros(NB, dtype=torch.float64) for cb in CODE_BINS}
        self.ties = 0
        self.rows_sd, self.rows_mr, self.rows_nu = [], [], []

    def add(self, w, eta):
        w = w.float()
        absmax = w.abs().amax(dim=1, keepdim=True)
        scale = absmax * (1.0 / QMAX)                 # the rollout quantizer's scale, as a multiplication
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = w / scale
        c = torch.round(q)
        phi = (q - c).double()
        keep = c.abs() < QMAX                          # the absmax element itself sits on the grid by construction
        p = phi[keep]
        self.ties += int((p.abs() == 0.5).sum())
        self.h += torch.histc(p, bins=NB, min=-0.5, max=0.5)
        self.hks += torch.histc(p, bins=NKS, min=-0.5, max=0.5)
        ca = c.abs()[keep]
        for lo, hi in CODE_BINS:
            sel = (ca >= lo) & (ca <= hi)
            self.hc[(lo, hi)] += torch.histc(p[sel], bins=NB, min=-0.5, max=0.5)
        rms = w.double().pow(2).mean(dim=1).sqrt()
        sc = scale.squeeze(1).double()
        self.rows_sd.append(rms / sc)
        self.rows_mr.append(absmax.squeeze(1).double() / rms.clamp_min(1e-30))
        self.rows_nu.append(eta / sc)

    @staticmethod
    def uniformity(h):
        n = float(h.sum())
        if n == 0:
            return {"n": 0}
        dens = h * len(h) / n
        return {"n": int(n), "max_bin_dev": float((dens - 1).abs().max())}

    def summary(self):
        n = float(self.h.sum())
        cdf = torch.cumsum(self.hks, 0) / n
        grid = torch.arange(1, NKS + 1, dtype=torch.float64) / NKS
        edge = int(round(0.05 * NB))                   # bins within 0.05 of either boundary
        boundary = float((self.h[:edge].sum() + self.h[-edge:].sum()) / n)
        sd, mr, nu = (torch.cat(x) for x in (self.rows_sd, self.rows_mr, self.rows_nu))
        quant = lambda t: [float(v) for v in torch.quantile(t.float(), torch.tensor(QS))]
        return {
            "elements": int(n), "rows": int(sd.numel()),
            "max_bin_dev": self.uniformity(self.h)["max_bin_dev"],
            "ks": float((cdf - grid).abs().max()),
            "boundary_share": boundary, "boundary_ratio": boundary / 0.10,
            "ties": self.ties / n,
            "by_code": {f"{lo}-{hi}": self.uniformity(h) for (lo, hi), h in self.hc.items()},
            "sigma_over_delta_q": quant(sd), "rows_sigma_below_delta": float((sd < 1).double().mean()),
            "max_over_rms_q": quant(mr), "nu_q": quant(nu),
        }


def linear_tensors(path):
    """Yield (layer, proj, FP32 tensor) for the decoder linear weights of a safetensors checkpoint."""
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"no .safetensors files in {path}")
    for f in files:
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                m = NAME.search(key)
                if m:
                    yield int(m.group(1)), m.group(2), st.get_tensor(key).float()


def resolve(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])


def analyse(tensors, eta):
    by_proj = {p: Acc() for p in PROJ}
    allacc = Acc()
    per_layer = {}
    for layer, proj, w in tensors:
        by_proj[proj].add(w, eta)
        allacc.add(w, eta)
        per_layer.setdefault(layer, []).append(eta / ((w.abs().amax(dim=1) * (1.0 / QMAX)).double().clamp_min(1e-30)))
    out = {"all": allacc.summary(), "by_proj": {p: a.summary() for p, a in by_proj.items() if a.rows_sd}}
    out["layer_median_nu"] = {str(l): float(torch.cat(v).median()) for l, v in sorted(per_layer.items())}
    return out


def report(name, r, eta):
    lines = [f"== {name} (eta = {eta:g})"]
    hdr = (f"{'proj':10s} {'rows':>8s} {'max bin dev':>11s} {'KS':>7s} {'boundary/U':>10s} {'ties':>9s}"
           f" {'dev |c|<=15':>11s} {'16-63':>7s} {'64-126':>7s} {'sigma/D 1%':>10s} {'50%':>6s} {'99%':>6s}"
           f" {'rows s/D<1':>10s} {'max/rms 50%':>11s} {'99%':>6s} {'nu 10%':>9s} {'50%':>9s} {'90%':>9s}")
    lines.append(hdr)
    for p, s in list(r["by_proj"].items()) + [("all", r["all"])]:
        bc = s["by_code"]
        dev = lambda key: f"{bc[key]['max_bin_dev']:.3f}" if bc[key]["n"] else "-"
        lines.append(
            f"{p:10s} {s['rows']:8d} {s['max_bin_dev']:11.4f} {s['ks']:7.4f} {s['boundary_ratio']:10.4f}"
            f" {s['ties']:9.2e} {dev('0-15'):>11s} {dev('16-63'):>7s} {dev('64-126'):>7s}"
            f" {s['sigma_over_delta_q'][0]:10.2f} {s['sigma_over_delta_q'][2]:6.2f} {s['sigma_over_delta_q'][4]:6.2f}"
            f" {s['rows_sigma_below_delta']:10.2e} {s['max_over_rms_q'][2]:11.1f} {s['max_over_rms_q'][4]:6.1f}"
            f" {s['nu_q'][1]:9.2e} {s['nu_q'][2]:9.2e} {s['nu_q'][3]:9.2e}")
    lm = r["layer_median_nu"]
    lines.append("median nu by decoder layer: " + " ".join(f"{l}:{v:.2e}" for l, v in lm.items()))
    return "\n".join(lines)


def self_test():
    """Known answers: Gaussian rows satisfy U; rows placed on the grid violate it; nu = eta / Delta exactly."""
    g = torch.Generator().manual_seed(0)
    gauss = [(0, "q_proj", torch.randn(256, 512, generator=g) * 0.02)]
    r = analyse(iter(gauss), 1e-6)["all"]
    ok1 = r["max_bin_dev"] < 0.05 and abs(r["boundary_ratio"] - 1) < 0.05 and r["sigma_over_delta_q"][2] > 20
    w = torch.randint(-60, 61, (256, 512), generator=g).float()
    w[:, 0] = 127.0                                    # absmax 127, so Delta = 1 and every element sits on the grid
    r2 = analyse(iter([(0, "q_proj", w * 0.01)]), 1e-6)["all"]
    ok2 = r2["max_bin_dev"] > 10                       # all mass in the central bins: U fails as it should
    delta = 1.27 * (1.0 / QMAX) * 0.01 / 0.01          # absmax 1.27 * 0.01 / 127 = 1e-4 per row
    ok3 = abs(r2["nu_q"][2] - 1e-6 / (1.27 * (1.0 / QMAX))) < 1e-9
    print(f"self-test: Gaussian rows uniform {ok1} (max dev {r['max_bin_dev']:.4f}, boundary ratio {r['boundary_ratio']:.4f},"
          f" median sigma/Delta {r['sigma_over_delta_q'][2]:.1f}); on-grid rows flagged {ok2}"
          f" (max dev {r2['max_bin_dev']:.1f}); nu = eta/Delta {ok3}")
    return ok1 and ok2 and ok3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--eta", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    texts, data = [], {}
    for m in a.models:
        r = analyse(linear_tensors(resolve(m)), a.eta)
        data[m] = r
        texts.append(report(m, r, a.eta))
        print(texts[-1], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write("E2 static analysis (e2_static.py)\n" + "\n\n".join(texts) + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(data, f, indent=1)


if __name__ == "__main__":
    main()
