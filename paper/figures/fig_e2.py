"""Figure 2 of the paper: tracking and persistence on the instrumented RL run (E2), straight from the analysis output.

  python -P paper/figures/fig_e2.py     (needs matplotlib; -P keeps this folder and the cwd off sys.path)

Reads experiments/e2/results/e2_r3_s1/dynamic.json (written by the frozen e2_dynamic.py) and
experiments/e2/results/bf16_update_sim.txt (the exploratory simulation), and writes fig_e2.pdf next to this script:
  (a) Corr(phi_k, phi_0) in bins of the cumulative displacement a = frac(|q_k - q_0|), at every full-dump sync after
      the first (bins with at least 10^4 elements), against the persistence proposition's 1 - 6a(1 - a). The registered
      test compares each bin with its mean prediction, the bin mean of 1 - 6a(1 - a), which depends on where the
      bin's elements sit (early syncs crowd near a = 0). So each bin is drawn at the displacement whose prediction
      equals that mean (on the branch of the bin's centre); its vertical distance from the line is then exactly the
      tested deviation, and the position is within 0.01 of the bin's mean displacement;
  (b) the capture slope sum(dc u) / sum(u^2) of the five one-step pairs, with the registered band [0.95, 1.05], and
      the same statistic for synthetic rows stored in BF16 (two seeds) or in FP32.
Colors: the validated categorical pair blue #2a78d6 / orange #eb6834 of Figure 1; the FP32 control is drawn in muted
ink. Marker shapes repeat the identity, so it is never carried by color alone.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "experiments" / "e2" / "results"
OUT = HERE / "fig_e2.pdf"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BAND = "#222222", "#555555", "#e4e4e4", "#ececec"
A_BINS, MIN_BIN = 20, 10_000
PAIRS = ["1-2", "2-3", "65-66", "257-258", "435-436"]

d = json.load(open(RESULTS / "e2_r3_s1" / "dynamic.json", encoding="utf-8"))
sim = {}
pat = re.compile(r"^(BF16 s1|BF16|FP32)\s+pair\s+(\d+-\d+)\s.*capture\s+([0-9.]+)\s*$")
for line in open(RESULTS / "bf16_update_sim.txt", encoding="utf-8"):
    m = pat.match(line.strip())
    if m:
        label = "FP32" if m.group(1) == "FP32" else "BF16"
        sim.setdefault(label, {}).setdefault(m.group(2), []).append(float(m.group(3)))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#b5b5b5", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.88, wspace=0.30)
for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)

# (a) Persistence: measured correlation by displacement bin, all recorded syncs, against 1 - 6a(1 - a).
ax = axes[0]
xs, ys, devs = [], [], []
for k, s in d["persistence"].items():
    for i, (c, p, n) in enumerate(zip(s["corr"], s["pred"], s["n"])):
        if n >= MIN_BIN:
            half = max(12 + 24 * p, 0.0) ** 0.5 / 12          # 1 - 6a(1 - a) = p  <=>  a = 1/2 -+ half
            xs.append(0.5 - half if (i + 0.5) / A_BINS < 0.5 else 0.5 + half)
            ys.append(c)
            devs.append(abs(c - p))
grid_a = [i / 400 for i in range(0, 281)]
ax.plot(grid_a, [1 - 6 * a * (1 - a) for a in grid_a], color=INK, lw=1.0, zorder=2,
        label=r"prediction $1-6a(1-a)$")
ax.plot(xs, ys, ls="none", marker="o", ms=4.2, mfc=BLUE, mec="white", mew=0.6, zorder=3,
        label=f"measured, {len(d['persistence'])} syncs")
ax.axhline(0, color="#b5b5b5", lw=0.6, zorder=1)
ax.set_xlim(0, 0.7)
ax.set_ylim(-0.6, 1.05)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_xlabel(r"Cumulative displacement $a=\mathrm{frac}(|q_k-q_0|)$")
ax.set_ylabel(r"$\mathrm{Corr}(\varphi_k,\varphi_0)$")
ax.set_title("(a) Persistence of the rounding error", loc="left")
ax.legend(loc="upper right", frameon=False, handlelength=1.6)

# (b) Capture slope per one-step pair, with the registered band, and the simulation at the first three pairs.
ax = axes[1]
ax.axhspan(0.95, 1.05, color=BAND, zorder=0, lw=0)
ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(len(PAIRS) - 0.55, 0.953, "registered band", ha="right", va="bottom", fontsize=7.5, color=MUTED)
run = [d["pairs"][p]["capture_slope"] for p in PAIRS]
ax.plot(range(len(PAIRS)), run, ls="none", marker="o", ms=5.5, mfc=BLUE, mec="white", mew=0.7, zorder=4,
        label="E2 run")
for label, dx, marker, color in (("BF16", -0.2, "D", ORANGE), ("FP32", 0.2, "s", MUTED)):
    px, py = [], []
    for j, p in enumerate(PAIRS):
        for v in sim.get(label, {}).get(p, []):
            px.append(j + dx)
            py.append(v)
    name = "simulation, BF16 storage" if label == "BF16" else "simulation, FP32 storage"
    ax.plot(px, py, ls="none", marker=marker, ms=4.2, mfc="white", mec=color, mew=1.1, zorder=3, label=name)
ax.set_xticks(range(len(PAIRS)))
ax.set_xticklabels(["${}{{-}}{}$".format(*p.split("-")) for p in PAIRS])
ax.set_xlim(-0.5, len(PAIRS) - 0.5)
ax.set_ylim(0.90, 1.03)
ax.set_yticks([0.90, 0.95, 1.00])
ax.set_xlabel("One-step pair (syncs)")
ax.set_ylabel("Capture slope")
ax.set_title("(b) Capture of one-step updates", loc="left")
ax.legend(loc="lower right", frameon=False, handlelength=1.2, bbox_to_anchor=(1.0, 0.04))

fig.savefig(OUT, metadata={"CreationDate": None, "Creator": None, "Producer": None})
print(f"wrote {OUT.name}: {len(xs)} persistence points from {len(d['persistence'])} syncs, largest deviation"
      f" {max(devs):.4f}; capture {[round(v, 3) for v in run]}; simulation {sim}")
