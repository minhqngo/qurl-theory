"""Figure 1 of the paper: the runs of the UAQ experiment (S3), straight from the run logs.

  python -P paper/figures/fig_s3_curves.py     (needs matplotlib; -P keeps this folder and the cwd off sys.path)

Reads experiments/s3/results/uaq_R{1..6}_s{1..5}/metrics.jsonl (whichever runs exist) and writes
fig_s3_curves.pdf next to this script:
  (a) greedy GSM8K test accuracy at each validation (every 29 steps);
  (b) the actor's entropy, median over each epoch of 29 steps, at the epoch's middle step.
BF16 rollouts (R1, R2) are blue, INT8 rollouts (R3-R6) orange; a run whose validation at step 29 is below 10%
(an early stall) is dashed. Colors: the validated categorical pair blue #2a78d6 / orange #eb6834.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "experiments" / "s3" / "results"
OUT = HERE / "fig_s3_curves.pdf"
VAL = "val-core/openai/gsm8k/acc/mean@1"
EPOCH, EPOCHS = 29, 15
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#222222", "#555555", "#e4e4e4"


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs.setdefault(r["step"], {}).update(r["data"])
    return recs


runs = {}
for arm in range(1, 7):
    for seed in range(1, 6):
        p = RESULTS / f"uaq_R{arm}_s{seed}" / "metrics.jsonl"
        if p.is_file():
            runs[arm, seed] = load(p)

steps_val = [e * EPOCH for e in range(1, EPOCHS + 1)]
steps_ent = [((e - 1) * EPOCH + 1 + e * EPOCH) // 2 for e in range(1, EPOCHS + 1)]


def val_curve(r):
    return [100 * r[s][VAL] for s in steps_val]


def ent_curve(r):
    return [st.median(r[k]["actor/entropy"] for k in range((e - 1) * EPOCH + 1, e * EPOCH + 1))
            for e in range(1, EPOCHS + 1)]


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#b5b5b5", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75))
fig.subplots_adjust(left=0.075, right=0.94, bottom=0.16, top=0.80, wspace=0.32)


def style(r):
    arm, _ = r
    if arm <= 2:
        return dict(color=BLUE, lw=1.0, ls="-", zorder=3)
    stalled = 100 * runs[r][29][VAL] < 10
    return dict(color=ORANGE, lw=1.0, ls=(0, (3.5, 2)) if stalled else "-", alpha=0.9, zorder=2)


# INT8 runs first, so the BF16 runs stay visible on top.
order = sorted(runs, key=lambda r: (r[0] <= 2, r))
for ax, xs, curve in ((axes[0], steps_val, val_curve), (axes[1], steps_ent, ent_curve)):
    for r in order:
        ax.plot(xs, curve(runs[r]), **style(r))
    ax.set_xlim(0, 440)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Training step")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)

axes[0].set_ylim(0, 60)
axes[0].set_yticks(range(0, 61, 10))
axes[0].set_ylabel("GSM8K test accuracy (%)")
axes[0].set_title("(a) Greedy validation accuracy", loc="left")
axes[1].set_ylim(0, 0.8)
axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axes[1].set_ylabel("Entropy (median per epoch)")
axes[1].set_title("(b) Policy entropy", loc="left")

# Direct labels at the right end, in text ink (identity is not carried by color alone).
for ax, xs, curve in ((axes[0], steps_val, val_curve), (axes[1], steps_ent, ent_curve)):
    bf = st.median(curve(runs[r])[-1] for r in runs if r[0] <= 2)
    q8 = st.median(curve(runs[r])[-1] for r in runs if r[0] >= 3 and 100 * runs[r][29][VAL] >= 10)
    for y, text in ((bf, "BF16"), (q8, "INT8")):
        ax.text(xs[-1] + 10, y, text, va="center", ha="left", fontsize=8, color=INK, clip_on=False)

handles = [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=ORANGE, lw=1.4),
           Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (3.5, 2)))]
labels = ["BF16 rollouts (R1, R2)", "INT8 rollouts (R3 to R6)", "INT8, early stall"]
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=2.4, columnspacing=2.0)

fig.savefig(OUT, metadata={"CreationDate": None, "Creator": None, "Producer": None})
print(f"wrote {OUT.name}: {len(runs)} runs, {sum(1 for r in runs if r[0] >= 3 and 100 * runs[r][29][VAL] < 10)} with an early stall")
