"""Writes the data of the S3 training-curves figure (fig_s3_curves.tex) from the run logs.

  python -I paper/figures/s3_curves.py

Reads experiments/s3/results/uaq_R{1..6}_s{1,2,3}/metrics.jsonl and writes, next to this script:
  s3_val.dat  greedy GSM8K test accuracy (%) at each validation step (29, 58, ..., 435)
  s3_ent.dat  the actor's entropy, median over each epoch of 29 steps, at the epoch's middle step
One column per run (R1s1 ... R6s3).
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "experiments" / "s3" / "results"
VAL = "val-core/openai/gsm8k/acc/mean@1"
RUNS = [(a, s) for a in range(1, 7) for s in (1, 2, 3)]
EPOCH = 29
EPOCHS = 15


def load(arm, seed):
    recs = {}
    with open(RESULTS / f"uaq_R{arm}_s{seed}" / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs.setdefault(r["step"], {}).update(r["data"])
    return recs


def write(name, rows):
    header = "step " + " ".join(f"R{a}s{s}" for a, s in RUNS)
    with open(HERE / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(row) + "\n")


runs = {r: load(*r) for r in RUNS}

val_rows = []
for e in range(1, EPOCHS + 1):
    step = e * EPOCH
    val_rows.append([str(step)] + [f"{100 * runs[r][step][VAL]:.2f}" for r in RUNS])
write("s3_val.dat", val_rows)

ent_rows = []
for e in range(1, EPOCHS + 1):
    lo, hi = (e - 1) * EPOCH + 1, e * EPOCH
    mid = (lo + hi) // 2
    cells = []
    for r in RUNS:
        cells.append(f"{st.median(runs[r][k]['actor/entropy'] for k in range(lo, hi + 1)):.4f}")
    ent_rows.append([str(mid)] + cells)
write("s3_ent.dat", ent_rows)
print(f"wrote s3_val.dat and s3_ent.dat for {len(RUNS)} runs")
