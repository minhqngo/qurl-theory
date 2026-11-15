"""Check an E2 dump directory (after the smoke test, and before analysing the E2 run).
  python e2_check_dumps.py <dumps dir>
Checks, for every full-dump sync: its buckets merge into one state with the same tensor names as sync 1's, every
decoder linear weight present in FP32; layer 0's live INT8 codes and scales equal quantize_rows of the dumped Q/K/V
weights (so the dumps are what the rollout quantizer saw); and the row samples of that sync equal the corresponding
rows of the full dump. Reports the syncs found, the size on disk and how far the weights moved from sync 1."""
import glob
import os
import re
import sys

import torch

QMAX = 127.0
LINEAR = re.compile(r"layers\.\d+\.(?:self_attn|mlp)\.(?:q|k|v|o|gate|up|down)_proj\.weight$")


def quantize_rows(w):          # the rollout quantizer (qurl_verl.int8_rollout.quantize_rows), without importing VeRL
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(w / scale).clamp_(-QMAX, QMAX).to(torch.int8), scale


def load_full(d, k):
    state = {}
    for f in sorted(glob.glob(os.path.join(d, "full", f"sync{k:04d}_b*.pt"))):
        state.update(torch.load(f, map_location="cpu"))
    return state


def main(d):
    syncs = sorted({int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(d, "full", "sync*_b*.pt"))})
    row_syncs = sorted({int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(d, "rows", "sync*_b*.pt"))})
    if not syncs or not row_syncs or not os.path.exists(os.path.join(d, "rows", "index.pt")):
        print(f"no complete dumps yet in {d}: full dumps at {len(syncs)} syncs, row samples at {len(row_syncs)}."
              " Is the run still starting (the first sync comes after VeRL's start-up), or did it fail? Look at the"
              " run's train.log in ~/qurl/s3/results/<tag>/.")
        return False
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    print(f"full dumps at syncs {syncs}; row samples at {len(row_syncs)} syncs ({row_syncs[0]}..{row_syncs[-1]});"
          f" {size / 1e9:.1f} GB")
    index = torch.load(os.path.join(d, "rows", "index.pt"))
    ok, ref, names = True, None, None
    for k in syncs:
        st = load_full(d, k)
        lin = sorted(n for n in st if LINEAR.search(n))
        if names is None:
            names, ref = sorted(st), {n: st[n] for n in lin}
        same = sorted(st) == names
        fp32 = all(st[n].dtype == torch.float32 for n in lin)
        live = torch.load(os.path.join(d, "full", f"sync{k:04d}_live_qkv0.pt"))
        w = torch.cat([st[n] for n in lin if ".layers.0.self_attn." in n and n.split(".")[-2] in ("q_proj",)] +
                      [st[n] for n in lin if ".layers.0.self_attn." in n and n.split(".")[-2] in ("k_proj",)] +
                      [st[n] for n in lin if ".layers.0.self_attn." in n and n.split(".")[-2] in ("v_proj",)])
        codes, scale = quantize_rows(w)
        exact = torch.equal(live["codes"], codes) and torch.equal(live["scale"], scale)
        rows_ok = None
        rf = sorted(glob.glob(os.path.join(d, "rows", f"sync{k:04d}_b*.pt")))
        if rf:
            rows = {}
            for f in rf:
                rows.update(torch.load(f, map_location="cpu"))
            rows_ok = all(torch.equal(rows[n], st[n][index[n]]) for n in rows)
        moved = max(float((st[n] - ref[n]).abs().max()) for n in lin)
        line_ok = same and fp32 and exact and rows_ok is not False
        ok &= line_ok
        print(f"sync {k:4d}: {len(st)} tensors ({len(lin)} linear, FP32 {fp32}); same names as the first {same};"
              f" live layer-0 INT8 codes = quantize_rows(dump) {exact}; row samples match {rows_ok};"
              f" max |theta - theta_first| {moved:.3e}  {'OK' if line_ok else 'PROBLEM'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
