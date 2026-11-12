# Usage: python int8_flip_rate.py results/<tag>/int8_dump   (needs torch)
# First real-stack look at Proposition 4 (flip probability |d| / Delta) from the H100 R3 pilot's layer-0 QKV dumps.
# Per row: Delta = scale at sync 1; |d| is estimated by the move of the row's absmax element between syncs,
# 127 * |scale2 - scale1| (scale = absmax / 127); the observed flip rate is the share of the row's codes that changed.
# Caveats: one sample of |d| per row (the absmax element), and part of the flips come from the grid itself moving
# (every code boundary shifts when the absmax changes).
import sys

import torch

d = sys.argv[1]
s1 = torch.load(f"{d}/sync1_qkv_layer0.pt")
s2 = torch.load(f"{d}/sync2_qkv_layer0.pt")
c1, c2 = s1["codes"].int(), s2["codes"].int()
sc1, sc2 = s1["scale"].double().squeeze(1), s2["scale"].double().squeeze(1)
obs = (c1 != c2).double().mean(dim=1)
dmove = 127.0 * (sc2 - sc1).abs()
pred = (dmove / sc1).clamp(max=1.0)
jit = pred * c1.abs().double().mean(dim=1) / 127.0       # extra flips from the moving grid, ~ |q| * relative change
bounds = [0]
for w in s1["logical_widths"]:
    bounds.append(bounds[-1] + w)
print("shard  rows  |d| median   Delta median  observed flips  |d|/Delta   +grid jitter  corr(obs, pred)")
for name, a, b in zip("qkv", bounds[:-1], bounds[1:]):
    o, p, j = obs[a:b], pred[a:b], jit[a:b]
    r = torch.corrcoef(torch.stack([o, p]))[0, 1].item()
    print(f"{name}_proj {b - a:5d}  {dmove[a:b].median().item():.3e}   {sc1[a:b].median().item():.3e}    "
          f"{100 * o.mean().item():.4f}%       {100 * p.mean().item():.4f}%   {100 * (p + j).mean().item():.4f}%     {r:.2f}")
print(f"all rows: observed {100 * obs.mean().item():.4f}% vs predicted {100 * pred.mean().item():.4f}% "
      f"(+ grid jitter {100 * (pred + jit).mean().item():.4f}%)")
