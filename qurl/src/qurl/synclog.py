"""The INT8 refit's per-sync log: code flips, row-scale changes, quantization error and ties, one JSON line per sync.

For every decoder layer and each of the 7 linear projections (PROJS), a record holds
  flip           the share of INT8 codes that changed since the previous sync
  scale_changed  the share of rows whose scale changed since the previous sync
  qerr           the relative quantization error ||W - Q(W)|| / ||W||
  ties           the share of codes that were ties at the attempt's first sync (only for attempts that start at
                 step 0, whose first sync carries the run's starting weights)
  flip_ties      the share of those ties that flipped since the previous sync
(flip, scale_changed and flip_ties are None at the attempt's first sync, which has no history), and for the q/k/v
weights of three layers (first, middle, last) under "pred": Proposition 4's prediction of the flip share,
mean(min(1, |W_k - W_(k-1)| / Delta)) with Delta the new grid step, next to the flip share recomputed from the previous
sync's weights ("observed_all_rows"). The latter must equal "flip" up to FP32 rounding of its mean: a check of the
history.

The comparison is against the logging replica's own copy of all previous codes, kept in pinned CPU memory (6.5 GB for a
7B model, about a second of transfers per sync), and of all row scales on the device. Neither vLLM's live weights nor a
row subset can serve as the history: VeRL puts vLLM to sleep (level 2) between rollouts, which discards its weights,
and a strided row subset aliases with the attention heads' dimension structure, while a random one is noisy for small
tensors (k_proj). The ties' indices stay in CPU memory (about 0.1 GB for 7B).

The log is device-agnostic: on CUDA it runs exactly as in the experiments (pinned host memory, a stream
synchronization before the history is overwritten); on the CPU it runs the same arithmetic without them.
"""
from __future__ import annotations

import json
import os
import time

import torch

from .quant import quantize_rows, tie_indices

PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
"""The 7 linear weights of a decoder layer, in the order of the statistics' second axis."""

QKV = ("q_proj", "k_proj", "v_proj")

# Columns of the per-sync statistics [layer, projection, column], as pairs (count, out of):
_FLIPS, _CODES, _SCALES_CHANGED, _ROWS, _SQ_ERR, _SQ_NORM, _TIE_FLIPS, _TIES = range(8)


def claim(path: str) -> bool:
    """True for exactly one process: the first to create `path` (its pid is written into it)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def tracked_layers(layers: int) -> list[int]:
    """The layers whose q/k/v weights carry Proposition 4's prediction: the first, the middle and the last."""
    return sorted({0, layers // 2, layers - 1})


class SyncLog:
    """One replica's per-sync log (see the module docstring), appended to `path` as JSON lines.

    Use: start_sync(device) when a sync begins, add(...) for every re-quantized weight, write(sync) when the sync is
    complete. `attempt` and `start_step` are recorded with every sync; ties are only detected when start_step is 0.
    """

    def __init__(self, path: str, layers: int, *, attempt: str = "0", start_step: int = 0):
        self.path = path
        self.layers = layers
        self.attempt = attempt
        self.start_step = start_step
        self.tracked = tracked_layers(layers)
        self.ties: dict[tuple[int, str], torch.Tensor] = {}    # flat indices of the ties (CPU), start_step 0 only
        self.codes_per_layer: dict[str, int] = {}
        self._prev: dict[tuple[int, str], torch.Tensor] = {}        # previous FP32 weights, tracked q/k/v only
        self._hist_codes: dict[tuple[int, str], torch.Tensor] = {}  # previous codes (pinned CPU memory on CUDA)
        self._hist_scale: dict[tuple[int, str], torch.Tensor] = {}  # previous row scales, all rows
        self._stats: torch.Tensor | None = None
        self._pred: dict[tuple[int, str], float] = {}
        self._obs: dict[tuple[int, str], float] = {}

    def start_sync(self, device: torch.device | str | None) -> None:
        """Reset the per-sync statistics, on `device` (where the INT8 layers live)."""
        # [layer, projection, (changed codes, codes, changed scales, rows, sq. error, sq. norm, changed codes among
        #  the ties, ties)]
        self._stats = torch.zeros(self.layers, len(PROJS), 8, dtype=torch.float64, device=device)
        self._pred = {}
        self._obs = {}

    @torch.no_grad()
    def add(self, layer: int, proj: str, w: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> None:
        """Accumulate one tensor's flips and scale changes (against this log's own history) and its error.

        `w` is the synced weight [rows, in] (any float dtype), `codes` and `scale` its quantize_rows(w).
        """
        w = w.float()
        key = (layer, proj)
        st = self._stats[layer, PROJS.index(proj)]
        self.codes_per_layer[proj] = codes.numel()
        prev_codes, prev_scale = self._hist_codes.get(key), self._hist_scale.get(key)
        if prev_codes is not None:   # the previous sync's codes
            changed = prev_codes.to(codes.device, non_blocking=True) != codes
            st[_FLIPS] += changed.sum()
            st[_CODES] += codes.numel()
            st[_SCALES_CHANGED] += (prev_scale != scale).sum()
            st[_ROWS] += scale.numel()
            ties = self.ties.get(key)
            if ties is not None:
                st[_TIE_FLIPS] += changed.view(-1)[ties.to(codes.device)].sum()
                st[_TIES] += ties.numel()
            del changed
        else:
            prev_codes = torch.empty(codes.shape, dtype=torch.int8, pin_memory=codes.is_cuda)
            self._hist_codes[key] = prev_codes
            if self.start_step == 0:   # the run's starting weights: e.g. a BF16 checkpoint, or its UAQ rescaling
                self.ties[key] = tie_indices(w, scale).cpu()
        if codes.is_cuda:
            torch.cuda.current_stream(codes.device).synchronize()   # the comparison above has read the old codes
        prev_codes.copy_(codes)
        self._hist_scale[key] = scale.clone()
        st[_SQ_ERR] += (w - codes.float() * scale).double().pow(2).sum()
        st[_SQ_NORM] += w.double().pow(2).sum()
        if layer in self.tracked and proj in QKV:
            prev = self._prev.get(key)
            if prev is not None:
                # Proposition 4: flip probability min(1, |d| / Delta), Delta the new grid step; and the observed flip
                # share on the same (all) rows, the previous codes recomputed from the previous weights
                self._pred[key] = float(torch.clamp((w - prev).abs() / scale, max=1.0).mean())
                self._obs[key] = float((quantize_rows(prev)[0] != codes).float().mean())
            self._prev[key] = w.clone()

    def record(self, sync: int) -> dict:
        """The JSON record of the current sync (all its weights added)."""
        st = self._stats.cpu()
        n = self.layers

        def col(j, k):   # count / out-of, or None where there is nothing to count (no history yet; no ties)
            return [float(st[i, j, k] / st[i, j, k + 1]) if st[i, j, k + 1] > 0 else None for i in range(n)]

        return {"attempt": self.attempt, "start_step": self.start_step, "sync": sync, "time": time.time(),
                "layers": n,
                "codes_per_layer": dict(self.codes_per_layer),
                "flip": {p: col(j, _FLIPS) for j, p in enumerate(PROJS)},
                "scale_changed": {p: col(j, _SCALES_CHANGED) for j, p in enumerate(PROJS)},
                "qerr": {p: [float((st[i, j, _SQ_ERR] / st[i, j, _SQ_NORM]).sqrt()) for i in range(n)]
                         for j, p in enumerate(PROJS)},
                "ties": {p: [len(self.ties[(i, p)]) / self.codes_per_layer[p] if (i, p) in self.ties else None
                             for i in range(n)] for p in PROJS},
                "flip_ties": {p: col(j, _TIE_FLIPS) for j, p in enumerate(PROJS)},
                "pred": {"layers": self.tracked,
                         **{p: [self._pred.get((i, p)) for i in self.tracked] for p in QKV},
                         "observed_all_rows": {p: [self._obs.get((i, p)) for i in self.tracked] for p in QKV}}}

    def write(self, sync: int) -> dict:
        """Append the current sync's record to the log file; returns it."""
        rec = self.record(sync)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
