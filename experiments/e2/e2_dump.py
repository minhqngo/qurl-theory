"""E2 (P1 experiments report, S5): record the FP32 master weights that VeRL syncs into the INT8 rollout engine.

E2WorkerExtension subclasses S3's QuRLWorkerExtension, which stays unchanged: every synced tensor still goes through
the INT8 refit, and this class only taps the stream on its way in. VeRL syncs the actor's weights before every
rollout, so sync k carries theta_{k-1}: sync 1 is the initial checkpoint, sync 436 the weights after step 435.
The synced weights arrive as the FP32 master weights (S3's refit log), so the dumps are exactly what the rollout
quantizer saw.

Environment (read in the vLLM worker process; set by e2_run.sh):
  QURL_E2_DIR          output directory. Without it the class behaves exactly like QuRLWorkerExtension.
  QURL_E2_FULL_SYNCS   comma-separated syncs at which every synced tensor is saved in FP32 (default FULL_DEFAULT)
  QURL_E2_ROW_LAYERS   decoder layers whose linear weights also give a fixed row sample at every sync (0,6,12,18,23)
  QURL_E2_ROWS         rows per matrix in that sample (16), a fixed random subset of each matrix's output rows
Outputs, under QURL_E2_DIR:
  full/sync{k:04d}_b{b:02d}.pt    {name: FP32 tensor} for every tensor of received bucket b of sync k
  full/sync{k:04d}_live_qkv0.pt   layer 0's live INT8 codes [out, in] and scales after the sync (exactness check)
  rows/index.pt                   {name: LongTensor of sampled row indices}
  rows/sync{k:04d}_b{b:02d}.pt    {name: FP32 rows [R, in]}
and one "[qurl e2]" line per bucket in the console log.
"""
import os
import re
import time
import zlib

import torch

from qurl_verl.int8_rollout import _LINEAR, QuRLWorkerExtension

FULL_DEFAULT = "1,2,3,5,9,17,33,65,66,129,257,258,435,436"
FIRST = "layers.0.self_attn.q_proj.weight"   # every sync carries this tensor exactly once, in its first bucket
_LAYER = re.compile(r"layers\.(\d+)\.")


def _ints(text):
    return {int(s) for s in text.split(",") if s.strip()}


OUT = os.environ.get("QURL_E2_DIR")
FULL = _ints(os.environ.get("QURL_E2_FULL_SYNCS", FULL_DEFAULT))
ROW_LAYERS = _ints(os.environ.get("QURL_E2_ROW_LAYERS", "0,6,12,18,23"))
ROWS = int(os.environ.get("QURL_E2_ROWS", "16"))


def row_index(name: str, n_rows: int) -> torch.Tensor:
    """A fixed, sorted random subset of a matrix's rows, the same in every process and at every sync."""
    g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    return torch.randperm(n_rows, generator=g)[:min(ROWS, n_rows)].sort().values


def _save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # a crash never leaves a half-written file under the final name


class E2WorkerExtension(QuRLWorkerExtension):
    """QuRL's INT8 refit, plus E2's taps on the synced FP32 weights."""

    def _update_weights(self, weights, peft_config, base_sync_done):
        if not OUT:
            return super()._update_weights(weights, peft_config, base_sync_done)
        weights = list(weights)
        if any(name.endswith(FIRST) for name, _ in weights):
            self._e2_sync = getattr(self, "_e2_sync", 0) + 1
            self._e2_bucket = 0
        k = getattr(self, "_e2_sync", 0)
        b = getattr(self, "_e2_bucket", 0)
        self._e2_bucket = b + 1

        # Copy before the refit consumes the bucket: the received tensors may be views into a reused IPC buffer.
        full = {}
        if k in FULL:
            full = {name: t.detach().to("cpu", dtype=torch.float32, copy=True) for name, t in weights}
        rows = {}
        if not hasattr(self, "_e2_index"):
            self._e2_index = {}
        for name, t in weights:
            if _LINEAR.match(name) and int(_LAYER.search(name).group(1)) in ROW_LAYERS:
                idx = self._e2_index.setdefault(name, row_index(name, t.shape[0]))
                rows[name] = t.detach()[idx.to(t.device)].to("cpu", dtype=torch.float32, copy=True)

        result = super()._update_weights(weights, peft_config, base_sync_done)

        t0 = time.time()
        if full:
            os.makedirs(os.path.join(OUT, "full"), exist_ok=True)
            _save(full, os.path.join(OUT, "full", f"sync{k:04d}_b{b:02d}.pt"))
            if any(name.endswith(FIRST) for name in full):
                self._e2_save_live_qkv0(k)
        if rows:
            os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
            index_path = os.path.join(OUT, "rows", "index.pt")
            if not os.path.exists(index_path) or getattr(self, "_e2_index_saved", 0) != len(self._e2_index):
                _save(dict(self._e2_index), index_path)
                self._e2_index_saved = len(self._e2_index)
            _save(rows, os.path.join(OUT, "rows", f"sync{k:04d}_b{b:02d}.pt"))
        print(f"[qurl e2] sync {k} bucket {b}: {len(weights)} tensors; saved {len(full)} full, {len(rows)} row samples"
              f" in {time.time() - t0:.1f} s", flush=True)
        return result

    def _e2_save_live_qkv0(self, k):
        modules = dict(self.model_runner.model.named_modules())
        name = next(n for n in modules if n.endswith("layers.0.self_attn.qkv_proj"))
        layer = modules[name]
        _save({"codes": layer.weight.data.t().cpu().clone(), "scale": layer.weight_scale.data.cpu().clone(),
               "logical_widths": list(layer.logical_widths), "module": name},
              os.path.join(OUT, "full", f"sync{k:04d}_live_qkv0.pt"))
