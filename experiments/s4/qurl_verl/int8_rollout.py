"""QuRL's INT8 W8A8 rollout for VeRL v0.9.1 + vLLM 0.24, re-quantized at every weight sync, on one or more GPUs (S4).

S4 version of experiments/s3/qurl_verl/int8_rollout.py. The quantizer and the refit are unchanged (S3 verified them:
codes bit-identical to an independent quantization after sync 1). What is new:
  - Several rollout replicas. With tensor_model_parallel_size=1, VeRL starts one vLLM replica per GPU and sends each
    the full weights (every FSDP rank all-gathers them), so the refit runs unchanged in every replica. Tensor
    parallelism is refused: a row-parallel layer (o_proj, down_proj) would hold a slice of every row, and its per-row
    scale needs the whole row.
  - A completeness check per replica: every sync must re-quantize all 7 linear weights of every layer, else it raises.
  - Optional per-sync logging for E2/E3 (QURL_E3_PATH): for every layer and projection, the share of INT8 codes that
    changed since the previous sync (code flips), the share of rows whose scale changed, and the relative quantization
    error ||W - Q(W)|| / ||W||. For the q/k/v weights of three layers (first, middle, last) it also logs Proposition
    4's prediction of the flip share, mean(min(1, |W_k - W_(k-1)| / Delta)), from the previous sync's weights.
    Exactly one replica writes (the first to create the claim file), so the log is not repeated per GPU.
    The comparison uses the logging replica's own copy of the previous sync's codes, all of them, in pinned CPU
    memory (6.5 GB for 7B; about a second of transfers per sync), and of all row scales on the GPU. For the tracked
    q/k/v tensors the flip share is also recomputed from the previous weights ("observed_all_rows"); it must equal the
    logged flip share exactly, which checks the history.
    vLLM's live weights cannot serve as the history: VeRL puts vLLM to sleep (level 2) between rollouts, which
    discards them (the laptop smoke test read back zeros and counted 98% flips for unchanged weights). A row subset is
    not used: a strided one aliases with the attention heads' dimension structure (on the laptop, every 16th row of
    q_proj, the same 4 of the 64 dimensions of every head, flipped 2-4x more often than the whole tensor), and a random
    one is noisy for small tensors (k_proj).
  - The test dump (QURL_INT8_DUMP_DIR) is written by one replica only, for the same reason.

vLLM side. The rollout engine is launched as an INT8 W8A8 compressed-tensors model, with the same quantizer as S1's
llm-compressor checkpoints (int8_w8a8_config.json: channel-wise symmetric weights, dynamic per-token symmetric
activations, lm_head kept in BF16). VeRL's dummy load format means every linear layer starts with garbage codes, which
the first weight sync overwrites. Hydra overrides (run.sh adds them for the INT8 arms):
  +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=compressed-tensors
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=<{quantization_config: ..., max_position_embeddings: ...}>
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=qurl_verl.int8_rollout.QuRLWorkerExtension

Weight sync. In vLLM 0.24, a CompressedTensorsW8A8Int8 layer holds an int8 `weight` and a float32 `weight_scale`
[out, 1]; after process_weights_after_loading both the CUTLASS (H100) and the Triton (fallback) kernel keep `weight`
as a transposed [in, out] view of the [out, in] codes. _update_weights quantizes each synced q/k/v/o/gate/up/down
weight row by row (scale = row absmax / 127, codes = round(W / scale) clamped to +-127: the quantizer of Theorem 1
and S2) and writes codes and scales into that shard's rows of the live layer. Per-row scales make the fused shards
(q/k/v in qkv_proj, gate/up in gate_up_proj) independent, so each tensor can be handled as it arrives. Everything
else (embeddings, norms, biases) goes through VeRL's ordinary load path.
"""
import json
import os
import re
import time

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

QMAX = 127.0
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "int8_w8a8_config.json")
_LINEAR = re.compile(r"^(?P<prefix>.*layers\.(?P<layer>\d+)\.)(?P<block>self_attn|mlp)\."
                     r"(?P<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
_FUSED = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
          "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
          "o_proj": ("o_proj", None), "down_proj": ("down_proj", None)}
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

E3_PATH = os.environ.get("QURL_E3_PATH")          # per-sync flip / scale / error log (jsonl), written by one replica
DUMP_DIR = os.environ.get("QURL_INT8_DUMP_DIR")   # test hook: layer 0's fused QKV codes at the first syncs
ATTEMPT = os.environ.get("QURL_RUN_ATTEMPT", "0")  # set per launch by run.sh, so a resumed run claims afresh
START_STEP = int(os.environ.get("QURL_START_STEP", "0"))   # run.sh: 0, or the checkpoint step a resumed attempt loads
DUMP_MODULE_SUFFIX = "layers.0.self_attn.qkv_proj"
DUMP_SYNCS = 2


def quantization_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


@torch.no_grad()
def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row absmax INT8: returns int8 codes [rows, in] and float32 scales [rows, 1].

    The scale is absmax * fl32(1/127), written as a multiplication on purpose: PyTorch divides a CUDA tensor by a
    Python scalar through its reciprocal but a CPU tensor by true division, and the two differ by one ulp in ~5% of
    rows. Either way the quantizer is absmax-based and exactly homogeneous under power-of-two rescaling (Theorem 1).
    """
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True) * (1.0 / QMAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(w / scale).clamp_(-QMAX, QMAX).to(torch.int8)
    return codes, scale


def _is_int8_layer(module) -> bool:
    w = getattr(module, "weight", None)
    return w is not None and w.dtype == torch.int8 and getattr(module, "weight_scale", None) is not None


def _claim(path: str) -> bool:
    """True for exactly one process: the first to create `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def _tp_world_size() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:   # noqa: BLE001  (not initialized: a single-process engine)
        return 1


class QuRLWorkerExtension(vLLMColocateWorkerExtension):
    """VeRL's colocated vLLM worker extension with an INT8 W8A8 refit path."""

    def update_weights_from_ipc(self, *args, **kwargs):
        # After the last bucket, VeRL re-runs vLLM's process_weights_after_loading on the whole model (meant for
        # unquantized models, where it is harmless). For the INT8 model it is wrong twice over: it would transpose
        # the int8 weights a second time, and it fails on the attention layers' KV-scale attributes, which the first
        # pass already consumed. Nothing needs re-processing here: the refit writes codes and scales in the kernel's
        # layout, and every other synced tensor (embeddings, norms, biases) is loaded in place. So the call is made a
        # no-op for the duration of the sync.
        import vllm.model_executor.model_loader.utils as loader_utils

        original = loader_utils.process_weights_after_loading
        loader_utils.process_weights_after_loading = lambda *a, **k: None
        try:
            return super().update_weights_from_ipc(*args, **kwargs)
        finally:
            loader_utils.process_weights_after_loading = original

    # ---- per-replica state, created at the first sync -------------------------------------------------------------
    def _qurl_init(self, modules):
        if getattr(self, "_qurl_ready", False):
            return
        tp = _tp_world_size()
        if tp != 1:
            raise RuntimeError(f"[qurl int8 refit] tensor parallelism ({tp}) is not supported: per-row scales of "
                               f"row-parallel layers need whole rows; run the rollout with tensor_model_parallel_size=1")
        int8 = {n: m for n, m in modules.items() if _is_int8_layer(m)}
        self._qurl_expected = sum(len(m.logical_widths) for m in int8.values())
        self._qurl_layers = sum(1 for n in int8 if n.endswith("self_attn.qkv_proj"))
        if self._qurl_expected != 7 * self._qurl_layers:
            raise RuntimeError(f"[qurl int8 refit] expected 7 INT8 weights per layer, found {self._qurl_expected} "
                               f"for {self._qurl_layers} layers")
        self._qurl_sync = 0
        self._qurl_count = 0
        L = self._qurl_layers
        self._qurl_tracked = sorted({0, L // 2, L - 1})
        self._qurl_prev = {}          # (layer, proj) -> previous FP32 weights, tracked q/k/v only (Proposition 4)
        self._qurl_hist_codes = {}    # (layer, proj) -> previous codes (pinned CPU memory)
        self._qurl_hist_scale = {}    # (layer, proj) -> previous row scales, all rows
        self._qurl_codes_per_layer = {}
        self._qurl_e3 = bool(E3_PATH) and _claim(f"{E3_PATH}.{ATTEMPT}.claim")
        # The dump is of the run's starting weights: only the first attempt writes it (a resumed attempt starts from
        # a checkpoint, and its sync 1 would overwrite the dump with trained weights).
        self._qurl_dump = (bool(DUMP_DIR) and not os.path.exists(os.path.join(DUMP_DIR, "sync1_qkv_layer0.pt"))
                           and _claim(os.path.join(DUMP_DIR, f"claim.{ATTEMPT}")))
        self._qurl_info = ""
        self._qurl_ready = True

    def _qurl_new_sync(self):
        self._qurl_sync += 1
        self._qurl_count = 0
        self._qurl_seen = set()
        self._qurl_dumped_shards = set()
        if self._qurl_e3:
            dev = torch.device("cuda", torch.cuda.current_device())
            # [layer, projection, (changed codes, codes, changed scales, rows, sq. error, sq. norm)]
            self._qurl_stats = torch.zeros(self._qurl_layers, len(PROJS), 6, dtype=torch.float64, device=dev)
            self._qurl_pred = {}
            self._qurl_obs = {}

    def _qurl_finish_sync(self):
        """All weights of this sync arrived: print the replica's line and, in the logging replica, write E3."""
        print(f"[qurl int8 refit] pid {os.getpid()} sync {self._qurl_sync}: {self._qurl_count} linear weights "
              f"re-quantized to INT8{self._qurl_info}", flush=True)
        if not self._qurl_e3:
            return
        st = self._qurl_stats.cpu()

        def col(j, k):   # None where there is no history yet (the attempt's first sync)
            return [float(st[i, j, k] / st[i, j, k + 1]) if st[i, j, k + 1] > 0 else None
                    for i in range(self._qurl_layers)]

        rec = {"attempt": ATTEMPT, "start_step": START_STEP, "sync": self._qurl_sync, "time": time.time(),
               "layers": self._qurl_layers,
               "codes_per_layer": dict(self._qurl_codes_per_layer),
               "flip": {p: col(j, 0) for j, p in enumerate(PROJS)},
               "scale_changed": {p: col(j, 2) for j, p in enumerate(PROJS)},
               "qerr": {p: [float((st[i, j, 4] / st[i, j, 5]).sqrt()) for i in range(self._qurl_layers)]
                        for j, p in enumerate(PROJS)},
               "pred": {"layers": self._qurl_tracked,
                        **{p: [self._qurl_pred.get((i, p)) for i in self._qurl_tracked]
                           for p in ("q_proj", "k_proj", "v_proj")},
                        "observed_all_rows": {p: [self._qurl_obs.get((i, p)) for i in self._qurl_tracked]
                                              for p in ("q_proj", "k_proj", "v_proj")}}}
        with open(E3_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _update_weights(self, weights, peft_config, base_sync_done):
        model = self.model_runner.model
        modules = dict(model.named_modules())
        self._qurl_init(modules)
        rest = []
        for name, tensor in weights:
            m = _LINEAR.match(name)
            if m is None:
                rest.append((name, tensor))
                continue
            # A sync starts with the first linear weight after a complete one (whatever the bucket order). A weight
            # seen twice before its sync is complete means the previous sync was incomplete: refuse.
            if self._qurl_sync == 0 or self._qurl_count == self._qurl_expected:
                self._qurl_new_sync()
            elif name in self._qurl_seen:
                raise RuntimeError(f"[qurl int8 refit] sync {self._qurl_sync} re-quantized {self._qurl_count} of "
                                   f"{self._qurl_expected} linear weights before {name!r} arrived again")
            self._qurl_seen.add(name)
            fused, shard = _FUSED[m["proj"]]
            module_name = m["prefix"] + m["block"] + "." + fused
            layer = modules.get(module_name)
            if layer is None or not _is_int8_layer(layer):
                # Loading BF16 values into int8 parameters would silently corrupt the rollout model: refuse.
                raise RuntimeError(f"[qurl int8 refit] no INT8 layer {module_name!r} for synced weight {name!r}; "
                                   f"is vLLM running the compressed-tensors INT8 W8A8 config?")
            codes_t = layer.weight.data            # [in, out] view of the [out, in] codes (CUTLASS and Triton)
            out_total = sum(layer.logical_widths)
            assert codes_t.shape == (tensor.shape[1], out_total), (name, tuple(codes_t.shape), tuple(tensor.shape))
            r0 = sum(layer.logical_widths[:shard]) if shard is not None else 0
            r1 = r0 + tensor.shape[0]
            if not self._qurl_info:   # what the rollout actually runs: the synced dtype and vLLM's INT8 GEMM kernel
                scheme = getattr(layer, "scheme", None)          # compressed-tensors attaches the scheme to the layer
                kernel = type(getattr(scheme, "kernel", None)).__name__
                self._qurl_info = f" (synced weights arrive as {tensor.dtype}; INT8 kernel {kernel})"
            w = tensor.to(codes_t.device)
            codes, scale = quantize_rows(w)
            if self._qurl_e3:
                self._qurl_log_tensor(int(m["layer"]), m["proj"], w, codes, scale)
            codes_t.t()[r0:r1].copy_(codes)
            layer.weight_scale.data[r0:r1].copy_(scale)
            self._qurl_count += 1
            if self._qurl_dump and module_name.endswith(DUMP_MODULE_SUFFIX) and self._qurl_sync <= DUMP_SYNCS:
                self._qurl_dumped_shards.add(shard)
                if len(self._qurl_dumped_shards) == len(layer.logical_widths):
                    os.makedirs(DUMP_DIR, exist_ok=True)
                    torch.save({"codes": codes_t.t().cpu().clone(), "scale": layer.weight_scale.data.cpu().clone(),
                                "logical_widths": list(layer.logical_widths), "module": module_name},
                               os.path.join(DUMP_DIR, f"sync{self._qurl_sync}_qkv_layer0.pt"))
            if self._qurl_count == self._qurl_expected:
                self._qurl_finish_sync()
        if rest:
            super()._update_weights(rest, peft_config=peft_config, base_sync_done=base_sync_done)

    @torch.no_grad()
    def _qurl_log_tensor(self, i, proj, w, codes, scale):
        """Accumulate one tensor's flips and scale changes (against this replica's own history) and its error."""
        w = w.float()
        key = (i, proj)
        st = self._qurl_stats[i, PROJS.index(proj)]
        self._qurl_codes_per_layer[proj] = codes.numel()
        prev_codes, prev_scale = self._qurl_hist_codes.get(key), self._qurl_hist_scale.get(key)
        if prev_codes is not None:   # the previous sync's codes, kept in pinned CPU memory
            st[0] += (prev_codes.to(codes.device, non_blocking=True) != codes).sum()
            st[1] += codes.numel()
            st[2] += (prev_scale != scale).sum()
            st[3] += scale.numel()
        else:
            prev_codes = torch.empty(codes.shape, dtype=torch.int8, pin_memory=True)
            self._qurl_hist_codes[key] = prev_codes
        torch.cuda.current_stream().synchronize()   # the comparison above has read the old codes
        prev_codes.copy_(codes)
        self._qurl_hist_scale[key] = scale.clone()
        st[4] += (w - codes.float() * scale).double().pow(2).sum()
        st[5] += w.double().pow(2).sum()
        if i in self._qurl_tracked and proj in ("q_proj", "k_proj", "v_proj"):
            prev = self._qurl_prev.get(key)
            if prev is not None:
                # Proposition 4: flip probability min(1, |d| / Delta), Delta the new grid step; and the observed flip
                # share on the same (all) rows, the previous codes recomputed from the previous weights
                self._qurl_pred[key] = float(torch.clamp((w - prev).abs() / scale, max=1.0).mean())
                self._qurl_obs[key] = float((quantize_rows(prev)[0] != codes).float().mean())
            self._qurl_prev[key] = w.clone()
