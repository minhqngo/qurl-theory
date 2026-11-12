"""E0 (INT8): build compressed-tensors INT8 W8A8 checkpoints (QuRL's INT8 scheme).

Usage (llm-compressor venv): python make_int8_w8a8.py <out_root> <name>=<bf16_dir> [...]

Scheme "W8A8": INT8 weights with symmetric per-output-channel absmax scales (static) and INT8 activations
with dynamic per-token scales. Data-free (round-to-nearest), so no calibration set is used. Each input
checkpoint is quantized from its own weights, as QuRL would quantize the UAQ'd model before rollout.
"""
import os
import sys

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

out_root = sys.argv[1]
for spec in sys.argv[2:]:
    name, src = spec.split("=", 1)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(src)
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]))
    dst = os.path.join(out_root, f"int8_w8a8_{name}")
    model.save_pretrained(dst, save_compressed=True)
    tok.save_pretrained(dst)
    print("saved", dst)
