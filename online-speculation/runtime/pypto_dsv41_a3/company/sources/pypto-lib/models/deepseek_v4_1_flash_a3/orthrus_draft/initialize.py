# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export a checkpoint-initialized Orthrus head without loading the model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from ..config import FLASH
from ..weights import Checkpoint
from .head import head_geometry, read_head


def assemble_layer(checkpoint, description, layer, shapes):
    prefix = f"layers.{layer}.attn."
    tensors = {}
    for name, shape in shapes.items():
        if not name.startswith(prefix):
            continue
        value = checkpoint.tensor(name)
        scheme = description[name]
        if scheme == "W8A8_DYNAMIC":
            scale = checkpoint.tensor(name + "_scale")
            offset = checkpoint.tensor(name + "_offset")
            if (value.dtype != torch.int8 or scale.shape != (value.shape[0], 1)
                    or offset.shape != scale.shape or torch.count_nonzero(offset)):
                raise ValueError(f"Invalid symmetric channel quantization for {name}")
            value = value.float() * scale
        elif scheme != "FLOAT":
            raise ValueError(f"Unsupported draft projection quantization for {name}")
        dtype = torch.float32 if name.endswith(".attn_sink") else torch.bfloat16
        value = value.to(dtype).contiguous()
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid initialized draft tensor {name}")
        tensors[name] = value
    return tensors


def initialize(checkpoint, output, mask, block):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((checkpoint.root / "config.json").read_text())
    reserved = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id"):
        value = config.get(name)
        reserved.update(value if isinstance(value, list) else (value,))
    if not 0 <= mask < FLASH.vocab_size or mask in reserved:
        raise ValueError("The draft mask must be a valid dedicated token ID")
    layers, shapes = head_geometry()
    description = json.loads((checkpoint.root / "quant_model_description.json").read_text())
    manifest = dict(format="orthrus-dsv41-head-v1", precision="bf16-exchange", artifact_role="initialized-plumbing",
                    mask_token_id=mask, block_size=block, logit_alignment="next-token",
                    engram_policy="committed-prefix-anchor-mask", layers=layers)
    for key, name in (("config", "config.json"), ("index", "quant_model_weights.safetensors.index.json"),
                      ("tokenizer", "tokenizer.json")):
        manifest[f"base_{key}_sha256"] = sha256((checkpoint.root / name).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging:
        head = Path(staging)
        weight_map, total = {}, 0
        for layer in range(len(layers)):
            tensors = assemble_layer(checkpoint, description, layer, shapes)
            shard_name = f"model-{layer+1:05d}-of-{len(layers):05d}.safetensors"
            save_file(tensors, head / shard_name)
            weight_map.update(dict.fromkeys(tensors, shard_name))
            total += sum(t.nbytes for t in tensors.values())
        (head / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (head / "model.safetensors.index.json").write_text(json.dumps(
            dict(weight_map=weight_map, metadata=dict(total_size=total)), indent=2,
        ) + "\n")
        read_head(checkpoint, head, block)
        head.rename(output)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--block", type=int, choices=(8, 16, 32), default=16)
    args = parser.parse_args()
    size = initialize(Checkpoint(args.checkpoint), args.output, args.mask_token_id, args.block)
    print(json.dumps(dict(head=str(args.output), weight_bytes=size)), flush=True)


if __name__ == "__main__":
    main()
