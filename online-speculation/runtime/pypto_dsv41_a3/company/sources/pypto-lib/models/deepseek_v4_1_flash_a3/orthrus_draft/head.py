# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate and place a TP-sharded BF16 Orthrus head."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from safetensors import safe_open

from ..config import FLASH
from ..weights import bf16_nz_shape
from .projections import pack_draft_input_projection, pack_draft_output_projection, prepare_draft_operands, read_draft_weights


def head_geometry():
    """Return the draft layer roles and canonical checkpoint shapes."""
    expected_layers = []
    for layer, ratio in enumerate(FLASH.compress_ratios):
        kv_source = max((i for i in FLASH.kv_source_layer_ids if i <= layer and FLASH.compress_ratios[i] == ratio),
                        default=None)
        index_source = max((i for i in FLASH.index_source_layer_ids if i <= layer and FLASH.compress_ratios[i] == ratio),
                           default=None)
        expected_layers.append(dict(layer=layer, ratio=ratio, kv_source=kv_source, index_source=index_source,
                                    candidate_source=FLASH.candidate_source_layer_id if ratio == 1 else None))
    h = FLASH
    common = {
        "attn_sink": (h.num_attention_heads,),
        "wq_a.weight": (h.q_lora_rank, h.hidden_size),
        "q_norm.weight": (h.q_lora_rank,),
        "wq_b.weight": (h.num_attention_heads * h.head_dim, h.q_lora_rank),
        "wkv.weight": (h.head_dim, h.hidden_size),
        "kv_norm.weight": (h.head_dim,),
        "wo_a.weight": (h.o_groups * h.o_lora_rank, h.num_attention_heads * h.head_dim // h.o_groups),
        "wo_b.weight": (h.hidden_size, h.o_groups * h.o_lora_rank),
    }
    indexer = {
        "indexer.wq_b.weight": (h.index_n_heads * h.index_head_dim, h.q_lora_rank),
        "indexer.weights_proj.weight": (h.index_n_heads, h.hidden_size),
    }
    shapes = {
        f"layers.{layer}.attn.{name}": shape
        for layer, entry in enumerate(expected_layers)
        for name, shape in (common | (indexer if layer == entry["index_source"] else {})).items()
    }
    return expected_layers, shapes


def read_head(checkpoint, head, block):
    """Match the exchange manifest and every shard header before device startup."""
    head = Path(head)
    manifest = json.loads((head / "manifest.json").read_text())
    role = manifest["artifact_role"]
    mask_token = manifest["mask_token_id"]
    if (manifest["format"] != "orthrus-dsv41-head-v1" or manifest["precision"] != "bf16-exchange"
            or manifest["logit_alignment"] != "next-token" or manifest["engram_policy"] != "committed-prefix-anchor-mask"
            or role not in ("initialized-plumbing", "trained") or type(mask_token) is not int
            or not 0 <= mask_token < FLASH.vocab_size):
        raise ValueError("Unsupported Orthrus head exchange contract")
    if manifest["block_size"] not in (8, 16, 32):
        raise ValueError("Invalid Orthrus head block size")
    if manifest["block_size"] != block and role == "trained":
        raise ValueError("Orthrus head block size differs from the draft graph")
    for key, name in (("config", "config.json"), ("index", "quant_model_weights.safetensors.index.json"),
                      ("tokenizer", "tokenizer.json")):
        if manifest[f"base_{key}_sha256"] != sha256((checkpoint.root / name).read_bytes()).hexdigest():
            raise ValueError(f"Orthrus head was exported against a different {name}")
    config = json.loads((checkpoint.root / "config.json").read_text())["text_config"]
    expected_layers, shapes = head_geometry()
    if manifest["layers"] != expected_layers:
        raise ValueError("Orthrus head roles differ from the A3 backbone")
    indices = list(head.glob("*.safetensors.index.json"))
    if len(indices) != 1:
        raise ValueError("Expected one Orthrus head safetensors index")
    files = json.loads(indices[0].read_text())["weight_map"]
    if files.keys() != shapes.keys():
        raise ValueError("Orthrus head index differs from the A3 tensor set")
    shards = defaultdict(set)
    for name, shard in files.items():
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Orthrus head index references an invalid shard")
        shards[shard].add(name)
    for shard, names in shards.items():
        with safe_open(head / shard, framework="pt") as artifact:
            if set(artifact.keys()) != names:
                raise ValueError(f"Orthrus head shard {shard} differs from its index")
            for name in names:
                value = artifact.get_slice(name)
                dtype = "F32" if name.endswith(".attn_sink") else "BF16"
                if value.get_dtype() != dtype or tuple(value.get_shape()) != shapes[name]:
                    raise ValueError(f"Invalid Orthrus head tensor {name}")
    return config, manifest, files


def bind_head(runtime, head, world, tp, artifact):
    """Stream the exchange artifact into TP-sharded device banks once."""
    config, manifest, files = artifact
    layers = manifest["layers"]
    index_owners = tuple(layer for layer, role in enumerate(layers) if role["index_source"] == layer)
    banks = [{} for _ in range(world)]
    total_bytes = 0

    def upload(name, payload, slot, slots, ranks):
        row_bytes = payload.nbytes
        for rank in ranks:
            bank = banks[rank].get(name)
            if bank is None:
                shape = (slots * payload.shape[0], *payload.shape[1:])
                bank = banks[rank][name] = runtime.alloc_tensor(shape, payload.dtype, worker_id=rank)
            runtime.copy_to(bank.data_ptr, payload.data_ptr(), row_bytes,
                            dst_offset=slot * row_bytes, worker_id=rank)
        return len(ranks) * row_bytes

    for layer in range(len(layers)):
        tensors = read_draft_weights(head, manifest, files, layer)
        for lane in range(tp):
            operands = prepare_draft_operands(tensors, config, tp_rank=lane, tp=tp)
            wo = pack_draft_output_projection(operands)
            wo_a, wo_b = wo["wo_a"], wo["wo_b"]
            payloads = {
                "draft_q_a_kv": pack_draft_input_projection(operands).unsqueeze(0),
                "draft_q_gamma": operands["q_norm"].unsqueeze(0),
                "draft_q_b": operands["wq_b"].unsqueeze(0),
                "draft_kv_gamma": operands["kv_norm"].unsqueeze(0),
                "draft_sink": operands["sink"].unsqueeze(0),
                "draft_wo_a": wo_a.reshape(bf16_nz_shape(*wo_a.shape)),
                "draft_wo_b": wo_b.reshape(bf16_nz_shape(1, *wo_b.shape)),
            }
            ranks = tuple(range(lane, world, tp))
            for name, payload in payloads.items():
                total_bytes += upload(name, payload, layer, len(layers), ranks)
            if layer in index_owners:
                slot = index_owners.index(layer)
                total_bytes += upload("draft_index_q", operands["idx_wq"].unsqueeze(0), slot, len(index_owners), ranks)
                total_bytes += upload("draft_index_weight", operands["idx_weight"].unsqueeze(0),
                                      slot, len(index_owners), ranks)
    return banks, manifest, total_bytes
