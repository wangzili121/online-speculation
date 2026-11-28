# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare BF16 DSV4.1 diffusion projections for a native PyPTO TP rank."""

from .swa import HEAD_DIM

QKV_TENSORS = ("wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "attn_sink")
OUTPUT_TENSORS = ("wo_a.weight", "wo_b.weight")
INDEX_TENSORS = ("indexer.wq_b.weight", "indexer.weights_proj.weight")


def read_draft_weights(head, manifest, files, layer):
    from contextlib import ExitStack

    from safetensors import safe_open

    if not 0 <= layer < len(manifest["layers"]):
        raise ValueError("Draft layer is outside the checkpoint")
    owns_index = manifest["layers"][layer]["index_source"] == layer
    names = QKV_TENSORS + OUTPUT_TENSORS + (INDEX_TENSORS if owns_index else ())
    with ExitStack() as stack:
        weights, shards = {}, {}
        for name in names:
            key = f"layers.{layer}.attn.{name}"
            file = files[key]
            if file not in shards:
                shards[file] = stack.enter_context(safe_open(head / file, framework="pt", device="cpu"))
            weights[name] = shards[file].get_tensor(key)
        return weights


def prepare_draft_operands(weights, config, *, tp_rank, tp=4):
    operands = prepare_qkv(weights, config, tp_rank=tp_rank, tp=tp)
    operands.update(prepare_output(weights, config, tp_rank=tp_rank, tp=tp))
    if INDEX_TENSORS[0] in weights:
        operands.update(idx_wq=weights[INDEX_TENSORS[0]].T.contiguous(),
                        idx_weight=weights[INDEX_TENSORS[1]].T.contiguous())
    return operands


def prepare_qkv(weights, config, *, tp_rank, tp=4):
    """Keep the exchanged head precision and shard only the query heads."""
    import torch

    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"], config["head_dim"]
    if not 0 <= tp_rank < tp or heads % tp or dim != HEAD_DIM:
        raise ValueError("Invalid DSV4.1 draft head partition")
    required = {
        "wq_a.weight": (latent, hidden),
        "q_norm.weight": (latent,),
        "wq_b.weight": (heads * dim, latent),
        "wkv.weight": (dim, hidden),
        "kv_norm.weight": (dim,),
        "attn_sink": (heads,),
    }
    for name, shape in required.items():
        weight = weights[name]
        dtype = torch.float32 if name == "attn_sink" else torch.bfloat16
        if weight.shape != shape or weight.dtype != dtype or weight.device.type != "cpu":
            raise ValueError(f"Invalid DSV4.1 draft operand {name}")
    local_heads = heads // tp
    low, high = tp_rank * local_heads, (tp_rank + 1) * local_heads
    return dict(
        wq_a=weights["wq_a.weight"].T.contiguous(),
        q_norm=weights["q_norm.weight"].contiguous(),
        wq_b=weights["wq_b.weight"][low * dim:high * dim].T.contiguous(),
        wkv=weights["wkv.weight"].T.contiguous(),
        kv_norm=weights["kv_norm.weight"].contiguous(),
        sink=weights["attn_sink"][low:high].contiguous(),
    )


def pack_draft_input_projection(operands):
    """Lay out the Q-A and KV channels consecutively for one resident GEMM."""
    import torch

    return torch.cat((operands["wq_a"], operands["wkv"]), dim=1)


def make_packed_draft_qkv(config, *, tp=4, capacity=16, expose_joint=False):
    """Project a BF16 draft head using the 1122 A3 GEMM, RMSNorm and RoPE."""
    import pypto.language as pl
    from ..gemm import gemm_bf16_to_bf16 as gemm
    from ..activation_quant import _tile_rsqrt_for_norm
    from ..rope import make_pipelined_rope

    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"] // tp, config["head_dim"]
    q_width, joint_width = heads * dim, latent + dim
    eps = config["rms_norm_eps"]
    q_rope = make_pipelined_rope(
        capacity, heads, dim, inverse=False, name_hint="orthrus_q_rope", allow_early_resolve=False,
    )
    kv_rope = make_pipelined_rope(
        capacity, 1, dim, inverse=False, name_hint="orthrus_kv_rope", allow_early_resolve=False,
    )

    @pl.jit.inline
    def norm_projection(
        joint: pl.Tensor[[capacity, joint_width], pl.BF16],
        gamma: pl.Tensor,
        output: pl.Tensor,
        active: pl.Scalar[pl.INT32],
        columns: pl.constexpr,
        first_col: pl.constexpr,
    ):
        # Load directly from the joined GEMM's row stride. A view with a
        # narrower declared width would require repacking before a native norm.
        for task in pl.spmd((active + 7) // 8, name_hint="orthrus_qkv_norm"):
            row = task * 8
            rows = pl.min(8, active - row)
            raw = pl.load(joint, [row, first_col], [8, columns], valid_shape=[rows, columns])
            padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, columns)
            value = pl.cast(padded, pl.FP32)
            weight_row = pl.tensor.view(gamma, [1, columns])
            gamma_row = pl.cast(pl.load(weight_row, [0, 0], [1, columns]), pl.FP32)
            sum_scratch = pl.create_tile([8, columns], dtype=pl.FP32)
            mean = pl.mul(pl.row_sum(pl.mul(value, value), tmp_tile=sum_scratch), 1.0 / columns)
            inverse_scratch = pl.create_tile([8, 1], dtype=pl.FP32)
            inv = _tile_rsqrt_for_norm(pl.add(mean, eps), inverse_scratch)
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma_row)
            narrowed = pl.cast(normalized, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(narrowed, rows, columns), [row, 0], output)
        return output

    @pl.jit.inline(auto_scope=False)
    def project(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a_kv: pl.Tensor[[hidden, joint_width], pl.BF16],
        q_gamma: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, q_width], pl.BF16],
        kv_gamma: pl.Tensor[[dim], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        low_rank: pl.Out[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.Out[pl.Tensor[[capacity, q_width], pl.BF16]],
        temporary: pl.Out[pl.Tensor[[capacity, dim], pl.BF16]],
        qa_kv: pl.Out[pl.Tensor[[capacity, joint_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        q_raw = pl.create_tensor([capacity, q_width], dtype=pl.BF16)
        kv_normalized = pl.create_tensor([capacity, dim], dtype=pl.BF16)
        start = pl.system.task_dummy(deps=[])
        qa_kv, _ = gemm(x, wq_a_kv, qa_kv, active, start, hidden, joint_width,
                        128, 256, 32, "orthrus_qa_kv", 0, 0, False, 16, 2)
        low_rank = norm_projection(qa_kv, q_gamma, low_rank, active, latent, 0)
        kv_normalized = norm_projection(qa_kv, kv_gamma, kv_normalized, active, dim, latent)
        q_raw, _ = gemm(low_rank, wq_b, q_raw, active, start, latent, q_width,
                        128, 256, 32, "orthrus_qb", 0, 0, False, 16, 2)
        query = q_rope(q_raw, cosine, sine, query, active)
        temporary = kv_rope(kv_normalized, cosine, sine, temporary, active)
        if active < capacity:
            # K8 uses a 16-row KV carrier. Cube multiplies the masked lanes
            # as well, so those values must be zero even with reused buffers.
            for task in pl.spmd((capacity - active + 7) // 8, name_hint="orthrus_zero_kv_padding"):
                row = active + task * 8
                rows = pl.min(8, capacity - row)
                zero = pl.tile.full([8, dim], dtype=pl.BF16, value=0.0)
                pl.store(pl.set_validshape(zero, rows, dim), [row, 0], temporary)
        return low_rank, query, temporary

    if expose_joint:
        return project

    @pl.jit
    def dispatch(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a_kv: pl.Tensor[[hidden, joint_width], pl.BF16],
        q_gamma: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, q_width], pl.BF16],
        kv_gamma: pl.Tensor[[dim], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        low_rank: pl.Out[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.Out[pl.Tensor[[capacity, q_width], pl.BF16]],
        temporary: pl.Out[pl.Tensor[[capacity, dim], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        joint = pl.create_tensor([capacity, joint_width], dtype=pl.BF16)
        return project(x, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine,
                       low_rank, query, temporary, joint, active)

    return dispatch


def prepare_output(weights, config, *, tp_rank, tp=4):
    """Assign complete Wo-A groups and their matching Wo-B columns to this rank."""
    import torch

    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    width = config["num_attention_heads"] * config["head_dim"]
    hidden = config["hidden_size"]
    if not 0 <= tp_rank < tp or groups % tp or width % groups:
        raise ValueError("Invalid DSV4.1 grouped output partition")
    wo_a, wo_b = (weights[name] for name in OUTPUT_TENSORS)
    if any(value.dtype != torch.bfloat16 or value.device.type != "cpu" for value in (wo_a, wo_b)):
        raise ValueError("Grouped output requires BF16 CPU weights")
    if wo_a.shape != (groups * o_rank, width // groups) or wo_b.shape != (hidden, groups * o_rank):
        raise ValueError("Invalid DSV4.1 grouped output shape")
    local_groups = groups // tp
    low, high = tp_rank * local_groups, (tp_rank + 1) * local_groups
    return dict(wo_a=wo_a.view(groups, o_rank, width // groups)[low:high].contiguous(),
                wo_b=wo_b[:, low * o_rank:high * o_rank].T.contiguous())


def pack_draft_output_projection(operands):
    """Use the same resident Cube layout as the 1122 target Wo projection."""
    from ..weights import pack_bf16_nz

    return dict(wo_a=pack_bf16_nz(operands["wo_a"].transpose(-2, -1).contiguous()),
                wo_b=pack_bf16_nz(operands["wo_b"]))


def make_packed_draft_output_tape(block, last_layers):
    """Direct the original Wo intermediates to the requested suffix slot."""
    import pypto.language as pl
    from ..config import FLASH as model, TP_SIZE as tp
    from ..o_proj import decode_o_proj_nz, decode_o_proj_nz_capture

    heads, dim = model.num_attention_heads // tp, model.head_dim
    groups, rank, hidden = model.o_groups // tp, model.o_lora_rank, model.hidden_size
    width, middle, phase = heads * dim, groups * rank, model.qk_rope_head_dim // 2
    tape_rows = max(1, last_layers * block)

    @pl.jit.inline(auto_scope=False)
    def project(
        attention: pl.Tensor[[block * heads, dim], pl.FP32],
        wo_a: pl.Tensor[[groups, width // groups, rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[middle, hidden], pl.BF16, pl.NZ],
        cosine: pl.Tensor[[block, phase], pl.FP32],
        sine: pl.Tensor[[block, phase], pl.FP32],
        contribution: pl.Tensor[[block, hidden], pl.BF16],
        tape_input: pl.Tensor[[tape_rows, width], pl.BF16],
        tape_latent: pl.Tensor[[tape_rows, middle], pl.BF16],
        head_major: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32],
    ):
        attended = pl.create_tensor([block, width], dtype=pl.BF16)
        if head_major != 0:
            for task in pl.spmd(block * heads, name_hint="orthrus_swa_to_wo"):
                token, head = task // heads, task % heads
                head_values = pl.load(attention, [head * block + token, 0], [1, dim])
                pl.store(pl.cast(head_values, pl.BF16, mode="rint"), [token, head * dim], attended)
        else:
            flat = pl.reshape(attention, [block, width])
            for token in pl.spmd(block, name_hint="orthrus_context_to_wo"):
                for column in pl.range(0, width, dim * 4):
                    quartet = pl.load(flat, [token, column], [1, dim * 4])
                    pl.store(pl.cast(quartet, pl.BF16, mode="rint"), [token, column], attended)
        if last_layers > 0:
            if slot >= 0:
                saved_input = pl.slice(tape_input, [block, width], [slot * block, 0])
                saved_latent = pl.slice(tape_latent, [block, middle], [slot * block, 0])
                decode_o_proj_nz_capture(attended, wo_a, wo_b, cosine, sine, contribution, block,
                                          saved_input, saved_latent)
            else:
                decode_o_proj_nz(attended, wo_a, wo_b, cosine, sine, contribution, block)
        else:
            decode_o_proj_nz(attended, wo_a, wo_b, cosine, sine, contribution, block)
        return contribution

    return project
