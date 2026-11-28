# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only draft attention over the native packed cache."""



def make_packed_draft_projection(block_size):
    """Bind the native mHC normalization and BF16 draft projections."""
    from dataclasses import asdict

    import pypto.language as pl
    from ..config import FLASH, TP_SIZE
    from ..hc import hc_head_norm

    from .projections import make_packed_draft_qkv

    capacity = max(16, block_size)
    hidden, dim, latent = FLASH.hidden_size, FLASH.head_dim, FLASH.q_lora_rank
    heads = FLASH.num_attention_heads // TP_SIZE
    width = heads * dim
    mix = FLASH.hc_mult
    groups, output_rank = FLASH.o_groups // TP_SIZE, FLASH.o_lora_rank
    project = make_packed_draft_qkv(asdict(FLASH), tp=TP_SIZE, capacity=capacity, expose_joint=True)

    @pl.jit.inline(auto_scope=False)
    def prepare(
        x_hc: pl.Tensor, pre_mix: pl.Tensor, norm: pl.Tensor,
        wq_a_kv: pl.Tensor, q_gamma: pl.Tensor, wq_b: pl.Tensor, kv_gamma: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor,
        joint: pl.Out[pl.Tensor[[capacity, latent + dim], pl.BF16]],
    ):
        x = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        hc_head_norm(x_hc, pre_mix, norm, x, one_token=False)
        low_rank = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        query = pl.create_tensor([capacity, width], dtype=pl.BF16)
        temporary = pl.create_tensor([capacity, dim], dtype=pl.BF16)
        low_rank, query, temporary = project(
            x, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine, low_rank, query, temporary, joint, block_size,
        )
        return x, low_rank, pl.slice(query, [block_size, width], [0, 0]), temporary

    return prepare, (capacity, mix, hidden, heads, dim, width, groups, output_rank)
