# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expand the compact Orthrus draft seed on each active TP rank."""

import pypto.language as pl

from ..config import FLASH
from ..engram import ENGRAM_K
from ..input_pack import pack_x_hc


def draft_engram_prefix_rows(block):
    # After the last distinct row, the longest n-gram contains only mask tokens.
    return min(block, 1 + FLASH.engram_max_ngram_size)


def make_draft_seed(block, capacity):
    hidden, lanes = FLASH.hidden_size, FLASH.hc_mult
    prefix = draft_engram_prefix_rows(block)
    width = ENGRAM_K
    tile = 512

    @pl.jit.inline(auto_scope=False)
    def expand(
        embeddings: pl.Tensor[[2, hidden], pl.BF16],
        embedding_ids: pl.Tensor[[capacity], pl.INT64],
        compact: pl.Tensor[[2, prefix, width], pl.BF16],
        x_hc: pl.Tensor[[capacity, lanes, hidden], pl.FP32],
        engram: pl.Tensor[[2, block, width], pl.BF16],
    ):
        x_hc = pack_x_hc(embedding_ids, embeddings, x_hc)
        source = pl.reshape(compact, [2 * prefix, width])
        dest = pl.reshape(engram, [2 * block, width])
        workers = pl.max(1, pl.min(2 * block, pl.system.available_aiv_count()))
        for worker in pl.spmd(workers, name_hint="orthrus_expand_engram"):
            for row in pl.range(worker, 2 * block, workers):
                index = (row // block) * prefix + pl.min(row % block, prefix - 1)
                for column in pl.range(0, width, tile):
                    values = pl.load(source, [index, column], [1, tile])
                    pl.store(values, [row, column], dest)
        return x_hc, engram

    return expand
