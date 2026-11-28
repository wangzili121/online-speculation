# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frozen router input VJP on the saved DSV4.1 Top-K routes."""


def make_packed_router_vjp(rows, hidden, experts, topk, temperature, route_scale, *, inline=False):
    import pypto.language as pl

    route_pad = (topk + 7) // 8 * 8
    kernel = pl.jit.inline(auto_scope=False) if inline else pl.jit

    @kernel
    def router(
        normalized: pl.Tensor[[rows, hidden], pl.BF16],
        route_ids: pl.Tensor[[rows, 16], pl.INT32],
        droute: pl.Tensor[[rows, route_pad], pl.FP32],
        weight: pl.Tensor[[experts, hidden], pl.FP32],
        routed_dx: pl.Tensor[[rows, hidden], pl.FP32],
        result: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(rows, name_hint="orthrus_router_selected_scores_vjp"):
            row = pl.tile.get_block_idx()
            logits = pl.tile.full([1, route_pad], dtype=pl.FP32, value=0.)
            dx = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            if row < active:
                x = pl.cast(pl.slice(normalized, [1, hidden], [row, 0]), pl.FP32)
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                for slot in pl.range(topk):
                    expert = pl.read(route_ids, [row, slot])
                    w = pl.slice(weight, [1, hidden], [expert, 0])
                    dot_strips = pl.reshape(pl.mul(x, w), [8, hidden // 8])
                    partials = pl.row_expand_mul(ones, pl.row_sum(dot_strips))
                    dot = pl.row_sum(pl.transpose(partials, 0, 1))
                    pl.tile.write(logits, [0, slot], pl.read(dot, [0, 0]) / temperature)
                tail = pl.exp(pl.neg(pl.abs(logits)))
                negative_exp = pl.exp(pl.minimum(logits, 0.))
                negative_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(logits), 10.), 0.), 1.)
                softplus = pl.maximum(
                    pl.add(pl.maximum(logits, 0.), pl.log(pl.add(tail, 1.))),
                    pl.mul(negative_exp, negative_mask),
                )
                scores = pl.sqrt(softplus)
                slopes = pl.div(negative_exp, pl.mul(pl.add(tail, 1.),
                               pl.mul(pl.maximum(scores, 1e-20), 2. * temperature)))
                total = pl.cast(1e-20, pl.FP32)
                weighted = pl.cast(0., pl.FP32)
                for slot in pl.range(topk):
                    score = pl.tile.read(scores, [0, slot])
                    total = total + score
                    weighted = weighted + score * pl.read(droute, [row, slot])
                dx = pl.slice(routed_dx, [1, hidden], [row, 0])
                for slot in pl.range(topk):
                    expert = pl.read(route_ids, [row, slot])
                    dw = route_scale / total * (pl.read(droute, [row, slot]) - weighted / total)
                    w = pl.slice(weight, [1, hidden], [expert, 0])
                    dx = pl.add(dx, pl.mul(w, dw * pl.tile.read(slopes, [0, slot])))
            result[row:row + 1, :] = dx
        return result

    if inline:
        return router

    @pl.jit.host
    def entry(
        normalized: pl.Tensor[[rows, hidden], pl.BF16],
        route_ids: pl.Tensor[[rows, 16], pl.INT32],
        droute: pl.Tensor[[rows, route_pad], pl.FP32],
        weight: pl.Tensor[[experts, hidden], pl.FP32],
        routed_dx: pl.Tensor[[rows, hidden], pl.FP32],
        result: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        router(normalized, route_ids, droute, weight, routed_dx, result, active, device=0)
    return entry
