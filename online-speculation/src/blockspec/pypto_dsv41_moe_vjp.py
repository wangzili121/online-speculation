"""PyPTO leaves for the fixed-route DSV4.1 MoE reverse pass."""


def make_moe_combine_vjp(rows=32, hidden=5120, topk=6, *, inline=False):
    """Return gradients for shared output, selected expert rows and weights."""
    if rows < 1 or hidden < 64 or hidden % 64 or topk != 6:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and Top-K=6")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64

    @kernel
    def moe_combine_vjp(
        output_gradient: pl.Tensor,
        expert_outputs: pl.Tensor,
        route_weights: pl.Tensor,
        shared_gradient: pl.Out[pl.Tensor],
        expert_gradient: pl.Out[pl.Tensor],
        weight_gradient: pl.Out[pl.Tensor],
    ):
        # One block owns a complete token row. Route gradients occupy six
        # values in a 16-FP32 row, avoiding concurrent whole-cache-line stores.
        with pl.spmd(rows, name_hint="orthrus_dsv41_moe_combine_backward"):
            token = pl.tile.get_block_idx()
            for slot in pl.range(topk):
                task = token * topk + slot
                total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                weight = pl.read(route_weights, [token, slot])
                for part in pl.range(hidden // tile):
                    dy = pl.cast(
                        pl.slice(output_gradient, [1, tile], [token, part * tile]), pl.FP32,
                    )
                    expert = pl.cast(
                        pl.slice(expert_outputs, [1, tile], [task, part * tile]), pl.FP32,
                    )
                    if slot == 0:
                        shared_gradient[token:token + 1, part * tile:(part + 1) * tile] = pl.cast(
                            dy, pl.BF16, mode="rint",
                        )
                    expert_gradient[task:task + 1, part * tile:(part + 1) * tile] = pl.cast(
                        pl.mul(dy, weight), pl.BF16, mode="rint",
                    )
                    product = pl.reshape(pl.mul(dy, expert), [8, 8])
                    partial = pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(product),
                    )
                    total = pl.add(
                        total,
                        pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.transpose(partial, 0, 1)),
                        ),
                    )
                pl.write(weight_gradient, [token, slot], pl.read(total, [0, 0]))
        return shared_gradient, expert_gradient, weight_gradient

    @pl.jit.host
    def entry(
        output_gradient: pl.Tensor,
        expert_outputs: pl.Tensor,
        route_weights: pl.Tensor,
        shared_gradient: pl.Out[pl.Tensor],
        expert_gradient: pl.Out[pl.Tensor],
        weight_gradient: pl.Out[pl.Tensor],
    ):
        moe_combine_vjp(
            output_gradient, expert_outputs, route_weights,
            shared_gradient, expert_gradient, weight_gradient, device=0,
        )

    return moe_combine_vjp if inline else entry


def make_normalized_route_vjp(rows=32, topk=6, *, inline=False):
    """Differentiate normalized selected scores while keeping Top-K IDs fixed."""
    if rows < 1 or topk != 6:
        raise ValueError("expected DSV4.1 rows and Top-K=6")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def normalized_route_vjp(
        selected_scores: pl.Tensor,
        weight_gradient: pl.Tensor,
        score_gradient: pl.Out[pl.Tensor],
        route_scale: pl.Scalar[pl.FP32],
    ):
        # ``score_gradient`` is [rows, 16], so each token owns one complete
        # 64-byte FP32 cache line even though only six values are meaningful.
        with pl.spmd(rows, name_hint="orthrus_dsv41_route_normalize_backward"):
            token = pl.tile.get_block_idx()
            total = pl.cast(0., pl.FP32)
            weighted = pl.cast(0., pl.FP32)
            for slot in pl.range(topk):
                score = pl.read(selected_scores, [token, slot])
                total = total + score
                weighted = weighted + score * pl.read(weight_gradient, [token, slot])
            for slot in pl.range(topk):
                result = route_scale / total * (
                    pl.read(weight_gradient, [token, slot]) - weighted / total
                )
                pl.write(score_gradient, [token, slot], result)
        return score_gradient

    @pl.jit.host
    def entry(
        selected_scores: pl.Tensor,
        weight_gradient: pl.Tensor,
        score_gradient: pl.Out[pl.Tensor],
        route_scale: pl.Scalar[pl.FP32],
    ):
        normalized_route_vjp(
            selected_scores, weight_gradient, score_gradient, route_scale, device=0,
        )

    return normalized_route_vjp if inline else entry
