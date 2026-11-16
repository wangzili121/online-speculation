"""Explicit derivatives for the parameterized GLM draft MLA output."""

from .pypto_glm53_mla import HEADS, VALUE


def make_output_backward():
    """Differentiate the BF16 MLA output projection with FP32 accumulation."""
    import pypto.language as pl
    from models.glm5_3_flash.config import D

    rows, width = 16, HEADS * VALUE
    columns, reduction = 64, 128

    @pl.jit
    def backward(
        context: pl.Tensor[[rows, width], pl.BF16], weight: pl.Tensor[[D, width], pl.BF16],
        upstream: pl.Tensor[[rows, D], pl.BF16], weight_grad: pl.Out[pl.Tensor[[D, width], pl.FP32]],
        context_grad: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
    ):
        for tile in pl.spmd((D // columns) * (width // columns), name_hint="orthrus_glm53_dwo"):
            output_column = tile // (width // columns) * columns
            input_column = tile % (width // columns) * columns
            dy = pl.slice(upstream, [rows, columns], [0, output_column])
            values = pl.slice(context, [rows, columns], [0, input_column])
            dw = pl.matmul(dy, values, a_trans=True, out_dtype=pl.FP32)
            weight_grad[output_column:output_column + columns, input_column:input_column + columns] = dw

        for tile in pl.spmd(width // columns, name_hint="orthrus_glm53_dcontext"):
            column = tile * columns
            dc = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, D, reduction, stage=2):
                grad_tile = pl.slice(upstream, [rows, reduction], [0, start])
                weight_tile = pl.slice(weight, [reduction, columns], [start, column])
                dc = pl.matmul_acc(dc, grad_tile, weight_tile, init_cond=start == 0)
            context_grad[:, column:column + columns] = dc
        return weight_grad, context_grad

    @pl.jit.host
    def entry(context: pl.Tensor, weight: pl.Tensor, upstream: pl.Tensor,
              weight_grad: pl.Out[pl.Tensor], context_grad: pl.Out[pl.Tensor]):
        backward(context, weight, upstream, weight_grad, context_grad, device=0)

    return entry


def make_joint_softmax_backward(rows, history_capacity, block_capacity=128):
    """Apply the score adjoint across the shared history and the draft block."""
    import pypto.language as pl

    row_tile, tile = 8, 128
    if rows < row_tile or rows % row_tile or min(history_capacity, block_capacity) < tile:
        raise ValueError("joint MLA softmax requires eight query heads and 128-column tiles")
    if history_capacity % tile or block_capacity % tile:
        raise ValueError("history and draft probabilities must be tile-aligned")
    capacity = history_capacity + block_capacity

    @pl.jit
    def backward(
        probability: pl.Tensor[[rows, capacity], pl.FP32],
        upstream: pl.Tensor[[rows, capacity], pl.FP32],
        score_gradient: pl.Out[pl.Tensor[[rows, capacity], pl.FP32]],
    ):
        for fragment in pl.spmd(rows // row_tile, name_hint="orthrus_glm53_joint_softmax_backward"):
            begin = fragment * row_tile
            # One row dot across both regions, including the mass contributed
            # by the temporary draft keys. Padding has probability zero.
            dot = pl.full([row_tile, 8], dtype=pl.FP32, value=0.0)
            unit = pl.full([row_tile, 8], dtype=pl.FP32, value=1.0)
            for column in pl.range(0, capacity, tile):
                p = probability[begin:begin + row_tile, column:column + tile]
                g = upstream[begin:begin + row_tile, column:column + tile]
                dot = pl.add(dot, pl.row_expand_mul(unit, pl.row_sum(pl.mul(p, g))))
            coupling = pl.row_max(dot)
            for column in pl.range(0, capacity, tile):
                p = probability[begin:begin + row_tile, column:column + tile]
                g = upstream[begin:begin + row_tile, column:column + tile]
                score_gradient[begin:begin + row_tile, column:column + tile] = pl.mul(p, pl.row_expand_sub(g, coupling))
        return score_gradient

    @pl.jit.host
    def entry(probability: pl.Tensor, upstream: pl.Tensor, score_gradient: pl.Out[pl.Tensor]):
        backward(probability, upstream, score_gradient, device=0)

    return entry
