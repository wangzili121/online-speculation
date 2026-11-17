"""Propagate the online loss through GLM's frozen vocabulary projection."""


def make_vocab_backward(local_vocab, model_width, rows=16, *, device_entry=False):
    """Return a TP-local FP32 activation gradient; the caller sums the ranks."""
    return make_frozen_bf16_input_backward(local_vocab, model_width, rows, device_entry=device_entry)


def make_frozen_bf16_input_backward(local_vocab, model_width, rows=16, *, device_entry=False, refine=False):
    """Apply a frozen BF16 weight transpose to an FP32 upstream gradient."""
    import pypto.language as pl

    reduction = columns = 64
    if rows < 16 or rows % 16 or model_width <= 0 or model_width % columns or local_vocab <= 0:
        raise ValueError("GLM vocabulary backward requires aligned rows and model width")
    vocab_pad = (local_vocab + reduction - 1) // reduction * reduction

    @pl.jit
    def vocab_backward(
        grad_logits: pl.Tensor[[rows, vocab_pad], pl.FP32],
        vocabulary: pl.Tensor[[vocab_pad, model_width], pl.BF16],
        partial: pl.Out[pl.Tensor[[rows, model_width], pl.FP32]],
    ):
        for tile in pl.spmd(model_width // columns, name_hint="orthrus_glm53_vocab_backward"):
            column = tile * columns
            acc = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, vocab_pad, reduction, stage=2):
                grad = pl.slice(grad_logits, [rows, reduction], [0, start])
                weights = pl.slice(vocabulary, [reduction, columns], [start, column])
                high = pl.cast(grad, pl.BF16, mode="rint")
                low = pl.cast(pl.sub(grad, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
                acc = pl.matmul_acc(acc, high, weights, init_cond=start == 0)
                acc = pl.matmul_acc(acc, low, weights)
                if refine:
                    rest = pl.sub(pl.sub(grad, pl.cast(high, pl.FP32)), pl.cast(low, pl.FP32))
                    acc = pl.matmul_acc(acc, pl.cast(rest, pl.BF16, mode="rint"), weights)
            partial[:, column:column + columns] = acc
        return partial

    @pl.jit.host
    def entry(grad_logits: pl.Tensor, vocabulary: pl.Tensor, partial: pl.Out[pl.Tensor]):
        vocab_backward(grad_logits, vocabulary, partial, device=0)

    return vocab_backward if device_entry else entry


def make_final_norm_backward(model_width, epsilon, rows=16, streams=4, *, device_entry=False):
    """Differentiate the frozen RMSNorm and distribute to the mean-pooled streams."""
    import pypto.language as pl

    tile = 128
    if model_width < tile or model_width % tile or rows < 1 or streams < 1 or epsilon <= 0:
        raise ValueError("invalid GLM output normalization geometry")

    @pl.jit
    def norm_backward(
        activation: pl.Tensor[[rows, model_width], pl.BF16], scale: pl.Tensor[[model_width], pl.BF16],
        gradient: pl.Tensor[[rows, model_width], pl.FP32],
        output: pl.Out[pl.Tensor[[rows * streams, model_width], pl.FP32]],
    ):
        for row in pl.spmd(rows, name_hint="orthrus_glm53_final_norm_backward"):
            squares = pl.cast(0.0, pl.FP32)
            dots = pl.cast(0.0, pl.FP32)
            for col in pl.range(0, model_width, tile):
                x = pl.cast(activation[row:row + 1, col:col + tile], pl.FP32)
                gamma = pl.cast(pl.reshape(scale[col:col + tile], [1, tile]), pl.FP32)
                g = pl.mul(gradient[row:row + 1, col:col + tile], gamma)
                aligned = pl.full([8, tile], dtype=pl.FP32, value=0.0)
                square_rows = pl.col_expand(aligned, pl.mul(x, x))
                product_rows = pl.col_expand(aligned, pl.mul(x, g))
                squares = squares + pl.read(pl.row_sum(square_rows), [0, 0])
                dots = dots + pl.read(pl.row_sum(product_rows), [0, 0])
            mean = squares * (1.0 / model_width) + epsilon
            inverse = pl.read(pl.recip(pl.sqrt(pl.add(pl.full([1, 8], dtype=pl.FP32, value=0.0), mean))), [0, 0])
            correction = dots * inverse * inverse * (1.0 / model_width)
            for col in pl.range(0, model_width, tile):
                x = pl.cast(activation[row:row + 1, col:col + tile], pl.FP32)
                gamma = pl.cast(pl.reshape(scale[col:col + tile], [1, tile]), pl.FP32)
                g = pl.mul(gradient[row:row + 1, col:col + tile], gamma)
                dx = pl.mul(pl.sub(g, pl.mul(x, correction)), inverse * (1.0 / streams))
                for stream in pl.unroll(streams):
                    output_row = row * streams + stream
                    output[output_row:output_row + 1, col:col + tile] = dx
        return output

    @pl.jit.host
    def entry(activation: pl.Tensor, scale: pl.Tensor, gradient: pl.Tensor, output: pl.Out[pl.Tensor]):
        norm_backward(activation, scale, gradient, output, device=0)

    return norm_backward if device_entry else entry
