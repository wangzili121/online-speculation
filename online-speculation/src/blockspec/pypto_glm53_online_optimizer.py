"""FP32 AdamW state and a BF16 inference copy for draft MLA parameters."""


def make_adamw_update(rows, columns, *, device_entry=False):
    if rows < 1 or columns < 128 or columns % 128:
        raise ValueError("the packed MLA parameter requires 128-aligned columns")
    import pypto.language as pl

    tile_width = 128
    parts = columns // tile_width

    @pl.jit
    def update(
        gradient: pl.Tensor[[rows, columns], pl.FP32],
        master: pl.InOut[pl.Tensor[[rows, columns], pl.FP32]],
        first_moment: pl.InOut[pl.Tensor[[rows, columns], pl.FP32]],
        second_moment: pl.InOut[pl.Tensor[[rows, columns], pl.FP32]],
        serving: pl.Out[pl.Tensor[[rows, columns], pl.BF16]],
        lr: pl.Scalar[pl.FP32], weight_decay: pl.Scalar[pl.FP32],
        inverse_bias_one: pl.Scalar[pl.FP32], inverse_bias_two: pl.Scalar[pl.FP32],
    ):
        for worker in pl.spmd(48, name_hint="orthrus_glm53_adamw"):
            for task in pl.range(worker, rows * parts, 48):
                row, col = task // parts, task % parts * tile_width
                grad = gradient[row:row + 1, col:col + tile_width]
                first = pl.add(pl.mul(first_moment[row:row + 1, col:col + tile_width], 0.9), pl.mul(grad, 0.1))
                second = pl.add(pl.mul(second_moment[row:row + 1, col:col + tile_width], 0.999),
                                pl.mul(pl.mul(grad, grad), 0.001))
                step = pl.div(pl.mul(first, inverse_bias_one), pl.add(pl.sqrt(pl.mul(second, inverse_bias_two)), 1e-8))
                previous = master[row:row + 1, col:col + tile_width]
                parameter = pl.sub(previous, pl.mul(pl.add(step, pl.mul(previous, weight_decay)), lr))
                first_moment[row:row + 1, col:col + tile_width] = first
                second_moment[row:row + 1, col:col + tile_width] = second
                master[row:row + 1, col:col + tile_width] = parameter
                serving[row:row + 1, col:col + tile_width] = pl.cast(parameter, pl.BF16, mode="rint")
        return master, first_moment, second_moment, serving

    @pl.jit.host
    def entry(gradient: pl.Tensor, master: pl.InOut[pl.Tensor], first_moment: pl.InOut[pl.Tensor],
              second_moment: pl.InOut[pl.Tensor], serving: pl.Out[pl.Tensor],
              lr: pl.Scalar[pl.FP32], weight_decay: pl.Scalar[pl.FP32],
              inverse_bias_one: pl.Scalar[pl.FP32], inverse_bias_two: pl.Scalar[pl.FP32]):
        update(gradient, master, first_moment, second_moment, serving,
               lr, weight_decay, inverse_bias_one, inverse_bias_two, device=0)

    return update if device_entry else entry


def make_mla_output_step():
    """Update the actual draft output projection from an upstream suffix gradient."""
    import pypto.language as pl
    from models.glm5_3_flash.config import D

    from .pypto_glm53_mla import HEADS, VALUE
    from .pypto_glm53_online_attention import make_output_backward

    backward = make_output_backward(device_entry=True)
    update = make_adamw_update(D, HEADS * VALUE, device_entry=True)

    @pl.jit.host
    def step(
        context: pl.Tensor, upstream: pl.Tensor,
        master: pl.InOut[pl.Tensor], first: pl.InOut[pl.Tensor], second: pl.InOut[pl.Tensor],
        serving: pl.InOut[pl.Tensor], weight_gradient: pl.InOut[pl.Tensor], context_gradient: pl.Out[pl.Tensor],
        lr: pl.Scalar[pl.FP32], weight_decay: pl.Scalar[pl.FP32],
        inverse_bias_one: pl.Scalar[pl.FP32], inverse_bias_two: pl.Scalar[pl.FP32],
    ):
        computed_weight_gradient, _ = backward(context, serving, upstream, weight_gradient, context_gradient, device=0)
        update(computed_weight_gradient, master, first, second, serving,
               lr, weight_decay, inverse_bias_one, inverse_bias_two, device=0)

    return step


def make_mla_head_adamw(ranks):
    """Stage an update of the differentiable MLA prefix on every resident rank."""
    import pypto.language as pl
    from .pypto_glm53_head import head_layout

    layout, _ = head_layout()
    length = layout["index_q"][0]
    update = make_adamw_update(1, length, device_entry=True)

    @pl.jit.host
    def entry(
        gradient: pl.Tensor[[ranks, 1, length], pl.FP32],
        master: pl.InOut[pl.Tensor[[ranks, 1, length], pl.FP32]],
        first: pl.InOut[pl.Tensor[[ranks, 1, length], pl.FP32]],
        second: pl.InOut[pl.Tensor[[ranks, 1, length], pl.FP32]],
        staged: pl.Out[pl.Tensor[[ranks, 1, length], pl.BF16]],
        lr: pl.Scalar[pl.FP32], weight_decay: pl.Scalar[pl.FP32],
        inverse_bias_one: pl.Scalar[pl.FP32], inverse_bias_two: pl.Scalar[pl.FP32],
    ):
        for rank in pl.range(ranks):
            update(gradient[rank], master[rank], first[rank], second[rank], staged[rank],
                   lr, weight_decay, inverse_bias_one, inverse_bias_two, device=rank)

    return entry


def make_mla_head_publish(ranks):
    """Copy a completed staged update into the selected live head layer."""
    import pypto.language as pl
    from .pypto_glm53_head import head_layout

    layout, full = head_layout()
    length = layout["index_q"][0]
    tile = 128

    @pl.jit
    def publish(
        staged: pl.Tensor[[1, length], pl.BF16],
        live: pl.InOut[pl.Tensor[[1, full], pl.BF16]],
    ):
        for lane in pl.spmd(48, name_hint="glm_publish_mla"):
            for part in pl.range(lane, length // tile, 48):
                begin = part * tile
                live[:, begin:begin + tile] = staged[:, begin:begin + tile]
        return live

    @pl.jit.host
    def entry(
        staged: pl.Tensor[[ranks, 1, length], pl.BF16],
        live: pl.InOut[pl.Tensor[[ranks, 1, full], pl.BF16]],
    ):
        for rank in pl.range(ranks):
            publish(staged[rank], live[rank], device=rank)

    return entry
