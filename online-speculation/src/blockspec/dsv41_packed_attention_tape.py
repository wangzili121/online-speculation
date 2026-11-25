"""Capture the exact mixed Attention boundary without replaying TP communication."""


def packed_attention_tape_shapes(block, last_layers, hidden, copies):
    return {
        "attention_tape_residual": ((max(1, last_layers * block * copies), hidden), "float32"),
        "attention_tape_mixes": ((max(1, last_layers * block), 32), "float32"),
        "attention_tape_delta": ((max(1, last_layers * block), hidden), "bfloat16"),
    }


def make_packed_attention_post_capture(block, last_layers, model, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.hc import hc_decode_mixes, hc_post_collapse
    from models.deepseek_v4_1_flash_a3.o_proj import prefill_tp_output_all_reduce

    hidden, copies = model.hidden_size, model.hc_mult
    shapes = packed_attention_tape_shapes(block, last_layers, hidden, copies)

    @pl.jit.inline(auto_scope=False)
    def mix(
        x_hc: pl.Tensor[[block, copies, hidden], pl.FP32],
        incoming: pl.Tensor[[block, copies], pl.FP32],
        contribution: pl.Tensor[[block, hidden], pl.BF16],
        function: pl.Tensor,
        scale: pl.Tensor,
        bias: pl.Tensor,
        output_window: pld.DistributedTensor,
        arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
        output: pl.Tensor[[block, copies, hidden], pl.FP32],
        next_premix: pl.Tensor[[block, copies], pl.FP32],
        collapsed: pl.Tensor[[block, hidden], pl.BF16],
        saved_residual: pl.Out[pl.Tensor[shapes["attention_tape_residual"][0], pl.FP32]],
        saved_mixes: pl.Out[pl.Tensor[shapes["attention_tape_mixes"][0], pl.FP32]],
        saved_delta: pl.Out[pl.Tensor[shapes["attention_tape_delta"][0], pl.BF16]],
        capture_owner: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        post = pl.create_tensor([block, copies], dtype=pl.FP32)
        residual_mix = pl.create_tensor([block, copies, copies], dtype=pl.FP32)
        hc_decode_mixes(x_hc, function, scale, bias, next_premix, post, residual_mix, block)
        reduced = pl.create_tensor([block, hidden], dtype=pl.BF16)
        prefill_tp_output_all_reduce(contribution, output_window, arrived, reduced, rank // tp * tp, rank % tp,
                                     block, epoch, owner_rows=False)
        hc_post_collapse(reduced, x_hc, post, residual_mix, next_premix, output, collapsed)
        if last_layers > 0:
            if rank == capture_owner and slot >= 0:
                flat = pl.reshape(x_hc, [block * copies, hidden])
                coefficients = pl.reshape(residual_mix, [block, copies * copies])
                with pl.spmd(block, name_hint="orthrus_capture_attention_post"):
                    token = pl.tile.get_block_idx()
                    target = slot * block + token
                    saved_delta[target:target + 1, :] = reduced[token:token + 1, :]
                    mix_row = pl.tile.full([1, 32], dtype=pl.FP32, value=0.)
                    for stream in pl.range(copies):
                        source = token * copies + stream
                        dest = target * copies + stream
                        saved_residual[dest:dest + 1, :] = flat[source:source + 1, :]
                        pl.tile.write(mix_row, [0, stream], pl.read(incoming, [token, stream]))
                        pl.tile.write(mix_row, [0, copies + stream], pl.read(post, [token, stream]))
                    for feature in pl.range(copies * copies):
                        pl.tile.write(mix_row, [0, 2 * copies + feature], pl.read(coefficients, [token, feature]))
                    pl.store(mix_row, [target, 0], saved_mixes)
        return output, next_premix, collapsed

    return mix
