"""Capture the exact mixed Attention boundary without replaying TP communication."""


def packed_attention_tape_shapes(block, last_layers, hidden, copies):
    return {
        "attention_tape_residual": ((max(1, last_layers * block * copies), hidden), "float32"),
        "attention_tape_mixes": ((max(1, last_layers * block), 32), "float32"),
        "attention_tape_delta": ((max(1, last_layers * block), hidden), "bfloat16"),
    }


def packed_wo_tape_shapes(block, last_layers, model, tp):
    rows = max(1, last_layers * block)
    return {
        "attention_tape_wo_input": ((rows, model.num_attention_heads // tp * model.head_dim), "bfloat16"),
        "attention_tape_wo_latent": ((rows, model.o_groups // tp * model.o_lora_rank), "bfloat16"),
    }


def packed_core_tape_shapes(block, last_layers, model, tp):
    rows = max(block, last_layers * block)
    heads, dim = model.num_attention_heads // tp, model.head_dim
    return {
        "attention_tape_normed": ((rows, model.hidden_size), "bfloat16"),
        "attention_tape_query": ((rows * heads, dim), "bfloat16"),
        "attention_tape_temporary": ((rows, dim), "bfloat16"),
        "attention_tape_context": ((rows * heads, dim), "float32"),
        "attention_tape_lse": ((rows, heads), "float32"),
        "attention_tape_selection": ((rows, model.index_topk), "int32"),
        "attention_tape_lengths": ((rows,), "int32"),
    }


def packed_qkv_tape_shapes(block, last_layers, model):
    rows = max(1, last_layers * block)
    return {
        "attention_tape_q_normed": ((rows, model.q_lora_rank), "bfloat16"),
        "attention_tape_qkv_raw": ((rows, model.q_lora_rank + model.head_dim), "bfloat16"),
    }


def make_packed_qkv_capture(block, last_layers, model):
    import pypto.language as pl

    latent, joint = model.q_lora_rank, model.q_lora_rank + model.head_dim
    shapes = packed_qkv_tape_shapes(block, last_layers, model)

    @pl.jit.inline(auto_scope=False)
    def save(
        low: pl.Tensor[[block, latent], pl.BF16], raw: pl.Tensor[[block, joint], pl.BF16],
        saved_low: pl.Out[pl.Tensor[shapes["attention_tape_q_normed"][0], pl.BF16]],
        saved_raw: pl.Out[pl.Tensor[shapes["attention_tape_qkv_raw"][0], pl.BF16]],
        owner: pl.Scalar[pl.BOOL], slot: pl.Scalar[pl.INT32],
    ):
        if slot >= 0:
            for row in pl.spmd(block, name_hint="orthrus_save_projection_operands"):
                dest = slot * block + row
                saved_low[dest:dest + 1, :] = low[row:row + 1, :]
                if owner:
                    saved_raw[dest:dest + 1, :] = raw[row:row + 1, :]

    return save


def make_packed_attention_core_capture(block, last_layers, model, tp):
    """Retain exact forward operands in the token-major layout used by the VJP."""
    import pypto.language as pl

    heads, dim = model.num_attention_heads // tp, model.head_dim
    hidden, topk = model.hidden_size, model.index_topk
    rows = block * heads
    shapes = packed_core_tape_shapes(block, last_layers, model, tp)
    saved_rows = shapes["attention_tape_lengths"][0][0]

    @pl.jit.inline(auto_scope=False)
    def save(
        normed: pl.Tensor[[block, hidden], pl.BF16],
        query: pl.Tensor[[block, heads * dim], pl.BF16],
        temporary: pl.Tensor[[block, dim], pl.BF16],
        context: pl.Tensor[[rows, dim], pl.FP32],
        lse: pl.Tensor[[rows, 1], pl.FP32],
        selection: pl.Tensor[[block, topk], pl.INT32],
        lengths: pl.Tensor[[block], pl.INT32],
        saved_normed: pl.Out[pl.Tensor[[saved_rows, hidden], pl.BF16]],
        saved_query: pl.Out[pl.Tensor[[saved_rows * heads, dim], pl.BF16]],
        saved_temporary: pl.Out[pl.Tensor[[saved_rows, dim], pl.BF16]],
        saved_context: pl.Out[pl.Tensor[[saved_rows * heads, dim], pl.FP32]],
        saved_lse: pl.Out[pl.Tensor[[saved_rows, heads], pl.FP32]],
        saved_selection: pl.Out[pl.Tensor[[saved_rows, topk], pl.INT32]],
        saved_lengths: pl.Out[pl.Tensor[[saved_rows], pl.INT32]],
        swa: pl.Scalar[pl.BOOL], slot: pl.Scalar[pl.INT32],
    ):
        if slot >= 0:
            first = slot * block
            flat_query = pl.reshape(query, [rows, dim])
            with pl.spmd(rows, name_hint="orthrus_save_attention_context"):
                row = pl.tile.get_block_idx()
                token, head = row // heads, row % heads
                source = row
                if swa:
                    source = head * block + token
                dest = first * heads + row
                saved_query[dest:dest + 1, :] = flat_query[row:row + 1, :]
                saved_context[dest:dest + 1, :] = context[source:source + 1, :]
            with pl.spmd(block, name_hint="orthrus_save_attention_inputs"):
                token = pl.tile.get_block_idx()
                dest = first + token
                saved_normed[dest:dest + 1, :] = normed[token:token + 1, :]
                saved_temporary[dest:dest + 1, :] = temporary[token:token + 1, :]
                values = pl.tile.full([1, heads], dtype=pl.FP32, value=0.)
                for head in pl.range(heads):
                    source = token * heads + head
                    if swa:
                        source = head * block + token
                    pl.tile.write(values, [0, head], pl.read(lse, [source, 0]))
                pl.store(values, [dest, 0], saved_lse)
                if not swa:
                    saved_selection[dest:dest + 1, :] = selection[token:token + 1, :]
            length_input = pl.reshape(lengths, [1, block])
            length_output = pl.reshape(saved_lengths, [1, saved_rows])
            for lane in pl.spmd(1, name_hint="orthrus_save_attention_lengths"):
                length_row = pl.tile.full([1, block], dtype=pl.INT32, value=0)
                if not swa:
                    length_row = pl.load(length_input, [0, 0], [1, block])
                pl.store(length_row, [lane, first], length_output)

    return save


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
