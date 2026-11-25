"""VJPs of draft Wo-B, grouped Wo-A and inverse RoPE in resident layouts."""

from dataclasses import dataclass
from math import prod
from time import perf_counter


def make_packed_wo_input_vjp(block, model, tp, *, compensated=True, world=1, parameters=False):
    import pypto.language as pl
    import pypto.language.distributed as pld

    heads, dim, groups = model.num_attention_heads // tp, model.head_dim, model.o_groups // tp
    hidden, rank = model.hidden_size, model.o_lora_rank
    width, middle = heads * dim, groups * rank
    group_width, tail = width // groups, model.qk_rope_head_dim
    c0 = 16
    b_shape = (hidden // c0 * middle, c0)
    a_shape = (groups, rank // c0 * group_width, c0)

    def transpose_nz(group_count, source_width, target_width, hint):
        kt, nt = 128, 64
        blocks = target_width // nt

        @pl.jit.inline(auto_scope=False)
        def vjp(gradient: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active: pl.Scalar[pl.INT32]):
            packed = pl.reshape(weight, [group_count * (source_width // c0) * target_width, c0])
            with pl.spmd(((active + 15) // 16) * group_count * blocks, name_hint=hint):
                task = pl.tile.get_block_idx()
                row = task // (group_count * blocks) * 16
                group = task // blocks % group_count
                col = task % blocks * nt
                acc = pl.create_tensor([16, nt], dtype=pl.FP32)
                for step in pl.range(source_width // kt):
                    begin = step * kt
                    g = pl.slice(gradient, [16, kt], [row, group * source_width + begin])
                    high = pl.cast(g, pl.BF16, mode="rint")
                    tile = pl.create_tensor([nt, kt], dtype=pl.BF16)
                    for frag in pl.range(kt // c0):
                        base = (group * (source_width // c0) + begin // c0 + frag) * target_width + col
                        line = pl.slice(packed, [nt, c0], [base, 0])
                        tile[:, frag * c0:(frag + 1) * c0] = line
                    acc = pl.matmul_acc(acc, high, tile, b_trans=True, init_cond=(step == 0))
                    if compensated:
                        low = pl.cast(pl.sub(g, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
                        acc = pl.matmul_acc(acc, low, tile, b_trans=True, init_cond=False)
                target = group * target_width + col
                output[row:row + 16, target:target + nt] = acc
            return output

        return vjp

    wo_b_vjp = transpose_nz(1, hidden, middle, "orthrus_packed_wo_b_nz_vjp")
    wo_a_vjp = transpose_nz(groups, rank, group_width, "orthrus_packed_wo_a_nz_vjp")

    def outer_nz(group_count, iw, ow, hint):
        row_tile = 128
        col_tiles = ow // c0
        tasks = group_count * (iw // row_tile) * col_tiles

        @pl.jit.inline(auto_scope=False)
        def vjp(x: pl.Tensor, seed: pl.Tensor, output: pl.Tensor, active: pl.Scalar[pl.INT32]):
            out = pl.reshape(output, [group_count * col_tiles * iw, c0])
            high = pl.create_tensor([block, group_count * ow], dtype=pl.BF16)
            low = pl.create_tensor([block, group_count * ow], dtype=pl.BF16)
            with pl.spmd(((active + 15) // 16) * 16, name_hint="orthrus_packed_wo_seed_split"):
                row = pl.tile.get_block_idx()
                for column in pl.range(0, group_count * ow, c0 * 4):
                    if row < active:
                        g = pl.load(seed, [row, column], [1, c0 * 4])
                        rounded = pl.cast(g, pl.BF16, mode="rint")
                        pl.store(rounded, [row, column], high)
                        pl.store(pl.cast(pl.sub(g, pl.cast(rounded, pl.FP32)), pl.BF16, mode="rint"), [row, column], low)
                    else:
                        zero = pl.tile.full([1, c0 * 4], dtype=pl.BF16, value=0.)
                        pl.store(zero, [row, column], high)
                        pl.store(zero, [row, column], low)

            with pl.spmd(tasks, name_hint=hint):
                task = pl.tile.get_block_idx()
                group = task // (col_tiles * (iw // row_tile))
                row = task % (iw // row_tile) * row_tile
                column_block = task // (iw // row_tile) % col_tiles
                col = group * ow + column_block * c0
                result = pl.create_tensor([row_tile, c0], dtype=pl.FP32)
                for tile in pl.range((active + 15) // 16):
                    start = tile * 16
                    inputs = pl.slice(x, [16, row_tile], [start, group * iw + row])
                    primary = pl.slice(high, [16, c0], [start, col])
                    correction = pl.slice(low, [16, c0], [start, col])
                    result = pl.matmul_acc(result, inputs, primary, a_trans=True, init_cond=(tile == 0))
                    result = pl.matmul_acc(result, inputs, correction, a_trans=True, init_cond=False)
                dest = (group * col_tiles + column_block) * iw + row
                out[dest:dest + row_tile, :] = result
            return output

        return vjp

    @pl.jit.inline(auto_scope=False)
    def reverse_input(
        delta: pl.Tensor[[block, hidden], pl.FP32],
        wo_a: pl.Tensor[a_shape, pl.BF16],
        wo_b: pl.Tensor[b_shape, pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32],
        sine: pl.Tensor[[block, tail // 2], pl.FP32],
        context: pl.Tensor[[block * heads, dim], pl.FP32],
        hidden_gradient: pl.Tensor[[block, middle], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.create_tensor([block, width], dtype=pl.FP32)
        wo_b_vjp(delta, wo_b, hidden_gradient, active)
        wo_a_vjp(hidden_gradient, wo_a, rotated, active)
        x = pl.reshape(rotated, [block * heads, dim])
        with pl.spmd(block * heads, name_hint="orthrus_packed_wo_inverse_rope_vjp"):
            row = pl.tile.get_block_idx()
            token = row // heads
            if token < active:
                vector = pl.load(x, [row, 0], [1, dim])
                rotary = pl.tile.slice(vector, [1, tail], [0, dim - tail])
                selector = pl.cast(pl.tile.arange(0, [1, tail], dtype=pl.INT32), pl.FP32)
                pairs = pl.cast(pl.mul(selector, .5), pl.INT32, mode="trunc")
                parity = pl.sub(selector, pl.mul(pl.cast(pairs, pl.FP32), 2.))
                swap = pl.cast(pl.sub(pl.add(selector, 1.), pl.mul(parity, 2.)), pl.INT32)
                scratch = pl.create_tile([1, tail], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                swapped = pl.tile.gather(rotary, swap, scratch)
                c = pl.tile.gather(pl.load(cosine, [token, 0], [1, tail // 2]), pairs, scratch)
                s = pl.tile.gather(pl.load(sine, [token, 0], [1, tail // 2]), pairs, scratch)
                s = pl.mul(s, pl.sub(pl.mul(parity, 2.), 1.))
                value = pl.add(pl.mul(rotary, c), pl.mul(swapped, s))
                pl.store(pl.tile.slice(vector, [1, dim - tail], [0, 0]), [row, 0], context)
                pl.store(value, [row, dim - tail], context)
            else:
                context[row:row + 1, :] = pl.full([1, dim], dtype=pl.FP32, value=0.)
        return context

    if parameters:
        weight_a_vjp = outer_nz(groups, group_width, rank, "orthrus_packed_wo_a_weight_vjp")
        weight_b_vjp = outer_nz(1, middle, hidden, "orthrus_packed_wo_b_weight_vjp")

        @pl.jit
        def reverse_all(
            delta: pl.Tensor[[block, hidden], pl.FP32],
            wo_a: pl.Tensor[a_shape, pl.BF16],
            wo_b: pl.Tensor[b_shape, pl.BF16],
            cosine: pl.Tensor[[block, tail // 2], pl.FP32],
            sine: pl.Tensor[[block, tail // 2], pl.FP32],
            rotated: pl.Tensor[[block, width], pl.BF16],
            latent: pl.Tensor[[block, middle], pl.BF16],
            context: pl.Out[pl.Tensor[[block * heads, dim], pl.FP32]],
            grad_a: pl.Out[pl.Tensor[a_shape, pl.FP32]],
            grad_b: pl.Out[pl.Tensor[b_shape, pl.FP32]],
            active: pl.Scalar[pl.INT32],
        ):
            hidden_gradient = pl.create_tensor([block, middle], dtype=pl.FP32)
            reverse_input(delta, wo_a, wo_b, cosine, sine, context, hidden_gradient, active)
            weight_a_vjp(rotated, hidden_gradient, grad_a, active)
            weight_b_vjp(latent, delta, grad_b, active)
            return context, grad_a, grad_b

        @pl.jit.host
        def parameter_dispatch(
            delta: pl.Tensor[[world, block, hidden], pl.FP32],
            wo_a: pl.Tensor[[world, *a_shape], pl.BF16],
            wo_b: pl.Tensor[[world, *b_shape], pl.BF16],
            cosine: pl.Tensor[[world, block, tail // 2], pl.FP32],
            sine: pl.Tensor[[world, block, tail // 2], pl.FP32],
            rotated: pl.Tensor[[world, block, width], pl.BF16],
            latent: pl.Tensor[[world, block, middle], pl.BF16],
            context: pl.Out[pl.Tensor[[world, block * heads, dim], pl.FP32]],
            grad_a: pl.Out[pl.Tensor[[world, *a_shape], pl.FP32]],
            grad_b: pl.Out[pl.Tensor[[world, *b_shape], pl.FP32]],
            owner: pl.Scalar[pl.INT32],
            active: pl.Scalar[pl.INT32],
        ):
            for idx in pl.range(pld.world_size()):
                if idx >= owner and idx < owner + tp:
                    reverse_all(delta[idx], wo_a[idx], wo_b[idx], cosine[idx], sine[idx],
                                rotated[idx], latent[idx], context[idx], grad_a[idx], grad_b[idx], active, device=idx)
        return parameter_dispatch

    @pl.jit
    def reverse(
        delta: pl.Tensor[[block, hidden], pl.FP32],
        wo_a: pl.Tensor[a_shape, pl.BF16],
        wo_b: pl.Tensor[b_shape, pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32],
        sine: pl.Tensor[[block, tail // 2], pl.FP32],
        context: pl.Out[pl.Tensor[[block * heads, dim], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        hidden_gradient = pl.create_tensor([block, middle], dtype=pl.FP32)
        return reverse_input(delta, wo_a, wo_b, cosine, sine, context, hidden_gradient, active)

    if world > 1:
        @pl.jit.host
        def dispatch(
            delta: pl.Tensor[[world, block, hidden], pl.FP32],
            wo_a: pl.Tensor[[world, *a_shape], pl.BF16],
            wo_b: pl.Tensor[[world, *b_shape], pl.BF16],
            cosine: pl.Tensor[[world, block, tail // 2], pl.FP32],
            sine: pl.Tensor[[world, block, tail // 2], pl.FP32],
            context: pl.Out[pl.Tensor[[world, block * heads, dim], pl.FP32]],
            owner: pl.Scalar[pl.INT32],
            active: pl.Scalar[pl.INT32],
        ):
            for rank in pl.range(pld.world_size()):
                if rank >= owner and rank < owner + tp:
                    reverse(delta[rank], wo_a[rank], wo_b[rank], cosine[rank], sine[rank], context[rank],
                            active, device=rank)
        return dispatch

    @pl.jit.host
    def entry(
        delta: pl.Tensor[[block, hidden], pl.FP32],
        wo_a: pl.Tensor[a_shape, pl.BF16],
        wo_b: pl.Tensor[b_shape, pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32],
        sine: pl.Tensor[[block, tail // 2], pl.FP32],
        context: pl.Out[pl.Tensor[[block * heads, dim], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        reverse(delta, wo_a, wo_b, cosine, sine, context, active, device=0)
    return entry


@dataclass(frozen=True)
class PackedWoInputGradient:
    owner: int
    layer: int
    rows: int
    context: object
    residual: object
    seconds: float
    parameter_gradients: dict | None = None


class PackedWoInputVJP:
    """Run one TP group on its existing Wo banks and the target's immutable RoPE tables."""

    def __init__(self, proposer, graph, model, *, parameters=False):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.ranks = tuple(range(proposer.world))
        self.block, self.tp = proposer.block, proposer.tp
        self.banks = proposer.ranks
        groups, heads = model.o_groups // self.tp, model.num_attention_heads // self.tp
        dim, hidden, rank = model.head_dim, model.hidden_size, model.o_lora_rank
        self.wo_shapes = ((groups, rank // 16 * (heads * dim // groups), 16),
                          (hidden // 16 * groups * rank, 16))
        self.rope_shape = (self.block, model.qk_rope_head_dim // 2)
        shape = (self.block * heads, dim)
        self.context = self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=idx)
                                    for idx in self.ranks])
        self.gradients = ()
        if parameters:
            self.first_layer = model.num_hidden_layers - proposer.last_layers
            self.tape_shapes = ((self.block, heads * dim), (self.block, groups * rank))
            self.gradients = tuple(self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=idx)
                                                for idx in self.ranks]) for shape in self.wo_shapes)

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.ranks)

    def _view(self, bank, shape, offset):
        offset *= bank.dtype.itemsize
        return self.device_tensor(bank.data_ptr + offset, shape, bank.dtype, buffer=bank.buffer,
                                  byte_offset=bank.byte_offset + offset)

    def __call__(self, student, seed):
        if seed.owner != student.owner or seed.rows > student.rows:
            raise ValueError("Wo backward requires the student capture that produced the Attention seed")
        layer = seed.layer
        weights = [self._stack([self._view(bank[name], shape, layer * prod(shape)) for bank in self.banks])
                   for name, shape in zip(("draft_wo_a", "draft_wo_b"), self.wo_shapes, strict=True)]
        profile = int(layer >= 2)
        rope = [self._stack([self._view(bank[name], self.rope_shape,
                                       (profile * bank[name].shape[1] + student.position) * self.rope_shape[1])
                             for bank in self.banks])
                for name in ("freqs_cos", "freqs_sin")]
        start = perf_counter()
        inputs = (seed.delta, *weights, *rope)
        outputs = (self.context,)
        parameter_gradients = None
        if self.gradients:
            slot = layer - self.first_layer
            if slot < 0:
                raise ValueError("Wo parameter backward requires the saved layer")
            tape = []
            for name, shape in zip(("attention_tape_wo_input", "attention_tape_wo_latent"),
                                   self.tape_shapes, strict=True):
                tape.append(self._stack([
                    self._view(student.ffn_tape.get(idx, bank)[name], shape, slot * prod(shape))
                    for idx, bank in enumerate(self.banks)
                ]))
            inputs += tuple(tape)
            outputs += self.gradients
            parameter_gradients = dict(zip(("draft_wo_a", "draft_wo_b"), self.gradients, strict=True))
        self.runtime.run(self.graph, *inputs, *outputs, seed.owner, seed.rows, config=self.config)
        return PackedWoInputGradient(seed.owner, layer, seed.rows, self.context, seed.residual,
                                     perf_counter() - start, parameter_gradients)
