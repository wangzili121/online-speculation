"""Resident gradient of the 1122 draft Q-A, Q-B, KV, and RMS weights."""

from dataclasses import dataclass
from math import prod
from time import perf_counter


def make_packed_qkv_vjp(block, model, *, world, tp):
    """Reduce shared activation gradients before forming the replicated weight gradients."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    hidden, latent, dim = model.hidden_size, model.q_lora_rank, model.head_dim
    heads = model.num_attention_heads // tp
    width, joint, tail = heads * dim, latent + dim, model.qk_rope_head_dim
    epsilon = model.rms_norm_eps

    @pl.jit.inline
    def rope_vjp(seed: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, row: pl.Scalar[pl.INT32],
                 token: pl.Scalar[pl.INT32]):
        rotated = pl.load(seed, [row, dim - tail], [1, tail])
        index = pl.cast(pl.tile.arange(0, [1, tail], dtype=pl.INT32), pl.FP32)
        pairs = pl.cast(pl.mul(index, .5), pl.INT32, mode="trunc")
        parity = pl.sub(index, pl.mul(pl.cast(pairs, pl.FP32), 2.))
        other = pl.cast(pl.sub(pl.add(index, 1.), pl.mul(parity, 2.)), pl.INT32)
        scratch = pl.create_tile([1, tail], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        swapped = pl.tile.gather(rotated, other, scratch)
        c = pl.tile.gather(pl.load(cosine, [token, 0], [1, tail // 2]), pairs, scratch)
        s = pl.tile.gather(pl.load(sine, [token, 0], [1, tail // 2]), pairs, scratch)
        return pl.add(pl.mul(rotated, c), pl.mul(swapped, pl.mul(s, pl.add(pl.mul(parity, -2.), 1.))))

    def transpose_matmul(source, target, hint):
        kt, nt = 128, 64
        cols = target // nt

        @pl.jit.inline(auto_scope=False)
        def project(
            hi: pl.Tensor, lo: pl.Tensor, weights: pl.Tensor, output: pl.Tensor,
            live: pl.Scalar[pl.INT32],
        ):
            with pl.spmd((block // 16) * cols, name_hint=hint):
                task = pl.tile.get_block_idx()
                row, col = (task // cols) * 16, (task % cols) * nt
                if row < live:
                    result = pl.create_tensor([16, nt], dtype=pl.FP32)
                    for chunk in pl.range(source // kt):
                        start = chunk * kt
                        w = pl.slice(weights, [nt, kt], [col, start])
                        primary = pl.slice(hi, [16, kt], [row, start])
                        correction = pl.slice(lo, [16, kt], [row, start])
                        result = pl.matmul_acc(result, primary, w, b_trans=True, init_cond=(chunk == 0))
                        result = pl.matmul_acc(result, correction, w, b_trans=True)
                    output[row:row + 16, col:col + nt] = result
                else:
                    output[row:row + 16, col:col + nt] = pl.full([16, nt], dtype=pl.FP32, value=0.)
            return output

        return project

    def parameter_matmul(in_width, out_width, hint):
        mt, nt = 64, 128
        columns = out_width // nt

        @pl.jit.inline(auto_scope=False)
        def project(
            x: pl.Tensor, hi: pl.Tensor, lo: pl.Tensor, output: pl.Tensor,
            live: pl.Scalar[pl.INT32],
        ):
            with pl.spmd((in_width // mt) * columns, name_hint=hint):
                task = pl.tile.get_block_idx()
                row, col = (task // columns) * mt, (task % columns) * nt
                result = pl.create_tensor([mt, nt], dtype=pl.FP32)
                for chunk in pl.range((live + 15) // 16):
                    start = chunk * 16
                    data = pl.slice(x, [16, mt], [start, row])
                    primary = pl.slice(hi, [16, nt], [start, col])
                    correction = pl.slice(lo, [16, nt], [start, col])
                    result = pl.matmul_acc(result, data, primary, a_trans=True, init_cond=(chunk == 0))
                    result = pl.matmul_acc(result, data, correction, a_trans=True)
                output[row:row + mt, col:col + nt] = result
            return output

        return project

    query_input_vjp = transpose_matmul(width, latent, "orthrus_q_b_input_vjp")
    joined_input_vjp = transpose_matmul(joint, hidden, "orthrus_q_a_kv_input_vjp")
    query_weight_vjp = parameter_matmul(latent, width, "orthrus_q_b_weight_vjp")
    joined_weight_vjp = parameter_matmul(hidden, joint, "orthrus_q_a_kv_weight_vjp")

    @pl.jit.inline(auto_scope=False)
    def local_projection(
        dq: pl.Tensor[[block * heads, dim], pl.FP32],
        dkv: pl.Tensor[[block, dim], pl.FP32],
        q_b: pl.Tensor[[latent, width], pl.BF16],
        q_normed: pl.Tensor[[block, latent], pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32],
        sine: pl.Tensor[[block, tail // 2], pl.FP32],
        joint_seed: pl.Out[pl.Tensor[[block, joint], pl.FP32]],
        q_weight_grad: pl.Out[pl.Tensor[[latent, width], pl.FP32]],
        active_q: pl.Scalar[pl.INT32],
    ):
        q_hi = pl.create_tensor([block * heads, dim], dtype=pl.BF16)
        q_lo = pl.create_tensor([block * heads, dim], dtype=pl.BF16)
        for row in pl.spmd(block * heads, name_hint="orthrus_query_rope_vjp"):
            token = row // heads
            if token < active_q:
                main = pl.load(dq, [row, 0], [1, dim - tail])
                rope = rope_vjp(dq, cosine, sine, row, token)
                q_head = pl.cast(main, pl.BF16, mode="rint")
                q_tail = pl.cast(rope, pl.BF16, mode="rint")
                pl.store(q_head, [row, 0], q_hi)
                pl.store(q_tail, [row, dim - tail], q_hi)
                pl.store(pl.cast(pl.sub(main, pl.cast(q_head, pl.FP32)), pl.BF16, mode="rint"), [row, 0], q_lo)
                pl.store(pl.cast(pl.sub(rope, pl.cast(q_tail, pl.FP32)), pl.BF16, mode="rint"), [row, dim - tail], q_lo)
            else:
                zero = pl.tile.full([1, dim], dtype=pl.BF16, value=0.)
                pl.store(zero, [row, 0], q_hi)
                pl.store(zero, [row, 0], q_lo)
        high = pl.reshape(q_hi, [block, width])
        low = pl.reshape(q_lo, [block, width])
        joint_seed = query_input_vjp(high, low, q_b, joint_seed, active_q)
        for row in pl.spmd(block, name_hint="orthrus_temporary_kv_rope_vjp"):
            main = pl.load(dkv, [row, 0], [1, dim - tail])
            rope = rope_vjp(dkv, cosine, sine, row, row)
            pl.store(main, [row, latent], joint_seed)
            pl.store(rope, [row, latent + dim - tail], joint_seed)
        q_weight_grad = query_weight_vjp(q_normed, high, low, q_weight_grad, active_q)
        return joint_seed, q_weight_grad

    def rms_vjp(columns, offset, hint):
        tile = 128

        @pl.jit.inline(auto_scope=False)
        def differentiate(
            raw: pl.Tensor[[block, joint], pl.BF16],
            incoming: pl.Tensor[[block, joint], pl.FP32],
            weight: pl.Tensor[[1, columns], pl.BF16],
            hi: pl.Out[pl.Tensor[[block, joint], pl.BF16]],
            lo: pl.Out[pl.Tensor[[block, joint], pl.BF16]],
            dgamma: pl.Out[pl.Tensor[[1, columns], pl.FP32]],
        ):
            partials = pl.create_tensor([block, columns], dtype=pl.FP32)
            for row in pl.spmd(block, name_hint=hint):
                x = pl.cast(pl.load(raw, [row, offset], [1, columns]), pl.FP32)
                d = pl.load(incoming, [row, offset], [1, columns])
                g = pl.cast(pl.load(weight, [0, 0], [1, columns]), pl.FP32)
                weighted = pl.mul(d, g)
                reduction = pl.create_tile([8, columns // 8], dtype=pl.FP32)
                square_rows = pl.reshape(pl.mul(x, x), [8, columns // 8])
                squares_by_row = pl.row_sum(square_rows, tmp_tile=reduction)
                squares = pl.tile.read(squares_by_row, [0, 0])
                for segment in pl.unroll(1, 8):
                    squares = squares + pl.tile.read(squares_by_row, [segment, 0])
                arg = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
                arg = pl.add(arg, squares * (1. / columns) + epsilon)
                inverse = pl.create_tile([1, 8], dtype=pl.FP32)
                inv = pl.tile.read(pl.tile.rsqrt(arg, inverse, high_precision=True), [0, 0])
                dot_rows = pl.reshape(pl.mul(x, weighted), [8, columns // 8])
                dot_by_row = pl.row_sum(dot_rows, tmp_tile=reduction)
                dot = pl.tile.read(dot_by_row, [0, 0])
                for segment in pl.unroll(1, 8):
                    dot = dot + pl.tile.read(dot_by_row, [segment, 0])
                correction = dot * (1. / columns) * inv * inv * inv
                dx = pl.sub(pl.mul(weighted, inv), pl.mul(x, correction))
                rounded = pl.cast(dx, pl.BF16, mode="rint")
                pl.store(rounded, [row, offset], hi)
                pl.store(pl.cast(pl.sub(dx, pl.cast(rounded, pl.FP32)), pl.BF16, mode="rint"), [row, offset], lo)
                pl.store(pl.mul(pl.mul(x, inv), d), [row, 0], partials)
            for panel in pl.spmd(columns // tile, name_hint="orthrus_qkv_norm_weight_vjp"):
                col = panel * tile
                value = pl.full([1, tile], dtype=pl.FP32, value=0.)
                for row in pl.range(block):
                    value = pl.add(value, pl.slice(partials, [1, tile], [row, col]))
                dgamma[0:1, col:col + tile] = value
            return hi, lo, dgamma

        return differentiate

    q_rms = rms_vjp(latent, 0, "orthrus_query_norm_input_vjp")
    kv_rms = rms_vjp(dim, latent, "orthrus_kv_norm_input_vjp")

    @pl.jit.inline(auto_scope=False)
    def project_and_send(
        dq: pl.Tensor[[block * heads, dim], pl.FP32], dkv: pl.Tensor[[block, dim], pl.FP32],
        q_normed: pl.Tensor[[block, latent], pl.BF16], q_b: pl.Tensor[[latent, width], pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32], sine: pl.Tensor[[block, tail // 2], pl.FP32],
        dq_b: pl.Out[pl.Tensor[[latent, width], pl.FP32]],
        peer_seeds: pld.DistributedTensor[[tp * block, joint], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], lane: pl.Scalar[pl.INT32],
        active_q: pl.Scalar[pl.INT32],
        notified: pl.Array[1, pl.TASK_ID],
    ):
        projection_seed = pl.create_tensor([block, joint], dtype=pl.FP32)
        local_projection(dq, dkv, q_b, q_normed, cosine, sine, projection_seed, dq_b, active_q)
        with pl.spmd(block, name_hint="orthrus_qkv_tp_seed_publish") as sent:
            row = pl.tile.get_block_idx()
            pld.tensor.put(dst=peer_seeds, peer=owner, src=projection_seed,
                           dst_offsets=[lane * block + row, 0], src_offsets=[row, 0], shape=[1, joint])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False) as announced:
            pld.system.notify(target=ready, peer=owner, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        notified[0] = announced

    @pl.jit
    def peer_vjp(
        dq: pl.Tensor[[block * heads, dim], pl.FP32], dkv: pl.Tensor[[block, dim], pl.FP32],
        q_normed: pl.Tensor[[block, latent], pl.BF16], q_b: pl.Tensor[[latent, width], pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32], sine: pl.Tensor[[block, tail // 2], pl.FP32],
        dq_b: pl.Out[pl.Tensor[[latent, width], pl.FP32]],
        peer_seeds: pld.DistributedTensor[[tp * block, joint], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], lane: pl.Scalar[pl.INT32], active_q: pl.Scalar[pl.INT32],
    ):
        notified = pl.array.create(1, pl.TASK_ID)
        project_and_send(dq, dkv, q_normed, q_b, cosine, sine, dq_b,
                         peer_seeds, ready, owner, lane, active_q, notified)
        return dq_b

    @pl.jit
    def owner_vjp(
        dq: pl.Tensor[[block * heads, dim], pl.FP32], dkv: pl.Tensor[[block, dim], pl.FP32],
        x: pl.Tensor[[block, hidden], pl.BF16], q_normed: pl.Tensor[[block, latent], pl.BF16],
        raw: pl.Tensor[[block, joint], pl.BF16],
        qa_kv: pl.Tensor[[hidden, joint], pl.BF16], q_b: pl.Tensor[[latent, width], pl.BF16],
        q_scale: pl.Tensor[[1, latent], pl.BF16], kv_scale: pl.Tensor[[1, dim], pl.BF16],
        cosine: pl.Tensor[[block, tail // 2], pl.FP32], sine: pl.Tensor[[block, tail // 2], pl.FP32],
        dx: pl.Out[pl.Tensor[[block, hidden], pl.FP32]],
        dqa_kv: pl.Out[pl.Tensor[[hidden, joint], pl.FP32]],
        dq_b: pl.Out[pl.Tensor[[latent, width], pl.FP32]],
        dq_scale: pl.Out[pl.Tensor[[1, latent], pl.FP32]], dkv_scale: pl.Out[pl.Tensor[[1, dim], pl.FP32]],
        peer_seeds: pld.DistributedTensor[[tp * block, joint], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], active_q: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        notified = pl.array.create(1, pl.TASK_ID)
        project_and_send(dq, dkv, q_normed, q_b, cosine, sine, dq_b,
                         peer_seeds, ready, owner, 0, active_q, notified)
        with pl.at(level=pl.Level.CORE_GROUP, deps=[notified], allow_early_resolve=False) as gathered:
            pld.system.wait(signal=ready, offsets=[0, 0], expected=epoch * tp, cmp=pld.WaitCmp.Ge)
        total = pl.create_tensor([block, joint], dtype=pl.FP32)
        with pl.spmd(block, deps=[gathered], name_hint="orthrus_qkv_tp_seed_sum"):
            row = pl.tile.get_block_idx()
            summed = pl.full([1, joint], dtype=pl.FP32, value=0.)
            for peer in pl.range(tp):
                source = peer * block + row
                summed = pl.add(summed, peer_seeds[source:source + 1, :])
            total[row:row + 1, :] = summed
        hi = pl.create_tensor([block, joint], dtype=pl.BF16)
        lo = pl.create_tensor([block, joint], dtype=pl.BF16)
        q_rms(raw, total, q_scale, hi, lo, dq_scale)
        kv_rms(raw, total, kv_scale, hi, lo, dkv_scale)
        joined_input_vjp(hi, lo, qa_kv, dx, block)
        joined_weight_vjp(x, hi, lo, dqa_kv, block)
        return dx, dqa_kv, dq_b, dq_scale, dkv_scale

    dp = world // tp

    @pl.jit.host
    def dispatch(
        dq: pl.Tensor[[world, block * heads, dim], pl.FP32], dkv: pl.Tensor[[world, block, dim], pl.FP32],
        x: pl.Tensor[[dp, block, hidden], pl.BF16], q_normed: pl.Tensor[[world, block, latent], pl.BF16],
        raw: pl.Tensor[[dp, block, joint], pl.BF16],
        qa_kv: pl.Tensor[[dp, hidden, joint], pl.BF16], q_b: pl.Tensor[[world, latent, width], pl.BF16],
        q_scale: pl.Tensor[[dp, 1, latent], pl.BF16], kv_scale: pl.Tensor[[dp, 1, dim], pl.BF16],
        cosine: pl.Tensor[[world, block, tail // 2], pl.FP32], sine: pl.Tensor[[world, block, tail // 2], pl.FP32],
        dx: pl.Out[pl.Tensor[[dp, block, hidden], pl.FP32]],
        dqa_kv: pl.Out[pl.Tensor[[dp, hidden, joint], pl.FP32]],
        dq_b: pl.Out[pl.Tensor[[world, latent, width], pl.FP32]],
        dq_scale: pl.Out[pl.Tensor[[dp, 1, latent], pl.FP32]], dkv_scale: pl.Out[pl.Tensor[[dp, 1, dim], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active_q: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        seed_buffer = pld.alloc_window_buffer([tp * block, joint], dtype=pl.FP32)
        ready_buffer = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        replica = owner // tp
        for rank in pl.range(pld.world_size()):
            seed = pld.window(seed_buffer, [tp * block, joint], dtype=pl.FP32)
            ready = pld.window(ready_buffer, [1, 1], dtype=pl.INT32)
            if rank == owner:
                owner_vjp(dq[rank], dkv[rank], x[replica], q_normed[rank], raw[replica], qa_kv[replica], q_b[rank],
                          q_scale[replica], kv_scale[replica], cosine[rank], sine[rank], dx[replica], dqa_kv[replica],
                          dq_b[rank], dq_scale[replica], dkv_scale[replica], seed, ready, owner, active_q, epoch,
                          device=rank)
            elif rank > owner and rank < owner + tp:
                peer_vjp(dq[rank], dkv[rank], q_normed[rank], q_b[rank], cosine[rank], sine[rank], dq_b[rank],
                         seed, ready, owner, rank - owner, active_q, device=rank)
    return dispatch


@dataclass(frozen=True)
class PackedQKVGradient:
    owner: int
    layer: int
    rows: int
    normalized: object
    residual: object
    parameter_gradients: dict
    seconds: float


class PackedQKVVJP:
    """Bind saved projection operands and reuse the resident draft weights."""

    def __init__(self, proposer, graph, model):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.block, self.tp = proposer.block, proposer.tp
        self.ranks = tuple(range(proposer.world))
        self.owners = self.ranks[::self.tp]
        self.epochs = [0] * len(self.owners)
        self.banks = proposer.ranks
        self.first_layer = model.num_hidden_layers - proposer.last_layers
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.hidden, self.latent, self.dim = model.hidden_size, model.q_lora_rank, model.head_dim
        self.width = model.num_attention_heads // self.tp * self.dim
        self.joint = self.latent + self.dim
        self.rope_shape = (self.block, model.qk_rope_head_dim // 2)
        self.outputs = tuple(self._stack([
            self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank) for rank in ranks
        ], ranks) for shape, ranks in (
            ((self.block, self.hidden), self.owners), ((self.hidden, self.joint), self.owners),
            ((self.latent, self.width), self.ranks), ((1, self.latent), self.owners),
            ((1, self.dim), self.owners),
        ))

    def _stack(self, shards, ranks):
        return self.stacked(shards, (len(shards), *shards[0].shape), ranks)

    def _view(self, bank, shape, elements):
        offset = elements * bank.dtype.itemsize
        return self.device_tensor(bank.data_ptr + offset, shape, bank.dtype, buffer=bank.buffer,
                                  byte_offset=bank.byte_offset + offset)

    def __call__(self, student, seed):
        owner, layer = seed.owner, seed.layer
        if seed.owner != student.owner or seed.rows > student.rows or layer < self.first_layer:
            raise ValueError("Draft projection backward requires the original captured layer")
        slot = layer - self.first_layer
        replica = owner // self.tp
        tape = tuple(self._stack([
            self._view(student.ffn_tape.get(rank, self.banks[rank])[name], shape, slot * prod(shape))
            for rank in ranks
        ], ranks) for name, shape, ranks in (
            ("attention_tape_normed", (self.block, self.hidden), self.owners),
            ("attention_tape_q_normed", (self.block, self.latent), self.ranks),
            ("attention_tape_qkv_raw", (self.block, self.joint), self.owners),
        ))
        weights = tuple(self._stack([
            self._view(self.banks[rank][name], shape, layer * prod(shape)) for rank in ranks
        ], ranks) for name, shape, ranks in (
            ("draft_q_a_kv", (self.hidden, self.joint), self.owners),
            ("draft_q_b", (self.latent, self.width), self.ranks),
            ("draft_q_gamma", (1, self.latent), self.owners),
            ("draft_kv_gamma", (1, self.dim), self.owners),
        ))
        profile = int(layer >= 2)
        phase = tuple(self._stack([
            self._view(bank[name], self.rope_shape,
                       (profile * bank[name].shape[1] + student.position) * self.rope_shape[1])
            for bank in self.banks
        ], self.ranks) for name in ("freqs_cos", "freqs_sin"))
        self.epochs[replica] += 1
        start = perf_counter()
        self.runtime.run(self.graph, seed.query, seed.temporary, *tape, *weights, *phase,
                         *self.outputs, owner, seed.rows, self.epochs[replica], config=self.config)
        normalized, *gradients = self.outputs
        params = dict(zip(("draft_q_a_kv", "draft_q_b", "draft_q_gamma", "draft_kv_gamma"),
                          gradients, strict=True))
        return PackedQKVGradient(owner, layer, self.block, normalized.shards[replica], seed.residual,
                                 {**seed.parameter_gradients, **params}, perf_counter() - start)
