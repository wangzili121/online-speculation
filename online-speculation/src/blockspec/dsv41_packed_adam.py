"""Device-resident Adam and head publication for the packed A3 draft."""

from dataclasses import dataclass
from math import gcd, isfinite, prod, sqrt
from time import perf_counter


@dataclass(frozen=True)
class PackedParameter:
    name: str
    shape: tuple
    replicated: bool = False
    fp32: bool = False

    @property
    def program_key(self):
        return prod(self.shape), self.replicated, self.fp32


def packed_parameters(model, tp):
    hidden, latent, dim = model.hidden_size, model.q_lora_rank, model.head_dim
    heads, groups, rank = model.num_attention_heads // tp, model.o_groups // tp, model.o_lora_rank
    return (
        PackedParameter("draft_q_a_kv", (hidden, latent + dim), True),
        PackedParameter("draft_q_gamma", (1, latent), True),
        PackedParameter("draft_q_b", (latent, heads * dim)),
        PackedParameter("draft_kv_gamma", (1, dim), True),
        PackedParameter("draft_sink", (1, heads), fp32=True),
        PackedParameter("draft_wo_a", (groups, rank // 16 * (heads * dim // groups), 16)),
        PackedParameter("draft_wo_b", (hidden // 16 * groups * rank, 16)),
    )


def make_packed_adam_programs(size, replicated, fp32, *, world, tp):
    """Stage gradients, apply Adam, then send the new execution shards to each DP rank."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    from .pypto_dsv41_optimizer_programs import make_reduce, make_stats

    lanes = 1 if replicated else tp
    sources = world // tp if replicated else world
    tile = gcd(size, 2048) if size > 2048 else size
    rows = size // tile
    packet = gcd(rows, 16)
    execution_dtype = pl.FP32 if fp32 else pl.BF16
    statistics, parts = make_stats((size,), inline=True)
    reduce, groups = make_reduce(parts, inline=True)
    final_reduce, _ = make_reduce(groups, inline=True)

    @pl.jit.inline(auto_scope=False)
    def copy_gradient(source: pl.Tensor[[rows, tile], pl.FP32], state: pl.Tensor[[4, size], pl.FP32]):
        destination = pl.reshape(state[3], [rows, tile])
        for index in pl.spmd(rows // packet, name_hint="orthrus_adam_stage_gradient"):
            row = index * packet
            for offset in pl.range(packet):
                at = row + offset
                destination[at:at + 1, :] = source[at:at + 1, :]

    @pl.jit.inline(auto_scope=False)
    def check_gradient(source: pl.Tensor[[size], pl.FP32], sums: pl.Tensor, flags: pl.Tensor):
        if parts == 1:
            statistics(source, sums, flags)
        else:
            partials = pl.create_tensor([parts, 8], dtype=pl.FP32)
            finite = pl.create_tensor([parts, 8], dtype=pl.FP32)
            statistics(source, partials, finite)
            if groups == 1:
                reduce(partials, finite, sums, flags)
            else:
                totals = pl.create_tensor([groups, 8], dtype=pl.FP32)
                ok = pl.create_tensor([groups, 8], dtype=pl.FP32)
                reduce(partials, finite, totals, ok)
                final_reduce(totals, ok, sums, flags)

    @pl.jit
    def local(
        source: pl.Tensor[[size], pl.FP32], state: pl.InOut[pl.Tensor[[4, size], pl.FP32]],
        sums: pl.Out[pl.Tensor[[1, 8], pl.FP32]], flags: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
    ):
        copy_gradient(pl.reshape(source, [rows, tile]), state)
        check_gradient(source, sums, flags)

    @pl.jit
    def receive(
        gradient: pld.DistributedTensor[[rows, tile], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        state: pl.InOut[pl.Tensor[[4, size], pl.FP32]], sums: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
        flags: pl.Out[pl.Tensor[[1, 8], pl.FP32]], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as arrived:
            pld.system.wait(signal=ready, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        destination = pl.reshape(state[3], [rows, tile])
        with pl.spmd(rows // packet, deps=[arrived], name_hint="orthrus_adam_receive_gradient"):
            row = pl.tile.get_block_idx() * packet
            for offset in pl.range(packet):
                at = row + offset
                destination[at:at + 1, :] = gradient[at:at + 1, :]
        check_gradient(state[3], sums, flags)

    @pl.jit
    def send(
        source: pl.Tensor[[size], pl.FP32], gradient: pld.DistributedTensor[[rows, tile], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32], target: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        values = pl.reshape(source, [rows, tile])
        with pl.spmd(rows // packet, name_hint="orthrus_adam_send_gradient") as sent:
            row = pl.tile.get_block_idx() * packet
            pld.tensor.put(dst=gradient, peer=target, src=values, dst_offsets=[row, 0],
                           src_offsets=[row, 0], shape=[packet, tile])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False):
            pld.system.notify(target=ready, peer=target, offsets=[0, 0], value=epoch, op=pld.NotifyOp.Set)

    @pl.jit.host
    def stage(
        gradient: pl.Tensor[[sources, size], pl.FP32], state: pl.InOut[pl.Tensor[[lanes, 4, size], pl.FP32]],
        sums: pl.Out[pl.Tensor[[lanes, 1, 8], pl.FP32]], flags: pl.Out[pl.Tensor[[lanes, 1, 8], pl.FP32]],
        owner: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        window = pld.alloc_window_buffer([rows, tile], dtype=pl.FP32)
        arrive = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            values = pld.window(window, [rows, tile], dtype=pl.FP32)
            ready = pld.window(arrive, [1, 1], dtype=pl.INT32)
            if rank < lanes:
                if owner == 0:
                    local(gradient[rank], state[rank], sums[rank], flags[rank], device=rank)
                else:
                    receive(values, ready, state[rank], sums[rank], flags[rank], epoch, device=rank)
            elif rank >= owner and rank < owner + lanes:
                source = rank
                if replicated:
                    source = rank // tp
                send(gradient[source], values, ready, rank - owner, epoch, device=rank)

    @pl.jit
    def step_rank(
        execution: pl.Tensor[[size], execution_dtype], state: pl.InOut[pl.Tensor[[4, size], pl.FP32]],
        rate: pl.Scalar[pl.FP32], bias1: pl.Scalar[pl.FP32], bias2: pl.Scalar[pl.FP32],
        clip: pl.Scalar[pl.FP32], initialize: pl.Scalar[pl.INT32],
    ):
        master = pl.reshape(state[0], [rows, tile])
        first = pl.reshape(state[1], [rows, tile])
        second = pl.reshape(state[2], [rows, tile])
        gradient = pl.reshape(state[3], [rows, tile])
        weight = pl.reshape(execution, [rows, tile])
        for index in pl.spmd(rows // packet, name_hint="orthrus_resident_adam"):
            row = index * packet
            for offset in pl.range(packet):
                at = row + offset
                if initialize != 0:
                    starting = pl.slice(weight, [1, tile], [at, 0])
                    if not fp32:
                        starting = pl.cast(starting, pl.FP32)
                    master[at:at + 1, :] = starting
                    first[at:at + 1, :] = pl.full([1, tile], dtype=pl.FP32, value=0.)
                    second[at:at + 1, :] = pl.full([1, tile], dtype=pl.FP32, value=0.)
                else:
                    grad = pl.mul(pl.slice(gradient, [1, tile], [at, 0]), clip)
                    mean = pl.add(pl.mul(pl.slice(first, [1, tile], [at, 0]), .9), pl.mul(grad, 1. - .9))
                    square = pl.mul(grad, grad)
                    variance = pl.add(pl.mul(pl.slice(second, [1, tile], [at, 0]), .999), pl.mul(square, 1. - .999))
                    denominator = pl.add(pl.sqrt(pl.div(variance, bias2)), 1e-8)
                    delta = pl.div(pl.mul(mean, rate), pl.mul(denominator, bias1))
                    master[at:at + 1, :] = pl.sub(pl.slice(master, [1, tile], [at, 0]), delta)
                    first[at:at + 1, :] = mean
                    second[at:at + 1, :] = variance

    @pl.jit.host
    def step(
        execution: pl.Tensor[[world, size], execution_dtype],
        state: pl.InOut[pl.Tensor[[lanes, 4, size], pl.FP32]],
        rate: pl.Scalar[pl.FP32], bias1: pl.Scalar[pl.FP32], bias2: pl.Scalar[pl.FP32],
        clip: pl.Scalar[pl.FP32], initialize: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(lanes):
            step_rank(execution[rank], state[rank], rate, bias1, bias2, clip, initialize, device=rank)

    peer_stride = 1 if replicated else tp

    @pl.jit
    def publish_rank(
        state: pl.Tensor[[4, size], pl.FP32], execution: pl.Out[pl.Tensor[[size], execution_dtype]],
        window: pld.DistributedTensor[[rows, tile], execution_dtype],
        signal: pld.DistributedTensor[[1, 1], pl.INT32], rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        master = pl.reshape(state[0], [rows, tile])
        weight = pl.reshape(execution, [rows, tile])
        with pl.spmd(rows // packet, name_hint="orthrus_cast_draft_head") as cast:
            index = pl.tile.get_block_idx() * packet
            for offset in pl.range(packet):
                row = index + offset
                value = pl.slice(master, [1, tile], [row, 0])
                if not fp32:
                    value = pl.cast(value, pl.BF16, mode="rint")
                weight[row:row + 1, :] = value
        with pl.spmd(rows // packet, deps=[cast], name_hint="orthrus_publish_draft_head") as sent:
            row = pl.tile.get_block_idx() * packet
            for peer in pl.range(rank + peer_stride, world, peer_stride):
                pld.tensor.put(dst=window, peer=peer, src=weight, dst_offsets=[row, 0],
                               src_offsets=[row, 0], shape=[packet, tile])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False):
            for peer in pl.range(rank + peer_stride, world, peer_stride):
                pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=epoch, op=pld.NotifyOp.Set)

    @pl.jit
    def publish_receive(
        execution: pl.Out[pl.Tensor[[size], execution_dtype]],
        window: pld.DistributedTensor[[rows, tile], execution_dtype],
        signal: pld.DistributedTensor[[1, 1], pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as arrived:
            pld.system.wait(signal=signal, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        weight = pl.reshape(execution, [rows, tile])
        with pl.spmd(rows // packet, deps=[arrived], name_hint="orthrus_bind_draft_head"):
            index = pl.tile.get_block_idx() * packet
            for offset in pl.range(packet):
                row = index + offset
                weight[row:row + 1, :] = window[row:row + 1, :]

    @pl.jit.host
    def publish(
        execution: pl.Out[pl.Tensor[[world, size], execution_dtype]],
        state: pl.Tensor[[lanes, 4, size], pl.FP32], epoch: pl.Scalar[pl.INT32],
    ):
        space = pld.alloc_window_buffer([rows, tile], dtype=execution_dtype)
        signals = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(space, [rows, tile], dtype=execution_dtype)
            signal = pld.window(signals, [1, 1], dtype=pl.INT32)
            if rank < lanes:
                publish_rank(state[rank], execution[rank], window, signal, rank, epoch, device=rank)
            else:
                publish_receive(execution[rank], window, signal, epoch, device=rank)

    return stage, step, publish


def compile_packed_adam(config, model, world, tp):
    compiled = {}
    for parameter in packed_parameters(model, tp):
        key = parameter.program_key
        if key not in compiled:
            programs = make_packed_adam_programs(*key, world=world, tp=tp)
            compiled[key] = tuple(entry.compile(config=config) for entry in programs)
    return compiled


@dataclass(frozen=True)
class PackedUpdate:
    version: int
    applied: bool
    teacher_rows: int
    loss: float
    gradient_norm: float | None
    clip: float
    reverse_seconds: float
    stage_seconds: float
    optimizer_seconds: float
    publication_seconds: float


@dataclass
class _ParameterState:
    weights: object
    state: object


class PackedAdam:
    """Keep one FP32 state per unique TP shard and send updated device weights to its replicas."""

    def __init__(self, proposer, model, programs):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.programs = proposer.runtime, proposer.config, programs
        self.parameters = packed_parameters(model, proposer.tp)
        self.ranks, self.tp = tuple(range(proposer.world)), proposer.tp
        self.owners = self.ranks[::self.tp]
        self.banks = proposer.ranks
        self.layers = range(model.num_hidden_layers - proposer.last_layers, model.num_hidden_layers)
        self.tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.params = {}
        self.version = 0
        self.epoch = 0
        self.failed = False
        self.host = torch.empty((self.tp, 2, 8), dtype=torch.float32)
        self.diagnostics = {
            shared: tuple(self._allocate((1, 8), ranks) for _ in range(2))
            for shared, ranks in ((True, self.ranks[:1]), (False, self.ranks[:self.tp]))
        }

    def _stack(self, shards, ranks):
        return self.stacked(shards, (len(shards), *shards[0].shape), ranks)

    def _allocate(self, shape, ranks):
        import torch

        return self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank)
                            for rank in ranks], ranks)

    def _flat(self, bank, size, layer=0):
        offset = layer * size * bank.dtype.itemsize
        return self.tensor(bank.data_ptr + offset, (size,), bank.dtype, buffer=bank.buffer,
                           byte_offset=bank.byte_offset + offset)

    def _bind(self, layer, parameter):
        key = layer, parameter.name
        bound = self.params.get(key)
        if bound is None:
            size, shared, _ = parameter.program_key
            ranks = self.ranks[:1 if shared else self.tp]
            weights = self._stack([self._flat(bank[parameter.name], size, layer) for bank in self.banks], self.ranks)
            state = self._allocate((4, size), ranks)
            _, initialize, _ = self.programs[parameter.program_key]
            self.runtime.run(initialize, weights, state, 0., 1., 1., 1., 1, config=self.config)
            bound = self.params[key] = _ParameterState(weights, state)
        return bound

    def _stage(self, gradient, parameter):
        size, shared, _ = parameter.program_key
        bound = self._bind(gradient.layer, parameter)
        stage, _, _ = self.programs[parameter.program_key]
        sums, flags = self.diagnostics[shared]
        source = gradient.parameter_gradients[parameter.name]
        devices = self.owners if shared else self.ranks
        flat = self._stack([self._flat(shard, size) for shard in source.shards], devices)
        self.epoch += 1
        self.runtime.run(stage, flat, bound.state, sums, flags, gradient.owner, self.epoch, config=self.config)
        transfers = [
            (out.data_ptr(), tensor.data_ptr, out.nbytes, rank)
            for index, rank in enumerate(sums.worker_ids)
            for out, tensor in zip(self.host[index], (sums.shards[index], flags.shards[index]), strict=True)
        ]
        read_many = getattr(self.runtime, "copy_from_many", None)
        if read_many is None:
            for dst, src, size, rank in transfers:
                self.runtime.copy_from(dst, src, size, worker_id=rank)
        else:
            read_many(transfers)
        norm_sq, valid = 0., True
        for norm, flags in self.host[:len(sums.worker_ids)]:
            norm_sq += float(norm.sum(dtype=self.host.dtype))
            valid &= bool(flags.all())
        return norm_sq, valid

    def update(self, differentiate, *, rate, max_norm=1., prepare=None, agree=None):
        """Run a synchronous reverse and publish after every layer and gradient have been checked."""
        if self.failed:
            raise RuntimeError("The draft optimizer is unusable after failed weight publication")
        if not isfinite(rate) or rate <= 0 or not isfinite(max_norm) or max_norm <= 0:
            raise ValueError("Packed Adam requires a positive finite learning rate and clip norm")
        seen = []
        norm_sq, finite, stage_seconds = 0., True, 0.

        def consume(gradient):
            nonlocal norm_sq, finite, stage_seconds
            if gradient.layer not in self.layers or gradient.layer in seen:
                raise ValueError("Packed Adam requires one gradient for each trainable layer")
            start = perf_counter()
            for parameter in self.parameters:
                source = prepare(gradient, parameter) if prepare is not None else gradient
                partial, valid = self._stage(source, parameter)
                norm_sq += partial
                finite &= valid
            stage_seconds += perf_counter() - start
            seen.append(gradient.layer)

        start = perf_counter()
        loss = differentiate(consume)
        reverse_seconds = perf_counter() - start - stage_seconds
        if loss is None:
            return None
        if set(seen) != set(self.layers):
            raise ValueError("Packed Adam requires a complete suffix gradient")
        finite = finite and isfinite(norm_sq) and isfinite(loss.value)
        norm = sqrt(norm_sq) if finite else None
        clip = min(1., max_norm / (norm + 1e-6)) if finite else 0.
        step = self.version + 1
        if agree is not None:
            finite = agree(step, finite, norm, clip)
        if not finite:
            return PackedUpdate(self.version, False, loss.rows, loss.value, None, 0.,
                                reverse_seconds, stage_seconds, 0., 0.)

        jobs = [(self.programs[item.program_key], self.params[layer, item.name])
                for layer in seen for item in self.parameters]
        try:
            start = perf_counter()
            for (_, apply, _), bound in jobs:
                self.runtime.run(apply, bound.weights, bound.state, rate, 1. - .9 ** step, 1. - .999 ** step,
                                 clip, 0, config=self.config)
            optimizer_seconds = perf_counter() - start
            start = perf_counter()
            for (_, _, publish), bound in jobs:
                self.epoch += 1
                self.runtime.run(publish, bound.weights, bound.state, self.epoch, config=self.config)
            publication_seconds = perf_counter() - start
        except BaseException:
            # A failed device dispatch can leave some in-place state or execution shards changed.
            # The session must stop rather than serve a partially published head.
            self.failed = True
            raise
        self.version = step
        return PackedUpdate(step, True, loss.rows, loss.value, norm, clip, reverse_seconds, stage_seconds,
                            optimizer_seconds, publication_seconds)
