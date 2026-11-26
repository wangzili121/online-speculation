"""Publish a shared Orthrus head from node-resident last-N gradients."""

from math import gcd
from time import perf_counter
from types import SimpleNamespace

from .dsv41_online_profile import timed_stage
from .dsv41_packed_adam import PackedAdam, make_packed_adam_programs, packed_parameters


def make_gradient_exchange_pack(size, shared, *, world, tp):
    """Place the request owner's FP32 gradients in HCCL-exportable buffers."""
    import pypto.language as pl

    lanes = 1 if shared else tp
    rows = size // gcd(size, 2048)
    tile = size // rows
    packet = gcd(rows, 16)
    sources = world // tp if shared else world

    @pl.jit
    def copy(source: pl.Tensor[[size], pl.FP32], send: pl.Out[pl.Tensor[[size], pl.FP32]]):
        x = pl.reshape(source, [rows, tile])
        y = pl.reshape(send, [rows, tile])
        for index in pl.spmd(rows // packet, name_hint="adam_exchange_copy"):
            for offset in pl.range(packet):
                row = index * packet + offset
                y[row:row + 1, :] = x[row:row + 1, :]

    @pl.jit
    def zero(send: pl.Out[pl.Tensor[[size], pl.FP32]]):
        y = pl.reshape(send, [rows, tile])
        for index in pl.spmd(rows // packet, name_hint="adam_exchange_zero"):
            for offset in pl.range(packet):
                row = index * packet + offset
                y[row:row + 1, :] = pl.full([1, tile], dtype=pl.FP32, value=0.)

    @pl.jit.host
    def pack(
        gradient: pl.Tensor[[sources, size], pl.FP32],
        send: pl.Out[pl.Tensor[[world, size], pl.FP32]],
        owner: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(world):
            if owner >= 0 and rank >= owner and rank < owner + lanes:
                source = rank
                if shared:
                    source = rank // tp
                copy(gradient[source], send[rank], device=rank)
            else:
                zero(send[rank], device=rank)

    return pack


def _entries(model, tp):
    return {parameter.program_key: parameter.name for parameter in packed_parameters(model, tp)}


def make_staged_adam_graphs(model, world, tp):
    graphs = {}
    for (size, shared, fp32), name in _entries(model, tp).items():
        pack = make_gradient_exchange_pack(size, shared, world=world, tp=tp)
        adam = make_packed_adam_programs(size, shared, fp32, world=world, tp=tp)
        graphs.update((f"{name}/{stage}", entry) for stage, entry in
                      zip(("pack", "stage", "apply", "publish"), (pack, *adam), strict=True))
    return graphs


def load_staged_adam_programs(directory, model, tp):
    from pypto.ir import DistributedCompiledProgram

    return {
        key: tuple(DistributedCompiledProgram.from_dir(directory / name / stage)
                   for stage in ("pack", "stage", "apply", "publish"))
        for key, name in _entries(model, tp).items()
    }


class StagedPackedAdam:
    """Exchange each unique gradient once, apply locally, then publish on both hosts."""

    def __init__(self, draft, reverse, model, programs):
        self.reverse = reverse
        self.link = draft.target.link
        self.transport = draft.ffn.moe.transport
        self.adam = PackedAdam(draft, model, {key: entries[1:] for key, entries in programs.items()})
        self.pack = {key: entries[0] for key, entries in programs.items()}
        self.runtime = draft.runtime
        self.packet_capacity = max(key[0] for key in programs)
        self.packets = None

    def _buffers(self, size, shared):
        adam = self.adam
        if self.packets is None:
            import torch

            def allocate():
                shards = [self.runtime.alloc_tensor((self.packet_capacity,), torch.float32,
                                                    worker_id=rank, exportable=True)
                          for rank in adam.ranks]
                for rank, shard in enumerate(shards):
                    self.transport.mappings.register(rank, shard)
                return shards

            self.packets = allocate(), allocate()
        # Each parameter is consumed before taking the next view. HCCL resolves
        # the view at offset zero through the backing allocation's IPC mapping.
        send, received = (adam._stack([adam._flat(shard, size) for shard in shards], adam.ranks)
                          for shards in self.packets)
        placeholder = adam._stack(received.shards[::adam.tp], adam.owners) if shared else received
        return send, received, placeholder

    def _exchange(self, stage, layer, gradient, parameter):
        adam = self.adam
        key = parameter.program_key
        size, shared, _ = key
        send, received, placeholder = self._buffers(size, shared)
        source, owner = placeholder, -1
        if gradient is not None:
            shards = gradient.parameter_gradients[parameter.name].shards
            ranks = adam.owners if shared else adam.ranks
            source = adam._stack([adam._flat(shard, size) for shard in shards], ranks)
            owner = gradient.owner
        with timed_stage("online_gradient_pack"):
            self.runtime.run(self.pack[key], source, send, owner, config=adam.config)
        transfers = [((x, y, parameter.name),) for x, y in zip(send.shards, received.shards, strict=True)]
        self.transport.exchange_resident(f"{stage}/{layer}/{parameter.name}", transfers,
                                         collective="all_reduce", group="dp")
        output = placeholder if shared else received
        return SimpleNamespace(layer=layer, owner=0, parameter_gradients={parameter.name: output})

    def update(self, stage, *, owner, feedback, finished, rate, max_norm):
        """Block the next draft until both hosts acknowledge the same head version."""
        receipt = None
        start = perf_counter()

        def differentiate(consume):
            nonlocal receipt
            receipt = self.reverse.run(stage, owner=owner, feedback=feedback, finished=finished,
                                       consume=lambda layer, source: consume(SimpleNamespace(layer=layer, source=source)))
            if receipt is not None:
                return SimpleNamespace(value=receipt["loss"], rows=receipt["teacher_rows"])
            return None

        def agree(step, finite, norm, clip):
            decision = dict(step=step, finite=finite, norm=norm, clip=clip, rate=rate)
            peer = self.link.exchange(f"{stage}/decision", decision)
            if peer["step"] != step or peer["rate"] != rate:
                raise ValueError("Hosts disagree on the next Adam step")
            if not finite or not peer["finite"]:
                return False
            if peer["norm"] != norm or peer["clip"] != clip:
                raise ValueError("Hosts disagree on the Adam gradient norm")
            return True

        def prepare(gradient, parameter):
            return self._exchange(stage, gradient.layer, gradient.source, parameter)

        update = self.adam.update(differentiate, rate=rate, max_norm=max_norm, prepare=prepare, agree=agree)
        if update is None:
            return None
        publication = dict(version=update.version, applied=update.applied)
        if self.link.exchange(f"{stage}/published", publication) != publication:
            self.adam.failed = True
            raise ValueError("Hosts published different Orthrus head versions")
        return {**receipt, **vars(update), "update_seconds": perf_counter() - start}
