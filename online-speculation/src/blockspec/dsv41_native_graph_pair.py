"""Dispatch retained device graphs through Simpler L4 and two rank-local L3s."""

from contextlib import contextmanager
from dataclasses import dataclass
from struct import pack, unpack


@dataclass(frozen=True)
class NativeChipGraph:
    graph: object
    name: str
    chip: object
    host_tensor_indices: tuple[int, ...] = ()

    @classmethod
    def from_program(cls, rank_graph, program):
        from pypto.runtime.distributed_runner import _assemble_chip_callables

        callables, runtime, sdma = _assemble_chip_callables(program)
        if len(callables) != 1 or runtime != "tensormap_and_ringbuffer" or sdma:
            raise ValueError("Expected the pinned 1122 whole-graph chip runtime")
        name, = callables
        return cls(rank_graph, name, callables[name], tuple(sorted(callables.host_tensor_indices[name])))

    def bind(self, handle):
        return bind_chip(self.name, handle, self.host_tensor_indices)


def bind_chip(name, handle, host_tensor_indices):
    from pypto.runtime.distributed_runner import _ChipCallables

    callables = _ChipCallables()
    callables.add(name, handle, {"host_tensor_indices": host_tensor_indices})
    return callables


def remote_chip(args, scalar_offset):
    """Resolve a binary registered on the parent without rebuilding it on node8."""
    from simpler.remote_l3_session import get_inner_handle

    digest = pack("<4Q", *(args.scalar(scalar_offset + index) for index in range(4)))
    return get_inner_handle(digest.hex())


@contextmanager
def remote_mailbox_view(args):
    """Materialize the daemon's named Host mapping from the wire tensor descriptor."""
    from ctypes import c_ubyte
    from simpler.buffer import ImportContext, ImportRegistry
    from simpler.comm_endpoints import HOST_CPU

    tensor = args.tensor(0)
    registry = ImportRegistry(ImportContext(deployment=HOST_CPU))
    try:
        address = registry.materialize(tensor.buffer).base + tensor.byte_offset
        backing = (c_ubyte * tensor.shapes[0]).from_address(address)
        with memoryview(backing).cast("B") as view:
            yield view
    finally:
        registry.close()


@contextmanager
def native_peer_mailbox(pair, local_buffer):
    """Share control and input/output bytes, keeping weights on their original host."""
    local = memoryview(local_buffer).cast("B")
    try:
        remote = pair.parent.remote_malloc(worker=pair.nodes[1], nbytes=len(local))
        # The parent closes any outstanding remote allocations after a failed dispatch.
        yield NativePeerMailbox(pair, local, remote)
        pair.parent.remote_free(remote)
    finally:
        local.release()


class NativePeerMailbox:
    def __init__(self, pair, local, remote):
        self.pair = pair
        self.local = local
        self.remote = remote

    def exchange(self, config, opcode, payloads, reply_sizes, *, phase=None):
        from ctypes import create_string_buffer
        from simpler.task_interface import DataType, RemoteTensorRef, TaskArgs, TensorArgType

        local_in, remote_in = payloads
        local_size, remote_size = reply_sizes
        capacity = len(self.local)
        if max(len(local_in), len(remote_in), local_size, remote_size) > capacity:
            raise ValueError("Native graph message exceeds its prepared mailbox")
        self.local[:len(local_in)] = local_in
        if remote_in:
            source = create_string_buffer(remote_in, len(remote_in))
            self.pair.parent.remote_copy_to(self.remote, source, len(remote_in))
        arguments = (TaskArgs(), TaskArgs())
        arguments[1].add_tensor(
            RemoteTensorRef(self.remote, shape=(capacity,), dtype=DataType.UINT8), TensorArgType.INOUT,
        )
        for args, payload_size, reply_size in zip(arguments, map(len, payloads), reply_sizes, strict=True):
            args.add_scalar(opcode)
            args.add_scalar(payload_size)
            args.add_scalar(reply_size)
        self.pair.run(config, arguments, phase=phase)
        remote_reply = create_string_buffer(remote_size)
        if remote_size:
            self.pair.parent.remote_copy_from(self.remote, remote_reply, remote_size)
        return bytes(self.local[:local_size]), remote_reply.raw


@contextmanager
def open_native_graph_pair(phases, *, endpoint, peer_listen_host, devices, platform, comm_profile,
                           timeout, local_endpoint, remote_endpoint):
    """Attach both rank ranges; the local factory returns a module-level callback."""
    from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

    local_devices, remote_devices = map(tuple, devices)
    world = len(local_devices) + len(remote_devices)
    if not local_devices or not remote_devices or any(item.graph.world_size != world for item in phases.values()):
        raise ValueError("The two device lists must cover the compiled global rank space")
    runtime = "tensormap_and_ringbuffer"
    local = Worker(level=3, platform=platform, runtime=runtime, num_sub_workers=0, device_ids=local_devices,
                   global_device_ranks=tuple(range(len(local_devices))), comm_profile=comm_profile)
    parent = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=timeout)
    attached = False
    try:
        local_id = parent.add_worker(local)
        attached = True
        remote_id = parent.add_remote_worker(RemoteWorkerSpec(
            endpoint=endpoint, platform=platform, runtime=runtime, device_ids=remote_devices,
            global_device_ranks=tuple(range(len(local_devices), world)), session_listen_host=peer_listen_host,
            comm_profile=comm_profile,
        ))
        # The forked L3 gets local handles. The daemon receives the same binaries through the L4 registry.
        local_bound = {name: phase.bind(local.register(phase.chip)) for name, phase in phases.items()}
        local_call = parent.register(local_endpoint(local_bound))
        remote_call = parent.register(RemoteCallable(remote_endpoint), workers=[remote_id])
        digests = {name: parent.register(phase.chip).digest for name, phase in phases.items()}
        parent.init()
        members = tuple(
            (node, worker) for node, cards in ((local_id, local_devices), (remote_id, remote_devices))
            for worker in range(len(cards))
        )
        yield NativeGraphPair(parent, phases, (local_id, remote_id), (local_call, remote_call), members, digests)
    finally:
        try:
            parent.close()
        finally:
            if not attached:
                local.close()


class NativeGraphPair:
    def __init__(self, parent, phases, nodes, callbacks, members, digests):
        self.parent = parent
        self.phases = phases
        self.nodes = nodes
        self.callbacks = callbacks
        self.members = members
        self.digests = digests
        self._used_phases = set()

    def run(self, config, arguments, *, phase=None, initialize=None):
        """Submit both hosts before waiting for either host's device graph."""
        def submit(orch, _args, cfg):
            if phase is not None:
                graph = self.phases[phase].graph
                domain = graph.allocate_domain(orch, f"dsv41/{phase}", self.members)
                if initialize is not None:
                    initialize(orch, domain)
                for args in arguments:
                    args.add_scalar(domain.domain_id)
                for word in unpack("<4Q", self.digests[phase]):
                    arguments[1].add_scalar(word)
            for node, callback, args in zip(self.nodes, self.callbacks, arguments, strict=True):
                orch.submit_next_level(callback, args, cfg, worker=node)

        self.parent.run(submit, config=config)
        if phase is not None:
            self._used_phases.add(phase)

    def reserve_windows(self, config):
        def reserve(orch, _args, _cfg):
            for phase, item in self.phases.items():
                item.graph.allocate_domain(orch, f"dsv41/{phase}", self.members)

        self.parent.run(reserve, config=config)
        self._used_phases.update(self.phases)

    def release_windows(self, config):
        def release(orch, _args, _cfg):
            for phase in sorted(self._used_phases):
                orch.get_domain(f"dsv41/{phase}").release()

        self.parent.run(release, config=config)
        self._used_phases.clear()
