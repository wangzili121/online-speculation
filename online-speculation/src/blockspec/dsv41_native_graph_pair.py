"""Dispatch retained device graphs through Simpler L4 and two rank-local L3s."""

from contextlib import contextmanager
from dataclasses import dataclass
from struct import pack, unpack


@dataclass(frozen=True)
class NativeChipGraph:
    graph: object
    chips: dict
    abi: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_program(cls, rank_graph, program):
        from pypto.runtime.distributed_runner import _assemble_chip_callables

        callables, runtime, sdma = _assemble_chip_callables(program)
        if runtime != "tensormap_and_ringbuffer" or sdma:
            raise ValueError("Expected the pinned 1122 chip runtime")
        abi = tuple((name, tuple(sorted(callables.host_tensor_indices[name]))) for name in callables)
        return cls(rank_graph, callables, abi)

    def bind(self, register):
        return bind_chips(self.abi, (register(self.chips[name]) for name, _ in self.abi))


def bind_chips(abi, handles):
    from pypto.runtime.distributed_runner import _ChipCallables

    callables = _ChipCallables()
    for (name, host_tensor_indices), handle in zip(abi, handles, strict=True):
        callables.add(name, handle, {"host_tensor_indices": host_tensor_indices})
    return callables


def remote_chip(args, scalar_offset):
    """Resolve a binary registered on the parent without rebuilding it on node8."""
    from simpler.remote_l3_session import get_inner_handle

    digest = pack("<4Q", *(args.scalar(scalar_offset + index) for index in range(4)))
    return get_inner_handle(digest.hex())


def remote_chips(abi, args, offset):
    return bind_chips(abi, (remote_chip(args, offset + index * 4) for index in range(len(abi))))


@contextmanager
def host_mailbox_view(args):
    """Materialize the local or daemon Host mailbox carried by the task."""
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
def native_peer_mailbox(pair, capacity):
    """Share control and input/output bytes, keeping weights on their original host."""
    local = pair.parent.create_buffer(capacity)
    remote = pair.parent.remote_malloc(worker=pair.nodes[1], nbytes=capacity)
    # Keep the shared local buffer until the parent has joined its children.
    with local.host_view() as view:
        yield NativePeerMailbox(pair, local, view, remote)
    pair.parent.remote_free(remote)


class NativePeerMailbox:
    def __init__(self, pair, local_buffer, local, remote):
        from ctypes import create_string_buffer
        from simpler.remote_l3_limits import MAX_FRAME_PAYLOAD_BYTES
        from simpler.task_interface import DataType

        self.pair = pair
        self.local = local
        self.local_tensor = local_buffer.tensor((len(local),), DataType.UINT8)
        self.remote = remote
        self.remote_staging = create_string_buffer(len(local))
        # Leave space for the command envelope within Simpler's frame limit.
        self.remote_chunk_size = MAX_FRAME_PAYLOAD_BYTES - 64

    def _copy_remote(self, transfer, nbytes):
        from ctypes import addressof

        address = addressof(self.remote_staging)
        for offset in range(0, nbytes, self.remote_chunk_size):
            size = min(nbytes - offset, self.remote_chunk_size)
            transfer(self.remote, address + offset, size, offset=offset)

    def exchange(self, config, opcode, payloads, reply_sizes, *, phase=None):
        from simpler.task_interface import DataType, RemoteTensorRef, TaskArgs, TensorArgType

        packets = tuple(payload if isinstance(payload, tuple) else (payload,) for payload in payloads)
        local_bytes, remote_bytes = sizes = tuple(sum(map(len, packet)) for packet in packets)
        local_size, remote_size = reply_sizes
        capacity = len(self.local)
        if max(local_bytes, remote_bytes, local_size, remote_size) > capacity:
            raise ValueError("Native graph message exceeds its prepared mailbox")

        # Copy each header and image directly into its send buffer. Keep a
        # contiguous remote buffer without allocating a joined packet.
        with memoryview(self.remote_staging).cast("B") as remote:
            for destination, packet in zip((self.local, remote), packets, strict=True):
                start = 0
                for part in packet:
                    end = start + len(part)
                    destination[start:end] = part
                    start = end
        if remote_bytes:
            self._copy_remote(self.pair.parent.remote_copy_to, remote_bytes)
        arguments = (TaskArgs(), TaskArgs())
        arguments[0].add_tensor(self.local_tensor, TensorArgType.INOUT)
        arguments[1].add_tensor(
            RemoteTensorRef(self.remote, shape=(capacity,), dtype=DataType.UINT8), TensorArgType.INOUT,
        )
        for args, payload_size, reply_size in zip(arguments, sizes, reply_sizes, strict=True):
            args.add_scalar(opcode)
            args.add_scalar(payload_size)
            args.add_scalar(reply_size)
        self.pair.run(config, arguments, phase=phase)
        if remote_size:
            self._copy_remote(self.pair.parent.remote_copy_from, remote_size)
        return bytes(self.local[:local_size]), self.remote_staging[:remote_size]


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
        local_bound = {name: phase.bind(local.register) for name, phase in phases.items()}
        local_call = parent.register(local_endpoint(local_bound))
        remote_call = parent.register(RemoteCallable(remote_endpoint), workers=[remote_id])
        digests = {name: tuple(parent.register(phase.chips[chip]).digest for chip, _ in phase.abi)
                   for name, phase in phases.items()}
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
        phases = phase if isinstance(phase, tuple) else (phase,) if phase is not None else ()
        if len(phases) > 1 and any(self.phases[name].graph.window is not None for name in phases):
            raise ValueError("A batch of native graphs cannot share collective windows")

        def submit(orch, _args, cfg):
            for name in phases:
                graph = self.phases[name].graph
                domain = graph.allocate_domain(orch, f"dsv41/{name}", self.members)
                if initialize is not None:
                    initialize(orch, domain)
                for args in arguments:
                    args.add_scalar(domain.domain_id if domain is not None else 0)
                for digest in self.digests[name]:
                    for word in unpack("<4Q", digest):
                        arguments[1].add_scalar(word)
            for node, callback, args in zip(self.nodes, self.callbacks, arguments, strict=True):
                orch.submit_next_level(callback, args, cfg, worker=node)

        self.parent.run(submit, config=config)
        self._used_phases.update(name for name in phases if self.phases[name].graph.window is not None)

    def reserve_windows(self, config):
        reserved = {}

        def reserve(orch, _args, _cfg):
            for phase, item in self.phases.items():
                graph = item.graph
                if graph.window is None:
                    continue
                size = graph.window["window_size"]
                try:
                    graph.allocate_domain(orch, f"dsv41/{phase}", self.members)
                except Exception as error:
                    raise RuntimeError(
                        f"Fabric window {phase!r} failed for {len(self.members)} ranks "
                        f"({size} bytes per rank, {len(reserved)} earlier windows reserved): {error}"
                    ) from error
                reserved[phase] = size

        self.parent.run(reserve, config=config)
        self._used_phases.update(reserved)
        return reserved

    def release_windows(self, config):
        def release(orch, _args, _cfg):
            for phase in sorted(self._used_phases):
                orch.get_domain(f"dsv41/{phase}").release()

        self.parent.run(release, config=config)
        self._used_phases.clear()
