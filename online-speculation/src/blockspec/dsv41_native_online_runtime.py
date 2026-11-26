"""Use the existing packed learner with allocations resident on both L3 nodes."""

from bisect import bisect_right
from ctypes import addressof, memmove, string_at
from dataclasses import dataclass
import json
from math import prod
from struct import Struct

from .dsv41_native_rank_dispatch import RankLocalStack
from .dsv41_online_profile import timed_stage


ARENA, TRAIN_GRAPH = 7, 8
HEADER = Struct("<I")


def encode_command(record, data=b""):
    text = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
    return b"".join((HEADER.pack(len(text)), text, data))


def decode_command(payload):
    size, = HEADER.unpack_from(payload)
    stop = HEADER.size + size
    if stop > len(payload):
        raise ValueError("Native learner command header exceeds its payload")
    return json.loads(bytes(payload[HEADER.size:stop])), payload[stop:]


def _describe(tensor):
    return (tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."))


@dataclass(frozen=True)
class _ProxyBuffer:
    key: str | int
    rank: int
    base: int
    nbytes: int

    def tensor(self, *args, **kwargs):
        raise TypeError("Resolve this resident tensor on its owning L3 worker")


class _LocalShards:
    """Indexed slices from a stack, with its original global tensor geometry."""

    def __init__(self, shape, shards):
        self.shape = shape
        self.shards = shards

    def __getitem__(self, key):
        index, *tail = key if isinstance(key, tuple) else (key,)
        if isinstance(index, slice):
            width = index.stop - index.start
            if index.step not in (None, 1) or index.start % width:
                raise ValueError("Compiled NZ view is not a whole rank shard")
            index = index.start // width
            shard = self.shards[index]
            if not tail or len(shard.shape) != len(tail) + 1 or shard.shape[0] != width:
                raise ValueError("Compiled NZ view does not match the resident physical geometry")
            expected = shard.shape[1:]
        else:
            shard = self.shards[index]
            expected = shard.shape
        if any(item not in (slice(None), slice(0, limit)) for item, limit in zip(tail, expected)):
            raise ValueError("Compiled operand attempted a partial slice of a rank shard")
        if len(tail) > len(expected):
            raise ValueError("Compiled operand used more indices than its rank shard")
        return shard


class NativeOnlineArena:
    """Map learner tensor names to owned device allocations on one host."""

    def __init__(self, memory, ranks, banks):
        self.memory = memory
        self.workers = {rank: worker for worker, rank in enumerate(ranks)}
        self.banks = banks
        self.named = {(rank, f"{bank}/{name}"): tensor
                      for bank, per_rank in banks.items()
                      for rank, values in per_rank.items() for name, tensor in values.items()}
        self.dynamic = {}

    def _get(self, rank, key):
        return self.named[rank, key] if isinstance(key, str) else self.dynamic[rank, key]

    def prepare(self, allocations, releases):
        import torch

        for rank, key in releases:
            tensor = self.dynamic[rank, key]
            worker = self.workers[rank]
            self.memory.free_tensor(tensor, worker_id=worker)
            del self.dynamic[rank, key]
        for rank, key, shape, dtype in allocations:
            if (rank, key) in self.dynamic:
                raise ValueError("Duplicate native learner allocation")
            self.dynamic[rank, key] = self.memory.alloc_tensor(shape, getattr(torch, dtype),
                                                               worker_id=self.workers[rank])

    def _view(self, reference):
        import torch
        from pypto.runtime import DeviceTensor

        rank, key, offset, shape, dtype = reference
        resident = self._get(rank, key)
        return DeviceTensor(resident.data_ptr + offset, shape, getattr(torch, dtype), buffer=resident.buffer,
                            byte_offset=resident.byte_offset + offset)

    def bind(self, values):
        operands = {}
        for name, value in values.items():
            if isinstance(value, (int, float)):
                operands[name] = value
            elif "stack" in value:
                shape, slots = value["stack"]
                operands[name] = RankLocalStack(
                    _LocalShards(shape, {index: self._view(view) for index, view in slots}),
                    tuple(range(shape[0])), shape[0],
                )
            elif "tensor" in value:
                operands[name] = self._view(value["tensor"])
            else:
                raise ValueError(f"Unknown native learner operand for {name}")
        return operands

    def control(self, command, data, reply):
        from ctypes import c_char

        operation = command["op"]
        self.prepare(command["allocate"], command["release"])
        if operation == "describe":
            layout = {bank: {rank: {name: _describe(tensor) for name, tensor in values.items()}
                             for rank, values in per_rank.items()} for bank, per_rank in self.banks.items()}
            response = encode_command(layout)
            if len(response) > len(reply):
                raise ValueError("Resident training layout exceeds the prepared mailbox")
            reply[:len(response)] = response
            return
        if operation == "prepare":
            return
        if operation == "load_many":
            offset = 0
            for address, size in command["reads"]:
                self.read(address, reply[offset:offset + size])
                offset += size
            if offset != len(reply):
                raise ValueError("Native learner readback size does not match its request")
            return
        if operation == "load":
            self.read(command["address"], reply)
            return
        rank, key, offset = command["address"]
        tensor = self._get(rank, key)
        worker = self.workers[rank]
        base = tensor.data_ptr - tensor.byte_offset
        if operation == "store":
            view = (c_char * len(data)).from_buffer(data)
            self.memory.copy_to(base, addressof(view), len(data), worker_id=worker,
                                dst_offset=tensor.byte_offset + offset)
        else:
            raise ValueError(f"Unknown native learner arena operation {operation}")

    def read(self, address, reply):
        from ctypes import c_char

        rank, key, offset = address
        tensor = self._get(rank, key)
        view = (c_char * len(reply)).from_buffer(reply)
        self.memory.copy_from(addressof(view), tensor.data_ptr - tensor.byte_offset, len(reply),
                              worker_id=self.workers[rank], src_offset=tensor.byte_offset + offset)


class NativeOnlineRuntime:
    """Run the existing packed replay and optimizer against logical device handles."""

    def __init__(self, mailbox, config, programs, *, ranks_per_host):
        self.mailbox = mailbox
        self.call_config = config
        self.programs = {program: (name, graph) for name, (program, graph) in programs.items()}
        self.ranks_per_host = ranks_per_host
        self.logicals = []
        self.buffers = {}
        self.allocations = ([], [])
        self.releases = ([], [])
        self.next_key = 0
        self.next_base = 512

    def _proxy(self, rank, key, shape, dtype):
        from pypto.runtime import DeviceTensor

        base = self.next_base
        buffer = _ProxyBuffer(key, rank, base, prod(shape) * dtype.itemsize)
        self.next_base = (base + buffer.nbytes + 511) // 512 * 512
        self.logicals.append(base)
        self.buffers[base] = buffer
        return DeviceTensor(base, shape, dtype, buffer=buffer)

    def _node(self, rank):
        return rank // self.ranks_per_host

    def _message(self, node, op, fields=None, data=b""):
        return encode_command({
            "allocate": self.allocations[node], "release": self.releases[node], "op": op, **(fields or {}),
        }, data)

    def _send(self, opcode, payloads, sizes=(0, 0), *, phase=None):
        result = self.mailbox.exchange(self.call_config, opcode, payloads, sizes, phase=phase)
        for queue in (*self.allocations, *self.releases):
            queue.clear()
        return result

    def load_banks(self):
        import torch

        cap = min(len(self.mailbox.local), 512 * 1024)
        data = self._send(ARENA, tuple(self._message(node, "describe") for node in range(2)), (cap, cap))
        banks = {}
        for response in data:
            records, _ = decode_command(response)
            for bank, descriptors in records.items():
                result = banks.setdefault(bank, {})
                for rank, values in descriptors.items():
                    rank = int(rank)
                    if rank in result:
                        raise ValueError("Duplicate native learner bank rank")
                    result[rank] = {
                        name: self._proxy(rank, f"{bank}/{name}", shape, getattr(torch, dtype))
                        for name, (shape, dtype) in values.items()
                    }
        return banks

    def prepare(self):
        self._send(ARENA, tuple(self._message(node, "prepare") for node in range(2)))

    def alloc_tensor(self, shape, dtype, *, worker_id, init=None):
        key = self.next_key
        self.next_key += 1
        tensor = self._proxy(worker_id, key, tuple(shape), dtype)
        self.allocations[self._node(worker_id)].append((worker_id, key, *_describe(tensor)))
        if init is not None:
            value = init.contiguous()
            self.copy_to(tensor.data_ptr, value.data_ptr(), value.nbytes, worker_id=worker_id)
        return tensor

    def free_tensor(self, tensor, *, worker_id):
        buffer = tensor.buffer
        if (buffer.rank != worker_id or not isinstance(buffer.key, int) or tensor.byte_offset
                or self.buffers.get(buffer.base) is not buffer):
            raise ValueError("Only the owner may release a whole learner allocation")
        queue = self.allocations[self._node(worker_id)]
        for index, (rank, key, *_rest) in enumerate(queue):
            if (rank, key) == (worker_id, buffer.key):
                del queue[index]
                break
        else:
            self.releases[self._node(worker_id)].append((worker_id, buffer.key))
        del self.buffers[buffer.base]
        del self.logicals[bisect_right(self.logicals, buffer.base) - 1]

    def _address(self, ptr, rank, offset, size):
        index = bisect_right(self.logicals, ptr) - 1
        buffer = self.buffers[self.logicals[index]] if index >= 0 else None
        if buffer is None or buffer.rank != rank or not 0 <= ptr + offset - buffer.base <= buffer.nbytes - size:
            raise ValueError("Transfer exceeds its native learner buffer")
        return (rank, buffer.key, ptr + offset - buffer.base)

    def _transfer(self, op, address, size, data=b""):
        owner = self._node(address[0])
        payloads = tuple(
            self._message(node, op, {"address": address}, data) if node == owner else self._message(node, "prepare")
            for node in range(2)
        )
        sizes = tuple(size if node == owner and op == "load" else 0 for node in range(2))
        return self._send(ARENA, payloads, sizes)[owner]

    def copy_to(self, dst, src, nbytes, *, worker_id, dst_offset=0, src_offset=0):
        address = self._address(dst, worker_id, dst_offset, nbytes)
        self._transfer("store", address, nbytes, string_at(src + src_offset, nbytes))

    def copy_from(self, dst, src, nbytes, *, worker_id, dst_offset=0, src_offset=0):
        address = self._address(src, worker_id, src_offset, nbytes)
        memmove(dst + dst_offset, self._transfer("load", address, nbytes), nbytes)

    def copy_from_many(self, transfers):
        """Return a whole set of loss/gradient diagnostics in one control round."""
        reads, destinations = ([], []), ([], [])
        sizes = [0, 0]
        for dst, src, nbytes, rank in transfers:
            node = self._node(rank)
            reads[node].append((self._address(src, rank, 0, nbytes), nbytes))
            destinations[node].append((dst, sizes[node], nbytes))
            sizes[node] += nbytes
        payloads = tuple(self._message(node, "load_many", {"reads": reads[node]}) for node in range(2))
        with timed_stage("native_learner_diagnostics_read"):
            outputs = self._send(ARENA, payloads, tuple(sizes))
        for output, copies in zip(outputs, destinations, strict=True):
            for dst, start, nbytes in copies:
                memmove(dst, output[start:start + nbytes], nbytes)

    def _reference(self, tensor, rank):
        buffer = tensor.buffer
        if self.buffers.get(buffer.base) is not buffer or buffer.rank != rank:
            raise ValueError("Native learner operand is not an allocation owned by the named rank")
        return (rank, buffer.key, tensor.byte_offset, *_describe(tensor))

    def _graph_values(self, program, arguments):
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        parameters = program._get_metadata()[0]
        values = [{}, {}]
        for parameter, arg in zip(parameters, arguments, strict=True):
            if parameter.shape is None:
                for items in values:
                    items[parameter.name] = arg
            elif isinstance(arg, StackedDeviceTensor):
                slots = ([], [])
                for index, (rank, shard) in enumerate(zip(arg.worker_ids, arg.shards, strict=True)):
                    slots[self._node(rank)].append((index, self._reference(shard, rank)))
                for node in range(2):
                    values[node][parameter.name] = {"stack": (arg.shape, slots[node])}
            elif isinstance(arg, DeviceTensor):
                rank = arg.buffer.rank
                values[self._node(rank)][parameter.name] = {"tensor": self._reference(arg, rank)}
            else:
                raise TypeError(f"Expected a resident operand for {parameter.name}")
        return values

    def _submit_graphs(self, phases, values):
        payloads = tuple(self._message(node, "graph", {"graphs": [
            (name, operands[node]) for name, operands in zip(phases, values, strict=True)
        ]}) for node in range(2))
        self._send(TRAIN_GRAPH, payloads, phase=phases[0] if len(phases) == 1 else phases)

    def run(self, program, *arguments, config=None):
        name, _ = self.programs[program]
        with timed_stage(f"native_learner/{name}"):
            self._submit_graphs((name,), (self._graph_values(program, arguments),))

    def run_adam_apply(self, jobs):
        """Submit independent local Adam updates together, before publication."""
        batch = {}
        with timed_stage("native_learner/adam_apply_batch"):
            for program, arguments in jobs:
                name, graph = self.programs[program]
                if not name.endswith("/apply") or graph.window is not None:
                    raise ValueError("Batched Adam apply requires a rank-local optimizer graph")
                if name in batch:
                    self._submit_graphs(tuple(batch), tuple(batch.values()))
                    batch.clear()
                batch[name] = self._graph_values(program, arguments)
            if batch:
                self._submit_graphs(tuple(batch), tuple(batch.values()))
