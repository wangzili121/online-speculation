"""A native PyPTO request transaction, including recurrent-state rollback."""

from collections import deque
from dataclasses import dataclass
from math import prod
from pathlib import Path
import shutil
import threading
import weakref

from .greedy import host_greedy_tokens, verify_transaction


@dataclass(frozen=True)
class Snapshot:
    length: int
    pages: tuple[int, ...]
    state: tuple


class SnapshotPool:
    """快照存活期间独占对应缓冲；池满时由调用方回退到独立副本。"""

    def __init__(self, conv_shape, recurrent_shape, capacity=2):
        import torch
        conv_bytes, recurrent_bytes = prod(conv_shape) * 2, prod(recurrent_shape) * 4
        stride = conv_bytes + recurrent_bytes
        size = capacity * stride
        if Path("/dev/shm").is_dir() and shutil.disk_usage("/dev/shm").free < size + (64 << 20):
            raise RuntimeError(f"state snapshots require at least {size + (64 << 20)} free shared bytes")
        self.storage = torch.empty((capacity, stride), dtype=torch.uint8).share_memory_()
        self.states = tuple((row[:conv_bytes].view(torch.bfloat16).reshape(conv_shape),
                             row[conv_bytes:].view(torch.float32).reshape(recurrent_shape))
                            for row in self.storage)
        self.busy = set()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            index = next((i for i in range(len(self.states)) if i not in self.busy), None)
            if index is None:
                return None
            self.busy.add(index)
        return index, self.states[index]

    def release(self, index):
        with self.lock:
            self.busy.remove(index)


class NativeTarget:
    def __init__(self, runner, request, slot_type, *, rows=16):
        self.runner, self.request, self.slot_type, self.rows = runner, request, slot_type, rows

    @property
    def resident(self):
        return self.runner.requests[self.request.request_id]

    def snapshot(self):
        resident = self.resident
        pool = getattr(self.runner, "verification_states", None)
        entry = None if pool is None else pool.acquire()
        if entry is not None:
            index, state = entry
            try:
                for host, device in zip(state, self.runner.resident[:2], strict=True):
                    offset = resident.slot * host.numel() * host.element_size()
                    self.runner._copy_offset(host, device, offset, False)
                snapshot = Snapshot(resident.pages.length, tuple(resident.pages.pages), state)
                weakref.finalize(snapshot, pool.release, index)
                return snapshot
            except BaseException:
                pool.release(index)
                raise
        return Snapshot(resident.pages.length, tuple(resident.pages.pages),
                        self.runner._read_state(resident.slot))

    def restore(self, snapshot):
        resident = self.resident
        pages = resident.pages
        if tuple(pages.pages[:len(snapshot.pages)]) != snapshot.pages:
            raise RuntimeError("verification replaced committed target pages")
        self.runner._write_state(resident.slot, *snapshot.state)
        for page in pages.pages[len(snapshot.pages):]:
            self.runner.cache._drop(page)
        del pages.pages[len(snapshot.pages):]
        pages.length = snapshot.length

    def forward(self, inputs):
        output = []
        for offset in range(0, len(inputs), self.rows):
            tokens = tuple(inputs[offset:offset + self.rows])
            slot = self.slot_type(self.request, tokens, self.resident.pages.length, "decode", False)
            self.runner.run_batch([slot])
            host = getattr(self.runner, "verification_logits", None)
            if host is not None:
                # 一次读取当前 tile 的有效行，复用 fork 前注册的共享主机缓冲。
                logits = host[:len(tokens)]
                offset = self.resident.slot * host.numel() * host.element_size()
                self.runner.worker.copy_from(logits.data_ptr(), self.runner.logits.data_ptr,
                    logits.numel() * logits.element_size(), src_offset=offset)
                output.extend(host_greedy_tokens(logits, name="native target verifier"))
                continue
            for row in range(len(tokens)):
                logits = self.runner._logit_row(self.resident.slot, row)
                if not logits.isfinite().all():
                    raise FloatingPointError("native target returned nonfinite verifier logits")
                output.append(int(logits.float().argmax()))
        return output


class OrthrusRunner:
    """Keep OneServing's HTTP/scheduler contract while buffering accepted tokens.

    First implementation is serialized TP1. The native scheduler consumes one
    token at a time; buffered tokens require no extra model invocation.
    """

    def __init__(self, native, draft, slot_type):
        if getattr(native, "device_ids", None):
            raise ValueError("Orthrus transaction adapter currently requires the native TP1 runner")
        self.native, self.draft, self.slot_type = native, draft, slot_type
        self.pending = {}
        self.rounds = self.accepted = self.restores = 0
        self.allow_prefix_hits = False

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def allow_prefix_hits(self):
        return self.native.allow_prefix_hits

    @allow_prefix_hits.setter
    def allow_prefix_hits(self, enabled):
        if enabled:
            raise ValueError("prefix reuse requires a separate Orthrus acceptance run")
        self.native.allow_prefix_hits = False

    def start(self):
        try:
            self.native.start()
            self.draft.start(self.native)
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            self.draft.close()
        finally:
            self.native.close()
            self.pending.clear()

    def admit(self, request):
        if self.native.requests:
            raise RuntimeError("initial Orthrus PyPTO runner requires max_active=1")
        self.pending[request.request_id] = deque()
        try:
            return self.native.admit(request)
        except BaseException:
            self.pending.pop(request.request_id, None)
            raise

    def release(self, request):
        self.pending.pop(request.request_id, None)
        return self.native.release(request)

    def run_batch(self, slots):
        if len(slots) != 1:
            raise ValueError("initial Orthrus PyPTO runner supports one active request")
        slot = slots[0]
        request = slot.request
        if slot.phase == "prefill":
            return self.native.run_batch(slots)
        pending = self.pending[request.request_id]
        if pending:
            position, anchor, token = pending[0]
            if slot.position != position or tuple(slot.token_ids) != (anchor,):
                raise RuntimeError("buffered accepted prefix no longer matches the scheduled request")
            pending.popleft()
            return [token]
        remaining = request.max_tokens - len(request.generated)
        width = min(self.draft.block_size, self.draft.context - slot.position, remaining)
        if width < 2:
            return self.native.run_batch(slots)
        candidates = tuple(self.draft.propose(request, slot.token_ids[0], width))
        if len(candidates) != width - 1:
            raise RuntimeError("draft must propose exactly K-1 tokens")
        target = NativeTarget(self.native, request, self.slot_type)
        result = verify_transaction(target, slot.token_ids[0], candidates,
                                    budget=remaining, stop_tokens=request.stop_token_ids)
        self.rounds += 1
        self.accepted += result.accepted
        self.restores += result.committed_inputs < width
        pending.extend((slot.position + index, slot.token_ids[0] if index == 0 else result.tokens[index - 1], token)
                       for index, token in enumerate(result.tokens))
        return [pending.popleft()[2]]
