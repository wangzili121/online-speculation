"""A native PyPTO request transaction, including recurrent-state rollback."""

from collections import deque
from dataclasses import dataclass

from .greedy import verify_transaction


@dataclass(frozen=True)
class Snapshot:
    length: int
    pages: tuple[int, ...]
    state: tuple


class NativeTarget:
    def __init__(self, runner, request, slot_type, *, rows=16):
        self.runner, self.request, self.slot_type, self.rows = runner, request, slot_type, rows

    @property
    def resident(self):
        return self.runner.requests[self.request.request_id]

    def snapshot(self):
        resident = self.resident
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
