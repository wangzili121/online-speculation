from ctypes import addressof, create_string_buffer, memmove
import json
from types import SimpleNamespace

import torch

from blockspec.dsv41_native_forward import DRAFT, DRAFT_CAPTURE, ROWS_AND_MASK, NativeForwardTarget
from blockspec.dsv41_native_resident import NativeRankMemory


class Buffer:
    def __init__(self, size):
        self.storage = create_string_buffer(size)
        self.base = addressof(self.storage)
        self.nbytes = size


class Worker:
    def __init__(self):
        self.host_allocations = 0

    def create_buffer(self, size):
        self.host_allocations += 1
        return Buffer(size)

    def copy_to(self, device, host, *, dst_offset, nbytes):
        memmove(device.base + dst_offset, host.base, nbytes)

    def copy_from(self, host, device, *, src_offset, nbytes):
        memmove(host.base, device.base + src_offset, nbytes)


def test_rank_memory_reuses_one_staging_buffer_for_tp_and_offset_reads():
    worker = Worker()
    runtime = NativeRankMemory(worker, staging_bytes=5)
    local = [Buffer(16) for _ in range(2)]
    expected = b"0123456789"
    source = create_string_buffer(expected)
    for rank, buffer in enumerate(local):
        runtime.allocations[rank, buffer.base] = buffer
    runtime.upload_to_ranks(
        addressof(source), len(expected),
        ((rank, buffer.base, rank + 2) for rank, buffer in enumerate(local)),
    )
    for rank, buffer in enumerate(local):
        output = create_string_buffer(10)
        runtime.copy_from(addressof(output), buffer.base, 10, worker_id=rank, src_offset=rank + 2)
        assert output.raw == expected
        assert buffer.storage.raw[:rank + 2] == bytes(rank + 2)
    assert worker.host_allocations == 1


def test_native_draft_keeps_global_counts_and_routes_images_to_their_owner():
    dispatched = []
    samples = torch.arange(4 * 16 * 8, dtype=torch.int32).reshape(4, 16, 8)

    def exchange(cfg, op, inputs, outputs, **kw):
        dispatched.append((op, inputs, outputs, kw))
        return samples[:2].numpy().tobytes(), samples[2:].numpy().tobytes()

    endpoint = SimpleNamespace(exchange=exchange)
    images = dict.fromkeys(("prefill", "decode"), SimpleNamespace(fields=({},)))
    artifacts = SimpleNamespace(sample_shape=(16, 8))
    target = NativeForwardTarget(endpoint, None, artifacts, images, {}, None)
    target.staged[2].update({0: b"owner-zero", 3: b"owner-three"})
    target.sampled[2] = "old"
    target.run(2, (16, 0, 0, 16), capture_owner=12)
    opcode, messages, replies, options = dispatched.pop()
    assert opcode == DRAFT and options == {"phase": "draft"}
    assert replies == (1024, 1024)
    assert ROWS_AND_MASK.unpack_from(messages[0]) == (16, 0, 0, 16, 1)
    assert ROWS_AND_MASK.unpack_from(messages[1]) == (16, 0, 0, 16, 8)
    assert tuple(DRAFT_CAPTURE.unpack_from(msg, ROWS_AND_MASK.size) for msg in messages) == ((12,), (12,))
    size = ROWS_AND_MASK.size + DRAFT_CAPTURE.size
    assert tuple(msg[size:] for msg in messages) == (b"owner-zero", b"owner-three")
    assert target.staged[2] == {}
    assert torch.equal(target.read_sampled(2), samples)
    assert dispatched == []


def test_native_checks_first_sample_on_each_dp_in_smoke():
    target = NativeForwardTarget.__new__(NativeForwardTarget)
    target.sample_checks = {}
    target.read_sampled = lambda index: torch.arange(12, dtype=torch.int32).reshape(4, 3, 1)
    seen = []

    def check(index, dp, row, token):
        seen.append((index, dp))
        return {"row": row, "token": token}

    target.check_sample = check
    output = torch.empty((1, 1), dtype=torch.int32)
    for phase in (0, 1):
        for dp in (0, 2, 0, 2):
            target.read(phase, dp, "sampled_ids", output, row=1)
    assert seen == [(0, 0), (0, 2), (1, 0), (1, 2)]
    report = json.loads(json.dumps(target.sample_checks))
    assert report["decode"]["2"] == {"row": 1, "token": 7}
