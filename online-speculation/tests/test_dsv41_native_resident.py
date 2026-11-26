from ctypes import addressof, create_string_buffer, memmove
import json
from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_native_forward import (
    DRAFT, DRAFT_CAPTURE, READ_ROWS, ROWS_AND_MASK, ForwardNode, NativeForwardTarget, NativeSampleError,
)
from blockspec.dsv41_native_graph_pair import NativePeerMailbox
from blockspec.dsv41_native_resident import NativePrefillInputPacket, NativeRankMemory, NativeResidualInputPacket


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


def test_native_mailbox_copies_large_frames_at_the_matching_source_and_destination_offsets():
    payload = b"0123456789abcdefghijklmn"
    mailbox = NativePeerMailbox.__new__(NativePeerMailbox)
    mailbox.remote_staging = create_string_buffer(payload, len(payload))
    mailbox.remote = create_string_buffer(len(payload))
    mailbox.remote_chunk_size = 10
    calls = []

    def send(handle, address, nbytes, *, offset):
        calls.append((offset, nbytes))
        memmove(addressof(handle) + offset, address, nbytes)

    def receive(handle, address, nbytes, *, offset):
        memmove(address, addressof(handle) + offset, nbytes)

    mailbox._copy_remote(send, len(payload))
    assert calls == [(0, 10), (10, 10), (20, 4)]
    assert mailbox.remote.raw == payload
    mailbox.remote_staging.raw = bytes(len(payload))
    mailbox._copy_remote(receive, len(payload))
    assert mailbox.remote_staging.raw == payload


@pytest.mark.parametrize(("rows", "repeat_to"), ((2, 8), (2, 16), (2, 32), (1, None), (8, None), (16, None)))
def test_native_residual_packet_reconstructs_the_full_input_image(rows, repeat_to):
    def allocate():
        residual_bytes = max(16, repeat_to or rows) * 4 * 32 * 4
        image = torch.empty(residual_bytes + 64, dtype=torch.uint8)
        residual = image[:residual_bytes].view(torch.float32).reshape(-1, 4, 32)
        return image, residual, NativeResidualInputPacket(
            SimpleNamespace(images=[None, None, image], fields=[None, None, {"x_hc": residual}]),
        )

    source, streams, outgoing = allocate()
    target, _, incoming = allocate()
    stream_bytes = streams.nbytes
    source[stream_bytes:] = torch.arange(64, dtype=torch.uint8)
    for value in (5.0, -1.0):
        embeddings = torch.arange(rows * 32, dtype=torch.float32).view(rows, 32) + value
        streams[:rows].copy_(embeddings[:, None, :])
        if repeat_to is not None:
            streams[rows:repeat_to].copy_(embeddings[-1])
        streams[repeat_to or rows:].zero_()
        target.fill_(255)
        packet = bytearray(outgoing.pack(rows))
        assert len(packet) == incoming.size(rows) == rows * 32 * 4 + 64
        incoming.stage(packet, rows, repeat_to=repeat_to)
        assert torch.equal(source, target)


def test_native_prefill_packet_restore_padding_without_touching_control_bytes():
    def allocate():
        image = torch.arange(352).remainder(251).to(torch.uint8)
        fields = {
            "embedding": image[32:96].view(torch.bfloat16).view(4, 8),
            "engram_lookup": image[128:192].view(torch.bfloat16).view(2, 4, 4),
            "window_indices": image[224:320].view(torch.int32).view(2, 4, 3),
        }
        return image, fields, NativePrefillInputPacket(
            SimpleNamespace(images=[None, None, image], fields=[None, None, fields]),
        )

    source, fields, sender = allocate()
    target, _, receiver = allocate()
    for rows in (4, 1, 3, 2):
        fields["embedding"][:rows].fill_(rows)
        fields["embedding"][rows:].zero_()
        fields["engram_lookup"][:, :rows].fill_(-rows)
        fields["engram_lookup"][:, rows:].zero_()
        fields["window_indices"][:, :rows].fill_(5 + rows)
        fields["window_indices"][:, rows:].fill_(-1)
        packet = memoryview(sender.pack(rows))
        assert len(packet) == receiver.size(rows) == len(source) - (4 - rows) * 56
        target.fill_(233)
        receiver.stage(packet, rows)
        assert torch.equal(source, target)


def test_native_draft_keeps_global_counts_and_routes_images_to_their_owner():
    dispatched = []
    samples = torch.arange(4 * 16 * 8, dtype=torch.int32).reshape(4, 16, 8)

    def exchange(cfg, op, inputs, outputs, **kw):
        dispatched.append((op, inputs, outputs, kw))
        return samples[:2].numpy().tobytes(), samples[2:].numpy().tobytes()

    target = NativeForwardTarget.__new__(NativeForwardTarget)
    target.mailbox = SimpleNamespace(exchange=exchange)
    target.config = None
    target.sample_shape = (16, 8)
    target.staged = ({}, {}, {0: b"owner-zero", 1: None, 2: None, 3: b"owner-three"})
    target.sampled = [None, None, "old"]
    target.run(2, (16, 0, 0, 16), capture_owner=12)
    opcode, packets, replies, options = dispatched.pop()
    assert tuple(parts[1:] for parts in packets) == ((b"owner-zero",), (b"owner-three",))
    messages = tuple(b"".join(parts) for parts in packets)
    assert opcode == DRAFT and options == {"phase": "draft"}
    assert replies == (1024, 1024)
    assert ROWS_AND_MASK.unpack_from(messages[0]) == (16, 0, 0, 16, 1 | 1 << 5)
    assert ROWS_AND_MASK.unpack_from(messages[1]) == (16, 0, 0, 16, 8 | 1 << 6)
    assert tuple(DRAFT_CAPTURE.unpack_from(msg, ROWS_AND_MASK.size) for msg in messages) == ((12,), (12,))
    size = ROWS_AND_MASK.size + DRAFT_CAPTURE.size
    assert tuple(msg[size:] for msg in messages) == (b"owner-zero", b"owner-three")
    assert target.staged[2] == {}
    assert torch.equal(target.read_sampled(2), samples)
    assert dispatched == []

    # Replay the same packet on each of the node-local Draft receivers.
    stages = []
    for node_id, (msg, active, idle) in enumerate(zip(messages, (0, 3), (1, 2), strict=True)):
        node = ForwardNode(node_id, {"draft": {}})
        draft_inputs = SimpleNamespace(images=[None, None, SimpleNamespace(nbytes=len(msg[size:]))])
        node.bank = SimpleNamespace(tp=4, upload_idle=lambda io, dp: stages.append(("idle", dp, io)))
        node.draft_packet = SimpleNamespace(size=lambda _rows: len(msg[size:]))
        count_buffer = bytearray(16 * 4)
        node.draft = SimpleNamespace(
            draft=SimpleNamespace(inputs=draft_inputs), capture_args={}, arguments={"ids": None},
            capture_parameter="capture", count_buffer=SimpleNamespace(host_view=lambda: memoryview(count_buffer)),
        )
        graph = SimpleNamespace(submit=lambda *_args, **_kw: stages.append(("submit", node_id)))
        node.artifacts = SimpleNamespace(draft=SimpleNamespace(graph=graph), sample_parameters={"draft": "ids"})
        node.stage_draft = lambda dp, image, **_kw: stages.append(("active", dp, image))
        node.bind_samples = lambda *_args: None
        rpc = SimpleNamespace(tensor=lambda _idx: None, scalar=lambda _idx: 0)
        node.forward(None, rpc, None, msg, 1024, "draft")
        assert stages == [("active", active, msg[size:]), ("idle", idle, draft_inputs), ("submit", node_id)]
        stages.clear()


@pytest.mark.parametrize(("bad_logit", "reason"), (
    (float("nan"), "non-finite logits"), (3.0, "sampled ID disagrees with greedy logits"),
))
def test_native_smoke_reports_bad_second_row_in_later_verify_dispatch(bad_logit, reason):
    target = NativeForwardTarget.__new__(NativeForwardTarget)
    target.artifacts = SimpleNamespace(vocab_size=3)
    target.sample_checks = {}
    target.read_sampled = lambda index: torch.ones((4, 3, 1), dtype=torch.int32)
    logits = torch.tensor([[0.0, 2.0, 0.0]] * 2)

    def read(index, dp, out, *, row, capture_owner):
        assert (index, dp, row, capture_owner) == (1, 2, 1, -1)
        out.copy_(logits)

    target.read_logits = read
    output = torch.empty((2, 1), dtype=torch.int32)
    target.read(1, 2, "sampled_ids", output, row=1)
    logits[1, 2] = bad_logit
    with pytest.raises(NativeSampleError) as failure:
        target.read(1, 2, "sampled_ids", output, row=1)
    assert failure.value.evidence == dict(phase="decode", dp=2, row=2, token=1, reason=reason)
    report = json.loads(json.dumps(target.sample_checks))
    assert report["decode"]["2"] == {"dispatches": 1, "rows": 2}


def test_native_draft_diagnostics_read_the_captured_logits_on_the_owner_only():
    node = ForwardNode(1)
    calls = []
    ordinary = [SimpleNamespace(shape=(4, 3), dtype=torch.float32, data_ptr=ptr) for ptr in node.ranks]
    captured = SimpleNamespace(shape=(4, 3), dtype=torch.float32, data_ptr=100)

    def copy_from(_dst, src, size, *, worker_id, src_offset):
        calls.append((src, size, worker_id, src_offset))

    node.bank = SimpleNamespace(tp=4, memory=SimpleNamespace(copy_from=copy_from))
    node.draft = SimpleNamespace(draft=SimpleNamespace(ranks=[{"logits": tensor} for tensor in ordinary]),
                                 capture={8: {"logits": captured}})
    for dp in (2, 3):
        node.read(READ_ROWS.pack(2, dp, 1, 8), bytearray(24))
    assert calls == [(captured.data_ptr, 24, 0, 12), (ordinary[4].data_ptr, 24, 4, 12)]
