from ctypes import addressof, create_string_buffer, memmove
import json
from pathlib import Path
from runpy import run_path
import sys
from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_native_forward import (
    CLEAR, DRAFT, DRAFT_CAPTURE, READ_ROWS, ROWS_AND_MASK, ForwardNode, NativeForwardTarget, NativeSampleError,
    prepare_target_graphs,
)
from blockspec.dsv41_native_graph_pair import NativePeerMailbox
from blockspec.dsv41_native_resident import (
    NativePrefillInputPacket, NativeRankMemory, NativeResidentForward, NativeResidualInputPacket,
)


def test_native_window_failure_preserves_contract_and_stage(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    main = run_path(str(scripts / "run_dsv41_pypto_native_ar.py"))["main"]
    output, workload = tmp_path / "result.json", tmp_path / "input.jsonl"
    workload.write_text('{"id":"a","prompt":"one"}\n')
    manifest = SimpleNamespace(
        capacities=(272, 16), context=512, draft=None, online={},
        tokenizer=SimpleNamespace(encode=lambda _: SimpleNamespace(ids=[1])),
        identity=b"test", native_windows=lambda: {"prefill": 4096},
    )

    def fail(_artifacts, *, on_stage, **_connection):
        on_stage("reserving_native_windows")
        running = json.loads(output.read_text())
        assert not running["complete"] and not running["passed"]
        assert running["stage"] == "reserving_native_windows"
        raise OSError("injected device window import error")

    monkeypatch.setitem(main.__globals__, "ForwardArtifacts", lambda *_a, **_kw: manifest)
    monkeypatch.setitem(main.__globals__, "check_native_windows", fail)
    monkeypatch.setattr(sys, "argv", [
        "run", "--graphs", "unused", "--checkpoint", "unused", "--workload", str(workload),
        "--output", str(output), "--windows-only", "--endpoint", "peer", "--peer-listen-host", "peer",
    ])
    with pytest.raises(SystemExit, match="1"):
        main()

    report = json.loads(output.read_text())
    assert not report["passed"] and report["complete"]
    assert report["planned_windows"] == {"prefill": 4096}
    assert report["failure"] == dict(
        phase="reserving_native_windows", error_type="OSError", message="injected device window import error",
    )


@pytest.mark.parametrize("reusable", (True, False))
def test_prepare_targets_checks_the_pair_before_reusing_it(tmp_path, monkeypatch, reusable):
    previous = tmp_path / "previous"
    destination = tmp_path / "new" / "graphs"
    valid = previous / "last-1-tv" / "graphs"
    incompatible = previous / "last-zz" / "graphs"
    for root in (valid, incompatible):
        for phase in ("prefill", "decode"):
            path = root / phase
            path.mkdir(parents=True)
            (path / "description").write_text(root.parent.name)

    def manifest(path, *, mode, capacity, request_slots):
        assert (capacity, request_slots) == ((272 if mode == "prefill" else 16), 1)
        return dict(history_capacity=512, moe_slab_rows=64 if mode == "prefill" else 16)

    def load_program(path):
        # The first candidate has a valid Prefill paired with a simulator Decode.
        usable = path.name == "prefill" or (path.parent == valid and reusable)
        return SimpleNamespace(platform="a2a3" if usable else "a2a3sim")

    builds = []
    driver = SimpleNamespace(load_manifest=manifest, compile_generation_graphs=lambda *paths, **_: builds.append(paths))
    monkeypatch.setitem(sys.modules, "models.deepseek_v4_1_flash_a3.driver", driver)
    monkeypatch.setitem(sys.modules, "pypto.ir", SimpleNamespace(
        DistributedCompiledProgram=SimpleNamespace(from_dir=load_program),
    ))
    monkeypatch.setattr("blockspec.dsv41_native_forward.NativeRankGraph",
                        lambda *_: SimpleNamespace(window={}))
    prepare_target_graphs(destination, previous)
    if reusable:
        assert builds == []
        assert (destination / "decode" / "description").read_text() == "last-1-tv"
    else:
        assert builds == [(destination / "prefill", destination / "decode")]
        assert not destination.exists()


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


def test_native_release_coalesces_contiguous_pages_and_preserves_other_slots():
    cache = torch.arange(24, dtype=torch.int32).reshape(3, 4, 2)
    state = torch.arange(6, dtype=torch.int32).reshape(3, 1, 2)
    destination = {name: tensor.new_full(tensor.shape, -1) for name, tensor in
                   (("window_cache", cache), ("state_cache", state))}
    uploads = []

    def copy(src, size, destinations):
        uploads.append(size)
        for worker, ptr, offset in destinations:
            assert worker == 0
            memmove(ptr + offset, src, size)

    bank = NativeResidentForward.__new__(NativeResidentForward)
    bank.ranks, bank.tp = (0,), 1
    bank.initial_cache = dict(window_cache=cache, state_cache=state)
    bank.memory = SimpleNamespace(upload_to_ranks=copy)
    bank.buffers = {"prefill": [{name: SimpleNamespace(data_ptr=value.data_ptr())
                                for name, value in destination.items()}]}
    bank.clear(0, 0, [1, 0, 3])
    assert torch.equal(destination["window_cache"][:, [0, 1, 3]], cache[:, [0, 1, 3]])
    assert torch.all(destination["window_cache"][:, 2] == -1)
    assert torch.equal(destination["state_cache"], state)
    assert uploads == [16, 8] * 3 + [24]
    uploads.clear()
    bank.clear(0, 0, [0, 1, 2, 3])
    assert torch.equal(destination["window_cache"], cache)
    assert uploads == [cache.nbytes, state.nbytes]


def test_native_release_sends_one_command_with_each_hosts_owned_requests():
    calls = []
    nodes = [ForwardNode(0), ForwardNode(1)]
    for node in nodes:
        node.bank = SimpleNamespace(clear=lambda dp, slot, pages: calls.append((dp, slot, list(pages))))

    def exchange(_config, opcode, payloads, sizes):
        assert opcode == CLEAR and sizes == (0, 0)
        calls.append("dispatch")
        for node, raw in zip(nodes, payloads, strict=True):
            node.clear(memoryview(raw))

    target = NativeForwardTarget.__new__(NativeForwardTarget)
    target.config = None
    target.mailbox = SimpleNamespace(exchange=exchange)
    target.clear_many(((0, 0, [0, 1]), (2, 0, [3]), (3, 0, []), (1, 0, [2])))
    assert calls == ["dispatch", (0, 0, [0, 1]), (1, 0, [2]), (2, 0, [3]), (3, 0, [])]


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


@pytest.mark.parametrize("block", (8, 16, 32))
def test_native_draft_packet_reconstructs_repeated_engram_rows(block):
    prefix_rows = 5
    residual_bytes = max(block, 16) * 4 * 8 * 4
    engram_bytes = 2 * block * 16 * 2

    def allocate():
        image = torch.arange(residual_bytes + 32 + engram_bytes).remainder(251).to(torch.uint8)
        embeddings = image[:residual_bytes].view(torch.float32).reshape(-1, 4, 8)
        engram = image[residual_bytes + 16:-16].view(torch.bfloat16).reshape(2, block, 16)
        return image, embeddings, engram

    def packet(image, embeddings, engram):
        inputs = SimpleNamespace(images=[None, None, image],
                                 fields=[None, None, {"x_hc": embeddings, "engram_rows": engram}])
        return NativeResidualInputPacket(inputs, engram_prefix_rows=prefix_rows)

    image, embeddings, engram = allocate()
    embeddings.zero_()
    embeddings[0].fill_(3)
    embeddings[1:block].fill_(-4)
    engram[:, :prefix_rows].copy_(torch.arange(2 * prefix_rows * 16).reshape(2, prefix_rows, 16))
    engram[:, prefix_rows:].copy_(engram[:, prefix_rows - 1:prefix_rows])
    decoded, *fields = allocate()
    incoming = packet(decoded, *fields)
    decoded.fill_(0xff)
    encoded = bytearray(packet(image, embeddings, engram).pack(2))

    assert len(encoded) == incoming.size(2) == 2 * 8 * 4 + 32 + 2 * prefix_rows * 16 * 2
    incoming.stage(encoded, 2, repeat_to=block)
    assert torch.equal(image, decoded)


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
