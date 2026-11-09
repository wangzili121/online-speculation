from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.greedy import correct, verify_transaction
from blockspec.pypto_qwen38.head import merge_parameter
from blockspec.pypto_qwen38.target import NativeTarget, OrthrusRunner


@pytest.mark.parametrize("reject", range(32))
@pytest.mark.parametrize("budget", (1, 2, 17, 31, 32, 40))
def test_correction_every_rejection_and_output_limit(reject, budget):
    candidates = list(range(31))
    target = [*candidates, 100]
    if reject < 31:
        target[reject] = 200
    result = correct(candidates, target, budget=budget)
    expected = [*candidates, 100] if reject == 31 else [*candidates[:reject], 200]
    assert result.tokens == tuple(expected[:budget])
    assert result.committed_inputs == len(result.tokens)
    assert result.accepted == min(reject, budget)


@pytest.mark.parametrize("stop", range(31))
def test_stop_does_not_commit_the_emitted_anchor(stop):
    result = correct(range(31), [*range(31), 99], budget=40, stop_tokens={stop})
    assert result.tokens == tuple(range(stop + 1))
    assert result.committed_inputs == stop + 1


class ToyTarget:
    def __init__(self, fail=False):
        self.inputs, self.fail = [], fail

    def snapshot(self):
        return tuple(self.inputs)

    def restore(self, state):
        self.inputs = list(state)

    def forward(self, tokens):
        result = []
        for token in tokens:
            self.inputs.append(token)
            if self.fail and token == 3:
                raise RuntimeError("device failure")
            result.append(token + 1)
        return result


@pytest.mark.parametrize("reject", range(31))
def test_transaction_restores_and_replays_only_accepted_ar_inputs(reject):
    target = ToyTarget()
    candidates = list(range(1, 32))
    candidates[reject] = 300
    result = verify_transaction(target, 0, candidates, budget=64)
    assert result.tokens == tuple(range(1, reject + 2))
    assert target.inputs == list(range(reject + 1))


def test_transaction_exception_restores_the_original_state():
    target = ToyTarget(fail=True)
    target.inputs = [-1]
    with pytest.raises(RuntimeError, match="device failure"):
        verify_transaction(target, 0, list(range(1, 32)), budget=32)
    assert target.inputs == [-1]


def test_all_accepts_commit_the_ar_block_not_the_bonus():
    target = ToyTarget()
    result = verify_transaction(target, 0, list(range(1, 32)), budget=32)
    assert result.tokens == tuple(range(1, 33))
    assert target.inputs == list(range(32))


def test_packed_qkv_is_gathered_by_projection_section(monkeypatch):
    import blockspec.pypto_qwen38.head as module
    monkeypatch.setattr(module, "KEY_WIDTH", 2)
    monkeypatch.setattr(module, "VALUE_WIDTH", 6)
    monkeypatch.setattr(module, "tensor_schema", lambda layer, tp: {"in_proj_qkv.weight": (5, 1)})
    a = torch.tensor([[10], [20], [30], [31], [32]], dtype=torch.bfloat16)
    b = torch.tensor([[11], [21], [33], [34], [35]], dtype=torch.bfloat16)
    result = merge_parameter(0, "in_proj_qkv.weight", [a, b])
    assert result[:, 0].tolist() == [10, 11, 20, 21, 30, 31, 32, 33, 34, 35]


def test_row_parallel_projection_gathers_columns(monkeypatch):
    import blockspec.pypto_qwen38.head as module
    monkeypatch.setattr(module, "tensor_schema", lambda layer, tp: {"o_proj.weight": (2, 1)})
    a = torch.tensor([[1], [2]], dtype=torch.bfloat16)
    b = torch.tensor([[3], [4]], dtype=torch.bfloat16)
    assert torch.equal(merge_parameter(3, "o_proj.weight", [a, b]), torch.cat([a, b], dim=1))


def test_replicated_norms_must_match(monkeypatch):
    import blockspec.pypto_qwen38.head as module
    monkeypatch.setattr(module, "tensor_schema", lambda layer, tp: {"norm.weight": (2,)})
    value = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="differs across ranks"):
        merge_parameter(0, "norm.weight", [value, value + 1])
    with pytest.raises(ValueError, match="finite BF16"):
        merge_parameter(0, "norm.weight", [value, value * float("nan")])


def test_gdn_decay_parameter_preserves_native_fp32(monkeypatch):
    import blockspec.pypto_qwen38.head as module
    monkeypatch.setattr(module, "tensor_schema", lambda layer, tp: {"A_log": (2,)})
    a = torch.tensor([0.12345678, 1.2345678], dtype=torch.float32)
    b = torch.tensor([2.345678, 3.456789], dtype=torch.float32)
    actual = merge_parameter(0, "A_log", [a, b])
    assert actual.dtype == torch.float32 and torch.equal(actual, torch.cat([a, b]))
    with pytest.raises(ValueError, match="invalid draft"):
        merge_parameter(0, "A_log", [a, b.bfloat16()])


def test_native_restore_releases_new_pages_and_restores_gdn():
    pages = SimpleNamespace(length=3, pages=[7])
    resident = SimpleNamespace(pages=pages, slot=0)
    dropped, restored = [], []
    runner = SimpleNamespace(requests={"r": resident},
                             _read_state=lambda slot: ("conv", "matrix"),
                             _write_state=lambda slot, *state: restored.append(state),
                             cache=SimpleNamespace(_drop=dropped.append))
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), None)
    snapshot = target.snapshot()
    pages.pages += [8, 9]
    pages.length = 31
    target.restore(snapshot)
    assert pages.pages == [7] and pages.length == 3
    assert dropped == [8, 9] and restored == [("conv", "matrix")]


def test_native_restore_rejects_replacement_of_committed_pages():
    pages = SimpleNamespace(length=3, pages=[7])
    runner = SimpleNamespace(requests={"r": SimpleNamespace(pages=pages, slot=0)},
                             _read_state=lambda slot: (0, 0))
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), None)
    snapshot = target.snapshot()
    pages.pages[0] = 99
    with pytest.raises(RuntimeError, match="replaced committed"):
        target.restore(snapshot)


@pytest.mark.parametrize("reject", (0, 1, 15, 16, 30, None))
def test_native_transaction_across_pages_and_recurrent_replay(reject):
    class Runner:
        def __init__(self):
            self.pages = SimpleNamespace(length=3, pages=[1])
            self.requests = {"r": SimpleNamespace(pages=self.pages, slot=0)}
            self.cache = SimpleNamespace(_drop=self.drop)
            self.refs = {1: 1}
            self.state = 42

        def drop(self, page):
            assert self.refs.pop(page) == 1

        def _read_state(self, slot):
            return (self.state, self.state)

        def _write_state(self, slot, conv, recurrent):
            assert conv == recurrent
            self.state = conv

        def run_batch(self, slots):
            slot = slots[0]
            assert slot.position == self.pages.length
            end = self.pages.length + len(slot.token_ids)
            while len(self.pages.pages) * 4 < end:
                page = max(self.refs) + 1
                self.refs[page] = 1
                self.pages.pages.append(page)
            self.pages.length = end
            self.state += sum(slot.token_ids)
            self.predictions = [token + 1 for token in slot.token_ids]

        def _logit_row(self, slot, row):
            logits = torch.zeros(512)
            logits[self.predictions[row]] = 1
            return logits

    def slot_type(request, token_ids, position, phase, completes_prefill):
        return SimpleNamespace(request=request, token_ids=token_ids, position=position)

    runner = Runner()
    candidates = list(range(1, 32))
    if reject is not None:
        candidates[reject] = 300
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), slot_type)
    result = verify_transaction(target, 0, candidates, budget=32)
    expected = list(range(1, 33 if reject is None else reject + 2))
    assert list(result.tokens) == expected
    assert runner.pages.length == 3 + len(expected)
    assert runner.state == 42 + sum(range(len(expected)))
    assert set(runner.pages.pages) == runner.refs.keys()


def test_runner_buffers_only_matching_request_positions():
    native = SimpleNamespace(allow_prefix_hits=True, requests={"r": None})
    runner = OrthrusRunner(native, None, None)
    from collections import deque
    runner.pending["r"] = deque([(4, 1, 2), (5, 2, 3)])
    request = SimpleNamespace(request_id="r")
    slot = SimpleNamespace(phase="decode", request=request, position=4, token_ids=(1,))
    assert runner.run_batch([slot]) == [2]
    with pytest.raises(RuntimeError, match="no longer matches"):
        runner.run_batch([slot])
    assert len(runner.pending["r"]) == 1


def test_baseline_source_and_runtime_pins_are_enforced(monkeypatch, tmp_path):
    import blockspec.pypto_qwen38.baseline as module
    package = tmp_path / module.PACKAGE
    package.mkdir()
    (package / "pypto.pin").write_text(module.PYPTO_COMMIT + "\n")
    with monkeypatch.context() as patch:
        patch.setattr(module.subprocess, "check_output", lambda argv, **kwargs:
                      module.ONESERVING_COMMIT + "\n" if "rev-parse" in argv else "")
        assert module.checked_source(tmp_path) == package
        (package / "pypto.pin").write_text("wrong\n")
        with pytest.raises(ValueError, match="dependency pin changed"):
            module.checked_source(tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "wrong\n")
    with pytest.raises(ValueError, match="must be pinned"):
        module.checked_source(tmp_path)


def test_runtime_does_not_allow_a_pin_bypass(monkeypatch):
    from blockspec.pypto_qwen38.baseline import runtime_identity
    monkeypatch.setenv("PYPTO_SKIP_RUNTIME_PIN_CHECK", "1")
    with pytest.raises(ValueError, match="must remain enabled"):
        runtime_identity(None)


def test_native_lm_head_tp_is_not_treated_as_full_model_tp():
    with pytest.raises(ValueError, match="requires the native TP1"):
        OrthrusRunner(SimpleNamespace(device_ids=(0, 1)), None, None)


def test_neural_draft_refuses_a_stale_anchor_boundary():
    from blockspec.pypto_qwen38.draft import NeuralDraft
    draft = NeuralDraft(None, SimpleNamespace(block_size=32))
    draft.native = SimpleNamespace(requests={"r": SimpleNamespace(pages=SimpleNamespace(length=4))})
    request = SimpleNamespace(request_id="r", prompt_ids=[1, 2, 3], generated=[4])
    with pytest.raises(RuntimeError, match="history boundary"):
        draft.propose(request, 4, 32)


def test_neural_draft_does_not_allocate_or_free_weights_during_a_round():
    from blockspec.pypto_qwen38.draft import NeuralDraft
    head = SimpleNamespace(block_size=32, mask_token_id=1,
                           banks=lambda layer: [torch.zeros(1) for _ in range(9)])
    draft = NeuralDraft(None, head)

    def fail_free(handle):
        raise RuntimeError("mailbox cleanup failure")

    def run(name, arguments):
        if name == "gdn":
            raise RuntimeError("original device execution failure")

    draft.native = SimpleNamespace(
        requests={"r": SimpleNamespace(pages=SimpleNamespace(length=1, pages=[0]), slot=0)},
        rope={name: torch.zeros(512, 64) for name in ("cos", "sin")},
        worker=SimpleNamespace(alloc_tensor=lambda *a, **k: object(), free_tensor=fail_free),
        banks=[None] * 8, resident=[None] * 2)
    draft.ids = draft.cos = draft.sin = draft.table = draft.mask = None
    draft.hidden = [None, None]
    draft.weights = SimpleNamespace(for_layer=lambda layer: [None] * 9)
    draft._upload = lambda *args: None
    draft._run = run
    request = SimpleNamespace(request_id="r", prompt_ids=[5], generated=[7])
    with pytest.raises(RuntimeError, match="original device execution failure"):
        draft.propose(request, 7, 32)


def tiny_head(monkeypatch):
    from blockspec.pypto_qwen38.head import HeadSource
    head = HeadSource.__new__(HeadSource)
    head._prepared = head.shared_weights = None
    head.preparation_seconds = 0.0
    head.prepared_manifest = head.preparation_backend = None
    full = {name: torch.arange(256, dtype=torch.bfloat16) / 256
            if name.endswith("norm.weight") else torch.ones((2, 2), dtype=torch.bfloat16)
            for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                         "q_norm.weight", "k_norm.weight")}
    gdn = {name: torch.ones((2, 2), dtype=torch.bfloat16) for name in
           ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight")}
    gdn.update({"A_log": torch.arange(48, dtype=torch.bfloat16) / 48,
                "dt_bias": torch.zeros(48, dtype=torch.bfloat16),
                "norm.weight": torch.ones(128, dtype=torch.bfloat16),
                "conv1d.weight": torch.ones((2, 1, 4), dtype=torch.bfloat16)})
    monkeypatch.setattr(head, "layer", lambda layer: full if layer % 4 == 3 else gdn)
    return head


def test_head_preparation_preserves_values_and_reuses_one_shared_storage(monkeypatch):
    head = tiny_head(monkeypatch)
    expected = [head.banks(layer) for layer in range(64)]
    for layer in range(64):
        if layer % 4 != 3:
            expected[layer][5] = expected[layer][5].float()
    head.prepare()
    pool = head.shared_weights
    monkeypatch.setattr(head, "layer", lambda layer: pytest.fail("head must not be re-read"))
    head.prepare()
    assert head.shared_weights is pool and pool.is_shared()
    for layer in range(64):
        for actual, wanted in zip(head.banks(layer), expected[layer], strict=True):
            assert actual.dtype == wanted.dtype and torch.equal(actual, wanted)
            assert actual.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
            assert actual.is_shared()


class WeightWorker:
    def __init__(self, free):
        self.free = free
        self.handles, self.freed = [], []
        self.uploads = 0
        self.operations = []

    def device_memory_info(self, worker_id=0):
        self.operations.append(("memory", worker_id))
        return self.free, self.free

    def alloc_tensor(self, shape, dtype, init=None, worker_id=0):
        self.operations.append(("allocate", worker_id))
        payload = torch.empty(shape, dtype=dtype)
        handle = SimpleNamespace(shape=tuple(shape), dtype=dtype, data_ptr=payload.data_ptr(), payload=payload)
        self.handles.append(handle)
        if init is not None:
            payload.copy_(init)
            self.uploads += 1
        return handle

    def copy_to(self, dst, src, size, worker_id=0):
        self.operations.append(("copy", worker_id))
        import ctypes
        ctypes.memmove(dst, src, size)
        self.uploads += 1

    def free_tensor(self, handle, worker_id=0):
        self.operations.append(("free", worker_id))
        self.freed.append(handle)


@pytest.mark.parametrize("mode,expected", (("resident", "resident"), ("stream", "stream"), ("auto", "resident")))
def test_device_head_reuses_allocations_and_resident_head_never_reuploads(monkeypatch, mode, expected):
    from blockspec.pypto_qwen38.weights import DraftWeights
    head = tiny_head(monkeypatch)
    head.prepare()
    worker = WeightWorker(1 << 30)
    weights = DraftWeights(head, mode)
    weights.start(worker)
    count, uploads = len(worker.handles), worker.uploads
    for _ in range(2):
        for layer in range(64):
            handles = weights.for_layer(layer)
            for handle, value in zip(handles, head.banks(layer), strict=True):
                assert torch.equal(handle.payload, value)
    assert weights.storage == expected and len(worker.handles) == count
    assert worker.uploads == uploads if expected == "resident" else worker.uploads > uploads
    weights.close()
    assert len(worker.freed) == count


def test_auto_streaming_and_explicit_resident_memory_gate(monkeypatch):
    from blockspec.pypto_qwen38.weights import DraftWeights
    head = tiny_head(monkeypatch)
    head.prepare()
    worker = WeightWorker((256 << 20) + head.shared_weights.numel() // 2)
    resident = DraftWeights(head, "resident")
    with pytest.raises(RuntimeError, match="resident head requires"):
        resident.start(worker)
    assert not worker.handles
    resident.close()
    weights = DraftWeights(head)
    weights.start(worker)
    assert weights.storage == "stream"
    weights.close()


@pytest.mark.parametrize("mode", ("resident", "stream"))
def test_all_head_memory_operations_use_the_assigned_worker(monkeypatch, mode):
    from blockspec.pypto_qwen38.weights import DraftWeights

    class AssignedWorker(WeightWorker):
        def device_memory_info(self, worker_id=0):
            assert worker_id == 1
            return super().device_memory_info(worker_id)

        def alloc_tensor(self, shape, dtype, init=None, worker_id=0):
            assert worker_id == 1
            return super().alloc_tensor(shape, dtype, init, worker_id)

        def copy_to(self, dst, src, size, worker_id=0):
            assert worker_id == 1
            return super().copy_to(dst, src, size, worker_id)

        def free_tensor(self, handle, worker_id=0):
            assert worker_id == 1
            return super().free_tensor(handle, worker_id)

    head = tiny_head(monkeypatch)
    head.prepare()
    worker = AssignedWorker(1 << 30)
    weights = DraftWeights(head, mode, worker_id=1)
    weights.start(worker)
    for layer in range(64):
        assert all(torch.equal(handle.payload, value)
                   for handle, value in zip(weights.for_layer(layer), head.banks(layer), strict=True))
    weights.close()
    assert len(worker.handles) == len(worker.freed)


def test_head_free_failure_keeps_the_owner_and_remaining_allocations_for_retry(monkeypatch):
    from blockspec.pypto_qwen38.weights import DraftWeights
    head = tiny_head(monkeypatch)
    head.prepare()
    worker = WeightWorker(1 << 30)
    weights = DraftWeights(head, "resident", worker_id=1)
    weights.start(worker)
    release = worker.free_tensor
    calls = []

    def fail_once(handle, worker_id=0):
        calls.append(worker_id)
        if len(calls) == 2:
            raise RuntimeError("temporary free failure")
        return release(handle, worker_id)

    worker.free_tensor = fail_once
    with pytest.raises(RuntimeError, match="temporary free"):
        weights.close()
    assert weights.worker is worker and len(weights.handles) == len(worker.handles) - 1
    with pytest.raises(RuntimeError, match="not started"):
        weights.for_layer(0)
    weights.close()
    assert calls == [1] * len(calls)
    assert weights.worker is None and not weights.handles
    assert len(worker.freed) == len(worker.handles)


@pytest.mark.parametrize("worker_id", (0, 1))
@pytest.mark.parametrize("group_size", (1, 4))
def test_packed_stream_reuses_owned_buffers_and_copies_each_layer_or_group_once(monkeypatch, worker_id, group_size):
    import math
    import sys
    from blockspec.pypto_qwen38.weights import DraftWeights
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", SimpleNamespace(
        DataType=SimpleNamespace(BFLOAT16=torch.bfloat16, FLOAT32=torch.float32)))

    class PackedWorker(WeightWorker):
        def alloc_tensor(self, shape, dtype, init=None, worker_id=0):
            parent = super().alloc_tensor(shape, dtype, init, worker_id=worker_id)

            def view(shape, dtype, *, byte_offset):
                size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                value = parent.payload[byte_offset:byte_offset + size].view(dtype).reshape(shape)
                return SimpleNamespace(payload=value)

            parent.buffer = SimpleNamespace(tensor=view)
            return parent

    head = tiny_head(monkeypatch)
    head.prepare()
    worker = PackedWorker(1 << 30)
    weights = DraftWeights(head, "stream", "packed", worker_id=worker_id, group_size=group_size)
    weights.start(worker)
    count = 1 if group_size == 4 else 2
    assert len(worker.handles) == count
    for _ in range(2):
        for base in range(0, 64, group_size):
            banks = weights.for_group(base) if group_size == 4 else [weights.for_layer(base)]
            # 同组的三个 GDN 参数必须同时保持正确，不能全部别名到最后一次上传。
            for index, handles in enumerate(banks):
                for actual, expected in zip(handles, head.banks(base + index), strict=True):
                    assert actual.payload.dtype == expected.dtype
                    assert torch.equal(actual.payload, expected)
    assert worker.uploads == 128 // group_size
    if group_size == 4:
        with pytest.raises(RuntimeError, match="for_group"):
            weights.for_layer(0)
    weights.close()
    assert len(worker.freed) == count and not weights.packed
    assert all(owner == worker_id for _, owner in worker.operations)


def test_resident_four_layer_groups_use_distinct_parameters_without_upload(monkeypatch):
    from blockspec.pypto_qwen38.weights import DraftWeights
    head = tiny_head(monkeypatch)
    head.prepare()
    worker = WeightWorker(1 << 30)
    weights = DraftWeights(head, "resident", "packed", group_size=4)
    weights.start(worker)
    uploads = worker.uploads
    for base in range(0, 64, 4):
        for index, handles in enumerate(weights.for_group(base)):
            assert all(torch.equal(h.payload, value) for h, value in zip(handles, head.banks(base + index), strict=True))
    assert worker.uploads == uploads
    weights.close()


def test_packed_stream_rejects_a_bank_outside_the_prepared_storage(monkeypatch):
    import sys
    from blockspec.pypto_qwen38.weights import DraftWeights
    monkeypatch.setitem(sys.modules, "pypto.runtime.task_interface", SimpleNamespace(
        DataType=SimpleNamespace(BFLOAT16=torch.bfloat16, FLOAT32=torch.float32)))
    head = tiny_head(monkeypatch)
    head.prepare()
    banks = head.banks
    monkeypatch.setattr(head, "banks", lambda layer: [v.clone() for v in banks(layer)])
    worker = WeightWorker(1 << 30)
    weights = DraftWeights(head, "stream", "packed")
    with pytest.raises(ValueError, match="packed storage"):
        weights.start(worker)
    assert not worker.handles
    weights.close()


def test_host_selection_preserves_every_finite_bf16_value_and_first_tie():
    from blockspec.pypto_qwen38.greedy import host_greedy_tokens
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.bfloat16)
    finite = values[values.isfinite()]
    pairs = torch.stack((finite, finite.flip(0)), dim=-1)
    assert host_greedy_tokens(pairs) == pairs.float().argmax(-1).tolist()
    ties = finite[:, None].expand(-1, 3).contiguous()
    assert host_greedy_tokens(ties) == [0] * len(finite)
    # 逐个覆盖全部 BF16 NaN 编码和正负无穷，而非只检查一整组特殊值。
    for encoding in encodings[~values.isfinite()]:
        matrix = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        matrix.view(torch.int16)[2, 1] = encoding
        with pytest.raises(FloatingPointError, match="nonfinite"):
            host_greedy_tokens(matrix)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_host_selection_rejects_nonfinite_fp32_values(bad):
    from blockspec.pypto_qwen38.greedy import host_greedy_tokens
    values = torch.tensor([[5., bad, -1.], [0., 0., 0.]])
    with pytest.raises(FloatingPointError, match="nonfinite"):
        host_greedy_tokens(values)


@pytest.mark.parametrize("nonfinite", (False, True))
def test_verifier_reads_active_rows_in_bulk_with_the_correct_slot(nonfinite):
    import ctypes
    payload = torch.full((2, 16, 8), float("nan"), dtype=torch.bfloat16)
    host = torch.empty((16, 8), dtype=torch.bfloat16).share_memory_()
    resident = SimpleNamespace(slot=1, pages=SimpleNamespace(length=0))
    copies = []

    def run(slots):
        for row in range(len(slots[0].token_ids)):
            payload[1, row].zero_()
            payload[1, row, 1] = payload[1, row, 4] = 3  # 并列最大值仍选最小 token ID。
        if nonfinite:
            payload[1, 0, 0] = float("nan")
        resident.pages.length += len(slots[0].token_ids)

    def copy(dst, src, size, *, src_offset):
        copies.append((size, src_offset))
        ctypes.memmove(dst, src + src_offset, size)

    runner = SimpleNamespace(requests={"r": resident}, run_batch=run,
        verification_logits=host, logits=SimpleNamespace(data_ptr=payload.data_ptr()),
        worker=SimpleNamespace(copy_from=copy), _logit_row=lambda *a: pytest.fail("must use bulk readback"))
    make_slot = lambda request, token_ids, *a: SimpleNamespace(token_ids=token_ids)
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), make_slot, rows=2)
    if nonfinite:
        with pytest.raises(FloatingPointError, match="nonfinite"):
            target.forward([10, 11, 12])
    else:
        assert target.forward([10, 11, 12]) == [1, 1, 1]
        assert copies == [(32, 256), (16, 256)]


def test_snapshot_buffers_are_not_reused_while_a_snapshot_is_alive():
    import gc
    from blockspec.pypto_qwen38.target import SnapshotPool
    pool = SnapshotPool((2, 3), (2, 2))
    values = (torch.ones((2, 3), dtype=torch.bfloat16), torch.ones((2, 2), dtype=torch.float32))
    private_reads = []

    def copy(host, device, offset, to_device):
        assert offset == 0 and not to_device
        host.copy_(device)

    def read(slot):
        private_reads.append(slot)
        return tuple(v.clone() for v in values)

    runner = SimpleNamespace(requests={"r": SimpleNamespace(slot=0, pages=SimpleNamespace(length=1, pages=[0]))},
        resident=values, verification_states=pool, _copy_offset=copy, _read_state=read)
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), None)
    first = target.snapshot()
    for value in values:
        value.fill_(2)
    second = target.snapshot()
    for value in values:
        value.fill_(3)
    third = target.snapshot()  # 两个共享槽均在使用，必须获得独立副本。
    assert private_reads == [0]
    assert all(torch.equal(v, torch.ones_like(v)) for v in first.state)
    assert all(torch.equal(v, torch.full_like(v, 2)) for v in second.state)
    del first
    gc.collect()
    for value in values:
        value.fill_(4)
    fourth = target.snapshot()
    assert private_reads == [0]
    assert all(torch.equal(v, torch.full_like(v, 2)) for v in second.state)
    assert all(torch.equal(v, torch.full_like(v, 3)) for v in third.state)
    assert all(torch.equal(v, torch.full_like(v, 4)) for v in fourth.state)


def test_failed_snapshot_copy_releases_its_buffer():
    from blockspec.pypto_qwen38.target import SnapshotPool
    pool = SnapshotPool((2, 3), (2, 2))

    def fail(*args):
        raise RuntimeError("copy failed")

    runner = SimpleNamespace(requests={"r": SimpleNamespace(slot=0, pages=SimpleNamespace(length=1, pages=[0]))},
        resident=[None, None], verification_states=pool, _copy_offset=fail)
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), None)
    with pytest.raises(RuntimeError, match="copy failed"):
        target.snapshot()
    assert not pool.busy
