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
