"""设备收尾的主机协议、异常清理及快照所有权。kernel 数值由 NPU 脚本核对。"""

import gc
from types import SimpleNamespace

import pytest

from blockspec.pypto_qwen38.serving import DeviceServing, DeviceState
from blockspec.pypto_qwen38.target import NativeTarget


def test_device_selector_reads_only_small_result_and_rejects_nonfinite():
    serving = DeviceServing(17)
    calls = []

    def run(name, *args):
        calls.append((name, args))
        for row in range(3):
            serving.result_values[row], serving.result_values[32 + row] = row + 7, 1

    def copy(dst, src, size):
        assert size == 256

    serving.run = run
    serving.result = SimpleNamespace(data_ptr=1)
    serving.native = SimpleNamespace(worker=SimpleNamespace(copy_from=copy))
    assert serving.select("logits", 112, 3, target=True) == [7, 8, 9]
    assert calls == [("target_select", ("logits", serving.result, 112, 3))]
    serving.run = lambda *a: None
    serving.result_values[33] = 0
    with pytest.raises(FloatingPointError, match="target"):
        serving.select("logits", 112, 3, target=True)


@pytest.mark.parametrize("begin,width,target", [(-1, 1, False), (0, 0, False), (0, 33, True), (127, 2, True), (1, 32, False)])
def test_device_selector_rejects_out_of_bounds_before_dispatch(begin, width, target):
    serving = DeviceServing(17)
    serving.run = lambda *a: pytest.fail("invalid rows must not dispatch")
    with pytest.raises(ValueError, match="rows"):
        serving.select(None, begin, width, target=target)


def test_failed_save_releases_device_snapshot_slot():
    serving = DeviceServing(17)
    serving.native = SimpleNamespace(resident=["conv", "recurrent"])
    serving.states = ["saved_conv", "saved_rec"]

    def fail(*args):
        raise RuntimeError("copy failure")

    serving.run = fail
    with pytest.raises(RuntimeError, match="copy failure"):
        serving.acquire(7)
    assert not serving.busy


def test_device_snapshot_lifetime_nested_pool_and_owner():
    serving = DeviceServing(17)
    resident = SimpleNamespace(slot=7, pages=SimpleNamespace(length=16, pages=[1]))
    calls = []
    native = SimpleNamespace(requests={"r": resident}, resident=["conv", "recurrent"],
                             cache=SimpleNamespace(_drop=lambda p: calls.append(("drop", p))))
    serving.native, serving.states = native, ["saved_conv", "saved_rec"]
    serving.run = lambda name, *args: calls.append((name, args))
    native.device_serving = serving
    target = NativeTarget(native, SimpleNamespace(request_id="r"), None)
    first, second = target.snapshot(), target.snapshot()
    assert serving.busy == {0, 1} and serving.acquire(0) is None
    assert isinstance(first.state, DeviceState)
    resident.pages.length = 33
    resident.pages.pages.append(2)
    target.restore(first)
    assert resident.pages.length == 16 and resident.pages.pages == [1]
    assert calls[-2:] == [("restore", ("saved_conv", "saved_rec", "conv", "recurrent", 0, 7)), ("drop", 2)]
    del first
    gc.collect()
    assert serving.busy == {1}
    with pytest.raises(ValueError, match="belong"):
        serving.restore(DeviceState(object(), 1), 0)
    del second
    gc.collect()
    assert not serving.busy


def test_device_verifier_avoids_duplicate_selection_and_full_logits_copy():
    calls = []
    resident = SimpleNamespace(slot=7, pages=SimpleNamespace(length=0))
    runner = SimpleNamespace(requests={"r": resident}, logits="logits",
        device_serving=SimpleNamespace(select=lambda *a, **k: calls.append((a, k)) or [4] * a[2]),
        run_verifier=lambda slots: calls.append(("forward", slots[0].token_ids)))
    slot = lambda request, tokens, *a: SimpleNamespace(token_ids=tokens)
    target = NativeTarget(runner, SimpleNamespace(request_id="r"), slot, rows=2)
    assert target.forward([10, 11, 12]) == [4, 4, 4]
    assert calls == [("forward", (10, 11)), (("logits", 112, 2), {"target": True}),
                     ("forward", (12,)), (("logits", 112, 1), {"target": True})]


@pytest.mark.parametrize("mode", ("copy", "already_compact", "full_pool", "copy_failure"))
def test_device_compaction_moves_state_before_committing_slot_metadata(mode):
    serving = DeviceServing(17)
    initial = 0 if mode == "already_compact" else 7
    resident = SimpleNamespace(slot=initial)
    slot_ids = [None] * 8
    slot_ids[initial] = "r"
    native = SimpleNamespace(slot_ids=slot_ids, requests={"r": resident}, resident=["conv", "rec"])
    serving.native, serving.states = native, ["saved_conv", "saved_rec"]
    calls = []

    def run(name, *args):
        calls.append((name, args))
        if mode == "copy_failure":
            raise RuntimeError("copy failure")

    serving.run = run
    if mode == "full_pool":
        serving.busy = {0, 1}
    slots = [SimpleNamespace(request=SimpleNamespace(request_id="r"))]
    if mode in {"full_pool", "copy_failure"}:
        with pytest.raises(RuntimeError):
            serving.compact(slots)
        assert resident.slot == 7 and native.slot_ids is slot_ids
        assert serving.busy == ({0, 1} if mode == "full_pool" else set())
    else:
        serving.compact(slots)
        assert resident.slot == 0 and native.slot_ids == ["r"] + [None] * 7
        assert not serving.busy
        assert calls == ([] if mode == "already_compact" else [
            ("save", ("conv", "rec", "saved_conv", "saved_rec", 7, 0)),
            ("restore", ("saved_conv", "saved_rec", "conv", "rec", 0, 0))])
