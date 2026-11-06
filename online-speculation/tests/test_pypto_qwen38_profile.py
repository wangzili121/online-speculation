import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("profile_summary", Path(__file__).parents[1] / "scripts/summarize_qwen38_pypto_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(start, duration, epoch=1):
    return dict(ph="X", cat="event", name="attention(block)", ts=start, dur=duration, args=dict(run_epoch=epoch))


def test_device_window_accounts_for_overlap_and_excludes_non_task_views():
    trace = dict(traceEvents=[event(10, 20), event(15, 40), dict(ph="X", cat="host", ts=0, dur=1000)])
    actual = module.summarize(trace)
    assert actual["device_task_window_us"] == 45  # 并行 block 不能按 20+40 累加。
    assert actual["block_records"] == 2


@pytest.mark.parametrize("events", [[], [event(0, float("nan"))], [event(0, -1)], [event(0, 1), event(2, 1, 2)]])
def test_incomplete_or_mixed_profiles_are_rejected(events):
    with pytest.raises(ValueError):
        module.summarize(dict(traceEvents=events))
