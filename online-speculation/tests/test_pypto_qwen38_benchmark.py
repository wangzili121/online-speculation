"""The acceptance fixture must preserve real draft work and failures."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).parents[1] / "scripts/benchmark_qwen38_pypto.py"
spec = importlib.util.spec_from_file_location("pypto_speed_benchmark", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_oracle_replaces_tokens_after_running_and_measuring_the_real_draft():
    calls = []

    def propose(request, anchor, width):
        calls.append((anchor, width))
        return [99] * (width - 1)

    draft = SimpleNamespace(propose=propose, last_timings={"weights_seconds": 2.1, "device_seconds": .35})
    meter = module.DraftMeter(draft)
    meter.mode, meter.reference = "oracle_keep_real_draft", [10, 20, 30, 40]
    assert meter(SimpleNamespace(generated=[10]), 10, 4) == [20, 30, 40]
    assert calls == [(10, 4)]
    assert meter.calls == 1 and meter.candidates == 3
    assert meter.phases == draft.last_timings


def test_oracle_does_not_hide_a_failure_in_real_draft_execution():
    def fail(*args):
        raise RuntimeError("device computation failed")

    meter = module.DraftMeter(SimpleNamespace(propose=fail))
    meter.mode, meter.reference = "oracle_keep_real_draft", [10, 20, 30]
    with pytest.raises(RuntimeError, match="device computation failed"):
        meter(SimpleNamespace(generated=[10]), 10, 3)


@pytest.mark.parametrize("anchor,width", [(99, 2), (10, 4)])
def test_oracle_rejects_a_different_prefix_or_incomplete_reference(anchor, width):
    meter = module.DraftMeter(SimpleNamespace(propose=lambda *args: [0] * (width - 1), last_timings={}))
    meter.mode, meter.reference = "oracle_keep_real_draft", [10, 20]
    with pytest.raises(AssertionError, match="anchor|cover"):
        meter(SimpleNamespace(generated=[10]), anchor, width)
