import importlib.util
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).parents[1] / "scripts"
watch_spec = importlib.util.spec_from_file_location("watch_glm53_validation", scripts / "watch_glm53_validation.py")
watch = importlib.util.module_from_spec(watch_spec)
watch_spec.loader.exec_module(watch)
sys.modules.setdefault("watch_glm53_validation", watch)
spec = importlib.util.spec_from_file_location("watch_npu_resources", scripts / "watch_npu_resources.py")
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def snapshot(*, busy=(), high_memory=()):
    rows = []
    for card in range(8):
        memory = 59000 if card in high_memory else 3500
        rows.extend([f"| {card} 910B3 | OK | 90 |",
                     f"| 0 | 0000:C1:00.0 | 0 0 / 0 {memory} / 65536 |",
                     f"| {card} 0 123 VLLMWorker_TP 57000 |" if card in busy else
                     f"| No running processes found in NPU {card} |"])
    return "\n".join(rows)


@pytest.mark.parametrize("busy,high_memory,expected", [
    ((), (), list(range(8))),
    ((0, 1), (), list(range(2, 8))),
    ((4, 5), (0, 1), [2, 3, 6, 7]),
])
def test_resource_probe_requires_both_empty_process_list_and_low_memory(monkeypatch, busy, high_memory, expected):
    calls = []

    def read(command, **kwargs):
        calls.append(command)
        return snapshot(busy=busy, high_memory=high_memory)

    monkeypatch.setattr(resources.subprocess, "check_output", read)
    state = resources.probe("test-host")
    assert state["idle"] == expected
    assert state["unavailable"] == sorted(set(range(8)) - set(expected))
    assert state["eight_cards_idle"] == (len(expected) == 8)
    assert state["action"] == "observe_only"
    assert calls == [["/usr/local/bin/npu-smi", "info"]]


def test_incomplete_snapshot_cannot_claim_a_window(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *a, **k: "incomplete")
    with pytest.raises(ValueError, match="incomplete"):
        resources.probe("test-host")
