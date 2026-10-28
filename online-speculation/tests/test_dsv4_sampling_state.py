import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("sampling_state", SCRIPTS / "check_dsv4_sampling_state.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
sys.path.pop(0)


def test_equal_probabilities_require_full_acceptance():
    assert PROBE.check_rejection([200, 201], [200, 201, 200], 200) == 2
    with pytest.raises(AssertionError, match="equal distributions"):
        PROBE.check_rejection([200, 201], [200], 200)


@pytest.mark.parametrize("token,correction", [(210, 211), (220, 221), (230, 230)])
def test_recovery_uses_positive_difference_not_target_distribution(token, correction):
    assert PROBE.check_rejection([token, token], [token, correction], token) == 1
    wrong = token + 1 if correction == token else token
    with pytest.raises(AssertionError, match="positive"):
        PROBE.check_rejection([token, token], [token, wrong], token)


def test_prefix_and_support_are_checked():
    with pytest.raises(AssertionError, match="prefix"):
        PROBE.check_rejection([210], [211, 210], 210)
    with pytest.raises(AssertionError, match="support"):
        PROBE.check_rejection([], [999], 210)
    for values in ([], [210, 211]):
        with pytest.raises(AssertionError, match="length"):
            PROBE.check_rejection([], values, 210)


def test_frequency_gate_rejects_bias_and_too_little_evidence():
    assert PROBE.frequency_check([210] * 350 + [211] * 650, 210)["observed"] == 0.35
    with pytest.raises(AssertionError, match="mismatch"):
        PROBE.frequency_check([210] * 1000, 210)
    with pytest.raises(AssertionError, match="insufficient"):
        PROBE.frequency_check([200, 201], 200)


def test_fixtures_vary_by_request_and_slot():
    assert PROBE.fixture(200) == PROBE.fixture(200, 30)
    assert PROBE.fixture(210, 0) != PROBE.fixture(210, 1)
    assert PROBE.fixture(210, 0) != PROBE.fixture(230, 0)


def test_failure_injection_is_bounded_and_keeps_native_preemption(monkeypatch, tmp_path):
    class NativeScheduler:
        def __init__(self):
            self.running = [object()]
            self.kv_cache_manager = SimpleNamespace(allocate_slots=lambda *a, **k: "allocated")
            self.preempted = []

        def _preempt_request(self, request, timestamp, **kwargs):
            self.preempted.append((request.request_id, timestamp, kwargs))

    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = NativeScheduler
    request_module = ModuleType("vllm.v1.request")
    request_module.RequestStatus = SimpleNamespace(RUNNING="running")
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, request_module.__name__, request_module)
    monkeypatch.setenv("ORTHRUS_TEST_ALLOCATION_LOG", str(tmp_path / "events.jsonl"))
    spec = importlib.util.spec_from_file_location("test_scheduler", SCRIPTS / "dsv4_test_scheduler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.AllocationFailureScheduler()
    allocate = scheduler.kv_cache_manager.allocate_slots
    ordinary = SimpleNamespace(status="running", num_output_tokens=65, max_tokens=512,
                               request_id="ordinary", num_computed_tokens=100)
    scheduler.running.append(ordinary)
    assert allocate(ordinary, 32) == "allocated"
    for i in range(5):
        request = SimpleNamespace(status="running", num_output_tokens=65, max_tokens=1536,
                                  request_id=str(i), num_computed_tokens=100)
        scheduler.running[-1] = request
        assert allocate(request, 32) == (None if i < 4 else "allocated")
        assert allocate(request, 32) == "allocated"
        scheduler._preempt_request(request, 12.0, drop_stale_output=True)
    assert len(scheduler.preempted) == 5
    assert all(row[2] == {"drop_stale_output": True} for row in scheduler.preempted)
    assert len(scheduler._injected_requests) == 4
