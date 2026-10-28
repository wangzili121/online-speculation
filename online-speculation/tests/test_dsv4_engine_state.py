import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/check_dsv4_engine_state.py"
SPEC = importlib.util.spec_from_file_location("engine_state", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.mark.parametrize("position", range(32))
def test_controlled_candidates_cover_every_rejection_and_full_acceptance(position):
    candidates = PROBE.make_candidates(19, position)
    assert len(candidates) == 31
    assert PROBE.expected_greedy_tokens(candidates, 19) == [19] * (position + 1)
    # Scheduler may shorten proposals near a request's budget or page boundary.
    prefix = candidates[:7]
    PROBE.check_scheduled_candidates(prefix, candidates)
    assert PROBE.expected_greedy_tokens(prefix, 19) == [19] * (min(position, 7) + 1)


def test_request_swap_and_invalid_rejection_are_not_accepted():
    with pytest.raises(AssertionError, match="this request"):
        PROBE.check_scheduled_candidates([23, 23], [19] * 31)
    with pytest.raises(ValueError):
        PROBE.make_candidates(19, 32)
    with pytest.raises(ValueError):
        PROBE.make_candidates(19, -1)
    assert PROBE.expected_greedy_tokens([], 19) == [19]


def test_shadow_check_requires_full_logits_and_identical_schedule():
    row = dict(kind="verify", computed=127, committed_before=127, candidates=[],
               raw_logits_sha256="hash", sampled=[19])
    events = [dict(row, phase=phase) for phase in ("shadow-real", "shadow-no-head")]
    assert PROBE.compare_shadow_logits(events)["all_target_logits_bitwise_equal"]
    changed_logits = [events[0], dict(events[1], raw_logits_sha256="different")]
    result = PROBE.compare_shadow_logits(changed_logits)
    assert not result["all_target_logits_bitwise_equal"] and result["different_steps"] == [0]
    for field, value in (("computed", 128), ("sampled", [20])):
        different = [events[0], dict(events[1], **{field: value})]
        with pytest.raises(AssertionError, match="step 0"):
            PROBE.compare_shadow_logits(different)
    with pytest.raises(AssertionError, match="complete"):
        PROBE.compare_shadow_logits(events[:1])
    with pytest.raises(AssertionError, match="complete"):
        PROBE.compare_shadow_logits([])


def test_baseline_nonrepeatability_does_not_excuse_adaptation_difference():
    worker = {key: {"all_target_logits_bitwise_equal": True} for key in
              ("shadow", "token_only_vs_real", "token_only_repeat")}
    worker["token_only_repeat"] = {"all_target_logits_bitwise_equal": False}
    result = PROBE.assess_shadow_results([worker])
    assert result["adaptation_reference_passed"]
    assert not result["reference_repeat_exact"] and not result["all_shadow_logits_exact"]
    for key in ("shadow", "token_only_vs_real"):
        broken = dict(worker, **{key: {"all_target_logits_bitwise_equal": False}})
        assert not PROBE.assess_shadow_results([worker, broken])["adaptation_reference_passed"]


def test_shadow_assessment_rejects_missing_evidence():
    with pytest.raises(ValueError, match="worker results"):
        PROBE.assess_shadow_results([])
    with pytest.raises(KeyError):
        PROBE.assess_shadow_results([{}])
