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


def test_logit_diagnostics_distinguish_future_rows_and_argmax_changes():
    import torch

    reference = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    future = PROBE.logit_errors(reference, reference + torch.tensor([[0.0, 0.0], [0.0, 0.125]]))
    assert future["max_abs"] == 0.125 and future["first_row_max_abs"] == 0
    assert future["changed_argmax_rows"] == 0 and not future["first_row_argmax_changed"]
    tie = PROBE.logit_errors(reference, reference + torch.tensor([[0.0, 0.125], [0.0, 0.0]]))
    assert tie["first_row_argmax_changed"] and tie["changed_argmax_rows"] == 1
    assert tie["reference_first_token"] == 0 and tie["actual_first_token"] == 1
    with pytest.raises(ValueError, match="matching nonempty"):
        PROBE.logit_errors(reference, reference[:1])


def test_token_only_shadow_does_not_advance_discarded_prefill():
    from types import SimpleNamespace

    import torch

    worker = PROBE.EngineStateWorker()
    worker.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        requests={key: SimpleNamespace(sampling_params=SimpleNamespace(allowed_token_ids=[token]))
                  for key, token in (("a", 19), ("b", 23))},
        discard_request_indices=SimpleNamespace(np=[1]), num_discarded_requests=1,
    )
    worker._state_probe = dict(phase="shadow-token-only-1", plans={"19": [0, 31], "23": [0, 31]},
                               rounds={}, pending={})
    kwargs = dict(next_token_ids=torch.tensor([19, 23]),
                  common_attn_metadata=SimpleNamespace(seq_lens=torch.tensor([127, 128])))
    worker._probe_propose(None, **kwargs)
    assert worker._state_probe["rounds"] == {"a": 1}
    worker.model_runner.num_discarded_requests = 0
    result = worker._probe_propose(None, **kwargs)
    assert worker._state_probe["rounds"] == {"a": 2, "b": 1}
    assert result[0].tolist() == [19] * 31
    assert result[1].tolist() == [24] + [23] * 30
