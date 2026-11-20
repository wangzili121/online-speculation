"""防止单基线收益、诊断和不完整流被判为双基线验收通过。"""

import pytest

from blockspec.pypto_qwen38.online_study_metrics import confirmation_metrics


def case(ar=100, frozen=100, online=110):
    return dict(ar_tokens=ar, ar_seconds=1, frozen_tokens=frozen, frozen_seconds=1,
                online_tokens=online, online_seconds=1, continuous=True, full_cohort=True,
                resource_policy_ok=True, performance_run=True)


def test_requires_both_baselines_in_every_repeat():
    assert confirmation_metrics([case()] * 3)["target_met"]
    assert not confirmation_metrics([case(frozen=108)] * 3)["target_met"]
    assert not confirmation_metrics([case(ar=108)] * 3)["target_met"]
    assert not confirmation_metrics([case(online=120), case(online=120), case(online=104)])["target_met"]


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_requires_exactly_three_confirmations(count):
    assert not confirmation_metrics([case()] * count)["target_met"]


def test_exactly_five_percent_is_not_over_five_percent():
    assert not confirmation_metrics([case(online=105)] * 3)["target_met"]


@pytest.mark.parametrize("flag", ["continuous", "full_cohort", "resource_policy_ok", "performance_run"])
def test_excludes_capped_partial_extra_card_or_diagnostic_runs(flag):
    row = case(); row[flag] = False
    assert not confirmation_metrics([case(), case(), row])["target_met"]


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timing_fails_closed(seconds):
    row = case(); row["online_seconds"] = seconds
    assert not confirmation_metrics([case(), case(), row])["target_met"]


def test_pooled_rate_uses_total_tokens_over_total_time():
    left = case(ar=1000, frozen=1000, online=1); left["online_seconds"] = .0009
    right = case(ar=1, frozen=1, online=1000); right["online_seconds"] = 900
    result = confirmation_metrics([left, right, left])
    assert all(g > .05 for g in result["individual_gains"])
    assert result["pooled_online_tok_s"] == pytest.approx(1002 / 900.0018)
    assert not result["target_met"]
