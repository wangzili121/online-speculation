import pytest

from experiments.arllm.summarize_gsm8k_ablations import (
    RUNNER_PATH,
    RESULT_COMPATIBLE_RUNNERS,
    _implementation_provenance,
    _is_method_summary,
)


def _variant(runner: str, algorithm: str = "algorithm") -> dict[str, str]:
    return {RUNNER_PATH: runner, "src/inference_scaling/algorithm.py": algorithm}


def test_result_compatible_runners_keep_algorithm_provenance() -> None:
    runners = sorted(RESULT_COMPATIBLE_RUNNERS)

    algorithm, observed_runners, note = _implementation_provenance(
        [_variant(runners[0]), _variant(runners[1])]
    )

    assert algorithm == {"src/inference_scaling/algorithm.py": "algorithm"}
    assert observed_runners == runners
    assert "result-compatible runner semantics" in note


def test_unknown_runner_or_algorithm_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible runners"):
        _implementation_provenance([_variant("first"), _variant("second")])
    runner = next(iter(RESULT_COMPATIBLE_RUNNERS))
    with pytest.raises(ValueError, match="different algorithms"):
        _implementation_provenance(
            [_variant(runner, "first"), _variant(runner, "second")]
        )


def test_non_method_aggregate_summary_is_ignored() -> None:
    assert _is_method_summary({"method": "conditional_is", "tag": "budget"})
    assert not _is_method_summary({"comparison": {"fresh": {}, "replay": {}}})
