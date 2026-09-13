from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from inference_scaling.experimental.arllm.streaming_is import (
    FrozenStreamingISEstimator,
    ordinary_importance_log_weight,
)
from inference_scaling.shared.rng import SeedStream


def _filled(order):
    estimator = FrozenStreamingISEstimator(2)
    estimator.add_history("history-0", 0, math.log(2.0))
    estimator.freeze((("fresh-0",), ("fresh-1a", "fresh-1b")))
    values = {
        "fresh-0": (0, math.log(4.0)),
        "fresh-1a": (1, math.log(1.0)),
        "fresh-1b": (1, math.log(3.0)),
    }
    for sample_id in order:
        candidate, value = values[sample_id]
        estimator.consume_fresh(sample_id, candidate, value)
    return estimator


def test_streaming_is_is_order_independent_after_the_budget_is_frozen() -> None:
    left = _filled(("fresh-0", "fresh-1a", "fresh-1b"))
    right = _filled(("fresh-1b", "fresh-0", "fresh-1a"))

    assert left.final_log_weights() == pytest.approx(right.final_log_weights())
    assert left.final_log_weights() == pytest.approx((math.log(3.0), math.log(2.0)))
    assert left.snapshot().effective_sample_sizes == pytest.approx((1.8, 1.6))
    assert left.select(SeedStream(7), "step") == right.select(SeedStream(7), "step")


def test_streaming_is_rejects_unfrozen_unknown_or_duplicate_evaluations() -> None:
    estimator = FrozenStreamingISEstimator(1)
    with pytest.raises(RuntimeError, match="before the design"):
        estimator.consume_fresh("fresh", 0, 0.0)
    estimator.freeze((("fresh",),))
    with pytest.raises(ValueError, match="not in the frozen design"):
        estimator.consume_fresh("other", 0, 0.0)
    estimator.consume_fresh("fresh", 0, 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        estimator.consume_fresh("fresh", 0, 0.0)


def test_ordinary_importance_weight_is_unclipped() -> None:
    value = ordinary_importance_log_weight(
        reward=2.0,
        reward_temperature=0.5,
        target_logprob=-3.0,
        behavior_logprob=-5.0,
    )
    assert value == pytest.approx(6.0)


def test_streaming_is_accepts_parallel_reward_completions_safely() -> None:
    estimator = FrozenStreamingISEstimator(2)
    sample_ids = tuple(f"fresh-{index}" for index in range(40))
    estimator.freeze((sample_ids[::2], sample_ids[1::2]))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                estimator.consume_fresh,
                sample_id,
                index % 2,
                math.log(index + 1),
            )
            for index, sample_id in enumerate(sample_ids)
        ]
        for future in futures:
            future.result()

    snapshot = estimator.snapshot()
    assert snapshot.complete
    assert snapshot.received_fresh == len(sample_ids)
    assert snapshot.contribution_counts == (20, 20)


def test_streaming_is_rolls_back_a_failed_update_callback() -> None:
    fail = True

    def callback(_snapshot, _contribution):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("callback failed")

    estimator = FrozenStreamingISEstimator(1, on_update=callback)
    estimator.freeze((("fresh",),))
    with pytest.raises(RuntimeError, match="callback failed"):
        estimator.consume_fresh("fresh", 0, 0.0)

    assert estimator.snapshot().received_fresh == 0
    assert estimator.snapshot().contribution_counts == (0,)
    estimator.consume_fresh("fresh", 0, 0.0)
    assert estimator.snapshot().complete
