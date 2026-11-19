"""Keep capped learning separate from continuous full-stream measurements."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/benchmark_qwen38_oneserving_tp2_online.py"
spec = importlib.util.spec_from_file_location("tp2_online_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_continuous_measurement_rejects_a_step_cap():
    assert benchmark.update_policy(None, require_continuous=True) == "continuous_periodic"
    assert benchmark.update_policy(64) == "capped_then_frozen"
    with pytest.raises(ValueError, match="cannot have a step cap"):
        benchmark.update_policy(64, require_continuous=True)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_update_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.update_policy(limit)


def test_full_stream_and_active_window_are_reconstructible():
    rows = [dict(seconds=2.0, tokens=[[1, 2], [3]],
                 stats=dict(rounds=2, updates=1, publications=1),
                 learner=dict(rounds=2, optimizer_version=161, published_version=161)),
            dict(seconds=1.0, tokens=[[4], [5, 6, 7]],
                 stats=dict(rounds=2, updates=0, publications=0),
                 learner=dict(rounds=2, optimizer_version=161, published_version=161))]
    timeline = benchmark.request_timeline(rows)
    assert [item["requests"] for item in timeline] == [[1, 2], [3, 4]]
    assert sum(sum(item["output_tokens"]) for item in timeline) == 7
    assert sum(item["seconds"] for item in timeline) == 3.0
    assert timeline[0]["stats"]["updates"] == 1
    assert timeline[1]["learner"]["rounds"] == 2
