from __future__ import annotations

import pytest

from experiments.dllm.gsm8k_replay_benchmark import summarize


def _record(index: int, fresh_correct: bool, warm_correct: bool):
    return {
        "problem_index": index,
        "fresh_only": {
            "correct": fresh_correct,
            "seconds": 10.0,
            "main_flops": 100.0,
        },
        "warm_replay": {
            "correct": warm_correct,
            "online_seconds": 5.0,
            "end_to_end_seconds": 12.0,
            "online_main_flops": 30.0,
            "online_proposal_flops": 10.0,
            "cache_build_main_flops": 50.0,
            "cache_build_proposal_flops": 20.0,
            "history_used": 2,
            "fresh_used": 1,
        },
    }


def test_replay_summary_names_each_speedup_and_compute_baseline_explicitly():
    summary = summarize((_record(0, True, True), _record(1, False, True)))

    assert summary["fresh_only"]["accuracy"] == 0.5
    assert summary["warm_replay"]["accuracy"] == 1.0
    assert summary["comparisons"]["online_wall_clock_speedup_vs_fresh_only"] == 2.0
    assert summary["comparisons"]["end_to_end_wall_clock_speedup_vs_fresh_only"] == 10 / 12
    assert summary["comparisons"]["online_compute_reduction_vs_fresh_only"] == 0.6
    assert summary["comparisons"]["end_to_end_compute_reduction_vs_fresh_only"] == pytest.approx(
        -0.1
    )
