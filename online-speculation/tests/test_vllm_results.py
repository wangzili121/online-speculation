import json

import pytest

from blockspec.vllm_ascend.results import load_trials, rank_trials


def trial(root, name, *, throughput, workload="same", preemptions=0,
          success=1.0):
    directory = root / name
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "trial": {
            "trial_id": name, "requests": 64, "max_num_seqs": 64,
            "max_num_batched_tokens": 65536,
            "gpu_memory_utilization": 0.94,
            "graph_mode": "FULL_AND_PIECEWISE", "async_scheduling": 0,
        },
        "inputs": {"workload_sha256": workload},
    }))
    (directory / "result.json").write_text(json.dumps({
        "summary": {
            "success_rate": success, "request_throughput": throughput,
            "output_token_throughput": throughput * 128,
            "slo_goodput": 0.0,
            "ttft_seconds": {"p95": 2.0},
            "tpot_seconds": {"p95": 0.04},
        },
        "engine": {
            "counter_delta": {
                "vllm:num_preemptions_total": preemptions,
                "vllm:spec_decode_num_accepted_tokens_total": 25,
                "vllm:spec_decode_num_draft_tokens_total": 100,
            },
            "max_running": 20, "max_waiting": 44,
            "max_kv_cache_usage": 0.98,
        },
    }))


def test_trials_are_ranked_and_failed_runs_are_excluded(tmp_path):
    trial(tmp_path, "slow", throughput=1.0)
    trial(tmp_path, "fast", throughput=2.0, preemptions=1)
    trial(tmp_path, "failed", throughput=3.0, success=0.9)
    rows = load_trials(tmp_path)
    assert [row["trial_id"] for row in rank_trials(rows)] == ["fast", "slow"]
    assert [row["trial_id"] for row in rank_trials(
        rows, require_no_preemption=True)] == ["slow"]
    assert rows[0]["acceptance_rate"] == 0.25


def test_different_workloads_are_not_ranked_together(tmp_path):
    trial(tmp_path, "a", throughput=1.0, workload="a")
    trial(tmp_path, "b", throughput=2.0, workload="b")
    with pytest.raises(ValueError, match="different workloads"):
        rank_trials(load_trials(tmp_path))
