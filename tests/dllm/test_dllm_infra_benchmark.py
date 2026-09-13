from __future__ import annotations

import hashlib

import pytest

from experiments.dllm.benchmark_infra import _aggregate
from experiments.dllm.runtime import validate_llada_weights


def _compute(*, seconds: float, flops: int, calls: int) -> dict[str, float | int]:
    return {
        "seconds": seconds,
        "output_token_ids": [1, 2],
        "main_compute": {
            "estimated_active_flops": flops,
            "forward_calls": calls,
        },
        "proposal_compute": {"estimated_active_flops": 0},
    }


def test_infra_aggregate_names_both_comparison_directions():
    report = _aggregate(
        [
            {
                "family": "block_continuous_batching",
                "arms": {
                    "sequential": _compute(seconds=4.0, flops=100, calls=4),
                    "batched": _compute(seconds=2.0, flops=100, calls=1),
                },
            }
        ]
    )

    comparison = report["families"]["block_continuous_batching"]["comparison"]
    assert comparison["baseline_arm"] == "sequential"
    assert comparison["optimized_arm"] == "batched"
    assert comparison["optimized_over_baseline"]["wall_clock_factor"] == 0.5
    assert comparison["baseline_over_optimized"]["wall_clock_speedup"] == 2.0
    assert comparison["output_match_on_every_example"] is True


def test_shared_weight_validation_accepts_the_pinned_manifest(tmp_path):
    content = b"weight shard"
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / "model.safetensors").write_bytes(content)
    config = {
        "model": {
            "path": str(tmp_path),
            "weight_files": ["model.safetensors"],
            "weight_sha256": [digest],
            "weight_bytes": [len(content)],
        }
    }

    assert validate_llada_weights(config) == {"model.safetensors": digest}


def test_shared_weight_validation_rejects_misaligned_columns(tmp_path):
    config = {
        "model": {
            "path": str(tmp_path),
            "weight_files": ["model.safetensors"],
            "weight_sha256": [],
            "weight_bytes": [1],
        }
    }

    with pytest.raises(ValueError, match="columns have different lengths"):
        validate_llada_weights(config)
