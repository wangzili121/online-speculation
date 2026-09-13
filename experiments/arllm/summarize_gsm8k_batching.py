"""Compare the preserved-group scheduler with its ungrouped baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.shared.artifacts import file_sha256 as _sha256


def _prefill_saved(method: dict[str, Any], path: str) -> int:
    compute = method[path]
    return sum(
        int(backend["shared_prefill_tokens_saved"])
        for backend in (compute["base_backend"], compute["proposal_backend"])
        if backend is not None
    )


def _compare_method(
    baseline: dict[str, Any], grouped: dict[str, Any]
) -> dict[str, Any]:
    baseline_async_seconds = float(baseline["asynchronous_continuous_batching_seconds"])
    grouped_async_seconds = float(grouped["asynchronous_continuous_batching_seconds"])
    baseline_async_flops = int(
        baseline["asynchronous_compute"]["estimated_dense_forward_flops"]
    )
    grouped_async_flops = int(
        grouped["asynchronous_compute"]["estimated_dense_forward_flops"]
    )
    grouped_sync_flops = int(
        grouped["synchronous_compute"]["estimated_dense_forward_flops"]
    )
    return {
        "baseline_async_seconds": baseline_async_seconds,
        "grouped_async_seconds": grouped_async_seconds,
        "baseline_over_grouped_async_wall_time_factor": (
            baseline_async_seconds / grouped_async_seconds
        ),
        "grouped_sync_over_async_wall_time_factor": float(
            grouped["wall_time_speedup_synchronous_over_asynchronous"]
        ),
        "baseline_async_estimated_dense_forward_flops": baseline_async_flops,
        "grouped_async_estimated_dense_forward_flops": grouped_async_flops,
        "baseline_over_grouped_async_flop_factor": (
            baseline_async_flops / grouped_async_flops
        ),
        "grouped_async_over_sync_flop_factor": grouped_async_flops / grouped_sync_flops,
        "baseline_async_shared_prefill_tokens_saved": _prefill_saved(
            baseline, "asynchronous_compute"
        ),
        "grouped_async_shared_prefill_tokens_saved": _prefill_saved(
            grouped, "asynchronous_compute"
        ),
        "grouped_output_exact_match_count_vs_its_sync_path": int(
            grouped["output_exact_match_count"]
        ),
        "grouped_answer_match_count_vs_its_sync_path": int(
            grouped["answer_match_count"]
        ),
        "grouped_synchronous_accuracy": float(grouped["synchronous_accuracy"]),
        "grouped_asynchronous_accuracy": float(grouped["asynchronous_accuracy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--grouped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    grouped = json.loads(args.grouped.read_text(encoding="utf-8"))
    for field in ("examples", "workers", "algorithm_config", "models"):
        if baseline[field] != grouped[field]:
            raise ValueError(f"batching reports differ in {field}")
    baseline_methods = set(baseline["methods"])
    grouped_methods = set(grouped["methods"])
    if baseline_methods != grouped_methods:
        raise ValueError("batching reports contain different methods")
    implementation_changes = {
        path: {
            "baseline": baseline["implementation_sha256"].get(path),
            "grouped": grouped["implementation_sha256"].get(path),
        }
        for path in sorted(
            set(baseline["implementation_sha256"])
            | set(grouped["implementation_sha256"])
        )
        if baseline["implementation_sha256"].get(path)
        != grouped["implementation_sha256"].get(path)
    }
    expected_change = {"src/inference_scaling/arllm/backends/batching.py"}
    if set(implementation_changes) != expected_change:
        raise ValueError(
            "the batching comparison changed implementation files other than the scheduler"
        )

    report = {
        "schema_version": 1,
        "benchmark": "GSM8K preserved-group continuous batching comparison",
        "examples": int(grouped["examples"]),
        "workers": int(grouped["workers"]),
        "comparison_contract": {
            "baseline_over_grouped_async_wall_time_factor": (
                "ungrouped continuous-batching seconds divided by preserved-group "
                "continuous-batching seconds; values above one favor preserved groups"
            ),
            "grouped_sync_over_async_wall_time_factor": (
                "preserved-group run's per-prompt synchronous seconds divided by its "
                "continuous-batching seconds; values above one are a scheduling speedup"
            ),
            "baseline_over_grouped_async_flop_factor": (
                "ungrouped asynchronous FLOPs divided by preserved-group asynchronous "
                "FLOPs; values above one are a measured compute reduction"
            ),
            "grouped_async_over_sync_flop_factor": (
                "preserved-group asynchronous FLOPs divided by its synchronous FLOPs; "
                "values above one are padding or live-path overhead"
            ),
            "cross_run_output_scope": (
                "the two scheduler runs did not store per-example token hashes, so the "
                "baseline-over-grouped timing is a same-configuration live-workload "
                "comparison; exact token and answer agreement are only asserted against "
                "the synchronous path inside each grouped run"
            ),
        },
        "implementation_changes": implementation_changes,
        "methods": {
            method: _compare_method(
                baseline["methods"][method], grouped["methods"][method]
            )
            for method in sorted(grouped_methods)
        },
        "provenance": {
            "baseline_path": str(args.baseline),
            "baseline_sha256": _sha256(args.baseline),
            "grouped_path": str(args.grouped),
            "grouped_sha256": _sha256(args.grouped),
            "postprocessor": "experiments/arllm/summarize_gsm8k_batching.py",
            "postprocessor_sha256": _sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
