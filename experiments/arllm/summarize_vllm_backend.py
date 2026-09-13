"""Validate and summarize a paired Transformers/vLLM throughput benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.shared.artifacts import file_sha256 as _sha256


def _positive_ratio(numerator: float, denominator: float, *, name: str) -> float:
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{name} requires two positive measurements")
    return numerator / denominator


def _role_setting(settings: dict[str, Any], role: str, name: str, default: Any) -> Any:
    role_settings = settings.get(role, {})
    if not isinstance(role_settings, dict):
        raise ValueError(f"runtime_config.vllm.{role} must be a table")
    return role_settings.get(name, settings.get(name, default))


def _engine_setting(settings: dict[str, Any], role: str, name: str, default: Any) -> Any:
    common = settings.get("engine_kwargs", {})
    role_settings = settings.get(role, {})
    role_engine = role_settings.get("engine_kwargs", {})
    if not isinstance(common, dict) or not isinstance(role_engine, dict):
        raise ValueError("vLLM engine_kwargs must be tables")
    return role_engine.get(name, common.get(name, default))


def _validate_fair_runtime(runtime: dict[str, Any]) -> None:
    vllm_settings = runtime.get("vllm", {})
    if not isinstance(vllm_settings, dict):
        raise ValueError("runtime_config.vllm must be a table")
    if not bool(vllm_settings.get("asynchronous", True)):
        raise ValueError("paired benchmark requires the asynchronous vLLM engine")
    transformers_dtype = str(runtime.get("dtype"))
    for role in ("base", "proposal"):
        role_dtype = str(
            _role_setting(vllm_settings, role, "dtype", transformers_dtype)
        )
        if role_dtype != transformers_dtype:
            raise ValueError(
                f"paired benchmark requires the same dtype for Transformers and vLLM {role}"
            )
        if _role_setting(vllm_settings, role, "quantization", None) is not None:
            raise ValueError(
                "paired benchmark does not mix vLLM quantization with an "
                "unquantized baseline"
            )
        if int(_role_setting(vllm_settings, role, "tensor_parallel_size", 1)) != 1:
            raise ValueError("paired benchmark requires tensor_parallel_size=1")
        if int(_role_setting(vllm_settings, role, "data_parallel_size", 1)) != 1:
            raise ValueError("paired benchmark requires data_parallel_size=1")
        pipeline_parallel_size = int(
            _engine_setting(vllm_settings, role, "pipeline_parallel_size", 1)
        )
        if pipeline_parallel_size != 1:
            raise ValueError("paired benchmark requires pipeline_parallel_size=1")
        exact_backend = str(
            _role_setting(vllm_settings, role, "exact_scoring_backend", "none")
        )
        if exact_backend != "none":
            raise ValueError(
                "paired benchmark excludes an additional exact scoring backend"
            )


def _validate_pair(
    transformers_report: dict[str, Any],
    vllm_report: dict[str, Any],
) -> tuple[str, ...]:
    if transformers_report.get("runtime_backend") != "transformers":
        raise ValueError("--transformers is not a Transformers backend report")
    if vllm_report.get("runtime_backend") != "vllm":
        raise ValueError("--vllm is not an asynchronous vLLM backend report")
    transformers_methods = set(transformers_report.get("methods", {}))
    vllm_methods = set(vllm_report.get("methods", {}))
    if not transformers_methods or transformers_methods != vllm_methods:
        raise ValueError("backend reports contain different or empty method sets")
    proposal_class = (
        "TransformersBackend"
        if any(method.endswith("small_proposal") for method in transformers_methods)
        else None
    )
    if transformers_report.get("runtime_backend_classes") != {
        "base": "TransformersBackend",
        "proposal": proposal_class,
    }:
        raise ValueError("--transformers did not instantiate the expected backend classes")
    if vllm_report.get("runtime_backend_classes") != {
        "base": "AsyncVLLMBackend",
        "proposal": "AsyncVLLMBackend" if proposal_class is not None else None,
    }:
        raise ValueError("--vllm did not instantiate asynchronous vLLM backends")

    matched_fields = (
        "examples",
        "workers",
        "experiment_config",
        "evaluation",
        "runtime_config",
        "algorithm_config",
        "models",
        "environment",
        "implementation_sha256",
    )
    for field in matched_fields:
        if transformers_report.get(field) != vllm_report.get(field):
            raise ValueError(f"backend reports differ in {field}")
    _validate_fair_runtime(transformers_report["runtime_config"])

    return tuple(sorted(transformers_methods))


def _method_comparison(
    transformers_method: dict[str, Any],
    vllm_method: dict[str, Any],
) -> dict[str, Any]:
    transformers_concurrent_seconds = float(
        transformers_method["asynchronous_continuous_batching_seconds"]
    )
    vllm_concurrent_seconds = float(
        vllm_method["asynchronous_continuous_batching_seconds"]
    )
    transformers_sequential_seconds = float(transformers_method["synchronous_seconds"])
    vllm_sequential_seconds = float(vllm_method["synchronous_seconds"])
    transformers_flops = int(
        transformers_method["asynchronous_compute"]["estimated_dense_forward_flops"]
    )
    vllm_flops = int(
        vllm_method["asynchronous_compute"]["estimated_dense_forward_flops"]
    )
    return {
        "transformers_concurrent_seconds": transformers_concurrent_seconds,
        "vllm_concurrent_seconds": vllm_concurrent_seconds,
        "transformers_over_vllm_concurrent_wall_time_factor": _positive_ratio(
            transformers_concurrent_seconds,
            vllm_concurrent_seconds,
            name="concurrent wall-time factor",
        ),
        "transformers_sequential_seconds": transformers_sequential_seconds,
        "vllm_sequential_seconds": vllm_sequential_seconds,
        "transformers_over_vllm_sequential_wall_time_factor": _positive_ratio(
            transformers_sequential_seconds,
            vllm_sequential_seconds,
            name="sequential wall-time factor",
        ),
        "transformers_concurrent_estimated_dense_forward_flops": transformers_flops,
        "vllm_concurrent_estimated_dense_forward_flops": vllm_flops,
        "transformers_over_vllm_logical_flop_factor": _positive_ratio(
            float(transformers_flops),
            float(vllm_flops),
            name="logical FLOP factor",
        ),
        "transformers_accuracy": float(transformers_method["asynchronous_accuracy"]),
        "vllm_accuracy": float(vllm_method["asynchronous_accuracy"]),
        "transformers_mean_output_tokens": float(
            transformers_method["asynchronous_mean_output_tokens"]
        ),
        "vllm_mean_output_tokens": float(
            vllm_method["asynchronous_mean_output_tokens"]
        ),
        "transformers_within_backend_exact_match_fraction": float(
            transformers_method["output_exact_match_fraction"]
        ),
        "vllm_within_backend_exact_match_fraction": float(
            vllm_method["output_exact_match_fraction"]
        ),
        "transformers_within_backend_answer_match_fraction": float(
            transformers_method["answer_match_fraction"]
        ),
        "vllm_within_backend_answer_match_fraction": float(
            vllm_method["answer_match_fraction"]
        ),
    }


def build_report(
    transformers_report: dict[str, Any],
    vllm_report: dict[str, Any],
    *,
    transformers_path: Path,
    vllm_path: Path,
) -> dict[str, Any]:
    methods = _validate_pair(transformers_report, vllm_report)
    comparisons = {
        method: _method_comparison(
            transformers_report["methods"][method],
            vllm_report["methods"][method],
        )
        for method in methods
    }
    transformers_total = sum(
        value["transformers_concurrent_seconds"] for value in comparisons.values()
    )
    vllm_total = sum(value["vllm_concurrent_seconds"] for value in comparisons.values())
    return {
        "schema_version": 1,
        "benchmark": "paired Transformers versus asynchronous vLLM runtime",
        "examples": int(transformers_report["examples"]),
        "workers": int(transformers_report["workers"]),
        "comparison_contract": {
            "primary_speedup": (
                "Transformers concurrent wall time divided by asynchronous vLLM "
                "concurrent wall time for the same method, requests, model, dtype, "
                "seeds, worker count, code revision, and hardware; values above one "
                "favor vLLM"
            ),
            "sequential_factor": (
                "Transformers one-prompt-at-a-time wall time divided by vLLM "
                "one-prompt-at-a-time wall time; it isolates backend execution from "
                "cross-prompt scheduling"
            ),
            "logical_flop_factor": (
                "Transformers estimated dense forward FLOPs divided by vLLM's estimate "
                "from executed token slots; it is not a measurement of fused-kernel, "
                "attention, padding, communication, or speculative-decoding FLOPs"
            ),
            "output_scope": (
                "Each source report checks concurrent output against its own sequential "
                "path. Source reports do not retain cross-backend token traces, so "
                "accuracy and output length are reported but cross-backend equality is "
                "not claimed"
            ),
            "timing_scope": (
                "One paired run is a throughput measurement, not a confidence interval; "
                "repeat the pair when timing uncertainty matters"
            ),
        },
        "matched_setting": {
            "experiment_config": transformers_report["experiment_config"],
            "evaluation": transformers_report["evaluation"],
            "runtime_config": transformers_report["runtime_config"],
            "algorithm_config": transformers_report["algorithm_config"],
            "models": transformers_report["models"],
            "environment": transformers_report["environment"],
            "implementation_sha256": transformers_report["implementation_sha256"],
        },
        "aggregate": {
            "transformers_concurrent_seconds": transformers_total,
            "vllm_concurrent_seconds": vllm_total,
            "transformers_over_vllm_concurrent_wall_time_factor": _positive_ratio(
                transformers_total,
                vllm_total,
                name="aggregate concurrent wall-time factor",
            ),
        },
        "methods": comparisons,
        "comparison_implementation": {
            "path": "experiments/arllm/summarize_vllm_backend.py",
            "sha256": _sha256(Path(__file__)),
        },
        "input_files": {
            "transformers": {
                "path": str(transformers_path),
                "sha256": _sha256(transformers_path),
            },
            "vllm": {
                "path": str(vllm_path),
                "sha256": _sha256(vllm_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transformers_report = json.loads(args.transformers.read_text(encoding="utf-8"))
    vllm_report = json.loads(args.vllm.read_text(encoding="utf-8"))
    report = build_report(
        transformers_report,
        vllm_report,
        transformers_path=args.transformers,
        vllm_path=args.vllm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
