from __future__ import annotations

import json
from copy import deepcopy

import pytest

from experiments.arllm.run_vllm_backend_benchmark import _matches_requested_run
from experiments.arllm.summarize_vllm_backend import build_report


def _method(seconds: float, flops: int, accuracy: float) -> dict:
    return {
        "asynchronous_continuous_batching_seconds": seconds,
        "synchronous_seconds": 2 * seconds,
        "asynchronous_compute": {"estimated_dense_forward_flops": flops},
        "asynchronous_accuracy": accuracy,
        "asynchronous_mean_output_tokens": 10.0,
        "output_exact_match_fraction": 1.0,
        "answer_match_fraction": 1.0,
    }


def _report(backend: str) -> dict:
    return {
        "runtime_backend": backend,
        "runtime_backend_classes": (
            {"base": "TransformersBackend", "proposal": None}
            if backend == "transformers"
            else {"base": "AsyncVLLMBackend", "proposal": None}
        ),
        "examples": 4,
        "workers": 2,
        "experiment_config": {"sha256": "config"},
        "evaluation": {"dataset_sha256": "data", "problem_indices": [1, 2, 3, 4]},
        "runtime_config": {
            "dtype": "float32",
            "vllm": {
                "asynchronous": True,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "exact_scoring_backend": "none",
            },
        },
        "algorithm_config": {"max_new_tokens": 16},
        "models": {"base": {"weight_sha256": "model"}},
        "environment": {"gpu": "same-gpu", "torch": "same-version"},
        "implementation_sha256": {"backend.py": "code"},
        "methods": {"base": _method(10.0, 100, 0.5)},
    }


def test_paired_backend_report_names_speedup_denominator(tmp_path) -> None:
    transformers = _report("transformers")
    vllm = _report("vllm")
    vllm["methods"]["base"] = _method(4.0, 80, 0.75)
    transformers_path = tmp_path / "transformers.json"
    vllm_path = tmp_path / "vllm.json"
    transformers_path.write_text("transformers", encoding="utf-8")
    vllm_path.write_text("vllm", encoding="utf-8")

    report = build_report(
        transformers,
        vllm,
        transformers_path=transformers_path,
        vllm_path=vllm_path,
    )

    comparison = report["methods"]["base"]
    assert comparison["transformers_over_vllm_concurrent_wall_time_factor"] == 2.5
    assert comparison["transformers_over_vllm_sequential_wall_time_factor"] == 2.5
    assert comparison["transformers_over_vllm_logical_flop_factor"] == 1.25
    assert comparison["transformers_accuracy"] == 0.5
    assert comparison["vllm_accuracy"] == 0.75
    assert report["aggregate"]["transformers_over_vllm_concurrent_wall_time_factor"] == 2.5


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evaluation", {"problem_indices": [9]}),
        ("runtime_config", {"dtype": "bfloat16"}),
        ("environment", {"gpu": "other-gpu"}),
        ("implementation_sha256", {"backend.py": "different-code"}),
    ],
)
def test_paired_backend_report_rejects_setting_drift(
    tmp_path, field: str, replacement: dict
) -> None:
    transformers = _report("transformers")
    vllm = deepcopy(_report("vllm"))
    vllm[field] = replacement
    transformers_path = tmp_path / "transformers.json"
    vllm_path = tmp_path / "vllm.json"
    transformers_path.write_text("transformers", encoding="utf-8")
    vllm_path.write_text("vllm", encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        build_report(
            transformers,
            vllm,
            transformers_path=transformers_path,
            vllm_path=vllm_path,
        )


def test_paired_backend_report_rejects_vllm_only_precision_change(tmp_path) -> None:
    transformers = _report("transformers")
    vllm = deepcopy(_report("vllm"))
    transformers["runtime_config"]["vllm"]["dtype"] = "bfloat16"
    vllm["runtime_config"]["vllm"]["dtype"] = "bfloat16"
    transformers_path = tmp_path / "transformers.json"
    vllm_path = tmp_path / "vllm.json"
    transformers_path.write_text("transformers", encoding="utf-8")
    vllm_path.write_text("vllm", encoding="utf-8")

    with pytest.raises(ValueError, match="same dtype"):
        build_report(
            transformers,
            vllm,
            transformers_path=transformers_path,
            vllm_path=vllm_path,
        )


def test_reuse_requires_the_requested_pair_settings(tmp_path) -> None:
    path = tmp_path / "report.json"
    report = _report("vllm")
    report["experiment_config"]["sha256"] = "config"
    report["evaluation"]["dataset_sha256"] = "data"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert _matches_requested_run(
        path,
        backend="vllm",
        config_sha256="config",
        data_sha256="data",
        limit=4,
        workers=2,
        methods={"base"},
    )
    assert not _matches_requested_run(
        path,
        backend="vllm",
        config_sha256="changed",
        data_sha256="data",
        limit=4,
        workers=2,
        methods={"base"},
    )
