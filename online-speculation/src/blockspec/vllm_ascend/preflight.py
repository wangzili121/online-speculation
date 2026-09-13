"""Fail-fast checks for the pinned Orthrus vLLM-Ascend environment."""

from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
import json
import os
from pathlib import Path
import time

from .artifacts import sha256_file
from .workload import read_manifest


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _has_orthrus_plugin():
    return any(entry.name == "blockspec_orthrus"
               for entry in metadata.entry_points(group="vllm.general_plugins"))


def check(*, model, workload, runtime_root, max_model_len,
          expected_version="0.23.0", profiler=None):
    import torch

    model = Path(model)
    runtime_root = Path(runtime_root)
    versions = {
        "vllm": _version("vllm"),
        "vllm_ascend": _version("vllm-ascend"),
        "torch": _version("torch"),
        "torch_npu": _version("torch-npu"),
        "online_speculation": _version("online-speculation"),
    }
    failures = []
    for name in ("vllm", "vllm_ascend"):
        value = versions[name] or ""
        if value.split("+", 1)[0] != expected_version:
            failures.append(f"{name} must be {expected_version}, found {value or 'missing'}")
    plugin_registered = _has_orthrus_plugin()
    if not plugin_registered:
        failures.append(
            "Orthrus vLLM plugin is not installed; run pip install -e . --no-deps")
    required_model_files = [model / "config.json", model / "tokenizer.json",
                            model / "model.safetensors"]
    for path in required_model_files:
        if not path.is_file():
            failures.append(f"missing model file: {path}")
    if (model / "model.safetensors").is_file() and (
            model / "model.safetensors").stat().st_size < 1024**3:
        failures.append("model.safetensors is unexpectedly small")
    marker_files = {
        "custom_proposer": runtime_root / "vllm_ascend/spec_decode/__init__.py",
        "probabilistic_correction": runtime_root / "vllm_ascend/worker/model_runner_v1.py",
        "k32_fallback": runtime_root / "vllm_ascend/attention/attention_v1.py",
        "verify_trace": runtime_root / "vllm_ascend/worker/model_runner_v1.py",
        "state_commit_trace": runtime_root / "vllm_ascend/worker/model_runner_v1.py",
    }
    markers = {}
    expected = {
        "custom_proposer": "custom_class",
        "probabilistic_correction": "_draft_probs",
        "k32_fallback": "min(requested, 16)",
        "verify_trace": "orthrus.verify.run",
        "state_commit_trace": "orthrus.state_commit",
    }
    for name, path in marker_files.items():
        present = path.is_file() and expected[name] in path.read_text(
            encoding="utf-8")
        markers[name] = present
        if not present:
            failures.append(f"runtime overlay marker missing: {name}")
    records = read_manifest(workload)
    overflow = [row["request_id"] for row in records
                if row["actual_prompt_tokens"] + row["max_tokens"] > max_model_len]
    if overflow:
        failures.append(f"{len(overflow)} workload requests exceed max_model_len")
    npu_available = bool(getattr(torch, "npu", None) and torch.npu.is_available())
    npu_count = torch.npu.device_count() if npu_available else 0
    if not npu_available or npu_count != 1:
        failures.append(f"exactly one visible NPU required, found {npu_count}")
    profile_dependencies = {}
    if profiler == "service":
        profile_dependencies = {
            "ms_service_profiler": _version("ms-service-profiler"),
            "tzdata": _version("tzdata"),
            "pytz": _version("pytz"),
            "module": find_spec("ms_service_profiler") is not None,
        }
        if profile_dependencies["ms_service_profiler"] != "26.0.0":
            failures.append("service profiler requires ms-service-profiler 26.0.0")
        if profile_dependencies["tzdata"] != "2025.3":
            failures.append("service profiler requires tzdata 2025.3")
        if profile_dependencies["pytz"] != "2026.3.post1":
            failures.append("service profiler requires pytz 2026.3.post1")
        try:
            import ms_service_profiler  # noqa: F401
            from ms_service_profiler import analyze  # noqa: F401
        except ImportError as exc:
            failures.append(f"service profiler hook import failed: {exc}")
        if not profile_dependencies["module"]:
            failures.append("ms_service_profiler module is unavailable")
    elif profiler == "torch":
        try:
            from torch_npu.profiler import profiler as torch_npu_profiler
            profile_dependencies["torch_npu_analyze"] = callable(
                getattr(torch_npu_profiler, "analyse", None))
        except ImportError as exc:
            failures.append(f"torch_npu profiler import failed: {exc}")
        if not profile_dependencies.get("torch_npu_analyze"):
            failures.append("torch_npu profiler analyze function is unavailable")
    result = {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "ok": not failures,
        "failures": failures,
        "versions": versions,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "npu_count": npu_count,
        "model_bytes": ((model / "model.safetensors").stat().st_size
                        if (model / "model.safetensors").is_file() else None),
        "workload_requests": len(records),
        "workload_sha256": sha256_file(workload),
        "runtime_markers": markers,
        "orthrus_plugin_registered": plugin_registered,
        "profiler": profiler,
        "profile_dependencies": profile_dependencies,
    }
    return result


def write(path, result):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def format_result(result):
    return json.dumps(result, ensure_ascii=True, indent=2)
