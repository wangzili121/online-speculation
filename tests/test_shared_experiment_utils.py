from __future__ import annotations

from dataclasses import dataclass
import hashlib
import subprocess

import pytest

from experiments.shared.artifacts import (
    adapter_hashes,
    cached_file_sha256,
    checkpoint_metadata_hashes,
    dataclass_snapshot_delta,
    directory_hashes,
    file_sha256,
    implementation_hashes,
    indexed_records,
    json_fingerprint,
    load_jsonl,
)
from experiments.shared.config_overrides import (
    apply_config_overrides,
    parse_override_value,
)
from experiments.shared.statistics import (
    estimated_pass_at_k,
    jensen_shannon_bits,
    probability_distribution,
    total_variation_distance,
    wilson_interval,
)
from experiments.shared.suite_runner import run_manifested_commands


def test_dotted_config_overrides_are_typed_isolated_and_validated() -> None:
    source = {"generation": {"temperature": 1.0}, "method": {"clip": 10.0}}
    result = apply_config_overrides(
        source,
        ("generation.temperature=0.7", "method.clip=none"),
    )

    assert result == {"generation": {"temperature": 0.7}, "method": {"clip": None}}
    assert source["generation"]["temperature"] == 1.0
    with pytest.raises(KeyError, match="unknown config override"):
        apply_config_overrides(source, ("generation.temperatur=0.7",))
    with pytest.raises(ValueError, match="must contain"):
        apply_config_overrides(source, ("generation.temperature",))


def test_override_parser_accepts_toml_values_and_plain_strings() -> None:
    assert parse_override_value("8") == 8
    assert parse_override_value("[1, 2]") == [1, 2]
    assert parse_override_value("random") == "random"


def test_shared_probability_estimators_have_known_values() -> None:
    assert estimated_pass_at_k(1, 4, 1) == 0.25
    assert estimated_pass_at_k(1, 4, 2) == 0.5
    assert estimated_pass_at_k(1, 4, 4) == 1.0
    interval = wilson_interval(5, 10)
    assert interval[0] < 0.5 < interval[1]

    left = probability_distribution({"a": 2})
    right = probability_distribution({"b": 2})
    assert total_variation_distance(left, right) == 1.0
    assert jensen_shannon_bits(left, right) == 1.0


def test_shared_artifact_helpers_are_stable_and_preserve_constants(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")
    assert file_sha256(path) == hashlib.sha256(b"artifact").hexdigest()
    cache = tmp_path / "cache"
    digest = cached_file_sha256(path, cache_directory=cache)
    assert digest == hashlib.sha256(b"artifact").hexdigest()
    path.write_bytes(b"changed")
    assert cached_file_sha256(path, cache_directory=cache) == hashlib.sha256(
        b"changed"
    ).hexdigest()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        cached_file_sha256(path, cache_directory=cache, expected="0" * 64)
    assert json_fingerprint({"b": 2, "a": "值"}) == json_fingerprint(
        {"a": "值", "b": 2}
    )

    @dataclass(frozen=True)
    class Snapshot:
        requests: int
        parameters: int

    assert dataclass_snapshot_delta(
        Snapshot(2, 10),
        Snapshot(5, 10),
        constant_fields={"parameters"},
    ) == {"requests": 3, "parameters": 10}


def test_implementation_and_checkpoint_hashes_cover_discovered_files(tmp_path) -> None:
    package = tmp_path / "src" / "inference_scaling"
    shared = tmp_path / "experiments" / "shared"
    package.mkdir(parents=True)
    shared.mkdir(parents=True)
    (package / "kernel.py").write_text("kernel = 1\n", encoding="utf-8")
    (shared / "artifacts.py").write_text("helper = 1\n", encoding="utf-8")
    entrypoint = tmp_path / "experiments" / "run.py"
    entrypoint.write_text("run = 1\n", encoding="utf-8")

    hashes = implementation_hashes(tmp_path, entrypoints=("experiments/run.py",))
    assert set(hashes) == {
        "experiments/run.py",
        "experiments/shared/artifacts.py",
        "src/inference_scaling/kernel.py",
    }

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assert set(checkpoint_metadata_hashes(checkpoint)) == {"config.json"}
    assert set(directory_hashes(checkpoint)) == {"config.json", "model.safetensors"}

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "training_state.json").write_text("{}", encoding="utf-8")
    assert set(adapter_hashes(adapter)) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_jsonl_helpers_validate_rows_and_identifiers(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"problem_index": 2}\n\n', encoding="utf-8")
    records = load_jsonl(path)
    assert indexed_records(records) == {2: {"problem_index": 2}}
    with pytest.raises(ValueError, match="duplicate"):
        indexed_records(records + records)


def test_suite_runner_resumes_only_pending_commands(monkeypatch, tmp_path) -> None:
    commands = (("python", "one"), ("python", "two"), ("python", "three"))
    manifest_path = tmp_path / "suite.json"
    first_calls: list[list[str]] = []

    def fail_second(command, **_kwargs):
        first_calls.append(list(command))
        if command[1] == "two":
            raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", fail_second)
    with pytest.raises(subprocess.CalledProcessError):
        run_manifested_commands(
            commands=commands,
            root=tmp_path,
            manifest_path=manifest_path,
            metadata={"family": "test"},
            dry_run=False,
        )
    assert first_calls == [["python", "one"], ["python", "two"]]

    resumed_calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: resumed_calls.append(list(command)),
    )
    manifest = run_manifested_commands(
        commands=commands,
        root=tmp_path,
        manifest_path=manifest_path,
        metadata={"family": "test"},
        dry_run=False,
    )
    assert resumed_calls == [["python", "two"], ["python", "three"]]
    assert manifest["status"] == "complete"
    assert manifest["completed_commands"] == 3

    with pytest.raises(ValueError, match="different command plan"):
        run_manifested_commands(
            commands=(("python", "changed"),),
            root=tmp_path,
            manifest_path=manifest_path,
            metadata={"family": "test"},
            dry_run=True,
        )
