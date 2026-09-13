from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.dllm.run_llada_suite import main
from experiments.shared.components import FULL_COMPONENTS


def _arguments(output: Path) -> list[str]:
    return [
        "run_llada_suite.py",
        "--config",
        "configs/gsm8k_llada_moe_3090.toml",
        "--profile",
        "smoke",
        "--tag",
        "unit",
        "--output-root",
        str(output),
        "--vrpo",
        "preflight",
        "--no-environment-check",
        "--methods",
        "base",
    ]


def test_llada_suite_executes_preflight_quality_and_replay_then_completes(
    monkeypatch, tmp_path
):
    calls = []

    def complete(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path))
    monkeypatch.setattr(subprocess, "run", complete)
    main()

    assert [Path(call[0][1]).name for call in calls] == [
        "train_gsm8k_vrpo.py",
        "gsm8k_reproduction.py",
        "gsm8k_replay_benchmark.py",
    ]
    manifest = json.loads(
        (tmp_path / "unit" / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["completed_commands"] == 3


def test_llada_suite_records_subprocess_failure(monkeypatch, tmp_path):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path))
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        main()

    manifest = json.loads(
        (tmp_path / "unit" / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["completed_commands"] == 0


def test_llada_suite_plans_passk_draws_with_one_model_load_per_method(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llada_suite.py",
            "--config",
            "configs/gsm8k_llada_moe_3090.toml",
            "--profile",
            "smoke",
            "--tag",
            "passk-unit",
            "--output-root",
            str(tmp_path),
            "--components",
            "passk",
            "--passk-limit",
            "1",
            "--passk-draws",
            "2",
            "--vrpo",
            "train",
            "--dry-run",
        ],
    )
    main()

    manifest = json.loads(
        (tmp_path / "passk-unit" / "suite_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    commands = manifest["commands"]
    reproduction = [command for command in commands if "gsm8k_reproduction.py" in command]
    assert len(reproduction) == 7
    assert all("--draws 2" in command for command in reproduction)
    assert sum("gsm8k_analysis.py" in command for command in commands) == 1
    assert manifest["components"] == ["passk"]


def test_llada_suite_coalesces_async_and_infra_into_one_full_benchmark(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llada_suite.py",
            "--config",
            "configs/gsm8k_llada_moe_3090.toml",
            "--profile",
            "smoke",
            "--tag",
            "infra-unit",
            "--output-root",
            str(tmp_path),
            "--components",
            "async",
            "infra",
            "--infra-limit",
            "2",
            "--vrpo",
            "skip",
            "--dry-run",
        ],
    )
    main()

    manifest = json.loads(
        (tmp_path / "infra-unit" / "suite_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    commands = [
        command for command in manifest["commands"] if "benchmark_infra.py" in command
    ]
    assert len(commands) == 1
    assert "--section all" in commands[0]
    assert "--limit 2" in commands[0]


def test_llada_suite_full_profile_defaults_to_every_common_component(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llada_suite.py",
            "--config",
            "configs/gsm8k_llada_moe_3090.toml",
            "--profile",
            "full",
            "--tag",
            "full-unit",
            "--output-root",
            str(tmp_path),
            "--vrpo",
            "train",
            "--dry-run",
        ],
    )
    main()

    manifest = json.loads(
        (tmp_path / "full-unit" / "suite_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["components"] == list(FULL_COMPONENTS)
    commands = manifest["commands"]
    assert sum("benchmark_infra.py" in command for command in commands) == 1
    assert any("gsm8k_replay_benchmark.py" in command for command in commands)
    assert any("--kind passk" in command for command in commands)
    assert any("--kind distribution" in command for command in commands)
