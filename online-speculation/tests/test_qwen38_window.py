"""Host-side device mapping and cleanup do not require an NPU."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def window(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_qwen38_validation_window
    return run_qwen38_validation_window


def test_pair_container_is_isolated_and_preserves_cann_python_paths(window, tmp_path):
    args = NS(devices=[2, 3], source=tmp_path / "src", model=tmp_path / "model", output=tmp_path / "result",
              image="runtime@sha256:123", max_model_len=32768, acceptance_args=["--blocks", "8"])
    command = window.container_command(args, "ours")
    assert "/dev/davinci2:/dev/davinci0" in command
    assert "/dev/davinci3:/dev/davinci1" in command
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1" in command
    assert "--privileged" not in command
    assert "PYTHONPATH=/work/src" not in command
    assert f"{args.source}:/work:ro" in command
    assert "/work/scripts/run_in_runtime.py" in command
    assert command[-2:] == ["--blocks", "8"]
    assert command[command.index("--max-model-len") + 1] == "32768"


def test_pair_can_yield_a_complete_window_to_the_pending_full_model(window, monkeypatch, tmp_path):
    directory = tmp_path / "validation"
    directory.mkdir()
    path = directory / "watch-status.json"
    monkeypatch.setattr(window.os, "kill", lambda *args: None)
    path.write_text(json.dumps(dict(pid=123, state="waiting", weights_verified=True)))
    assert window.eight_card_job_waiting(tmp_path, list(range(8)))
    assert not window.eight_card_job_waiting(tmp_path, [4, 5])
    path.write_text(json.dumps(dict(pid=123, state="validation_complete", weights_verified=True)))
    assert not window.eight_card_job_waiting(tmp_path, list(range(8)))
    monkeypatch.setattr(window.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    assert not window.eight_card_job_waiting(tmp_path, list(range(8)))


def test_uniform_fixture_runs_separate_bounded_calibration(window, tmp_path):
    args = NS(devices=[2, 3], source=tmp_path, model=tmp_path, output=tmp_path,
              image="runtime@sha256:123", max_model_len=8192, acceptance_args=["--block-size", "32"],
              uniform_calibration=True)
    command = window.container_command(args, "ours")
    assert "/work/scripts/validate_qwen38_vllm.py" in command
    assert "/work/scripts/accept_qwen38_vllm.py" not in command
    assert "ORTHRUS_VALIDATE_UNIFORM_CALIBRATION=1" in command
    assert "ORTHRUS_VALIDATE_SAMPLING_TRIALS=128" in command
    assert "--output-dir" not in command


def test_failed_container_creation_still_cleans_only_our_name(window, monkeypatch, tmp_path):
    calls, released = [], []
    snapshot = "\n".join(
        f"| {i} 910B3 | OK | 90 |\n| 0 | 0000:C1:00.0 | 0 0 / 0 3480 / 65536 |\n"
        f"| No running processes found in NPU {i} |" for i in range(8))

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, 1)
        return NS(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(window.subprocess, "run", run)
    monkeypatch.setattr(window.subprocess, "check_output", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(window, "acquire_devices", lambda *args: ["lease"])
    monkeypatch.setattr(window, "release_devices", lambda locks: released.extend(locks))
    output = tmp_path / "new-result"
    monkeypatch.setattr(sys, "argv", ["run", "--source", str(tmp_path), "--lease-root", str(tmp_path),
        "--model", str(tmp_path), "--output", str(output), "--image", "runtime@sha256:123",
        "--devices", "2", "3"])
    with pytest.raises(subprocess.TimeoutExpired):
        window.main()
    name = calls[0][calls[0].index("--name") + 1]
    assert ["docker", "stop", "--time", "15", name] in calls
    assert ["docker", "rm", "-f", name] in calls
    assert released == ["lease"]
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
