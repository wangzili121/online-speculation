import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest


spec = importlib.util.spec_from_file_location(
    "watch_glm53", Path(__file__).parents[1] / "scripts" / "watch_glm53_validation.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def snapshot(busy=(), memory=3480):
    rows = []
    for i in range(8):
        rows.extend([f"| {i} 910B3 | OK | 90 |",
                     f"| 0 | 0000:C1:00.0 | 0 0 / 0 {memory} / 65536 |"])
    for i in range(8):
        rows.append(f"| {i} 0 123 VLLMWorker_TP 57000 |" if i in busy else
                    f"| No running processes found in NPU {i} |")
    return "\n".join(rows)


def test_idle_requires_no_process_and_low_memory_not_only_zero_compute():
    assert watch.idle_devices(snapshot((0, 1))) == list(range(2, 8))
    assert watch.idle_devices(snapshot()) == list(range(8))
    assert watch.idle_devices(snapshot(memory=58000)) == []
    with pytest.raises(ValueError, match="incomplete"):
        watch.idle_devices("")


def test_device_leases_are_atomic_and_released_without_reserving_npus(tmp_path):
    original = watch.acquire_devices(tmp_path, [6, 7])
    assert original is not None
    assert watch.acquire_devices(tmp_path, [4, 5, 6, 7]) is None
    separate = watch.acquire_devices(tmp_path, [4, 5])
    assert separate is not None
    watch.release_devices(separate)
    watch.release_devices(original)
    complete = watch.acquire_devices(tmp_path, range(8))
    assert complete is not None
    watch.release_devices(complete)


def test_no_runtime_attestation_means_no_automatic_job(tmp_path):
    assert not watch.runtime_ready(tmp_path / "absent.json", tmp_path, "image@sha256:123")


def test_runtime_attestation_includes_inherited_proposer_dependencies(tmp_path):
    package, scripts = tmp_path / "src" / "blockspec", tmp_path / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    dependency = package / "sampling.py"
    dependency.write_text("version = 1\n")
    (scripts / "run_in_runtime.py").write_text("pass\n")
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(dict(status="passed", image="image@sha256:123", native_worker_imported=True, checked_devices=[0],
                                   sha256=watch.runtime_sources(tmp_path))))
    assert watch.runtime_ready(path, tmp_path, "image@sha256:123")
    dependency.write_text("version = 2\n")
    assert not watch.runtime_ready(path, tmp_path, "image@sha256:123")


def test_full_model_requires_all_eight_native_devices(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_in_runtime.py").write_text("pass\n")
    path = tmp_path / "ready.json"
    state = dict(status="passed", image="image@sha256:123", native_worker_imported=True,
                 checked_devices=[0], sha256=watch.runtime_sources(tmp_path))
    path.write_text(json.dumps(state))
    assert watch.runtime_ready(path, tmp_path, "image@sha256:123")
    assert not watch.runtime_ready(path, tmp_path, "image@sha256:123", 8)
    state["checked_devices"] = list(range(8))
    path.write_text(json.dumps(state))
    assert watch.runtime_ready(path, tmp_path, "image@sha256:123", 8)


def test_privileged_attestation_cannot_authorize_an_isolated_container(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_in_runtime.py").write_text("pass\n")
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(dict(status="passed", image="image@sha256:123", native_worker_imported=True,
                                   checked_devices=[0], privileged_runtime=True, sha256=watch.runtime_sources(tmp_path))))
    assert not watch.runtime_ready(path, tmp_path, "image@sha256:123")
    assert watch.runtime_ready(path, tmp_path, "image@sha256:123", privileged_runtime=True)


def test_privileged_runtime_is_explicit_and_cannot_remap_physical_cards():
    assert watch.container_access_args([2, 3]) == []
    assert watch.container_access_args([0], True) == ["--privileged"]
    assert watch.container_access_args(list(range(8)), True) == ["--privileged"]
    with pytest.raises(ValueError, match="identity-mapped"):
        watch.container_access_args([2, 3], True)


def test_network_environment_is_explicit_and_part_of_the_attestation(tmp_path, monkeypatch):
    for key in watch.runtime_network_environment():
        monkeypatch.delenv(key)
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("VLLM_HOST_IP", "192.0.2.1")
    monkeypatch.setenv("UNRELATED_SECRET", "not forwarded")
    environment = dict(HCCL_SOCKET_IFNAME="eth0", VLLM_HOST_IP="192.0.2.1")
    assert watch.runtime_network_environment() == environment
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_in_runtime.py").write_text("pass\n")
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(dict(status="passed", image="image@sha256:123", native_worker_imported=True,
                                   checked_devices=[0], network_environment=environment,
                                   sha256=watch.runtime_sources(tmp_path))))
    assert watch.runtime_ready(path, tmp_path, "image@sha256:123")
    monkeypatch.setenv("VLLM_HOST_IP", "192.0.2.2")
    assert not watch.runtime_ready(path, tmp_path, "image@sha256:123")


def test_operator_only_attestation_cannot_arm_native_jobs(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_in_runtime.py").write_text("pass\n")
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(dict(status="passed", image="image@sha256:123",
                                   sha256=watch.runtime_sources(tmp_path))))
    assert not watch.runtime_ready(path, tmp_path, "image@sha256:123")


def completion_fixture(root):
    model = root / "model"
    model.mkdir()
    manifest = root / "source_manifest.json"
    manifest.write_text("{}")
    state = dict(state="complete", model_id="Eco-Tech/GLM-5.3-Flash-w8a8",
                 directory=str(model), files=98, verified_files=98,
                 total_bytes=331764803315, verified_bytes=331764803315,
                 source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    path = root / "status.json"
    path.write_text(json.dumps(state))
    return path, model, state


def test_only_fully_verified_download_can_trigger_a_job(tmp_path):
    path, model, _ = completion_fixture(tmp_path)
    assert watch.weights_ready(path, model)


@pytest.mark.parametrize("field,value", [("verified_files", 97), ("verified_bytes", 123),
                                        ("model_id", "Eco-Tech/GLM-5.3-w8a8c8")])
def test_incomplete_or_wrong_checkpoint_is_rejected(tmp_path, field, value):
    path, model, state = completion_fixture(tmp_path)
    state[field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="fully verified"):
        watch.weights_ready(path, model)


def test_manifest_change_invalidates_download_completion(tmp_path):
    path, model, _ = completion_fixture(tmp_path)
    path.with_name("source_manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="manifest has changed"):
        watch.weights_ready(path, model)


def test_downloading_does_not_start_a_job(tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"state":"downloading"}')
    assert not watch.weights_ready(path, tmp_path / "model")


def test_timeout_releases_only_our_unique_container(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "wait"]:
            raise subprocess.TimeoutExpired(command, 1)
        return NS(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(watch.subprocess, "run", run)
    args = NS(output=tmp_path, source=tmp_path, model=tmp_path,
              image="runtime@sha256:123", job_timeout=1)
    with pytest.raises(subprocess.TimeoutExpired):
        watch.run_job(args, "baseline")
    name = calls[0][calls[0].index("--name") + 1]
    assert name.startswith("wangzili-glm53-baseline-")
    assert ["docker", "stop", "--time", "15", name] in calls
    assert ["docker", "rm", name] in calls


def test_native_job_remaps_cards_and_preserves_runtime_environment(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return NS(returncode=0, stdout="0", stderr="")
    monkeypatch.setattr(watch.subprocess, "run", run)
    args = NS(output=tmp_path, source=tmp_path, model=tmp_path, devices=[2, 3],
              image="runtime@sha256:123", job_timeout=1)
    watch.run_job(args, "orthrus", label="reload-bidir",
                  extra=("--head-dir", "/results/head", "--kda-mode", "bidirectional"))
    command = calls[0]
    assert "/dev/davinci2:/dev/davinci0" in command
    assert "/dev/davinci3:/dev/davinci1" in command
    assert command[command.index("--tp") + 1] == "2"
    assert "/work/scripts/run_in_runtime.py" in command
    assert command[command.index("-w") + 1] == "/runtime-cache"
    assert f"{tmp_path}:/work:ro" in command
    assert "PYTHONPATH=/work/src" not in command
    assert "--initialize-from-target" not in command
    assert "--privileged" not in command


def test_native_job_only_uses_privilege_after_explicit_selection(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return NS(returncode=0, stdout="0", stderr="")
    monkeypatch.setattr(watch.subprocess, "run", run)
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth0")
    args = NS(output=tmp_path, source=tmp_path, model=tmp_path, devices=[0],
              image="runtime@sha256:123", job_timeout=1, privileged_runtime=True)
    watch.run_job(args, "baseline")
    assert "--privileged" in calls[0]
    assert "/dev/davinci0:/dev/davinci0" in calls[0]
    assert "HCCL_SOCKET_IFNAME=eth0" in calls[0]
