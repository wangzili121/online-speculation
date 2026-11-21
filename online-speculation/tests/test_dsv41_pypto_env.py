"""Verify device-free inspection and explicitly mapped single-card probes."""

import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dsv41_pypto_env.sh"


def environment(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "if sys.argv[1:3] == ['image','inspect']: raise SystemExit(0)\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def test_preflight_mounts_weights_without_npu_devices(tmp_path):
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = environment(tmp_path) | {"TARGET_WEIGHTS": str(model)}
    result = subprocess.check_output(["bash", str(SCRIPT), "preflight"], text=True, env=env)
    args = json.loads(result)
    assert "--device" not in args
    assert f"{model}:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro" in args
    assert args[-1] == "preflight"


def test_probe_requires_a_device_and_maps_only_that_device(tmp_path):
    env = environment(tmp_path)
    env.pop("NPU_DEVICE", None)
    failure = subprocess.run(["bash", str(SCRIPT), "probe"], capture_output=True, text=True, env=env)
    assert failure.returncode != 0 and "NPU_DEVICE" in failure.stderr
    env["NPU_DEVICE"] = "6"
    result = subprocess.check_output(["bash", str(SCRIPT), "probe"], text=True, env=env)
    args = json.loads(result)
    assert args.count("--device") == 4
    assert "/dev/davinci6:/dev/davinci6" in args
    assert "ASCEND_RT_VISIBLE_DEVICES=6" in args
    assert args[-1] == "probe"


def test_compilation_probe_does_not_map_npu_devices(tmp_path):
    result = subprocess.check_output(["bash", str(SCRIPT), "compile-probe"], text=True, env=environment(tmp_path))
    args = json.loads(result)
    assert "--device" not in args
    assert args[-1] == "compile-probe"


def test_attention_check_mounts_current_sources_and_writes_host_reports(tmp_path):
    env = environment(tmp_path) | {"NPU_DEVICE": "6", "PYPTO_IMAGE": "image:offline"}
    result = subprocess.check_output(
        ["bash", str(SCRIPT), "attention-check", "--heads", "16"], text=True, env=env, cwd=tmp_path,
    )
    location, command = result.splitlines()
    output = Path(location.removeprefix("Reports: "))
    args = json.loads(command)
    assert output.parent.resolve() == tmp_path.resolve() and output.is_dir()
    assert f"{output}:/results" in args
    assert f"{SCRIPT.parents[1]}:/workspace/online-speculation:ro" in args
    assert args[args.index("--network") + 1] == "none"
    assert "/dev/davinci6:/dev/davinci6" in args
    assert args[-5:] == ["image:offline", "bash", "/workspace/online-speculation/scripts/compare_dsv41_pypto_swa.sh",
                         "--heads", "16"]


def test_handoff_checks_checkpoint_and_head_without_npu_by_default(tmp_path):
    model, head = tmp_path / "model", tmp_path / "head"
    model.mkdir()
    head.mkdir()
    (model / "config.json").write_text("{}")
    (head / "manifest.json").write_text("{}")
    env = environment(tmp_path) | {"TARGET_WEIGHTS": str(model), "DRAFT_HEAD": str(head)}
    env.pop("NPU_DEVICE", None)
    result = subprocess.check_output(
        ["bash", str(SCRIPT), "handoff-check"], text=True, env=env, cwd=tmp_path,
    )
    location, command = result.splitlines()
    args = json.loads(command)
    assert f"{Path(location.removeprefix('Reports: '))}:/results" in args
    assert f"{model}:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro" in args
    assert f"{head}:/workspace/draft-head:ro" in args
    assert "--device" not in args
    assert args[args.index("--network") + 1] == "none"
    assert args[-1].endswith("/scripts/check_dsv41_pypto_handoff.sh")

    result = subprocess.check_output(
        ["bash", str(SCRIPT), "handoff-check"], text=True, env=env | {"NPU_DEVICE": "0"}, cwd=tmp_path,
    )
    args = json.loads(result.splitlines()[-1])
    assert args.count("--device") == 4
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in args


def test_init_head_keeps_target_readonly_and_writes_output_without_npu(tmp_path):
    model, head = tmp_path / "model", tmp_path / "head"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = environment(tmp_path) | {"TARGET_WEIGHTS": str(model), "DRAFT_HEAD": str(head), "NPU_DEVICE": "6"}
    args = json.loads(subprocess.check_output(
        ["bash", str(SCRIPT), "init-head", "--mask-token-id", "129279"], text=True, env=env,
    ))
    assert "--device" not in args
    assert f"{model}:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro" in args
    assert f"{tmp_path}:/results" in args
    assert args[-5:] == ["--output", "/results/head", "--from-target", "--mask-token-id", "129279"]


def test_fabric_check_uses_host_network_and_only_one_device(tmp_path):
    env = environment(tmp_path) | {"NPU_DEVICE": "0", "HCCL_IF_IP": "192.0.2.7"}
    args = json.loads(subprocess.check_output(
        ["bash", str(SCRIPT), "fabric-check", "--rank", "0", "--master-address", "192.0.2.7"],
        text=True, env=env,
    ))
    assert args.count("--device") == 4
    assert "/dev/davinci0:/dev/davinci0" in args
    assert args[args.index("--network") + 1] == "host"
    assert "HCCL_IF_IP" in args
    assert args[-5:] == ["/workspace/online-speculation/scripts/probe_dsv41_pypto_fabric.py",
                         "--rank", "0", "--master-address", "192.0.2.7"]
