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
