"""Verify device-free inspection and explicitly mapped single-card probes."""

import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/dsv41_pypto_env.sh"


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
