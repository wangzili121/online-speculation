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
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    env.pop("PYPTO_VISIBLE_DEVICE", None)
    env.pop("PYPTO_RUNTIME_DEVICE", None)
    return env


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
    assert args[:-2].count("--device") == 4
    assert "/dev/davinci6:/dev/davinci6" in args
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in args
    assert f"{SCRIPT.parents[1]}:/workspace/online-speculation:ro" in args
    assert args[-4:] == ["/workspace/online-speculation/scripts/check_dsv41_pypto_environment.py",
                         "--device-probe", "--device", "0"]


def test_device_map_keeps_the_physical_mount_in_all_three_probes(tmp_path):
    env = environment(tmp_path) | {"NPU_DEVICE": "6"}
    result = subprocess.check_output(["bash", str(SCRIPT), "device-map"], text=True, env=env)
    probes = [json.loads(row) for row in result.splitlines()]
    assert len(probes) == 3
    for args in probes:
        assert "/dev/davinci6:/dev/davinci6" in args
        assert args.count("--device") == 4
        assert args[-1] == "6"
    assert "ASCEND_RT_VISIBLE_DEVICES" in probes[0] and "-u" in probes[0]
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in probes[1]
    assert "ASCEND_RT_VISIBLE_DEVICES=6" in probes[2]


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
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in args
    assert args[-8:] == ["image:offline", "env", "bash", "/workspace/online-speculation/scripts/compare_dsv41_pypto_swa.sh",
                         "--heads", "16", "--device", "0"]


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
        ["bash", str(SCRIPT), "handoff-check"], text=True,
        env=env | {"NPU_DEVICE": "6", "PYPTO_VISIBLE_DEVICE": "unset", "PYPTO_RUNTIME_DEVICE": "2"}, cwd=tmp_path,
    )
    args = json.loads(result.splitlines()[-1])
    assert args[:-2].count("--device") == 4
    assert "PYPTO_RUNTIME_DEVICE=2" in args
    assert "ASCEND_RT_VISIBLE_DEVICES" in args and "-u" in args
    assert args[-2:] == ["--device", "2"]


def test_init_head_keeps_target_readonly_and_writes_output_without_npu(tmp_path):
    model, head = tmp_path / "model", tmp_path / "head"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = environment(tmp_path) | {"TARGET_WEIGHTS": str(model), "DRAFT_HEAD": str(head), "NPU_DEVICE": "6"}
    args = json.loads(subprocess.check_output(
        ["bash", str(SCRIPT), "init-head", "--mask-token-id", "129279"], text=True, env=env,
    ))
    assert "--device" not in args
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"{model}:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro" in args
    assert f"{tmp_path}:/results" in args
    assert args[args.index("--entrypoint") + 1] == "/opt/pypto/venv/bin/python"
    assert args[-8:] == ["/workspace/online-speculation/scripts/export_dsv41_orthrus_head.py",
                         "--model", "/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8",
                         "--output", "/results/head", "--from-target", "--mask-token-id", "129279"]


def test_fabric_check_uses_host_network_and_only_one_device(tmp_path):
    env = environment(tmp_path) | {"NPU_DEVICE": "6", "HCCL_IF_IP": "192.0.2.7"}
    args = json.loads(subprocess.check_output(
        ["bash", str(SCRIPT), "fabric-check", "--rank", "0", "--master-address", "192.0.2.7"],
        text=True, env=env,
    ))
    assert args[:-2].count("--device") == 4
    assert "/dev/davinci6:/dev/davinci6" in args
    assert args[args.index("--network") + 1] == "host"
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in args
    assert "HCCL_IF_IP=192.0.2.7" in args
    assert args[-7:] == ["/workspace/online-speculation/scripts/probe_dsv41_pypto_fabric.py",
                         "--rank", "0", "--master-address", "192.0.2.7", "--device", "0"]


def test_company_fabric_check_selects_node_defaults(tmp_path):
    env = environment(tmp_path)
    for name in ("PYPTO_IMAGE", "NPU_DEVICE", "HCCL_IF_IP"):
        env.pop(name, None)
    nodes = (("node7", "0", "1", "7.150.6.33"), ("node8", "1", "0", "7.150.6.37"))
    for node, rank, card, address in nodes:
        args = json.loads(subprocess.check_output(
            ["bash", str(SCRIPT), "fabric-check", node], text=True, env=env,
        ))
        assert f"/dev/davinci{card}:/dev/davinci{card}" in args
        assert f"HCCL_IF_IP={address}" in args
        assert "orthrus-dsv41-pypto:72a62d1-offline" in args
        assert args[-6:] == ["--rank", rank, "--master-address", "7.150.6.33", "--device", "0"]

    args = json.loads(subprocess.check_output(
        ["bash", str(SCRIPT), "fabric-check", "node8"],
        text=True, env=env | {"NPU_DEVICE": "4", "PYPTO_IMAGE": "image:custom"},
    ))
    assert "/dev/davinci4:/dev/davinci4" in args
    assert "image:custom" in args

    result = subprocess.run(["bash", str(SCRIPT), "fabric-check", "node9"], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "fabric-check {node7|node8}" in result.stderr
