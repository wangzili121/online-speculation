import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).parents[1]


def test_host_launcher_pins_image_and_maps_eight_cards():
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "NODE_RANK": "1",
            "NODE0_IP": "10.0.0.1",
            "LOCAL_IP": "10.0.0.2",
            "NIC_NAME": "eth0",
            "MODE": "orthrus",
        }
    )
    result = subprocess.run(
        [ROOT / "scripts" / "deploy_dsv41_node.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    command = shlex.split(result.stdout.removeprefix("command: "))
    assert command[:5] == ["docker", "run", "--pull", "never", "-d"]
    assert command.count("--device") == 11
    assert command[-3].startswith("quay.io/ascend/vllm-ascend@sha256:")
    assert "NODE_RANK=1" in command
    assert "/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8:/workspace/target:ro" in command
    assert command[-1] == "scripts/serve_dsv41_orthrus.sh"


def test_two_node_deploy_lowers_both_roles():
    environment = os.environ | {
        "DRY_RUN": "1",
        "NODE0_IP": "10.0.0.1",
        "NODE1_IP": "10.0.0.2",
        "NODE1_SSH": "test@10.0.0.2",
        "NIC0": "eth2",
        "NIC1": "eth3",
        "HEAD_HOST": "/workspace/custom head",
    }
    script = ROOT / "scripts/dsv41_two_node.sh"

    def commands(action):
        output = subprocess.check_output([script, action], text=True, env=environment)
        return [shlex.split(line.removeprefix("command: ")) for line in output.splitlines()]

    main, peer = commands("up")
    for rank, command in enumerate((main, peer)):
        assert f"NODE_RANK={rank}" in command
        assert f"LOCAL_IP=10.0.0.{rank + 1}" in command
        assert f"NIC_NAME=eth{rank + 2}" in command
        assert "INIT_HEAD=1" in command
        assert "HEAD=/workspace/heads/custom head" in command
        assert command[-1] == "scripts/serve_dsv41_orthrus.sh"
    main, peer = commands("probe")
    for rank, command in enumerate((main, peer)):
        assert f"NODE_RANK={rank}" in command
        assert "MODEL=/workspace/target" not in command
        assert "scripts/probe_dsv41_cluster.py" in command[-3]


def test_experimental_single_a2_launcher_maps_one_eight_card_node():
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "LOCAL_IP": "10.0.0.1",
            "NIC_NAME": "eth0",
            "MODE": "target",
        }
    )
    result = subprocess.run(
        [ROOT / "scripts" / "deploy_dsv41_single_a2.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    commands = [shlex.split(line.removeprefix("command: ")) for line in result.stdout.splitlines()]
    docker_run, docker_exec = commands
    assert docker_run[:3] == ["docker", "run", "-d"]
    assert docker_run.count("--device") == 11
    assert "--ipc" in docker_run
    assert docker_exec[:3] == ["docker", "exec", "-d"]
    assert "MODE=target" in docker_exec
    assert "scripts/serve_dsv41_single_a2.sh" in docker_exec[-1]


def test_online_gsm8k_launcher_is_strict_and_uses_resident_factory(tmp_path):
    script = ROOT / "scripts" / "run_dsv41_pypto_online_gsm8k.sh"
    text = script.read_text()
    assert "set -euo pipefail" in text
    assert "pypto_dsv41_resident_primitives:create_primitives" in text
    assert "--online-study" in text and "--online-overrides" in text
    assert "last-N must be one of 1, 2, 4 or 8" in text
    assert "--no-trace-prefill" in text
    environment = os.environ.copy()
    environment["DSV41_GSM8K_ROOT"] = str(tmp_path / "missing-data")
    result = subprocess.run(
        [script, "node7", "smoke", "1"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "required online-update input is missing" in result.stderr
    assert "node must be node7 or node8" not in result.stderr


def test_hc_collapse_validator_is_a_bounded_single_leaf_check():
    script = ROOT / "scripts" / "validate_dsv41_pypto_hc_collapse_vjp.py"
    text = script.read_text()
    assert "make_hc_collapse_vjp" in text
    assert "--compile-only" in text and "--production" in text
    assert "validate_dsv41_pypto_online_programs" not in text
    environment = (ROOT / "scripts" / "dsv41_pypto_env.sh").read_text()
    assert "hc-collapse-vjp-check" in environment


def test_company_online_launcher_keeps_site_paths_in_one_entrypoint():
    script = ROOT / "scripts" / "run_dsv41_pypto_company_online.sh"
    text = script.read_text()
    assert "export CLUSTER_KEY_FILE=/home/wangzili/.orthrus-cluster.key" in text
    assert 'CLUSTER_KEY_FILE:-' not in text
    assert "/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8" in text
    assert "/workspace/data/orthrus-dsv41-gsm8k" in text
    assert "0,1,2,3,4,5,6,7" in text
    assert 'run_dsv41_pypto_online_gsm8k.sh" "$@"' in text


def test_cluster_source_pairing_tracks_the_fully_patched_kernel_tree():
    launcher = (ROOT / "scripts" / "run_dsv41_pypto_cluster.sh").read_text()
    patch = (ROOT / "runtime" / "pypto_dsv41_a3" / "serving-cluster.patch").read_text()
    assert "from pypto_serving_a3.source import A3_COMMIT, A3_TREE" in launcher
    assert "lib-online-expert-capture.patch" in launcher
    assert "Expected one DSV4.1 router state boundary" in launcher
    assert 'export ORTHRUS_CLUSTER_LIB_TREE="$(git -C "$PYPTO_LIB_ROOT" rev-parse' in launcher
    assert 'os.environ.get("ORTHRUS_CLUSTER_LIB_TREE", A3_CLUSTER_TREE)' in patch

