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
    commands = [shlex.split(line.removeprefix("command: ")) for line in result.stdout.splitlines()]
    docker_run, docker_exec = commands
    assert docker_run[:3] == ["docker", "run", "-d"]
    assert docker_run.count("--device") == 11
    assert docker_run[-4].startswith("quay.io/ascend/vllm-ascend@sha256:")
    assert docker_exec[:3] == ["docker", "exec", "-d"]
    assert "NODE_RANK=1" in docker_exec
    assert "scripts/serve_dsv41_orthrus.sh" in docker_exec[-1]


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

