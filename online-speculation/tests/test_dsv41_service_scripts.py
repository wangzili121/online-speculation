import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).parents[1]


def _dry_run(script: str, rank: int) -> list[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "NODE_RANK": str(rank),
            "NODE0_IP": "10.0.0.1",
            "LOCAL_IP": f"10.0.0.{rank + 1}",
            "NIC_NAME": "eth0",
            "MODEL": "/models/dsv41",
            "HEAD": "/models/orthrus-head",
            "PROFILE": "smoke-16k",
        }
    )
    result = subprocess.run(
        [ROOT / "scripts" / script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.startswith("command: ")
    return shlex.split(result.stdout.removeprefix("command: "))


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_target_service_matches_a2_two_node_topology():
    node0 = _dry_run("serve_dsv41_target.sh", 0)
    node1 = _dry_run("serve_dsv41_target.sh", 1)
    for command in (node0, node1):
        assert command[:3] == ["vllm", "serve", "/models/dsv41"]
        assert _argument(command, "--data-parallel-size") == "2"
        assert _argument(command, "--data-parallel-size-local") == "1"
        assert _argument(command, "--tensor-parallel-size") == "8"
        assert "--enable-expert-parallel" in command
        assert "--enforce-eager" in command
        assert _argument(command, "--tokenizer-mode") == "deepseek_v41"
        assert json.loads(_argument(command, "--engram-config")) == {
            "cpu_offload": True,
            "dp_shared_memory": True,
        }
    assert "--headless" not in node0
    assert "--data-parallel-start-rank" not in node0
    assert "--headless" in node1
    assert _argument(node1, "--data-parallel-start-rank") == "1"


def test_orthrus_service_adds_k32_eager_draft():
    command = _dry_run("serve_dsv41_orthrus.sh", 0)
    speculative = json.loads(_argument(command, "--speculative-config"))
    assert speculative == {
        "model": "/models/orthrus-head",
        "method": "dflash",
        "num_speculative_tokens": 31,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
        "enforce_eager": True,
    }
    assert _argument(command, "--compilation-config") == '{"cudagraph_mode":"NONE"}'


def test_experimental_single_a2_smoke_is_bounded():
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "MODE": "target",
            "MODEL": "/models/dsv41",
        }
    )
    result = subprocess.run(
        [ROOT / "scripts" / "serve_dsv41_single_a2.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    command = shlex.split(result.stdout.removeprefix("command: "))
    assert command[:3] == ["vllm", "serve", "/models/dsv41"]
    assert "--data-parallel-size" not in command
    assert _argument(command, "--tensor-parallel-size") == "8"
    assert _argument(command, "--max-model-len") == "16384"
    assert _argument(command, "--max-num-batched-tokens") == "512"
    assert _argument(command, "--max-num-seqs") == "1"
    assert json.loads(_argument(command, "--engram-config")) == {
        "cpu_offload": True,
        "dp_shared_memory": False,
    }


def test_experimental_single_a2_can_add_orthrus():
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "MODE": "orthrus",
            "MODEL": "/models/dsv41",
            "HEAD": "/models/head",
        }
    )
    result = subprocess.run(
        [ROOT / "scripts" / "serve_dsv41_single_a2.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    command = shlex.split(result.stdout.removeprefix("command: "))
    speculative = json.loads(_argument(command, "--speculative-config"))
    assert speculative["model"] == "/models/head"
    assert speculative["num_speculative_tokens"] == 31
    assert speculative["method"] == "dflash"

