#!/usr/bin/env python3
"""Wait for a downloaded checkpoint and eight idle cards, then run one API smoke."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from watch_dsv4_tp8 import check_port_available, idle_devices


REPO = Path(__file__).resolve().parents[1]
PINNED_IMAGE = "sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2"
MARKER = "V4.1 Orthrus last-layer update "


def checkpoint_ready(model):
    """Check every indexed shard's final byte using its safetensors header."""
    for filename in ("config.json", "tokenizer.json", "quant_model_description.json"):
        if (model / filename).stat().st_size <= 0:
            raise ValueError(f"empty {filename}")
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if len(shards) != 85:
        raise ValueError(f"expected the W4A8 1017 checkpoint, found {len(shards)} shards")
    for shard in sorted(shards):
        path = model / shard
        if path.parent != model:
            raise ValueError(f"unexpected shard path: {shard}")
        with path.open("rb") as source:
            header_size = int.from_bytes(source.read(8), "little")
            if not 0 < header_size <= 100_000_000:
                raise ValueError(f"invalid safetensors header length for {shard}")
            header = json.loads(source.read(header_size))
        offsets = (entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__")
        if path.stat().st_size != 8 + header_size + max(offsets):
            raise ValueError(f"wrong file length for {shard}")


def wait_for_idle(npu_smi, deadline):
    while time.monotonic() < deadline:
        try:
            snapshot = subprocess.check_output([npu_smi, "info"], text=True, timeout=10)
            if idle_devices(snapshot):
                return
        except subprocess.SubprocessError:
            pass
        time.sleep(5)
    raise TimeoutError("eight idle cards did not become available")


def draft_token_count(metrics):
    rows = re.findall(
        r"^vllm:spec_decode_num_draft_tokens(?:_total)?(?:\{[^\n]*\})? ([0-9.eE+\-]+)$",
        metrics, re.MULTILINE,
    )
    return sum(map(float, rows))


def run_model(args, mode, head, deadline, *, online=False):
    wait_for_idle(args.npu_smi, deadline)
    check_port_available(args.port)
    name = f"wangzili-dsv41-{mode}-{os.getpid()}" + ("-online" if online else "")
    image = subprocess.check_output(["docker", "image", "inspect", PINNED_IMAGE, "--format", "{{.Id}}"],
                                    text=True).strip()
    env = os.environ | {
        "ACK_EXCLUSIVE_NODE": "1",
        "PINNED_LOCAL_IMAGE_ID": image,
        "MODE": mode,
        "MODELS_HOST": str(args.model.parent.parent),
        "MODEL_HOST": str(args.model),
        "HEAD_HOST": str(head),
        "CONTAINER": name,
        "HOST": "127.0.0.1",
        "PORT": str(args.port),
        "LOCAL_IP": args.local_ip,
        "NIC_NAME": args.nic,
        "ORTHRUS_DSV41_ONLINE_STRIDE": "1" if online else "0",
    }
    print(f"starting {name}", flush=True)
    launcher = REPO / "scripts/deploy_dsv41_single_a2.sh"
    log = REPO / "local" / "dsv41-deploy" / name / "service.log"
    log.parent.mkdir(parents=True, exist_ok=False)
    api = f"http://127.0.0.1:{args.port}"
    try:
        subprocess.run(["bash", launcher], env=env, check=True, timeout=120)
        start_deadline = time.monotonic() + 2400
        while time.monotonic() < start_deadline:
            if log.exists():
                with log.open("rb") as source:
                    source.seek(max(0, log.stat().st_size - 48_000))
                    tail = source.read().decode(errors="replace")
                if "Engine core initialization failed" in tail or "Traceback (most recent call last)" in tail:
                    raise RuntimeError(f"engine startup failed, see {log}")
            try:
                with urlopen(api + "/health", timeout=3):
                    break
            except (OSError, URLError):
                time.sleep(5)
        else:
            raise TimeoutError(f"engine readiness timed out, see {log}")

        outputs = []
        prompts = (
            "List the first ten positive integers, separated by commas.",
            "Read this context and answer with one sentence. " + "The sky is blue. " * 80 + "What color is the sky?",
        )
        for prompt in prompts[:1] if online else prompts:
            body = json.dumps({
                "model": f"dsv41-{mode}-single-a2", "prompt": prompt, "max_tokens": 12 if online else 40,
                "temperature": 0, "return_token_ids": True,
            }).encode()
            request = Request(api + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=1200) as response:
                result = json.load(response)
            ids = result["choices"][0]["token_ids"]
            if not ids:
                raise RuntimeError("completion returned no token IDs")
            outputs.append(ids)
        if online and not any(MARKER in line and "published" in line for line in log.read_text().splitlines()):
            raise RuntimeError(f"no completed online publication recorded in {log}")
        if mode == "orthrus":
            for _ in range(10):
                with urlopen(api + "/metrics", timeout=10) as response:
                    metrics = response.read().decode()
                if draft_token_count(metrics) > 0:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("requests completed without any verifier draft tokens")
        return outputs
    finally:
        subprocess.run(["docker", "stop", "-t", "20", name], timeout=90, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--download-marker", type=Path, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--npu-smi", default="/usr/local/sbin/npu-smi")
    parser.add_argument("--port", type=int, default=19891)
    parser.add_argument("--watch-hours", type=float, default=24)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_hours * 3600
    local = REPO / "local"
    local.mkdir(exist_ok=True)
    with (local / "dsv41-single-a2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"waiting for {args.download_marker} and the pinned image", flush=True)
        while not args.download_marker.is_file() or subprocess.run(
            ["docker", "image", "inspect", PINNED_IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            if time.monotonic() >= deadline:
                raise TimeoutError("checkpoint or pinned image did not become ready")
            time.sleep(30)
        checkpoint_ready(args.model)
        models = args.model.parent.parent
        head = models / "orthrus-dsv41-head-v1-ar-init"
        print("checkpoint and container image verified, waiting for eight idle cards", flush=True)
        baseline = run_model(args, "target", head, deadline)
        if not head.exists():
            text = json.loads((args.model / "config.json").read_text())["text_config"]
            command = [
                "docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
                "--user", f"{os.getuid()}:{os.getgid()}", "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
                "-e", "OMP_NUM_THREADS=6", "-e", "MKL_NUM_THREADS=6",
                "-v", f"{models}:/models", "-v", f"{REPO}:/work:ro", "-e", "PYTHONPATH=/work/src",
                PINNED_IMAGE, "/work/scripts/export_dsv41_orthrus_head.py",
                "--model", "/models/" + str(args.model.relative_to(models)),
                "--output", "/models/" + head.name, "--mask-token-id", str(text["dspark_noise_token_id"]),
                "--from-target",
            ]
            subprocess.run(command, check=True, timeout=1800)
        orth = run_model(args, "orthrus", head, deadline)
        if baseline != orth:
            raise RuntimeError(f"greedy token IDs differ: target={baseline}, Orthrus={orth}")
        print(json.dumps({"greedy_exact": True, "token_ids": orth}), flush=True)
        if args.online:
            run_model(args, "orthrus", head, deadline, online=True)
            print("online update published and the request completed", flush=True)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    main()
