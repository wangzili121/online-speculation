#!/usr/bin/env python3
"""CPU 文件对照：连续更新和断点续训的同一步必须产生相同参数及 Adam 状态。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("continuous", "resumed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    manifests = [json.loads((path / "manifest.json").read_text())
                 for path in (args.continuous, args.resumed)]
    assert manifests[0] == manifests[1], "checkpoint identity, layout, step or payload differ"
    manifest = manifests[0]
    assert manifest["format"] == "orthrus-qwen38-pypto-lastn-adam-v2"
    digest, size = hashlib.sha256(), 0
    # 逐段直接比较整个 payload，包含 FP32 master 和两组 Adam moment。
    with (args.continuous / "optimizer.safetensors").open("rb") as continuous, \
            (args.resumed / "optimizer.safetensors").open("rb") as resumed:
        while True:
            left, right = continuous.read(16 << 20), resumed.read(16 << 20)
            assert left == right, f"optimizer payload differs at byte {size}"
            if not left:
                break
            digest.update(left)
            size += len(left)
    assert digest.hexdigest() == manifest["payload_sha256"], "checkpoint SHA256 mismatch"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        passed=True, step=manifest["step"],
        last_layers=len({parameter["layer"] for parameter in manifest["layout"]}),
        parameter_groups=len(manifest["layout"]), payload_bytes=size,
        payload_sha256=digest.hexdigest(), head=manifest["head"],
        manifest_equal=True, payload_byte_exact=True,
        continuous=str(args.continuous), resumed=str(args.resumed),
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
