#!/usr/bin/env python3
"""Compile native EP16 online reverse and Adam artifacts without loading weights."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from blockspec.dsv41_native_online_artifacts import (
    bundle_manifest, learner_geometry, make_native_online_graphs, read_native_online_programs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--loss", choices=("tv", "forward_kl"), default="tv")
    args = parser.parse_args()

    from models.deepseek_v4_1_flash_a3.config import EP_SIZE
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    block, context, layers = learner_geometry(args.draft_dir)
    start = perf_counter()
    layout = DistributedConfig(device_ids=list(range(EP_SIZE)), num_sub_workers=0)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    for name, graph in make_native_online_graphs(block, context, layers, args.loss).items():
        graph.compile(config=RunConfig(
            backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
            distributed_config=layout, save_kernels=True, save_kernels_dir=str(args.artifact_dir.resolve() / name),
        ))
        print(json.dumps(dict(stage="compiled", graph=name)), flush=True)
    manifest = bundle_manifest(args.artifact_dir, args.draft_dir, args.loss)
    read_native_online_programs(args.artifact_dir, manifest["programs"])
    (args.artifact_dir / "online_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(passed=True, graphs=len(manifest["programs"]), last_layers=layers,
                          seconds=perf_counter() - start)), flush=True)


if __name__ == "__main__":
    main()
