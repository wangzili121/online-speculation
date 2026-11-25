#!/usr/bin/env python3
"""Compile the Orthrus DSV4.1 A3 whole-model draft graph without loading weights."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from blockspec.dsv41_packed_proposer import draft_geometry, draft_graph_manifest
from blockspec.pypto_dsv41_packed_graph import make_packed_draft_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, choices=(16, 32), default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--decode-capacity", type=int, choices=(16,), required=True,
                        help="Use the native 16-row MoE dispatch configuration")
    parser.add_argument("--artifact-dir", type=Path,
                        help="Fresh directory for the loadable graph")
    parser.add_argument("--output", type=Path, help="Compilation report")
    args = parser.parse_args()

    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    pages, c1a_bucket, c2a_bucket = draft_geometry(args.context)
    graph = make_packed_draft_graph(pages, c1a_bucket, c2a_bucket, args.block)
    extra = {}
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=False)
        extra = dict(save_kernels=True, save_kernels_dir=str(args.artifact_dir.resolve()))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=list(range(EP_SIZE)), num_sub_workers=0),
                       **extra)

    start = perf_counter()
    program = graph.compile(config=config)
    manifest = draft_graph_manifest(args.context, args.block)
    (program.output_dir / "orthrus_draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = dict(compiled=True, npu_execution=False, layers=FLASH.num_hidden_layers,
                  block=args.block, context=args.context, seconds=perf_counter() - start,
                  work_dir=str(program.output_dir))
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
