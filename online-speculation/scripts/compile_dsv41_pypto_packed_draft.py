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
    parser.add_argument("--last-layers", type=int, choices=range(41), default=0,
                        metavar="N", help="Capture Attention and MoE state for the last N layers")
    parser.add_argument("--decode-capacity", type=int, choices=(16,), required=True,
                        help="Use the native 16-row MoE dispatch configuration")
    parser.add_argument("--artifact-dir", type=Path,
                        help="Fresh directory for the loadable graph")
    parser.add_argument("--local-ranks", type=int, choices=(8,),
                        help="Compile a draft bundle for each host in an 8+8 EP16 job")
    parser.add_argument("--slots-per-dp", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Compilation report")
    args = parser.parse_args()
    if args.local_ranks is not None and args.artifact_dir is None:
        parser.error("--local-ranks requires --artifact-dir")
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, MAX_BATCH_PER_DP
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    if not 1 <= args.slots_per_dp <= MAX_BATCH_PER_DP or (args.local_ranks is None and args.slots_per_dp != 1):
        parser.error("Invalid slot count for the requested draft bundle")

    pages, c1a_bucket, c2a_bucket = draft_geometry(args.context, args.slots_per_dp)
    graph = make_packed_draft_graph(pages, c1a_bucket, c2a_bucket, args.block, last_layers=args.last_layers,
                                    staged_ranks=args.local_ranks)
    extra = {}
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=False)
        extra = dict(save_kernels=True, save_kernels_dir=str(args.artifact_dir.resolve()))
    distributed = DistributedConfig(device_ids=list(range(args.local_ranks or EP_SIZE)), num_sub_workers=0)

    start = perf_counter()
    if args.local_ranks is None:
        config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                           distributed_config=distributed, **extra)
        directory = graph.compile(config=config).output_dir
    else:
        from blockspec.dsv41_packed_decode import make_decode_ffn_tail, make_decode_final_head
        from blockspec.dsv41_packed_ep_graph import make_hccl_ep_graphs
        from models.deepseek_v4_1_flash_a3 import driver

        directory = args.artifact_dir.resolve()
        ep = make_hccl_ep_graphs(args.decode_capacity, args.block, args.local_ranks, prefill=False)
        entries = {
            "attention": graph,
            "ffn-restore": make_decode_ffn_tail(args.block, args.local_ranks, last_layers=args.last_layers),
            "head": make_decode_final_head(args.local_ranks),
            **dict(zip(("ep-prepare", "ep-compute", "ep-restore"), ep, strict=True)),
        }
        for name, entry in entries.items():
            options = dict(distributed_config=distributed, save_kernels=True, save_kernels_dir=str(directory / name))
            if name == "head":
                specs = driver.build_specs(entry, mode="decode", capacity=args.block, prepare_windows=True)
                result = driver.run(fn=entry, specs=specs, compile_only=True,
                                    config=dict(options, backend_type="a2a3", execution_mode=False))
                if not result.passed:
                    raise SystemExit(result.error or 1)
            else:
                entry.compile(config=RunConfig(backend_type=BackendType.Ascend910,
                                               execution_mode=ExecutionMode.ONBOARD, **options))
        if args.last_layers:
            from blockspec.dsv41_staged_reverse import make_staged_reverse_graphs

            reverse = make_staged_reverse_graphs(args.block, args.last_layers, physical_pages=pages,
                                                local_ranks=args.local_ranks)
            for name, entry in reverse.items():
                entry.compile(config=RunConfig(
                    backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                    distributed_config=distributed, save_kernels=True,
                    save_kernels_dir=str(directory / "reverse" / name),
                ))
    manifest = draft_graph_manifest(args.context, args.block, last_layers=args.last_layers,
                                    local_ranks=args.local_ranks, request_slots=args.slots_per_dp)
    (directory / "orthrus_draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = dict(compiled=True, npu_execution=False, layers=FLASH.num_hidden_layers,
                  block=args.block, context=args.context, last_layers=args.last_layers, seconds=perf_counter() - start,
                  local_ranks=args.local_ranks, request_slots=args.slots_per_dp, work_dir=str(directory))
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
