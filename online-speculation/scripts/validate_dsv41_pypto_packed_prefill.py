#!/usr/bin/env python3
"""Compile the resident 1122 Prefill stages around a host-local TP group."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--slots-per-dp", type=int, default=1)
    parser.add_argument("--prefill-moe-slab-rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from blockspec.dsv41_packed_decode import make_decode_final_head
    from blockspec.dsv41_packed_ep_graph import make_hccl_ep_graphs
    from blockspec.dsv41_packed_prefill import make_prefill_ffn_restore, prefill_slabs
    from models.deepseek_v4_1_flash_a3 import driver, prefill_fwd
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    devices = list(map(int, args.devices.split(",")))
    ranks = len(devices)
    tokens, tp = prefill_fwd.MOE_TOKENS, prefill_fwd.TP_SIZE
    config = dict(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                  distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0), save_kernels=True)
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = {}

    def compile_stage(name, stage, rows):
        specs = driver.build_specs(stage, mode="prefill", capacity=rows,
                                   history_capacity=args.context, request_slots=args.slots_per_dp,
                                   prepare_windows=True)
        result = driver.run(fn=stage, specs=specs, compile_only=True,
                            config=dict(config, save_kernels_dir=str((args.output / name).resolve())))
        if not result.passed:
            raise SystemExit(result.error or 1)
        artifacts[name] = str(result.work_dir)

    compile_stage("attention", prefill_fwd.make_prefill_attention_stage(ranks), args.capacity)
    for width in dict.fromkeys(width for _, width in prefill_slabs(args.capacity, tokens, tp)):
        compile_stage(f"ffn-restore-{width}", make_prefill_ffn_restore(width, ranks), width)
        # These stages receive already-packed matrix views; compile through
        # the standard graph factory to retain their NZ layout metadata.
        stages = make_hccl_ep_graphs(tokens, width, ranks, prefill=True)
        for stage in stages:
            name = f"ep-{stage._func.__name__}-{width}"
            target = (args.output / name).resolve()
            stage.compile(config=RunConfig(**config, save_kernels_dir=str(target)))
            artifacts[name] = str(target)
    compile_stage("head", make_decode_final_head(ranks), args.capacity)
    manifest = driver.manifest("prefill", args.capacity, args.context,
                               slab_rows=tokens, request_slots=args.slots_per_dp)
    (args.output / "a3_manifest.json").write_text(json.dumps(dict(manifest, local_ranks=ranks), indent=2) + "\n")
    print(json.dumps(dict(passed=True, compiled=True, local_ranks=ranks, rows=args.capacity,
                          context=args.context, slots_per_dp=args.slots_per_dp, moe_tokens=tokens, artifacts=artifacts)),
          flush=True)


if __name__ == "__main__":
    main()
