#!/usr/bin/env python3
"""Compile the eight-device optimizer graph bundle for each EP16 host."""

import argparse
from pathlib import Path
from time import perf_counter

from blockspec.dsv41_staged_adam import make_staged_adam_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()

    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, TP_SIZE
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig

    start = perf_counter()
    world = EP_SIZE // 2
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    layout = DistributedConfig(device_ids=list(range(world)), num_sub_workers=0)
    graphs = make_staged_adam_graphs(FLASH, world, TP_SIZE)
    for name, graph in graphs.items():
        graph.compile(config=RunConfig(
            backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
            distributed_config=layout, save_kernels=True, save_kernels_dir=str(args.artifact_dir.resolve() / name),
        ))
        print(name, flush=True)
    print(f"Compiled {len(graphs)} optimizer graphs in {perf_counter() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
