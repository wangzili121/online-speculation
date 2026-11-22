#!/usr/bin/env python3
"""Compile and numerically validate the DSV4.1 selected-expert capture ABI."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")

    import torch
    import pypto.language as pl
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from models.deepseek_v4_1_flash_a3.prefill_moe import make_moe_combine

    torch.manual_seed(41)
    capacity, hidden, topk, world = 4, 128, 6, 2
    routes, packet_rows = capacity * topk, world * capacity * topk
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    def host_entry(combine):
        @pl.jit.host
        def entry(
            packet_routes: pl.Tensor,
            packet_hidden: pl.Tensor,
            packet_counts: pl.Tensor,
            route_weights: pl.Tensor,
            shared: pl.Tensor,
            routed: pl.InOut[pl.Tensor],
            output: pl.InOut[pl.Tensor],
            returned_routes: pl.InOut[pl.Tensor],
            expert_outputs: pl.Out[pl.Tensor],
            active: pl.Scalar[pl.INT32],
        ):
            combine(
                packet_routes, packet_hidden, packet_counts, route_weights, shared,
                routed, output, returned_routes, expert_outputs, active, device=0,
            )
        return entry

    combine = make_moe_combine(capacity, hidden, topk, world, weight_dtype=pl.BF16)
    entry = host_entry(combine)

    program = entry.compile(
        empty((packet_rows, 4), torch.int64), empty((packet_rows, hidden)),
        empty((world, 8), torch.int32), empty((capacity, topk), torch.float32),
        empty((capacity, hidden)), empty((capacity, hidden)), empty((capacity, hidden)),
        empty((routes, 4), torch.int64), empty((routes, hidden)), capacity,
        config=config,
    )
    # Compile the actual EP16/K32 serving geometry as well.  The small fixture
    # below remains the numerical oracle; this catches production-only tiling
    # or ABI failures without allocating the full packet on the host.
    production_capacity, production_hidden, production_world = 32, 5120, 16
    production_routes = production_capacity * topk
    production_packets = production_world * production_routes
    production_entry = host_entry(make_moe_combine(
        production_capacity, production_hidden, topk, production_world,
        weight_dtype=pl.BF16,
    ))
    production_entry.compile(
        empty((production_packets, 4), torch.int64),
        empty((production_packets, production_hidden)),
        empty((production_world, 8), torch.int32),
        empty((production_capacity, topk), torch.float32),
        empty((production_capacity, production_hidden)),
        empty((production_capacity, production_hidden)),
        empty((production_capacity, production_hidden)),
        empty((production_routes, 4), torch.int64),
        empty((production_routes, production_hidden)), production_capacity,
        config=config,
    )

    worker, handles = None, []
    report = {
        "passed": False, "compiled": True, "executed": False,
        "production_shape": [production_capacity, topk, production_hidden, production_world],
    }
    try:
        worker = DistributedWorker([program], config=config)

        def allocate(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.contiguous())
            handles.append(handle)
            return handle

        def read(handle, target):
            worker.copy_from(target.data_ptr(), handle.data_ptr, target.nbytes)
            return target

        packet_routes = torch.full((packet_rows, 4), -1, dtype=torch.int64)
        for token in range(capacity):
            for slot in range(topk):
                row = token * topk + slot
                packet_routes[row] = torch.tensor([slot, token, slot, 0])
        packet_hidden = torch.randn(packet_rows, hidden).bfloat16()
        packet_counts = torch.zeros(world, 8, dtype=torch.int32)
        packet_counts[0, 0] = routes
        route_weights = torch.rand(capacity, topk)
        shared = torch.randn(capacity, hidden).bfloat16()
        routed = torch.zeros_like(shared)
        output = torch.zeros_like(shared)
        returned = torch.full((routes, 4), -1, dtype=torch.int64)
        experts = torch.zeros(routes, hidden, dtype=torch.bfloat16)
        values = [
            packet_routes, packet_hidden, packet_counts, route_weights, shared,
            routed, output, returned, experts,
        ]
        device = [allocate(value) for value in values]
        worker.run(program, *device, capacity, config=config)
        actual_routed = read(device[5], torch.empty_like(routed))
        actual_output = read(device[6], torch.empty_like(output))
        actual_returned = read(device[7], torch.empty_like(returned))
        actual_experts = read(device[8], torch.empty_like(experts))

        expected_experts = packet_hidden[:routes]
        rounded_weights = route_weights.bfloat16().float()
        expected_routed = (
            expected_experts.view(capacity, topk, hidden).float()
            * rounded_weights.unsqueeze(-1)
        ).sum(1).bfloat16()
        expected_output = (expected_routed.float() + shared.float()).bfloat16()
        torch.testing.assert_close(actual_experts, expected_experts, atol=0, rtol=0)
        torch.testing.assert_close(actual_returned, packet_routes[:routes], atol=0, rtol=0)
        torch.testing.assert_close(actual_routed, expected_routed, atol=0, rtol=0)
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
        report.update(
            passed=True, executed=True, shape=[capacity, topk, hidden],
            expert_output_max_abs=float((actual_experts.float() - expected_experts.float()).abs().max()),
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
