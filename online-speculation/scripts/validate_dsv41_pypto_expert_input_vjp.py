#!/usr/bin/env python3
"""Compile and numerically validate the frozen W8A8 expert input VJP."""

import argparse
import json
from pathlib import Path


def pack_kn(value):
    """Pack logical INT8 [K,N] into the native [N/64*K,64] layout."""
    k, n = value.shape
    if n % 64:
        raise ValueError("N must align to 64")
    return value.reshape(k, n // 64, 64).permute(1, 0, 2).reshape(-1, 64).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_moe_vjp import make_frozen_w8a8_expert_input_vjp

    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")

    # Compile the production expert slab even though the numerical fixture is
    # intentionally compact enough to return quickly on a shared development card.
    make_frozen_w8a8_expert_input_vjp().compile(
        empty((64, 5120)), empty((64, 5120)),
        empty((5120 * (4608 // 64), 64), torch.int8), empty((4608,), torch.float32),
        empty((2304 * (5120 // 64), 64), torch.int8), empty((5120,), torch.float32),
        empty((64, 5120)), config=config,
    )

    rows, hidden, intermediate = 16, 256, 256
    program = make_frozen_w8a8_expert_input_vjp(rows, hidden, intermediate).compile(
        empty((rows, hidden)), empty((rows, hidden)),
        empty((hidden * (2 * intermediate // 64), 64), torch.int8),
        empty((2 * intermediate,), torch.float32),
        empty((intermediate * (hidden // 64), 64), torch.int8),
        empty((hidden,), torch.float32), empty((rows, hidden)), config=config,
    )

    torch.manual_seed(47)
    value = (torch.randn(rows, hidden) * .1).bfloat16()
    output_gradient = (torch.randn(rows, hidden) * .1).bfloat16()
    gate_codes = torch.randint(-8, 9, (hidden, 2 * intermediate), dtype=torch.int8)
    down_codes = torch.randint(-8, 9, (intermediate, hidden), dtype=torch.int8)
    gate_scale = torch.rand(2 * intermediate).mul_(.01).add_(.001)
    down_scale = torch.rand(hidden).mul_(.01).add_(.001)
    gate_weight = (gate_codes.float() * gate_scale).bfloat16()
    down_weight = (down_codes.float() * down_scale).bfloat16()
    gate_up = (value.float() @ gate_weight.float()).bfloat16().float()
    activation_gradient = (
        output_gradient.float() @ down_weight.float().T
    ).bfloat16().float()
    gate, up = gate_up.chunk(2, dim=-1)
    gate_clamped, up_clamped = gate.clamp(max=10.), up.clamp(-10., 10.)
    sigmoid = gate_clamped.sigmoid()
    dgate = activation_gradient * up_clamped * sigmoid * (
        1 + gate_clamped * (1 - sigmoid)
    ) * (gate <= 10.)
    dup = activation_gradient * gate_clamped * sigmoid * (
        (up >= -10.) & (up <= 10.)
    )
    gate_gradient = torch.cat((dgate, dup), -1).bfloat16()
    expected = (gate_gradient.float() @ gate_weight.float().T).bfloat16()

    result = torch.zeros_like(expected)
    worker, handles = None, []
    report = {"passed": False, "compiled": True, "executed": False}
    try:
        worker = DistributedWorker([program], config=config)

        def allocate(tensor):
            handle = worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor.contiguous())
            handles.append(handle)
            return handle

        output = allocate(result)
        worker.run(
            program, allocate(value), allocate(output_gradient), allocate(pack_kn(gate_codes)),
            allocate(gate_scale), allocate(pack_kn(down_codes)), allocate(down_scale), output,
            config=config,
        )
        worker.copy_from(result.data_ptr(), output.data_ptr, result.nbytes)
        torch.testing.assert_close(result, expected, atol=.03125, rtol=.03125)
        report.update(
            passed=True, executed=True, fixture=[rows, hidden, intermediate],
            production=[64, 5120, 2304],
            max_abs=float((result.float() - expected.float()).abs().max()),
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
