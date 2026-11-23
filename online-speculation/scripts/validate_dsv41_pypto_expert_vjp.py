#!/usr/bin/env python3
"""Compile and numerically validate the frozen DSV4.1 expert activation VJP."""

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
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_moe_vjp import make_clipped_swiglu_vjp

    torch.manual_seed(43)
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    small = make_clipped_swiglu_vjp(4, 128).compile(
        empty((4, 256)), empty((4, 128)), empty((4, 256)), config=config,
    )
    make_clipped_swiglu_vjp().compile(
        empty((32, 4608)), empty((32, 2304)), empty((32, 4608)), config=config,
    )
    report = {
        "passed": False,
        "compiled": True,
        "executed": False,
        "production_shape": [32, 2304],
    }
    worker, handles = None, []
    try:
        worker = DistributedWorker([small], config=config)

        def allocate(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.contiguous())
            handles.append(handle)
            return handle

        gate_up = (torch.randn(4, 256) * 7).bfloat16()
        output_gradient = torch.randn(4, 128).bfloat16()
        result = torch.zeros(4, 256, dtype=torch.bfloat16)
        result_handle = allocate(result)
        worker.run(small, allocate(gate_up), allocate(output_gradient), result_handle, config=config)
        worker.copy_from(result.data_ptr(), result_handle.data_ptr, result.nbytes)

        gate, up = gate_up.float().chunk(2, dim=-1)
        gate_clamped = gate.clamp(max=10.)
        up_clamped = up.clamp(-10., 10.)
        sigmoid = gate_clamped.sigmoid()
        expected = torch.cat((
            output_gradient.float() * up_clamped * sigmoid
            * (1 + gate_clamped * (1 - sigmoid)) * (gate <= 10.),
            output_gradient.float() * gate_clamped * sigmoid
            * ((up >= -10.) & (up <= 10.)),
        ), dim=-1).bfloat16()
        torch.testing.assert_close(result, expected, atol=.016, rtol=.016)
        report.update(
            passed=True,
            executed=True,
            fixture_shape=[4, 128],
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
