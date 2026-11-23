#!/usr/bin/env python3
"""Compile and numerically validate the exact frozen mHC-pre data VJP."""

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
    from blockspec.pypto_dsv41_tail_vjp import (
        make_hc_pre_data_vjp,
        make_hc_pre_projection_forward,
    )

    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0,
        ),
    )
    meta = lambda shape, dtype=torch.bfloat16: torch.empty(
        shape, dtype=dtype, device="meta",
    )

    # Compile the real last-N geometry first.  The compact execution below is
    # a fast numerical oracle and does not weaken this production ABI gate.
    make_hc_pre_projection_forward().compile(
        meta((32 * 4, 5120)), meta((24, 4 * 5120), torch.float32),
        meta((32, 24), torch.float32), meta((32, 8), torch.float32),
        config=config,
    )
    make_hc_pre_data_vjp().compile(
        meta((32 * 4, 5120)), meta((32, 4), torch.float32),
        meta((24, 4 * 5120), torch.float32), meta((32, 8), torch.float32),
        meta((32, 24), torch.float32), meta((32, 5120)),
        meta((32 * 4, 5120)), meta((32 * 4, 5120)),
        meta((32, 16), torch.float32), config=config,
    )

    rows, hidden, copies = 16, 64, 4
    channels, flat = 24, copies * hidden
    forward = make_hc_pre_projection_forward(rows, hidden).compile(
        meta((rows * copies, hidden)), meta((channels, flat), torch.float32),
        meta((rows, channels), torch.float32), meta((rows, 8), torch.float32),
        config=config,
    )
    backward = make_hc_pre_data_vjp(rows, hidden).compile(
        meta((rows * copies, hidden)), meta((rows, copies), torch.float32),
        meta((channels, flat), torch.float32), meta((rows, 8), torch.float32),
        meta((rows, channels), torch.float32), meta((rows, hidden)),
        meta((rows * copies, hidden)), meta((rows * copies, hidden)),
        meta((rows, 16), torch.float32), config=config,
    )

    torch.manual_seed(53)
    residual = (torch.randn(rows, copies, hidden) * .2).bfloat16()
    incoming = torch.randn(rows, copies, dtype=torch.float32) * .2
    function = torch.randn(channels, flat, dtype=torch.float32) * .02
    dprojection = torch.randn(rows, channels, dtype=torch.float32) * .1
    dcollapsed = (torch.randn(rows, hidden) * .1).bfloat16()
    passthrough = (torch.randn_like(residual.float()) * .05).bfloat16()

    flat_value = residual.float().flatten(1)
    expected_inverse = (
        flat_value.square().mean(-1, keepdim=True) + 1e-20
    ).rsqrt()
    expected_projection = flat_value @ function.T * expected_inverse
    reference = residual.float().detach().requires_grad_()
    reference_incoming = incoming.detach().requires_grad_()
    normalized = reference.flatten(1)
    normalized = normalized * (
        normalized.square().mean(-1, keepdim=True) + 1e-20
    ).rsqrt()
    projection = normalized @ function.T
    collapsed = torch.einsum(
        "tc,tcd->td", reference_incoming, reference,
    )
    reference_residual, reference_incoming_gradient = torch.autograd.grad(
        (projection, collapsed), (reference, reference_incoming),
        (dprojection, dcollapsed.float()),
    )
    expected_residual = (reference_residual + passthrough.float()).bfloat16()

    report = {
        "passed": False, "compiled": True, "executed": False,
        "production": [32, 4, 5120],
    }
    worker, handles = None, []
    try:
        worker = DistributedWorker([forward, backward], config=config)

        def allocate(value):
            handle = worker.alloc_tensor(
                tuple(value.shape), value.dtype, init=value.contiguous(),
            )
            handles.append(handle)
            return handle

        projection_output = torch.zeros(rows, channels, dtype=torch.float32)
        inverse_output = torch.zeros(rows, 8, dtype=torch.float32)
        residual_output = torch.zeros_like(residual.flatten(0, 1))
        incoming_output = torch.zeros(rows, 16, dtype=torch.float32)
        projection_handle, inverse_handle = (
            allocate(projection_output), allocate(inverse_output),
        )
        worker.run(
            forward, allocate(residual.flatten(0, 1)), allocate(function),
            projection_handle, inverse_handle, config=config,
        )
        worker.run(
            backward, allocate(residual.flatten(0, 1)), allocate(incoming),
            allocate(function), inverse_handle, allocate(dprojection),
            allocate(dcollapsed), allocate(passthrough.flatten(0, 1)),
            allocate(residual_output), allocate(incoming_output), config=config,
        )
        worker.copy_from(
            projection_output.data_ptr(), projection_handle.data_ptr,
            projection_output.nbytes,
        )
        worker.copy_from(
            inverse_output.data_ptr(), inverse_handle.data_ptr,
            inverse_output.nbytes,
        )
        # The last two allocations are the output handles.
        residual_handle, incoming_handle = handles[-2:]
        worker.copy_from(
            residual_output.data_ptr(), residual_handle.data_ptr,
            residual_output.nbytes,
        )
        worker.copy_from(
            incoming_output.data_ptr(), incoming_handle.data_ptr,
            incoming_output.nbytes,
        )
        torch.testing.assert_close(
            projection_output, expected_projection, atol=3e-4, rtol=3e-4,
        )
        torch.testing.assert_close(
            inverse_output[:, 0], expected_inverse[:, 0], atol=2e-6, rtol=2e-6,
        )
        torch.testing.assert_close(
            residual_output, expected_residual.flatten(0, 1),
            atol=.03125, rtol=.03125,
        )
        torch.testing.assert_close(
            incoming_output[:, :copies], reference_incoming_gradient,
            atol=.02, rtol=.003,
        )
        report.update(
            passed=True, executed=True, fixture=[rows, copies, hidden],
            projection_max_abs=float(
                (projection_output - expected_projection).abs().max()
            ),
            residual_max_abs=float(
                (residual_output.float() - expected_residual.flatten(0, 1).float())
                .abs().max()
            ),
            incoming_max_abs=float(
                (incoming_output[:, :copies] - reference_incoming_gradient)
                .abs().max()
            ),
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
