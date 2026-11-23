#!/usr/bin/env python3
"""Compile and execute only the final DSV4.1 mHC-collapse VJP leaf."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output must not already exist")

    def emit(report):
        text = json.dumps(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(text)

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_tail_vjp import (
        make_hc_collapse_forward,
        make_hc_collapse_vjp,
    )

    rows, hidden = (32, 5120) if args.production else (4, 128)
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0,
        ),
    )
    meta = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    forward = make_hc_collapse_forward(rows, hidden).compile(
        meta((rows * 4, hidden)), meta((rows, 4), torch.float32),
        meta((rows, hidden)), config=config,
    )
    backward = make_hc_collapse_vjp(rows, hidden).compile(
        meta((rows * 4, hidden)), meta((rows, 4), torch.float32),
        meta((rows, hidden)), meta((rows * 4, hidden)),
        meta((rows, 16), torch.float32), config=config,
    )
    report = {"passed": True, "compiled": True, "executed": False,
              "programs": 2, "shape": [rows, 4, hidden]}
    if args.compile_only:
        emit(report)
        return

    generator = torch.Generator().manual_seed(419)
    streams = torch.randn(rows, 4, hidden, generator=generator).bfloat16()
    premix = torch.randn(rows, 4, generator=generator, dtype=torch.float32)
    gradient = torch.randn(rows, hidden, generator=generator).bfloat16()
    worker = DistributedWorker([forward, backward], config=config)
    try:
        def allocate(value):
            return worker.alloc_tensor(
                tuple(value.shape), value.dtype, init=value.contiguous(),
            )

        handles = []
        try:
            values = [allocate(streams.flatten(0, 1)), allocate(premix), allocate(gradient)]
            handles.extend(values)
            collapsed_out = worker.alloc_tensor((rows, hidden), torch.bfloat16)
            stream_out = worker.alloc_tensor((rows * 4, hidden), torch.bfloat16)
            premix_out = worker.alloc_tensor(
                (rows, 16), torch.float32,
                init=torch.zeros(rows, 16, dtype=torch.float32),
            )
            handles.extend((collapsed_out, stream_out, premix_out))
            worker.run(forward, values[0], values[1], collapsed_out, config=config)
            worker.run(backward, *values, stream_out, premix_out, config=config)
            actual_collapsed = torch.empty(rows, hidden, dtype=torch.bfloat16)
            actual_stream = torch.empty(rows * 4, hidden, dtype=torch.bfloat16)
            actual_premix = torch.empty(rows, 16, dtype=torch.float32)
            worker.copy_from(
                actual_collapsed.data_ptr(), collapsed_out.data_ptr,
                actual_collapsed.nbytes,
            )
            worker.copy_from(
                actual_stream.data_ptr(), stream_out.data_ptr,
                actual_stream.nbytes,
            )
            worker.copy_from(
                actual_premix.data_ptr(), premix_out.data_ptr,
                actual_premix.nbytes,
            )
        finally:
            for handle in reversed(handles):
                worker.free_tensor(handle)
    finally:
        worker.close()

    expected_stream = (
        gradient.float().unsqueeze(1) * premix.unsqueeze(-1)
    ).bfloat16().flatten(0, 1)
    expected_collapsed = torch.einsum(
        "ti,tid->td", premix, streams.float(),
    ).bfloat16()
    expected_premix = torch.einsum(
        "td,tid->ti", gradient.float(), streams.float(),
    )
    torch.testing.assert_close(
        actual_collapsed, expected_collapsed, atol=.016, rtol=.016,
    )
    torch.testing.assert_close(actual_stream, expected_stream, atol=.016, rtol=.016)
    torch.testing.assert_close(
        actual_premix[:, :4], expected_premix, atol=.02, rtol=.002,
    )
    if torch.count_nonzero(actual_premix[:, 4:]):
        raise AssertionError("padded mHC coefficient gradients must remain zero")
    report.update(
        executed=True,
        collapsed_max_abs=float(
            (actual_collapsed.float() - expected_collapsed.float()).abs().max()
        ),
        stream_max_abs=float((actual_stream.float() - expected_stream.float()).abs().max()),
        premix_max_abs=float((actual_premix[:, :4] - expected_premix).abs().max()),
    )
    emit(report)


if __name__ == "__main__":
    main()
