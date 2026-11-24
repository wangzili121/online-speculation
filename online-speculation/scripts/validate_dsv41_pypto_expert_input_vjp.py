#!/usr/bin/env python3
"""Compile and numerically validate the frozen W8A8 expert input VJP."""

import argparse
import json
from pathlib import Path

from blockspec.pypto_dsv41_moe_vjp import EXPERT_DEQUANT_TILE


def pack_kn(value):
    """Pack logical INT8 [K,N] into the native [N/64*K,64] layout."""
    k, n = value.shape
    if n % 64:
        raise ValueError("N must align to 64")
    return value.reshape(k, n // 64, 64).permute(1, 0, 2).reshape(-1, 64).contiguous()


def validate(rows, hidden, intermediate, active, config, *, dequant_tile=EXPERT_DEQUANT_TILE):
    import torch
    from blockspec.pypto_dsv41_moe_vjp import make_frozen_w8a8_expert_input_vjp
    from blockspec.pypto_dsv41_resident_abi import resident_entry

    torch.manual_seed(47)
    value = torch.zeros((rows, hidden), dtype=torch.bfloat16)
    output_gradient = torch.zeros_like(value)
    value[:active] = torch.randn(active, hidden) * .1
    output_gradient[:active] = torch.randn(active, hidden) * .1
    gate_codes = torch.randint(-8, 9, (hidden, 2 * intermediate), dtype=torch.int8)
    down_codes = torch.randint(-8, 9, (intermediate, hidden), dtype=torch.int8)
    gate_scale = torch.rand(2 * intermediate).mul_(.01).add_(.001)
    down_scale = torch.rand(hidden).mul_(.01).add_(.001)
    gate_weight = (gate_codes.float() * gate_scale).bfloat16()
    down_weight = (down_codes.float() * down_scale).bfloat16()
    gate_up = (value[:active].float() @ gate_weight.float()).bfloat16().float()
    activation_gradient = (
        output_gradient[:active].float() @ down_weight.float().T
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

    result = torch.empty_like(value)
    chip = resident_entry(make_frozen_w8a8_expert_input_vjp(rows, hidden, intermediate, dequant_tile=dequant_tile))
    chip(
        value, output_gradient, pack_kn(gate_codes), gate_scale, pack_kn(down_codes), down_scale,
        result, config=config,
    )
    torch.testing.assert_close(result[:active], expected, atol=.003, rtol=.03)
    if torch.count_nonzero(result[active:]):
        raise AssertionError("Inactive expert rows must return zero")
    return {
        "shape": [rows, hidden, intermediate], "active_rows": active,
        "max_abs": float((result[:active].float() - expected.float()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--dequant-tile", type=int, choices=(16, 32, 64, 128), default=EXPERT_DEQUANT_TILE)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")

    import torch
    from pypto.backend import BackendType
    from pypto.runtime import ExecutionMode, RunConfig

    torch.set_num_threads(4)
    config = RunConfig(
        arch=BackendType.Ascend910, device_id=args.device,
        execution_mode=ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD,
    )
    shapes = ((16, 5120, 2304, 3), (32, 5120, 2304, 19), (32, 5120, 768, 31))
    if not args.production:
        shapes = ((16, 256, 256, 15),)
    report = {
        "passed": False, "compiled": False, "executed": False,
        "npu_execution": False, "simulator_execution": False, "cases": [],
    }
    try:
        for rows, hidden, intermediate, active in shapes:
            report["cases"].append(validate(rows, hidden, intermediate, active, config, dequant_tile=args.dequant_tile))
        report.update(
            passed=True, compiled=True, executed=True,
            npu_execution=not args.simulate, simulator_execution=args.simulate,
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "expert-input-vjp", **report}), flush=True)


if __name__ == "__main__":
    main()
