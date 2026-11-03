#!/usr/bin/env python3
"""Compare existing NPU multi-tensor primitives on TP4 draft gradient shapes."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from audit_qwen38_execution import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    torch.manual_seed(731)
    shapes = [(2560, 5120), (1536, 5120), (5120, 1536), (3072, 5120), (256, 5120),
              (5120,), (128,), (2560, 1, 4), (12,)]
    gradients = [torch.randn(shape, device="npu", dtype=torch.float32) for shape in shapes]

    def original_norm():
        total = torch.zeros((), device="npu")
        for gradient in gradients:
            total += gradient.square().sum()
        return total.sqrt()

    def foreach_norm():
        return torch.stack(torch._foreach_norm(gradients, 2)).square().sum().sqrt()

    expected = original_norm()
    report = dict(torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                  shapes=shapes, scope="TP4 gradient primitives; excludes model forward and TP communication")
    try:
        actual = foreach_norm()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        report["norm"] = dict(numerical_pass=True, expected=float(expected), actual=float(actual),
            original=measure(original_norm, args.repeats, torch.npu.synchronize),
            foreach=measure(foreach_norm, args.repeats, torch.npu.synchronize))
    except (RuntimeError, NotImplementedError, AssertionError) as error:
        report["norm"] = dict(numerical_pass=False, error=str(error))

    scale = torch.tensor(.999, device="npu")
    original = [gradient.clone() for gradient in gradients]
    for gradient in original:
        gradient.mul_(scale)
    try:
        actual = [gradient.clone() for gradient in gradients]
        torch._foreach_mul_(actual, scale)
        for a, b in zip(actual, original, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        def original_scale():
            for gradient in original:
                gradient.mul_(scale)

        def foreach_scale():
            torch._foreach_mul_(actual, scale)

        report["scale"] = dict(numerical_pass=True,
            original=measure(original_scale, args.repeats, torch.npu.synchronize),
            foreach=measure(foreach_scale, args.repeats, torch.npu.synchronize))
    except (RuntimeError, NotImplementedError, AssertionError) as error:
        report["scale"] = dict(numerical_pass=False, error=str(error))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
