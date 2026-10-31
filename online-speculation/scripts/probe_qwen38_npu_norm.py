#!/usr/bin/env python3
"""Check the existing CANN RMSNorm primitive for frozen residual norms."""

import json

import torch
import torch_npu


def main():
    torch_npu.npu.set_device(0)
    torch.manual_seed(47)
    results = []
    for rows in (1, 16, 32):
        value = torch.randn(1, rows, 5120, device="npu", dtype=torch.bfloat16).requires_grad_()
        weight = torch.randn(5120, device="npu", dtype=torch.bfloat16) * .1
        gamma = 1 + weight
        unit = torch.ones_like(weight)
        expected_input = value.detach().clone().requires_grad_()
        normalized = expected_input.float() * torch.rsqrt(
            expected_input.float().square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized.bfloat16() * gamma
        actual = torch_npu.npu_rms_norm(value, unit, 1e-6)[0] * gamma
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.003)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(value.grad, expected_input.grad, rtol=.025, atol=.0025)
        results.append(dict(rows=rows, forward_max_abs=float((actual.float() - expected.float()).abs().max()),
            gradient_max_abs=float((value.grad.float() - expected_input.grad.float()).abs().max()),
            gradient_relative_l2=float((value.grad.float() - expected_input.grad.float()).norm()
                                       / expected_input.grad.float().norm()), passed=True))
    print(json.dumps(dict(operator="torch_npu.npu_rms_norm + BF16 gamma", results=results), indent=2))


if __name__ == "__main__":
    main()
