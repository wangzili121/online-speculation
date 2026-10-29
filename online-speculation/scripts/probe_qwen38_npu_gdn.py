#!/usr/bin/env python3
"""Check the installed CANN GDN primitive before choosing a native backend."""

import json

import torch
import torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def main():
    torch.manual_seed(11)
    device = "npu:0"
    results = []
    for length in (1, 8, 32):
        q, k = [torch.nn.functional.normalize(torch.randn(length, 8, 128, device=device), dim=-1).bfloat16()
                for _ in range(2)]
        value = torch.randn(length, 24, 128, device=device).bfloat16()
        beta = torch.rand(length, 24, device=device).bfloat16()
        g = -torch.rand(length, 24, device=device)
        state = torch.zeros(1, 24, 128, 128, device=device, dtype=torch.bfloat16)
        try:
            output, final = torch_npu.npu_chunk_gated_delta_rule(
                q, k, value, beta=beta, initial_state=state,
                actual_seq_lengths=torch.tensor([length], dtype=torch.int32, device=device),
                scale=128 ** -.5, g=g)
            torch.npu.synchronize()
            reference, expected_state = torch_chunk_gated_delta_rule(
                q.cpu()[None].repeat_interleave(3, dim=2), k.cpu()[None].repeat_interleave(3, dim=2),
                value.cpu()[None], g.cpu()[None], beta.cpu()[None], output_final_state=True)
            results.append({"length": length, "finite": bool(torch.isfinite(output).all()),
                            "output_max_abs": float((output.cpu() - reference[0]).abs().max()),
                            "state_max_abs": float((final.cpu().transpose(-1, -2) - expected_state).abs().max())})
        except RuntimeError as error:
            results.append({"length": length, "error": str(error)[:1000]})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
