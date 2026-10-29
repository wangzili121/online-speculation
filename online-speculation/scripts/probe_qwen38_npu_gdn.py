#!/usr/bin/env python3
"""Check the installed CANN GDN primitive before choosing a native backend."""

import json

import torch
import torch_npu


def probe_recurrent():
    from vllm_ascend.utils import enable_custom_op
    from blockspec.vllm_ascend.qwen38_gdn_backend import recurrent_slices

    if not enable_custom_op():
        raise RuntimeError("native recurrent GDN op is unavailable")
    results = []
    for length, accepted in ((8, 27), (17, 1), (31, 16), (32, 32)):
        q, k = [torch.nn.functional.normalize(torch.randn(length, 8, 128), dim=-1).bfloat16()
                for _ in range(2)]
        value, beta = torch.randn(length, 24, 128).bfloat16(), torch.rand(length, 24).bfloat16()
        g, state = -torch.rand(length, 24), torch.randn(32, 24, 128, 128) * .01
        expected_state = state.clone()
        hidden, reference = state[accepted - 1].clone(), []
        for index in range(length):
            ki = k[index].float().repeat_interleave(3, 0)
            qi = q[index].float().repeat_interleave(3, 0) * 128 ** -.5
            hidden = hidden * g[index].exp()[:, None, None]
            delta = (value[index].float() - (hidden * ki[:, None]).sum(-1)) * beta[index, :, None].float()
            hidden = hidden + delta[:, :, None] * ki[:, None]
            expected_state[index] = hidden
            reference.append((hidden * qi[:, None]).sum(-1))
        npu_state = state.npu()
        output = recurrent_slices(
            torch.ops._C_ascend.npu_recurrent_gated_delta_rule,
            query=q.npu(), key=k.npu(), value=value.npu(), state=npu_state,
            g=g.npu(), beta=beta.npu(), scale=128 ** -.5,
            actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device="npu"),
            ssm_state_indices=torch.arange(32, dtype=torch.int32, device="npu"),
            num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device="npu"))
        torch.npu.synchronize()
        reference = torch.stack(reference)
        actual, final = output.float().cpu(), npu_state.cpu()
        torch.testing.assert_close(actual, reference, rtol=.01, atol=.002)
        torch.testing.assert_close(final, expected_state, rtol=.01, atol=.002)
        results.append(dict(length=length, previous_accepted=accepted,
                            output_max_abs=float((actual - reference).abs().max()),
                            state_max_abs=float((final - expected_state).abs().max())))
    return results


def main():
    from vllm_ascend.utils import enable_custom_op
    # 与 worker 一致，必须在首次 NPU 计算前注册镜像自带的 AscendC 算子。
    if not enable_custom_op():
        raise RuntimeError("native recurrent GDN op is unavailable")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
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
    print(json.dumps({"recurrent_slices": probe_recurrent()}))


if __name__ == "__main__":
    main()
