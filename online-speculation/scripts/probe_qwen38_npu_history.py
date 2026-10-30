#!/usr/bin/env python3
"""Check native convolution/SSM checkpoints against the draft history reader."""

import json
from types import SimpleNamespace as NS

import torch
import torch_npu  # noqa: F401


def main():
    from vllm_ascend.utils import enable_custom_op
    if not enable_custom_op():
        raise RuntimeError("native GDN operators are unavailable")
    from blockspec.vllm_ascend.qwen38_gdn_backend import recurrent_slices
    from blockspec.vllm_ascend.qwen38_history import gdn_history

    torch.manual_seed(17)
    device, length, channels = "npu:0", 32, (8 * 2 + 24) * 128
    indices = torch.randperm(40)[:32].to(device, torch.int32)
    cache_row = int(indices[0].item())
    conv = torch.zeros(40, 34, channels, dtype=torch.bfloat16, device=device)
    state = torch.randn(40, 24, 128, 128, device=device) * .01
    weight = torch.randn(channels, 4).bfloat16().to(device)
    previous = torch.randn(3, channels).bfloat16().to(device)
    conv[cache_row, :3] = previous
    accepted_before, results = 1, []
    for round_index, next_accepted in enumerate((17, 32, 1)):
        raw = torch.randn(length, channels).bfloat16().to(device)
        output = torch.empty_like(raw)
        meta = NS(spec_sequence_masks=torch.tensor([True], device=device),
                  spec_state_indices_tensor=indices[None], spec_decode_metadata=NS(
                      spec_causal_conv1d=NS(cache_indices=indices[None],
                                           num_accepted_tokens=torch.tensor([accepted_before], device=device))))
        initial = state[indices[accepted_before - 1].long()].float().cpu()
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output, raw, weight.T, conv_state=conv, bias_opt=None,
            query_start_loc_opt=torch.tensor([0, length], dtype=torch.int32, device=device),
            cache_indices_opt=indices[None], initial_state_mode_opt=None,
            num_accepted_tokens_opt=meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens,
            activation_mode=1, pad_slot_id=-1, run_mode=1)
        q, k = [torch.nn.functional.normalize(torch.randn(length, 8, 128), dim=-1).bfloat16()
                for _ in range(2)]
        value = torch.randn(length, 24, 128).bfloat16()
        g, beta = -torch.rand(length, 24), torch.rand(length, 24).bfloat16()
        recurrent_slices(torch.ops._C_ascend.npu_recurrent_gated_delta_rule,
                         query=q.to(device), key=k.to(device), value=value.to(device),
                         g=g.to(device), beta=beta.to(device), state=state, scale=128 ** -.5,
                         actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device=device),
                         ssm_state_indices=indices,
                         num_accepted_tokens=meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.to(torch.int32))
        expected, recurrent = [], initial
        for i in range(length):
            ki = k[i].float().repeat_interleave(3, 0)
            recurrent = recurrent * g[i].exp()[:, None, None]
            delta = (value[i].float() - (recurrent * ki[:, None]).sum(-1)) * beta[i, :, None].float()
            recurrent = recurrent + delta[:, :, None] * ki[:, None]
            expected.append(recurrent)
        native_before = conv.clone(), state.clone()
        errors = []
        for accepted in range(1, length + 1):
            c, s = gdn_history((conv, state), meta, accepted=accepted,
                               query_length=length, kernel_width=4)
            tail = torch.cat((previous, raw[:accepted]), dim=0)[-3:]
            torch.testing.assert_close(c[0, :, 1:], tail.T, rtol=0, atol=0)
            reference = expected[accepted - 1].transpose(-1, -2)[None]
            torch.testing.assert_close(s.cpu(), reference, rtol=1e-5, atol=1e-6)
            errors.append(float((s.cpu() - reference).abs().max()))
            c.zero_()
            s.zero_()
        for actual, before in zip((conv, state), native_before, strict=True):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
        results.append(dict(round=round_index, previous_accepted=accepted_before,
                            tested_accepted_positions=32, state_max_abs=max(errors), cache_unchanged=True))
        previous = torch.cat((previous, raw[:next_accepted]), dim=0)[-3:]
        accepted_before = next_accepted
    print(json.dumps(dict(passed=True, rounds=results)))


if __name__ == "__main__":
    main()
