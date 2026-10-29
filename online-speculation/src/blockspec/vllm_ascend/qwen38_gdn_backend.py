"""Adapters for the installed GDN prefill and short recurrent primitives."""

import functools


def recurrent_slices(operation, *, query, key, value, state, actual_seq_lengths,
                     ssm_state_indices, num_accepted_tokens, g, beta, scale, limit=16):
    """Keep every verifier state while passing at most 16 rows per kernel call."""
    import torch

    length = query.shape[0]
    if actual_seq_lengths.numel() != 2 or ssm_state_indices.numel() < length or length < 1:
        raise ValueError("synchronous batch-one recurrent verification required")
    # 上轮只接受到哪个位置，就从那个位置的状态开始；先复制，避免被本轮覆盖。
    selected = ssm_state_indices[num_accepted_tokens.long() - 1].long()
    initial = state.index_select(0, selected).clone()
    outputs = []
    for start in range(0, length, limit):
        end = min(start + limit, length)
        slots = ssm_state_indices[start:end].contiguous()
        state.index_copy_(0, slots[:1].long(), initial)
        outputs.append(operation(
            query=query[start:end].contiguous(), key=key[start:end].contiguous(),
            value=value[start:end].contiguous(), state=state,
            actual_seq_lengths=actual_seq_lengths.new_tensor([0, end - start]),
            ssm_state_indices=slots, num_accepted_tokens=None,
            g=None if g is None else g[start:end].contiguous(),
            beta=None if beta is None else beta[start:end].contiguous(), scale=scale))
        initial = state.index_select(0, slots[-1:].long()).clone()
    return torch.cat(outputs, dim=0)


def install_recurrent_hook():
    import torch
    from vllm.config import get_current_vllm_config
    from .qwen38_attention import _enabled

    original = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    if getattr(original, "_qwen38_recurrent_hook", False) is True:
        return

    @functools.wraps(original)
    def recurrent(**kwargs):
        if not _enabled(get_current_vllm_config()) or kwargs.get("num_accepted_tokens") is None:
            return original(**kwargs)
        return recurrent_slices(original, **kwargs)

    recurrent._qwen38_recurrent_hook = True
    torch.ops._C_ascend.npu_recurrent_gated_delta_rule = recurrent


def install_cann_gdn_prefill():
    import torch
    import torch_npu
    from vllm.config import get_current_vllm_config
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    from vllm_ascend.ops import gdn

    original = gdn.chunk_gated_delta_rule
    if getattr(original, "_qwen38_cann_backend", False):
        return
    if not hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
        raise RuntimeError("this torch_npu build does not provide the CANN GDN prefill op")

    @functools.wraps(original)
    def prefill(q, k, v, g, beta, *, initial_state=None, output_final_state=False,
                cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False, **kwargs):
        config = get_current_vllm_config()
        enabled = (config.additional_config or {}).get("qwen38_gdn_prefill_backend") == "cann"
        if not enabled:
            return original(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                            head_first=head_first, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        if set(kwargs) - {"prebuilt_meta"}:
            raise ValueError(f"unsupported CANN GDN options: {sorted(set(kwargs) - {'prebuilt_meta'})}")
        if (head_first or q.shape[0] != 1 or q.dtype != torch.bfloat16 or q.shape[-1] != 128
                or v.shape[-1] != 128 or initial_state is None or cu_seqlens is None):
            raise ValueError("CANN GDN adapter requires BF16 sequence-major 128-dim inputs and explicit state")
        if use_qk_l2norm_in_kernel:
            q, k = l2norm_fwd(q), l2norm_fwd(k)
        # HF/FLA 状态为 [N,H,K,V]，CANN 为 [N,H,V,K]；使用临时副本，
        # 仍由原生 GDN 编排将返回状态写入本轮对应的正式 cache 槽。
        state = initial_state.transpose(-1, -2).contiguous().to(torch.bfloat16)
        output, final = torch_npu.npu_chunk_gated_delta_rule(
            q[0].contiguous(), k[0].contiguous(), v[0].contiguous(),
            beta=beta[0].contiguous(), g=g[0].float().contiguous(), initial_state=state,
            actual_seq_lengths=cu_seqlens[1:].to(torch.int32).contiguous(), scale=q.shape[-1] ** -.5)
        final = final.transpose(-1, -2).contiguous().to(initial_state.dtype) if output_final_state else None
        return output[None], final

    prefill._qwen38_cann_backend = True
    gdn.chunk_gated_delta_rule = prefill
