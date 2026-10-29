"""Use the installed CANN GDN prefill op instead of the affected Triton path."""

import functools


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
