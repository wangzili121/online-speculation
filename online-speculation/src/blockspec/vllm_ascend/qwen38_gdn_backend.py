# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：原生 GDN 后端适配：处理递推 query 的分片与 prefill 路径，供原生模型及 draft 算子包装使用。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Adapters for the installed GDN prefill and short recurrent primitives."""

# 依赖：导入 functools。
import functools


# ── recurrent_slices ──
# 职责：按设备支持的 query 宽度切分 GDN 递推，保持跨片状态延续和索引对齐。
# 输入提示：query：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；key：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
def recurrent_slices(operation, *, query, key, value, state, actual_seq_lengths,
                     ssm_state_indices, num_accepted_tokens, g, beta, scale, limit=16):
    """Keep every verifier state while passing at most 16 rows per kernel call."""
    # 依赖：导入 torch。
    import torch

    length = query.shape[0]
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：synchronous batch-one recurrent verification required。
    if actual_seq_lengths.numel() != 2 or ssm_state_indices.numel() < length or length < 1:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("synchronous batch-one recurrent verification required")
    # 上轮只接受到哪个位置，就从那个位置的状态开始；先复制，避免被本轮覆盖。
    selected = ssm_state_indices[num_accepted_tokens.long() - 1].long()
    # 计算：按索引收集行/页/状态，返回新张量。
    initial = state.index_select(0, selected)
    # 整块只整理一次布局；沿 token 维切片不会再次破坏连续性。
    # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义。 此处使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    g = None if g is None else g.contiguous()
    # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    beta = None if beta is None else beta.contiguous()
    outputs = []
    for start in range(0, length, limit):
        end = min(start + limit, length)
        # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        slots = ssm_state_indices[start:end].contiguous()
        state.index_copy_(0, slots[:1].long(), initial)
        # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
        outputs.append(operation(
            query=query[start:end], key=key[start:end], value=value[start:end], state=state,
            actual_seq_lengths=actual_seq_lengths.new_tensor([0, end - start]),
            ssm_state_indices=slots, num_accepted_tokens=None,
            g=None if g is None else g[start:end],
            beta=None if beta is None else beta[start:end], scale=scale))
        if end < length:
            # 计算：按索引收集行/页/状态，返回新张量。
            initial = state.index_select(0, slots[-1:].long())
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


# ── install_recurrent_hook ──
# 职责：为原生 GDN 安装递推切片适配，避免短块 query 限制导致调用失败。
def install_recurrent_hook(vllm_config=None):
    # 依赖：导入 torch。
    import torch
    # 依赖：从 vllm.config 导入 get_current_vllm_config；供本模块调用。
    from vllm.config import get_current_vllm_config
    # 依赖：从 .qwen38_attention 导入 _enabled；供本模块调用。
    from .qwen38_attention import _enabled

    original = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    if getattr(original, "_qwen38_recurrent_hook", False) is True:
        # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
        return
    config = get_current_vllm_config() if vllm_config is None else vllm_config

    @functools.wraps(original)
    # ── install_recurrent_hook.recurrent ──
    # 职责：对原生递推调用做包装，转交保留状态语义的 recurrent_slices。
    def recurrent(**kwargs):
        if not _enabled(config) or kwargs.get("num_accepted_tokens") is None:
            return original(**kwargs)
        return recurrent_slices(original, **kwargs)

    recurrent._qwen38_recurrent_hook = True
    torch.ops._C_ascend.npu_recurrent_gated_delta_rule = recurrent


# ── install_cann_gdn_prefill ──
# 职责：把符合配置的原生 GDN prefill 切换为已有 CANN 路径；不是训练优化器。
def install_cann_gdn_prefill(vllm_config=None):
    # 依赖：导入 torch。
    import torch
    # 依赖：导入 torch_npu。
    import torch_npu
    # 依赖：从 vllm.config 导入 get_current_vllm_config；供本模块调用。
    from vllm.config import get_current_vllm_config
    # 依赖：从 vllm.third_party.flash_linear_attention.ops.l2norm 导入 l2norm_fwd；供本模块调用。
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    # 依赖：从 vllm_ascend.ops 导入 gdn；供本模块调用。
    from vllm_ascend.ops import gdn

    original = gdn.chunk_gated_delta_rule
    if getattr(original, "_qwen38_cann_backend", False):
        # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
        return
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：this torch_npu build does not provide the CANN GDN prefill op。
    if not hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise RuntimeError("this torch_npu build does not provide the CANN GDN prefill op")
    # 每个 worker 绑定一个模型；forward/RPC 不保证存在配置上下文。
    config = get_current_vllm_config() if vllm_config is None else vllm_config

    @functools.wraps(original)
    # ── install_cann_gdn_prefill.prefill ──
    # 职责：处理 GDN 初始文本前向并按需要输出最终状态；是主模型 prefill 的后端适配。
    # 输入提示：q：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
    def prefill(q, k, v, g, beta, *, initial_state=None, output_final_state=False,
                cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False, **kwargs):
        enabled = (config.additional_config or {}).get("qwen38_gdn_prefill_backend") == "cann"
        if not enabled:
            return original(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                            head_first=head_first, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        if set(kwargs) - {"prebuilt_meta"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError(f"unsupported CANN GDN options: {sorted(set(kwargs) - {'prebuilt_meta'})}")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：CANN GDN adapter requires BF16 sequence-major 128-dim inputs and explicit state。
        if (head_first or q.shape[0] != 1 or q.dtype != torch.bfloat16 or q.shape[-1] != 128
                or v.shape[-1] != 128 or initial_state is None or cu_seqlens is None):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("CANN GDN adapter requires BF16 sequence-major 128-dim inputs and explicit state")
        if use_qk_l2norm_in_kernel:
            # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
            q, k = l2norm_fwd(q), l2norm_fwd(k)
        # HF/FLA 状态为 [N,H,K,V]，CANN 为 [N,H,V,K]；使用临时副本，
        # 仍由原生 GDN 编排将返回状态写入本轮对应的正式 cache 槽。
        # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入；使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        state = initial_state.transpose(-1, -2).contiguous().to(torch.bfloat16)
        # 数据：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。 此处使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        output, final = torch_npu.npu_chunk_gated_delta_rule(
            q[0].contiguous(), k[0].contiguous(), v[0].contiguous(),
            beta=beta[0].contiguous(), g=g[0].float().contiguous(), initial_state=state,
            actual_seq_lengths=cu_seqlens[1:].to(torch.int32).contiguous(), scale=q.shape[-1] ** -.5)
        # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入；使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        final = final.transpose(-1, -2).contiguous().to(initial_state.dtype) if output_final_state else None
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return output[None], final

    prefill._qwen38_cann_backend = True
    gdn.chunk_gated_delta_rule = prefill
