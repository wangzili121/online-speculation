# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：计算后端：给双视图模型安装现成 NPU 算子或可微分组合。根据设备、形状和是否启用梯度选路，不等同于自研 kernel。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Existing NPU primitives with differentiable FP32-intermediate fallbacks."""

# 依赖：从 functools 导入 partial；供本模块调用。
from functools import partial
# 依赖：从 types 导入 MethodType；供本模块调用。
from types import MethodType

# 依赖：导入 torch。
import torch
# 依赖：从 torch 导入 nn；供本模块调用。
from torch import nn
# 依赖：从 torch.nn 导入 functional；供本模块调用。
from torch.nn import functional as F


# ── swiglu ──
# 职责：对 gate/up 两半计算 SiLU(gate)*up；NPU 优先使用现成融合算子。
def swiglu(packed):
    # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
    if packed.device.type == "npu":
        # 依赖：导入 torch_npu。
        import torch_npu
        return torch_npu.npu_swiglu(packed)
    # 数据：门控特征；Attention 输出乘 sigmoid(gate)，GDN norm 使用相应 SiLU gate。 此处转为 FP32 做分数/归一化/梯度等中间计算。
    gate, up = packed.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(packed.dtype)


# ── l2norm ──
# 职责：对 Q/K 最后一维做 L2 归一化；无梯度且形状符合时走已有高效实现。
# 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
def l2norm(value):
    # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
    if value.device.type == "npu" and not torch.is_grad_enabled() and value.shape[-1] == 128:
        # 依赖：从 vllm.third_party.flash_linear_attention.ops.l2norm 导入 l2norm_fwd；供本模块调用。
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        return l2norm_fwd(value)
    # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
    full = value.float()
    return (full * torch.rsqrt(full.square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype)


# ── causal_conv ──
# 职责：GDN 的因果深度卷积和 SiLU；推理符合限制时走原生算子，训练/其他形状用可微组合。
def causal_conv(x, weight, bias=None, activation="silu", seq_idx=None):
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：unpacked SiLU causal convolution required。
    if seq_idx is not None or activation not in {"silu", "swish"}:
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("unpacked SiLU causal convolution required")
    # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
    if (x.device.type != "npu" or x.dtype != torch.bfloat16 or torch.is_grad_enabled()
            or bias is not None or weight.shape[-1] != 4 or x.shape[0] != 1):
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if torch.is_inference(weight) and torch.is_grad_enabled() and x.requires_grad:
            # 计算：新建独立张量存储；与只建立别名不同。
            weight = weight.clone()
        # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。 此处转为 FP32 做分数/归一化/梯度等中间计算。
        out = F.conv1d(x.float(), weight[:, None].float(), None if bias is None else bias.float(),
                       padding=weight.shape[-1] - 1, groups=weight.shape[0])[..., :x.shape[-1]]
        return F.silu(out).to(x.dtype)
    # 处理完整输入（可能带有 HF 拼接的历史）；临时卷积状态绝不指向主模型。
    # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    raw = x[0].T.contiguous()
    state = x.new_zeros((1, 34, x.shape[1]))
    tail = x.new_zeros((3, x.shape[1]))
    # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
    result = []
    for start in range(0, raw.shape[0], 32):
        part = raw[start:start + 32]
        state[0, :3] = tail
        # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。
        out = torch.empty_like(part)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out, part, weight.T, conv_state=state, bias_opt=None,
            query_start_loc_opt=torch.tensor([0, part.shape[0]], dtype=torch.int32, device=x.device),
            cache_indices_opt=torch.zeros((1, part.shape[0]), dtype=torch.int32, device=x.device),
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=torch.ones(1, dtype=torch.int32, device=x.device),
            activation_mode=1, pad_slot_id=-1, run_mode=1)
        # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
        result.append(out)
        # 计算：沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
        tail = torch.cat((tail, part), dim=0)[-3:]
    return torch.cat(result).T[None]


# ── causal_conv_update ──
# 职责：处理单步卷积状态滚动；state 必须是临时工作状态，不能把 draft 更新写进主模型。
# 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
def causal_conv_update(hidden, state, weight, bias=None, activation="silu"):
    # 计算：沿指定维拼接；是新结果，不等于向原生持久 cache 提交；重排维度表达，元素不变；可能在必要时产生拷贝。
    updated = torch.cat((state[..., 1:], hidden.reshape(*state.shape[:2], 1)), dim=-1)
    # 数据：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
    output = causal_conv(updated, weight, bias, activation)[..., -1:]
    # 执行：把值写入既有目标存储；是否可写需看目标是 draft、master 还是工作 cache。
    state.copy_(updated)
    # 输出：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
    return output


# ── delta_rule ──
# 职责：选择 GDN 递推后端：短块推理用原生递推，训练按配置选 torch 或 FLA 可微实现。
# 输入提示：query：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；key：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
# 输入布局通常 [B,T,heads,dim]，不同于 Full Attention 的 BNSD；状态布局还需与原生算子转换。
# 原生短块推理为各行写 checkpoint，训练可选 FLA 或 Torch 路径，输出是特征及可选最终状态。
def delta_rule(query, key, value, *, fallback, groups, training_fallback=None, **kwargs):
    normalize = kwargs.pop("use_qk_l2norm_in_kernel", False)
    if normalize:
        # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义。
        query, key = l2norm(query), l2norm(key)
    # 数据：原生模型主体，后续共享参数/缓存来自这里。
    native = (query.device.type == "npu" and query.dtype == torch.bfloat16
              and query.shape[0] == 1 and 1 <= query.shape[1] <= 32
              and query.shape[-1] == value.shape[-1] == 128 and not torch.is_grad_enabled()
              and kwargs.get("cu_seqlens") is None
              and not set(kwargs) - {"g", "beta", "initial_state", "output_final_state", "cu_seqlens"})
    if not native:
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if (training_fallback is not None and torch.is_grad_enabled()
                and query.device.type == "npu" and query.dtype == torch.bfloat16):
            return training_fallback(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
        # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
        if query.device.type == "npu" and query.shape[1] < 64 and "chunk_size" not in kwargs:
            # 依赖：从 transformers.models.qwen3_5.modeling_qwen3_5 导入 torch_chunk_gated_delta_rule；供本模块调用。
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
            if fallback is torch_chunk_gated_delta_rule:
                # Keep the existing chunk algorithm, without padding short draft blocks to 64.
                kwargs["chunk_size"] = 1 << (max(1, query.shape[1]) - 1).bit_length()
        return fallback(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
    # 依赖：从 .qwen38_gdn_backend 导入 recurrent_slices；供本模块调用。
    from .qwen38_gdn_backend import recurrent_slices
    length = query.shape[1]
    state = torch.empty((length, value.shape[2], 128, 128), device=query.device, dtype=torch.float32)
    initial = kwargs.get("initial_state")
    # 递推算子写出每行 checkpoint；仅首行需要历史初态，其他行不预先清零。
    if initial is None:
        state[:1].zero_()
    else:
        # 执行：把值写入既有目标存储；是否可写需看目标是 draft、master 还是工作 cache；交换张量维度以适配 head/token 或 K/V 布局。
        state[:1].copy_(initial.transpose(-1, -2))
    # HF 已经展开了 GQA 头；原生递推可直接接收少量 Q/K 头。
    operation = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    operation = getattr(operation, "__wrapped__", operation)
    # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。 此处转为 FP32 做分数/归一化/梯度等中间计算。
    out = recurrent_slices(operation, query=query[0, :, ::groups],
        key=key[0, :, ::groups], value=value[0],
        g=kwargs["g"][0].float(), beta=kwargs["beta"][0], state=state,
        actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device=query.device),
        ssm_state_indices=torch.arange(length, dtype=torch.int32, device=query.device),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=query.device), scale=128 ** -.5)
    # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；交换张量维度以适配 head/token 或 K/V 布局。
    final = state[-1:].transpose(-1, -2).contiguous() if kwargs.get("output_final_state", False) else None
    # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
    return out[None], final


# ── GatedRMSNorm ──
# 职责：类型 `GatedRMSNorm`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class GatedRMSNorm(nn.Module):
    # ── GatedRMSNorm.__init__ ──
    # 职责：保存 GDN 的 norm 权重和 epsilon，供前向选择融合或可微分实现。
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.variance_epsilon = weight, eps

    # ── GatedRMSNorm.forward ──
    # 职责：对 GDN 输出做 RMSNorm 并乘 SiLU(gate)；无梯度优先融合，训练走可微路径。
    # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；gate：门控特征；Attention 输出乘 sigmoid(gate)，GDN norm 使用相应 SiLU gate。
    def forward(self, value, gate):
        weight = self.weight
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if value.device.type == "npu" and not torch.is_grad_enabled():
            # 依赖：从 vllm_ascend.ops.layernorm 导入 layer_norm_fwd_npu；供本模块调用。
            from vllm_ascend.ops.layernorm import layer_norm_fwd_npu
            return layer_norm_fwd_npu(value, weight, None, self.variance_epsilon, z=gate,
                group_size=None, norm_before_gate=True, is_rms_norm=True, activation="swish")[0]
        # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
        if torch.is_inference(weight) and torch.is_grad_enabled() and (value.requires_grad or gate.requires_grad):
            # 计算：新建独立张量存储；与只建立别名不同。
            weight = weight.clone()
        # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
        full = value.float()
        # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
        if value.device.type == "npu":
            # 依赖：导入 torch_npu。
            import torch_npu
            # 数据：归一化后的浮点特征，送入 mixer 或 MLP。 此处转为 FP32 做分数/归一化/梯度等中间计算。
            normalized = torch_npu.npu_rms_norm(full, weight.float(), self.variance_epsilon)[0]
            return (normalized * F.silu(gate.float())).to(value.dtype)
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
        normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (normalized * weight.float() * F.silu(gate.float())).to(value.dtype)


# ── configure_gdn ──
# 职责：给 mixer 装配归一化、卷积和 delta rule 实现；这里选择后端，并不执行一次完整推理。
def configure_gdn(mixer, training_backend="torch"):
    # 依赖：从 transformers.models.qwen3_5.modeling_qwen3_5 导入 torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule；供本模块调用。
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
    )
    training_fallback = None
    if training_backend == "fla":
        # 依赖：从 importlib.metadata 导入 version；供本模块调用。
        from importlib.metadata import version
        # 依赖：导入 os。
        import os
        # 依赖：导入 sys。
        import sys
        dependencies = os.environ.get("ORTHRUS_QWEN38_FLA_PATH")
        if dependencies and dependencies not in sys.path:
            # FLA captures its device context on import; load only in the initialized worker.
            sys.path.insert(0, dependencies)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：the optional GDN training backend requires fla-core==0.6.0。
        if version("fla-core") != "0.6.0":
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("the optional GDN training backend requires fla-core==0.6.0")
        # 依赖：从 fla.ops.gated_delta_rule 导入 chunk_gated_delta_rule；供本模块调用。
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # 依赖：从 fla.utils.hardware 导入 IS_NPU；供本模块调用。
        from fla.utils.hardware import IS_NPU
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：FLA must be loaded with an available NPU runtime and writable Triton cache。
        if not IS_NPU:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("FLA must be loaded with an available NPU runtime and writable Triton cache")
        training_fallback = chunk_gated_delta_rule
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：GDN training backend must be torch or fla。
    elif training_backend != "torch":
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("GDN training backend must be torch or fla")
    mixer.norm = GatedRMSNorm(mixer.norm.weight, mixer.layer_norm_epsilon)
    mixer.causal_conv1d_fn = causal_conv
    mixer.causal_conv1d_update = causal_conv_update
    groups = mixer.num_v_heads // mixer.num_k_heads
    mixer.chunk_gated_delta_rule = partial(delta_rule, fallback=torch_chunk_gated_delta_rule,
                                           groups=groups, training_fallback=training_fallback)
    mixer.recurrent_gated_delta_rule = partial(delta_rule, fallback=torch_recurrent_gated_delta_rule, groups=groups)


# ── SharedRotary ──
# 职责：类型 `SharedRotary`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SharedRotary:
    # ── SharedRotary.__init__ ──
    # 职责：引用原生 RoPE cos/sin 缓存及旋转配置；不是再训练一套位置参数。
    def __init__(self, original):
        self.cache = original.cos_sin_cache
        self.sections = tuple(original.mrope_section)
        self.interleaved = original.mrope_interleaved
        self.dim = original.rotary_dim

    # ── SharedRotary.__deepcopy__ ──
    # 职责：控制深复制的共享语义；返回 self 表示对象继续共享，而非创建新的通信组/缓存。
    def __deepcopy__(self, memo):
        return self

    # ── SharedRotary.positions ──
    # 职责：根据严格历史长度创建当前块的绝对位置索引，供 RoPE 读取。
    def positions(self, start, length, device):
        return torch.arange(start, start + length, device=device)

    # ── SharedRotary.coefficients ──
    # 职责：从共享 RoPE 表按位置取 cos/sin 并调整布局与 dtype。
    # 输入提示：positions：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE。
    def coefficients(self, positions, dtype):
        # 计算：按索引收集行/页/状态，返回新张量。
        cos, sin = self.cache.index_select(0, positions).chunk(2, -1)
        return tuple(torch.cat((value, value), dim=-1)[None, None].to(dtype) for value in (cos, sin))


# ── full_attention ──
# 职责：draft/AR 视图的 Full Attention 执行：投影 Q/K/V 与 gate，norm/RoPE，拼入工作历史，融合 Attention，门控和 O 投影。
# 调用者：configure_attention 用 MethodType 替换 mixer.forward 后，经 _HybridLayer.forward 到这里。
# 输入：hidden_states [1,K,H]、位置数据、可见性 mask 和临时 cache。
# 输出：(经门控和 O 投影的特征 [1,K,H], None)；不是最终 token logits。
# 本函数负责 draft/模型视图的 Attention；qwen38_attention.py 则负责原生 verifier 的分页 query 限制。
def full_attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
    if (hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] != 1 or kwargs
            or self.config._attn_implementation != "sdpa"):
        return self._reference_forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
    # 依赖：导入 torch_npu。
    import torch_npu

    batch, length, _ = hidden_states.shape
    shape = (batch, length, -1, self.head_dim)
    # 数据：Q 与输出 gate 的投影结果，随后拆分；本模型不是只有普通 Q 投影。
    projected_q, projected_k, projected_v = (projection(hidden_states)
        for projection in (self.q_proj, self.k_proj, self.v_proj))
    # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；门控特征；Attention 输出乘 sigmoid(gate)，GDN norm 使用相应 SiLU gate。 此处重排维度表达，元素不变；可能在必要时产生拷贝。
    query, gate = projected_q.reshape(batch, length, -1, 2 * self.head_dim).chunk(2, -1)
    # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局；K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义。 此处交换张量维度以适配 head/token 或 K/V 布局。
    query, key = query.reshape(shape).transpose(1, 2), projected_k.reshape(shape).transpose(1, 2)
    # 数据：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。 此处交换张量维度以适配 head/token 或 K/V 布局。
    value = projected_v.reshape(shape).transpose(1, 2)
    native_rope = self._native_rope
    if native_rope is not None:
        # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。
        row = None if past_key_values is None else past_key_values.layers[self.layer_idx].keys
        # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
        past = 0 if row is None else row.shape[2]
        # 数据：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE。
        positions = native_rope.positions(past, length, hidden_states.device)

    # 梯度选路：训练需要保留 autograd；无梯度推理才使用仅支持推理的融合路径。
    if native_rope is not None and not torch.is_grad_enabled():
        # 计算：重排维度表达，元素不变；可能在必要时产生拷贝；沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
        packed = torch.cat((projected_q, projected_k, projected_v), dim=-1).reshape(length, -1)
        # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        cos_sin = native_rope.cache.index_select(0, positions).to(packed.dtype)[None].expand(3, -1, -1).contiguous()
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q, k, v, raw_gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=packed, q_weight=1 + self.q_norm.weight, k_weight=1 + self.k_norm.weight,
            cos_sin=cos_sin, num_q_heads=query.shape[1], num_kv_heads=key.shape[1], head_size=self.head_dim,
            eps=self.config.rms_norm_eps, mrope_section=list(native_rope.sections),
            is_interleaved=native_rope.interleaved, rope_dim=native_rope.dim, has_gate=True)
        # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局。 此处交换张量维度以适配 head/token 或 K/V 布局。
        query = q.reshape(shape).transpose(1, 2)
        # 数据：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。 此处交换张量维度以适配 head/token 或 K/V 布局。
        key, value = (part.reshape(shape).transpose(1, 2) for part in (k, v))
        # 数据：门控特征；Attention 输出乘 sigmoid(gate)，GDN norm 使用相应 SiLU gate。 此处重排维度表达，元素不变；可能在必要时产生拷贝。
        gate = raw_gate.reshape(batch, length, -1)
    else:
        # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
        cos, sin = (native_rope.coefficients(positions, query.dtype) if native_rope is not None
                    else tuple(component[:, None].contiguous() for component in position_embeddings))
        rotary_dim = cos.shape[-1]

        # ── full_attention.normalize_rotate ──
        # 职责：对 Q 或 K 先做 FP32 中间计算的 RMSNorm，再只旋转 RoPE 对应维度，最后回到执行 dtype。
        # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        def normalize_rotate(value, weight):
            # 计算：转为 FP32 做分数/归一化/梯度等中间计算。
            full = value.float()
            # 原生融合分支先形成 BF16 gamma，但 norm 与 RoPE 之间不舍入。
            # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
            normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
            # 数据：归一化后的浮点特征，送入 mixer 或 MLP。 此处转为 FP32 做分数/归一化/梯度等中间计算。
            normalized = normalized * (1 + weight).float()
            # 计算：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；转为 FP32 做分数/归一化/梯度等中间计算。
            rotated = torch_npu.npu_rotary_mul(normalized[..., :rotary_dim].contiguous(), cos.float(), sin.float())
            return torch.cat((rotated, normalized[..., rotary_dim:]), dim=-1).to(value.dtype).contiguous()

        # 数据：Attention/GDN 的 Q 特征，注意 batch/head/sequence 维布局。
        query = normalize_rotate(query, self.q_norm.weight)
        # 数据：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义。
        key = normalize_rotate(key, self.k_norm.weight)
    if past_key_values is not None:
        # 数据：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
        key, value = past_key_values.update(key, value, self.layer_idx)
    # FAS 直接支持 GQA；保留少量 KV 头，避免复制整段历史。
    # PyTorch 加性 mask 中的 -inf 对应 Ascend FAS 的 True（禁止读取）。
    # 数据：NPU Attention 的禁止访问 mask；True 表示屏蔽，与部分 PyTorch bool mask 语义相反。 此处使张量布局连续，满足算子/通信输入要求；必要时会拷贝。
    blocked = (None if attention_mask is None else (~attention_mask if attention_mask.dtype == torch.bool
               else torch.isneginf(attention_mask)).contiguous())
    # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。 此处调用现成 NPU 融合 Attention；mask 和传入的历史决定可见范围。
    out = torch_npu.npu_fusion_attention(query, key.contiguous(), value.contiguous(),
        query.shape[1], input_layout="BNSD", atten_mask=blocked, scale=self.scaling,
        keep_prob=1., sparse_mode=0)[0]
    # 数据：当前计算输出，随后还可能经过 head 合并、门控或 O 投影。 此处重排维度表达，元素不变；可能在必要时产生拷贝。
    out = out.transpose(1, 2).reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
    # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
    return self.o_proj(out), None


# ── configure_attention ──
# 职责：保留 reference forward，并把已有 NPU 算子包装绑定为该 mixer 的 forward。
def configure_attention(mixer, native_rope=None):
    mixer._reference_forward = mixer.forward
    mixer._native_rope = None if native_rope is None else SharedRotary(native_rope)
    mixer.forward = MethodType(full_attention, mixer)
