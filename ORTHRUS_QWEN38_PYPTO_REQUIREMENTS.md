# Orthrus-Qwen3.8 NPU 推理适配与算子设计

## 1. 背景与适配目标

Orthrus 用一个 draft 分支并行预测多个 token，再由原模型一次验证这些候选。本文面向 Qwen3.8-27B 文本模型，说明将该分支交给 PyPTO 执行时，需要保留哪些模型计算、调整哪些算子入口，以及如何与主模型协作。范围仅为推理，不包含训练和反向算子。

### 1.1 草稿分支与主模型的关系

Qwen3.8-27B 的文本骨干沿用 Qwen3.5 系列的混合结构，共 64 层：48 层 GDN 和 16 层 Full Attention。两类层的排列与 draft 参数所在位置如下：

```text
共享 Embedding
    |
    v
[ GDN -> GDN -> GDN -> Full Attention ] x 16
    | 每层：共享 input norm -> 本层 mixer -> 残差
    |                       -> 共享 post norm / MLP -> 残差
    v
共享 Final Norm / LM head

每层 mixer 有两套参数：
  AR mixer      ：原模型参数，产生正式历史
  Draft mixer   ：独立参数，读取 AR 历史，在临时工作区计算候选
```

每轮 draft 的输入是 `anchor + (K-1) 个 MASK`。anchor 是已经确定的最后一个 token；MASK 是待预测位置的占位符，不是未来答案。设 anchor 的绝对位置为 $a$，则可读取的主模型历史为 $[0,a)$，当前块占据 $a$ 到 $a+K-1$。本实现常用 $K=32$，即一个 anchor 和 31 个候选位置；K 是配置，不是算法常数。

当前混合层策略记为 `orthrus_qwen35_reference`：**Full Attention 在当前块内双向计算，GDN 则从 AR 历史状态出发，保持因果递推。** 这是依据 Orthrus Qwen3.5 参考实现采用的模型扩展；原始普通 Qwen3 算法没有 GDN，不能把“所有层都改成双向 SDPA”作为这里的默认定义。训练 head 必须与这一策略匹配。

### 1.2 现有推理路径的适配边界

PyPTO 已有 Qwen3.8 整网前向的实现基础。本文固定参考 `pypto-lib` PR #1503 的提交 `1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9`，模块路径均相对于 `models/qwen3_8_27b/`。该 PR 及配套 `pypto-serving` PR #281 尚未合入，公开验证范围为单卡、单请求、最多 128 个位置，不包含投机解码，不能等同于已经支持本报告的执行方式。

**核心增量是两个 mixer 入口，不是固定新增若干个底层 kernel。** Full Attention 要从单个 AR query 扩展为整块 query，并改变可见范围；GDN 要从带历史初值的单步，扩展为 K 个位置连续推进，同时隔离状态写入。投影、norm、RoPE、MLP 等公式不因 Orthrus 改变，但仍要接好独立 draft 权重和 TP 分片，不能只替换一个 mask 就完成整网接入。

下面按两条计算路径依次说明：先定义计算，再对照 PyPTO 现有入口，最后给出当前 Torch-NPU 实现。

## 2. Attention 模块设计

### 2.1 Full Attention：历史前缀与双向块

#### 2.1.1 可见性变化与计算定义

普通 AR 层中，第 $i$ 个位置只能读取历史及当前块的前 $i+1$ 个位置；draft 的块内输入没有未来答案，因此每行可以读取整个临时块。主模型历史保持原样，不重新做双向计算。

按单个 query head 书写，令 $K_H,V_H$ 为主模型历史，$q_i^{AR}$ 和 $K_{AR,0:i},V_{AR,0:i}$ 为当前 AR 块的 query 和因果前缀，则原有计算为：

$$
o_i^{AR}=\mathrm{softmax}\left(q_i^{AR}[K_H;K_{AR,0:i}]^\top/\sqrt d\right)[V_H;V_{AR,0:i}].
$$

$0:i$ 包含位置 $i$，分号表示沿 token 维拼接。Orthrus 由本层 draft 参数产生 $Q_D,K_D,V_D$，计算改为：

$$
O_D=\mathrm{softmax}\left(Q_D[K_H;K_D]^\top/\sqrt d\right)[V_H;V_D].
$$

两者仍是同一个 SDPA 公式，变化是**所有 draft query 都读取同一历史前缀及整个临时块**。这里没有 DSV4 的 sink、压缩 KV 或 indexer。

该模型使用 GQA：24 个 query head 共享 4 个 KV head，$d=256$。Q/K 先做 RMSNorm，仅前 64 维做 RoPE；Attention 输出乘以 query 投影产生的 sigmoid gate，再经过 O projection。Q/K norm 的缩放按 `1 + weight` 解释，这些前后处理也属于模型语义，不能只对齐 softmax 而省略它们。

#### 2.1.2 PyPTO 实现基础与适配方案

现有 `attention_state.py` 已有 Q/K norm、部分 RoPE、GQA 和输出门控。`full_network_core.py::attention_context_full` 从连续 KV 读取历史，但只计算第一个有效 query，并按当前位置限制可见范围；`append_kv_full` 则负责追加主模型 K/V。

Orthrus 需要保留前后处理，将 Attention 入口扩展为 K 行 query，另行传入临时块 K/V。历史长度固定为 $a$，当前块不加因果限制，且不调用 AR 的 KV append。这里“只读”限制的是历史输入；算子仍会写自己的输出和临时缓冲。draft K/V 不会成为正式历史，即使对应候选最终被接受，也由主模型重新计算其正式 KV。

首版可接连续历史，与下述 Torch-NPU 入口对照；若接入主模型分页缓存，再根据页表读取。历史和临时块既可放在一次 Attention 中计算，也可分别计算后按 LSE 合并，数学定义不要求拆成两个 kernel。

#### 2.1.3 Torch-NPU 实现

现有实现先用 draft 投影产生 Q/K/V，并复用镜像中的 `triton_split_qkv_rmsnorm_mrope` 完成拆分、Q/K norm 和部分 RoPE。随后在本轮私有的 `_ScratchCache` 中拼接历史与临时 KV，调用融合 Attention：

```python
# query/key/value 已完成投影及位置处理。
# update 只更新 draft 的临时对象，不写主模型 KV。
key, value = past_key_values.update(key, value, self.layer_idx)
out = torch_npu.npu_fusion_attention(
    query, key.contiguous(), value.contiguous(),
    query.shape[1], input_layout="BNSD", atten_mask=blocked,
    scale=self.scaling, keep_prob=1., sparse_mode=0,
)[0]
out = out.transpose(1, 2).reshape(batch, length, -1)
out = out * gate.reshape(batch, length, -1).sigmoid()
return self.o_proj(out), None
```

布局为 `[batch, head, token, dim]`，融合算子直接接收较少的 KV head，不需要复制整段历史来展开 GQA。当前单请求、无 padding 的 draft 中，所有传入 KV 都可见；若有 padding，则显式屏蔽无效位置。Torch 的布尔 SDPA mask 用 `True` 表示可见，而这里的 FAS mask 用 `True` 表示禁止读取，适配层负责转换。

主模型分页 KV 当前通过页表收集为独立连续快照，并非零拷贝分页直读。快照之后的拼接只发生在 draft 私有对象中。以下折叠源码给出实际 Attention 函数和历史读取函数，依赖本仓库的模型对象及运行环境。

<details>
<summary>完整函数：Full Attention 与分页历史读取</summary>

```python
def full_attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    if (hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] != 1 or kwargs
            or self.config._attn_implementation != "sdpa"):
        return self._reference_forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
    import torch_npu

    batch, length, _ = hidden_states.shape
    shape = (batch, length, -1, self.head_dim)
    projected_q, projected_k, projected_v = (projection(hidden_states)
        for projection in (self.q_proj, self.k_proj, self.v_proj))
    query, gate = projected_q.reshape(batch, length, -1, 2 * self.head_dim).chunk(2, -1)
    query, key = query.reshape(shape).transpose(1, 2), projected_k.reshape(shape).transpose(1, 2)
    value = projected_v.reshape(shape).transpose(1, 2)
    native_rope = self._native_rope
    if native_rope is not None:
        row = None if past_key_values is None else past_key_values.layers[self.layer_idx].keys
        past = 0 if row is None else row.shape[2]
        positions = native_rope.positions(past, length, hidden_states.device)

    if native_rope is not None and not torch.is_grad_enabled():
        packed = torch.cat((projected_q, projected_k, projected_v), dim=-1).reshape(length, -1)
        cos_sin = native_rope.cache.index_select(0, positions).to(packed.dtype)[None].expand(3, -1, -1).contiguous()
        q, k, v, raw_gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=packed, q_weight=1 + self.q_norm.weight, k_weight=1 + self.k_norm.weight,
            cos_sin=cos_sin, num_q_heads=query.shape[1], num_kv_heads=key.shape[1], head_size=self.head_dim,
            eps=self.config.rms_norm_eps, mrope_section=list(native_rope.sections),
            is_interleaved=native_rope.interleaved, rope_dim=native_rope.dim, has_gate=True)
        query = q.reshape(shape).transpose(1, 2)
        key, value = (part.reshape(shape).transpose(1, 2) for part in (k, v))
        gate = raw_gate.reshape(batch, length, -1)
    else:
        cos, sin = (native_rope.coefficients(positions, query.dtype) if native_rope is not None
                    else tuple(component[:, None].contiguous() for component in position_embeddings))
        rotary_dim = cos.shape[-1]

        def normalize_rotate(value, weight):
            full = value.float()
            # 原生融合分支先形成 BF16 gamma，但 norm 与 RoPE 之间不舍入。
            normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
            normalized = normalized * (1 + weight).float()
            rotated = torch_npu.npu_rotary_mul(normalized[..., :rotary_dim].contiguous(), cos.float(), sin.float())
            return torch.cat((rotated, normalized[..., rotary_dim:]), dim=-1).to(value.dtype).contiguous()

        query = normalize_rotate(query, self.q_norm.weight)
        key = normalize_rotate(key, self.k_norm.weight)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)
    # FAS 直接支持 GQA；保留少量 KV 头，避免复制整段历史。
    # PyTorch 加性 mask 中的 -inf 对应 Ascend FAS 的 True（禁止读取）。
    blocked = (None if attention_mask is None else (~attention_mask if attention_mask.dtype == torch.bool
               else torch.isneginf(attention_mask)).contiguous())
    out = torch_npu.npu_fusion_attention(query, key.contiguous(), value.contiguous(),
        query.shape[1], input_layout="BNSD", atten_mask=blocked, scale=self.scaling,
        keep_prob=1., sparse_mode=0)[0]
    out = out.transpose(1, 2).reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
    return self.o_proj(out), None


def paged_history(cache, block_table, length):
    """Native [page, token, KV head, dim] -> owned [1, KV head, token, dim]."""
    if (cache.ndim != 4 or block_table.ndim != 2 or block_table.shape[0] != 1
            or length < 1):
        raise ValueError("batch-one native paged KV and a positive prefix required")
    pages = (length + cache.shape[1] - 1) // cache.shape[1]
    if pages > block_table.shape[1]:
        raise ValueError("committed prefix exceeds native block table")
    indices = block_table[0, :pages].to(device=cache.device, dtype=torch.long)
    return cache.index_select(0, indices).flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0).contiguous()
```

</details>

### 2.2 GDN：带历史初值的块递推

#### 2.2.1 原有递推与 draft 状态

GDN 不计算 softmax Attention，而是先做宽度为 4 的因果逐通道卷积，再用门控 Delta Rule 推进一个矩阵状态。Q/K 各有 16 个 head，V 有 48 个 head，每个 head 维度为 128；每组三个 V head 共享一组 Q/K。

省略 batch 和 head 下标，令 $q_i,k_i$ 为 L2 归一化后的列向量，$v_i$ 为 value，$S_i$ 为处理完位置 $i$ 后的矩阵状态。门控和衰减为：

$$
\beta_i=\sigma(b_i),\qquad g_i=-\exp(A_{\log})\,\mathrm{softplus}(u_i+\Delta t).
$$

$b_i,u_i$ 来自本层投影，$\sigma$ 为 sigmoid，$A_{\log},\Delta t$ 为每个 V head 的参数；$g_i$ 是该 head 的标量衰减。状态采用原生 `[V,K]` 布局，即 $S\in\mathbb R^{d_v\times d_k}$，单步计算为：

$$
\overline S_i=e^{g_i}S_{i-1},\qquad \delta_i=\beta_i(v_i-\overline S_i k_i).
$$

$$
S_i=\overline S_i+\delta_i k_i^\top,\qquad o_i=S_iq_i/\sqrt{d_k}.
$$

输出再经过本层 gated RMSNorm、`SiLU(z)` 门控和 O projection。该 norm 使用自己的 weight，不采用 Full Attention 的 `1 + weight` 解释。

**AR 与 draft 使用相同递推公式，区别在参数和状态的归属。** AR 从上一段已接受的正式状态继续，用主模型参数计算；draft 以 $S_{-1}^{D}=S_{a-1}^{AR}$ 为初态副本，用 draft 参数在工作区中计算 K 行，结束后丢弃临时状态。卷积也要带入历史尾部的三个原始投影输入，而不是从零开始，或读取卷积后的结果。

#### 2.2.2 PyPTO 实现基础与适配方案

现有 `delta_state.py` 已实现 Qwen GDN 的卷积、head 分组、标量衰减、Delta Rule 和门控 norm，不需要用 GLM KDA 代替。`decode_layer.py::delta_decode_into` 接受 `previous_conv` 和 `previous_recurrent`，但一次只处理一个有效 token。

所需变化是提供**非零历史初值下的块入口**：按顺序处理 K 行卷积输入，把每步递推状态传给下一步，并返回每个位置的输出。首版可以编排已有单步计算，再评估分块融合；不能把同一序列的 K 个位置当成 K 条互不相关的 batch。现有入口里的 16 行 padding tile 不等于已经支持 16 或 32 个有效位置，短 prefill 中的零历史假设也不能直接沿用。

状态布局必须显式约定：原生 NPU cache 为 `[V,K]`，HF 与所参考 PyPTO 入口为 `[K,V]`，需要转置。两维恰好都是 128，仅核对 shape 无法发现转置错误。draft 只需要自己的临时推进状态；如果主模型验证也迁入 PyPTO，则还需能恢复任意接受前缀的状态，见第 3 节。

#### 2.2.3 Torch-NPU 实现

当前路径复用现成的因果卷积、L2 norm、GDN 递推和门控 norm。进入递推前，适配层先将 AR 初态复制进自有工作区，再调用镜像已有算子：

```python
length = query.shape[1]
state = torch.empty(
    (length, value.shape[2], 128, 128),
    device=query.device, dtype=torch.float32,
)
# HF 初态为 [K,V]，原生工作区为 [V,K]；后者不与 AR cache 共用存储。
state[:1].copy_(initial.transpose(-1, -2))
operation = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
operation = getattr(operation, "__wrapped__", operation)
out = recurrent_slices(
    operation, query=query[0, :, ::groups], key=key[0, :, ::groups],
    value=value[0], g=log_decay[0].float(), beta=beta[0], state=state,
    actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device=query.device),
    ssm_state_indices=torch.arange(length, dtype=torch.int32, device=query.device),
    num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=query.device),
    scale=128 ** -.5,
)
```

这里省略了空历史分支，完整函数见下方。`groups` 是 V head 与 Q/K head 的比例；HF 输入已展开分组，传给原生接口时恢复为少量 Q/K head。当前 A2 递推接口每次最多处理 16 行，`recurrent_slices` 将 K32 切为两段，后段继续使用前段末状态；这不是两次整网起草，也不改变块内的因果关系。

卷积同样只使用私有状态，按顺序衔接历史尾部与当前块。GDN 输出经现成门控 norm 返回本层残差路径。以下源码保留实际调用和边界处理，供 PyPTO 对照输入、状态布局及分段语义，而不是要求逐句照搬 Torch 编排。

<details>
<summary>完整函数：因果卷积、GDN 递推分段和门控 norm</summary>

```python
def causal_conv(x, weight, bias=None, activation="silu", seq_idx=None):
    if seq_idx is not None or activation not in {"silu", "swish"}:
        raise ValueError("unpacked SiLU causal convolution required")
    if (x.device.type != "npu" or x.dtype != torch.bfloat16 or torch.is_grad_enabled()
            or bias is not None or weight.shape[-1] != 4 or x.shape[0] != 1):
        if torch.is_inference(weight) and torch.is_grad_enabled() and x.requires_grad:
            weight = weight.clone()
        out = F.conv1d(x.float(), weight[:, None].float(), None if bias is None else bias.float(),
                       padding=weight.shape[-1] - 1, groups=weight.shape[0])[..., :x.shape[-1]]
        return F.silu(out).to(x.dtype)
    # 处理完整输入（可能带有 HF 拼接的历史）；临时卷积状态绝不指向主模型。
    raw = x[0].T.contiguous()
    state = x.new_zeros((1, 34, x.shape[1]))
    tail = x.new_zeros((3, x.shape[1]))
    result = []
    for start in range(0, raw.shape[0], 32):
        part = raw[start:start + 32]
        state[0, :3] = tail
        out = torch.empty_like(part)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out, part, weight.T, conv_state=state, bias_opt=None,
            query_start_loc_opt=torch.tensor([0, part.shape[0]], dtype=torch.int32, device=x.device),
            cache_indices_opt=torch.zeros((1, part.shape[0]), dtype=torch.int32, device=x.device),
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=torch.ones(1, dtype=torch.int32, device=x.device),
            activation_mode=1, pad_slot_id=-1, run_mode=1)
        result.append(out)
        tail = torch.cat((tail, part), dim=0)[-3:]
    return torch.cat(result).T[None]


def delta_rule(query, key, value, *, fallback, groups, training_fallback=None, **kwargs):
    normalize = kwargs.pop("use_qk_l2norm_in_kernel", False)
    if normalize:
        query, key = l2norm(query), l2norm(key)
    native = (query.device.type == "npu" and query.dtype == torch.bfloat16
              and query.shape[0] == 1 and 1 <= query.shape[1] <= 32
              and query.shape[-1] == value.shape[-1] == 128 and not torch.is_grad_enabled()
              and kwargs.get("cu_seqlens") is None
              and not set(kwargs) - {"g", "beta", "initial_state", "output_final_state", "cu_seqlens"})
    if not native:
        if (training_fallback is not None and torch.is_grad_enabled()
                and query.device.type == "npu" and query.dtype == torch.bfloat16):
            return training_fallback(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
        if query.device.type == "npu" and query.shape[1] < 64 and "chunk_size" not in kwargs:
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
            if fallback is torch_chunk_gated_delta_rule:
                # Keep the existing chunk algorithm, without padding short draft blocks to 64.
                kwargs["chunk_size"] = 1 << (max(1, query.shape[1]) - 1).bit_length()
        return fallback(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
    from .qwen38_gdn_backend import recurrent_slices
    length = query.shape[1]
    state = torch.empty((length, value.shape[2], 128, 128), device=query.device, dtype=torch.float32)
    initial = kwargs.get("initial_state")
    # 递推算子写出每行 checkpoint；仅首行需要历史初态，其他行不预先清零。
    if initial is None:
        state[:1].zero_()
    else:
        state[:1].copy_(initial.transpose(-1, -2))
    # HF 已经展开了 GQA 头；原生递推可直接接收少量 Q/K 头。
    operation = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    operation = getattr(operation, "__wrapped__", operation)
    out = recurrent_slices(operation, query=query[0, :, ::groups],
        key=key[0, :, ::groups], value=value[0],
        g=kwargs["g"][0].float(), beta=kwargs["beta"][0], state=state,
        actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32, device=query.device),
        ssm_state_indices=torch.arange(length, dtype=torch.int32, device=query.device),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=query.device), scale=128 ** -.5)
    final = state[-1:].transpose(-1, -2).contiguous() if kwargs.get("output_final_state", False) else None
    return out[None], final


def recurrent_slices(operation, *, query, key, value, state, actual_seq_lengths,
                     ssm_state_indices, num_accepted_tokens, g, beta, scale, limit=16):
    """Keep every verifier state while passing at most 16 rows per kernel call."""
    import torch

    length = query.shape[0]
    if actual_seq_lengths.numel() != 2 or ssm_state_indices.numel() < length or length < 1:
        raise ValueError("synchronous batch-one recurrent verification required")
    # 上轮只接受到哪个位置，就从那个位置的状态开始；先复制，避免被本轮覆盖。
    selected = ssm_state_indices[num_accepted_tokens.long() - 1].long()
    initial = state.index_select(0, selected)
    # 整块只整理一次布局；沿 token 维切片不会再次破坏连续性。
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    g = None if g is None else g.contiguous()
    beta = None if beta is None else beta.contiguous()
    outputs = []
    for start in range(0, length, limit):
        end = min(start + limit, length)
        slots = ssm_state_indices[start:end].contiguous()
        state.index_copy_(0, slots[:1].long(), initial)
        outputs.append(operation(
            query=query[start:end], key=key[start:end], value=value[start:end], state=state,
            actual_seq_lengths=actual_seq_lengths.new_tensor([0, end - start]),
            ssm_state_indices=slots, num_accepted_tokens=None,
            g=None if g is None else g[start:end],
            beta=None if beta is None else beta[start:end], scale=scale))
        if end < length:
            initial = state.index_select(0, slots[-1:].long())
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class GatedRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.variance_epsilon = weight, eps

    def forward(self, value, gate):
        weight = self.weight
        if value.device.type == "npu" and not torch.is_grad_enabled():
            from vllm_ascend.ops.layernorm import layer_norm_fwd_npu
            return layer_norm_fwd_npu(value, weight, None, self.variance_epsilon, z=gate,
                group_size=None, norm_before_gate=True, is_rms_norm=True, activation="swish")[0]
        if torch.is_inference(weight) and torch.is_grad_enabled() and (value.requires_grad or gate.requires_grad):
            weight = weight.clone()
        full = value.float()
        if value.device.type == "npu":
            import torch_npu
            normalized = torch_npu.npu_rms_norm(full, weight.float(), self.variance_epsilon)[0]
            return (normalized * F.silu(gate.float())).to(value.dtype)
        normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (normalized * weight.float() * F.silu(gate.float())).to(value.dtype)
```

</details>

### 2.3 开发范围汇总

| 计算路径 | PyPTO 已有基础 | 需要适配的部分 |
| --- | --- | --- |
| Full Attention | Q/K norm、部分 RoPE、GQA、输出门控，以及连续历史的单 query Attention。 | K 行 draft query 读取相同历史及整个临时块；不执行 AR KV append；扩展有效长度和容量，按集成方式接连续或分页历史。 |
| GDN | 带历史的单步卷积和 Delta Rule、head 分组、门控 norm。 | 非零初态下连续处理 K 行，卷积和递推写入私有工作区；对齐状态布局，不把块内位置作为独立 batch。 |

这是两条模型计算路径的需求划分，不规定最终 kernel 数量。norm、投影等已有组件可以按实际性能选择融合；共享 MLP 和 TP 通信属于完整层执行，不应另列成 Orthrus 新 Attention 算法。

## 3. 模型执行与推理引擎接入

### 3.1 一轮投机如何执行

完整 draft 并不是只调用一次 Attention。执行器要让当前块依次通过 64 层：`共享 input norm -> draft mixer -> 残差 -> 共享 post norm / MLP -> 残差`，最后经共享 Final Norm 和 LM head 产生候选分布。共享的是主模型权重，不是本轮中间结果，因此 MLP 等仍需对 draft 输入重新计算，但不必复制第二套 backbone。

```text
已接受的主模型历史
    |
    +-> 分页 KV 快照 / GDN 卷积尾部和递推初态
    |              |
    |       anchor + MASK 块 -> 64 层 draft -> 候选及其概率
    |                                             |
    +----------------------------------------> 原生 AR 整块验证
                                                   |
                                    接受前缀 -> 保留 AR 对应历史
                                    拒绝后缀 -> 回收 KV / 恢复 GDN
```

候选由主模型验证后，正式历史来自这次 **AR 计算**，不是把 draft KV 或 GDN 状态转正。Full Attention 由引擎控制接受前缀的 KV 可见长度；GDN 则必须恢复到接受位置的递推状态和卷积尾部，不能把块末状态简单“截短”。现有实现使用逐位置 checkpoint，另一种可行方式是重新计算接受前缀。

### 3.2 PyPTO 与引擎的接口

若仅将 draft mixer 交给 PyPTO，vLLM-Ascend 可以继续负责主模型、共享 MLP、TP 通信和原生验证采样。调用方需传入 draft 参数、历史边界、KV 或 GDN 初态，以及独立输出和工作区；同时约定张量布局、设备地址、有效长度、stream 同步和缓冲生命周期。两个运行时都在 NPU 上，不代表缓存指针可以未经约定直接互用。

若将整段 draft 迁入 PyPTO，再接齐共享权重调用、残差、MLP 和跨卡通信即可；若连主模型也迁入 PyPTO，还要提供上一节的块验证和接受前缀恢复。缓存分配、页表与接受决策由引擎管理，算子只按输入页表或状态地址执行读取与写入，二者不能混为一个新增 kernel。

参考 PR 是单卡完整规格，而现有 Torch-NPU 接入支持 TP 分片。以 TP2 为例，每 rank 的 Full Attention 为 12 个 Q head、2 个 KV head，GDN 为 8 个 Q/K head、24 个 V head；TP4 分别为 6/1 和 4/12。PyPTO 入口须参数化这些尺寸并匹配分片权重，不能直接沿用单卡常量。head 的基模、K、mask token、混合层策略和 TP 规格也应一致。

### 3.3 可运行实现与验证范围

本仓库已有完整 Torch-NPU 执行器和 vLLM-Ascend 接入，不仅是以上两个计算片段。下列命令在固定运行环境中加载真实模型，执行起草、验证和缓存检查；未提供 head 时从主模型 mixer 复制初始化，只用于功能验证。

```bash
cd online-speculation
python scripts/run_in_runtime.py scripts/validate_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

模型路径按实际挂载修改；使用训练 head 时指定 `--head-dir`，mask token 和 TP 规格以该 head 的配置为准。这里的 mask ID 仅是初始化功能用例配置，不是所有 head 的通用取值。

现有记录覆盖真实 27B 的 TP2 起草与原生验证、K=2/8/17/32、全接受及连续拒绝后的状态恢复、分页历史读取、缓存隔离、head 导出重载和采样校准；单算子还对照了双向 Full Attention、非零初态 GDN、分段边界和相关前后处理。另有 TP4 加载及 draft 更新记录，但不把训练反向列入本报告的 PyPTO 开发范围。

这些结果可作为 PyPTO 逐模块和整网接入的参照，**不是 PyPTO Orthrus 已验证的证据**。原生不同计算形状存在 K32 数值分叉，严格逐 token 一致性没有全部通过；初始化 head 也不代表已获得业务接受率或加速收益。当前推理以 BF16、eager、同步单请求为验证范围，不据此推断量化、图模式或动态 batch 已完成适配。

开发顺序可沿用上述模块边界：先对照连续历史的 Full Attention 和非零初态 GDN，再接整层及真实共享历史，最后验证各接受位置恢复、TP 分片和长上下文；分页直读与更大融合在这一基础上按测量结果推进。

## 参考资料

- PyPTO 实现范围与固定版本：[pypto-lib PR #1503](https://github.com/hw-native-sys/pypto-lib/pull/1503)、[pypto-serving PR #281](https://github.com/hw-native-sys/pypto-serving/pull/281)。
- 混合层算法参考：[Orthrus Qwen3.5 实现](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py)。
- 本仓库记录：[运行与配置](online-speculation/docs/QWEN38_VLLM.md)、[验收摘要](online-speculation/docs/QWEN38_ACCEPTANCE_SUMMARY.md)、[详细验证](online-speculation/docs/QWEN38_VLLM_VALIDATION.md)、[推理算子审计](online-speculation/docs/QWEN38_INFERENCE_OPS_AUDIT.md)。
