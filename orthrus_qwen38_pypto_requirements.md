# Orthrus-Qwen3.8 NPU 推理适配与算子设计

## 1. 背景与适配目标

大模型的自回归推理通常逐个生成 token：当前 token 确定后，才能继续计算下一个位置。投机解码先由草稿分支提出多个候选，再由主模型一次验证，尝试减少主模型逐 token 执行的轮数。实际收益既取决于候选接受率，也取决于起草、验证和状态管理的额外开销。

Orthrus 采用共享主干的草稿结构，在模型各层增加独立的 draft 模块，复用主模型的 embedding、外层 norm、MLP 和 LM head，并读取主模型已经生成的历史。用于 Qwen3.8-27B 时，草稿分支分别适配 Full Attention 和 GDN：前者在当前块内做双向 Attention，后者从历史状态的副本继续因果递推。

下文用 AR 表示主模型的自回归分支，draft 表示预测候选的分支。这里只讨论文本推理，不包含训练和反向算子。

### 1.1 草稿分支与主模型的关系

Qwen3.8-27B 的模型主干共有 64 层：每三个 GDN 层后接一个 Full Attention 层，重复 16 次，共 48 层 GDN、16 层 Full Attention。Full Attention 读取历史 token 的 KV，GDN 则通过卷积和矩阵递推累积历史信息。因此，草稿分支在这两类层中使用不同的方式接上主模型历史。

Orthrus 为每层的 Full Attention 或 GDN 另外保留一套独立的 draft 参数。起草时使用这套参数，embedding、外层 norm、MLP 和 LM head 仍与主模型共享：

```text
共享 Embedding
    |
    v
[ GDN -> GDN -> GDN -> Full Attention ] x 16
    | 每层：共享 input norm -> Full Attention / GDN -> 残差
    |                       -> 共享 post norm / MLP -> 残差
    v
共享 Final Norm / LM head

每层的 Full Attention 或 GDN 有两套参数：
  AR 参数    ：原模型参数，产生正式历史
  Draft 参数 ：独立参数，读取 AR 历史，在临时工作区计算候选
```

draft 每轮输入一个块：`anchor + (K-1) 个 MASK`。anchor 是已经确定的最后一个 token，MASK 占据接下来要预测的位置，输入中没有未来答案。设 anchor 位于 `a`，它之前的 `[0,a)` 直接读取主模型历史；anchor 和 MASK 则占据 `a` 到 `a+K-1`，在本轮一起计算。常用的 `K=32` 就是一个 anchor 加 31 个候选位置，K 本身可以配置。

参考 Orthrus 的 Qwen3.5 混合模型实现，Qwen3.8 的草稿分支采用以下计算方式：在 Full Attention 层，块内位置可以互相读取信息；在 GDN 层，当前实现仍从 AR 历史状态出发，按顺序递推。代码中将这一配置记为 `orthrus_qwen35_reference`。原始普通 Qwen3 没有 GDN，因此不能直接把它的双向 SDPA 套到所有层。加载训练 head 时，也要确认它采用同样的混合层策略。

### 1.2 PyPTO 已有基础与本次改动范围

本文的算子设计基线为 `pypto-lib` PR #1503 的提交 `1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9`，下文该基线的文件均位于 `models/qwen3_8_27b/`。可运行接入采用固定的 OneServing `7aa40ba` 交付包及配套 PyPTO，支持单卡、单活动请求、512 个总位置；环境和运行入口见 [PyPTO Serving 接入](online-speculation/docs/QWEN38_PYPTO_PORT.md)。这两份基线的接口不同，下面的计算契约相同。

**Orthrus 主要扩展 Full Attention 和 GDN 的调用方式**。Full Attention 一次计算 K 行 query，每行读取同一段历史和整个草稿块；GDN 则先复制主模型的历史状态，再用这份数据依次计算 K 个位置，不修改主模型原来的状态。这两条路径已在可运行接入中实现；以下解释计算要求，最后说明整网验证范围。

投影、norm、RoPE 和 MLP 的公式可以沿用。接入时仍要给它们绑定正确的 draft 或共享权重，按 TP 规格分片，再与主模型交换历史。下面先说明这两类模块内部要改什么，第 3 节再把它们放回完整推理流程。

## 2. Attention 模块设计

### 2.1 Full Attention：历史前缀与双向块

#### 2.1.1 计算方式与可见范围

先看普通 AR 层。它按从左到右的顺序预测，第 `i` 个位置只能读取历史和当前块的前 `i+1` 个位置，避免看到未来答案。draft 输入的是 anchor 和 MASK，没有未来答案可泄露，因此每行可以读取整个当前块。双向只发生在这个块里，之前的主模型历史保持不变。

按单个 query head 书写，令 `K_H`、`V_H` 为主模型历史，`q_i^AR` 为当前 AR query，`K_AR,0:i`、`V_AR,0:i` 为当前块截至位置 `i` 的 K/V，则原有计算为：

$$
o_i^{AR}=\mathrm{softmax}\left(q_i^{AR}[K_H;K_{AR,0:i}]^\top/\sqrt d\right)[V_H;V_{AR,0:i}].
$$

`0:i` 包含位置 `i`，分号表示沿 token 维拼接。Orthrus 由本层 draft 参数产生 `Q_D`、`K_D`、`V_D`，计算改为：

$$
O_D=\mathrm{softmax}\left(Q_D[K_H;K_D]^\top/\sqrt d\right)[V_H;V_D].
$$

两式的 Attention 公式不变，区别在于可见范围：**每个 draft query 都能读取同一段历史和整个草稿块**。

Q/K 归一化、部分 RoPE、GQA 及输出处理沿用 Qwen 原有实现。

#### 2.1.2 PyPTO 已有实现与所需修改

设计基线中的 `attention_state.py` 已包含上述 Q/K norm、部分 RoPE、GQA 和门控。其历史 Attention 位于 `full_network_core.py::attention_context_full`，从连续 KV 中读取数据，只计算第一个有效 query，并按当前位置限制可见范围；新 K/V 由另一个入口 `append_kv_full` 追加。

因此可以保留前后处理，重点扩展 Attention 入口：一次接收 K 行 query，并额外接收临时块 K/V。所有行的历史长度都设为 `a`，临时块全部可见。由于这些 K/V 由 draft 参数计算，不是主模型的正式历史，起草阶段不调用 AR 的 KV append，只写自己的输出和工作区。候选接受后，正式 KV 由主模型验证时重新计算。

可运行接入保留连续历史对照，并提供 `paged` 实现：按页表将历史和整个有效草稿块直接搬入 L1，融合 QK、softmax、PV 和门控，不生成连续 KV 副本。按有效长度选择 128/256/512 容量，Torch 数学参考保持独立；详见 [kernel 接口](online-speculation/docs/QWEN38_PYPTO_KERNEL_TASK.md)。

#### 2.1.3 Torch-NPU 实现

现有 Torch-NPU 版本就是按上述顺序执行：先用 draft 权重投影，再调用`triton_split_qkv_rmsnorm_mrope` 完成QKV拆分、Q/K norm 和部分 RoPE，再在临时缓存中拼接历史与草稿 KV，调用融合 Attention：

以下摘取当前 BF16、单请求、无 padding、无梯度的融合推理分支。`self` 为 draft Attention，`past_key_values` 为已装入历史 KV 快照的本轮 `_ScratchCache`；输入检查、mask 转换和其他执行分支仍保留在后面的完整函数中。

```python
# draft 投影；下面只展示单请求、无 padding 的融合推理分支。
batch, length, _ = hidden_states.shape
shape = (batch, length, -1, self.head_dim)
projected_q, projected_k, projected_v = (projection(hidden_states)
    for projection in (self.q_proj, self.k_proj, self.v_proj))
num_q_heads = projected_q.shape[-1] // (2 * self.head_dim)
num_kv_heads = projected_k.shape[-1] // self.head_dim

# 按历史长度准备本轮位置编码，复用主模型的只读 RoPE 系数表。
native_rope = self._native_rope
history_key = past_key_values.layers[self.layer_idx].keys
past = 0 if history_key is None else history_key.shape[2]
positions = native_rope.positions(past, length, hidden_states.device)
packed = torch.cat((projected_q, projected_k, projected_v), dim=-1).reshape(length, -1)
cos_sin = native_rope.cache.index_select(0, positions).to(packed.dtype)
cos_sin = cos_sin[None].expand(3, -1, -1).contiguous()

# 现成融合算子：QKV 拆分、Q/K norm、部分 RoPE，并拆出输出门控值。
q, k, v, raw_gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
    qkv=packed, q_weight=1 + self.q_norm.weight, k_weight=1 + self.k_norm.weight,
    cos_sin=cos_sin, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
    head_size=self.head_dim, eps=self.config.rms_norm_eps,
    mrope_section=list(native_rope.sections),
    is_interleaved=native_rope.interleaved, rope_dim=native_rope.dim, has_gate=True,
)
query, key, value = (part.reshape(shape).transpose(1, 2) for part in (q, k, v))

# ORTHRUS：在本轮 _ScratchCache 中拼接历史和草稿 KV，不写主模型缓存。
key, value = past_key_values.update(key, value, self.layer_idx)

# 历史与整个草稿块全部可见；原生 GQA 不需要复制 KV head。
out = torch_npu.npu_fusion_attention(
    query, key.contiguous(), value.contiguous(), num_q_heads,
    input_layout="BNSD", atten_mask=None, scale=self.scaling,
    keep_prob=1., sparse_mode=0,
)[0]
out = out.transpose(1, 2).reshape(batch, length, -1)
out = out * raw_gate.reshape(batch, length, -1).sigmoid()
return self.o_proj(out), None
```

历史 KV 先按页表收集为连续张量，再与本轮草稿 KV 拼接，以 `[batch, head, token, dim]` 布局送入融合 Attention。算子原生支持 GQA，无需展开复制 KV head。

当前按单请求、无 padding 场景执行，历史和整个草稿块均可见，调用时不设置 mask。完整 Attention 和历史读取函数见下方折叠代码。

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

#### 2.2.1 从历史状态继续计算当前块

Full Attention 通过读取历史 KV 获取信息，GDN 则用卷积状态和递推矩阵保留历史。处理当前块时，输入先经过宽度为 4 的因果逐通道卷积，再通过门控 Delta Rule 更新矩阵状态。

该模型的 GDN 中，Q/K 各有 16 个 head，V 有 48 个 head，每个 head 的维度为 128，每三个 V head 共用一组 Q/K。

省略 batch 和 head 下标，令 `q_i`、`k_i` 为 L2 归一化后的列向量，`v_i` 为 value，`S_i` 为处理完位置 `i` 后的矩阵状态。门控和衰减为：

$$
\beta_i=\sigma(b_i),\qquad g_i=-\exp(A_{\log})\,\mathrm{softplus}(u_i+\Delta t).
$$

`b_i`、`u_i` 来自本层投影，`σ` 为 sigmoid，`A_log`、`Δt` 为每个 V head 的参数；`g_i` 是该 head 的标量衰减。状态采用原生 `[V,K]` 布局，也就是 `d_v × d_k` 矩阵，单步计算为：

$$
\overline S_i=e^{g_i}S_{i-1},\qquad \delta_i=\beta_i(v_i-\overline S_i k_i).
$$

$$
S_i=\overline S_i+\delta_i k_i^\top,\qquad o_i=S_iq_i/\sqrt{d_k}.
$$

算出 `o_i` 后，再做本层 gated RMSNorm、`SiLU(z)` 门控和 O 投影。这里 norm 直接使用自己的 weight，与 Full Attention 中的 `1 + weight` 不同。

这套公式在 AR 和 draft 中相同，**需要改变的是用哪套参数、把状态写到哪里**。AR 用主模型参数，从上一段正式状态继续；draft 则复制 anchor 之前的状态，用独立参数在工作区中处理 K 行，矩阵初态为：

$$
S_{-1}^{D}=S_{a-1}^{AR}.
$$

卷积也要带入历史最后三个原始投影输入，这样块的第一个位置才能接上此前的序列。它需要的是卷积前输入，不是卷积输出。临时状态在起草结束后丢弃。

#### 2.2.2 PyPTO 已有实现与所需修改

设计基线中的 `delta_state.py` 已实现这里的卷积、head 分组、标量衰减、Delta Rule 和门控 norm，可以作为 Qwen GDN 的基础。`decode_layer.py::delta_decode_into` 也接受 `previous_conv` 和 `previous_recurrent`，能够接上历史，不过一次只处理一个有效 token。

因此要把单步入口扩展为整块入口：从已有历史初态出发，按顺序处理 K 行，把每步状态交给下一步，并返回全部位置的输出。可以先逐步调用已有计算，再考虑分块融合。同一请求的这些位置有先后依赖，不能作为独立 batch 同时更新一个状态。现有 16 行 padding tile 只说明缓冲的补齐尺寸，不代表能处理同样数量的有效位置；短 prefill 的零历史假设也需要改为本轮传入的历史初态。

传递状态时还有一个容易忽略的差别：原生 NPU cache 为 `[V,K]`，HF 和所参考 PyPTO 入口为 `[K,V]`，两者需要转置。因为两维恰好都是 128，shape 相同也可能把矩阵方向用反。draft 只在副本上继续递推；主模型验证后的状态恢复是另一件事，放在第 3 节说明。

#### 2.2.3 Torch-NPU 实现

现有 NPU 路径复用因果卷积、L2 norm、GDN 递推和门控 norm。下面截取其中的状态交接：先给本轮分配工作区，把 AR 初态复制进去，再调用镜像已有的递推算子。

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

这里省略了空历史分支，完整函数见下方。`groups` 是 V head 与 Q/K head 的比例；HF 输入已按这个比例展开，原生接口支持分组，因此传入时再恢复为少量 Q/K head。当前 A2 接口一次最多处理 16 行，所以 `recurrent_slices` 将 K32 分成两段，后一段接着前一段末状态计算。只有本层递推分段，模型不需要为此跑两遍。

卷积也按同样原则执行：把历史尾部接到当前块之前，只写私有状态。递推输出经过门控 norm 和 O 投影后，回到本层残差计算。下面保留完整调用，便于对照输入布局和跨段状态；PyPTO 可以使用自己的分块方式，只要计算关系一致。

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

这两条路径接通后，每类层内部的计算就有了基础。接下来还要把它们放回完整模型，使用共享层传递当前块的特征，再交给主模型验证。norm、投影等可以继续复用或按性能融合，最终 kernel 数量由实现方式决定。

可运行 PyPTO 接入已覆盖两行要求：Full Attention 采用 K32 分页双向计算，GDN 从目标历史的临时副本连续处理两个 16 行 tile。共享层仍调用交付包的原生计算；可选四层合并 HOST 编排进一步减少提交次数，不改变上述公式。

## 3. 模型执行与推理引擎接入

### 3.1 一轮投机如何执行

起草时，当前块先经过共享 Embedding，再通过 64 层。每层按 `共享 input norm -> draft Full Attention / GDN -> 残差 -> 共享 post norm / MLP -> 残差` 执行，最后经过共享 Final Norm 和 LM head，得到候选分布。MLP 等虽然使用主模型权重，但 draft 输入不同，仍需重新计算，不能直接拿主模型的中间结果代替；共享权重省下的是第二套 backbone 参数。

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

主模型随后按因果规则验证候选。接受前缀对应的 **AR 计算结果** 才成为正式历史，draft KV 和 GDN 状态不会转正。对 Full Attention，引擎保留接受位置之前的 KV，并限制后续可读长度；对 GDN，还要取回那个位置对应的递推矩阵和卷积尾部。因为块末矩阵已经包含后续位置的信息，不能简单缩短长度来回滚。现有实现保存逐位置 checkpoint，也可以通过重新计算接受前缀恢复状态。

### 3.2 PyPTO 与引擎的接口

当前接入在同一个 OneServing `DistributedWorker` 中运行主模型和完整 draft。`HeadSource` 读取训练 head，`DraftWeights` 管理驻留或流式上传，`NeuralDraft.propose` 绑定共享 backbone 与目标历史；Full Attention 读取分页池，GDN 在私有工作区更新状态。PyPTO 执行全部模型计算。

`OrthrusRunner` 对接原生 scheduler，`NativeTarget` 调用原生 AR 程序分块验证候选；批量回读有效 logits 后做 greedy 校正。拒绝后缀时恢复 GDN 和页表长度，只重放接受前缀；已验证输出由 scheduler 逐 token 交付。缓存分配、页表和接受决策由引擎管理，算子按传入的长度和布局读写数据。

`--draft-group-size 4` 将 3 个 GDN 层与 1 个 Full Attention 层编排为一次 HOST 提交，head 同时按组上传。该配置只改变执行粒度，K 仍为 32；共享层继续使用原生计算。完整 TP 尚需对接原生的局部 backbone 和归约接口。

跨卡时还要调整尺寸。参考 PR 使用单卡完整规格，现有 Torch-NPU 接入则有 TP 分片：TP2 每 rank 的 Full Attention 为 12 个 Q head、2 个 KV head，GDN 为 8 个 Q/K head、24 个 V head；TP4 分别为 6/1 和 4/12。因此 PyPTO 入口需要按 TP 配置接收尺寸和分片权重，不能固定为单卡常量。加载 head 时也要核对基模、K、mask token、混合层策略和 TP 规格。

### 3.3 可运行实现与验证范围

上面的函数用于单独对照计算，完整流程可以通过仓库已有的 Torch-NPU 执行器和 vLLM-Ascend 接入运行。下列命令加载真实模型，执行起草、验证和缓存检查。没有提供 head 时，程序从主模型对应的 Full Attention 或 GDN 模块复制参数初始化，用来检查链路是否能运行。

```bash
cd online-speculation
python scripts/run_in_runtime.py scripts/validate_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

运行时按挂载位置修改模型路径；已有训练 head 则指定 `--head-dir`，并使用它对应的 mask token 和 TP 规格。命令中的 mask ID 只用于初始化功能用例。

现有验证分为两层。计算上，已对照双向 Full Attention、非零初态 GDN、跨分段状态和相关前后处理；真实模型上，已在 27B、TP2 中覆盖 K=2/8/17/32 的起草与原生验证、全接受和连续拒绝后的恢复、分页历史、缓存隔离、head 导出重载及采样校准。另有 TP4 加载和 draft 更新记录，训练反向不列入本次 PyPTO 需求。

PyPTO 路径也已在真实 27B、K32 上完成独立验证：52 项 Attention 检查、128 项真实输入逐层 HF 对照、六个历史边界、八组全接受/拒绝/截断用例、TP2 文件重载及 scheduler 续写均通过。原生与分页后端、分页逐层与四层合并两组对照的有效 logits 和候选均逐字节相同。CPU 回归共 372 项通过。

上述 PyPTO 验证范围为 TP1、一个活动请求、512 个总位置及 greedy。四层合并将单轮起草耗时中位数从 2.600 s 降为 2.315 s，不能当作业务吞吐收益；当前仍主要受流式 head 上传限制。完整 TP、长上下文和采样需要单独接入与验证。代码入口、原始记录和复现命令见 [执行与验证](online-speculation/docs/QWEN38_PYPTO_PAGED_VALIDATION.md)。

## 参考资料

- PyPTO 实现范围与固定版本：[pypto-lib PR #1503](https://github.com/hw-native-sys/pypto-lib/pull/1503)、[pypto-serving PR #281](https://github.com/hw-native-sys/pypto-serving/pull/281)。
- 混合层算法参考：[Orthrus Qwen3.5 实现](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py)。
- 本仓库记录：[运行与配置](online-speculation/docs/QWEN38_VLLM.md)、[验收摘要](online-speculation/docs/QWEN38_ACCEPTANCE_SUMMARY.md)、[详细验证](online-speculation/docs/QWEN38_VLLM_VALIDATION.md)、[推理算子审计](online-speculation/docs/QWEN38_INFERENCE_OPS_AUDIT.md)。
