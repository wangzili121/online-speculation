# Orthrus-Qwen3.8 NPU 推理适配与算子设计

## 1. 背景与适配目标

Orthrus 先让 draft 分支一次预测多个 token，再交给主模型验证。draft 与主模型共享大部分权重，但用独立参数处理 token 之间的信息。要把它放到 PyPTO 上运行，需要先看清楚 Qwen3.8-27B 每类层怎样计算，再判断现有实现哪些能沿用、哪些需要调整。

下文用 AR 表示主模型的自回归分支，draft 表示预测候选的分支。这里只讨论文本推理，不包含训练和反向算子。

### 1.1 草稿分支与主模型的关系

Qwen3.8-27B 的文本部分沿用 Qwen3.5 系列的混合结构：每三个 GDN 层后接一个 Full Attention 层，重复 16 次，共 64 层。Full Attention 从历史 KV 中读取信息，GDN 则通过卷积和矩阵递推保留历史。两类层对历史的处理不同，draft 也需要分别适配。

每层中，负责这部分计算的模块以下称为 mixer。Orthrus 为 mixer 准备独立的 draft 参数，其他层参数继续与主模型共享：

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

draft 每轮输入一个块：`anchor + (K-1) 个 MASK`。anchor 是已经确定的最后一个 token，MASK 占据接下来要预测的位置，输入中没有未来答案。设 anchor 位于 `a`，它之前的 `[0,a)` 直接读取主模型历史；anchor 和 MASK 则占据 `a` 到 `a+K-1`，在本轮一起计算。常用的 `K=32` 就是一个 anchor 加 31 个候选位置，K 本身可以配置。

在 Full Attention 层，块内位置可以互相读取信息；在 GDN 层，当前实现仍从 AR 历史状态出发，按顺序递推。**这两种计算合在一起，构成 `orthrus_qwen35_reference` 策略**，依据的是 Orthrus Qwen3.5 混合模型参考实现。原始普通 Qwen3 没有 GDN，因此不能直接把它的双向 SDPA 套到所有层。加载训练 head 时，也要确认它采用同样的混合层策略。

### 1.2 PyPTO 已有基础与本次改动范围

PyPTO 已有 Qwen3.8 整网前向的实现基础，位于 `pypto-lib` PR #1503。本文按提交 `1d454a1b8f7af18c9b176d8aeaa60dc565a6a9e9` 核对，下文文件均位于 `models/qwen3_8_27b/`。该 PR 及配套 `pypto-serving` PR #281 尚未合入，公开验证的是单卡、单请求、最多 128 个位置的普通推理，尚不包含本报告的投机执行方式。

在这个基础上，**主要要扩展两类 mixer 的调用方式**。Full Attention 需要一次计算 K 行 query，让它们读取同一段历史和整个当前块；GDN 需要带着已有历史状态连续处理 K 行，而且只更新临时副本。前者改变可见范围，后者改变输入长度和状态传递方式，因而不能用一次 mask 修改概括全部工作。

投影、norm、RoPE 和 MLP 的公式可以沿用。接入时仍要给它们绑定正确的 draft 或共享权重，按 TP 规格分片，再与主模型交换历史。下面先说明两类 mixer 内部要改什么，第 3 节再把它们放回完整推理流程。

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

对比两式，softmax Attention 本身没有变，变化在传给它的 KV：**每个 draft query 读取相同的历史前缀，以及整个临时块**。这条路径不涉及 DSV4 的 sink、压缩 KV 或 indexer。

Attention 前后还要保留 Qwen 自身的处理。该模型有 24 个 query head、4 个 KV head，以 GQA 方式共享，head 维度为 `d=256`。Q/K 先做 RMSNorm，再对前 64 维做 RoPE；Attention 输出乘 query 投影产生的 sigmoid gate，最后经过 O 投影。Q/K norm 的缩放使用 `1 + weight`。这些步骤都参与最终结果，移植时要和 Attention 一起对齐。

#### 2.1.2 PyPTO 已有实现与所需修改

现有 `attention_state.py` 已包含上述 Q/K norm、部分 RoPE、GQA 和门控。历史 Attention 位于 `full_network_core.py::attention_context_full`，目前从连续 KV 中读取数据，只计算第一个有效 query，并按当前位置限制可见范围；新 K/V 由另一个入口 `append_kv_full` 追加。

因此可以保留前后处理，重点扩展 Attention 入口：一次接收 K 行 query，并额外接收临时块 K/V。所有行的历史长度都设为 `a`，临时块全部可见。由于这些 K/V 由 draft 参数计算，不是主模型的正式历史，起草阶段不调用 AR 的 KV append，只写自己的输出和工作区。候选接受后，正式 KV 由主模型验证时重新计算。

首版可以先接连续历史，与下面的 Torch-NPU 实现比较结果。之后若要直接读取主模型分页缓存，再增加页表读取。历史和临时块可以放进同一次 Attention，也可以分开计算，再用 LSE（log-sum-exp）恢复两部分的相对权重后合并；具体拆成几个 kernel，取决于接口和性能，不影响上面的定义。

#### 2.1.3 Torch-NPU 实现

现有 Torch-NPU 版本就是按上述顺序执行：先用 draft 权重投影，再调用镜像中的 `triton_split_qkv_rmsnorm_mrope` 完成拆分、Q/K norm 和部分 RoPE。随后由本轮私有的 `_ScratchCache` 拼接历史与临时 KV，交给融合 Attention：

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

输入布局为 `[batch, head, token, dim]`，融合算子直接支持 GQA，所以保留原来的少量 KV head，不复制整段历史。当前单请求且没有 padding，传入的历史和临时块都可见；有 padding 时再屏蔽无效位置。这里还要转换 mask 的布尔含义：Torch SDPA 的 `True` 表示可见，FAS 的 `True` 则表示禁止读取。

在进入这段计算之前，适配层已按页表把主模型 KV 收集成独立连续快照。后续拼接只发生在 draft 私有对象中，不影响主模型，但仍有一次历史复制，并非零拷贝分页读取。折叠部分保留实际 Attention 和历史读取函数，运行时依赖本仓库的模型对象与环境。

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

GDN 与前一节不同：它不逐条读取历史 KV，而是带着历史状态继续计算。输入先经过宽度为 4 的因果逐通道卷积，再用门控 Delta Rule 更新矩阵状态。Q/K 各有 16 个 head，V 有 48 个 head，每个 head 维度为 128，三个 V head 共享一组 Q/K。

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

现有 `delta_state.py` 已实现这里的卷积、head 分组、标量衰减、Delta Rule 和门控 norm，可以直接作为 Qwen GDN 的基础。`decode_layer.py::delta_decode_into` 也接受 `previous_conv` 和 `previous_recurrent`，能够接上历史，不过一次只处理一个有效 token。

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

## 3. 模型执行与推理引擎接入

### 3.1 一轮投机如何执行

起草时，当前块先经过共享 Embedding，再通过 64 层。每层按 `共享 input norm -> draft mixer -> 残差 -> 共享 post norm / MLP -> 残差` 执行，最后经过共享 Final Norm 和 LM head，得到候选分布。MLP 等虽然使用主模型权重，但 draft 输入不同，仍需重新计算，不能直接拿主模型的中间结果代替；共享权重省下的是第二套 backbone 参数。

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

PyPTO 可以先只接管 Full Attention 和 GDN 这两类 mixer，主模型、共享 MLP、TP 通信和验证采样继续由现有 vLLM-Ascend 执行器负责。每次调用时，执行器传入 draft 参数、历史长度、KV 或 GDN 初态，并提供输出和临时工作区。

调用双方首先要对齐张量布局、设备地址和有效长度，再约定 stream 怎样同步、临时缓冲何时释放，确保 PyPTO 读到的是本轮有效历史。缓存分配、页表和接受决策仍由引擎管理，算子只按这些信息读写数据。

在此基础上，可以再把整段 draft 迁入 PyPTO，接齐共享层、残差和跨卡通信。如果主模型也改由 PyPTO 执行，还要实现上一节的块验证和接受前缀恢复。这样可以先验证局部计算，再逐步扩大执行范围。

跨卡时还要调整尺寸。参考 PR 使用单卡完整规格，现有 Torch-NPU 接入则有 TP 分片：TP2 每 rank 的 Full Attention 为 12 个 Q head、2 个 KV head，GDN 为 8 个 Q/K head、24 个 V head；TP4 分别为 6/1 和 4/12。因此 PyPTO 入口需要按 TP 配置接收尺寸和分片权重，不能固定为单卡常量。加载 head 时也要核对基模、K、mask token、混合层策略和 TP 规格。

### 3.3 可运行实现与验证范围

上面的函数用于单独对照计算，完整流程可以通过仓库已有的 Torch-NPU 执行器和 vLLM-Ascend 接入运行。下列命令加载真实模型，执行起草、验证和缓存检查。没有提供 head 时，程序从主模型 mixer 复制参数初始化，用来检查链路是否能运行。

```bash
cd online-speculation
python scripts/run_in_runtime.py scripts/validate_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 --mask-token-id 1 --block-size 32 \
  --max-model-len 2048 --kv-cache-memory-bytes 8589934592
```

运行时按挂载位置修改模型路径；已有训练 head 则指定 `--head-dir`，并使用它对应的 mask token 和 TP 规格。命令中的 mask ID 只用于初始化功能用例。

现有验证分为两层。计算上，已对照双向 Full Attention、非零初态 GDN、跨分段状态和相关前后处理；真实模型上，已在 27B、TP2 中覆盖 K=2/8/17/32 的起草与原生验证、全接受和连续拒绝后的恢复、分页历史、缓存隔离、head 导出重载及采样校准。另有 TP4 加载和 draft 更新记录，训练反向不列入本次 PyPTO 需求。

这些是 Torch-NPU 路径的结果，可以作为 PyPTO 接入的对照，但 PyPTO Orthrus 本身尚未验证。数值检查也还没有全部通过：原生不同计算形状下存在 K32 数值分叉，严格逐 token 一致性尚未达成。当前覆盖 BF16、eager、同步单请求，量化、图模式和动态 batch 需要另行适配。初始化 head 用来验证链路，业务接受率和加速收益需要训练 head 另行评估。

因此，接下来先让两类 PyPTO mixer 与现有结果对齐，再放回整层、接入真实主模型历史，最后检查各接受位置恢复、TP 分片和长上下文。计算和流程确认后，再根据搬运与执行耗时决定是否增加分页直读或进一步融合。

## 参考资料

- PyPTO 实现范围与固定版本：[pypto-lib PR #1503](https://github.com/hw-native-sys/pypto-lib/pull/1503)、[pypto-serving PR #281](https://github.com/hw-native-sys/pypto-serving/pull/281)。
- 混合层算法参考：[Orthrus Qwen3.5 实现](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py)。
- 本仓库记录：[运行与配置](online-speculation/docs/QWEN38_VLLM.md)、[验收摘要](online-speculation/docs/QWEN38_ACCEPTANCE_SUMMARY.md)、[详细验证](online-speculation/docs/QWEN38_VLLM_VALIDATION.md)、[推理算子审计](online-speculation/docs/QWEN38_INFERENCE_OPS_AUDIT.md)。
