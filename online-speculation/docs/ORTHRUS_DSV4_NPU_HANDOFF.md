# Orthrus-DSV4 NPU 推理实现与 PyPTO 算子需求

## 1. 要支持的计算是什么

**Orthrus-DSV4 的核心需求，是让 diffusion Attention 一次处理整个待预测块，同时读取主模型已经算好的历史。Attention 公式没有改变，变化在于使用哪套参数、哪些 KV 可见，以及本轮是否更新持久状态。**

本文以 DeepSeek-V4-Flash-0731、Ascend A2 为对象，讨论推理，不涉及训练和在线更新。
现成算子实现包含 BF16 单层 demo 和基于固定 v0.26 环境的 W8A8 服务路径。
PyPTO 对照版本固定为 [pypto-lib 4c908f3](https://github.com/hw-native-sys/pypto-lib/tree/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark)，不把它的能力泛化成所有版本的能力。

### 1.1 从 DSV4 到 Orthrus

原生 DSV4 每层先把当前输入投影为 query 和 KV，再根据层类型读取历史。SWA 只使用滑动窗口；HCA 在滑窗之外读取 C128 压缩历史；CSA 用 indexer 从 C4 压缩历史选出相关行，再做 Attention。正常 AR 路径还要更新原始 KV、压缩 KV 和压缩器状态，供下一步解码使用。

Orthrus 保留这三种层的结构，但每层增加自己的 diffusion Attention 参数。每轮输入不是一个新 token，而是 `anchor + MASK × 31`：anchor 是本轮已知的起始 token，MASK 是未来位置的占位符。当前服务块长 `K=32`，用于提出后续 31 个候选。

设 anchor 的位置为 $a$，则本轮读取的历史是位置 $0,\ldots,a-1$。全部 $K$ 个 query 都读取这段固定历史，并且互相读取整个临时块。**“双向”不要求周围位置先生成答案，而是让它们在同一层交换当前中间特征。**

```text
anchor + MASK × 31
        |
共享 embedding
        |
43 层 block，每层：
  共享 mHC / 外层 norm
        |
  diffusion Q/KV 投影，产生 K 行 query 和临时块 KV
        |
        +-- SWA：滑窗历史 -------------------------+
        +-- HCA：滑窗历史 + C128 压缩历史 ----------+--> 加入整个临时块
        +-- CSA：滑窗历史 + indexer 选中的 C4 历史 --+    统一归一化，sink 计一次
        |
  diffusion 输出投影
        |
  共享 mHC / 外层 norm / MoE
        |
共享 LM head -> 候选 token 及其概率
        |
主模型按因果 Attention 验证 -> 接受前缀或拒绝补采样
```

diffusion 分支独立持有 Attention 投影、内部 norm、sink，以及 CSA 的 indexer query/weight 参数；embedding、外层 norm、mHC、MoE 和 LM head 共享主模型参数。共享权重不等于复用本轮激活，draft 仍需经过这些计算。

主模型历史只读，临时 draft KV 不转正。验证时由主模型重新计算候选的 target KV，再由引擎维护接受前缀和拒绝后的状态。

### 1.2 为什么不能只改一个 mask 参数

在能接收完整 Q/K/V 和任意 mask 的 SDPA 接口里，这个可见性规则可以直接表达。但 PyPTO 的 DSV4 实现不是一个接收完整矩阵的通用 SDPA：它把分页读取、因果边界、压缩器和 cache 写回编排成了专门的 decode 路径，有些 Attention 还融合了逆 RoPE 和输出整理。

因此需要调整的是**数据入口和计算编排**：改用 diffusion 权重，固定历史边界，加入单独存放的临时块，并绕开 AR 写回。不是重新发明 Attention，也不是每个基础算子都重写。

下面按公共双向块、SWA、HCA、CSA 四个模块说明。**四个模块是需求划分，不是必须新增四个物理 kernel。** 公共块可以融合进历史 Attention，已有合并能力也可以继续使用。

### 1.3 如何阅读实现

每个模块先说明 PyPTO 现有计算，再给出 Orthrus 公式、现成算子实现和开发要求；完整函数放在紧邻的折叠区。源码不是省略参数的伪代码，函数名和实现取自本仓库文件，来源版本为 `360ab06` 所在代码快照。

W8A8 源码是服务模块中的完整函数或类方法，依赖固定版本 vLLM-Ascend，不是可逐段粘贴运行的独立脚本。第 4 节另附完整 BF16 demo、运行脚本和公共辅助模块，不依赖 vLLM metadata；后面的链接仅用于定位文件。

## 2. 四个计算模块

公式以单个 head、反量化后的实数表示。定义可见 KV 行集合 $C$、query $q$、模型缩放 $\alpha$ 和 sink logit $s$：

$$
\mathcal A(q,C;s)=
\frac{\sum_{c_j\in C}\exp(\alpha q c_j^\top)c_j}
{\exp(s)+\sum_{c_j\in C}\exp(\alpha q c_j^\top)}.
\tag{1}
$$

DSV4 在这里用同一条 shared KV 向量 $c_j$ 作为 key 和 value。sink 仅增加 softmax 分母，等价于一个 value 为零的额外项。$AR$ 和 $D$ 分别表示主模型和 diffusion 分支；下文的拼接符号 $\Vert$ 表示把多组 KV 一起纳入 Attention，不要求物理复制拼接。

### 2.1 公共双向块：让 K 个位置互相可见

**PyPTO 已有什么。** `dspark_attention.py` 已经实现整块双向 Attention：它先把本轮 draft KV 按 slot mapping 写入 DSpark 的私有 cache，再从这个 cache 收集历史滑窗和完整 draft 块。每个 query 都看得到全部块内位置，不使用因果三角遮挡。

计算内部采用分块 QK、softmax、PV：矩阵单元计算 QK/PV，向量单元维护最大值、指数和及累加输出，通过流水线同步连接。块宽来自 `DSPARK_SPEC_TOKENS`，可见行数和临时空间随配置生成。这说明已有可复用的双向计算和 online-softmax，不意味着现有入口已适用于 Orthrus 的 `K=32`、三类目标历史和只读要求。

**Orthrus 要什么。** 由 diffusion 投影生成临时块 $B^D=\{c_0^D,\ldots,c_{K-1}^D\}$，直接作为 Attention 输入，不先提交到持久 cache。每个位置 $i$ 的块内计算为：

$$
M_{ij}^{D}=0\quad(0\le i,j<K),\qquad
O_{B,i}=\mathcal A(q_i^D,B^D;-\infty),\qquad
L_{B,i}=\log\sum_{j=0}^{K-1}\exp\!\left(\alpha q_i^D(c_j^D)^\top\right).
\tag{2}
$$

这里 $M$ 是 additive mask，零表示允许读取；$O_{B,i}$ 是只对块内归一化的输出，$L_{B,i}$ 是不含 sink 的 LSE。

历史与临时块最终必须共享同一个 softmax 分母。现成算子分段计算时，每段返回输出 $O_r$ 和不含 sink 的 $L_r$，随后：

$$
L=\operatorname{logsumexp}(s,L_1,\ldots,L_R),\qquad
O=\sum_{r=1}^{R}\exp(L_r-L)O_r.
\tag{3}
$$

$R$ 是段数；空历史段使用 $O_r=0,L_r=-\infty$。sink 作为 $(0,s)$ 加入一次，不能给每段各加一次。该式在实数计算下等价于式 (1)，实际低精度计算仍需数值容差验证。

**现成算子怎样做。** BF16 demo 分别调用 FIA 计算滑窗、压缩历史和块内输出，再用 Torch 的 FP32 LSE 运算合并。W8A8 服务把滑窗与块合为一次 FIA，减少一次调用；压缩历史另算，最后用 `npu_attention_update` 合并，避免用多次 Torch 操作完成跨段归一化。

这里的 FIA 是完整的 `torch_npu.npu_fused_infer_attention_score` 调用，不是新增 kernel。它返回 Attention 输出和 LSE；`sparse_mode=0` 配合显式 mask/有效长度表达非因果读取。以下是服务实际使用的封装，包含分页布局转换和空历史处理。


<details>
<summary>完整源码：FIA 封装与 LSE 布局转换</summary>

```python
def _normalize_fia_lse(lse: torch.Tensor, batch: int, queries: int) -> torch.Tensor:
    """Normalize Ascend FIA BSND LSE to ``[B,Q,H,1]``."""
    if lse.ndim == 4 and lse.shape[0] == batch and lse.shape[2] == queries:
        return lse.transpose(1, 2)
    if lse.ndim == 4 and lse.shape[:2] == (batch, queries):
        return lse
    raise RuntimeError(f"unexpected FIA LSE shape: {tuple(lse.shape)}")

def _fia_segment(
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_lengths: Sequence[int] | torch.Tensor,
    key_lengths: Sequence[int] | torch.Tensor,
    scale: float,
    *,
    allow_empty: bool = False,
    block_table: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    if query.ndim != 4 or key_value.ndim != 4:
        raise ValueError("Orthrus FIA expects BSND query and key/value")
    kv_heads = key_value.shape[2]
    block_size = key_value.shape[1] if block_table is not None else 0
    # FIA paged KV is [blocks, heads, block_size, dim]. DSV4 has one
    # KV head, so this transpose only changes the view, without copying KV.
    if block_table is not None:
        key_value = key_value.transpose(1, 2)
    key_value = key_value.contiguous()
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        key_value,
        key_value,
        num_heads=query.shape[2],
        num_key_value_heads=kv_heads,
        input_layout="BSND",
        sparse_mode=0,
        scale=scale,
        atten_mask=atten_mask,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths=query_lengths,
        actual_seq_lengths_kv=key_lengths,
        softmax_lse_flag=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Ascend FIA did not return output and LSE")
    output, lse = result[:2]
    lse = _normalize_fia_lse(lse, query.shape[0], query.shape[1])
    empty = None
    if allow_empty and torch.is_tensor(key_lengths):
        empty = key_lengths.to(device=output.device) == 0
    elif not torch.is_tensor(key_lengths) and any(length == 0 for length in key_lengths):
        empty = torch.tensor(
            [length == 0 for length in key_lengths], dtype=torch.bool, device=output.device
        )
    if empty is not None:
        output = torch.where(
            empty.view(-1, 1, 1, 1), torch.zeros_like(output), output
        )
        lse = torch.where(
            empty.view(-1, 1, 1, 1),
            torch.full_like(lse, float("-inf")),
            lse,
        )
    return output, lse
```

</details>


LSE 合并函数同时列出 Torch 参考路径和 NPU 路径。Attention 调用方负责把 sink 追加为零输出的一段；它没有混入各个局部 Attention。


<details>
<summary>完整源码：Torch LSE 合并与 npu_attention_update</summary>

```python
def merge_attention_segments(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention segments exactly.

    Every output is ``sum(exp(score) * value) / sum(exp(score))`` for one
    segment and every LSE is ``log(sum(exp(score)))`` with a final singleton
    dimension. Empty segments must be omitted by the caller.
    """
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    lse_stack = torch.stack([lse.float() for lse in logsumexp], dim=0)
    merged_lse = torch.logsumexp(lse_stack, dim=0)
    weights = torch.exp(lse_stack - merged_lse.unsqueeze(0))
    merged = sum(
        output.float() * weight
        for output, weight in zip(outputs, weights, strict=True)
    )
    return merged.to(outputs[0].dtype), merged_lse

def merge_attention_outputs(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Merge segmented attention outputs using the native NPU update op."""
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if len(shape) < 2 or lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    if outputs[0].device.type == "npu":
        import torch_npu

        head_dim = shape[-1]
        merged, _ = torch_npu.npu_attention_update(
            [lse.float().reshape(-1) for lse in logsumexp],
            [output.reshape(-1, head_dim) for output in outputs],
            0,
        )
        return merged.view(shape)

    merged, _ = merge_attention_segments(outputs, logsumexp)
    return merged
```

</details>


**PyPTO 应怎样改。** 复用 DSpark 的块内 QK/softmax/PV 计算，把临时块改成独立输入，去掉“先写私有 cache、再读回来”的依赖；将 K、padding 和临时空间纳入形状配置。公共块可以并入 SWA/HCA/CSA 的流水线，不必单独落地一个 kernel。

PyPTO 的 HCA 已经使用 online-softmax 状态合并，因此**不需要为 LSE 公式从零再造一种合并算法**。设一段保存最大值 $m_r$、指数和 $\ell_r$、未归一化分子 $u_r$：

$$
\ell_r=\sum_j e^{z_{rj}-m_r},\qquad
u_r=\sum_j e^{z_{rj}-m_r}v_{rj},\qquad
L_r=m_r+\log\ell_r,\quad O_r=u_r/\ell_r.
\tag{4}
$$

$z_{rj}$ 和 $v_{rj}$ 分别是该段第 $j$ 行的注意力分数和 value；式 (4) 的输出转换用于非空段。空段置 $m_r=-\infty,\ell_r=0,u_r=0$，合并时贡献为零。已有状态可以直接扩展到临时块：

$$
m=\max(s,m_1,\ldots,m_R),\qquad
O=\frac{\sum_r e^{m_r-m}u_r}
{e^{s-m}+\sum_r e^{m_r-m}\ell_r}.
\tag{5}
$$

式 (3) 是现成算子返回 LSE 时的接口形式，式 (5) 是 PyPTO 融合实现更容易复用的状态形式，两者表达同一计算。

源码定位：[DSpark 双向 Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/dspark_attention.py)、[HCA 合并](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。

### 2.2 SWA：固定历史窗口，再接上整个临时块

**PyPTO 已有什么。** `decode_sparse_attn_swa.py` 从分页 cache 读取每个 AR query 的滑窗。多 query 验证时，调用方已经写入当前 token 的 KV；Attention 先为第一个 query 收集窗口，再复用这份数据，为后续 query 追加新 token、移出过期 token。每行的可见范围随位置向右移动，仍是因果 Attention。

这里要区分：滑窗 Attention 子函数负责读取，cache 写入发生在它的调用链中，不能笼统地说“所有 Attention kernel 都自己写 cache”。

**Orthrus 要什么。** 设 $W(t)$ 表示截至位置 $t$ 的最近至多 $w$ 行原始 KV，本模型 $w=128$；$p$ 是 AR query 位置。区别是：

| 现有 AR 语义 | Orthrus-SWA 语义 |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert B^D;s^D)$ |

所有 draft query 共用 $W(a-1)$，不能因为第 $i$ 行位置更靠右就丢掉更多历史；整个 $B^D$ 对每行开放。只有历史 padding 被遮挡，块内不做因果裁剪。

**现成算子怎样做。** 服务按每个请求的历史长度生成窗口行号，再经页表读取主模型 SWA cache；随后将这段短历史与临时块 `cat` 成连续 KV，一次调用 FIA。历史窗口最多 128 行，因此这里只收集窗口，不把整个长上下文展开。输出还未包含 sink，稍后和其他历史段一起归一化。


<details>
<summary>完整源码：滑窗行号、分页读取、padding mask 与联合 FIA</summary>

```python
def tail_logical_indices(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """Build front-packed logical indices for each request's trailing window."""
    if seq_lens.ndim != 1 or width <= 0:
        raise ValueError("seq_lens must be rank one and width must be positive")
    lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, width))
    starts = seq_lens - lengths
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = starts[:, None] + offsets[None, :]
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1)

def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_indices: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Gather logical token indices from ``[blocks, block, heads, dim]`` cache.

    ``logical_indices`` may be ``[batch, tokens]`` or
    ``[batch, queries, tokens]``. Invalid ``-1`` entries are returned as zero;
    their lengths must also be supplied to the attention kernel.
    """
    if cache.ndim not in (3, 4):
        raise ValueError(f"unsupported paged cache shape: {tuple(cache.shape)}")
    if block_table.ndim != 2 or logical_indices.ndim not in (2, 3):
        raise ValueError("block_table and logical_indices have invalid rank")
    if block_table.shape[0] != logical_indices.shape[0]:
        raise ValueError("cache gather batch dimensions differ")
    block_size = cache.shape[1]
    valid = logical_indices >= 0
    safe = logical_indices.clamp_min(0).long()
    logical_blocks = torch.div(safe, block_size, rounding_mode="floor")
    if validate and logical_blocks.numel():
        if int(logical_blocks.max().item()) >= block_table.shape[1]:
            raise ValueError("logical index exceeds block table capacity")
    batch_shape = logical_indices.shape
    batch_ids = torch.arange(
        batch_shape[0], device=logical_indices.device, dtype=torch.long
    ).view((batch_shape[0],) + (1,) * (logical_indices.ndim - 1))
    physical_blocks = block_table[batch_ids, logical_blocks]
    if validate and torch.any(valid & (physical_blocks < 0)):
        raise ValueError("logical index maps to an unallocated cache block")
    physical_blocks = physical_blocks.clamp_min(0).long()
    offsets = safe % block_size
    if logical_indices.ndim == 3 and logical_indices.shape[0] > 1 and cache.is_contiguous():
        # C4 has a different sparse list per query. A single row index avoids
        # materializing the much larger two-index gather on the NPU.
        rows = physical_blocks * block_size + offsets
        gathered = cache.flatten(0, 1).index_select(0, rows.flatten()).view(
            *logical_indices.shape, *cache.shape[2:])
        mask = ~valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
        return gathered.masked_fill(mask, 0)
    gathered = cache[physical_blocks, offsets]
    mask = valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
    return torch.where(mask, gathered, torch.zeros_like(gathered))

def history_and_block_mask(
    history_lengths: torch.Tensor, history_width: int, block_width: int
) -> torch.Tensor:
    """Mask padded history, while all queries see the entire temporary block."""
    offsets = torch.arange(history_width + block_width, device=history_lengths.device)
    masked = (offsets[None, :] >= history_lengths[:, None]) & (offsets[None, :] < history_width)
    return masked[:, None, None, :].expand(-1, 1, block_width, -1).contiguous()
```

</details>


<details>
<summary>完整源码：SWA 历史与双向块的一次联合计算</summary>

```python
def _swa_block_segment(query, history, local, lengths, scale):
    # Both branches use the same query and KV format. One softmax avoids two
    # FIA launches and an intermediate normalized output; only padding is masked.
    kv = torch.cat((history, local), dim=1)
    mask = history_and_block_mask(lengths, history.shape[1], local.shape[1])
    batch = query.shape[0]
    return _fia_segment(
        query, kv, [query.shape[1]] * batch, [kv.shape[1]] * batch, scale,
        atten_mask=mask,
    )
```

</details>


上述布尔 mask 中 `True` 表示屏蔽位置，和式 (2) 的 additive mask 不是同一编码。历史长度小于窗口时，补齐的行必须被屏蔽，不能只把 KV 填零，因为零 KV 仍会进入 softmax 分母。

**PyPTO 应怎样改。** 保留分页地址转换和窗口预取，把每行窗口改为统一的 anchor 前窗口；取消 AR 的逐行追加/滑移逻辑，改为读取独立临时块的全部 K 行。可以复用“每请求只取一次公共历史”的优化思路，但不能原样保留因果边界。调用方绕过 draft 的 target KV 写回，Attention 将历史和临时块共同归一化。

源码定位：[PyPTO SWA](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py)。

### 2.3 HCA：在同一窗口之外加入 C128 历史

SWA 解决了近处的历史。HCA 还需要更远的上下文，但读取的是主模型按 ratio-128 生成的压缩行，而不是重新展开全部 token。

**PyPTO 已有什么。** `decode_hca.py` 的 AR 路径执行 Q/KV 投影、原始 KV 写回、ratio-128 压缩和状态更新，再启动 Attention。`decode_sparse_attn_hca.py` 分别处理滑窗和有效 C128 行，最后合并两组 online-softmax 状态，加一次 sink，再做逆 RoPE 和输出布局整理。它已经有分页读取、压缩历史 Attention 和归一化合并，不缺这些基础能力。

**Orthrus 要什么。** 记 $H_{128}(t)$ 为截至位置 $t$ 已完成的 C128 压缩行：

| 现有 AR 语义 | Orthrus-HCA 语义 |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_{128}(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_{128}(a-1)\Vert B^D;s^D)$ |

anchor 前有 $a$ 个历史 token，因此可读完整 C128 行数为 $\lfloor a/128\rfloor$；尚未完成的压缩组仍在目标模型状态中，不让 draft 将它补完。HCA 读取全部有效 C128 行，不做 indexer Top-K。

**现成算子怎样做。** SWA 和块内部分沿用第 2.2 节。压缩历史采用两种原生入口：页大小满足当前 FIA 条件时直接传页表；否则枚举全部有效压缩行，调用 `npu_sparse_flash_attention`。这里“sparse”是分页读取接口，并不改变 HCA 的全量有效历史语义。只有显式选择 `gather` 回退时才把压缩行收集成连续张量。


<details>
<summary>完整源码：C128 分页 FIA、原生 sparse 路径与 gather 回退</summary>

```python
def _c128_segment(query, cache, block_table, lengths, maximum, scale, *, backend=None):
    backend = backend or os.environ.get("ORTHRUS_C128_BACKEND", "auto")
    if backend not in {"auto", "sparse", "gather"}:
        raise ValueError("ORTHRUS_C128_BACKEND must be auto, sparse or gather")
    # FIA supports large pages directly; sparse attention also reads the
    # smaller DSV4 pages without materializing a contiguous history buffer.
    if backend == "auto" and 128 <= cache.shape[1] <= 1024 and cache.shape[1] % 128 == 0:
        return _fia_segment(
            query, cache, [BLOCK_SIZE] * query.shape[0], lengths, scale,
            allow_empty=True, block_table=block_table,
        )
    if backend != "gather":
        width = max(128, (maximum + 127) // 128 * 128)
        offsets = torch.arange(width, device=lengths.device, dtype=torch.int32)
        indices = torch.where(offsets[None, None, :] < lengths[:, None, None], offsets, -1)
        indices = indices.expand(-1, query.shape[1], -1).contiguous()
        return _sparse_history_segment(query, cache, block_table, indices, lengths, scale)
    offsets = torch.arange(max(maximum, 1), device=lengths.device)
    indices = torch.where(offsets[None, :] < lengths[:, None], offsets[None, :], -1)
    kv = gather_paged_cache(cache, block_table, indices)
    return _fia_segment(
        query, kv, [BLOCK_SIZE] * query.shape[0], lengths, scale, allow_empty=True,
    )

def _sparse_history_segment(query, cache, block_table, indices, lengths, scale):
    import torch_npu

    batch, width, heads, _ = query.shape
    counts = (indices >= 0).sum(-1)

    # The MLA API requires a separate RoPE pair. DSV4 already includes RoPE in
    # its 512-d query/key, so a zero auxiliary query preserves q@k exactly.
    # key_rope is a strided read-only view, not another KV allocation.
    auxiliary_query = torch.zeros(batch, width, heads, 64, device=query.device, dtype=query.dtype)
    out, maximum, total = torch_npu.npu_sparse_flash_attention(
        query, cache, cache, indices.to(torch.int32).unsqueeze(2), scale,
        block_table=block_table,
        actual_seq_lengths_query=torch.full_like(lengths, width),
        actual_seq_lengths_kv=lengths,
        query_rope=auxiliary_query, key_rope=cache[..., -64:],
        sparse_mode=0, attention_mode=2,
        layout_query="BSND", layout_kv="PA_BSND", return_softmax_lse=True,
    )
    lse = (maximum + total.log()).squeeze(1).unsqueeze(-1)
    empty = (counts == 0)[..., None, None]
    return out.masked_fill(empty, 0), lse.masked_fill(empty, -torch.inf)
```

</details>


`_sparse_history_segment` 同时供 CSA 使用。该算子接口要求分离的 RoPE 参数，但 DSV4 的 512 维 query/KV 已包含 RoPE 分量，因此代码额外传入零 query，让辅助点积为零，避免重复加入位置项；`key_rope` 只是目标 cache 的只读视图。其返回的最大值和指数和通过 $L=m+\log\ell$ 转成统一 LSE。

**PyPTO 应怎样改。** 保留 C128 分页读取、QK/PV 和状态合并，将有效长度固定到 anchor 前，增加公共双向块贡献。绕过 draft 的 compressor 和持久状态写回，同时调整任务依赖：Attention 应等待历史可读和 diffusion Q/KV 就绪，不再等待被跳过的 AR 写回任务。完成所有部分的合并后再做一次逆 RoPE/输出整理。主模型的压缩规则不变。

源码定位：[HCA 编排](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_hca.py)、[HCA Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。

### 2.4 CSA：由 diffusion query 检索 C4 历史

CSA 与 HCA 的区别是压缩率更低、可选历史更多，所以先由 indexer 选行。这里包含“选哪些行”和“对选中行做 Attention”两个阶段，但它们属于同一条 CSA 路径，不额外算一种模型层。

**PyPTO 已有什么。** `decode_csa.py` 和 `decode_indexer.py` 使用 AR 低秩 query 生成 indexer query，执行 RoPE、Hadamard 变换和量化，再与历史 index key 评分；各 head 的 ReLU 分数按权重求和，分段 Top-K 后归并。历史不超过 512 行时，现有代码还有直接枚举全部有效行、跳过评分排序的短历史分支。

随后 `decode_sparse_attn_csa.py` 读取滑窗和选中的 C4 行，计算 Attention 并处理 sink/逆 RoPE。正常 AR 编排同时更新 C4 compressor、indexer key/scale cache 和原始 KV。检索范围按各 query 的当前位置计算；它不是可以直接套用的“所有 query 共用 anchor 历史”入口。

**Orthrus 要什么。** 历史 index key 和 C4 KV 继续使用主模型已经生成的内容，query 及其 head 权重改用 diffusion 分支参数。令 $k_u^I$ 为历史 index key，$q_{i,h}^{I,D}$ 为第 $i$ 个 draft 位置、第 $h$ 个 indexer head 的 query，$w_{i,h}^D$ 为含模型缩放的权重；$H_I$ 是 indexer head 数：

$$
A_{i,u}^{D}=\sum_{h=1}^{H_I}w_{i,h}^{D}
\operatorname{ReLU}\!\left(q_{i,h}^{I,D}(k_u^I)^\top\right),\qquad
I_i^D=\operatorname{TopIdx}_{k_I}
\{A_{i,u}^D:0\le u<\lfloor a/4\rfloor\}.
\tag{6}
$$

$k_I=512$ 是最多选取的压缩行数，不是 draft 块长 $K=32$；不足 512 行时只使用有效行。各 query 检索的历史范围相同，但 query 不同，选出的 $I_i^D$ 可以不同。临时双向块全部可见，不参与 Top-K。

| 现有 AR 语义 | Orthrus-CSA 语义 |
| --- | --- |
| $I_p^{AR}$ 由 AR query/权重产生，范围为 $0\le u<\lfloor(p+1)/4\rfloor$ | $I_i^D$ 由 diffusion query/权重产生，范围固定为 $0\le u<\lfloor a/4\rfloor$ |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_4[I_p^{AR}];s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_4[I_i^D]\Vert B^D;s^D)$ |

$H_4[I]$ 是按行号 $I$ 选出的 C4 压缩 KV。需要替换的是 indexer 参数和查询语义，不是重写 Top-K 排序算法。

**现成算子怎样做。** W8A8 服务使用量化 projection、RoPE、Hadamard 和 query 量化，调用原生 `npu_vllm_quant_lightning_indexer`（QLI）。配套 metadata 使用非因果模式，让 K 行都读取完整 anchor 历史，不能沿用 target 验证时的因果 metadata。下面是完整类方法，其中 `indexer` 是 diffusion 分支对象，`target_kv` 是主模型 cache。


<details>
<summary>完整源码：diffusion indexer 查询、量化及 QLI 调用</summary>

```python
@staticmethod
def _c4_topk_readonly(
    indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    qr_scale: torch.Tensor | None,
    target_kv: tuple[torch.Tensor, ...],
    metadata,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Run the diffusion-side indexer over all committed C4 history."""
    import torch_npu

    request = _request_metadata(metadata)
    indexer_wq_b = indexer.wq_b
    if _dsa_runtime._is_w8a8_dynamic(indexer_wq_b) and qr_scale is not None:
        query = torch_npu.npu_quant_matmul(
            qr,
            indexer_wq_b.weight,
            indexer_wq_b.weight_scale,
            pertoken_scale=qr_scale,
            bias=indexer_wq_b.bias,
            output_dtype=hidden_states.dtype,
        )
    else:
        query = _linear(indexer_wq_b, qr)
    query = query.view(-1, indexer.n_heads, indexer.head_dim)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        query.unsqueeze(1),
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[indexer.head_dim - indexer.rope_head_dim, indexer.head_dim],
    )
    query = _dsa_runtime.rotate_activation(query, metadata.hadamard)
    query, query_scale = DeviceOperator.indexer_quantize_query(query)
    _, _, _, indexer_key, indexer_scale, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
        target_kv, 4
    )
    weights = _linear(indexer.weights_proj, hidden_states) * (
        indexer.softmax_scale * indexer.n_heads**-0.5
    )
    # Each query reads the same committed prefix. The target's causal QLI
    # metadata would shorten the history for the first K-1 query rows.
    qli_metadata = getattr(request, "orthrus_qli_metadata", None)
    if qli_metadata is None:
        qli_metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=request.query_start_loc[1:],
            actual_seq_lengths_key=request.seq_lens,
            num_heads_q=indexer.n_heads,
            num_heads_k=1,
            head_dim=indexer.head_dim,
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=request.seq_lens.shape[0],
            max_seqlen_q=BLOCK_SIZE,
            max_seqlen_k=request.max_seq_lens,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=indexer.index_topk,
            sparse_mode=0,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(query.device),
        )
        request.orthrus_qli_metadata = qli_metadata
    topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query,
        key=indexer_key,
        weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
        query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(
            query_scale
        ),
        key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(
            indexer_scale
        ),
        actual_seq_lengths_query=request.query_start_loc[1:],
        actual_seq_lengths_key=request.seq_lens,
        block_table=request.block_table,
        metadata=qli_metadata,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=indexer.index_topk,
        sparse_mode=0,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=4,
        return_value=False,
    )
    return topk
```

</details>


QLI 使用原始 token 历史长度和 `cmp_ratio=4`，返回的是**压缩行号**，不能再除以 4；结果还要按 $\lfloor a/4\rfloor$ 排除无效项，并把有效项排在前面。随后交给 sparse Attention 的有效长度则是压缩行数，不能与 QLI 的长度单位混用。压缩行号经 C4 页表读取对应数据；`gather + FIA` 仅作为兼容回退。


<details>
<summary>完整源码：C4 行号过滤与整理</summary>

```python
def front_pack_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move valid non-negative indices to the front of the last dimension."""
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    valid = indices >= 0
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    packed = torch.gather(indices, -1, order)
    packed_valid = torch.gather(valid, -1, order)
    return torch.where(packed_valid, packed, -1), packed_valid

def c4_history_indices(
    indices: torch.Tensor, history_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """QLI returns compressed row IDs, not original-token positions."""
    complete_rows = torch.div(history_lengths, 4, rounding_mode="floor")
    valid = (indices >= 0) & (indices < complete_rows[:, None, None])
    return front_pack_indices(torch.where(valid, indices, -1))
```

</details>


<details>
<summary>完整源码：C4 sparse Attention 与 gather 回退</summary>

```python
def _c4_segment(query, cache, block_table, indices, lengths, scale, *, validate=False):
    batch, width, heads, dim = query.shape
    backend = os.environ.get("ORTHRUS_C4_BACKEND", "sparse")
    if backend == "gather":
        counts = (indices >= 0).sum(-1)
        kv = gather_paged_cache(cache, block_table, indices, validate=validate)
        out, lse = _fia_segment(
            query.reshape(batch * width, 1, heads, dim),
            kv.reshape(batch * width, indices.shape[-1], 1, dim),
            [1] * (batch * width), counts.flatten(), scale, allow_empty=True,
        )
        return out.view_as(query), lse.view(batch, width, heads, 1)
    if backend != "sparse":
        raise ValueError("ORTHRUS_C4_BACKEND must be sparse or gather")
    if validate:
        logical_blocks = indices.clamp_min(0).long() // cache.shape[1]
        if int(logical_blocks.max()) >= block_table.shape[1]:
            raise ValueError("C4 index exceeds block table capacity")
        batch_ids = torch.arange(batch, device=indices.device)[:, None, None]
        physical = block_table[batch_ids, logical_blocks]
        if bool(((indices >= 0) & ((physical < 0) | (physical >= cache.shape[0]))).any()):
            raise ValueError("C4 index maps to an invalid physical cache block")
    return _sparse_history_segment(query, cache, block_table, indices, lengths, scale)
```

</details>


BF16 demo 使用非量化 `npu_lightning_indexer`，将选中的 C4 行 `gather` 后交给 FIA，便于独立阅读计算。W8A8 服务默认直接从分页 cache 做 sparse Attention，避免为每个 query 展开选中的大张量。两者不是同一条性能路径。

**PyPTO 应怎样改。** 复用已有量化评分、Top-K 归并和分页读取，绑定 diffusion indexer 权重，批量处理 K 个 query；将完整评分分支和短历史枚举分支的有效边界都固定到 anchor 前。跳过 draft 的 indexer compressor/key/scale 写回，只保留 query 侧计算，然后将选中 C4、滑窗与公共双向块统一归一化。

实现时还要区分两种地址：原始滑窗的物理 slot 与 C4 的逻辑压缩行号不能混用。量化尺度、Hadamard 约定、无效项 `-1` 和每个请求的有效长度需要与现成实现对齐。

源码定位：[CSA 编排](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_csa.py)、[Indexer](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_indexer.py)、[CSA Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py)。


## 3. 把四个模块接成一层

前面回答了“读哪些 KV、怎样计算 Attention”。一层真正运行还需要输入投影、输出投影以及与主模型的衔接；这些不是额外的 Attention 语义，但不能仅凭公式相同就认为已经无条件兼容。

### 3.1 输入与输出投影

**输入端。** 对 DSV4-Flash，每行 4096 维 hidden 先经 `wq_a` 降到 1024 维，做内部 RMSNorm，再经 `wq_b` 生成各 head 的 query，执行 head RMSNorm 和 partial RoPE。另一条 `wkv` 生成 512 维 shared KV，再做 norm/partial RoPE，作为本轮临时块。完整模型共有 64 个 Attention head，TP 时使用本 rank 的分片；RoPE 作用于其中的 64 维。

W8A8 服务在 `wq_a` 和 `wkv` 都使用动态 W8A8 时，共用一次输入动态量化，避免相同 hidden 重复量化。低秩 query 的 norm 和动态量化调用融合算子，matmul 使用 head 自己的权重与 scale；这里没有把主模型的量化 scale 套到 diffusion 权重上。

PyPTO 的 `qkv_proj_rope.py` 已有这些基础运算。适配重点是绑定 diffusion 参数，接收 K 行输入，匹配 TP/量化布局，并把生成的 KV 作为临时张量交给 Attention，而非送入目标 cache 写回。


<details>
<summary>完整源码：W8A8 Q/KV 投影、norm、量化和 partial RoPE</summary>

```python
def _linear(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    output = module(value)
    return output[0] if isinstance(output, tuple) else output

def _query_readonly(
    draft_impl,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_a: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project a diffusion query without constructing or updating draft KV."""
    import torch_npu

    if q_a is None:
        q_a = _linear(draft_impl.wq_a, hidden_states)
    if _dsa_runtime._is_w8a8_dynamic(draft_impl.wq_b):
        qr, qr_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
            q_a, draft_impl.q_norm.weight, epsilon=draft_impl.eps
        )
        query = torch_npu.npu_quant_matmul(
            qr,
            draft_impl.wq_b.weight,
            draft_impl.wq_b.weight_scale,
            pertoken_scale=qr_scale,
            bias=draft_impl.wq_b.bias,
            output_dtype=hidden_states.dtype,
        ).unflatten(-1, (draft_impl.n_local_heads, draft_impl.head_dim))
    else:
        qr = _linear(draft_impl.q_norm, q_a)
        query = _linear(draft_impl.wq_b, qr).unflatten(
            -1, (draft_impl.n_local_heads, draft_impl.head_dim)
        )
        qr_scale = None
    query = DeviceOperator.apply_dsa_q_rms(
        query, draft_impl.eps, draft_impl.q_norm_without_weight
    )
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        query.unsqueeze(1),
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
    )
    return query, qr, qr_scale

def _qkv_readonly(draft_impl, hidden_states, cos, sin):
    """Share input quantization across Q/KV, without a target-cache write."""
    import torch_npu

    if (_dsa_runtime._is_w8a8_dynamic(draft_impl.wq_a)
            and _dsa_runtime._is_w8a8_dynamic(draft_impl.wkv)):
        hs_int8, hs_scale = torch_npu.npu_dynamic_quant(hidden_states)

        def project(module):
            return torch_npu.npu_quant_matmul(
                hs_int8, module.weight, module.weight_scale,
                pertoken_scale=hs_scale, bias=module.bias,
                output_dtype=hidden_states.dtype,
            )

        q_a, kv = project(draft_impl.wq_a), project(draft_impl.wkv)
    else:
        q_a = _linear(draft_impl.wq_a, hidden_states)
        kv = _linear(draft_impl.wkv, hidden_states)
    query, qr, qr_scale = _query_readonly(
        draft_impl, hidden_states, cos, sin, q_a=q_a
    )
    kv = draft_impl.kv_norm(kv).view(-1, 1, draft_impl.head_dim)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        kv.unsqueeze(1), cos, sin, rotary_mode="interleave",
        partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
    )
    return query, qr, qr_scale, kv
```

</details>


**输出端。** 三条 Attention 路径完成联合归一化后，先做 inverse partial RoPE，再执行 DSV4 的分组低秩输出投影：按 8 组整理 head，每组投影到 1024 维，再合并映射回 4096 维 hidden。服务复用 diffusion Attention 对象的 `_forward_o_proj`；demo 显式使用 `npu_transpose_batchmatmul` 和 Linear，完整实现见第 4 节。

PyPTO 的 `decode_o_proj.py` 已有输出投影及相关融合。可以沿用计算，但需要核对 K 行的 padding、TP 通信和 mHC 输入输出约定。所对照的 HCA/CSA 等独立运行入口显式限制 TP 为 1/2/4，不能据此承诺 TP8 可直接使用；这是适配和验证项，不代表要另造 matmul。

源码定位：[Q/KV 投影](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/qkv_proj_rope.py)、[输出投影](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_o_proj.py)。

### 3.2 W8A8 服务如何连接这些函数

服务中的 `_history_segments` 是三条路径的统一入口。它从 target cache 取出 SWA 和压缩 cache，先计算所有层都需要的“滑窗 + 临时块”，再按 `compress_ratio` 加入 C4 或 C128 段；最后由 `_attention_readonly` 追加一次 sink、合并并做输出投影。

以下计算方法使用的对象和布局如下，避免只看到变量名而不知道它们从哪里来：

| 输入或对象 | 来源与含义 |
| --- | --- |
| `self.self_attn` / `draft_impl` | 当前层的 diffusion Attention，持有独立投影、内部 norm、sink 和 CSA indexer 参数 |
| `target_attention` / `target_kv` | 绑定的主模型层及其持久 cache；用原生布局解包，不复制整个历史 |
| `hidden_states` / `positions` | 共享 backbone 产生的 `[batch × K, hidden]` 和对应绝对位置；每个请求第一行位置是 anchor |
| `query` / `local_kv` | 投影结果；Attention 内整理为 `[batch, K, local_heads, 512]` 和 `[batch, K, 1, 512]` |
| `layer_metadata` | 主模型 cache 的页表、有效长度、RoPE 表等运行时信息；Orthrus 路径要求历史长度对应 anchor 前的前缀 |
| `_dsa_runtime` / `DeviceOperator` | 固定版本 vLLM-Ascend 的 DSA 量化与 cache 布局适配接口 |
| `BLOCK_SIZE` | 本服务契约中的 draft 块长 32，不是 KV 分页大小；cache 的页大小由实际张量/页表决定 |

下面的函数保留真实调用参数。`_metadata_for_target` 用于不同固定 runtime 布局的兼容；`_build_kv_cache`、`get_forward_context`、设备算子注册及模型构造由 vLLM-Ascend 提供，不是本报告省略的 Attention 算法。


<details>
<summary>完整源码：metadata 获取与目标层查找</summary>

```python
def _request_metadata(metadata):
    helper = getattr(_dsa_runtime, "_require_req_metadata", None)
    if helper is not None:
        return helper(metadata)
    for name in ("prefill", "decode"):
        request = getattr(metadata, name, None)
        if request is not None:
            return request
    raise RuntimeError("DSV4 metadata has neither a prefill nor decode request view")

def _metadata_for_target(target_wrapper, context_metadata, compress_ratio: int):
    """Normalize the release and nightly DSA metadata layouts."""
    prefix = target_wrapper.prefix
    entries = [
        value
        for name, value in sorted(context_metadata.items())
        if name.startswith(prefix)
    ]
    expected = 5 if compress_ratio == 4 else 3 if compress_ratio == 128 else 1
    if len(entries) != expected:
        raise RuntimeError(
            f"Orthrus expected {expected} target metadata entries under {prefix!r}, "
            f"got {len(entries)}"
        )
    if compress_ratio == 4:
        attention, _, _, indexer, swa = entries
    elif compress_ratio == 128:
        attention, _, swa = entries
        indexer = None
    else:
        (swa,) = entries
        attention = None
        indexer = None
    return SimpleNamespace(attention=attention, swa=swa, indexer=indexer)

def _target_attention(self) -> nn.Module:
    target = self._target_attention_ref() if self._target_attention_ref else None
    if target is None:
        raise RuntimeError("Orthrus diffusion head is not bound to the DSV4 target")
    return target
```

</details>


<details>
<summary>完整源码：三类历史路径的统一编排</summary>

```python
def _history_segments(
    self,
    query: torch.Tensor,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    qr_scale: torch.Tensor | None,
    target_kv: tuple[torch.Tensor, ...],
    layer_metadata,
    layer_name: str,
    local_kv: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    draft_impl = self.self_attn.dsa_attn.dsa_attn.impl
    validate_cache = os.environ.get("ORTHRUS_VERIFY_CACHE_CHECKSUM", "0") in {"1", "full"}
    compress_cache, swa_cache, state_cache, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
        target_kv, self.compress_ratio
    )
    swa_req = _request_metadata(layer_metadata.swa)
    seq_lens = swa_req.seq_lens
    batch = int(seq_lens.shape[0])
    if query.shape[0] != batch * BLOCK_SIZE:
        raise RuntimeError(
            f"Orthrus requires a dense Bx{BLOCK_SIZE} query block, got "
            f"{query.shape[0]} tokens for {batch} requests"
        )
    query_bsnd = query.view(batch, BLOCK_SIZE, query.shape[1], query.shape[2])
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []

    swa_indices = tail_logical_indices(seq_lens, draft_impl.window_size)
    swa_lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, draft_impl.window_size))
    swa_kv = gather_paged_cache(
        swa_cache,
        swa_req.block_table,
        swa_indices,
        validate=validate_cache,
    )
    out, lse = _swa_block_segment(
        query_bsnd,
        swa_kv,
        local_kv.view(batch, BLOCK_SIZE, 1, local_kv.shape[-1]),
        swa_lengths,
        draft_impl.softmax_scale,
    )
    outputs.append(out)
    lses.append(lse)

    if self.compress_ratio > 1:
        common = layer_metadata.attention
        assert common is not None and compress_cache is not None and state_cache is not None
        compressed_req = _request_metadata(common)
        if self.compress_ratio == 4:
            assert draft_impl.indexer is not None
            assert layer_metadata.indexer is not None
            indexer_metadata = layer_metadata.indexer
            if hasattr(indexer_metadata, "compressor"):
                indexer_metadata = indexer_metadata.compressor.cache
            indexer_req = _request_metadata(indexer_metadata)
            topk = self._c4_topk_readonly(
                draft_impl.indexer,
                hidden_states,
                qr,
                qr_scale,
                target_kv,
                layer_metadata.indexer,
                indexer_req.cos[layer_name][: hidden_states.shape[0]],
                indexer_req.sin[layer_name][: hidden_states.shape[0]],
            )
            if topk is None:
                raise RuntimeError("C4 diffusion-side indexer returned no candidates")
            if topk.ndim == 3:
                if topk.shape[1] != 1:
                    raise RuntimeError(f"unexpected C4 top-k shape: {tuple(topk.shape)}")
                topk = topk[:, 0]
            topk, _ = c4_history_indices(topk.view(batch, BLOCK_SIZE, -1), seq_lens)
            out, lse = _c4_segment(
                query_bsnd,
                compress_cache,
                compressed_req.block_table,
                topk,
                torch.div(seq_lens, 4, rounding_mode="floor"),
                draft_impl.softmax_scale,
                validate=validate_cache,
            )
            outputs.append(out)
            lses.append(lse)
        else:
            compressed_lengths = torch.div(seq_lens, 128, rounding_mode="floor")
            maximum = getattr(compressed_req, "max_seq_lens", None)
            maximum = (int(maximum) // 128 if maximum is not None
                       else compressed_req.block_table.shape[1] * compress_cache.shape[1])
            out, lse = _c128_segment(
                query_bsnd,
                compress_cache,
                compressed_req.block_table,
                compressed_lengths,
                maximum,
                draft_impl.softmax_scale,
            )
            outputs.append(out)
            lses.append(lse)
    return outputs, lses
```

</details>


<details>
<summary>完整源码：读取目标状态、Attention 合并和输出投影</summary>

```python
def _attention_readonly(
    self, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    context = get_forward_context()
    if context is None or context.attn_metadata is None:
        return torch.zeros_like(hidden_states)
    target_attention = self._target_attention()
    target_wrapper = target_attention.dsa_attn
    target_impl = target_wrapper.dsa_attn.impl
    draft_impl = self.self_attn.dsa_attn.dsa_attn.impl
    layer_name = target_wrapper.dsa_attn.layer_name
    if hasattr(target_impl, "_get_layer_metadata"):
        layer_metadata = target_impl._get_layer_metadata(
            layer_name, context.attn_metadata
        )
    else:
        layer_metadata = _metadata_for_target(
            target_wrapper, context.attn_metadata, self.compress_ratio
        )
    target_kv = _build_kv_cache(target_wrapper, context)
    _, swa_cache, _, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
        target_kv, self.compress_ratio
    )
    common = layer_metadata.attention or layer_metadata.swa
    req = _request_metadata(common)
    swa_req = _request_metadata(layer_metadata.swa)
    num_tokens = hidden_states.shape[0]
    cos = req.cos[layer_name][:num_tokens]
    sin = req.sin[layer_name][:num_tokens]
    guard_mode = os.environ.get("ORTHRUS_VERIFY_CACHE_CHECKSUM", "0")
    debug = guard_mode in {"1", "full"}
    batch = num_tokens // BLOCK_SIZE
    full_check = guard_mode == "full" and batch not in self._full_checked_batches
    checksums = []
    if debug:
        torch.npu.synchronize()
        anchors = positions.view(batch, BLOCK_SIZE)[:, 0]
        if not torch.equal(anchors, swa_req.seq_lens.to(anchors.dtype)):
            raise RuntimeError("Orthrus history length does not match anchor position")
        for index, cache in enumerate(target_kv):
            if torch.is_tensor(cache):
                checksums.append((index, cache_checksum(cache, full=full_check)))

    query, qr, qr_scale, local_kv = _qkv_readonly(
        draft_impl, hidden_states, cos, sin
    )
    batch = num_tokens // BLOCK_SIZE
    outputs, lses = self._history_segments(
        query, hidden_states, qr, qr_scale, target_kv, layer_metadata, layer_name,
        local_kv,
    )
    sink_lse = draft_impl.attn_sink.view(1, 1, -1, 1).expand(
        batch, BLOCK_SIZE, -1, -1
    )
    outputs.append(torch.zeros_like(outputs[0]))
    lses.append(sink_lse)
    merged = merge_attention_outputs(outputs, lses)
    merged = merged.reshape(num_tokens, query.shape[1], query.shape[2])
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        merged.unsqueeze(1),
        cos,
        -sin,
        rotary_mode="interleave",
        partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
    )
    output = torch.empty_like(hidden_states)
    draft_impl._forward_o_proj(merged, output)
    if debug:
        torch.npu.synchronize()
        for index, before in checksums:
            assert_cache_unchanged(before, target_kv[index], f"layer-{self.layer_idx}:{index}")
        if full_check:
            self._full_checked_batches.add(batch)
    instant("orthrus.dsv4.attention", {
        "layer": self.layer_idx, "compress_ratio": self.compress_ratio,
        "batch": batch, "block_size": BLOCK_SIZE,
        "query_shape": list(query.shape), "segments": len(outputs),
        "cache_guard": "full" if full_check else "sampled" if debug else "off",
        "checked_elements": sum(before.numel() for _, before in checksums),
    })
    return output
```

</details>


上面的 cache guard 是可选诊断开关，正常计算不依赖校验张量。代码在无真实 metadata 的初始化/profile 阶段返回占位输出；这不等于执行了一次有效 draft，真实执行必须走完整的 metadata/cache 路径。

Attention 回到 block 后，继续调用共享主模型的 mHC、外层 norm 和 MoE。以下方法把这段衔接也列全：只有中间的 Attention 走 diffusion 分支，外层参数仍来自目标层。


<details>
<summary>完整源码：Orthrus block 与共享 backbone 的衔接</summary>

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    target_layer: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states.clone()
    hidden_states, post, comb = target_layer.hc_pre(
        hidden_states,
        target_layer.hc_attn_fn,
        target_layer.hc_attn_scale,
        target_layer.hc_attn_base,
    )
    hidden_states = target_layer.input_layernorm(hidden_states)
    hidden_states = self._attention_readonly(positions, hidden_states)
    hidden_states = target_layer.hc_post(hidden_states, residual, post, comb)

    residual = hidden_states.clone()
    hidden_states, post, comb = target_layer.hc_pre(
        hidden_states,
        target_layer.hc_ffn_fn,
        target_layer.hc_ffn_scale,
        target_layer.hc_ffn_base,
    )
    hidden_states = target_layer.post_attention_layernorm(hidden_states)
    hidden_states = target_layer.mlp(hidden_states, input_ids)
    return target_layer.hc_post(hidden_states, residual, post, comb)
```

</details>


### 3.3 kernel 与引擎各负责什么

kernel/计算图负责正确读取地址、遵守可见范围、不执行未授权写回，并产生 Attention 输出。它收到目标 cache 指针后是否写内存，不能靠框架的“只读意图”自动保证，因此需要绕过 AR 的写回算子。

引擎负责分配与回收页、保证读写顺序、构造 anchor/MASK、绑定 head、传递候选概率以及验证后的提交/回滚。**被接受的 token 保留的是 verifier 算出的 target KV，不是把 draft KV 复制或转正。** 压缩器状态也属于目标状态，由目标路径维护。

对 PyPTO 的交付可以据此界定：输入是 diffusion 参数、整块 hidden/位置和只读目标历史视图，输出是本层 draft hidden；不把 verifier/scheduler 的职责塞进 Attention kernel，也不能让 draft 顺手执行 AR 的 cache 更新。

### 3.4 开发范围归纳

| 模块 | 已有可复用部分 | 需要实现的适配 |
| --- | --- | --- |
| 公共双向块 | DSpark 的整块可见、QK/PV、online-softmax；HCA 的状态合并 | 临时块独立输入、K/布局适配；贡献并入统一 softmax，sink 只计一次 |
| SWA | 分页滑窗读取、公共历史预取、Attention | 固定 anchor 历史；用全可见临时块替代逐 query 因果追加 |
| HCA | C128 读取、滑窗/压缩状态合并 | 固定完整压缩行数；加入临时块；绕过 draft compressor/写回并调整依赖 |
| CSA | query 变换、量化评分/Top-K、C4 分页 Attention | diffusion indexer 参数；两种历史长度分支统一 anchor 边界；跳过 key/cache 写回并加入临时块 |

这是四项计算适配，不是四个从零开发的基础算子。公共投影和 backbone 沿用现有数学，但仍要完成权重、shape、TP 和运行时接线。**本文的 Torch-NPU 组合已实现；对应的 PyPTO Orthrus 路径是开发需求，不能当作已交付能力。**

## 4. 完整 BF16 单层 Demo

这个 demo 不依赖 PyPTO，也不要求加载整个主模型：它构造一层 diffusion Attention，输入一个 K 行 hidden 块和模拟历史，依次运行 SWA、HCA、CSA，包含输入投影、indexer、Attention、sink 合并、逆 RoPE 和输出投影。它使用 Torch-NPU 的真实算子，不是用 Python 模拟 FIA 或 indexer。

完整源码分为三个现有文件，下面全部列出，不省略函数体：


<details>
<summary>完整源码 1/3：公共辅助模块 dsv4_hybrid.py</summary>

```python
"""Exact read-only attention primitives for the DSV4 Orthrus backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def front_pack_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move valid non-negative indices to the front of the last dimension."""
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    valid = indices >= 0
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    packed = torch.gather(indices, -1, order)
    packed_valid = torch.gather(valid, -1, order)
    return torch.where(packed_valid, packed, -1), packed_valid


def c4_history_indices(
    indices: torch.Tensor, history_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """QLI returns compressed row IDs, not original-token positions."""
    complete_rows = torch.div(history_lengths, 4, rounding_mode="floor")
    valid = (indices >= 0) & (indices < complete_rows[:, None, None])
    return front_pack_indices(torch.where(valid, indices, -1))


def merge_attention_segments(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention segments exactly.

    Every output is ``sum(exp(score) * value) / sum(exp(score))`` for one
    segment and every LSE is ``log(sum(exp(score)))`` with a final singleton
    dimension. Empty segments must be omitted by the caller.
    """
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    lse_stack = torch.stack([lse.float() for lse in logsumexp], dim=0)
    merged_lse = torch.logsumexp(lse_stack, dim=0)
    weights = torch.exp(lse_stack - merged_lse.unsqueeze(0))
    merged = sum(
        output.float() * weight
        for output, weight in zip(outputs, weights, strict=True)
    )
    return merged.to(outputs[0].dtype), merged_lse


def merge_attention_outputs(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Merge segmented attention outputs using the native NPU update op."""
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if len(shape) < 2 or lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    if outputs[0].device.type == "npu":
        import torch_npu

        head_dim = shape[-1]
        merged, _ = torch_npu.npu_attention_update(
            [lse.float().reshape(-1) for lse in logsumexp],
            [output.reshape(-1, head_dim) for output in outputs],
            0,
        )
        return merged.view(shape)

    merged, _ = merge_attention_segments(outputs, logsumexp)
    return merged


def tail_logical_indices(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """Build front-packed logical indices for each request's trailing window."""
    if seq_lens.ndim != 1 or width <= 0:
        raise ValueError("seq_lens must be rank one and width must be positive")
    lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, width))
    starts = seq_lens - lengths
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = starts[:, None] + offsets[None, :]
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1)


def history_and_block_mask(
    history_lengths: torch.Tensor, history_width: int, block_width: int
) -> torch.Tensor:
    """Mask padded history, while all queries see the entire temporary block."""
    offsets = torch.arange(history_width + block_width, device=history_lengths.device)
    masked = (offsets[None, :] >= history_lengths[:, None]) & (offsets[None, :] < history_width)
    return masked[:, None, None, :].expand(-1, 1, block_width, -1).contiguous()


def compressed_logical_indices(
    seq_lens: torch.Tensor, ratio: int, maximum: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete C4/C128 cache rows and their per-request lengths.

    DSV4 keeps an incomplete compression group in the target state cache. It
    becomes a readable compressed KV row only after ``ratio`` source tokens.
    """
    if ratio not in (4, 128):
        raise ValueError("compressed history ratio must be 4 or 128")
    lengths = torch.div(seq_lens, ratio, rounding_mode="floor")
    width = int(maximum or max(int(lengths.max().item()), 1))
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = offsets[None, :].expand(seq_lens.shape[0], -1)
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1), lengths


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_indices: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Gather logical token indices from ``[blocks, block, heads, dim]`` cache.

    ``logical_indices`` may be ``[batch, tokens]`` or
    ``[batch, queries, tokens]``. Invalid ``-1`` entries are returned as zero;
    their lengths must also be supplied to the attention kernel.
    """
    if cache.ndim not in (3, 4):
        raise ValueError(f"unsupported paged cache shape: {tuple(cache.shape)}")
    if block_table.ndim != 2 or logical_indices.ndim not in (2, 3):
        raise ValueError("block_table and logical_indices have invalid rank")
    if block_table.shape[0] != logical_indices.shape[0]:
        raise ValueError("cache gather batch dimensions differ")
    block_size = cache.shape[1]
    valid = logical_indices >= 0
    safe = logical_indices.clamp_min(0).long()
    logical_blocks = torch.div(safe, block_size, rounding_mode="floor")
    if validate and logical_blocks.numel():
        if int(logical_blocks.max().item()) >= block_table.shape[1]:
            raise ValueError("logical index exceeds block table capacity")
    batch_shape = logical_indices.shape
    batch_ids = torch.arange(
        batch_shape[0], device=logical_indices.device, dtype=torch.long
    ).view((batch_shape[0],) + (1,) * (logical_indices.ndim - 1))
    physical_blocks = block_table[batch_ids, logical_blocks]
    if validate and torch.any(valid & (physical_blocks < 0)):
        raise ValueError("logical index maps to an unallocated cache block")
    physical_blocks = physical_blocks.clamp_min(0).long()
    offsets = safe % block_size
    if logical_indices.ndim == 3 and logical_indices.shape[0] > 1 and cache.is_contiguous():
        # C4 has a different sparse list per query. A single row index avoids
        # materializing the much larger two-index gather on the NPU.
        rows = physical_blocks * block_size + offsets
        gathered = cache.flatten(0, 1).index_select(0, rows.flatten()).view(
            *logical_indices.shape, *cache.shape[2:])
        mask = ~valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
        return gathered.masked_fill(mask, 0)
    gathered = cache[physical_blocks, offsets]
    mask = valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def cache_checksum(cache: torch.Tensor, *, full: bool = False) -> torch.Tensor:
    """Snapshot a bounded sample, or all elements for explicit debug checks."""
    flat = cache.reshape(-1)
    if full:
        return flat.clone()
    max_samples = 65_536
    stride = max((flat.numel() + max_samples - 1) // max_samples, 1)
    return flat[::stride][:max_samples].clone()


def assert_cache_unchanged(
    before: torch.Tensor, cache: torch.Tensor, name: str
) -> None:
    after = cache.reshape(-1) if before.numel() == cache.numel() else cache_checksum(cache)
    equal = before == after
    if before.is_floating_point():
        equal = equal | (torch.isnan(before) & torch.isnan(after))
    if not bool(torch.all(equal).item()):
        changed = ~equal
        changed_count = int(changed.sum().item())
        finite = changed & torch.isfinite(before) & torch.isfinite(after)
        max_abs = 0.0
        if bool(torch.any(finite).item()):
            max_abs = float(
                (before[finite].float() - after[finite].float()).abs().max().item()
            )
        raise RuntimeError(
            f"Orthrus draft modified target cache {name}: "
            f"sample_changed={changed_count}/{before.numel()}, max_abs={max_abs}"
        )
```

</details>


<details>
<summary>完整源码 2/3：单层模型 torch_npu_dsv4_layer.py</summary>

```python
"""使用 Torch-NPU 算子实现一层 Orthrus-DSV4 Attention

Orthrus 让 K 个 draft 位置都能读取主模型已接受的历史，以及临时 draft 块内
的全部位置：

    draft hidden states -> diffusion Q/KV 投影
                        -> 主模型历史 + 双向 draft 块
                        -> LSE 合并 -> diffusion 输出投影

历史张量模拟只读的主模型 cache。本模块参照 ``dsv4.py``，但不依赖 vLLM
metadata。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch_npu
from torch import nn

from .dsv4_hybrid import merge_attention_segments


AttentionKind = Literal["swa", "hca", "csa"]


@dataclass(frozen=True)
class NativeDSV4Config:
    """一层 diffusion attention 所需的 DSV4 结构参数。"""

    # 主 Attention 和 MLA 风格低秩投影的维度。
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    # CSA indexer 的维度以及 SWA 历史窗口大小。
    index_heads: int
    index_head_dim: int
    index_topk: int
    window_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    @classmethod
    def dsv4_flash(cls) -> "NativeDSV4Config":
        """DeepSeek-V4-Flash-0731 模型配置中的真实维度。"""
        return cls(
            hidden_size=4096,
            num_heads=64,
            head_dim=512,
            rope_head_dim=64,
            q_lora_rank=1024,
            o_lora_rank=1024,
            o_groups=8,
            index_heads=64,
            index_head_dim=128,
            index_topk=512,
            window_size=128,
        )

    def validate(self) -> None:
        if self.head_dim < self.rope_head_dim or self.index_head_dim < self.rope_head_dim:
            raise ValueError("RoPE dimensions exceed an attention head")
        if self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be even")
        if self.num_heads % self.o_groups:
            raise ValueError("num_heads must be divisible by o_groups")


@dataclass(frozen=True)
class SyntheticTargetHistory:
    """只读主模型 cache：原始 KV、HCA C128、CSA C4 和 index key。"""

    raw_kv: torch.Tensor
    c4_kv: torch.Tensor
    c128_kv: torch.Tensor
    index_key_pages: torch.Tensor
    index_block_table: torch.Tensor
    index_key_lengths: torch.Tensor


@dataclass(frozen=True)
class NativeLayerOutput:
    """临时 draft 输出；CSA 还返回选中的历史位置。"""

    output: torch.Tensor
    topk_indices: torch.Tensor | None


def make_synthetic_history(
    config: NativeDSV4Config,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    c4_length: int | None = None,
    c128_length: int = 16,
    index_block_size: int = 128,
) -> SyntheticTargetHistory:
    """构造布局与 DSV4 兼容的模拟 cache。"""
    c4_length = c4_length or max(config.index_topk + index_block_size, index_block_size)
    if c4_length < config.index_topk:
        raise ValueError("C4 history must cover index_topk")
    pages = math.ceil(c4_length / index_block_size)
    padded_c4 = pages * index_block_size

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype) * 0.02

    raw_kv = randn(batch, config.window_size, 1, config.head_dim)
    c4_kv = randn(batch, c4_length, 1, config.head_dim)
    c128_kv = randn(batch, c128_length, 1, config.head_dim)
    index_key_pages = randn(
        batch * pages, index_block_size, 1, config.index_head_dim
    )
    index_block_table = torch.arange(
        batch * pages, device=device, dtype=torch.int32
    ).view(batch, pages)
    index_key_lengths = torch.full(
        (batch,), c4_length, device=device, dtype=torch.int32
    )
    if padded_c4 != c4_length:
        # 分页 index cache 保留 padding，但有效长度会排除这些位置。
        index_key_pages = index_key_pages.contiguous()
    return SyntheticTargetHistory(
        raw_kv=raw_kv,
        c4_kv=c4_kv,
        c128_kv=c128_kv,
        index_key_pages=index_key_pages,
        index_block_table=index_block_table,
        index_key_lengths=index_key_lengths,
    )


def _normalize_fia_lse(lse: torch.Tensor, batch: int, queries: int) -> torch.Tensor:
    if lse.ndim == 4 and lse.shape[0] == batch and lse.shape[2] == queries:
        return lse.transpose(1, 2)
    if lse.ndim == 4 and lse.shape[:2] == (batch, queries):
        return lse
    raise RuntimeError(f"unexpected FIA LSE shape: {tuple(lse.shape)}")


def _native_fia_segment(
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_lengths: list[int],
    key_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算一段历史或块内 Attention，并返回输出和 LSE。"""
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        key_value.contiguous(),
        key_value.contiguous(),
        num_heads=query.shape[2],
        num_key_value_heads=key_value.shape[2],
        input_layout="BSND",
        sparse_mode=0,
        scale=query.shape[-1] ** -0.5,
        atten_mask=None,
        block_table=None,
        block_size=0,
        actual_seq_lengths=query_lengths,
        actual_seq_lengths_kv=key_lengths,
        softmax_lse_flag=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("FIA did not return output and softmax LSE")
    output, lse = result[:2]
    return output, _normalize_fia_lse(lse, query.shape[0], query.shape[1])


class TorchNPUOrthrusDSV4Attention(nn.Module):
    """DSV4 单层中的 Orthrus diffusion attention 分支。"""

    def __init__(
        self,
        config: NativeDSV4Config,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        factory = {"device": device, "dtype": dtype}

        # ORTHRUS 改动：增加独立于 AR 分支的可训练 diffusion Q 投影。
        self.wq_a = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False, **factory
        )
        self.q_norm = nn.Parameter(torch.ones(config.q_lora_rank, **factory))
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.num_heads * config.head_dim,
            bias=False,
            **factory,
        )

        # ORTHRUS 改动：生成临时块 K/V，不写入主模型 cache。
        self.wkv = nn.Linear(
            config.hidden_size, config.head_dim, bias=False, **factory
        )
        self.kv_norm = nn.Parameter(torch.ones(config.head_dim, **factory))

        # ORTHRUS 改动：为每个 draft 位置生成 CSA 检索 query。
        self.index_wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_heads * config.index_head_dim,
            bias=False,
            **factory,
        )
        self.index_weights = nn.Linear(
            config.hidden_size, config.index_heads, bias=False, **factory
        )

        # DSV4 分组输出投影，将结果映射回共享 backbone 的 hidden size。
        group_dim = config.num_heads * config.head_dim // config.o_groups
        self.wo_a = nn.Parameter(
            torch.empty(config.o_groups, group_dim, config.o_lora_rank, **factory)
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
            **factory,
        )
        self.attn_sink = nn.Parameter(
            torch.zeros(config.num_heads, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            "head_norm", torch.ones(config.head_dim, **factory), persistent=False
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.wq_a,
            self.wq_b,
            self.wkv,
            self.index_wq_b,
            self.index_weights,
            self.wo_b,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.wo_a, mean=0.0, std=0.01)

    def _rotary_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0,
                    config.rope_head_dim,
                    2,
                    device=positions.device,
                    dtype=torch.float32,
                )
                / config.rope_head_dim
            )
        )
        angles = positions.reshape(-1, 1).float() * inv_freq.view(1, -1)
        angles = angles.repeat_interleave(2, dim=-1)
        shape = (positions.numel(), 1, 1, config.rope_head_dim)
        return angles.cos().view(shape), angles.sin().view(shape)

    def _partial_rope(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        rope_dim = self.config.rope_head_dim
        flat = value.reshape(-1, value.shape[-2], value.shape[-1])
        nope, rope = flat[..., :-rope_dim], flat[..., -rope_dim:]
        rotated = torch_npu.npu_rotary_mul(
            rope.reshape(rope.shape[0], rope.shape[1], 1, rope_dim).float(),
            cos,
            -sin if inverse else sin,
            rotary_mode="interleave",
        ).to(value.dtype)
        return torch.cat((nope, rotated.reshape_as(rope)), dim=-1).view_as(value)

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """为 K 个 draft 位置生成 diffusion Q 和块内 K/V。"""
        config = self.config
        batch, block, _ = hidden_states.shape
        q_a = self.wq_a(hidden_states)
        qr, _ = torch_npu.npu_rms_norm(
            q_a, self.q_norm, epsilon=config.rms_norm_eps
        )
        query = self.wq_b(qr).view(
            batch, block, config.num_heads, config.head_dim
        )
        query, _ = torch_npu.npu_rms_norm(
            query, self.head_norm, epsilon=config.rms_norm_eps
        )
        local_kv = self.wkv(hidden_states)
        local_kv, _ = torch_npu.npu_rms_norm(
            local_kv, self.kv_norm, epsilon=config.rms_norm_eps
        )
        local_kv = local_kv.view(batch, block, 1, config.head_dim)
        cos, sin = self._rotary_tables(positions)
        return (
            self._partial_rope(query, cos, sin),
            self._partial_rope(local_kv, cos, sin),
            qr,
            hidden_states,
        )

    def _native_indexer(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        history: SyntheticTargetHistory,
    ) -> torch.Tensor:
        """使用 diffusion query 为每个 draft 位置选择 C4 历史。"""
        config = self.config
        batch, block, _ = hidden_states.shape
        query = self.index_wq_b(qr).view(
            batch, block, config.index_heads, config.index_head_dim
        )
        cos, sin = self._rotary_tables(positions)
        query = self._partial_rope(query, cos, sin)
        weights = self.index_weights(hidden_states) * (
            config.index_head_dim**-0.5 * config.index_heads**-0.5
        )
        query_lengths = torch.full(
            (batch,), block, dtype=torch.int32, device=hidden_states.device
        )
        result = torch_npu.npu_lightning_indexer(
            query=query,
            key=history.index_key_pages,
            weights=weights,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_key=history.index_key_lengths,
            block_table=history.index_block_table,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=config.index_topk,
            sparse_mode=0,
        )
        indices = result[0]
        if indices.ndim == 4:
            if indices.shape[2] != 1:
                raise RuntimeError(f"unexpected indexer output: {tuple(indices.shape)}")
            indices = indices[:, :, 0]
        if indices.shape != (batch, block, config.index_topk):
            raise RuntimeError(f"unexpected indexer output: {tuple(indices.shape)}")
        return indices.long()

    @staticmethod
    def _gather_c4(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """从 C4 cache 中取出每个 draft query 选中的历史行。"""
        batch, queries, selected = indices.shape
        source = cache[:, None].expand(-1, queries, -1, -1, -1)
        gather_index = indices[..., None, None].expand(
            batch, queries, selected, cache.shape[2], cache.shape[3]
        )
        return torch.gather(source, 2, gather_index)

    def _output_projection(
        self,
        attention: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """执行 DSV4 inverse partial RoPE 和分组输出投影。"""
        config = self.config
        batch, block, _, _ = attention.shape
        cos, sin = self._rotary_tables(positions)
        attention = self._partial_rope(attention, cos, sin, inverse=True)
        grouped = attention.reshape(
            batch * block,
            config.o_groups,
            config.num_heads * config.head_dim // config.o_groups,
        )
        low_rank = torch_npu.npu_transpose_batchmatmul(
            grouped,
            self.wo_a,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
        return self.wo_b(low_rank.reshape(batch, block, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        history: SyntheticTargetHistory,
        kind: AttentionKind,
    ) -> NativeLayerOutput:
        """为 K 位置 draft 块执行一层 SWA、HCA 或 CSA。"""
        config = self.config
        batch, block, hidden = hidden_states.shape
        if hidden != config.hidden_size or positions.shape != (batch, block):
            raise ValueError("hidden_states or positions have incompatible shapes")
        query, local_kv, qr, hidden_states = self._project(hidden_states, positions)
        query_lengths = [block] * batch
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []

        # K 个 query 读取同一份已接受历史，但不修改它。
        raw_out, raw_lse = _native_fia_segment(
            query,
            history.raw_kv,
            query_lengths,
            [history.raw_kv.shape[1]] * batch,
        )
        outputs.append(raw_out)
        lses.append(raw_lse)

        topk_indices: torch.Tensor | None = None
        if kind == "hca":
            # HCA 额外读取 C128 压缩历史。
            compressed_out, compressed_lse = _native_fia_segment(
                query,
                history.c128_kv,
                query_lengths,
                [history.c128_kv.shape[1]] * batch,
            )
            outputs.append(compressed_out)
            lses.append(compressed_lse)
        elif kind == "csa":
            # CSA 为每个 draft 位置选择并读取 C4 历史。
            topk_indices = self._native_indexer(
                qr, hidden_states, positions, history
            )
            selected_c4 = self._gather_c4(history.c4_kv, topk_indices)
            flat_query = query.reshape(
                batch * block, 1, config.num_heads, config.head_dim
            )
            flat_c4 = selected_c4.reshape(
                batch * block, config.index_topk, 1, config.head_dim
            )
            compressed_out, compressed_lse = _native_fia_segment(
                flat_query,
                flat_c4,
                [1] * (batch * block),
                [config.index_topk] * (batch * block),
            )
            outputs.append(compressed_out.view_as(query))
            lses.append(
                compressed_lse.view(batch, block, config.num_heads, 1)
            )
        elif kind != "swa":
            raise ValueError(f"unknown attention kind: {kind}")

        # ORTHRUS 改动：不加因果 mask，K 个 draft 位置可以互相读取。
        local_out, local_lse = _native_fia_segment(
            query, local_kv, query_lengths, [block] * batch
        )
        outputs.append(local_out)
        lses.append(local_lse)

        # DSV4 attention sink 与其他部分共享 softmax 分母。
        outputs.append(torch.zeros_like(local_out))
        lses.append(
            self.attn_sink.view(1, 1, -1, 1).expand(batch, block, -1, -1)
        )
        # ORTHRUS 改动：用 LSE 精确合并只读历史与双向 draft 块。
        attention, _ = merge_attention_segments(outputs, lses)

        # 候选验证以及 cache 提交/回滚由推理引擎负责。
        output = self._output_projection(attention, positions)
        return NativeLayerOutput(
            output=output,
            topk_indices=topk_indices,
        )
```

</details>


<details>
<summary>完整源码 3/3：运行脚本 demo_torch_npu_dsv4_layer.py</summary>

```python
#!/usr/bin/env python3
"""使用 Torch-NPU 算子运行一层 Orthrus-DSV4 diffusion attention。"""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from blockspec.vllm_ascend.torch_npu_dsv4_layer import (
    NativeDSV4Config,
    TorchNPUOrthrusDSV4Attention,
    make_synthetic_history,
)


def run_demo(config: NativeDSV4Config, block_size: int) -> dict:
    """对一个 Orthrus draft 块分别运行 SWA、HCA 和 CSA。"""
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    batch = 1
    layer = TorchNPUOrthrusDSV4Attention(config, device=device, dtype=dtype)
    # 用随机输入模拟共享 backbone 生成的 K 位置块。
    hidden = torch.randn(
        batch, block_size, config.hidden_size, device=device, dtype=dtype
    ) * 0.02
    positions = torch.arange(
        1024, 1024 + block_size, device=device, dtype=torch.long
    ).view(batch, block_size)
    # 用模拟 cache 代替主模型已经接受的历史。
    history = make_synthetic_history(
        config,
        batch=batch,
        device=device,
        dtype=dtype,
    )
    rows = []
    with torch.inference_mode():
        # DSV4 混合使用三种层，因此 Orthrus 需要全部支持。
        for kind in ("swa", "hca", "csa"):
            result = layer(hidden, positions, history, kind)
            rows.append(
                {
                    "kind": kind,
                    "output_shape": list(result.output.shape),
                    "topk_shape": (
                        list(result.topk_indices.shape)
                        if result.topk_indices is not None
                        else None
                    ),
                }
            )
    torch.npu.synchronize()
    return {
        "config": {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "index_heads": config.index_heads,
            "index_head_dim": config.index_head_dim,
            "index_topk": config.index_topk,
            "window_size": config.window_size,
            "block_size": block_size,
        },
        "paths": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261027)

    result = run_demo(NativeDSV4Config.dsv4_flash(), args.block_size)
    print(
        json.dumps(
            {
                "device": "npu:0",
                "dtype": "torch.bfloat16",
                "native_ops": [
                    "torch_npu.npu_rms_norm",
                    "torch_npu.npu_rotary_mul",
                    "torch_npu.npu_lightning_indexer",
                    "torch_npu.npu_fused_infer_attention_score",
                    "torch_npu.npu_transpose_batchmatmul",
                ],
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

</details>


在匹配的 Torch-NPU/CANN 环境中，进入仓库的 `online-speculation` 目录后运行：

```bash
PYTHONPATH=src python scripts/demo_torch_npu_dsv4_layer.py --block-size 32
```

运行脚本默认使用 `npu:0`，输出三条路径的输出形状和 CSA 的 Top-K 形状。仓库路径分别为：
[公共模块](../src/blockspec/vllm_ascend/dsv4_hybrid.py)、
[单层模型](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py)、
[运行脚本](../scripts/demo_torch_npu_dsv4_layer.py)。

**demo 的边界。** 模型维度取自 DSV4-Flash，但权重、历史内容和输入是随机构造的；RoPE 表使用演示配置，不是生产长上下文配置。模拟历史长度和位置用于覆盖接口，不是某条真实序列的 checkpoint 回放。demo 不包含主模型 compressor、共享 MoE、调度器或验证器，也不能据此判断候选质量、真实长上下文正确性或加速比。

为了阅读直观，demo 分别计算历史与块内 Attention，并将 C4 选中行收集成连续张量。服务路径则使用联合 SWA/块 FIA、量化 QLI、分页 sparse Attention 和原生合并。开发 PyPTO 时以第 2 节数学语义为准，用 demo 理解完整数据流，用服务路径对照实际分页和量化接口，不需要照搬 demo 的中间张量物化方式。

当前服务契约固定 K=32；Attention 数学本身不要求 K=32。调整 K 不必重新发明算法，但需要同步 head/调度契约、编译形状和临时空间，并重新验证。

## 5. 交付与验收边界

完整推理代码还包含 [proposer](../src/blockspec/vllm_ascend/dsv4_proposer.py) 和 [固定版本接入补丁](../runtime/vllm_ascend_v0_26/README.md)，负责把本文计算放进 vLLM-Ascend 的模型执行与验证流程；配置和启动方法见 [部署说明](DSV4_DEPLOYMENT.md)。本文展示的是计算实现，不将单层 demo 等同于完整服务。

已有实测覆盖 A2 算子对照、四层 W8A8 TP2、全部候选接受长度、概率传递、拒绝恢复和物理页复用；抢占恢复使用受控分配失败触发。完整 43 层 TP8、graph、1M 上下文和持续负载尚未完成验收。初始化 head 只能检查推理接线，接受率和加速收益要用训练好的 head 评估，见 [实测记录](DSV4_VALIDATION_20261028.md)。

PyPTO 交付按三层验证即可明确责任：先检查四个计算模块的输出、LSE/归一化和有效行选择；再检查整层投影、分页、空历史、多请求以及目标 cache/压缩状态不变；最后接入引擎检查接受、拒绝和恢复。数值对照应固定权重、输入、精度和布局，不能只检查“能运行”或输出形状。完整检查入口见 [验证说明](DSV4_VALIDATION_RUNBOOK.md)。
