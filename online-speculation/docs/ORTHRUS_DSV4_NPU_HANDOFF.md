# Orthrus-DSV4 NPU 推理适配与算子设计

## 1. 背景与适配目标

大模型的自回归推理需要依次生成 token：当前 token 确定后，才能继续计算下一个位置。投机解码通过草稿分支提前提出多个候选，再由主模型批量验证，尝试减少主模型逐 token 执行的轮数。能否获得加速，既取决于候选接受率，也取决于草稿生成、验证以及状态管理的额外开销。

Orthrus 采用共享主干的草稿结构，在模型各层增加独立的 diffusion Attention，通过块内双向 Attention 一次预测多个位置。在本文的 DSV4 适配中，embedding、外层 norm、mHC、MoE 和 LM head 与主模型共享，因此它既不是另起一个完整的小模型，也不是只在最终 hidden 后附加一层预测头。草稿分支仍执行各层计算，但不复制整套主干参数，并且能够读取主模型已经生成的历史 KV。

将这一结构用于 DeepSeek-V4，需要处理其混合 Attention：SWA 读取滑动窗口，HCA 还读取 C128 压缩历史，CSA 则通过 indexer 选择 C4 压缩历史。适配工作的重点，是在保留三类历史表示和检索规则的前提下，让 diffusion query 同时读取主模型历史与临时双向块，并避免草稿计算改变目标模型状态。

本文聚焦 DeepSeek-V4-Flash-0731 在 Ascend A2 上的推理计算与引擎接入，不包含训练和在线更新。各模块以固定 v0.26 环境下的服务代码说明实现，单层 demo 则用于独立演示计算过程。当前 head 为初始化版本，只用于链路验证，不代表训练后的接受率或加速收益。

### 1.1 草稿分支与主模型的关系

每轮草稿输入由一个已知的 anchor token 和多个 MASK token 组成。anchor 是本轮预测的起点；MASK 是尚未生成位置的占位符，不是答案 token。当前服务使用块长 $K=32$，即一个 anchor 和 31 个待预测位置。

设 anchor 的绝对位置为 $a$，主模型已经提供位置 $0,\ldots,a-1$ 的历史状态。草稿块中各位置使用自己的 diffusion query，读取这段历史对应的滑窗或压缩表示，同时读取整个临时块。块内位置交换的是当前层的中间特征，因此不需要先等待其他位置生成 token。

```text
anchor + MASK × 31
        |
共享 embedding
        |
43 层 block，每层执行：
  共享 mHC / 外层 norm
        |
  diffusion Q/KV 投影 -> K 行 query + 临时块 KV
        |
  按该层类型选择一条历史路径：
    SWA：滑窗历史
    HCA：滑窗历史 + C128 压缩历史
    CSA：滑窗历史 + indexer 选中的 C4 历史
        |
  历史与整个临时块共同参与 Attention，sink 只计一次
        |
  diffusion 输出投影
        |
  共享 mHC / 外层 norm / MoE
        |
共享 LM head -> 候选 token 及其概率
        |
主模型因果验证 -> 接受前缀或拒绝补采样 -> 更新目标模型状态
```

每层独立持有 diffusion Q/KV/O 投影、Attention 内部 norm、sink；CSA 层还持有自己的 indexer query/weight 参数。共享的是主干权重和已生成的目标历史，而不是本轮全部激活。由于输入与 Attention 可见性不同，draft 仍需计算自己的中间结果。

草稿阶段产生的 KV 仅用于当前块。主模型验证候选时，会按自身参数和因果规则重新计算 target KV；接受后保留的是这些 target KV，而不是将 draft KV 转为主模型历史。两者的参数和上下文不同，不能相互替代。

### 1.2 现有推理路径的适配边界

从数学上看，Orthrus 的双向计算仍是 scaled dot-product Attention，可见范围可以由 mask 表达。但现有 PyPTO DSV4 路径并非一个接收完整 Q/K/V 和任意 mask 的通用 SDPA 接口：它把分页读取、因果边界、压缩器更新、cache 写回以及部分后处理编排成了专门的 decode 计算图。

因此，适配不仅要改变块内 mask，还要改变数据从哪里读取、哪些历史行有效、临时块放在哪里，以及是否执行 AR 状态更新。这些变化不要求重新设计矩阵乘、softmax 或 Top-K，却需要修改已有融合路径的输入和编排。

Attention 侧可分为四个逻辑模块：**公共双向块、SWA、HCA、CSA**。公共块供三类历史路径使用，可以融合到对应计算中；四个逻辑模块不等于必须新增四个独立 kernel。下文以各模块的计算需求为单位，同时说明原有语义、目标语义、PyPTO 实现基础和现有 Torch-NPU 实现。

## 2. Attention 模块设计

### 2.1 双向块 Attention 与联合归一化

#### 2.1.1 可见性变化与计算定义

普通自回归模型即使一次处理多个位置，第 $i$ 个位置也只能读取本块中 $j\le i$ 的位置，否则会看到未来答案。Orthrus 的块内输入则是 anchor 和 MASK，不包含未来答案，因此可以让每个位置读取整个块，以并行生成候选。

仅考虑当前块，原有因果 mask 与 Orthrus mask 的区别为：

$$
M_{ij}^{AR}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i,
\end{cases}
\qquad
M_{ij}^{D}=0,\quad 0\le i,j<K.
\tag{1}
$$

$M$ 是加在 Attention 分数上的 additive mask，零表示可见，$-\infty$ 表示被屏蔽；上标 $AR$ 和 $D$ 分别表示主模型与 diffusion 分支。这个变化只涉及临时块，历史仍然限定在 anchor 之前，不能读取其他预测块或未来目标状态。

为统一三类历史路径，定义单个 head 的 Attention：

$$
\mathcal A(q,C;s)=
\frac{\sum_{c_j\in C}\exp(\alpha q c_j^\top)c_j}
{\exp(s)+\sum_{c_j\in C}\exp(\alpha q c_j^\top)}.
\tag{2}
$$

$q$ 是 query，$C$ 是该 query 可见的 KV 行集合，$\alpha$ 是模型规定的缩放，$s$ 是 Attention sink 的 logit。DSV4 在这里使用同一 shared KV 向量 $c_j$ 作为 key 和 value；sink 等价于一个 value 为零的额外项，只增加分母。公式按反量化后的实数表示，低精度实现仍需检查数值误差。

设本轮临时块为 $B^D=\{c_0^D,\ldots,c_{K-1}^D\}$。若只计算块内部分，则第 $i$ 行返回：

$$
O_{B,i}=\mathcal A(q_i^D,B^D;-\infty),\qquad
L_{B,i}=\log\sum_{j=0}^{K-1}
\exp\!\left(\alpha q_i^D(c_j^D)^\top\right).
\tag{3}
$$

这里不在块内单独计入 sink，因为块内输出还要与历史共同归一化。$O_{B,i}$ 是局部归一化输出，$L_{B,i}$ 是该部分的 log-sum-exp，简称 LSE。

历史和块内 Attention 可以融合计算，也可以分别调用现成算子。分开计算时，不能直接相加各自的输出：每部分都除以了不同的 softmax 分母，需要根据其 LSE 恢复相对权重。设第 $r$ 段返回 $O_r,L_r$，共有 $R$ 段，则：

$$
L=\operatorname{logsumexp}(s,L_1,\ldots,L_R),\qquad
O=\sum_{r=1}^{R}\exp(L_r-L)O_r.
\tag{4}
$$

sink 按 $(O_s,L_s)=(0,s)$ 加入一次。空历史段使用 $O_r=0,L_r=-\infty$，对结果没有贡献。在实数计算下，该式与把所有可见 KV 一起代入式 (2) 等价；它是调用多个现成算子时的组合方法，不要求 PyPTO 也必须拆成多个 kernel。

#### 2.1.2 PyPTO 实现基础与适配方案

`dspark_attention.py` 已经具备整块双向可见的计算。它先将当前 draft KV 按 slot mapping 写入 DSpark 的私有 cache，再从同一 cache 收集历史滑窗与整个 draft 块。其块内 mask 与 Orthrus 的目标相同，区别主要在参数来源、历史来源以及临时块的存放方式。

该实现采用分块 QK、softmax、PV：矩阵单元计算 QK/PV，向量单元维护最大值、指数和与累加分子，通过流水线同步衔接。块宽取自 `DSPARK_SPEC_TOKENS`，相应的可见行数和临时空间由配置生成。适配 Orthrus 时，可以复用这些计算，但应让临时块直接作为输入，不再依赖“写入私有 cache 后再读取”的流程，并为 K=32 及相应 batch/布局配置临时空间。

归一化合并也有现成基础。PyPTO HCA 已经维护各段 online-softmax 状态 $(m_r,\ell_r,u_r)$，其中 $m_r$ 是最大分数，$\ell_r$ 是减去最大值后的指数和，$u_r$ 是同样缩放的加权 value 分子：

$$
\ell_r=\sum_j e^{z_{rj}-m_r},\qquad
u_r=\sum_j e^{z_{rj}-m_r}v_{rj}.
\tag{5}
$$

$z_{rj}$ 和 $v_{rj}$ 分别是第 $r$ 段第 $j$ 行的分数和 value。现有 HCA 将滑窗与压缩历史合并；支持 Orthrus 后，只需把临时块也纳入同一组状态：

$$
m=\max(s,m_1,\ldots,m_R),\qquad
O=\frac{\sum_r e^{m_r-m}u_r}
{e^{s-m}+\sum_r e^{m_r-m}\ell_r}.
\tag{6}
$$

非空段满足 $L_r=m_r+\log\ell_r$、$O_r=u_r/\ell_r$，因此式 (4) 和式 (6) 是同一归一化的两种接口形式。空段置 $m_r=-\infty,\ell_r=0,u_r=0$。PyPTO 可以直接复用状态合并，不需要先产生 LSE 再调用额外合并 kernel；关键是全部贡献合并后才执行输出后处理，且 sink 不重复计算。

#### 2.1.3 Torch-NPU 实现

服务使用 `torch_npu.npu_fused_infer_attention_score`（FIA）计算 Attention 分段。query 整理为 `[batch, K, local_heads, dim]`；KV 可以是连续的“滑窗 + 临时块”，也可以是带页表的目标压缩历史。`sparse_mode=0` 配合显式 padding mask 和有效长度，表达 Orthrus 的非因果读取，而不是沿用 AR 的三角 mask。

下面是实际服务的完整 FIA 封装，包括分页布局转换、LSE 布局整理和空历史处理。空段必须返回零输出及 $-\infty$ LSE，不能让 padding 进入联合 softmax。

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

SWA 历史和临时块已在第 2.2 节合为一次 FIA，不再单独计算一遍块内 Attention。压缩历史按层类型另外返回输出与 LSE；调用方将 sink 追加为零输出的一段，再由下列函数调用原生 `npu_attention_update` 完成式 (4)。

`merge_attention_outputs` 在 NPU 上调用原生合并算子，并保留用于非 NPU 对照的 Torch 回退；完整定义见附录 A。

<details>
<summary>完整源码：分段输出与 LSE 合并</summary>

```python
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

PyPTO 可以直接沿用式 (6) 的 online-softmax 状态完成同一归一化，无需先输出 LSE 再调用独立 kernel。这里的 FIA/AttentionUpdate 组合，是利用已有 NPU 算子实现数学契约的服务路径，不是对 PyPTO 融合粒度的限制。

源码定位：[DSpark Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/dspark_attention.py)、[HCA 状态合并](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。

### 2.2 SWA：滑动窗口历史与临时块

#### 2.2.1 历史边界与计算定义

双向块解决当前 K 个位置之间的交互，但候选还需要已有上下文。SWA 提供最近一段原始 KV，因此它既是独立 SWA 层的历史来源，也是 HCA/CSA 路径的局部历史部分。

设 $W(t)$ 表示截至位置 $t$ 的最近至多 $w$ 行原始 KV，DSV4-Flash 的窗口宽度 $w=128$；$p$ 是 AR query 的绝对位置。原有计算与 Orthrus 的区别为：

| 原有 AR Attention | Orthrus-SWA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert B^D;s^D)$ |

$\Vert$ 表示两组 KV 共同参与 Attention，不要求物理拼接。AR 窗口随 query 位置移动，包含当前 query 对应的真实 KV；Orthrus 的 K 个 query 则读取同一个 anchor 前窗口，再读取整块临时 KV。即使第 $i$ 个 draft 位置更靠后，也不能按 AR 规则缩短它可读的 anchor 历史。

#### 2.2.2 PyPTO 实现基础与适配方案

`decode_sparse_attn_swa.py` 已经实现分页滑窗读取和 Attention。在多 query 的 AR 验证路径中，调用方先写入待验证 token 的 KV；Attention 为第一个 query 收集窗口，再为后续 query 追加新位置、移出过期位置。该优化避免重复收集整个窗口，但可见范围仍然是逐行因果的。

Orthrus 可以复用分页地址转换、预取和公共历史共享，但不能沿用上述逐行追加/滑移规则。窗口行号应由 anchor 前历史长度统一生成，临时块单独提供，每个 query 都读取完整块。历史不足窗口宽度时还需屏蔽 padding，否则填零的 KV 也会进入 softmax 分母。

cache 写入和 Attention 读取并不总在同一个函数内。现有滑窗子函数主要负责读取，写回由其调用链完成；适配时既要调整读取范围，也要让调用方绕过 draft 的 target KV 写回。仅修改 Attention mask 而保留 AR 写回编排，仍不满足只读要求。

#### 2.2.3 Torch-NPU 实现

服务直接读取主模型的分页 SWA cache。`tail_logical_indices` 根据每个请求的 anchor 前历史长度生成最近窗口的逻辑行号，`gather_paged_cache` 通过页表映射到物理位置，再收集最多 128 行历史。这里只展开短窗口，不展开整个长上下文。

以下是实际读取和 mask 构造的完整函数。无效行号以 `-1` 表示；布尔 mask 的 `True` 表示屏蔽，与式 (1) 的 additive mask 编码不同。所有 query 使用同一个历史有效长度。

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

得到滑窗后，`_swa_block_segment` 将其与 diffusion 投影生成的临时 KV 拼接，用一次 FIA 同时计算历史与块内 Attention，减少调用和后续合并次数。块内全量可见，只屏蔽历史 padding。

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

这一段尚未加入 sink。纯 SWA 层随后只需合并 sink；HCA/CSA 层还会加入压缩历史。PyPTO 可以直接在融合 kernel 中分别读取分页历史与临时块，避免物理 `cat`，但必须保持相同的可见性和全局 softmax。

源码定位：[PyPTO SWA](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py)。

### 2.3 HCA：C128 压缩历史 Attention

#### 2.3.1 压缩历史范围与计算定义

SWA 仅保存近处的原始上下文。HCA 在此基础上读取主模型按 ratio-128 生成的压缩行，使 query 能访问更长的历史。Orthrus 不需要对草稿块再运行一次目标 compressor，而是使用已经完成的压缩历史，将草稿块保持为未压缩的临时 KV。

记 $H_{128}(t)$ 为截至位置 $t$ 已完成的 C128 压缩行：

| 原有 AR Attention | Orthrus-HCA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_{128}(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_{128}(a-1)\Vert B^D;s^D)$ |

anchor 前有 $a$ 个 token，因此读取 $\lfloor a/128\rfloor$ 个已完成压缩行。未完成压缩组仍保存在目标模型的 compressor state 中，draft 不能补完或更新它。这里读取全部有效 C128 行，不通过 indexer 做 Top-K 截断。

#### 2.3.2 PyPTO 实现基础与适配方案

`decode_hca.py` 的正常 AR 路径执行投影、原始 KV 写回、ratio-128 压缩与状态更新，再启动 Attention。`decode_sparse_attn_hca.py` 分别处理滑窗和 C128 历史，合并各自的 online-softmax 状态，加入一次 sink，然后执行逆 RoPE 和输出布局整理。

因此，HCA 已有分页读取、压缩历史计算和联合归一化。Orthrus 的新增工作是把有效历史长度固定到 anchor 前、加入临时双向块，并绕过本轮草稿的 compressor 和 cache 写回。状态合并可以直接按式 (6) 扩展；应在全部贡献合并后做一次后处理，而不是分别处理各段再相加。

计算图依赖也需相应调整。原有 Attention 可能等待 AR 写回任务完成；跳过该任务后，要把依赖改成“目标历史可读、diffusion Q/KV 已就绪”，不能只删除写回调用而留下失效的同步关系。

#### 2.3.3 Torch-NPU 实现

服务直接使用目标模型的 C128 分页 cache。页大小满足当前 FIA 接口条件时，传入页表做直接分页 Attention；不满足时，枚举全部有效 C128 行，通过 `npu_sparse_flash_attention` 读取较小分页。sparse 接口在这里仅改变读取方式，不做 Top-K，也不减少 HCA 的历史范围。

默认选择由 `_c128_segment` 完成。显式配置 `ORTHRUS_C128_BACKEND=gather` 时，才退回连续展开后的 FIA；这是兼容和对照路径，不是服务默认采用的全部历史展开。下面给出选择逻辑与原生 sparse 调用的完整源码。

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

`_sparse_history_segment` 同时供 CSA 使用。其原生接口要求独立的 RoPE 参数，但 DSV4 的 512 维 query/KV 已包含 RoPE 分量，因此传入全零辅助 query，使额外点积项为零；`key_rope` 是目标 cache 的只读视图。算子返回的最大值和指数和通过 $L=m+\log\ell$ 转成 LSE，再与滑窗/块结果合并。

PyPTO 不需要模仿为适配现成接口而设置的零辅助参数，可以在已有 HCA 流水线中直接读取这些行并计算同一分数。需要保持的是完整 C128 可见性、目标压缩状态不变，以及包含临时块的正确归一化。

源码定位：[HCA 编排](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_hca.py)、[HCA Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。

### 2.4 CSA：C4 检索与稀疏 Attention

#### 2.4.1 Indexer 与 Attention 的计算定义

C4 的压缩率比 C128 低，同一上下文会保留更多压缩行。CSA 先由 indexer 为每个 query 选择相关历史，再对选中行计算 Attention。适配 Orthrus 时，历史 index key 和 C4 KV 仍来自主模型，query 及其检索权重改由 diffusion 分支生成，以对应本轮 hidden 和交付 head 中的 indexer 参数。

设 $k_u^I$ 为历史第 $u$ 行 index key，$q_{i,h}^{I,X}$ 为位置 $i$、head $h$ 的 indexer query，$w_{i,h}^{X}$ 为含模型缩放的 head 权重，$X\in\{AR,D\}$。两种路径共用同一评分形式：

$$
A_{i,u}^{X}=
\sum_{h=1}^{H_I}w_{i,h}^{X}
\operatorname{ReLU}\!\left(q_{i,h}^{I,X}(k_u^I)^\top\right).
\tag{7}
$$

$H_I$ 是 indexer head 数。变化在于 query/权重来源和有效历史范围：

| 原有 AR 检索 | Orthrus 检索 |
| --- | --- |
| $I_p^{AR}=\operatorname{TopIdx}_{k_I}\{A_{p,u}^{AR}:0\le u<\lfloor(p+1)/4\rfloor\}$ | $I_i^D=\operatorname{TopIdx}_{k_I}\{A_{i,u}^{D}:0\le u<\lfloor a/4\rfloor\}$ |

$k_I=512$ 是最多选取的压缩行数，与 draft 块长 $K=32$ 不同；历史不足时仅使用有效行。所有 draft query 的检索范围相同，但 query 不同，选中的行可以不同。当前临时块全量可见，不参与 Top-K。

记 $H_4[I]$ 为行号集合 $I$ 对应的 C4 KV，最终 Attention 为：

| 原有 AR Attention | Orthrus-CSA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_4[I_p^{AR}];s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_4[I_i^D]\Vert B^D;s^D)$ |

因此，CSA 需要独立 diffusion indexer 参数，但不需要重新设计评分公式或 Top-K 排序算法。参数替换与算子算法复用是两件不同的事。

#### 2.4.2 PyPTO 实现基础与适配方案

`decode_indexer.py` 已提供 indexer query 投影、RoPE、Hadamard 变换、量化评分和 Top-K 归并。`decode_csa.py` 还包含一个短历史分支：有效 C4 行不超过 512 时，直接枚举全部有效行，跳过评分排序。随后 `decode_sparse_attn_csa.py` 根据页表读取选中的压缩 KV，与滑窗一起做 Attention，并处理 sink 和逆 RoPE。

正常 AR 编排会同时更新原始 KV、C4 compressor、indexer key 和 scale cache。现有检索边界也根据各 query 的当前位置变化，而不是固定到一个共同 anchor。

Orthrus 可以复用评分、Top-K 归并和分页读取，但需要绑定 diffusion query/weight 投影，批量处理 K 行；完整评分与短历史枚举两个分支都应采用 anchor 前的有效长度。同时跳过草稿侧的 indexer compressor/key/scale 写回，仅运行 query 侧计算。最后将选中的 C4、滑窗和临时块纳入同一个 softmax。

#### 2.4.3 Torch-NPU 实现

服务使用 `npu_vllm_quant_lightning_indexer`（QLI）检索 C4 历史。低秩 query 经 diffusion indexer 投影、RoPE、Hadamard 和动态量化后送入算子；历史 key 及其 scale 直接读取主模型 cache。

配套 metadata 使用非因果模式，让 K 行 query 均读取完整 anchor 历史。若沿用 target 验证的因果 metadata，块内前几行会错误地少看历史。以下是实际调用的完整类方法，其中 `indexer` 是 diffusion 参数对象，`target_kv` 是只读目标 cache；metadata 缓存不是 KV 写回。

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

QLI 接收原始 token 历史长度和 `cmp_ratio=4`，返回的是压缩行号。结果不能再除以 4，而应按 $\lfloor a/4\rfloor$ 过滤越界/未完成行，并将有效项移到前面。随后 sparse Attention 的有效长度是压缩行数，两个接口的单位不能混用。

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

`_c4_segment` 默认把行号和页表直接交给第 2.3 节的原生 sparse Attention，不为每个 query 物化完整的 Top-512 KV 张量。只有选择 `ORTHRUS_C4_BACKEND=gather` 时才走收集后 FIA 的兼容路径。下面是完整分支，保留了默认路径与回退条件。

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

直接分页读取减少了中间 KV 展开，实际 workspace 和耗时仍需按形状测量。PyPTO 适配时应区分原始滑窗的物理 slot 与 C4 的逻辑压缩行号，并对齐 scale、Hadamard 约定和无效项 `-1`。

源码定位：[CSA 编排](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_csa.py)、[Indexer](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_indexer.py)、[CSA Attention](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py)。

## 3. 层级执行与推理引擎集成

四个 Attention 模块定义了历史与临时块如何交互。完整的一层还需要先由 diffusion 参数生成 Q/KV，再将 Attention 结果投影回 hidden，继续经过共享主干。前后处理和引擎接入不改变上述公式，但决定权重、布局和状态能否正确连接。

### 3.1 Diffusion 输入投影与输出投影

每行 4096 维 hidden 经 `wq_a` 降到 1024 维，经过内部 RMSNorm，再由 `wq_b` 生成各 head 的 query，执行 head RMSNorm 和 partial RoPE。KV 侧由 `wkv` 生成 512 维 shared KV，再执行 norm/RoPE，得到本轮临时块。完整模型共有 64 个 Attention head，TP 使用对应分片。

`_qkv_readonly` 在 Q/KV 投影都采用动态 W8A8 时，共用一次输入量化，再分别执行 `npu_quant_matmul`；低秩 query 使用 `npu_rms_norm_dynamic_quant` 合并 norm 与量化。这样可减少重复处理，同时保留各 diffusion 投影自身的权重和 scale。

下面给出完整投影代码，包括共享输入处理、query/KV 两条分支以及 partial RoPE。

<details>
<summary>完整源码：Q/KV 投影与 partial RoPE</summary>

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

联合 Attention 输出随后执行 inverse partial RoPE，再由 diffusion 对象的 `_forward_o_proj` 完成分组低秩输出投影，映射回 4096 维 hidden。这里复用固定版本 DSV4 的投影实现，调用位置见下一节的 `_attention_readonly`。

PyPTO 的 `qkv_proj_rope.py` 和 `decode_o_proj.py` 已包含相应基础计算，适配重点是独立权重/scale、K 行形状、TP 分片和 mHC 约定。所对照的 HCA/CSA 等独立入口显式支持 TP=1/2/4，仍需单独检查 TP8 的通信和编译形状。

源码定位：[Q/KV 投影](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/qkv_proj_rope.py)、[输出投影](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_o_proj.py)。

### 3.2 层级执行与共享主干

一层的执行顺序是：diffusion Q/KV 投影，联合计算滑窗与双向块，按层类型读取 C128 或 C4 历史，再合并各段输出与 sink，最后执行逆 RoPE 和输出投影。

服务路径还需从真实运行时中找到对应的目标模型层、cache 和 metadata。`_history_segments` 统一组织三条路径：先计算所有层都需要的“滑窗 + 临时块”，再按 `compress_ratio` 添加 C4 或 C128；`_attention_readonly` 负责取得运行上下文，追加一次 sink，完成合并与输出投影。

以下对象是服务函数的输入边界。它们解释了为什么单层 Attention 可以独立演示，而完整服务仍需要 proposer 和模型执行器提供上下文。

| 对象 | 来源与作用 |
| --- | --- |
| `self.self_attn` / `draft_impl` | diffusion Attention 参数对象，包含独立投影、内部 norm、sink 和 CSA indexer |
| `target_attention` / `target_kv` | 当前层绑定的主模型 Attention 及其持久 cache，使用原生布局，不复制整份历史 |
| `hidden_states` / `positions` | 共享主干产生的 `[batch × K, hidden]` 与绝对位置，每个请求第一行对应 anchor |
| `query` / `local_kv` | 在 Attention 内整理为 `[batch, K, local_heads, 512]` 和 `[batch, K, 1, 512]` |
| `layer_metadata` | 页表、有效历史长度、RoPE 表等信息；Orthrus 的历史边界对应 anchor 前的前缀 |
| `_dsa_runtime` / `DeviceOperator` | 固定 vLLM-Ascend 版本的量化和 cache 布局适配接口 |
| `BLOCK_SIZE` | 本服务的 draft 块长 32，不是 KV 分页大小；后者由 cache 布局单独确定 |

下面列出完整计算方法。`_build_kv_cache`、`get_forward_context`、设备算子注册和模型构造属于固定版本 vLLM-Ascend 的运行时依赖；这些方法不是可以脱离运行时单独粘贴执行的脚本。独立运行方式见第 4 节。


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


`_attention_readonly` 的 cache guard 是可选诊断功能，用于比较 draft 前后的目标状态；正常计算不依赖这些检查张量。初始化/profile 阶段若没有真实 metadata，代码返回占位输出，这不构成一次有效草稿生成。

Attention 计算完成后，block 继续执行共享层的 mHC、外层 norm 和 MoE。下面的方法展示这一衔接：只有 Attention 参数和计算来自 diffusion 分支，外层参数仍取自主模型。


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


### 3.3 Cache 生命周期与职责边界

目标历史的页分配、回收和请求归属由推理引擎管理；kernel 根据页表读取实际数据，并决定是否执行内存写入。这两个层次不能互相替代：框架把 cache 当作只读输入，并不会自动消除某个 AR 融合入口内部的写回副作用。

草稿执行期间，目标 SWA KV、压缩 KV、indexer cache 和 compressor state 都应保持不变；只产生临时 draft KV 和候选概率。候选交给主模型验证后，由目标路径生成新的 target KV，并由验证器与调度器维护接受前缀、拒绝恢复和后续请求状态。

因此，PyPTO 计算侧的输入包括 diffusion 权重、整块 hidden/位置以及只读目标历史视图，输出是对应的 draft hidden。proposer、verifier 和 scheduler 则负责 anchor/MASK 构造、候选传递和状态提交。算子实现无需承担完整调度，但必须保证自身不破坏目标历史。

## 4. 运行入口与实现范围

### 4.1 vLLM-Ascend 服务入口

服务由 vLLM-Ascend 加载主模型和独立 draft head，并提供目标分页 cache 与运行时上下文。附录 A 收录完整计算模块、公共辅助模块和启动脚本；独立单层运行入口见第 4.2 节。

完成 [固定版本安装](../runtime/vllm_ascend_v0_26/README.md) 后，在容器中的项目根目录运行：

```bash
MODEL=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
HEAD=/data/disk/models/orthrus-dsv4-head-v2-w8a8 \
TP=8 \
ORTHRUS_PREFLIGHT_MODE=production \
ORTHRUS_EXECUTION_MODE=eager \
ORTHRUS_DRAFT_SAMPLE_METHOD=probabilistic \
bash scripts/serve_dsv4_orthrus.sh
```

模型和 head 路径需按部署环境指定。`production` preflight 校验完整 43 层模型/head 契约，不表示生产验收已经通过；初始化 head 只用于链路验证。服务始终保留 [target-only 启动路径](../scripts/serve_dsv4_target.sh)，完整部署与回退方法见 [部署说明](DSV4_DEPLOYMENT.md)。

| 环节 | 关键实现 | 目的 |
| --- | --- | --- |
| 投影 | Q/KV 共享输入处理，融合 norm/quant | 减少重复量化和中间张量 |
| 滑窗与双向块 | 收集短窗口后，一次联合 FIA | 减少 Attention 调用和合并次数 |
| C128 | 按页大小选择 FIA 或 sparse 接口，读取全部有效行 | 避免展开完整压缩历史，不截断可见范围 |
| C4 | QLI 检索后，按行号直接做分页 sparse Attention | 避免为每个 query 展开 Top-K KV 大张量 |
| 归一化 | AttentionUpdate 合并各段输出，sink 计一次 | 保持全局 softmax，避免重复计入 sink |

以上是当前算子组合方式，不表示全模型吞吐已达到最优或八卡验收已完成。计算模块对应 `360ab06` 所在实现快照。

### 4.2 独立单层 Demo

单层 demo 不依赖 vLLM metadata，使用随机权重和模拟历史，串起投影、三类历史、双向块、sink 和输出投影，便于独立运行和理解计算过程。它采用 BF16，完整源码见附录 B；该 demo 不用于服务性能评估或真实长上下文验收。

```bash
PYTHONPATH=src python scripts/demo_torch_npu_dsv4_layer.py --block-size 32
```

实际部署仍以第 4.1 节为准。源码入口：[服务计算模块](../src/blockspec/vllm_ascend/dsv4.py)、[公共辅助模块](../src/blockspec/vllm_ascend/dsv4_hybrid.py)、[proposer](../src/blockspec/vllm_ascend/dsv4_proposer.py)、[单层 Demo](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py)。

当前服务契约固定 K=32。修改 K 不需要重新发明 Attention 数学，但要同步 head/调度契约、编译形状及临时空间，并重新验证。

## 5. 开发范围与验证要求

### 5.1 PyPTO 开发范围

主要工作是沿现有三条历史路径接入 Orthrus 的参数与状态语义，而不是从零开发全部基础计算。公共双向块可内联到各路径；公共投影和共享主干沿用原有数学，但仍需检查块形状、TP 和运行时接线。

| 模块 | 现有实现基础 | 适配内容 |
| --- | --- | --- |
| 公共双向块 | DSpark 块内双向 QK/PV、online-softmax，HCA 状态合并 | 独立临时 KV 输入；K/布局适配；贡献纳入统一归一化，sink 计一次 |
| SWA | 分页滑窗读取、公共历史预取、Attention | 固定 anchor 前窗口；全可见临时块替代逐行因果追加；绕过目标写回 |
| HCA | C128 读取、滑窗/压缩状态合并 | 固定完整压缩行数；增加临时块；跳过 draft compressor/写回并调整依赖 |
| CSA | query 变换、量化评分/Top-K、C4 分页 Attention | diffusion indexer 参数；短/长历史分支统一 anchor 边界；跳过 key/state 写回并加入临时块 |

实现顺序可从公共块与 SWA 开始，再接 HCA 的分页历史和状态合并，最后接 CSA 的量化 query、检索边界和 Top-K。这样每一步都有可独立对照的现成算子路径。具体 kernel 融合数量、流水线和 workspace 策略需要基于 PyPTO 实测确定，不能仅按逻辑模块数估算工作量。

### 5.2 正确性与服务验收

算子层先对照相同权重、输入、精度和布局下的参考输出，检查可见行、LSE/归一化、padding 和空历史；整层再覆盖输入输出投影、不同 batch、分页边界以及目标 cache/压缩状态不变。引擎层最后检查候选接受与拒绝、请求取消、页复用和恢复。仅检查输出形状或服务不崩溃不足以完成验收。

现有 Torch-NPU 实测已覆盖 A2 算子对照、四层 W8A8 TP2、全部候选接受长度、概率传递、拒绝恢复及物理页复用；抢占恢复通过受控分配失败触发。完整 43 层 TP8 主模型已完成健康检查、串行 greedy 重放及长上下文请求采集，但自身并发随机重放和一个混合长度 greedy 案例存在差异；切换 Orthrus 前被端口检查中断，Orthrus TP8 对照尚未完成。graph、1M 上下文和持续负载也未完成验收。初始化 head 用于检查推理接线，训练 head 的接受率与加速收益需要另行评估，详见 [实测记录](DSV4_VALIDATION_20261028.md) 与 [验证入口](DSV4_VALIDATION_RUNBOOK.md)。

以上证据属于 Torch-NPU 路径，不代表 PyPTO Orthrus 适配已交付。PyPTO 完成实现后，应在同一数学和状态契约下建立对应验证，再进入完整推理服务。

## 附录 A：服务计算模块与启动脚本

下面给出完整 `dsv4.py`、公共辅助模块和启动脚本，包含模型构造、共享主干绑定、权重加载及各计算路径。运行时还需固定版本引擎、vendor 算子、proposer，以及匹配的主模型和 head 权重。

<details>
<summary>完整源码 1/3：dsv4.py</summary>

```python
"""DeepSeek-V4 Orthrus diffusion head for vLLM-Ascend 0.26.

The target owns the complete DSV4 backbone and every historical cache. The
draft owns only attention/head parameters and reads SWA/C4/C128 history without
writing it. FIA handles the local window/block and large-page C128 history;
native sparse attention handles C4 and small-page C128. Each exposes an LSE.
"""

from __future__ import annotations

import os
import re
import weakref
from collections.abc import Iterable, Sequence
from types import SimpleNamespace

import torch
from torch import nn
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import PPMissingLayer
# The 0.26 rc1 monolithic package needs its operator registrations completed
# before DeviceOperator imports submodules from the same package.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import dsa_v1 as _dsa_runtime
from vllm_ascend.device.device_op import DeviceOperator
try:
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention
except ImportError:  # 0.26 release image before the package split
    from vllm_ascend.models.deepseek_v4 import DeepseekV4Attention

try:
    from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
except ImportError:  # 0.26 release image before the package split
    from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.ops.dsa import _build_kv_cache
from vllm_ascend.utils import enable_dsa_cp

from .dsv4_hybrid import (
    assert_cache_unchanged,
    cache_checksum,
    c4_history_indices,
    gather_paged_cache,
    history_and_block_mask,
    merge_attention_outputs,
    tail_logical_indices,
)
from .head_contract import BLOCK_SIZE, FORMAT_NAME, NUM_SPECULATIVE_TOKENS
from .trace import instant


_HEAD_WEIGHT = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.(?:attn|self_attn)\."
    r"(?:attn_sink|wq_a|q_norm|q_norm_without_weight|wq_b|wkv|kv_norm|"
    r"wo_a|wo_b|indexer\.(?:wq_b|weights_proj))(?:\.|$)"
)


def _install_orthrus_quant_aliases(quant_config, ratios: list[int]) -> None:
    if quant_config is None:
        return
    description = getattr(quant_config, "quant_description", None)
    if not isinstance(description, dict):
        return
    aliases = {}
    for name, value in description.items():
        normalized = name.removeprefix("model.")
        if _HEAD_WEIGHT.match(normalized):
            aliases["orthrus." + normalized.replace(".attn.", ".self_attn.", 1)] = value
    # The vendor DeepseekV4Attention constructor always creates target-side
    # compressors before Orthrus can detach them. They are absent from the head
    # artifact by contract, so give only those throwaway constructor modules a
    # local FLOAT policy. No compressor survives in named_parameters().
    for layer, ratio in enumerate(ratios):
        if ratio <= 1:
            continue
        roots = [f"orthrus.layers.{layer}.self_attn.compressor"]
        if ratio == 4:
            roots.append(f"orthrus.layers.{layer}.self_attn.indexer.compressor")
        for root in roots:
            aliases[f"{root}.wkv.weight"] = "FLOAT"
            aliases[f"{root}.wgate.weight"] = "FLOAT"
    description.update(aliases)


def _alias_cache(module: object | None, target_name: str) -> None:
    if module is not None:
        setattr(module, "kv_sharing_target_layer_name", target_name)


def _linear(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    output = module(value)
    return output[0] if isinstance(output, tuple) else output


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


class _NoTargetState(nn.Module):
    """Placeholder proving that target compressor state is not in the head."""

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Orthrus attempted to execute a target-only compressor")


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


class _OrthrusAttentionLayer(nn.Module):
    def __init__(self, vllm_config, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._full_checked_batches: set[int] = set()
        self.compress_ratio = int(config.compress_ratios[layer_idx])
        self.self_attn = DeepseekV4Attention(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=config.rope_parameters[
                "original_max_position_embeddings"
            ],
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"orthrus.layers.{layer_idx}.self_attn",
        )
        self._register_target_cache_aliases()
        self._drop_target_only_modules()
        self._target_attention_ref: weakref.ReferenceType[nn.Module] | None = None

    @property
    def _target_prefix(self) -> str:
        return f"model.layers.{self.layer_idx}.self_attn"

    def _register_target_cache_aliases(self) -> None:
        attention = self.self_attn
        wrapper = attention.dsa_attn
        _alias_cache(wrapper.swa_cache_layer, f"{self._target_prefix}.swa_cache")
        if self.compress_ratio > 1:
            _alias_cache(wrapper.dsa_attn, f"{self._target_prefix}.attn")
            assert attention.compressor is not None
            _alias_cache(
                attention.compressor.state_cache,
                f"{self._target_prefix}.compressor.state_cache",
            )
        if self.compress_ratio == 4:
            assert attention.indexer is not None
            assert attention.indexer.compressor is not None
            _alias_cache(attention.indexer.k_cache, f"{self._target_prefix}.indexer.k_cache")
            _alias_cache(
                attention.indexer.compressor.state_cache,
                f"{self._target_prefix}.indexer.compressor.state_cache",
            )

    def _drop_target_only_modules(self) -> None:
        """Remove compressor weights while retaining the diffusion-side indexer."""
        attention = self.self_attn
        wrapper = attention.dsa_attn
        impl = wrapper.dsa_attn.impl
        if self.compress_ratio > 1:
            missing = _NoTargetState()
            attention.compressor = missing
            wrapper.compressor = missing
            impl.compressor = missing
            for name in (
                "compressor_ape",
                "compressor_wkv",
                "compressor_wgate",
                "compressor_norm",
            ):
                if hasattr(impl, name):
                    setattr(impl, name, None)
        if self.compress_ratio == 4:
            assert attention.indexer is not None
            attention.indexer.compressor = _NoTargetState()
            for name in (
                "indexer_compress",
                "indexcom_ape",
                "indexcom_wkv",
                "indexcom_wgate",
                "indexcom_norm",
            ):
                if hasattr(impl, name):
                    setattr(impl, name, None)

    def bind_target(self, target_layer: nn.Module) -> None:
        target_attention = target_layer.self_attn
        if int(target_attention.compress_ratio) != self.compress_ratio:
            raise ValueError(
                f"layer {self.layer_idx} compress ratio mismatch: "
                f"head={self.compress_ratio}, target={target_attention.compress_ratio}"
            )
        self._target_attention_ref = weakref.ref(target_attention)

    def _target_attention(self) -> nn.Module:
        target = self._target_attention_ref() if self._target_attention_ref else None
        if target is None:
            raise RuntimeError("Orthrus diffusion head is not bound to the DSV4 target")
        return target

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


class _OrthrusDeepseekV4Model(nn.Module):
    def __init__(self, vllm_config, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.embed_tokens = PPMissingLayer()
        self.layers = nn.ModuleList(
            [_OrthrusAttentionLayer(vllm_config, config, index) for index in range(config.num_hidden_layers)]
        )
        self._target_model_ref: weakref.ReferenceType[nn.Module] | None = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        target_inner = getattr(target_model, "model", None)
        target_layers = getattr(target_inner, "layers", None)
        if target_inner is None or target_layers is None or len(target_layers) != len(self.layers):
            raise ValueError("draft/target DSV4 layer count mismatch")
        self._target_model_ref = weakref.ref(target_inner)
        for draft_layer, target_layer in zip(self.layers, target_layers, strict=True):
            draft_layer.bind_target(target_layer)
        leaked = [
            name
            for name, _ in self.named_parameters()
            if ".compressor." in name
        ]
        if leaked:
            raise RuntimeError(
                "target compressor parameters leaked into Orthrus draft: "
                + ", ".join(leaked[:8])
            )

    def _target_model(self) -> nn.Module:
        target = self._target_model_ref() if self._target_model_ref else None
        if target is None:
            raise RuntimeError("Orthrus draft is not bound to the live DSV4 target")
        return target

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        names = []
        for layer in self.layers:
            prefix = layer._target_prefix
            names.append(f"{prefix}.swa_cache")
            if layer.compress_ratio > 1:
                names.extend((f"{prefix}.attn", f"{prefix}.compressor.state_cache"))
            if layer.compress_ratio == 4:
                names.extend(
                    (
                        f"{prefix}.indexer.k_cache",
                        f"{prefix}.indexer.compressor.state_cache",
                    )
                )
        return names

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        target = self._target_model()
        hidden_states = target.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for draft_layer, target_layer in zip(self.layers, target.layers, strict=True):
            hidden_states = draft_layer(positions, hidden_states, target_layer, input_ids)
        return target.hc_head(
            hidden_states, target.hc_head_fn, target.hc_head_scale, target.hc_head_base
        )


class OrthrusDSV4DraftModel(DSparkDeepseekV4ForCausalLM):
    """Attention-only K32 diffusion head with a shared DSV4 backbone."""

    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)
        # The decorated DSpark parent installs its compile wrapper in __call__,
        # but invoking its constructor would allocate a second draft backbone.
        # Orthrus is intentionally eager, so initialize the inherited wrapper
        # contract without constructing the parent model.
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.do_not_compile = True
        if get_pp_group().world_size != 1:
            raise NotImplementedError("DSV4 Orthrus currently requires PP=1")
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if additional_config.get("enable_dsa_cp"):
            raise NotImplementedError(
                "DSV4 Orthrus does not yet support DSA context parallel; "
                "remove enable_dsa_cp and use the TP-only correctness backend"
            )
        if additional_config.get("enable_flashcomm1"):
            raise NotImplementedError(
                "DSV4 Orthrus does not yet support FlashComm1 sequence parallel; "
                "remove enable_flashcomm1 and use the TP-only correctness backend"
            )
        speculative = vllm_config.speculative_config
        if speculative is None or speculative.num_speculative_tokens != NUM_SPECULATIVE_TOKENS:
            raise ValueError("DSV4 Orthrus requires exactly 31 speculative tokens")
        config = speculative.draft_model_config.hf_config
        if getattr(config, "orthrus_head_format", None) != FORMAT_NAME:
            raise ValueError("DSV4 Orthrus requires an orthrus-dsv4-head-v2 artifact")
        ratios = list(getattr(config, "compress_ratios", []))[: config.num_hidden_layers]
        if len(ratios) != config.num_hidden_layers or any(r not in (0, 1, 4, 128) for r in ratios):
            raise ValueError("invalid DSV4 hybrid compress_ratios in diffusion head")
        if getattr(config, "sample_from_anchor", None) is not False:
            raise ValueError("Orthrus K32 requires sample_from_anchor=false")
        _install_orthrus_quant_aliases(vllm_config.quant_config, ratios)
        self.config = config
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        self.model = _OrthrusDeepseekV4Model(vllm_config, config)
        self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        self.num_expert_groups = getattr(config, "n_group", 1)
        self._target_ref: weakref.ReferenceType[nn.Module] | None = None

    def bind_shared_backbone(self, target_model: nn.Module) -> None:
        self._target_ref = weakref.ref(target_model)
        self.model.bind_shared_backbone(target_model)
        for name in (
            "expert_weights",
            "moe_layers",
            "moe_mlp_layers",
            "num_expert_groups",
            "num_moe_layers",
            "num_logical_experts",
            "num_physical_experts",
            "num_local_physical_experts",
            "num_routed_experts",
            "num_shared_experts",
            "num_redundant_experts",
        ):
            if hasattr(target_model, name):
                setattr(self, name, getattr(target_model, name))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return [False]

    def combine_hidden_states(self, states: torch.Tensor) -> torch.Tensor:
        return states[..., : self.config.hidden_size].contiguous()

    def precompute_and_store_context_kv(self, *args, **kwargs) -> None:
        del args, kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        if inputs_embeds is not None:
            raise ValueError("DSV4 Orthrus does not accept separate inputs_embeds")
        return self.model(input_ids=input_ids, positions=positions)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        target = self._target_ref() if self._target_ref else None
        if target is None:
            raise RuntimeError("Orthrus draft is not bound to the target LM head")
        return self.logits_processor(target.lm_head, target.model.norm(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        for source_name, loaded_weight in weights:
            normalized = source_name.removeprefix("model.")
            if normalized.startswith("orthrus.layers."):
                normalized = normalized.removeprefix("orthrus.")
            if not _HEAD_WEIGHT.match(normalized):
                continue
            name = "model." + normalized.replace(".attn.", ".self_attn.", 1)
            if name.endswith(".scale"):
                name = name.removesuffix(".scale") + ".weight_scale"
            param = params.get(name)
            if param is None:
                continue
            if name.endswith(".attn_sink") and not enable_dsa_cp():
                param.data.copy_(loaded_weight.narrow(0, tp_rank * heads_per_rank, heads_per_rank))
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            loaded.add(name)
        expected = {
            name
            for name in params
            if name.startswith("model.layers.") and _HEAD_WEIGHT.match(name.removeprefix("model."))
        }
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"Orthrus head is missing {len(missing)} parameters: {', '.join(missing[:8])}"
            )
        return loaded


# Kept as an import-time error message for stale deployment scripts.
OrthrusDSparkDeepseekV4ForCausalLM = OrthrusDSV4DraftModel
```

</details>

<details>
<summary>完整源码 2/3：dsv4_hybrid.py</summary>

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
<summary>完整源码 3/3：serve_dsv4_orthrus.sh</summary>

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
source "$repo_root/scripts/dsv4_service_env.sh"

: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8}"
: "${HEAD:=/data/disk/models/orthrus-dsv4-head-v2-w8a8}"
PORT="${PORT:-8901}"
TP="${TP:-8}"
MEMORY="${MEMORY:-0.90}"
DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy}"
DEFAULT_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-$DEFAULT_ADDITIONAL_CONFIG}"

case "$DRAFT_SAMPLE_METHOD" in
  greedy|probabilistic) ;;
  *)
    echo "ORTHRUS_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
    exit 1
    ;;
esac

execution_args=()
case "${ORTHRUS_EXECUTION_MODE:-eager}" in
  eager)
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
    execution_args+=(--enforce-eager)
    ;;
  graph)
    if [[ "${ORTHRUS_GRAPH_VALIDATED:-0}" != "1" ]]; then
      echo "Graph mode is gated; set ORTHRUS_GRAPH_VALIDATED=1 only after the NPU correctness matrix passes" >&2
      exit 1
    fi
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ;;
  *)
    echo "ORTHRUS_EXECUTION_MODE must be eager or graph" >&2
    exit 1
    ;;
esac
COMPILATION_CONFIG="${COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"

profile_args=()
if [[ -n "${ORTHRUS_TORCH_PROFILE_DIR:-}" ]]; then
  profiler_config="$("$python_bin" - "$ORTHRUS_TORCH_PROFILE_DIR" <<'PY'
import json
import sys

print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": sys.argv[1],
    "torch_profiler_with_stack": False,
    "torch_profiler_with_memory": False,
    "ignore_frontend": False,
}))
PY
)"
  profile_args+=(--profiler-config "$profiler_config")
fi

configure_dsv4_profile
configure_dsv4_runtime
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

preflight=(
  "$python_bin" "$repo_root/scripts/preflight_dsv4_orthrus.py"
  --model "$MODEL"
  --head "$HEAD"
  --tensor-parallel-size "$TP"
)
case "${ORTHRUS_PREFLIGHT_MODE:-production}" in
  production) preflight+=(--production) ;;
  development) ;;
  *)
    echo "ORTHRUS_PREFLIGHT_MODE must be production or development" >&2
    exit 1
    ;;
esac
if [[ "${REQUIRE_TRAINED_HEAD:-0}" == "1" ]]; then
  preflight+=(--require-trained)
fi
"${preflight[@]}"

SPECULATIVE_CONFIG="$("$python_bin" - "$HEAD" "$DRAFT_SAMPLE_METHOD" <<'PY'
import json
import sys

print(json.dumps({
    "method": "orthrus",
    "model": sys.argv[1],
    "num_speculative_tokens": 31,
    "parallel_drafting": True,
    "draft_sample_method": sys.argv[2],
    "rejection_sample_method": "standard",
    "quantization": "ascend",
    "enforce_eager": True,
}))
PY
)"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name dsv4-orthrus \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$MEMORY" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --quantization ascend \
  --block-size "$BLOCK_SIZE" \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  "${execution_args[@]}" \
  "${profile_args[@]}" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --compilation-config "$COMPILATION_CONFIG" \
  --additional-config "$ADDITIONAL_CONFIG"
```

</details>

## 附录 B：独立单层 Demo 完整源码

下面给出第 4.2 节 demo 的完整源码。模型类构造随机参数与模拟 cache，运行脚本依次调用三类 Attention；公共辅助模块与服务共用，demo 使用其中的 Torch LSE 合并函数。


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
