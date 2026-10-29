# Orthrus-DSV4 NPU 推理适配与算子设计

## 1. 背景与适配目标

大模型的自回归推理需要依次生成 token：当前 token 确定后，才能继续计算下一个位置。投机解码通过草稿分支提前提出多个候选，再由主模型批量验证，尝试减少主模型逐 token 执行的轮数。能否获得加速，既取决于候选接受率，也取决于草稿生成、验证以及状态管理的额外开销。

Orthrus 采用共享主干的草稿结构，在模型各层增加独立的 diffusion Attention，通过块内双向 Attention 一次预测多个位置。在本文的 DSV4 适配中，embedding、外层 norm、mHC、MoE 和 LM head 与主模型共享，因此它既不是另起一个完整的小模型，也不是只在最终 hidden 后附加一层预测头。草稿分支仍执行各层计算，但不复制整套主干参数，并且能够读取主模型已经生成的历史 KV。

将这一结构用于 DeepSeek-V4，需要处理其混合 Attention：SWA 读取滑动窗口，HCA 还读取 C128 压缩历史，CSA 则通过 indexer 选择 C4 压缩历史。适配工作的重点，是在保留三类历史表示和检索规则的前提下，让 diffusion query 同时读取主模型历史与临时双向块，并避免草稿计算改变目标模型状态。

本文聚焦 Orthrus-DeepSeek-V4-Flash-0731 在 Ascend A2 上的推理计算与引擎接入，不包含训练和在线更新。代码基于固定 v0.26 运行时，采用现成 Torch-NPU 算子实现三类历史读取与双向块计算。当前 head 为初始化版本，只用于链路验证，不代表训练后的接受率或加速收益。

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

Attention 侧可分为四个逻辑模块：**公共双向块、SWA、HCA、CSA**。公共块供三类历史路径使用，可以融合到对应计算中；四个逻辑模块不等于必须新增四个独立 kernel。下文以各模块的计算需求为单位，同时说明原有语义、目标语义、PyPTO 实现基础和现有 Torch-NPU 实现。 PyPTO 的实现基础以 `pypto-lib` 提交 `4c908f3` 中的 DSV4/DSpark 路径为准。

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

在 Torch-NPU 实现中，历史和当前块都交给 `npu_fused_infer_attention_score`（FIA）计算。它接收一批 query，以及这些 query 可以读取的 KV，返回 Attention 输出和这一部分的 LSE。双向 Attention 并没有换一套计算公式，区别在于送进算子的 KV 范围，以及哪些位置被 mask 屏蔽。

下面是 `_fia_segment` 的实际调用。`query` 的布局为 `[batch, K, local_heads, dim]`，每个请求一次处理 K 行；`key_value` 可以是连续的“滑窗历史 + 临时块”，也可以是主模型的分页压缩 cache。后一种情况需要额外提供页表，让算子根据逻辑位置找到实际存储的数据。

```python
result = torch_npu.npu_fused_infer_attention_score(
    query,
    # DSV4 的 shared KV 同时作为 key 和 value。
    key_value,
    key_value,
    num_heads=query.shape[2],
    num_key_value_heads=kv_heads,
    input_layout="BSND",
    # Orthrus 不使用逐行因果模式；历史 padding 仍由显式 mask 排除。
    sparse_mode=0,
    scale=scale,
    atten_mask=atten_mask,
    block_table=block_table,
    block_size=block_size,
    # 分页模式由长度限定历史；拼接模式同时依靠 mask 排除 padding。
    actual_seq_lengths=query_lengths,
    actual_seq_lengths_kv=key_lengths,
    # 返回本段 LSE，供压缩历史与 sink 的联合归一化使用。
    softmax_lse_flag=True,
)
```

`sparse_mode=0` 表示这里不使用 AR 的逐行因果规则，但历史边界仍然需要由有效长度和 mask 限制。例如，把滑窗与临时块一起传入时，历史中的 padding 要屏蔽，临时块的 K 个位置则全部可见；读取分页压缩历史时，只允许读取 anchor 之前已经完成的压缩行。

FIA 的输出只对本次输入的 KV 做了归一化。如果后面还要加入另一段历史，就需要保留 LSE，才能把两段结果放回同一个 softmax 分母中。完整封装还处理了输出布局和空历史：空段返回零输出、负无穷 LSE，因此不会对后续合并产生贡献。

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

当前实现先把 SWA 历史与临时块放在同一次 FIA 中计算。对于 HCA/CSA 层，压缩历史另外返回一组输出和 LSE；最后再加入 sink。sink 的 value 为零，因此可以表示成一段“零输出、LSE 等于 sink logit”的结果，与其他分段一起交给 `merge_attention_outputs`：

```python
# shape 来自 outputs[0]，各段的 batch/query/head 必须一一对应。
head_dim = shape[-1]
# 按相同顺序传入各段 LSE 和输出，让算子恢复统一 softmax 权重。
merged, _ = torch_npu.npu_attention_update(
    [lse.float().reshape(-1) for lse in logsumexp],
    [output.reshape(-1, head_dim) for output in outputs],
    0,
)
# 展平只是接口布局转换，合并后恢复原来的各位置、各 head。
return merged.view(shape)
```

这里传入的每一组输出都已经除以自己的 softmax 分母。直接把它们相加，会把每个分段当成同等权重；`npu_attention_update` 则根据对应的 LSE 计算各段应占的比例，得到式 (4) 的联合结果。展平操作只是为了匹配算子接口，同一个请求、同一个位置、同一个 head 的各段结果仍然一一对应。

对 PyPTO 来说，需要保持的是这次联合归一化。若融合 kernel 已经维护式 (6) 的 online-softmax 状态，就可以在内部直接合并，无需额外生成 LSE 再调用一个独立算子。

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

SWA 需要先确定本轮草稿能看到哪一段原始历史。`_history_segments` 收到的 `seq_lens` 是各请求在 anchor 之前的历史长度，`swa_cache` 和页表都来自主模型。以这个长度为终点，`tail_logical_indices` 为每个请求生成最近最多 128 行的逻辑位置，再由 `gather_paged_cache` 通过页表读出对应 KV：

```python
# 只生成每个请求的一组窗口行号，不随块内 query 位置向右滑动。
swa_indices = tail_logical_indices(seq_lens, draft_impl.window_size)
swa_lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, draft_impl.window_size))
# 通过目标页表读取历史；该调用不写 cache。
swa_kv = gather_paged_cache(
    swa_cache,
    swa_req.block_table,
    swa_indices,
    validate=validate_cache,
)
```

例如，anchor 前有 1000 个 token，读取的窗口就是位置 872–999。本轮 32 个 query 共用这份历史，不会随着块内位置向右移动而丢掉窗口左端。历史不足 128 行时，行号中的 `-1` 表示无效位置，收集出的缓冲区会用零填充这些行。

历史读取完成后，`_swa_block_segment` 把它与 diffusion 分支生成的临时 `local` KV 接在一起。对每个 query 来说，前半部分是同一段真实历史，后半部分是整个 anchor/MASK 块：

```python
def _swa_block_segment(query, history, local, lengths, scale):
    # ORTHRUS：历史与整个临时块共同参与一次 softmax，不逐 token 追加。
    kv = torch.cat((history, local), dim=1)
    # 只屏蔽历史 padding，临时块的全部 K 列始终可见。
    mask = history_and_block_mask(lengths, history.shape[1], local.shape[1])
    batch = query.shape[0]
    return _fia_segment(
        query, kv, [query.shape[1]] * batch, [kv.shape[1]] * batch, scale,
        atten_mask=mask,
    )
```

填零本身并不能排除 padding：零 key 的点积分数是零，经过指数运算后仍会进入 softmax 分母。因此还需要下面这段 mask 构造，`True` 表示该列被屏蔽：

```python
offsets = torch.arange(history_width + block_width, device=history_lengths.device)
# True 表示屏蔽：只命中历史 padding，不命中后面的临时块。
masked = (offsets[None, :] >= history_lengths[:, None]) & (offsets[None, :] < history_width)
# 将同一可见范围广播给全部 K 个 query，而不是构造三角 mask。
return masked[:, None, None, :].expand(-1, 1, block_width, -1).contiguous()
```

这个条件只屏蔽历史缓冲区中超出有效长度的部分。临时块位于历史缓冲区之后，不会命中该条件；同一 mask 又被广播给全部 K 个 query，因此块内没有上三角屏蔽，每个位置都可以读取整个块。FIA 随后对所有未被屏蔽的列共同做 softmax，这就实现了 SWA 公式中的 `W(a-1) ∥ Bᴰ`。

对于纯 SWA 层，这已经包含全部可见 KV，后面只需计入 sink；对于 HCA/CSA 层，它是三条路径共同使用的局部部分，还要与压缩历史合并。这里的 `cat` 是调用现成 FIA 的输入组织方式，PyPTO 可以分别读取历史和临时块，只要最终的可见范围和归一化结果一致。

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

HCA 在滑窗之外还需要读取 C128 压缩历史。这份 cache 已由主模型生成，draft 使用它时首先要确定哪些压缩组已经完成。代码将 anchor 前的 token 数除以 128 并向下取整，得到本轮可读的压缩行数；尚未凑齐一组的状态仍留在主模型 compressor 中：

```python
# 只读取完整压缩组：例如 1000 个历史 token 对应 7 行 C128。
compressed_lengths = torch.div(seq_lens, 128, rounding_mode="floor")
# maximum 用于确定索引缓冲区上界；实际可读范围仍由 compressed_lengths 限制。
maximum = getattr(compressed_req, "max_seq_lens", None)
maximum = (int(maximum) // 128 if maximum is not None
           else compressed_req.block_table.shape[1] * compress_cache.shape[1])
# 直接传入目标压缩 cache 和页表，不先收集整段长历史。
out, lse = _c128_segment(
    query_bsnd,
    compress_cache,
    compressed_req.block_table,
    compressed_lengths,
    maximum,
    draft_impl.softmax_scale,
)
```

`compressed_lengths` 决定每个请求实际能读多少行，`maximum` 则用于准备足够大的索引缓冲区，两者不能混用。取得长度后，`_c128_segment` 根据 cache 的页大小选择能处理该布局的接口。下面是默认 `auto` 路径：

```python
# 当前 FIA 接口支持的页大小，直接使用分页 FIA。
if backend == "auto" and 128 <= cache.shape[1] <= 1024 and cache.shape[1] % 128 == 0:
    return _fia_segment(
        query, cache, [BLOCK_SIZE] * query.shape[0], lengths, scale,
        allow_empty=True, block_table=block_table,
    )
if backend != "gather":
    # 其他页大小走 sparse 接口，但枚举全部有效历史行，不进行 Top-K。
    width = max(128, (maximum + 127) // 128 * 128)
    offsets = torch.arange(width, device=lengths.device, dtype=torch.int32)
    # 对齐产生的额外行用 -1 屏蔽，所有 query 共用相同历史边界。
    indices = torch.where(offsets[None, None, :] < lengths[:, None, None], offsets, -1)
    indices = indices.expand(-1, query.shape[1], -1).contiguous()
    return _sparse_history_segment(query, cache, block_table, indices, lengths, scale)
```

页大小满足 FIA 接口条件时，直接将 cache、页表和有效长度传入分页 FIA。其余受支持的页布局使用 `npu_sparse_flash_attention`：先列出每个请求的全部有效 C128 行，再让算子按这些行号读取分页 cache。这里虽然使用了 sparse 接口，却没有筛掉任何有效历史，也没有做 Top-K；对齐产生的额外索引以 `-1` 标记，不参与计算。

两条路径最终都返回压缩历史的输出与 LSE。FIA 直接提供 LSE；sparse 接口提供最大分数和缩放后的指数和，封装再通过 `L = m + log(total)` 转换。这样得到的结果就能与前面的“滑窗 + 临时块”使用同一套合并逻辑。

调用 sparse 接口还有一处布局适配：该接口要求单独提供 RoPE query/key，而 DSV4 的 512 维向量已经包含位置编码分量。代码将额外的 query 置零，key 使用目标 cache 的只读切片，使这部分附加点积为零，实际 Attention 分数仍然来自原来的 query/KV。这个处理是为了匹配现成接口，不是 Orthrus 新增的数学步骤，PyPTO 无需照搬。

完整函数还保留了 `ORTHRUS_C128_BACKEND=gather` 路径，用于先收集历史再调用 FIA 的兼容和对照测试。无论选择哪条路径，读取的都是同一批已完成压缩行，draft 不更新它们。

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

CSA 先检索，再对选中的 C4 行计算 Attention。这里有两种 query：前面生成的 Attention query 用于计算最终输出，而 indexer query 用于判断哪些历史行值得读取。`_c4_topk_readonly` 中的 `indexer` 属于 diffusion head，它用自己的投影、RoPE 和 Hadamard 变换生成检索 query；被检索的历史 index key 则来自主模型。

下面的代码同时说明了两侧数据的来源：`target_kv` 提供历史 key 和 scale，`indexer.weights_proj` 根据当前 draft hidden 生成各检索 head 的评分权重。它只运行 query 这一侧，不生成新的历史 key：

```python
# 只读取主模型的 index key/scale，不执行 key compressor 或 cache 写回。
_, _, _, indexer_key, indexer_scale, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
    target_kv, 4
)
# 检索 head 的权重来自 diffusion 分支当前 hidden，不沿用 AR query。
weights = _linear(indexer.weights_proj, hidden_states) * (
    indexer.softmax_scale * indexer.n_heads**-0.5
)
```

这些输入准备好后，QLI 计算检索分数并返回 Top-K 行号。下面的 `query` 指 indexer query，不是送入最终 Attention 的 query。`actual_seq_lengths_key` 给出 anchor 前的原始 token 数，`cmp_ratio=4` 告诉接口如何对应到压缩历史；`sparse_mode=0` 让每个 draft 位置都使用这段完整前缀：

```python
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
    # 这里是 anchor 前的原始 token 数，不是 C4 行数。
    actual_seq_lengths_key=request.seq_lens,
    block_table=request.block_table,
    # metadata 同样以非因果模式生成，不能复用 AR 的逐行因果配置。
    metadata=qli_metadata,
    query_quant_mode=0,
    key_quant_mode=0,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=indexer.index_topk,
    # K 行 query 的可检索边界相同，实际 Top-K 结果仍可各不相同。
    sparse_mode=0,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    # 返回值已经是 C4 压缩行号，后续不能再除以 4。
    cmp_ratio=4,
    return_value=False,
)
```

QLI 还接收一份 metadata，其中包含算子执行所需的长度和布局信息。它需要按同样的非因果规则构造：如果沿用 AR 的配置，算子会按各 query 的先后顺序缩短可读历史，导致块内靠前的 query 少看一部分已经存在的前缀。因此，完整方法既设置调用参数，也重新准备对应的 metadata。它还包含前面提到的 query 投影与变换：

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

QLI 返回后，当前双向块仍不在检索结果里。它本来就对所有 query 可见，已在 SWA 部分参加计算；这里的行号只用来选择 C4 历史。调用方将结果整理成 `[batch, K, topk]`，过滤未完成或越界的压缩行，再把有效行号和目标页表送入 Attention：

```python
# 保留原压缩行号，仅排除负数或 >= floor(anchor/4) 的项。
topk, _ = c4_history_indices(topk.view(batch, BLOCK_SIZE, -1), seq_lens)
# 按页表和选中行号读取 C4，不为每个 query 展开 Top-K KV 大张量。
out, lse = _c4_segment(
    query_bsnd,
    compress_cache,
    compressed_req.block_table,
    topk,
    # Attention 接口此处接收的是压缩行数，与前面 QLI 的长度单位不同。
    torch.div(seq_lens, 4, rounding_mode="floor"),
    draft_impl.softmax_scale,
    validate=validate_cache,
)
```

需要区分两个接口的长度单位：QLI 接收的是原始 token 数，返回值已经是 C4 压缩行号；后面的 Attention 则接收压缩行数。因此代码只对历史长度除以 4，不再对返回的行号做同样处理。

`_c4_segment` 默认调用与 HCA 共用的 `_sparse_history_segment`。HCA 提供全部有效行号，CSA 提供检索选出的行号，后面的分页读取和 Attention 计算方式相同。它返回 C4 部分的输出与 LSE，再与“滑窗 + 临时块”以及 sink 合并，得到这一层最终的 Attention 结果。

过滤产生的无效项以 `-1` 标记并移动到末尾，算子据此排除 padding。完整实现也保留 `ORTHRUS_C4_BACKEND=gather`，用于收集相同行号的 KV 后再调用 FIA，以便做兼容性检查和结果对照。对于 PyPTO，需要保持一致的是 diffusion 参数来源、检索边界、压缩行号含义和最终的归一化，而不必采用完全相同的中间张量组织方式。

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
<summary>完整源码：C4 分页 Attention 与兼容回退</summary>

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

以上四部分的 PyPTO 适配范围可归纳为下表。公共双向块可以直接融合进三条历史路径，因此这里描述的是计算职责，不是必须独立交付的四个 kernel。

| 模块 | 现有实现基础 | 适配内容 |
| --- | --- | --- |
| 公共双向块 | DSpark 块内双向 QK/PV、online-softmax，HCA 状态合并 | 独立临时 KV 输入；K/布局适配；贡献纳入统一归一化，sink 计一次 |
| SWA | 分页滑窗读取、公共历史预取、Attention | 固定 anchor 前窗口；全可见临时块替代逐行因果追加；绕过目标写回 |
| HCA | C128 读取、滑窗/压缩状态合并 | 固定完整压缩行数；增加临时块；跳过 draft compressor/写回并调整依赖 |
| CSA | query 变换、量化评分/Top-K、C4 分页 Attention | diffusion indexer 参数；短/长历史分支统一 anchor 边界；跳过 key/state 写回并加入临时块 |

这些改动之外的模型执行和引擎连接在下一节说明。实现后的验证应分别检查可见范围、分段归一化、分页边界和目标状态不变，再验证完整请求流程。

## 3. 模型执行与推理引擎接入

上面的实现解决了 Attention 计算，但要在服务中运行，还需要把它接到模型的一次完整前向，以及主模型的验证流程中。这里分工是：proposer（草稿提议器）准备输入并产生候选，draft 模型执行各层计算，主模型按原来的因果规则验证候选，推理引擎维护每个请求已经接受的内容和 cache。算子不负责决定接受哪些 token。

**服务启动时，先建立 head 与主模型的关系。** 主模型加载完整权重，引擎为它分配持久 cache；draft 只加载独立 Attention 参数，包括 Q/KV/O 投影、内部 norm、sink 和 CSA indexer。加载后，`bind_shared_backbone(target_model)` 将 draft 绑定到当前主模型：embedding、外层 norm、mHC、MoE 和 LM head 都引用已有模块。这样得到的是两条使用不同 Attention 的计算路径，而不是两份完整模型。

每层 draft 还要知道自己对应主模型的哪一层，以及该层使用的 SWA、压缩 KV、indexer cache 和页表。固定 v0.26 适配通过共享 cache 名称和目标层引用建立这些对应关系，后续执行直接使用目标 cache 的现有布局。共享关系只说明数据在哪里，draft 计算本身仍须保证不写入这些历史。

**一轮投机开始时，proposer 准备 anchor、MASK 和历史信息。** 假设本轮 anchor 的位置是 `a`，输入是这个已知 token 加 31 个 MASK，位置为 `a…a+31`；可读的目标历史则只到 `a-1`。同时，runner 为当前批次提供每个请求的有效长度、页表和位置编码表，这些运行时信息合称 metadata。算子据此区分“缓冲区中有多少空间”和“当前请求实际有多少有效历史”。

这一边界由 `AscendOrthrusProposer.set_inputs_first_pass` 明确设置。它复用现有批量输入的准备过程，再把 cache 可见长度修正为有效前缀，而不是把新构造的 K 行也算进去：

```python
effective_seq_lens = original_seq_lens
# 上轮被拒绝的 token 不属于本轮可读取的历史。
if num_rejected is not None:
    effective_seq_lens = effective_seq_lens - num_rejected
# 历史长度不包含新构造的 K 行草稿输入。
draft_cad.seq_lens = effective_seq_lens
draft_cad._seq_lens_cpu = original_seq_lens_cpu
if hasattr(draft_cad, "seq_lens_cpu"):
    draft_cad.seq_lens_cpu = original_ascend_seq_lens_cpu
draft_cad.max_seq_len = original_max_seq_len
# 块内按非因果方式计算，历史范围仍由前面的长度和页表限制。
draft_cad.causal = False
draft_cad.attn_mask = None
```

如果上一轮出现拒绝，`num_rejected` 会先从历史长度中扣掉。随后 `build_draft_attn_metadata` 根据各 cache group 的实际页表构造逐层信息，让 SWA、C4 和 C128 都指向同一个请求的正确历史。设置 `causal=False` 只改变草稿块内的可见性，并不授权它读取前缀之外的目标数据。

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

**输入就绪后，draft 按层执行前向计算。** 共享 embedding 先把 anchor/MASK 转成 hidden。每一层依次经过共享的 mHC 和外层 norm，再进入独立的 diffusion Attention；Attention 输出回到共享的 mHC、norm 和 MoE，然后继续下一层。这里复用的是参数，draft 输入与主模型输入不同，所以这些模块仍然需要为本轮 draft 重新计算激活。

进入 Attention 后，4096 维 hidden 先由 `wq_a` 投影为 1024 维低秩表示，经内部 norm 和 `wq_b` 得到各 head 的 query；另一条 `wkv` 分支生成 512 维临时 KV。两侧按模型约定处理 norm 和 partial RoPE，再调用第 2 节的历史与双向块计算。输出经过 inverse partial RoPE 和分组输出投影，回到 4096 维 hidden。

这些投影、norm 和位置编码使用已有 DSV4 实现，并不改变它们的数学定义；需要连接的是 diffusion 自己的参数、K 行输入形状以及对应 TP 分片。量化投影按自身参数携带的 scale 执行，符合条件时共用输入量化；这属于已有计算的调用安排，不另算一种 Orthrus Attention 算子。PyPTO 已有的 `qkv_proj_rope.py` 和 `decode_o_proj.py` 也应按同样边界复用，TP8 等配置仍需检查编译形状和通信。

下面摘录 `_attention_readonly` 的主计算过程。`draft_impl` 持有 diffusion 参数，`target_kv` 是当前层的目标历史，`layer_metadata` 提供页表和有效长度；`num_tokens=batch×K`，所以各段输出中的请求、位置和 head 可以一一对应：

```python
# 1. 生成本轮 diffusion query 与临时块 KV。
query, qr, qr_scale, local_kv = _qkv_readonly(
    draft_impl, hidden_states, cos, sin
)
batch = num_tokens // BLOCK_SIZE
# 2. 统一调用前述路径：所有层先算 SWA+块，HCA/CSA 再加对应压缩历史。
outputs, lses = self._history_segments(
    query, hidden_states, qr, qr_scale, target_kv, layer_metadata, layer_name,
    local_kv,
)
# 3. sink 只加入一次；零 value 只影响分母，不增加加权输出分子。
sink_lse = draft_impl.attn_sink.view(1, 1, -1, 1).expand(
    batch, BLOCK_SIZE, -1, -1
)
outputs.append(torch.zeros_like(outputs[0]))
lses.append(sink_lse)
# 4. 根据各段 LSE 做全局归一化，不能直接相加局部输出。
merged = merge_attention_outputs(outputs, lses)
merged = merged.reshape(num_tokens, query.shape[1], query.shape[2])
# 5. 合并完成后才执行 inverse RoPE 和 diffusion 输出投影。
torch.ops._C_ascend.inplace_partial_rotary_mul(
    merged.unsqueeze(1),
    cos,
    -sin,
    rotary_mode="interleave",
    partial_slice=[draft_impl.nope_head_dim, draft_impl.head_dim],
)
output = torch.empty_like(hidden_states)
draft_impl._forward_o_proj(merged, output)
```

`_history_segments` 先调用三类层都需要的“滑窗 + 临时块”计算，再按层类型加入 C4 或 C128 输出。所有结果合并后才做一次逆 RoPE 和输出投影。回到 block 时，代码继续调用 `target_layer` 上的 norm、mHC 和 MoE，因此这些共享模块没有被 Attention head 替代。

下列完整方法保留了投影分支、三类历史的调用、运行上下文读取和 block 衔接。它们依赖固定版本运行时；附录中的完整类定义还包含模型构造和参数加载。

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

**draft 前向结束后，proposer 把候选交给主模型验证。** 最后通过共享的末层 norm 和 LM head，得到 31 个待预测位置的词表分布。greedy 模式选取各位置概率最大的 token；随机模式按 draft 分布采样，并把对应分布一起交给验证器。传递候选 token 之外还要传递 draft 概率，是因为标准 rejection sampling 需要同时使用 draft 分布 `q` 和主模型分布 `p`。

主模型随后按原生 DSV4 的因果 Attention 处理候选块，生成验证概率和自己的 target KV。验证器决定可接受的前缀，并在需要时补采样下一个 token。主模型的验证计算没有改成双向；双向只发生在草稿块里。

**验证结果确定后，引擎更新请求状态，再准备下一轮。** 接受的候选成为输出前缀，对应的 target KV 成为有效历史；拒绝部分不再作为后续可见前缀，其物理页如何保留或复用由引擎管理。本轮 draft KV 只是 Attention 的临时输入，从未写入目标历史，也不会在接受后“转正”。因此，框架需要维护请求长度、页归属和拒绝恢复，而算子需要遵守这些长度和页表，并避免改写目标数据，两者职责不同。

在固定 v0.26 实现中，这条流程涉及以下接入，而不仅是增加一个 Attention 函数：

| 代码范围 | 接入内容 |
| --- | --- |
| vLLM 配置层 | 识别 `method="orthrus"`、head 模型类型和相应投机配置，使引擎能够加载并选择这条路径 |
| vLLM-Ascend | 创建专用 proposer，在加载后绑定共享主干；为混合 Attention 建立共享 cache 与 metadata；将 draft 概率传给现有验证流程 |
| 本项目的模型与 proposer | 实现 K=32 的输入和历史边界、独立 diffusion Attention、候选采样及参数加载；复用目标模型的主干计算和原生验证路径 |

这些接入由 `runtime/vllm_ascend_v0_26/patch_runtime.py` 对固定版本运行时应用，计算逻辑位于本项目的 `dsv4.py`、`dsv4_hybrid.py` 和 `dsv4_proposer.py`。当前使用 eager 正确性路径。PyPTO 后续实现的是第 2 节规定的计算，若要提供完整服务，仍需具备上述输入准备、候选验证和状态管理能力。

运行与验证方式保留在下面，便于复现。现有少卡测试不能代替完整 43 层 TP8 验收，初始化 head 也不能用于判断训练后的接受率或加速收益。

<details>
<summary>运行方式：完整服务与算子级对照</summary>

**完整服务**

先按 `runtime/vllm_ascend_v0_26/README.md` 安装固定版本适配，再在容器内的项目根目录运行。模型和 head 路径按部署环境指定：

```bash
MODEL=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
HEAD=/data/disk/models/orthrus-dsv4-head-v2-w8a8 \
TP=8 \
ORTHRUS_PREFLIGHT_MODE=production \
ORTHRUS_EXECUTION_MODE=eager \
ORTHRUS_DRAFT_SAMPLE_METHOD=probabilistic \
bash scripts/serve_dsv4_orthrus.sh
```

`production` preflight 校验完整 43 层模型与 head 的契约，不代表生产验收已经完成。目标模型单独运行和回退使用 `scripts/serve_dsv4_target.sh`。当前契约固定 K=32；修改 K 时需同步 head、调度、编译形状和临时空间，并重新验证。

**算子级对照**

无需加载完整模型，可以直接调用正文中的计算函数。以下脚本导入服务使用的 Attention 路径，构造合成历史，对照输出、LSE、cache 不变性和局部耗时：

```bash
# --device 指向当前可用的 NPU。
PYTHONPATH=src python scripts/benchmark_dsv4_attention_paths.py --case joined --device npu:0
PYTHONPATH=src python scripts/benchmark_dsv4_attention_paths.py --case sparse --device npu:0
PYTHONPATH=src python scripts/benchmark_dsv4_attention_paths.py --case c128-sparse --device npu:0
PYTHONPATH=src python scripts/check_dsv4_paged_npu.py
```

`joined` 覆盖滑窗/双向块和 sink 合并，`sparse` 覆盖给定行号后的 C4 Attention，`c128-sparse` 与分页脚本覆盖 C128 的两种读取方式。QLI 检索需另用 `scripts/check_dsv4_indexer_npu.py` 检查，再通过整层服务验证参数和状态的连接。分页脚本固定使用可见设备中的 `npu:0`，应在设备已经分配好的环境中运行。

这些 Attention 测试使用 BF16 Q/KV，调用的是正文中的同一组 FIA、分页 sparse Attention 和原生 LSE 合并函数。量化投影和 QLI 的输入格式另按各自接口处理。脚本保留 gather 等对照路径，用来检查同一数学结果；合成历史测试不能代替真实服务验收。

</details>

**验证结果与适用范围**

已在 A2 上验证双向块、SWA/C4/C128 历史读取、C4 indexer 检索，以及 LSE 和 sink 合并。测试覆盖空历史、分页边界和多请求等情况，计算结果在设定容差内与独立 FP32 参考一致，检查范围内未发现 draft 改写主模型 cache。

推理接入已在四层和 12 层真实截取模型上，使用初始化 head、双卡 TP2、K=32 和 eager 模式跑通。已检查候选生成与验证、随机概率传递、请求换序、KV 页复用、取消和恢复；其中四层测试用受控候选覆盖不同接受情况，用故障注入验证抢占恢复。跨批次的旧概率残留已修复，修复前后输出一致；这些结果不代表训练后 head 的接受率或加速收益。

完整 43 层 Orthrus 已在八卡上跑通启动、重启及 32K/128K 输入请求，但跨服务和并发输出对照仍有差异，尚未通过完整正确性验收。双卡定位的一个反例已脱离模型独立复现：原生 C4 按物理地址装载 KV，换页布局会产生一个 BF16 间隔的输出差异，统一物理布局后差异消失；这不能替代完整模型的归因。graph、1M 上下文和持续负载也未验收。以上是 Torch-NPU 实现的验证，PyPTO 仍需单独适配。详细结果保留在 `DSV4_VALIDATION_20261028.md`，复现方式见 `DSV4_VALIDATION_RUNBOOK.md`。

## 附录：完整实现

以下保留计算模块、公共辅助函数、proposer 和启动脚本的完整源码。它们共同构成这里展示的服务实现，运行时还需要安装固定版本引擎适配和 vendor 算子，并提供匹配的主模型与 head 权重。

<details>
<summary>完整源码 1/4：dsv4.py</summary>

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
<summary>完整源码 2/4：dsv4_hybrid.py</summary>

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
<summary>完整源码 3/4：dsv4_proposer.py</summary>

```python
"""Dedicated vLLM-Ascend proposer for DSV4 Orthrus parallel drafting."""

from __future__ import annotations

import copy
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

from .head_contract import BLOCK_SIZE, NUM_SPECULATIVE_TOKENS
from .draft_sampling import sample_seeded_parallel_draft
from .trace import instant


class AscendOrthrusProposer(AscendDSparkProposer):
    """Reuse Ascend's block-expansion buffers, not DSpark generation semantics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner=runner)
        if self.method not in {"orthrus", "dflash"}:
            raise ValueError(f"AscendOrthrusProposer received method={self.method!r}")
        if self.num_speculative_tokens != NUM_SPECULATIVE_TOKENS:
            raise ValueError("DSV4 Orthrus requires 31 speculative tokens")
        if self.sample_from_anchor is not False or self.num_query_per_req != BLOCK_SIZE:
            raise ValueError(
                "DSV4 Orthrus requires anchor plus 31 masked query positions"
            )
        if not self.speculative_config.parallel_drafting:
            raise ValueError("DSV4 Orthrus requires parallel_drafting=true")
        if self.speculative_config.rejection_sample_method != "standard":
            raise ValueError("DSV4 Orthrus requires standard rejection sampling")
        self.dynamic_spec = None
        # Eager is the correctness default. Graph support is enabled only after
        # the exact path passes the NPU test matrix.
        self.use_cuda_graph = False

    def model_returns_tuple(self) -> bool:
        # The generic base treats unknown methods as recurrent MTP models.
        # Orthrus emits one hidden tensor for the entire parallel block.
        return False

    def take_last_draft_probs(self):
        probabilities = super().take_last_draft_probs()
        instant("orthrus.dsv4.draft_probs", {
            "present": probabilities is not None,
            "shape": list(probabilities.shape) if probabilities is not None else None,
        })
        return probabilities

    def _sample_from_logits(self, logits, sampling_metadata):
        if (not self._enable_probabilistic_draft_probs
                or sampling_metadata.all_greedy or not sampling_metadata.generators):
            return super()._sample_from_logits(logits, sampling_metadata)
        trace_sampling = os.environ.get("ORTHRUS_TRACE_SAMPLING") == "1"
        if trace_sampling:
            generators = sampling_metadata.generators
            before = {i: [g.initial_seed(), g.get_offset()]
                      for i, g in generators.items()}
        result = sample_seeded_parallel_draft(
            logits, sampling_metadata.temperature, sampling_metadata.generators,
            self.num_speculative_tokens, use_fp64_gumbel=self.use_fp64_gumbel,
        )
        if trace_sampling:
            ids, probabilities = result
            instant("orthrus.dsv4.sampling", {
                "request_ids": list(self.runner.input_batch.req_ids),
                "generators_before": before,
                "generator_offsets_after": {i: g.get_offset() for i, g in generators.items()},
                "draft_ids": ids.view(-1, self.num_speculative_tokens).cpu().tolist(),
                "selected_probabilities": probabilities.gather(1, ids.long()[:, None])
                    .view(-1, self.num_speculative_tokens).cpu().tolist(),
            })
        return result

    def set_inputs_first_pass(self, *args, **kwargs):
        cad = kwargs.get("cad")
        if cad is None:
            raise TypeError("Orthrus set_inputs_first_pass requires cad")
        original_seq_lens = cad.seq_lens
        original_seq_lens_cpu = cad._seq_lens_cpu
        original_ascend_seq_lens_cpu = getattr(cad, "seq_lens_cpu", None)
        original_max_seq_len = cad.max_seq_len
        num_rejected = kwargs.get("num_rejected_tokens_gpu")
        result = super().set_inputs_first_pass(*args, **kwargs)
        num_tokens, token_indices, draft_cad, long_seq = result

        # DSpark appends the block to its writable draft cache. Orthrus reads
        # committed target history and computes the K32 local block separately,
        # so cache-visible sequence lengths must never include the query block.
        effective_seq_lens = original_seq_lens
        if num_rejected is not None:
            effective_seq_lens = effective_seq_lens - num_rejected
        draft_cad.seq_lens = effective_seq_lens
        draft_cad._seq_lens_cpu = original_seq_lens_cpu
        if hasattr(draft_cad, "seq_lens_cpu"):
            draft_cad.seq_lens_cpu = original_ascend_seq_lens_cpu
        draft_cad.max_seq_len = original_max_seq_len
        draft_cad.causal = False
        draft_cad.attn_mask = None
        return num_tokens, token_indices, draft_cad, long_seq

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        del num_actual_tokens
        if not self.draft_attn_groups:
            raise RuntimeError("Orthrus target cache groups were not initialized")
        per_layer: dict[str, Any] = {}
        shared_prefill_metadata: dict[str, Any] = {}
        shared_decode_metadata: dict[str, Any] = {}
        shared_dsa_metadata: dict[str, Any] = {}
        for group in self.draft_attn_groups:
            builder = group.get_metadata_builder()
            metadata_input = copy.copy(common_attn_metadata)
            gid = group.kv_cache_group_id
            block_table = self._per_group_block_table_buffers.get(gid)
            if block_table is None:
                raise RuntimeError(f"Orthrus has no block table for cache group {gid}")
            metadata_input.block_table_tensor = block_table[: metadata_input.num_reqs]
            slot_mapping = self._per_group_query_slot_mapping_buffers.get(gid)
            if slot_mapping is not None:
                metadata_input.slot_mapping = slot_mapping[:num_input_tokens]
            extra = (
                {
                    "prefill_ratio_to_sas_metadata": shared_prefill_metadata,
                    "decode_ratio_to_sas_metadata": shared_decode_metadata,
                    "common_ratio_to_sas_metadata": shared_dsa_metadata,
                    "block_size": group.kv_cache_spec.block_size,
                }
                if self.use_compress
                else {}
            )
            metadata = builder.build(
                0,
                metadata_input,
                self.runner.get_model(),
                **extra,
            )
            if hasattr(metadata, "causal"):
                metadata.causal = False
            if hasattr(metadata, "attn_mask"):
                metadata.attn_mask = None
            for layer_name in group.layer_names:
                per_layer[layer_name] = metadata
        first = self.draft_attn_groups[0].layer_names[0]
        return [per_layer], per_layer[first]

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        **kwargs,
    ) -> None:
        return super().dummy_run(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            **kwargs,
        )
```

</details>

<details>
<summary>完整源码 4/4：serve_dsv4_orthrus.sh</summary>

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
