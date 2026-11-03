# Orthrus-GLM5.3-Flash NPU 推理适配与算子设计

## 1. 背景与适配目标

Orthrus 在原模型中增加独立的 draft Attention 参数，一次提出多个候选 token，再由原模型验证。将这套方法用于 GLM5.3-Flash，既要处理稀疏 MLA 的历史检索，也要处理 KDA 的递推状态，不能直接照搬普通 Transformer 的双向 SDPA。

本文面向 GLM5.3-Flash 的文本推理，依次说明模型结构、各计算模块的公式、PyPTO 现有基础、所需适配和可运行的 Torch-NPU 实现。范围不包含训练、在线更新或多模态；Torch-NPU 验证也不等于 PyPTO 已实现 Orthrus。

### 1.1 草稿分支与主模型的关系

GLM5.3-Flash 共 45 层，排列为 `(KDA x 3 -> 稀疏 MLA) x 11 -> KDA`，即 34 层 KDA、11 层 MLA。MLA 位于从 0 开始计数的第 3、7、…、43 层。这里没有 DSV4 的 SWA、C4、C128 三类层。

```text
共享 Embedding
    |
    v
[ KDA -> KDA -> KDA -> 稀疏 MLA ] x 11 -> KDA
    |
    | 每层共享 mHC / 外层 norm，mixer 使用独立 draft 参数：
    |   KDA：AR 卷积尾部、递推初态的副本 -> 临时块递推
    |   MLA：draft indexer 查询历史池 -> 选中历史 + 全部临时块
    |
    | 共享 mHC / norm / Dense MLP 或 MoE
    v
共享 Final Norm / LM head -> 候选 -> 原模型验证
```

每轮输入为 `anchor + (K-1) 个 MASK`。anchor 是位置 $a$ 上已知的最后一个 token，MASK 是未知位置的占位符；读取的主模型历史止于 $[0,a)$。块长 K 包含 anchor，候选数为 K-1。当前原生 KDA 接口限制本接入使用 $K\le8$，这不是 Orthrus 或 PyPTO 的数学限制，单独 MLA 的 K32 用例也不能证明完整模型已支持 K32。

当前保留两种明确区分的实验策略：**因果 KDA + 双向 MLA**，以及**门控双向 KDA + 双向 MLA**。前者在 AR 状态副本上向前递推；后者增加从零状态出发的反向块，在输出归一化前融合。两者需要各自匹配的 head，均不能表述为原作者已经训练并确认的 GLM head。下文先说明公共需求，再给出反向分支的增量。

### 1.2 现有推理路径的适配边界

本文固定参考 `pypto-lib@2aef4534f8769d8ce8d49d9dde7e5aeebc472197` 和 `pypto-serving@5f548be23d7af120b20a40694601f5bb5b8b7a90`。下文库内文件名均相对于 `models/glm5_3_flash/`。该库已有 KDA、KPool、稀疏 MLA、mHC、MoE、量化投影和整网前向编排，不是从零开发 GLM。

但库内文档提到的 `pypto_serving.model.glm53.npu_runner.Glm53ModelRunner`，在所核对的 Serving 公开主线版本中没有对应模块。因此需要区分“已有模型计算代码”和“已有可直接部署的服务”，不对其他分支或未公开代码作推断。

**Attention 侧按三个计算入口组织需求：KPool 检索、历史与临时块联合 MLA、KDA 块执行。** 这不等于三个全新 kernel：检索和因果 KDA 主要复用现有核心，MLA 需要扩展联合归一化与两套 KV 展开。双向临时块属于 MLA 内部计算，不必再独立列作第四个算子。

## 2. Attention 模块设计

### 2.1 KPool：用 draft query 检索主模型历史

#### 2.1.1 原有检索与 draft 历史边界

KPool 不直接计算最终 Attention，而是先选择值得读取的历史 token。模型每四个历史位置生成一个池 key；这是主模型已有的压缩结果，不是简单均值，也不需要 draft 重新压缩历史。

先定义原有打分。设 AR 位置 $p$ 的可见长度为 $b_p$，完整池数为 $P_p=\lfloor b_p/4\rfloor$。$u_{ph}^{AR}$ 为 indexer 第 h 个 head 的 query，$\omega_{ph}^{AR}$ 为包含缩放的 head 权重，$\kappa_r^{AR}$ 为历史池 key，则：

$$
s_{pr}^{AR}=\sum_{h=1}^{32}\omega_{ph}^{AR}\,\mathrm{ReLU}\left((u_{ph}^{AR})^\top\kappa_r^{AR}\right),\qquad 0\le r<P_p.
$$

Orthrus 用 draft 的 query 和 head 权重打分，所有行的池数固定为 anchor 之前的 $P=\lfloor a/4\rfloor$：

$$
s_{ir}^{D}=\sum_{h=1}^{32}\omega_{ih}^{D}\,\mathrm{ReLU}\left((u_{ih}^{D})^\top\kappa_r^{AR}\right),\qquad 0\le r<P.
$$

$u$ 和 $\kappa$ 均为 128 维，$\omega$ 已包含 $128^{-1/2}32^{-1/2}$ 缩放。令 $m=\min(P,512)$，选中的池集合及展开后的历史 token 集合为：

$$
\mathcal R_i=\mathrm{TopK}_{m}(s_i^{D}),\qquad \mathcal H_i=\bigcup_{r\in\mathcal R_i}\{4r,4r+1,4r+2,4r+3\}\cup\{4P,\ldots,a-1\}.
$$

最后一项是不足一池的尾部；无尾部时为空。每行最多读取 2048 个池展开位置和 3 个尾部位置。各行可以选出不同的池，但都不能超过同一个历史边界；当前临时块随后全部参与 MLA，不经过历史 Top-K。

#### 2.1.2 PyPTO 实现基础与适配方案

`indexer.py` 的 `indexer_score / indexer_topk / indexer_expand` 已支持多行 query、逐行池数和历史尾部，读池与写池分离。`indexer_score` 用 `seg_start / pool_count` 描述历史，`indexer_expand` 接受 `tail_start / tail_count / kv_len`。同一 draft 请求将这些边界统一为 $a$ 即可，不需要重写 Top-K 算法。

需要补的是 draft 查询编排：使用 draft 的 `w_q_b` 和 head-weight 投影，不调用 `indexer_cache_write / indexer_pool_write`。现有 `indexer_proj` 同时算 query、key 和 gate，首版可复用后丢弃不用的输出，再抽出 query-only 投影；已有 `skip_query` 恰好是跳过 query，不是所需的 query-only 开关。

实现还需处理两个接口约束。第一，现有投影要求行数按 16 对齐，短块 padding 必须排除在有效 query 之外，相关 score tile 也需按实际 K 检查。第二，PyPTO 使用 Hadamard 变换后的 INT8 池 key、FP32 scale 和 query 量化，而当前 vLLM 路径读取未打包的 BF16 池。两者不能直接交换 cache 指针，须提供布局转换或适配入口，并核对缩放只计一次、量化后的 Top-K 选择符合约定。

#### 2.1.3 Torch-NPU 实现

现有 `select_history` 的主体就是上面的池打分和索引展开。输入 query 为 `[K,32,128]`，head 权重为 `[K,32]`，历史池为 `[P,128]`：

```python
# ORTHRUS：所有 query 共用 anchor 前的历史池，不写入新池。
scores = torch.einsum("qhd,pd->qhp", query.float(), pooled_keys.float())
scores = (scores.relu() * head_weights.float().unsqueeze(-1)).sum(1)
chosen = scores.topk(min(complete, topk // pool_size), dim=-1).indices
offsets = torch.arange(pool_size, device=query.device)
indices = (chosen.unsqueeze(-1) * pool_size + offsets).flatten(1)
tail = torch.arange(complete * pool_size, prefix_length, device=query.device)
return torch.cat((indices, tail.expand(query.shape[0], -1)), dim=-1)
```

这里先逐 head 做 ReLU，再加权求和，与所参考 PyPTO 的打分定义一致。固定 vLLM-Ascend 镜像另有先合并 query head 再打分的 Triton 入口，两者并不等价，不能只因它是融合算子就替换进 draft。主模型仍保持原生路径，draft head 需匹配本节公式。

<details>
<summary>完整函数：历史池打分与索引展开</summary>

```python
def select_history(query, head_weights, pooled_keys, prefix_length, *, topk, pool_size=4):
    """每行选择完整历史池，展开成 token 位置，再追加未满池的历史尾部。

    query: [Q, index_heads, index_dim]；head_weights 已包含 indexer 的缩放。
    pooled_keys 只包含 anchor 之前的完整池，不含临时 draft 的 key。
    """
    if (query.ndim != 3 or head_weights.shape != query.shape[:2]
            or pooled_keys.ndim != 2 or pooled_keys.shape[-1] != query.shape[-1]
            or type(prefix_length) is not int or prefix_length < 0
            or pool_size < 1 or topk < 1 or topk % pool_size):
        raise ValueError("invalid KPool read-only query contract")
    complete = prefix_length // pool_size
    if pooled_keys.shape[0] != complete:
        raise ValueError("pooled keys must end at the committed prefix")
    # ORTHRUS：所有 query 的历史边界相同，不随块内位置向右推进。
    scores = torch.einsum("qhd,pd->qhp", query.float(), pooled_keys.float())
    scores = (scores.relu() * head_weights.float().unsqueeze(-1)).sum(1)
    chosen = scores.topk(min(complete, topk // pool_size), dim=-1).indices
    offsets = torch.arange(pool_size, device=query.device)
    indices = (chosen.unsqueeze(-1) * pool_size + offsets).flatten(1)
    tail = torch.arange(complete * pool_size, prefix_length, device=query.device)
    return torch.cat((indices, tail.expand(query.shape[0], -1)), dim=-1)
```

</details>

### 2.2 MLA：历史与双向临时块共同归一化

#### 2.2.1 原有计算与联合 Attention

MLA 的历史保存 512 维 latent，而不是展开后的逐 head K/V。AR 用自身投影读取历史；draft 则用自己的 query 读取选中的 AR 历史和整个临时块。由于历史与临时块来自不同参数，不能默认二者使用同一套 KV 展开权重。

按单个 head 书写，令 $c_j^{AR},c_t^D\in\mathbb R^{512}$ 分别为历史与临时 latent，$W_K^{AR},W_K^D,W_V^{AR},W_V^D\in\mathbb R^{256\times512}$ 为相应展开权重。原有 AR 的 query 为 $q_p^{AR}\in\mathbb R^{256}$，可见集合为 $\mathcal H_p^{AR}$，计算为：

$$
z_{pj}^{AR}=(q_p^{AR})^\top W_K^{AR}c_j^{AR}/\sqrt{256}.
$$

$$
o_p^{AR}=\frac{\sum_{j\in\mathcal H_p^{AR}}e^{z_{pj}^{AR}}W_V^{AR}c_j^{AR}}{\sum_{j\in\mathcal H_p^{AR}}e^{z_{pj}^{AR}}}.
$$

Orthrus 的 $q_i^D\in\mathbb R^{256}$ 由 draft 投影生成。为区分输入来源，将历史段和临时块段的分数分别记为 $z_{ij}^{H}$、$z_{it}^{B}$：

$$
z_{ij}^{H}=(q_i^D)^\top W_K^{AR}c_j^{AR}/\sqrt{256},\qquad z_{it}^{B}=(q_i^D)^\top W_K^Dc_t^D/\sqrt{256}.
$$

联合输出为：

$$
o_i^D=\frac{\sum_{j\in\mathcal H_i}e^{z_{ij}^{H}}W_V^{AR}c_j^{AR}+\sum_{t=0}^{K-1}e^{z_{it}^{B}}W_V^Dc_t^D}{\sum_{j\in\mathcal H_i}e^{z_{ij}^{H}}+\sum_{t=0}^{K-1}e^{z_{it}^{B}}}.
$$

临时块中所有 $t$ 都可见，历史仍止于 anchor 前。这里没有 DSV4 的 Attention sink，也没有 RoPE。吸收 K 展开后虽然 query 变为 512 维，缩放仍按原始 256 维 query 计算。

当前实现分别调用现成 Attention 计算两段，再按同一个 softmax 分母合并。若两段返回局部输出 $o_{i,H},o_{i,B}$ 和自然对数 LSE $L_{i,H},L_{i,B}$，则：

$$
L_i=\log\left(e^{L_{i,H}}+e^{L_{i,B}}\right),\qquad o_i^D=e^{L_{i,H}-L_i}o_{i,H}+e^{L_{i,B}-L_i}o_{i,B}.
$$

空历史取输出零、LSE 为负无穷。两段输出要先经过各自 $W_V$，进入相同的 256 维 value 空间，再合并；不能先合并 512 维 latent 输出，最后统一乘 draft 的 $W_V$。分段是现成算子的组合方式，不是算法规定必须调用两个 kernel。

#### 2.2.2 PyPTO 实现基础与适配方案

`attention.py::decode_sparse_attn` 已能根据逐 query 索引读取分页 latent，支持多行 query，且自身只读 cache。写入由独立的 `mla_cache_write` 完成；它不是一个必须先删除内部因果 mask 才能使用的核心。

缺口在于当前入口只接一类 latent，只返回归一化后的 latent 输出。Orthrus 需要增加临时块分支、分别使用两套 KV 展开，并保留联合归一化所需的统计。建议的逻辑接口如下，名称不限定最终实现：

```text
draft_sparse_mla(
  q_history_absorbed[K,H,512], q_block_absorbed[K,H,512],
  ar_latent_pages, block_table, request_ids, history_indices[K,S],
  block_latent[K,512], W_V_AR[H,256,512], W_V_D[H,256,512]
) -> attention[K,H,256]
```

现有分块 softmax 内部已经维护最大分数 $m_r$、移位指数和 $\ell_r$ 和未归一化 latent 和 $u_r$。下标 $H,B$ 表示历史与临时块。保留这些量即可完成：

$$
m=\max(m_H,m_B),\qquad \alpha_H=e^{m_H-m},\qquad \alpha_B=e^{m_B-m}.
$$

$$
\ell=\alpha_H\ell_H+\alpha_B\ell_B,\qquad u=\alpha_H W_V^{AR}u_H+\alpha_B W_V^Du_B,\qquad o=u/\ell.
$$

$u_H,u_B$ 为 512 维，合并后的 $u$ 为 256 维。这与上面的 LSE 公式等价，可在同一计算图中完成，也可扩展原入口返回统计后再合并，不必为此强制新增独立的 LSE kernel。空段的权重置零，避免对负无穷做未定义运算。

输出编排也要调整：现有 `mla_epilog_decode` 将 AR 的 value 展开吸收进 `w_o_absorbed`，不能用这一套吸收权重统一处理两段。得到共同 value 输出后，可复用 `mla_epilog_prefill / mla_epilog_w8a8`，使用 draft 的实际 O 投影。

#### 2.2.3 Torch-NPU 实现

当前计算不展开整段历史的逐 head K/V，而是把 K 展开吸收到 query，再调用 `npu_fused_infer_attention_score`（FIA）计算 latent Attention。历史集合因 query 而异，按行读取；临时块所有 query 共用，只传一份 K/V。

```python
# selected 为每行检索到的历史 latent；block 是整个临时块。
selected = gather_history(history, indices)
q_ar = torch.einsum("qhd,hdr->qhr", query, ar_key)
q_draft = torch.einsum("qhd,hdr->qhr", query, draft_key)
ar_out, ar_lse = _segment(q_ar, selected, dim**-.5, backend)
draft_out, draft_lse = _block_segment(q_draft, block, dim**-.5, backend)
# 先按各自权重进入同一个 value 空间，再按 LSE 合并。
ar_out = torch.einsum("qhr,hvr->qhv", ar_out, ar_value)
draft_out = torch.einsum("qhr,hvr->qhv", draft_out, draft_value)
return merge_attention_outputs(
    (ar_out, draft_out), (ar_lse.unsqueeze(-1), draft_lse.unsqueeze(-1)),
)
```

变量 `q_ar` 仍来自 draft query，只是吸收了 AR 的 K 展开；它不是另一次主模型前向。FIA 返回局部输出与 LSE，合并封装在 NPU 路径调用 `npu_attention_update`；空历史另走安全分支。下面保留完整计算函数，包括 Torch 参考分支和空段处理。

<details>
<summary>完整函数：两段 Attention、value 展开与联合归一化</summary>

```python
def _segment(query, latent, scale, backend):
    """[Q,H,R] 对每行自己的 [Q,S,R] 计算输出及自然对数 LSE。"""
    if latent.shape[1] == 0:
        return torch.zeros_like(query), query.new_full(query.shape[:2], -torch.inf, dtype=torch.float32)
    if backend == "npu":
        import torch_npu

        latent = latent[:, :, None].contiguous()
        output, lse = torch_npu.npu_fused_infer_attention_score(
            query[:, None].contiguous(), latent,
            latent, num_heads=query.shape[1],
            num_key_value_heads=1, input_layout="BSND", sparse_mode=0,
            scale=scale, softmax_lse_flag=True,
            actual_seq_lengths=[1] * query.shape[0],
            actual_seq_lengths_kv=[latent.shape[1]] * query.shape[0],
        )[:2]
        return output[:, 0], lse.reshape(query.shape[:2]).float()
    if backend != "torch":
        raise ValueError("backend must be torch or npu")
    scores = torch.einsum("qhr,qsr->qhs", query.float(), latent.float()) * scale
    lse = scores.logsumexp(-1)
    output = torch.einsum("qhs,qsr->qhr", scores.softmax(-1), latent.float())
    return output.to(query.dtype), lse


def _block_segment(query, block, scale, backend):
    """所有 query 共读同一个双向块，不复制 K 份临时 KV。"""
    if backend != "npu":
        return _segment(query, block.expand(query.shape[0], -1, -1), scale, backend)
    import torch_npu

    latent = block[None, :, None].contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query[None].contiguous(), latent, latent,
        num_heads=query.shape[1], num_key_value_heads=1, input_layout="BSND",
        sparse_mode=0, scale=scale, softmax_lse_flag=True,
        actual_seq_lengths=[query.shape[0]], actual_seq_lengths_kv=[block.shape[0]],
    )[:2]
    return output[0], lse.reshape(query.shape[1], query.shape[0]).T.float()


def sparse_draft_attention(query, history, block, indices, *, ar_key, ar_value,
                           draft_key, draft_value, backend="torch"):
    """稀疏 MLA：主模型低秩历史 + 完整临时双向块，返回 [Q,H,V]。

    query=[Q,H,D]；history=[L,R]；block=[Q,R]；indices=[Q,S]。
    key/value 权重分别为 [H,D,R] / [H,V,R]，不要求 AR 与 draft 相同。
    """
    if (query.ndim != 3 or history.ndim != 2 or block.ndim != 2
            or block.shape[0] != query.shape[0] or history.shape[1] != block.shape[1]
            or indices.ndim != 2 or indices.shape[0] != query.shape[0]):
        raise ValueError("invalid sparse draft tensor layout")
    q_rows, heads, dim = query.shape
    latent_dim = block.shape[1]
    if (ar_key.shape != (heads, dim, latent_dim) or draft_key.shape != ar_key.shape
            or ar_value.shape[0] != heads or ar_value.shape[-1] != latent_dim
            or draft_value.shape != ar_value.shape):
        raise ValueError("incompatible AR/draft latent expansion weights")
    # ORTHRUS：历史按 AR 的 KV 展开权重解释，临时块按 draft 的权重解释。
    selected = gather_history(history, indices)
    q_ar = torch.einsum("qhd,hdr->qhr", query, ar_key)
    q_draft = torch.einsum("qhd,hdr->qhr", query, draft_key)
    ar_out, ar_lse = _segment(q_ar, selected, dim**-.5, backend)
    draft_out, draft_lse = _block_segment(q_draft, block, dim**-.5, backend)
    ar_out = torch.einsum("qhr,hvr->qhv", ar_out, ar_value)
    draft_out = torch.einsum("qhr,hvr->qhv", draft_out, draft_value)
    # 两段采用同一个 softmax 分母，不是各自归一化后直接相加。
    if backend == "npu" and indices.shape[1]:
        from .dsv4_hybrid import merge_attention_outputs

        return merge_attention_outputs((ar_out, draft_out),
                                       (ar_lse.unsqueeze(-1), draft_lse.unsqueeze(-1)))
    total_lse = torch.logaddexp(ar_lse, draft_lse)
    output = (ar_out.float() * (ar_lse - total_lse).exp().unsqueeze(-1)
              + draft_out.float() * (draft_lse - total_lse).exp().unsqueeze(-1))
    return output.to(query.dtype)


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

`gather_history` 可以接独立连续快照，也可接本轮借用的 `PagedLatent`：根据逻辑索引和页表，只收集命中的历史行。当前自动策略仅在足够长的历史采用后者，短历史保留快照；这属于数据读取编排，不改变检索集合或 Attention 公式。PyPTO 已有分页读取基础，可以直接在 kernel 中访问选中页，不必照搬 Torch 的中间物化方式。

### 2.3 KDA：带历史初值的临时块执行

#### 2.3.1 原有递推与两种 draft 策略

KDA 的历史是宽度 4 卷积保留的三行输入，以及每个 head 的 $128\times128$ 递推矩阵，而不是逐 token KV。draft 使用自己的投影，读取 AR 状态副本，计算结束后不写回主模型。

省略 head 下标，令 $q_i,k_i,v_i\in\mathbb R^{128}$ 为投影经因果逐通道卷积和 SiLU 后的向量，$\widehat q_i=\mathrm{L2Norm}(q_i)/\sqrt{128}$、$\widehat k_i=\mathrm{L2Norm}(k_i)$。$f_i,b_i$ 是衰减和更新门的原始投影，$A_{\log},d_t$ 是学习参数，则：

$$
\gamma_i=-5\,\sigma\left(e^{A_{\log}}\odot(f_i+d_t)\right),\qquad \beta_i=\sigma(b_i).
$$

$\sigma$ 是 sigmoid，$\gamma_i$ 是逐 key 通道衰减，不是 Qwen GDN 的标量衰减。状态采用 `[V,K]` 布局，单步计算为：

$$
\overline S_i=S_{i-1}\,\mathrm{diag}(e^{\gamma_i}),\qquad \delta_i=\beta_i(v_i-\overline S_i\widehat k_i).
$$

$$
S_i=\overline S_i+\delta_i\widehat k_i^\top,\qquad y_i^{\rightarrow}=S_i\widehat q_i.
$$

AR 用主模型参数计算并保留正式状态；因果 draft 保持同一公式，但以 $S_{-1}^{D}=\mathrm{copy}(S_{a-1}^{AR})$ 和历史卷积尾部副本为初态，在工作区连续推进 K 行。

门控双向策略额外将**当前块的原始投影输入倒序**，用零卷积历史和零矩阵状态执行同一递推，再把输出翻回原顺序，记作 $y_i^{\leftarrow}$。令 $\rho$ 为每 head 的反向门，则：

$$
\widetilde y_i=y_i^{\rightarrow}+\tanh(\rho)\odot y_i^{\leftarrow}.
$$

因果策略直接取 $\widetilde y_i=y_i^{\rightarrow}$。两种策略随后使用相同输出处理：

$$
o_i=W_O^D\,\mathrm{ConcatHeads}\left(\mathrm{RMSNorm}_{w^D}(\widetilde y_i)\odot\sigma(g_i)\right).
$$

$g_i$ 是输出门投影，$w^D$ 为该 norm 的权重；GLM 的输出门为 sigmoid，不是 Qwen 的 SiLU。反向只遍历当前块，不反向重算历史，融合发生在 gated norm 之前。$\rho=0$ 时退化为因果策略，验证反向是否真正生效应使用非零门。

#### 2.3.2 PyPTO 实现基础与适配方案

`kda.py` 的 `kda_conv_prefill` 已接收 `query_start_loc / state_rows`，能消费带历史尾部的块；`prefill_kda` 支持非零矩阵初态、变长序列和 16 行 chunk。因此因果 draft 主要是把历史状态换成私有副本，再复用 prefill 核心和 `kda_output`，不需要新发明递推算法。

不能将 K 行都指向同一 state row，再一次调用 `decode_kda`：decode 中每行是独立任务，这样会竞争同一个状态，而不是按序递推。应使用块 prefill，或者严格顺序编排单步入口，并核对状态布局、门函数、归一化 epsilon 与有效长度。

双向策略增加倒序打包、零状态初始化、第二次卷积/递推和门控融合的编排。反向输入必须在卷积前倒序，不能把正向卷积结果简单翻转代替；是否进一步融合由实测决定。K 变化涉及 tile、padding 与编译形状，不等于每换一个 K 就重写 KDA 算法。

#### 2.3.3 Torch-NPU 实现

当前递推复用与目标 GLM 相同的 `run_recurrent_kda`，先分配私有状态槽并复制初态。卷积由 `kda_convolve` 选择现成的原生因果卷积或 `conv1d + SiLU`，两者都不把 AR cache 传给可写参数。完整分支中的两种策略如下：

```python
# run 内部执行当前块卷积和 KDA；state/tail 均只作历史输入。
output = run(rows, tail, state, gate, beta)
if bidirectional:
    reverse = run(
        rows.flip(0), torch.zeros_like(tail), torch.zeros_like(state),
        gate.flip(0), beta.flip(0),
    )
    output = output + self.backward_gate.tanh()[None, :, None] * reverse.flip(0)
out_gate = self.g_b(g_a).reshape(-1, 8, 128).sigmoid()
return self.kda_o((self.kda_norm(output) * out_gate).flatten(1))
```

这里的 8 heads 是 demo 对应的 TP8 单分片规格，不是 KDA 的通用常量。当前原生接口可直接接收卷积结果的 Q/K/V 切片，不必全部复制为连续张量；其他参数仍遵守接口要求。卷积自动选择也以已测形状为依据：小 head 数时组合调用可能更快，不把“融合”直接等同于性能更好。

<details>
<summary>完整函数：卷积、Torch 递推参考与原生 KDA 调用</summary>

```python
def kda_convolve(rows, tail, weight, *, backend="torch"):
    """宽度 4 的 QKV 卷积；历史只读，融合路径只写临时状态。"""
    if rows.ndim != 2 or tail.shape != (3, rows.shape[1]) or weight.shape != (rows.shape[1], 4):
        raise ValueError("KDA convolution expects rows[K,C], tail[3,C], weight[C,4]")
    if backend == "auto":
        # 实测 32/64 heads 的融合调用更快，8 heads 的分配/调度成本反而更高。
        backend = "npu" if (rows.device.type == "npu" and rows.dtype == torch.bfloat16
                              and rows.shape[1] in (3 * 32 * 128, 3 * 64 * 128)
                              and 1 <= rows.shape[0] <= 8) else "torch"
    if backend == "torch":
        stream = torch.cat((tail, rows), 0).T[None].float()
        output = F.conv1d(stream, weight.float()[:, None], groups=rows.shape[1])
        return F.silu(output).to(rows.dtype)[0].T
    if backend != "npu" or rows.dtype != torch.bfloat16 or not 1 <= rows.shape[0] <= 32:
        raise ValueError("native KDA convolution requires BF16 and 1..32 rows")
    from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d

    # 原生 GLM 的 prefill 卷积可以消费一个带初始尾部的短块。
    # 槽 0 保留为 null，所有可写状态都由本次 draft 分配。
    state = rows.new_zeros((2, 3, rows.shape[1]))
    state[1].copy_(tail)
    # 原生 GLM 将卷积参数存为 FP32，调用融合算子前需与激活精度一致。
    return causal_conv1d(
        rows.contiguous(), weight.to(rows.dtype).T.contiguous(), state,
        torch.tensor([0, rows.shape[0]], dtype=torch.int32, device=rows.device),
        torch.ones(1, dtype=torch.int32, device=rows.device), run_mode=0,
        initial_state_mode=torch.ones(1, dtype=torch.bool, device=rows.device),
    )


def kda_reference(query, key, value, raw_gate, raw_beta, initial, a_log, dt_bias,
                  *, lower_bound=-5.):
    """因果 KDA 参考计算；状态为 [H,V,K]，仅返回临时状态，不写 initial。

    这不是“双向 KDA”的定义，不能把它称为普通 SDPA 的换 mask 版本。
    """
    q = F.normalize(query.float(), dim=-1, eps=1e-6) * query.shape[-1]**-.5
    k = F.normalize(key.float(), dim=-1, eps=1e-6)
    gate = lower_bound * torch.sigmoid(a_log.float().reshape(1, -1, 1).exp()
                                      * (raw_gate.float() + dt_bias.float().reshape(1, *q.shape[1:])))
    state = initial.float().clone()
    outputs, states = [], []
    for pos in range(query.shape[0]):
        state = state * gate[pos].exp().unsqueeze(-2)
        prediction = torch.einsum("hvk,hk->hv", state, k[pos])
        delta = (value[pos].float() - prediction) * raw_beta[pos].float().sigmoid().unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k[pos].unsqueeze(-2)
        outputs.append(torch.einsum("hvk,hk->hv", state, q[pos]))
        states.append(state.clone())
    return torch.stack(outputs).to(value.dtype), torch.stack(states)


def kda_native(query, key, value, raw_gate, raw_beta, initial, a_log, dt_bias,
               *, lower_bound=-5.):
    """调用目标模型同一层 recurrent KDA 接口，仅使用自有临时状态槽。"""
    from vllm_ascend.ops.kda import run_recurrent_kda

    length, heads, dim = query.shape
    if not 1 <= length <= 8 or dim != 128:
        raise ValueError("current native KDA supports 1..8 rows and 128-dim heads")
    states = initial.new_empty((length, heads, value.shape[-1], dim))
    states[0].copy_(initial)
    output = run_recurrent_kda(
        query[None], key[None], value[None], raw_gate[None], raw_beta[None], states,
        torch.tensor([0, length], dtype=torch.int32, device=query.device),
        torch.arange(length, dtype=torch.int32, device=query.device)[None],
        a_log.reshape(-1).contiguous(), dt_bias.contiguous(),
        beta_is_preprocessed=False, lower_bound=lower_bound,
    )
    return output[0], states
```

</details>

### 2.4 开发范围汇总

| 计算入口 | PyPTO 已有基础 | Orthrus 需要适配的部分 |
| --- | --- | --- |
| Draft KPool | 多行池打分、Top-K、索引展开，读写分离。 | 接 draft 查询权重和统一历史边界，跳过写池；处理短块 padding、INT8 池与现有历史布局的互通。检索核心不重写。 |
| 联合 MLA | 多 query 分页 latent 读取、稀疏 Attention、分块 softmax 统计。 | 加入整个临时块及两套 KV 展开；保留统计或在内部联合归一化；共同 value 输出后使用 draft O 投影。是主要计算扩展。 |
| Draft KDA | 带非零初态的块卷积、KDA prefill 和输出处理。 | 在私有状态上执行 K 行；可选策略再编排零状态反向块和门控融合。因果递推核心不重写。 |

表格划分的是三个计算入口，最终可以拆分或融合成不同数量的 kernel。共享 mHC、norm、MLP/MoE 等不产生新的 Attention 数学，但仍需模型执行和权重映射，不能将“可复用”理解为整网无需适配。

## 3. 模型执行与推理引擎接入

### 3.1 从计算模块到一轮投机

完整 draft 要依次通过 45 层，使用各层独立的 draft 参数，并复用主模型 embedding、外层 norm、mHC、MLP/MoE 和 LM head。共享的是权重，不是本轮激活，这些模块仍须对 anchor/MASK 输入重新计算，但不应复制第二套 backbone。

```text
主模型已接受的历史
    |
    +-> 复制 KPool、KDA 卷积尾部和递推状态
    +-> 固定 MLA 页表与只读历史引用
                    |
          anchor + MASK 块 -> 45 层 draft -> 候选及概率
                                               |
                                       原生 AR 整块验证
                                               |
                           接受前缀：保留 AR 算出的正式状态
                           拒绝后缀：回收 KV，恢复对应 KDA 状态
```

当前 vLLM 接入在同步 draft 期间固定历史边界，主模型不推进或回收被读取的页。MLA 历史引用在 proposal 返回、异常或下一次 metadata 捕获时失效，不能作为跨步骤、异步 replay 的快照。算子内部可以写自己的输出和临时状态，但不能改变这些 AR 历史。

接受 token 后保留的是 **target 按因果规则重新计算的状态**，draft latent、卷积尾部和递推矩阵永不转正。引擎决定接受前缀、管理页表与分配；算子按这些地址和有效长度读写。若验证器也迁入 PyPTO，还需逐位置 KDA/卷积 checkpoint 或拒绝后重放，只返回块末状态不够。现有 `prefill_kda` 和普通 MTP 编排不能自动替代该回滚接口。

### 3.2 整体 demo 与真实模型入口

独立 demo 位于 `online-speculation/benchmarks/glm53_orthrus_demo.py`。它包含实际投影和输出：MLA 从 4096 维输入生成低秩 query/latent、执行 32-head indexer、联合 Attention 和 O 投影；KDA 包含 QKV、更新门和两类 gate 投影、卷积、递推、可选反向块、gated norm 和 O 投影。二者直接调用上文核心函数，而非另一套示意计算。

demo 对应 TP8 单分片：MLA/KDA 各 8 个 head，indexer 保持 32 个 head，latent 为 512 维，query rank 为 1536。权重和历史是合成数据，两条分支分别返回结果，不是作为同一层相加；它不包含整网 mHC/MoE、分词或采样。

```bash
cd online-speculation

# 无需模型权重的 Torch 参考。
PYTHONPATH=src python benchmarks/glm53_orthrus_demo.py --backend torch

# 固定 Ascend/vLLM-Ascend 环境中的现成算子组合。
python scripts/run_in_runtime.py benchmarks/glm53_orthrus_demo.py \
  --backend npu --block-size 8 --prefix-length 2055 --kda-mode bidirectional
```

<details>
<summary>完整源码：两类 Attention 分支 demo</summary>

```python
#!/usr/bin/env python3
"""GLM-5.3-Flash 的两类 draft Attention，使用真实维度和现成 Torch/NPU 算子。

这是独立的计算 demo，不包含共享 mHC/MoE、分词、采样和服务调度。
权重及历史为合成数据；推理接入使用相同的 MLA/KDA 核心函数。
"""

import argparse
import json
import math

import torch
from torch import nn

from blockspec.vllm_ascend.glm53_ops import (
    kda_convolve, kda_native, kda_reference, select_history, sparse_draft_attention,
)


class DraftAttentionDemo(nn.Module):
    """一个 TP8 分片的几何：8 个 MLA/KDA head，indexer 仍有完整的 32 个 head。"""

    def __init__(self):
        super().__init__()
        self.heads, self.dim, self.rank = 8, 128, 512
        self.qkv_a = nn.Linear(4096, 1536 + 512, bias=False)
        self.q_norm, self.kv_norm = nn.RMSNorm(1536, eps=1e-5), nn.RMSNorm(512, eps=1e-5)
        self.q_b = nn.Linear(1536, 8 * 256, bias=False)
        self.index_q = nn.Linear(1536, 32 * 128, bias=False)
        self.index_weight = nn.Linear(4096, 32, bias=False)
        self.mla_o = nn.Linear(8 * 256, 4096, bias=False)
        for name in ("ar_key", "ar_value", "draft_key", "draft_value"):
            # 两套展开权重故意不同，不能把历史当成 draft latent 来解释。
            self.register_buffer(name, torch.randn(8, 256, 512) / math.sqrt(512))
        self.qkvbfg = nn.Linear(4096, 3 * 8 * 128 + 8 + 2 * 128, bias=False)
        self.f_b, self.g_b = nn.Linear(128, 8 * 128, bias=False), nn.Linear(128, 8 * 128, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(3 * 8 * 128, 1, 4) * .1)
        self.register_buffer("a_log", torch.zeros(8))
        self.register_buffer("dt_bias", torch.full((8 * 128,), -3.))
        self.kda_norm = nn.RMSNorm(128, eps=1e-5)
        self.kda_o = nn.Linear(8 * 128, 4096, bias=False)
        self.backward_gate = nn.Parameter(torch.full((8,), math.atanh(.25)))

    def mla(self, hidden, history, pools, *, backend):
        """完整 MLA 分支：投影 -> 历史检索 -> 历史/双向块联合 Attention -> O 投影。"""
        q_resid, latent = self.qkv_a(hidden).split((1536, 512), -1)
        q_resid = self.q_norm(q_resid)
        query = self.q_b(q_resid).reshape(-1, 8, 256)
        block = self.kv_norm(latent)
        index_q = self.index_q(q_resid).reshape(-1, 32, 128)
        weights = self.index_weight(hidden) * (128**-.5 * 32**-.5)
        # ORTHRUS：所有行止于同一个历史边界，临时块不参加历史 Top-K。
        indices = select_history(index_q, weights, pools, len(history), topk=2048)
        output = sparse_draft_attention(
            query, history, block, indices, ar_key=self.ar_key, ar_value=self.ar_value,
            draft_key=self.draft_key, draft_value=self.draft_value, backend=backend,
        )
        return self.mla_o(output.flatten(1))

    def _convolve(self, rows, tail, backend):
        # 宽度 4 的逐通道卷积，读取 3 行历史；不把新尾部写回 target。
        mixed = kda_convolve(rows, tail, self.conv_weight[:, 0],
                             backend="auto" if backend == "npu" else "torch")
        return mixed.reshape(-1, 3, 8, 128).unbind(1)

    def kda(self, hidden, tail, state, *, backend, bidirectional):
        """KDA 分支：临时历史递推，可选零状态反向块，在 gated norm 前合并。"""
        rows, beta, f_a, g_a = self.qkvbfg(hidden).split((3 * 8 * 128, 8, 128, 128), -1)
        gate = self.f_b(f_a).reshape(-1, 8, 128)
        core = kda_native if backend == "npu" else kda_reference

        def run(qkv, left, initial, raw_gate, raw_beta):
            q, k, v = self._convolve(qkv, left, backend)
            output, _ = core(q, k, v,
                             raw_gate.contiguous(), raw_beta.contiguous(), initial,
                             self.a_log, self.dt_bias)
            return output

        # ORTHRUS：从已接受位置的 AR 状态出发，但只更新 core 内部的临时副本。
        output = run(rows, tail, state, gate, beta)
        if bidirectional:
            reverse = run(rows.flip(0), torch.zeros_like(tail), torch.zeros_like(state),
                          gate.flip(0), beta.flip(0))
            output = output + self.backward_gate.tanh()[None, :, None] * reverse.flip(0)
        # GLM 的输出门是 sigmoid；不是将两路分别归一化后相加。
        out_gate = self.g_b(g_a).reshape(-1, 8, 128).sigmoid()
        return self.kda_o((self.kda_norm(output) * out_gate).flatten(1))

    def forward(self, hidden, history, pools, tail, state, *, backend="torch", bidirectional=False):
        return dict(mla=self.mla(hidden, history, pools, backend=backend),
                    kda=self.kda(hidden, tail, state, backend=backend, bidirectional=bidirectional))


def make_case(block_size=8, prefix_length=2051, *, device="cpu", dtype=torch.float32):
    """合成权重和只读历史；2051 覆盖 512 个完整池及 3 个尾部 token。"""
    if block_size < 1 or prefix_length < 0:
        raise ValueError("positive block size and nonnegative prefix length required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5308)
        model = DraftAttentionDemo().eval().requires_grad_(False).to(device=device, dtype=dtype)
        hidden = torch.randn(block_size, 4096).to(device=device, dtype=dtype)
        history = torch.randn(prefix_length, 512).to(device=device, dtype=dtype) * .1
        pools = torch.randn(prefix_length // 4, 128).to(device=device, dtype=dtype) * .1
        tail = torch.randn(3, 3 * 8 * 128).to(device=device, dtype=dtype) * .1
        state = torch.randn(8, 128, 128).to(device=device, dtype=torch.float32) * .01
    # state 是 FP32，a_log/dt_bias 与原生 KDA 的接口一致，保持 FP32。
    model.a_log = model.a_log.float()
    model.dt_bias = model.dt_bias.float()
    return model, (hidden, history, pools, tail, state)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "npu"), default="torch")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=2051)
    parser.add_argument("--kda-mode", choices=("causal", "bidirectional"), default="causal")
    args = parser.parse_args()
    if args.backend == "npu":
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import enable_custom_op

        if not 1 <= args.block_size <= 8:
            parser.error("the current native KDA entry supports block sizes 1..8")
        torch.npu.set_device(0)
        if not enable_custom_op():
            raise RuntimeError("native Ascend operators are unavailable")
    model, inputs = make_case(args.block_size, args.prefix_length,
                              device="npu" if args.backend == "npu" else "cpu",
                              dtype=torch.bfloat16 if args.backend == "npu" else torch.float32)
    outputs = model(*inputs, backend=args.backend, bidirectional=args.kda_mode == "bidirectional")
    print(json.dumps(dict(backend=args.backend, block_size=args.block_size, kda_mode=args.kda_mode,
                          prefix_length=args.prefix_length,
                          outputs={name: dict(shape=list(value.shape), mean=float(value.float().mean()))
                                   for name, value in outputs.items()},
                          scope="synthetic attention branches; not full-model serving acceptance"), indent=2))


if __name__ == "__main__":
    main()
```

</details>

真实模型推理由 `glm53_native.py / glm53_proposer.py` 接入原生 vLLM-Ascend。`scripts/run_glm53_vllm.py` 是有界验证程序，不是长驻 OpenAI 服务；其主模型及 verifier 使用 GLM 固定的 0.30 环境，不能沿用 DSV4 的 0.26 镜像预检结论。

将完整 draft 交给 PyPTO 时，除上面的三个入口外，还要加载共享权重与 draft head，接齐 mHC、残差、MLP/MoE、TP/EP 通信和输出。若仅替换计算模块，则可暂时保留现有 Torch 执行器。两种方式都需约定张量布局、stream 同步、页表和工作区生命周期。

head 契约需记录基模、K、MASK token、KDA 策略和参数布局；量化 head 使用自己的 scale，不能在参数更新后直接套用主模型 scale。这属于参数交付与加载，不是新增 Attention 算子。

### 3.3 已验证范围与后续验收

现成算子组合已做 CPU 数学与单卡 NPU 对照，覆盖不同 AR/draft KV 展开、空历史、超过 512 个池及尾部、分页与 padding、非零 KDA 初态、非零反向门和目标状态只读。单独 MLA 还覆盖 K32，但完整原生 KDA 接入仍限 K≤8。PyPTO 能力描述来自固定版本源码核对，尚未编译运行 PyPTO Orthrus kernel。

真实源权重的五层截层模型已有单/双卡 eager 验证。更新后的五层 TP1/K8 还完成 target-only、两种 draft、独立进程 head 重载及两种采样闭环；四个 draft/reload greedy 阶段各 96 个 token 与基线一致，重载哈希一致，强制分页与自动读取合计 444 次缓存审计通过。采样仅作功能验证，不代替分布检验。

九层四卡已有功能闭环，但 greedy 未验收；完整 45 层八卡尚未验收。截层、合成 demo 和初始化 head 都不能证明完整模型正确性、业务接受率或加速收益。长历史性能记录属于局部算子测试，不等同于整网长上下文验收。

建议先完成只读 KPool 查询、联合 MLA 和因果 KDA 私有状态入口，与 Torch-NPU 逐模块对照；反向 KDA 作为独立策略接入。随后验证整层、整网、TP/EP 和各接受位置恢复。必须核对同一 anchor 边界、临时块全可见、两套展开权重、空历史、跨页与池尾部，以及 proposal 前后 target 状态不变，再据耗时决定进一步融合。

## 参考资料

- 模型结构与维度：[GLM5.3-Flash 官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)。
- 固定实现版本：[pypto-lib](https://github.com/hw-native-sys/pypto-lib/tree/2aef4534f8769d8ce8d49d9dde7e5aeebc472197)、[pypto-serving](https://github.com/hw-native-sys/pypto-serving/tree/5f548be23d7af120b20a40694601f5bb5b8b7a90)。
- 本仓库记录：[运行与配置](online-speculation/docs/ORTHRUS_GLM53_FLASH.md)、[KDA 策略定义](online-speculation/docs/GLM53_KDA_DIRECTION.md)、[推理算子复核](online-speculation/docs/GLM53_INFERENCE_OPS_AUDIT.md)、[读取与性能分析](online-speculation/docs/GLM53_INFERENCE_OPTIMIZATION.md)。
