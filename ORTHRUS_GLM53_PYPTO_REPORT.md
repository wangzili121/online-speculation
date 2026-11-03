# Orthrus-GLM5.3-Flash NPU 推理适配与算子设计

## 1. 背景与适配目标

Orthrus 的做法是先让 draft 分支一次预测多个 token，再交给主模型验证。draft 复用主模型的大部分权重，只在负责处理 token 之间关系的模块上使用独立参数。这样接入 GLM5.3-Flash 时，首先要弄清楚：模型的哪些层要读取历史，哪些层要接着历史状态继续计算。

GLM5.3-Flash 同时使用稀疏 MLA 和 KDA，两类层保存历史的方式不同，所需修改也不同。下面先从这一区别讲起，再分别说明 PyPTO 已有什么、还要补什么，以及现有 Torch-NPU 代码怎样完成这些计算。本文只讨论文本推理，训练、在线更新和多模态不在范围内。

### 1.1 草稿分支与主模型的关系

GLM5.3-Flash 共 45 层：每三个 KDA 层后接一个稀疏 MLA 层，这样重复 11 次，最后再接一个 KDA 层。因此共有 34 层 KDA、11 层 MLA；MLA 位于从 0 开始计数的第 3、7、…、43 层。

MLA 保留可以逐条读取的历史表示，计算时先检索，再做 Attention。KDA 则把历史信息保存在卷积尾部和递推矩阵中，计算新位置时继续更新这个状态。因此，这里的适配应围绕 MLA 和 KDA 展开，而不是沿用 DSV4 的 SWA、C4、C128 分类。

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

draft 每轮输入一个块：`anchor + (K-1) 个 MASK`。anchor 是位置 $a$ 上已经确定的最后一个 token，MASK 占据接下来要预测的位置。历史只读取 $[0,a)$，而 anchor 和 MASK 一起参与本轮计算。因此，块长 K 包含 anchor，实际提出 K-1 个候选。当前原生 KDA 接口将整条推理路径限制在 $K\le8$；这是本实现的接口限制，不是 Orthrus 或 PyPTO 的数学限制。

MLA 可以让块内位置互相可见，KDA 却不是 softmax Attention，没有一张 mask 可以直接把它改成双向。为此，当前实现保留两种实验策略：**因果 KDA + 双向 MLA**，以及**门控双向 KDA + 双向 MLA**。前者从主模型历史状态的副本出发向前递推；后者再增加一次从零状态出发的块内反向递推，并在输出归一化前合并两路结果。下文会先讲两种策略共用的部分，再解释反向分支。

这两种策略是本项目的 GLM 适配选项，尚无原作者提供并确认的训练 head。交付 head 时需要注明采用哪一种策略，推理时保持一致。

### 1.2 PyPTO 已有基础与本次改动范围

PyPTO 库已经有 GLM5.3-Flash 的模型计算代码，包括 KDA、KPool、稀疏 MLA、mHC、MoE、量化投影和整网前向。本文以 `pypto-lib@2aef4534f8769d8ce8d49d9dde7e5aeebc472197` 为准，下文库内文件均位于 `models/glm5_3_flash/`。

在这些代码上支持 Orthrus，需要接好三个计算入口：先由 KPool 检索历史，再由 MLA 把选中历史和当前双向块一起计算；KDA 层则从历史初态继续处理整个块。**主要的计算扩展在联合 MLA，KPool 检索和因果 KDA 的核心算法可以复用。** 三个入口只是需求的划分，最后可以由多个 kernel 组合，也可以进一步融合。

模型计算之外还需要服务接入。本文核对的 `pypto-serving@5f548be23d7af120b20a40694601f5bb5b8b7a90` 公开主线，没有库内文档提到的 `pypto_serving.model.glm53.npu_runner.Glm53ModelRunner`。因此，已有模型代码还不足以说明该版本已经能直接启动 GLM 服务；第 3 节会另行说明引擎需要配合的部分。

## 2. Attention 模块设计

### 2.1 KPool：用 draft query 检索主模型历史

#### 2.1.1 检索过程与历史范围

MLA 不必读取所有历史 token，而是先用 KPool 找出一批相关位置。主模型每四个历史位置生成一个池 key，检索时先给池打分，再把选中的池展开为 token 索引。这些 key 是主模型已经计算好的压缩结果，draft 直接读取，不重新压缩历史。

原模型逐步生成时，每个位置只能检索截至自身边界的历史。设 AR 位置 $p$ 的可见长度为 $b_p$，完整池数为 $P_p=\lfloor b_p/4\rfloor$。$u_{ph}^{AR}$ 是 indexer 第 h 个 head 的 query，$\omega_{ph}^{AR}$ 是包含缩放的 head 权重，$\kappa_r^{AR}$ 是历史池 key，打分为：

$$
s_{pr}^{AR}=\sum_{h=1}^{32}\omega_{ph}^{AR}\,\mathrm{ReLU}\left((u_{ph}^{AR})^\top\kappa_r^{AR}\right),\qquad 0\le r<P_p.
$$

draft 的不同之处在于：一个块里的 query 同时提出，读取的都应是 anchor 之前的历史。因此换用 draft 的 query 和 head 权重后，所有行都使用同一个池数 $P=\lfloor a/4\rfloor$：

$$
s_{ir}^{D}=\sum_{h=1}^{32}\omega_{ih}^{D}\,\mathrm{ReLU}\left((u_{ih}^{D})^\top\kappa_r^{AR}\right),\qquad 0\le r<P.
$$

$u$ 和 $\kappa$ 均为 128 维，$\omega$ 已包含 $128^{-1/2}32^{-1/2}$ 缩放。令 $m=\min(P,512)$，选中的池集合及展开后的历史 token 集合为：

$$
\mathcal R_i=\mathrm{TopK}_{m}(s_i^{D}),\qquad \mathcal H_i=\bigcup_{r\in\mathcal R_i}\{4r,4r+1,4r+2,4r+3\}\cup\{4P,\ldots,a-1\}.
$$

最后一项补上不足四个 token 的历史尾部；没有尾部时该集合为空。这样每行最多选出 2048 个池展开位置和 3 个尾部位置。虽然不同 query 可以选中不同的池，但它们的历史边界相同。当前块不参加这次历史检索，而是在下一步 MLA 中全部可见。

#### 2.1.2 PyPTO 已有实现与所需修改

现有 `indexer.py` 已经把读池和写池分开，`indexer_score / indexer_topk / indexer_expand` 也支持多行 query。因此可以保留这三个计算步骤，只调整输入：query 来自 draft 权重，每一行的历史范围都按同一个 anchor 填写。具体对应 `seg_start / pool_count`，以及展开阶段的 `tail_start / tail_count / kv_len`。

为此需要增加一个 draft 查询入口，使用 draft 的 `w_q_b` 和 head-weight 投影，随后执行上述检索，但不调用 `indexer_cache_write / indexer_pool_write`。现有 `indexer_proj` 会同时计算 query、key 和 gate，首版可以直接复用并忽略不用的输出；后续再拆出只算 query 的投影，省去多余计算。已有 `skip_query` 表示跳过 query，不能用于这个目的。

算法可以沿用，输入格式还有两处需要处理。短块首先要补齐到投影要求的 16 行，但补出的行不能作为有效 query，score 的分块尺寸也要按实际 K 检查。其次，PyPTO 读取 Hadamard 变换后的 INT8 池 key 和 FP32 scale，并会量化 query；当前 vLLM 路径读取的是未打包的 BF16 池。接入时要转换格式，或提供能读取现有格式的入口，再检查缩放是否重复、量化后的 Top-K 结果是否符合约定。

#### 2.1.3 Torch-NPU 实现

Torch 版本按同样的顺序实现：算每个 query/head 与池 key 的分数，做 ReLU 和 head 加权，选池，最后展开索引并补上尾部。`select_history` 接收 `[K,32,128]` 的 query、`[K,32]` 的 head 权重和 `[P,128]` 的历史池：

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

这里保留了一个会影响结果的顺序：先对每个 head 的分数做 ReLU，再加权求和，与所参考 PyPTO 定义一致。固定 vLLM-Ascend 镜像中另一个 Triton 入口会先合并 query head 再打分，计算结果并不等价，因此没有用它替换 draft 检索。主模型继续走原生路径，draft head 则按本节公式使用。

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

#### 2.2.1 为什么历史和当前块要一起计算

上一步给出了每行要读取的历史位置，MLA 接下来计算这些位置和当前块对输出的贡献。历史中保存的是 512 维压缩表示，以下称为 latent；它还需要通过各 head 的权重展开成 K/V。主模型生成的历史使用 AR 展开权重，本轮 draft 生成的临时块使用 draft 展开权重，两套权重可能不同。

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

公式的分子同时累加历史和临时块的贡献，分母也把两部分放在同一个 softmax 中。所有临时位置 $t$ 都可见，历史仍止于 anchor 前。这里没有 DSV4 的 Attention sink，也不使用 RoPE。

为调用现成算子，当前实现先分别计算历史和临时块。两段各自做过 softmax，输出已经除以不同的分母，所以不能直接相加；需要保留每段的 log-sum-exp（LSE），用它恢复两段的相对权重。将局部输出记为 $o_{i,H},o_{i,B}$，自然对数 LSE 记为 $L_{i,H},L_{i,B}$，合并为：

$$
L_i=\log\left(e^{L_{i,H}}+e^{L_{i,B}}\right),\qquad o_i^D=e^{L_{i,H}-L_i}o_{i,H}+e^{L_{i,B}-L_i}o_{i,B}.
$$

合并前，两段还要分别乘各自的 $W_V$，都展开成 256 维 value 向量。若先混合 512 维 latent，再统一乘 draft 的 $W_V$，就把历史也按 draft 权重解释了，与上式不同。空历史则取零输出、负无穷 LSE，使其对结果没有贡献。PyPTO 可以在一次计算中完成这套归一化，不必沿用两次算子调用。

#### 2.2.2 PyPTO 已有实现与所需修改

现有 `attention.py::decode_sparse_attn` 已经能处理多行 query，并根据每行索引读取分页 latent。这个核心只读 cache，写入在独立的 `mla_cache_write` 中完成，因此分页读取本身可以保留；历史边界继续由传入索引限制。

还缺的是当前块及两套展开权重。原入口只接一类 latent，并直接返回归一化后的 latent 输出，已经丢掉了后续合并需要的 softmax 统计。因此要么扩展它，让历史和临时块在内部一起计算；要么让它返回统计，再与块内结果合并。下面给出需要传入的数据，函数名仅作示意：

```text
draft_sparse_mla(
  q_history_absorbed[K,H,512], q_block_absorbed[K,H,512],
  ar_latent_pages, block_table, request_ids, history_indices[K,S],
  block_latent[K,512], W_V_AR[H,256,512], W_V_D[H,256,512]
) -> attention[K,H,256]
```

现有分块 softmax 其实已经计算了这些统计，只是没有向外返回：$m_r$ 是本段最大分数，$\ell_r$ 是减去最大值后的指数和，$u_r$ 是按这些指数加权累加的 latent 向量。将历史和临时块分别记为 $H,B$，就可以继续计算：

$$
m=\max(m_H,m_B),\qquad \alpha_H=e^{m_H-m},\qquad \alpha_B=e^{m_B-m}.
$$

$$
\ell=\alpha_H\ell_H+\alpha_B\ell_B,\qquad u=\alpha_H W_V^{AR}u_H+\alpha_B W_V^Du_B,\qquad o=u/\ell.
$$

$u_H,u_B$ 仍是 512 维 latent 累加值，分别乘 $W_V$ 后，才得到 256 维的合并分子 $u$。这与 LSE 合并公式计算的是同一个结果，因此可以复用已有分块归约，不必另写一个独立的 LSE kernel。实现中将空段权重置零，避免负无穷之间相减。

完成 Attention 后还要接 O 投影。原有 `mla_epilog_decode` 为 AR 路径预先合并了 value 展开和输出投影，保存在 `w_o_absorbed` 中；两段使用不同 $W_V$ 时，这一套权重不再适用。按上式先得到 256 维 value 输出后，可以接回 `mla_epilog_prefill / mla_epilog_w8a8`，使用 draft 的实际 O 投影。

#### 2.2.3 Torch-NPU 实现

Torch-NPU 版本利用矩阵乘法的结合律，把 K 展开权重先乘到 query 上，而不是给整段历史展开逐 head K。这样可以直接用 512 维 latent 调用 `npu_fused_infer_attention_score`（FIA）。query 虽变为 512 维，分数缩放仍按原始 256 维计算。历史因 query 而异，按行收集；临时块则由所有 query 共读，只传一份 K/V。

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

代码中的 `q_ar` 仍由 draft query 得来，名字表示它乘了 AR 的 K 展开权重，并没有再运行一次主模型。两次 FIA 返回局部输出和 LSE，再由 `npu_attention_update` 合并；空历史单独处理。完整函数如下，同时保留了纯 Torch 对照实现。

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

历史数据由 `gather_history` 提供。短历史下，它从独立连续快照中选行；历史足够长时，也可以通过 `PagedLatent` 借用本轮页表，只收集命中的行。这两种方式读取的逻辑位置相同，只是搬运量和索引开销不同。PyPTO 已有分页读取能力，可以直接访问选中页，省去 Torch 中间收集出的张量。

### 2.3 KDA：带历史初值的临时块执行

#### 2.3.1 从历史状态继续计算当前块

MLA 根据索引读取一条条历史，KDA 则只带着一份状态继续往前算。这份状态包括宽度 4 卷积保留的三行输入，以及每个 head 的 $128\times128$ 矩阵。draft 先复制 anchor 之前的这两份状态，再用自己的投影处理当前块，计算过程中只改副本。

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

这套递推公式在 AR 和因果 draft 中不变。区别是 AR 使用主模型参数，并把结果留作后续历史；draft 使用独立参数，以 $S_{-1}^{D}=\mathrm{copy}(S_{a-1}^{AR})$ 和历史卷积尾部副本为起点，连续处理 K 行，结果只供本轮起草使用。

如果采用门控双向策略，还要让当前位置获得块内右侧的信息。做法是把**当前块的原始投影输入倒序**，从零卷积历史和零矩阵状态出发，再运行同一套递推。将这一路输出翻回原顺序，记为 $y_i^{\leftarrow}$，再用每 head 的参数 $\rho$ 控制它的贡献：

$$
\widetilde y_i=y_i^{\rightarrow}+\tanh(\rho)\odot y_i^{\leftarrow}.
$$

因果策略直接取 $\widetilde y_i=y_i^{\rightarrow}$。两种策略随后使用相同输出处理：

$$
o_i=W_O^D\,\mathrm{ConcatHeads}\left(\mathrm{RMSNorm}_{w^D}(\widetilde y_i)\odot\sigma(g_i)\right).
$$

$g_i$ 是输出门投影，$w^D$ 是该 norm 的权重。这里先合并两路，再做 norm、sigmoid 门控和 O 投影；GLM 的输出门与 Qwen 的 SiLU 不同。反向分支只遍历当前块，不重算历史。$\rho=0$ 时它不影响输出，所以验证双向分支时需要使用非零门。

#### 2.3.2 PyPTO 已有实现与所需修改

这部分可以从现有 prefill 入口接起。`kda.py::kda_conv_prefill` 已通过 `query_start_loc / state_rows` 支持带历史尾部的块，`prefill_kda` 也支持非零矩阵初态、变长序列和 16 行分块。给它们传入历史状态的私有副本，就能沿用因果递推，再接现有 `kda_output`。

之所以选 prefill，是因为同一个请求的 K 个位置有先后依赖。`decode_kda` 把各行当作独立任务，若让它们同时写同一个 state row，就会发生竞争，而不是连续递推。也可以逐步调用 decode，但必须保证顺序，并核对状态布局、门函数、归一化 epsilon 和有效长度。

在此基础上，双向策略再加倒序输入、零状态、第二次卷积/递推，以及输出前的门控合并。倒序必须发生在卷积之前；先做正向卷积再翻转，得到的邻域不同。首版可用现有算子组合完成，再根据耗时考虑融合。改变 K 时主要调整分块、padding 和编译形状，递推公式不变。

#### 2.3.3 Torch-NPU 实现

现有 NPU 版本调用与主模型相同的 `run_recurrent_kda`，但为 draft 另分配状态槽，再复制历史初态。前面的卷积由 `kda_convolve` 选择原生因果卷积或 `conv1d + SiLU`，两条路径也都只改临时状态。两种策略的差别集中在下面这段：

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

代码中的 8 heads 对应 demo 的 TP8 单分片。卷积输出的 Q/K/V 切片可以直接交给当前原生递推接口，省去连续化复制；其他参数仍按接口要求处理。卷积则按已测形状选择实现，因为 head 数较少时，分配和调用开销可能使融合版本更慢。

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

上述三个入口接通后，解决的是每类层内部怎样处理历史和当前块。下一步还要把它们放回 45 层模型中，接好共享权重、层间数据传递和主模型验证。这些工作不需要新的 Attention 公式，但决定了整条推理能否运行。

## 3. 模型执行与推理引擎接入

### 3.1 从计算模块到一轮投机

起草时，anchor/MASK 块先经过共享 Embedding，再依次通过 45 层。每层用本层 draft 参数执行 MLA 或 KDA，其他部分继续使用主模型的 norm、mHC 和 MLP/MoE，最后由共享 LM head 给出候选分布。这里共享的是权重：draft 输入不同，仍需重新计算各层激活，但无需保存第二套 backbone 参数。

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

先看历史怎样交给 draft。当前 vLLM 接入采用同步起草：这段时间主模型不继续执行，也不回收正在读取的页，因此 MLA 可以暂时借用历史引用。一次 proposal 返回、异常退出或获取下一轮 metadata 后，该引用就失效；若要异步执行或留作后续 replay，需要另行保存快照。KDA 和 KPool 则按图中方式复制所需状态，避免临时计算覆盖主模型历史。

再看候选怎样成为正式输出。主模型收到候选后，仍按因果规则计算和验证；接受前缀对应的主模型状态才会保留下来，draft 的 latent、卷积尾部和递推矩阵不会转成正式历史。接受几个 token、保留哪些页由引擎决定，算子按引擎提供的地址和长度执行。

如果连验证器也迁入 PyPTO，还需要恢复到任意接受位置的 KDA 状态。只返回块末矩阵不够，因为拒绝可能发生在块中间；需要逐位置 checkpoint，或者重新计算接受前缀。现有 `prefill_kda` 和普通 MTP 流程尚不包含这套整块拒绝恢复接口。

### 3.2 整体 demo 与真实模型入口

为了单独运行这些计算，仓库提供了 `online-speculation/benchmarks/glm53_orthrus_demo.py`。它把上面的核心函数接上输入和输出投影：MLA 从 4096 维输入生成 query/latent，经过 32-head indexer 和联合 Attention，再投影回 4096 维；KDA 则依次执行投影、卷积、递推、可选反向块、gated norm 和 O 投影。

demo 使用 TP8 单分片的尺寸：MLA/KDA 各 8 个 head，indexer 保留 32 个 head，latent 为 512 维，query rank 为 1536。它用合成权重和历史分别演示两类层，不把两类层的输出相加，也不执行整网 mHC/MoE、分词或采样。因此可以先用它对照计算，再进入真实模型测试。

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

真实模型由 `glm53_native.py / glm53_proposer.py` 接入原生 vLLM-Ascend，运行入口是 `scripts/run_glm53_vllm.py`。这个脚本加载模型、执行指定验证用例后退出，不是长驻 OpenAI 服务。它使用固定的 GLM 0.30 环境，卷积和 KDA API 与 DSV4 的 0.26 环境不同，需要分别检查兼容性。

PyPTO 接入可以分两步进行。先替换这三个计算入口，其余共享层和通信继续由现有 Torch 执行器负责；之后再把整段 draft 迁入 PyPTO，接齐共享权重、mHC、残差、MLP/MoE、TP/EP 通信和输出。无论停在哪一步，调用双方都要约定好张量布局、stream 同步、页表，以及临时工作区何时可以释放。

加载 head 时，还要核对基模、K、MASK token、KDA 策略和参数布局。量化 head 的 scale 应与它自己的参数匹配，不能直接套用主模型 scale。这些检查放在参数加载阶段完成，不需要另开发 Attention 算子。

### 3.3 已验证范围与后续验收

目前先验证了计算本身，再验证真实权重的截层模型。CPU 和单卡 NPU 对照覆盖两套不同的 KV 展开权重、空历史、超过 512 个池及尾部、分页与 padding、非零 KDA 初态和反向门，并检查起草没有修改目标状态。单独 MLA 还测过 K32，但整条原生 KDA 接入仍限 K≤8。

在真实源权重上，五层截层模型已有单/双卡 eager 验证。其中五层 TP1/K8 测过无投机基线、两种 draft、独立进程重载 head，以及两种策略下的采样流程。四个 draft/reload greedy 阶段各 96 个 token 与基线一致，重载前后 head 哈希一致；分页和自动读取两种方式合计通过 444 次缓存检查。采样部分只确认流程能完成，尚不代表分布检验通过。

还未完成的部分也要单独看：九层四卡已能完成起草和验证，但 greedy 未验收；完整 45 层八卡尚未验收。长历史记录是局部算子测试，尚不覆盖整网长上下文。PyPTO 部分目前只核对了固定版本源码，还没有编译运行 Orthrus kernel，以上 Torch-NPU 结果是它的对照依据，不是 PyPTO 验证结果。

因此，接下来应先让三个 PyPTO 入口与 Torch-NPU 结果对齐，再接整层和整网，最后检查 TP/EP 及各接受位置的状态恢复。对齐时重点看历史是否止于同一 anchor、块内是否全部可见、两套展开权重是否用对，以及空历史、跨页、池尾部和状态只读是否正确。反向 KDA 作为独立策略测试；计算和流程确认后，再根据耗时决定融合。初始化 head 在这里用于验证链路，业务接受率和加速收益需要训练 head 另行评估。

## 参考资料

- 模型结构与维度：[GLM5.3-Flash 官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)。
- 固定实现版本：[pypto-lib](https://github.com/hw-native-sys/pypto-lib/tree/2aef4534f8769d8ce8d49d9dde7e5aeebc472197)、[pypto-serving](https://github.com/hw-native-sys/pypto-serving/tree/5f548be23d7af120b20a40694601f5bb5b8b7a90)。
- 本仓库记录：[运行与配置](online-speculation/docs/ORTHRUS_GLM53_FLASH.md)、[KDA 策略定义](online-speculation/docs/GLM53_KDA_DIRECTION.md)、[推理算子复核](online-speculation/docs/GLM53_INFERENCE_OPS_AUDIT.md)、[读取与性能分析](online-speculation/docs/GLM53_INFERENCE_OPTIMIZATION.md)。
