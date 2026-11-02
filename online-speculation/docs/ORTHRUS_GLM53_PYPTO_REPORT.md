# Orthrus-GLM-5.3-Flash：PyPTO 推理算子适配需求

## 1. 目标与现有基础

Orthrus 用一组独立的 draft Attention 参数，一次预测多个候选 token，再交给原模型验证。draft 共享原模型的 embedding、外层 norm、mHC、MLP/MoE 和 LM head，也读取原模型已经计算好的历史状态；但本轮输入和注意力规则不同，因此仍要计算自己的逐层激活，并非直接复用主模型本轮的计算结果。

本文面向 **GLM-5.3-Flash 的文本推理**，不涉及训练、在线更新和多模态。目标是：在已有 PyPTO GLM 计算基础上，明确 draft 分支还需要什么输入、计算与状态隔离。文中的现成算子组合和 demo 已实现于 Torch/Torch-NPU；**不是宣称 PyPTO 已实现 Orthrus**。

### 模型结构与 draft 的位置

GLM-5.3-Flash 的 45 层为 `(KDA × 3 → 稀疏 MLA) × 11 → KDA`，共 34 层 KDA、11 层 MLA。MLA 层编号为 3、7、…、43，从 0 开始计数。它不是 DSV4 的 SWA/C4/C128 结构。

```text
anchor + MASK × (K−1)                原模型已接受的历史
          │                                │
    共享 embedding                         │
          │                                │
    每层：共享 mHC / norm                  │
          │                                │
          ├─ KDA 层：draft 投影 ── 读取卷积尾部和递推状态的临时副本
          │                         → 块内递推 → draft 输出投影
          │
          └─ MLA 层：draft 投影 ── 用 draft indexer 查询历史池
                                    → 选中的 AR latent + 全部临时块 latent
                                    → 联合 Attention → draft 输出投影
          │
    共享 mHC / norm / dense MLP 或 MoE
          │
    共享最终 norm / LM head → 候选 → 原模型验证
```

这里 $K$ 是包含 anchor 的块长，产生 $K-1$ 个候选；anchor 是位置 $a$ 的已知 token，输入历史仅覆盖 $[0,a)$，MASK 是未知位置的占位 token。当前原生 KDA 适配采用 $K\le8$。**8 不是 Orthrus 或 PyPTO 的数学限制**，也不代表所有路径已验收 $K=32$。

普通 Qwen3 的 Orthrus 可以用双向 SDPA 表达；GLM 的 KDA 是带状态的线性递推，不能通过修改 SDPA mask 变成双向。当前接入保留两种显式实验策略：

- **因果 KDA + 双向 MLA**：KDA 从 AR 状态出发，在临时副本上向前递推。
- **门控双向 KDA + 双向 MLA**：额外从零状态反向遍历当前块，在输出归一化前融合。

两种策略需要各自匹配的 head；都不是原作者已经训练并确认的 GLM head。下文先给出共用需求，再单独说明反向块的增量。

### PyPTO 当前支持到哪里

本次核对固定到 2026-11-03 可见的公开主线：`pypto-lib@2aef453`、`pypto-serving@5f548be`，完整提交号见文末。`pypto-lib/models/glm5_3_flash/` 已有 KDA、KPool indexer、稀疏 MLA，以及 mHC、MoE、量化投影和前向编排代码，不能将其当成尚未支持 GLM。

但库内文档提到的 `pypto_serving.model.glm53.npu_runner.Glm53ModelRunner`，在本次核对的 **Serving 公开主线中没有对应模块**。这是两个公开版本之间的差异；可能存在其他分支，但不能据此认定当前主线已提供可直接使用的 GLM 服务。

**Attention 侧按三个计算入口组织需求，其中 MLA 是主要计算扩展；Indexer 和因果 KDA 主要复用已有核心。三个入口不等于三个全新 kernel。**

双向临时块包含在 MLA 入口内，不必再单列一个独立算子；KDA 则使用自身的状态递推，不与 MLA 共用一个双向 Attention 核心。

| 计算入口 | 当前 PyPTO 已有能力 | Orthrus 所需改动与交付 |
| --- | --- | --- |
| **Draft KPool 检索** | `indexer.py` 的 `indexer_score / indexer_topk / indexer_expand` 已支持多行 query、每行可见池数和尾部 token；读池与写池分开。 | 用 draft 的 query/head-weight 投影，对全部行传入同一 anchor 历史边界，不执行历史 key/池写入。首先复用检索核心；再抽出 query-only 投影、处理 padding 和 cache ABI。无需重写 Top-K。 |
| **历史与临时块联合 MLA** | `attention.py` 的 `decode_sparse_attn` 已支持多行 query、分页 latent 和每行索引，底层只读 cache；内部已有分块 softmax 合并，但只输出归一化 latent 结果。 | 增加完整临时块分支，并让历史、临时块分别使用 AR/draft 的 KV 展开参数。扩展接口保留 softmax 统计或直接完成联合归一化，最后进入同一 value 空间。不能直接把现有输出相加。 |
| **Draft KDA 块执行** | `kda.py` 已有 `kda_conv_prefill / prefill_kda`，支持带历史状态的变长序列；decode 则是每行独立状态。 | 将 AR 卷积尾部和递推状态复制到临时槽，用 prefill 入口执行整个块，再复用 `kda_output`。因果策略无需新递推算法；双向策略再复用同一核心执行零状态反向块，并增加门控融合编排。 |

## 2. Draft KPool 检索

### 算法与原有路径的区别

Indexer 不计算最终 Attention，而是先挑出值得读取的历史 token。模型将每 4 个历史位置压缩为一个 key 池；draft 使用自己的 query 和 head 权重给这些历史池打分，选池后展开成 token 索引。**池 key 来自主模型的既有压缩结果，不在 draft 中重新压缩，也不是简单求均值。**

对一个 draft 块，令 $P=\lfloor a/4\rfloor$ 为完整历史池数，$u_{ih}\in\mathbb R^{128}$ 为第 $i$ 行、第 $h$ 个 indexer head 的 query，$\kappa_r^{AR}$ 为历史池 key，$\omega_{ih}$ 为包含缩放的 head 权重。32 个 head 的分数为

$$
s_{ir}=\sum_{h=1}^{32}\omega_{ih}
\operatorname{ReLU}\!\left(u_{ih}^{\mathsf T}\kappa_r^{AR}\right),
\qquad 0\le r<P.
\tag{1}
$$

选择 $m=\min(P,512)$ 个池，并追加不足一池的历史尾部：

$$
\mathcal R_i=\operatorname{TopK}_{m}(s_i),\qquad
\mathcal H_i=
\bigcup_{r\in\mathcal R_i}\{4r,4r+1,4r+2,4r+3\}
\;\cup\;\{4P,\ldots,a-1\}.
\tag{2}
$$

因此每行最多读 2048 个池展开 token，加 0 至 3 个尾部 token。各行可能选择不同的池，但历史截止位置都是 $a$；临时块的 $K$ 个位置随后全部参与 MLA，不经过这个 Top-K。

| 原模型 AR | Orthrus draft |
| --- | --- |
| 第 $p$ 个 query 使用自身位置对应的历史边界，并按正常前向产生、写入新 key。 | 第 $i$ 个 query 使用 draft 投影，所有行的边界固定为 $[0,a)$；只查询已有池，不生成新的目标历史。 |

### 当前小算子组合

以下为 `glm53_ops.py::select_history` 的计算主体。`query=[K,32,128]`，`pooled_keys=[P,128]`，`head_weights=[K,32]` 已含 $128^{-1/2}32^{-1/2}$ 缩放。

```python
def select_history(query, head_weights, pooled_keys, prefix_length,
                   *, topk=2048, pool_size=4):
    complete = prefix_length // pool_size
    scores = torch.einsum("qhd,pd->qhp", query.float(), pooled_keys.float())
    scores = (scores.relu() * head_weights.float().unsqueeze(-1)).sum(1)
    chosen = scores.topk(min(complete, topk // pool_size), dim=-1).indices
    offsets = torch.arange(pool_size, device=query.device)
    indices = (chosen.unsqueeze(-1) * pool_size + offsets).flatten(1)
    tail = torch.arange(complete * pool_size, prefix_length, device=query.device)
    return torch.cat((indices, tail.expand(query.shape[0], -1)), dim=-1)
```

### PyPTO 具体适配

现有 `indexer_score` 通过 `seg_start / pool_count` 描述各行的历史，`indexer_expand` 接收 `tail_start / tail_count / kv_len`。对同一 draft 请求，这些边界按同一个 $a$ 填写即可；不需要为“多行 query”另写一套 score 或 Top-K。

需要新增的是 **draft 查询入口的编排**：使用 draft 的 `w_q_b` 和 head-weight 投影；不调用 `indexer_cache_write / indexer_pool_write`。现有 `indexer_proj` 同时计算 query、key 和 gate，可先复用并丢弃不用的结果，再抽成 query-only 投影减少计算。它的 `skip_query` 是“跳过 query”，不是所需的 query-only 开关。

另有两个必须落实的接口细节：

- 当前 `indexer_proj` 要求行数按 16 对齐；短块需 padding，并保证 padding 行不进入实际块或历史。score 的部分 tile 同样需要验证行数对齐，不能只看动态维度就认定任意 $K$ 已可直接运行。
- PyPTO 的历史池采用 Hadamard 变换后的 INT8 key 和 FP32 scale，score 时还会量化 query；当前 vLLM demo 读取的是未打包的 BF16 池。两者不能直接交换 cache 指针。若接入 vLLM 历史，需明确转换/只读适配，或提供接受该 BF16 布局的入口；缩放只能计入一次。量化后 Top-K 的选择也需独立验证，不能用实数公式证明其逐项一致。

## 3. 历史与双向临时块的联合 MLA

### 算法与原有路径的区别

MLA 历史缓存保存 512 维 latent，而不是展开后的每头 K/V。每个 draft query 一方面读取 $\mathcal H_i$ 中的历史，另一方面读取整个临时块。两部分必须共同归一化，但它们的 KV 展开权重可以不同。

以下按单个 head 表示。$q_i^D\in\mathbb R^{256}$ 是 draft query，$c_j^{AR},c_t^D\in\mathbb R^{512}$ 分别是历史及临时块 latent；$W_K^{AR},W_K^D\in\mathbb R^{256\times512}$ 和 $W_V^{AR},W_V^D\in\mathbb R^{256\times512}$ 是相应展开权重。

$$
z_{ij}^{AR}=\frac{(q_i^D)^{\mathsf T}W_K^{AR}c_j^{AR}}{\sqrt{256}},
\qquad
z_{it}^{D}=\frac{(q_i^D)^{\mathsf T}W_K^Dc_t^D}{\sqrt{256}}.
\tag{3}
$$

| 原模型 AR 的单段计算 | Orthrus 的联合计算 |
| --- | --- |
| $\displaystyle o_p^{AR}=\frac{\sum_{j\in\mathcal H_p^{AR}}e^{z_{pj}^{AR}}W_V^{AR}c_j^{AR}}{\sum_{j\in\mathcal H_p^{AR}}e^{z_{pj}^{AR}}}$，query 和边界均来自 AR。 | $\displaystyle o_i^D=\frac{\sum_{j\in\mathcal H_i}e^{z_{ij}^{AR}}W_V^{AR}c_j^{AR}+\sum_{t=0}^{K-1}e^{z_{it}^{D}}W_V^Dc_t^D}{\sum_{j\in\mathcal H_i}e^{z_{ij}^{AR}}+\sum_{t=0}^{K-1}e^{z_{it}^{D}}}$，使用 draft query。 |

左列的 $z_{pj}^{AR}=(q_p^{AR})^{\mathsf T}W_K^{AR}c_j^{AR}/\sqrt{256}$；$\mathcal H_p^{AR}$ 是该 AR 位置按自身因果边界选出的集合。右列的分数则按式 (3) 计算。

GLM 这里没有 DSV4 的 Attention sink，也不需要 RoPE。块内所有 $t$ 均可见，但历史仍止于 anchor 前，不会读取未来 AR 状态。

当前组合分别计算历史和临时块。这是利用现有算子的一种实现方式，**不是算法要求必须分成两个 kernel**。令两段的局部输出、自然对数 LSE 为 $(o_{i,H},L_{i,H})$ 和 $(o_{i,B},L_{i,B})$，则

$$
L_i=\log\!\left(e^{L_{i,H}}+e^{L_{i,B}}\right),\qquad
o_i^D=e^{L_{i,H}-L_i}o_{i,H}+e^{L_{i,B}-L_i}o_{i,B}.
\tag{4}
$$

空历史取 $L_{i,H}=-\infty$、$o_{i,H}=0$。历史和块内输出要先分别经过各自的 $W_V$，到同一 256 维 value 空间后再合并；**不能直接合并两个 512 维 latent 输出，再统一乘 draft 的 $W_V$**。

### 当前 Torch-NPU 组合

下面给出实际调用的完整计算主体，省略的只有输入合法性检查。`indices=[K,S]` 为上一节的结果，`query=[K,H,256]`，历史和临时块分别为 `[a,512]`、`[K,512]`。历史按索引 gather；没有展开整段历史的每头 K/V。

```python
def segment(query, latent, scale):
    # 每行 query 视作独立 batch，读取自己的 selected latent。
    # query=[K,H,512]，latent=[K,S,512]。
    if latent.shape[1] == 0:
        return (torch.zeros_like(query),
                query.new_full(query.shape[:2], -torch.inf, dtype=torch.float32))
    kv = latent[:, :, None].contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query[:, None].contiguous(), kv, kv,
        num_heads=query.shape[1], num_key_value_heads=1,
        input_layout="BSND", sparse_mode=0, scale=scale,
        softmax_lse_flag=True,
        actual_seq_lengths=[1] * len(query),
        actual_seq_lengths_kv=[latent.shape[1]] * len(query),
    )[:2]
    return output[:, 0], lse.reshape(query.shape[:2]).float()


def block_segment(query, block, scale):
    # 所有 K 行共读一份临时块，以 sequence 维执行，不复制 K 份 KV。
    kv = block[None, :, None].contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query[None].contiguous(), kv, kv,
        num_heads=query.shape[1], num_key_value_heads=1,
        input_layout="BSND", sparse_mode=0, scale=scale,
        softmax_lse_flag=True, actual_seq_lengths=[len(query)],
        actual_seq_lengths_kv=[len(block)],
    )[:2]
    return output[0], lse.reshape(query.shape[1], len(query)).T.float()


def joint_mla(query, history, block, indices, ar_key, ar_value,
              draft_key, draft_value):
    # 历史与临时块使用各自的 K 展开权重吸收 query。
    q_ar = torch.einsum("qhd,hdr->qhr", query, ar_key)
    q_d = torch.einsum("qhd,hdr->qhr", query, draft_key)
    h, lh = segment(q_ar, history[indices], 256**-.5)
    b, lb = block_segment(q_d, block, 256**-.5)
    h = torch.einsum("qhr,hvr->qhv", h, ar_value)
    b = torch.einsum("qhr,hvr->qhv", b, draft_value)
    if indices.shape[1] == 0:
        return b
    # 每个 query/head 独立合并，不能简单 h+b。
    output, _ = torch_npu.npu_attention_update(
        [lh.reshape(-1), lb.reshape(-1)],
        [h.reshape(-1, 256), b.reshape(-1, 256)], 0,
    )
    return output.view_as(h)
```

这里用 FIA 返回局部输出和 LSE，再用 `npu_attention_update` 完成式 (4)。历史的选择因 query 而异，仍按行读取；临时块则全部行共享，只保存一份 KV。代码对应 `glm53_ops.py::sparse_draft_attention`；完整实现还提供纯 Torch 参考路径，供数值比较。

已测的 K=2/8、8/32/64 heads 以及单独的 K=32/8-head Attention 用例中，共享块调用约 168–175 μs，重复展开路径约 220–259 μs。该 K=32 检查仅针对 MLA，不能据此宣布完整 KDA/模型支持 K=32。

### PyPTO 具体适配

`decode_sparse_attn(absorbed_query, latent_cache, block_table, request_ids, topk_indices, output)` **已经是只读的多 query 分页计算**。cache 写入发生在独立的 `mla_cache_write`，因果边界来自上游索引，而不是这个 Attention 核心中不可修改的三角 mask。

当前缺口是：该接口只接收一类 latent，且只返回归一化后的 latent 输出，不能直接表达式 (3) 的两套展开权重和式 (4) 的联合分母。建议在保留现有页读取和 QK/PV 核心的基础上，补一个联合入口，逻辑接口如下，名称仅为建议：

```text
draft_sparse_mla(
  q_ar_absorbed[K,H,512], q_d_absorbed[K,H,512],
  ar_latent_pages, block_table, request_ids, history_indices[K,S],
  block_latent[K,512], W_V_AR[H,256,512], W_V_D[H,256,512]
) -> attention[K,H,256]
```

内部可把临时块加入同一计算图，也可先扩展历史入口，输出自然对数 LSE，再与临时块合并。已有代码内部维护块级 $(m,\ell,u)$，分别是最大分数、移位指数和与加权 latent 和。保留这些统计即可做联合归一化，不必强制新增一个独立的“LSE 合并 kernel”。不同段的 $u$ 仍须经过各自 $W_V$ 才能相加。

具体地，令下标 $H,B$ 代表历史和临时块，现有分块归约可扩展为

$$
\begin{aligned}
m&=\max(m_H,m_B),\quad \alpha_r=e^{m_r-m},\\
\ell&=\alpha_H\ell_H+\alpha_B\ell_B,\\
u&=\alpha_H W_V^{AR}u_H+\alpha_B W_V^D u_B,\qquad o=u/\ell.
\end{aligned}
$$

这里 $u_H,u_B$ 是各段以自身 $m_r$ 为基准的未归一化 512 维 latent 和，$u$ 才是合并后的 256 维 value 和。这与式 (4) 等价，明确了已有归约中哪些量能沿用、哪些必须增加 value 变换。

还需要调整输出编排：现有 decode 的 `mla_epilog_decode` 将 AR 的 value 展开吸收进输出权重 `w_o_absorbed`。draft 不能直接复用这一套吸收权重处理两段。先得到共同的 `[K,H,256]` 结果后，可以复用现有 `mla_epilog_prefill / mla_epilog_w8a8`，使用 draft 的实际 O 投影。

这部分是主要的计算扩展。现有 demo 先复制有效历史，再 gather 每行选中的 latent；PyPTO 可以直接按页读取选中历史，避免完整前缀复制和逐 query 的选中行物化。**这是可实现的优化方向，尚未提供 PyPTO 版性能实测。**

## 4. Draft KDA 的临时块执行

### 算法与原有路径的区别

KDA 的历史不是逐 token KV，而是宽度 4 卷积留下的 3 行输入，以及每个 head 的 $128\times128$ 递推矩阵。draft 从已接受位置的这两类状态出发，执行自己的投影、卷积和递推，但不得覆盖 AR 状态。

以下省略 head 下标，$q_i,k_i,v_i\in\mathbb R^{128}$ 为 draft 投影经过逐通道因果卷积和 SiLU 后的向量。$\widehat q_i=\operatorname{L2Norm}(q_i)/\sqrt{128}$、$\widehat k_i=\operatorname{L2Norm}(k_i)$；$f_i,b_i$ 是衰减和更新门的未激活值，$A_{\log},d_t$ 为学习参数。

$$
\gamma_i=-5\,\sigma\!\left(e^{A_{\log}}\odot(f_i+d_t)\right),
\qquad \beta_i=\sigma(b_i).
\tag{5}
$$

状态采用 value-by-key 布局 $S_i\in\mathbb R^{128\times128}$，按下式递推：

$$
\begin{aligned}
\bar S_i&=S_{i-1}\operatorname{diag}(e^{\gamma_i}),\\
\delta_i&=\beta_i\left(v_i-\bar S_i\widehat k_i\right),\\
S_i&=\bar S_i+\delta_i\widehat k_i^{\mathsf T},\\
y_i^{\rightarrow}&=S_i\widehat q_i.
\end{aligned}
\tag{6}
$$

| 原模型 AR | 因果 draft KDA |
| --- | --- |
| 用 AR 参数消费真实 token，并将更新状态保存为目标历史。 | 用 draft 参数消费 anchor/MASK，从 $S_{-1}=\operatorname{copy}(S_{a-1}^{AR})$ 和 AR 卷积尾部副本起步；状态仅用于本次 draft。递推公式不变。 |

门控双向策略再将当前块投影倒序，卷积尾部、递推矩阵均从零开始，复用式 (6)。将结果恢复原顺序记为 $y_i^{\leftarrow}$，在 gated RMSNorm 之前融合：

$$
\widetilde y_i=y_i^{\rightarrow}+\tanh(\rho)\odot y_i^{\leftarrow},\qquad
o_i=W_O^D\operatorname{ConcatHeads}
\left[\operatorname{RMSNorm}_{w^D}(\widetilde y_i)
\odot\sigma(g_i)\right].
\tag{7}
$$

$\rho$ 是每 head 的反向门参数，$g_i$ 是输出门投影；因果策略直接取 $\widetilde y_i=y_i^{\rightarrow}$。反向只遍历当前块，不反向重算历史。$\rho=0$ 时退化为因果策略，但判断反向分支是否生效必须使用非零门测试。

### 当前现成算子组合

完整分支的投影和输出见 demo 的 `kda()`；其核心实现如下。`rows=[K,3H·128]` 是 QKV 卷积输入，`tail=[3,3H·128]`，`initial=[H,128,128]` 是 FP32 状态。

```python
def convolve(rows, tail, conv_weight, heads):
    # kda_convolve 复用现成卷积；只写本次调用分配的临时状态。
    mixed = kda_convolve(rows, tail, conv_weight[:, 0], backend="auto")
    return mixed.reshape(-1, 3, heads, 128).unbind(1)


def recurrent(q, k, v, gate, beta, initial, a_log, dt_bias):
    # 使用目标 GLM 相同的 AscendC KDA 接口，状态槽由 draft 独立分配。
    length, heads, dim = q.shape
    states = initial.new_empty(length, heads, dim, dim)
    states[0].copy_(initial)
    output = run_recurrent_kda(
        q[None], k[None], v[None], gate[None], beta[None], states,
        torch.tensor([0, length], device=q.device, dtype=torch.int32),
        torch.arange(length, device=q.device, dtype=torch.int32)[None],
        a_log.reshape(-1).contiguous(), dt_bias.contiguous(),
        beta_is_preprocessed=False, lower_bound=-5.,
    )
    return output[0]


def block_kda(rows, tail, initial, gate, beta, conv_weight, a_log, dt_bias,
              backward_gate=None):
    heads = initial.shape[0]
    q, k, v = [x.contiguous() for x in convolve(rows, tail, conv_weight, heads)]
    forward = recurrent(q, k, v, gate.contiguous(), beta.contiguous(),
                        initial, a_log, dt_bias)
    if backward_gate is None:
        return forward
    rq, rk, rv = [x.contiguous() for x in
                  convolve(rows.flip(0), torch.zeros_like(tail), conv_weight, heads)]
    reverse = recurrent(rq, rk, rv, gate.flip(0).contiguous(), beta.flip(0).contiguous(),
                        torch.zeros_like(initial), a_log, dt_bias)
    return forward + backward_gate.tanh()[None, :, None] * reverse.flip(0)
```

此后两种策略都执行同一 gated RMSNorm 和 O 投影。双向策略不是两路分别归一化后相加，也不是再加一个 SDPA。

`kda_convolve` 在实测有收益的形状使用 GLM 原生融合卷积入口，核心调用如下。状态槽 0 保留为 null，槽 1 是独立的历史尾部副本；不会把目标状态传给可写参数：

```python
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d

state = rows.new_zeros((2, 3, rows.shape[1]))
state[1].copy_(tail)
mixed = causal_conv1d(
    rows.contiguous(), weight.to(rows.dtype).T.contiguous(), state,
    torch.tensor([0, len(rows)], dtype=torch.int32, device=rows.device),
    torch.ones(1, dtype=torch.int32, device=rows.device), run_mode=0,
    initial_state_mode=torch.ones(1, dtype=torch.bool, device=rows.device),
)
```

这与目标 GLM 使用同一现成 AscendC/FLA 卷积入口，不是自研 kernel。参数在模型内以 FP32 存储，调用前按原生路径转为激活精度。本次使用真实投影式的非连续 QKV 切片、FP32 参数，K=8 的 32/64 heads 分别约为 568/560 μs，原 `conv1d + SiLU` 约为 618/1067 μs；8 heads 原组合约 264 μs，比融合调用约 566 μs 更快。因此 `auto` 仅在已测的 32/64 heads、BF16 激活、K≤8 时选融合入口，其余保留 Torch 组合。CPU 参考始终可运行，不把“融合”当作自动更优的依据。

### PyPTO 具体适配

`kda_conv_prefill` 已接收 `query_start_loc` 和每请求的 `state_rows`，可以消费带非零历史尾部的完整块；`prefill_kda` 也已支持变长块、非零初始矩阵和 16 行 chunk。把状态参数换成临时副本，即可复用因果数学核心。**不要把 K 行都映射到同一 state row，再一次调用 `decode_kda`**：decode 的行是独立任务，这会让同一请求的递推发生竞争。

反向策略需要倒序打包当前块、清零独立状态、再次调用现有卷积和 KDA，再恢复顺序、按 head 门控融合。融合可先用逐元素操作接入 `kda_output` 前，是否另写融合 kernel 由 profiler 决定，不是新的 KDA 数学算法。反向卷积使用倒序的原始投影行，不能直接把正向卷积的 Q/K/V 倒序代替。

移植时仍需对齐 state 的 VK 布局、门函数、Q/K 归一化 epsilon、投影及状态精度。PyPTO 的 chunk 实现和逐 token 参考的低精度舍入不同，应做数值容差测试而非要求位级相等。块长变化涉及 tile/padding 和编译形状，不等于每换一个 $K$ 就重新发明递推算子。

## 5. 完整 demo 与推理引擎边界

可运行 demo 为 `benchmarks/glm53_orthrus_demo.py`。它补齐两类 Attention 各自的投影和输出，调用本项目真实的 `select_history / sparse_draft_attention / kda_native`，不是另外写一套与推理接入无关的示意算法。

| 部分 | demo 中的完整链路 |
| --- | --- |
| MLA | `4096 → Q/latent 低秩投影 → RMSNorm → 8×256 query`；32-head indexer 检索；AR/draft 两套 KV 展开；FIA、LSE 合并；O 投影回 4096。 |
| KDA | `4096 → QKV / beta / 两类 gate 投影`；宽度 4 逐通道卷积；8×128 head 的临时递推；可选反向块；gated RMSNorm；O 投影回 4096。 |

该几何对应 TP8 的单分片：MLA/KDA 各 8 个 head，Indexer 保留完整 32 个 head，latent 512、query rank 1536。权重和历史为合成输入。**两条分支分别演示，不是把它们当作同一层相加，也不包含 45 层、共享 mHC/MoE 或采样器。** 完整原生推理接入另由 `glm53_native.py / glm53_proposer.py` 和 `scripts/run_glm53_vllm.py` 提供。

从 `online-speculation` 目录运行：

```bash
# 纯 Torch 计算参考，不需要模型权重。
PYTHONPATH=src python benchmarks/glm53_orthrus_demo.py --backend torch

# 在已初始化的 Ascend/vLLM-Ascend 环境中运行；使用可见设备 0。
python scripts/run_in_runtime.py benchmarks/glm53_orthrus_demo.py \
  --backend npu --block-size 8 --prefix-length 2055 --kda-mode bidirectional
```

PyPTO 完整接入还需要一个 draft 前向入口：加载匹配策略的 head，执行共享主干，把请求的 anchor 边界、页表和临时 KDA 状态传给上述计算，跳过 target 的 cache 写入。现有 embedding、mHC、norm、FFN/MoE、LM head 不因 Orthrus 产生新的数学需求，但仍要完成权重布局、TP/EP、量化 scale、padding 和调用编排；“可复用”不等于不需要联调。

head 的加载契约应记录基模、块长、MASK token、混合策略和参数布局。量化 head 还需匹配自身 scale，不能在训练后的参数上直接沿用原模型 scale；这属于参数交付和加载，不是另一个 Attention 算子。

引擎负责分配缓存、调度请求、验证候选、采样和提交/拒绝。算子不决定接受几个 token，但必须服从只读 target 历史和独立临时状态的接口约束。**draft 的 latent、卷积和递推状态永不转正；接受后保留的是 target 按因果规则重新计算的状态。**

若将验证器也迁入 PyPTO，需要逐位置 KDA/卷积 checkpoint 或拒绝后重放，不能只保存候选块末尾状态。当前 `prefill_kda` 返回各位置输出并更新最终矩阵，不自动提供完整投机回滚协议；库文档描述的 MTP 顺序验证，也不能等同于 Orthrus 整块验证。此项属于引擎与状态接口工作，不应算作“再开发一个 Attention 算子”。

## 6. 验证范围与交付顺序

推理接入使用 GLM 的固定 **vLLM-Ascend 0.30** 镜像，不沿用 DSV4 的 0.26 镜像。两者的 KDA 与卷积 API 不同，不能将一个镜像的算子预检作为另一个镜像的兼容证明。本轮新增代码、数值对照、局部计时及原生接入复验见 [算子复核](GLM53_INFERENCE_OPS_AUDIT.md)。

本报告区分三类证据：

- **PyPTO 支持情况是公开源码核对**，尚未编译运行 PyPTO Orthrus kernel。不能把本项目 NPU 组合测试写成 PyPTO 验收。
- **现成算子组合已做 CPU 数学和单卡 NPU 测试**：包括两套不同 KV 展开权重与显式 SDPA 等价、空历史、KPool 尾部、超过 512 个池、双向信息流、非零反向门及状态只读。新增完整 Attention 分支的测试记录见 `validation/glm53/pypto-report-demo/`。
- **原生模型接入**：真实源权重的五层截层模型已通过相应单/双卡 eager 测试；九层四卡功能闭环通过，但 greedy 未验收；完整 45 层八卡仍未验收。截层、初始化 head 和合成 demo 都不能证明完整模型正确性或加速收益。

本轮全套 CPU 回归为 **624 项通过、44 项硬件专项跳过**。完整 Attention 分支的 NPU/FP32 对照覆盖 $K=2/8$、历史长度 0/3/2055 和两种 KDA 策略；还单独比较了共享块调用、原生卷积以及默认后端选择，卷积参数同时覆盖 BF16 与 FP32 存储。FP32 参考使用同一组已舍入权重和输入，完整分支的相对 RMS 容差预设为 4%；这是组合实现的有限数值检查，不是端到端模型误差界。具体计数、命令和结果以本轮复核记录为准，不与历史用例重复相加。

建议先交付 **只读 Indexer 编排 + 联合 MLA + 因果 KDA 临时状态入口**，在已有 Torch-NPU 路径上逐模块对照。门控反向 KDA作为独立策略扩展，复用同一递推核心。完成这些后，再处理完整 PyPTO 模型入口、整块验证与回滚，最后依据性能数据决定是否融合 query 投影、倒序打包和输出门控。

验收至少覆盖：所有 query 的历史止于同一 anchor；当前块无因果遮挡；不同 AR/draft 展开权重下式 (3) 正确；空历史不产生 NaN；跨页和池尾部索引正确；draft 前后 target 状态逐字节不变；反向块只影响临时分支；短块 padding 不成为可见 token。性能测试应单列历史 gather、FIA/联合 MLA、KDA、状态复制和调度耗时，不用初始化 head 宣称实际加速。

## 参考版本

- [GLM-5.3-Flash 官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)：用于核对混合结构和维度。
- [pypto-lib 固定版本](https://github.com/hw-native-sys/pypto-lib/tree/2aef4534f8769d8ce8d49d9dde7e5aeebc472197)：`models/glm5_3_flash/{indexer,attention,kda,decode_fwd,prefill_fwd}.py` 及对应模型文档。
- [pypto-serving 固定版本](https://github.com/hw-native-sys/pypto-serving/tree/5f548be23d7af120b20a40694601f5bb5b8b7a90)：本次公开主线的模型目录不含 GLM runner；不对其他分支或未公开代码作结论。

本项目对应源码：`src/blockspec/vllm_ascend/{glm53_ops,glm53_native,glm53_proposer}.py`；独立 demo：`benchmarks/glm53_orthrus_demo.py`；检查逻辑独立保存在 `tests/test_glm53_orthrus_demo.py`，不混入对外展示的计算分支。
