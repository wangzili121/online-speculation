# Orthrus-DSV4：PyPTO 推理算子需求

## 1. 目标与已有基础

Orthrus 在每层增加独立的 draft Attention，共享主模型的 embedding、mHC、外层 norm、MoE 和 LM head。输入为已知 anchor 加若干 MASK，经过一次前向产生候选，再由主模型验证。本文针对 `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8`，不涉及训练和在线更新。

DSV4 的 43 层包含 2 层 SWA、21 层 CSA/C4、20 层 HCA/C128。draft 读取各层对应的主模型历史，并让当前块内的位置彼此可见。它仍需计算自己的逐层激活，不能复用主模型本轮的结果；临时 draft KV 也不会成为接受后的主模型 KV。

**按四个逻辑模块组织需求：公共双向块、SWA、HCA、CSA。不是四套新 Attention 数学，也不要求恰好四个独立 kernel。** 主要增量是临时块输入、固定历史边界和只读执行入口。

本次源码核对固定到 `pypto-lib@2aef4534f8769d8ce8d49d9dde7e5aeebc472197`，核对日期为 2026-11-03。下表文件均相对于 `models/deepseek_v4_flash_dspark/`；这是源码能力分析，不是 PyPTO Orthrus 的运行验收。

| 计算模块 | 作用 | PyPTO 当前已有能力 | Orthrus 所需适配 |
| --- | --- | --- | --- |
| 公共双向块 | K 个 draft 位置彼此读取 | `dspark_attention.py::dspark_attention` 已有块内双向计算，同时把当前块 KV 写入 DSpark 私有 cache，并在最终归约中加入 sink。 | 复用 QK/softmax/PV 核心，接收临时 KV 并提供可联合归约的统计，或直接融合进后三类 Attention。不能把私有 cache 写入照搬到目标 cache。 |
| SWA | 当前块读取滑窗历史 | `decode_sparse_attn_swa.py` 已有按索引读取历史、局部 softmax 统计和最终合并；DSpark 也已有“窗口 + 双向块”。 | 使用 draft 参数和同一 anchor 的窗口索引，追加完整临时块；历史指针只读。优先适配已有联合计算，不重写 SDPA。 |
| HCA / C128 | 读取滑窗和全部 C128 历史 | `decode_sparse_attn_hca.py::sparse_attn_hca` 已分别维护局部段与压缩段的 softmax 状态；完整 `decode_hca.py` 还执行 compressor/cache 更新。 | 保留历史读取，另接临时块；有效压缩行数固定为 anchor 前的完整组数，跳过 compressor 和持久写入，再共同归约。 |
| CSA / C4 | Indexer 选 C4 历史后计算 Attention | `decode_indexer.py` 已有 query 投影、量化检索和 Top-K；`decode_sparse_attn_csa.py` 已有局部及 C4 稀疏读取和归约。 | 用 diffusion indexer 的 query/head-weight 参数查询同一历史边界，再接完整临时块。复用 Top-K，分离 query-only 执行与目标 compressor/cache 写入。 |

顶层编排与计算核心要分开看，不能概括成“现有 Attention 都会写 cache”。主要写入在 QKV/compressor 编排中；但 CSA 核心还保留一次 `ori_kv_flat[0:1] = ori_kv_flat[0:1]` 的依赖标记，严格只读入口应改为显式依赖，而非对共享历史执行 store。

## 2. 公共双向块与归一化

anchor 位于 a，历史范围为 `[0,a)`，块长为 K。第 i 行的实际位置仍为 `a+i`，第 0 行为 anchor，其余为 MASK。固定历史边界不等于把所有行的 RoPE 位置也改成 a。

DSV4 每行使用同一个 512 维向量作为 K/V。省略 batch/head 下标，q、c 是完成 norm/RoPE 后的 query、KV，s 是当前 head 的 sink logit，定义

$$
\mathcal A(q,\mathcal C;s)=
\frac{\sum_{c\in\mathcal C}e^{q^\top c/\sqrt{512}}c}
{e^s+\sum_{c\in\mathcal C}e^{q^\top c/\sqrt{512}}}.
$$

临时块为 $\mathcal B^D=\{c_0^D,\ldots,c_{K-1}^D\}$。每行读取全部临时块，没有三角遮挡；这些是 MASK 的中间特征，不是提前知道答案。

分段实现时，每段保留输出 $O_r$ 和自然对数归一化量 $L_r$，sink 不在各段重复加入：

$$
L=\log\!\left(e^s+\sum_r e^{L_r}\right),\qquad
O=\sum_r e^{L_r-L}O_r.
$$

这与一次拼接 softmax 等价，不要求必须分两次调用。PyPTO 可复用已有在线 softmax 的 $(m,\ell,u)$，在同一图内融合，不必新增独立“LSE kernel”。空段取 $O_r=0,L_r=-\infty$，sink 只计一次，随后仍执行原有 inverse RoPE 和 O 投影。

DSpark 已有双向可见性，需调整的是输入和归约边界。当前源码的 `DSPARK_SPEC_TOKENS=7`、`DECODE_SEQ=8` 与本项目 K=32 的约定不同，不能直接用同名常量替代 K。需先明确 query 行数、是否含 anchor、padding 和候选数，再编译对应形状；改变 K 不要求重新设计 Attention 算法。

## 3. SWA：滑窗历史与整个临时块

窗口宽度 W=128，令 $\mathcal W_a=\{c_j^{AR}\mid\max(0,a-W)\le j<a\}$：

| 主模型 AR | Orthrus draft |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},\mathcal W_{p+1};s^{AR})$ | $o_i^D=\mathcal A(q_i^D,\mathcal W_a\cup\mathcal B^D;s^D)$ |

AR 按每行位置推进；draft 各行共享同一历史窗口，并读取整个临时块。PyPTO 可保留窗口索引构造，但所有行的历史截止点取同一 anchor；真实 query 位置另用于 RoPE。

正式 Torch-NPU 路径把最多 128 行历史与 K 行临时 KV 合在一次 FIA 中，mask 只屏蔽历史 padding：

```python
# query=[B,K,H,512]；history=[B,W,1,512]；block=[B,K,1,512]
kv = torch.cat((history, block), dim=1)
mask = history_and_block_mask(history_lengths, history.shape[1], block.shape[1])
output, lse = torch_npu.npu_fused_infer_attention_score(
    query, kv, kv, input_layout="BSND", sparse_mode=0,
    num_heads=query.shape[2], num_key_value_heads=1, scale=512**-.5,
    atten_mask=mask, actual_seq_lengths=[query.shape[1]] * len(query),
    actual_seq_lengths_kv=[kv.shape[1]] * len(query), softmax_lse_flag=True,
)[:2]
lse = lse.transpose(1, 2)  # [B,H,K,1] -> [B,K,H,1]
```

三类 Attention 共用同一种块内双向数学。SWA 直接联合，HCA、CSA 再加入各自压缩历史。

## 4. HCA：C128 压缩历史

令 $\mathcal C_{128,a}=\{c_r^{AR,128}\mid0\le r<\lfloor a/128\rfloor\}$，只包含完成压缩的历史行，未满一组的状态不额外当作压缩 KV。

| 主模型 AR | Orthrus draft |
| --- | --- |
| 按 p 的因果边界读取局部与完成的 C128 行。 | $o_i^D=\mathcal A(q_i^D,\mathcal W_a\cup\mathcal C_{128,a}\cup\mathcal B^D;s^D)$ |

PyPTO 已有两类历史的归约状态，新增临时块输入，并分离“有效历史行数”和“query 位置”。不能把 `a+i` 当作已提交长度，也不能为了读历史调用目标 compressor。

当前 NPU 路径按页布局选择已有算子：页长是 128 的整数倍且在 FIA 支持范围内时直接调用分页 FIA；较小页调用 `npu_sparse_flash_attention`，索引覆盖**全部**有效 C128 行，不做 Top-K 截断。保留 `ORTHRUS_C128_BACKEND=gather` 对照回退。

两条路径返回输出和 LSE，与 SWA/临时块共同归一化。PyPTO 可以扩展现有 HCA 分块归约，不必照搬 Torch 的多次 API 调用。

## 5. CSA：Diffusion indexer 与 C4 历史

令 $u_{ih}^D$ 为 indexer query，$w_{ih}^D$ 为已含模型缩放的 head 权重，$k_r^{AR,I}$ 为目标历史 indexer key：

$$
\begin{aligned}
v_{ir}&=\sum_h w_{ih}^D\operatorname{ReLU}((u_{ih}^D)^\top k_r^{AR,I}),
&&0\le r<\lfloor a/4\rfloor,\\
\mathcal T_i&=\operatorname{TopK}_{\min(512,\lfloor a/4\rfloor)}(v_i),\\
o_i^D&=\mathcal A\left(q_i^D,
\mathcal W_a\cup\{c_r^{AR,4}:r\in\mathcal T_i\}\cup\mathcal B^D;s^D\right).
\end{aligned}
$$

AR 使用自身 query 和逐行历史边界；draft 使用自己的 query，全部行只检索 anchor 前历史。Top-K 不变，当前块不经检索而是完整可见。PyPTO 已有 Hadamard/量化检索，应沿用其变换、cache 和 scale 约定，不能靠浮点公式保证量化 Top-K 逐项一致。

正式实现使用原生量化 Lightning Indexer，设置非因果历史查询元数据。返回值是压缩行号，过滤未完成组即可，不能再次除以 4。随后直接按页读取 C4：

```python
# query=[B,K,H,512]；indices=[B,K,512]，无效项为 -1。
# RoPE 已纳入 DSV4 的 512 维向量，辅助 query 设零避免重复计分。
aux_q = query.new_zeros(*query.shape[:-1], 64)
output, maximum, total = torch_npu.npu_sparse_flash_attention(
    query, cache, cache, indices.to(torch.int32).unsqueeze(2), 512**-.5,
    block_table=block_table,
    actual_seq_lengths_query=torch.full_like(lengths, query.shape[1]),
    actual_seq_lengths_kv=lengths, query_rope=aux_q, key_rope=cache[..., -64:],
    sparse_mode=0, attention_mode=2, layout_query="BSND", layout_kv="PA_BSND",
    return_softmax_lse=True,
)
lse = (maximum + total.log()).squeeze(1).unsqueeze(-1)
empty = (indices >= 0).sum(-1)[..., None, None] == 0
output, lse = output.masked_fill(empty, 0), lse.masked_fill(empty, -torch.inf)
```

同一分页读取也用于 C128 小页，但索引含全部历史。PyPTO 已有稀疏读取与归约；需扩展临时块和联合归约接口，不是重新开发排序或页式存储。

## 6. 现有算子选择与完整接入

正式实现为 `src/blockspec/vllm_ascend/dsv4.py`。Q/KV 共用一次输入动态量化，使用已有量化 matmul、融合 RMSNorm/量化、partial RoPE 和原生 O 投影；mHC、MoE 等共享层继续调用主模型。最终用 `npu_attention_update` 合并各段，单独计入 FP32 sink。

固定镜像的 FIA v2 内置 sink 需降为 BF16，既有实验未通过原 LSE 容差，因此保留 FP32 sink 和原生合并。没有因为新 API 就替换，也没有把所有压缩历史搬入 dense Attention。**关键路径已使用融合/分页算子，但不宣称所有形状均达全局最优。**

独立单层 demo 为 `scripts/demo_torch_npu_dsv4_layer.py`，计算在 `torch_npu_dsv4_layer.py`：包含投影、norm/RoPE、三类 Attention、sink、inverse RoPE 和 O 投影。本轮将 SWA/块联合计算及原生合并对齐正式路径。demo 使用合成连续历史，CSA 的 gather 便于阅读，不能替代正式分页路径的性能测试。

```bash
python scripts/run_in_runtime.py scripts/demo_torch_npu_dsv4_layer.py --block-size 32
ORTHRUS_RUN_NPU_TESTS=1 PYTHONPATH=src python -m pytest -q tests/test_dsv4_layer_npu.py
python scripts/run_in_runtime.py scripts/benchmark_dsv4_attention_paths.py --case sparse
```

引擎负责 head 加载、共享层绑定、请求页表、anchor/MASK 输入、验证和接受前缀的提交/回滚；不是新增 Attention kernel。draft 不更新目标 compressor/indexer 状态；接受后保留 verifier 重新计算的因果 KV，不把 draft KV 转正。

PyPTO 应先明确历史/临时块 ABI，再适配 SWA、HCA、CSA，最后接共享模型前向与验证。共享层没有新增数学，但仍需对齐权重布局、量化描述、TP/EP 和 padding，不能凭同名算子就认定整模型已通。

## 7. 验证边界

本轮单卡与 CPU 结果见 `validation/dsv4/ops-20261103/`，覆盖联合归一化、sink、空历史、随机分页、C4/C128 索引及历史不变。这是现成算子组合的测试，不是 PyPTO 实现验收。

CPU 为 620 项通过、33 项跳过；单卡 SWA 联合路径 36 组、C4 6 组、C128 8 组及完整单层 6 组通过。合并专项中，原生合并约 229 μs，Torch 组合约 560 μs；与独立 FP32 拼接计算的最大绝对误差约 0.00358。这是局部算子数据，不是模型吞吐提升。

已有真实权重截层服务通过多种接受长度、连续拒绝、页复用和缓存检查；完整 43 层 TP8 曾启动并完成部分请求，尚未完成全部端到端验收。目标 C4 物理布局引起的舍入差异已有独立复现，见 `DSV4_C4_LAYOUT_REPRO.md`，不作为本项目新引入的缺陷，也不据此豁免未知差异。

八卡复验仍需同窗口运行 target-only、Orthrus 与 head 重载，核对长输出、多请求及 cache 生命周期。图模式、1M 上下文、业务 goodput 单独验收，不用初始化 head 推导加速结论。

参考：[PyPTO 固定源码](https://github.com/hw-native-sys/pypto-lib/tree/2aef4534f8769d8ce8d49d9dde7e5aeebc472197)。本文不覆盖 DSV4.1 的其他结构。
