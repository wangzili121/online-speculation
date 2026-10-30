# Orthrus-Qwen3.8：PyPTO 算子适配需求

评估日期：2026-10-30。对象：Qwen3.8-27B 文本模型；以当前 Torch/NPU 实现的计算语义为准。
PyPTO 代码核对版本：`pypto-lib@ab0fcd333d71701d063db4b9218ad2422f891ec4`。

## 1. 模型结构与适配范围

**核心是两条计算路径：Full Attention 的双向块计算，以及 GDN 的带初始状态递推。不是照搬 DSV4 的 SWA、HCA、CSA，也不等于要从零开发一组新算子。**

Qwen3.8-27B 的文本部分共有 64 层，结构如下：

```text
Embedding -> [GDN -> GDN -> GDN -> Full Attention] × 16 -> Norm -> LM head
              每一层还包含层外 Norm、残差连接和 Dense MLP
```

Full Attention 保存逐 token 的历史 K/V；GDN 保存卷积尾部输入和递推状态矩阵，不保存同样的历史 KV。
Orthrus 在每层增加独立的 draft mixer 参数，复用并冻结 embedding、层外 norm、MLP 和 LM head；复用参数不等于复用计算结果，draft 仍须执行这些共享模块。

当前适配沿用 Orthrus Qwen3.5 的混合模型定义：**Full Attention 块内双向，GDN 块内仍因果递推**，不把 GDN 替换成 SDPA。
一次输入为 `anchor + (K-1) 个 MASK`：anchor 是已知的最后一个 token，MASK 是待预测位置的占位符；主模型历史截止到 anchor 之前。
当前服务使用 K=32，即一次提出 31 个候选。K 是配置，不是算法固定常数；改变 K 不改变公式，但可能需要新的编译规格和 tiling。

两种接入范围必须区分：

- **仅用 PyPTO 加速 draft：**保留 vLLM-Ascend 的主模型验证与正式缓存管理，新增下述 draft 前向入口和状态接口即可。
- **独立 PyPTO 服务：**还须完成 Qwen3.8 主模型的 prefill/decode、验证块和状态恢复，不能只实现 draft Attention 就认为整套服务完成。

## 2. Full Attention：历史前缀加双向预测块

### 计算要求

设 anchor 的绝对位置为 \(a\)，块内位置为 \(i=0,\ldots,K-1\)，绝对位置为 \(a+i\)。
\(\mathbf K_H,\mathbf V_H\) 是主模型在 \(a\) 之前的历史；\(\mathbf Q_D,\mathbf K_D,\mathbf V_D\) 由本层 draft 参数计算。
对每个 query head，KV 按 GQA 分组映射，\(d=256\) 是 head 维度。普通 AR 与 draft 分别计算：

$$
\begin{aligned}
\mathbf o_i^{\mathrm{AR}}&=
\operatorname{softmax}\!\left(
\frac{\mathbf q_i^{\mathrm{AR}}[\mathbf K_H;\mathbf K_{\mathrm{AR},0:i}]^\top}{\sqrt d}
\right)[\mathbf V_H;\mathbf V_{\mathrm{AR},0:i}],\\
\mathbf O_D&=
\operatorname{softmax}\!\left(
\frac{\mathbf Q_D[\mathbf K_H;\mathbf K_D]^\top}{\sqrt d}
\right)[\mathbf V_H;\mathbf V_D].
\end{aligned}
$$

这里的分号表示沿序列维拼接，\(0:i\) 包含第 \(i\) 行。AR 使用主模型参数，第 \(i\) 行只看历史及当前块的 \(0\ldots i\)；draft 使用独立参数，每一行都看同一段历史及整个当前块。
单请求且输入全为有效位置时，不必构造大 mask；批量或 padding 输入仍须屏蔽无效位置和其他请求。
历史与临时块可以在同一 kernel 内计算，也可以分别计算输出/LSE 后精确合并，**不要求另外开发一个独立的 LSE 合并算子**。

该模型还有必须保留的前后处理：24 个 Q head、4 个 KV head、head_dim=256，GQA 比例为 6；Q/K 做 RMSNorm，仅前 64 维应用 RoPE。
当前执行路径的 Attention 输出还要乘 `sigmoid(output_gate)`，再经过 O projection。门控、投影布局和 norm 权重语义不能直接照搬普通 Qwen3。

### 现有基础与具体改动

PyPTO 的 `models/qwen3_14b/paged_attention_pypto.py` 已有 Q/K norm、RoPE、分页历史读取和 Attention，并把当前 K/V 写入持久 cache。
但这条实现针对每请求单 query decode，使用 head_dim=128、GQA 比例 5，且将预处理、cache append 和 Attention 编排在一起。
因此，不能只传一张新 mask 就直接运行本模型。

需要增加可接收 **K 行 query、只读历史、独立临时 K/V** 的入口，并适配上述 head 规格、部分 RoPE 和输出门控。
AR 路径保留因果语义；draft 路径关闭块内因果限制，且不执行目标 cache append。
首版可接收连续历史；若直接接 vLLM 分页历史，则还需约定 block table、有效长度和布局，避免每轮 gather。

下面是当前 Torch 路径的等价逻辑；省略了 TP 通信和精度转换，不是完整运行代码：

```python
# h 是经过本层共享 input norm 的 K 行 draft 表示。
q, gate, kd, vd = draft_project_and_split(h)  # query 与 gate 按 head 解包
q, kd = qk_norm_and_partial_rope(q, kd, positions=a + arange(K))
kh, vh = read_target_prefix(end=a)           # 只读 [0, a)，不包含 anchor
k_all, v_all = cat([kh, kd]), cat([vh, vd])
o = sdpa(q, k_all, v_all, is_causal=False, enable_gqa=True)
mixed = draft_o_proj(flatten_heads(o) * sigmoid(gate))
# kd/vd 只服务本轮 draft，不写进主模型持久 cache。
```

## 3. GDN：从主模型状态继续，但只更新临时副本

### 计算要求

**GDN 不是 softmax Attention，不能靠改 attention mask 支持。** 它先做宽度为 4 的逐通道因果卷积，再用门控 Delta Rule 更新历史状态。
Qwen3.8 的 Q/K 有 16 个 head，V 有 48 个 head，维度均为 128；一组 Q/K 对应三个 V head。

省略 batch/head 下标，令 \(\mathbf q_i,\mathbf k_i\) 为 L2 归一化后的列向量，\(\mathbf v_i\) 为 value。
采用原生缓存的 \(\mathbf S\in\mathbb R^{d_v\times d_k}\) 布局，\(d_k=d_v=128\)，单步递推为：

$$
\begin{aligned}
\beta_i&=\sigma(b_i),\qquad
g_i=-e^{A_{\log}}\operatorname{softplus}(u_i+\Delta t),\\
\overline{\mathbf S}_i&=e^{g_i}\mathbf S_{i-1},\\
\boldsymbol\delta_i&=\beta_i(\mathbf v_i-\overline{\mathbf S}_i\mathbf k_i),\\
\mathbf S_i&=\overline{\mathbf S}_i+\boldsymbol\delta_i\mathbf k_i^\top,\qquad
\mathbf o_i=\mathbf S_i\mathbf q_i/\sqrt{d_k}.
\end{aligned}
$$

其中 \(\sigma\) 为 sigmoid，\(b_i,u_i\) 是 draft 投影输出，\(A_{\log},\Delta t\) 是对应 head 的参数；\(g_i\) 是每个 V head 的标量，不是每通道独立向量。
\(\mathbf S_{-1}\) 取自主模型在 anchor 前的状态；卷积同时读取主模型已有的三个历史投影输入。
块内使用 draft 的卷积与递推参数计算，只更新临时工作状态；输出经本层 draft RMSNorm、`SiLU(z)` 门控及输出投影。

该递推可以用 chunk 算法并行实现，公式不要求逐 token 发起模型调用。但同一请求的不同位置有状态依赖，不能当作互不相关的 batch 行同时覆写同一状态槽。

### 现有基础与具体改动

PyPTO 的 `models/glm5_3_flash/kda.py` 已有投影、宽度 4 卷积、Q/K 归一化、decode 递推和 chunk prefill 的 DSL 实现，可以借用实现结构。
不过它是 **KDA，不是 Qwen GDN**：采用逐通道衰减、不同门控公式和 head 组织；其输出使用 sigmoid gate，也不同于这里的 SiLU。
该目录的模型索引仍标为 staging，不能把源码存在当作已完成 Qwen3.8 服务验收。

需要按上述 GDN 公式适配 head 分组、标量衰减、投影/卷积通道布局及输出门控，提供“传入 AR 初始状态、计算整个 draft 块”的调用方式。
现有原地更新接口可以接临时缓冲，不一定必须重写 kernel；关键是不能把 draft 更新写回主模型状态。
实现还须对齐 Q/K 归一化、状态累积和 norm/gate 的精度转换边界，不能仅凭代数公式相同判断数值已经对齐。

等价伪代码如下，递推内核可用已有 chunk 实现替换：

```python
conv_tail, s_ar = read_target_gdn_state(before=a)
s = clone(s_ar)                              # AR 历史状态只作初值
x, z, beta, log_decay = draft_gdn_project(h)
q, k, v = causal_conv4_and_split(conv_tail, x, draft_conv_weight)
q, k = l2_normalize(q), l2_normalize(k)
q, k = map_qk_to_value_heads(q, k)            # 每组 Q/K 对应三个 V head
for i in range(K):                           # 描述依赖，不要求 K 次 kernel launch
    s = exp(log_decay[i]) * s
    delta = beta[i] * (v[i] - s @ k[i])
    s = s + outer(delta, k[i])
    o[i] = s @ q[i] / sqrt(128)
mixed = draft_out_proj(draft_rmsnorm(o) * silu(z))
# 本轮 s、卷积状态和中间结果均不提交给 AR。
```

另有 [MegaGDN-PTO](https://github.com/huawei-csl/megagdn-pto) 提供 PTO-ISA 的 chunk GDN，并有 torch_npu/vLLM-Ascend 接入。
它是可优先评估的复用来源，**不是已经集成到 PyPTO 的同一套 DSL 算子**；仍需检查短块性能、初始状态、布局、运行时依赖和版本兼容，不能当作直接替换件。

## 4. 交付边界：算子、模型编排、推理引擎

| 工作项 | 当前基础与所需交付 | 归属 |
| --- | --- | --- |
| Full Attention 前后处理 | 适配 head_dim=256、GQA=6、部分 RoPE、Q/gate 布局、norm 和输出门控；不限于改 mask | Qwen3.8 基模适配 |
| Full Attention 块计算 | 在已有分页 Attention 基础上增加多 query、同一历史前缀和临时双向 K/V 的入口，不 append 目标 cache | Orthrus 特有 |
| GDN 前后处理及递推 | 从 KDA/PTO 实现复用可用部分，按 Qwen GDN 公式适配卷积、head 分组、门控、递推和精度；draft 接收 AR 状态副本 | 基模适配 + Orthrus 状态入口 |
| 验证后的状态恢复 | 仅替换 draft 时沿用原生 verifier；若主模型也移入 PyPTO，则须支持接受前缀的 GDN 状态选择或等价的正确恢复 | 算子输出契约 + 引擎编排 |

这些是能力交付项，**不是四个必须单独编译的 kernel**。投影、卷积、递推、输出可以按性能需求融合或拆分。
Embedding、GEMM、Dense MLP、残差等已有基础可复用，但仍要适配模型尺寸、权重布局、norm 的 `1+weight` 语义和 TP，不能整层照搬 Qwen3-14B。
本模型不需要 DSV4 的压缩 KV、indexer、SWA/HCA/CSA 分支或 sink 合并。

**引擎负责决定哪些 token 被接受；算子负责正确计算这些 token 对应的状态。** Full Attention KV 可按接受长度限制可见范围，GDN 则不能把最终矩阵简单“截短”。
若验证 K 行后只提交前 A 行 AR 输入（含本轮 anchor），引擎必须取得计算到第 A 行的递推状态和卷积尾部，再继续下一轮；新采样的下轮 anchor 尚未进入这些状态。
可以保留逐位置 checkpoint，也可以正确重算接受前缀；不能使用第 K 行的状态，更不能把 draft 状态转正。
分配/回收缓存、选择 checkpoint、request ID、版本和接受长度管理属于引擎，不要求算子自行完成调度。

## 5. 在线更新是另一份反向需求

**前向推理支持，不代表在线更新已经由 PyPTO 支持。** 可以先让 PyPTO 跑 draft 前向，保留 Torch 后缀重放和优化器；发布时同步权重版本，并验证两侧前向/重放一致。
如果连更新也迁入 PyPTO，则需单列反向实现和 autograd/运行时接入：

- 最后 1 层是 Full Attention：需要其投影、Q/K norm、RoPE、Attention 和 gate 的反向；后缀内冻结的 MLP、norm、LM head 仍需要输入梯度。
- 最后 4 层包含 3 个 GDN 和 1 个 Full Attention：额外需要卷积、GDN 递推、门控和输出 norm 的反向。

还需接入词表 softmax/TV loss、梯度裁剪、优化器和 TP 梯度归并，不能只补 Attention backward。
当前训练策略在后缀入口 detach，因此不要求对前面所有层反传；AR 历史状态也作为常量。
不能根据当前前向 DSL 文件就声称这些反向已经具备，也不能据此断言 PyPTO 本身无法承载训练。

## 6. 实现顺序与验证依据

建议先固定文本、单请求、K32 的前向语义，用现有 Torch 实现作参照：先分别对齐 Full Attention 和 GDN，再接完整层和整块验证，最后处理分页直读、融合及在线反向。
算子检查应包含非零初始状态、跨页/乱序页、不同 K 和有效长度、只读历史；引擎检查需遍历接受位置，并覆盖拒绝后只剩一个普通 decode token 的状态恢复。

现有 **Torch/NPU + 原生 vLLM-Ascend** 已在真实 27B、TP2、K32、eager、同步单请求下验证起草、验证、拒绝恢复、缓存隔离、last-1/last-4 更新及 head 重载。
扩展检查的 7 个 greedy 用例共 220 token 与同后端基线相同；固定反馈的 TV loss 从 0.5937 降至 0.5094。
这提供可运行参照，**不代表 PyPTO 已验证，也不代表已有训练 head 的吞吐收益**。当前 Full Attention draft 会先 gather 分页历史再用 SDPA，GDN draft 使用 Torch 组合路径，仍有可替换的性能空间。

## 依据与代码

- [Qwen3.8-27B 官方配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)：层数、层型及 head 规格。
- [Orthrus 混合模型定义，固定版本](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/models/modeling_orthrus_qwen3_5.py)：Full Attention 可见性与 GDN 状态种子。
- [PyPTO Qwen Attention，固定版本](https://github.com/hw-native-sys/pypto-lib/blob/ab0fcd333d71701d063db4b9218ad2422f891ec4/models/qwen3_14b/paged_attention_pypto.py)、[KDA](https://github.com/hw-native-sys/pypto-lib/blob/ab0fcd333d71701d063db4b9218ad2422f891ec4/models/glm5_3_flash/kda.py)：现有实现基础，未发现完整 Qwen3.8 模型入口。
- 本仓库：[双视图执行器](../src/blockspec/parallel/qwen38.py)、[主模型历史读取](../src/blockspec/vllm_ascend/qwen38_history.py)、[TP 权重与更新](../src/blockspec/vllm_ascend/qwen38_weights.py)。
- [原生部署入口](QWEN38_VLLM.md)与[完整验证记录](QWEN38_VLLM_VALIDATION.md)。
