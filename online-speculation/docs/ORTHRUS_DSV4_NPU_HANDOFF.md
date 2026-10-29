# Orthrus-DSV4 NPU 推理实现与 PyPTO 算子需求

## 1. 目标与整体结构

**目标：在不改变 Orthrus 算法的前提下，为 DSV4 实现双向 draft Attention，并接入 NPU 推理引擎。**
本报告覆盖 DeepSeek-V4-Flash-0731、Ascend A2、W8A8 推理及 BF16 单层演示，不涉及训练和在线更新。
Torch-NPU 服务基于固定 v0.26 镜像；PyPTO 能力对照固定到
[pypto-lib 4c908f3](https://github.com/hw-native-sys/pypto-lib/tree/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark)。

每轮 draft 输入由一个已知的 anchor token 和 31 个 MASK token 组成，块长 $K=32$。
MASK 是待预测位置的占位符；块内所有位置互相可见，同时读取 anchor 之前的主模型历史。

```text
anchor + MASK × 31
        |
共享 embedding
        |
43 层 block：
  共享 mHC / norm
  -> 本层 diffusion Attention（SWA、HCA 或 CSA）
  -> 共享 mHC / norm / MoE
        |
共享 LM head -> 31 个候选及其概率
        |
主模型因果验证 -> 接受前缀 / 拒绝补采样 -> 提交主模型状态
```

diffusion 分支持有独立的 Attention 投影、内部 norm、sink，以及 CSA 的 indexer query/weight 参数；
embedding、block 外层 norm、mHC、MoE 和 LM head 与主模型共享。
共享参数不等于共享本轮激活，draft 仍需执行这些计算。

## 2. 按计算模块划分的开发需求

**计算划分为公共双向块、SWA、HCA、CSA 四个模块；物理 kernel 数由融合方式决定。**
现成 Torch-NPU 算子能够组合表达这些计算。PyPTO 需要适配整块 query、历史可见范围、
状态只读和联合归一化，而非另造一套 Attention 数学。

下文公式按单个 head、反量化后的实数表示。$q$ 为 query，$C$ 为可见 KV 行集合，$c_j$ 为 DSV4 shared KV 行
（同时作为 key 和 value），$\alpha$ 为模型配置的缩放，$s$ 为 sink logit：

$$
\mathcal A(q,C;s)=
\frac{\sum_{c_j\in C}\exp(\alpha q c_j^\top)c_j}
{\exp(s)+\sum_{c_j\in C}\exp(\alpha q c_j^\top)}.
$$

上标 $AR$ / $D$ 分别表示主模型与 diffusion 分支的参数。各模块的其他符号在对应公式旁定义。
以下代码块是**调用顺序伪代码**，实际参数与布局以链接中的实现为准。

### 2.1 公共双向块与归一化

**作用与算法。** 让全部 $K$ 个 draft 位置互相读取中间特征，而不依赖这些位置先生成 token。
$B^D=\{c_0^D,\ldots,c_{K-1}^D\}$ 是临时块 KV，$q_i^D$ 是块内第 $i$ 个 query。
块内 additive mask 恒为零，没有因果三角遮挡：

$$
M_{ij}^{D}=0\quad(0\le i,j<K),\qquad
O_{B,i}=\mathcal A(q_i^D,B^D;-\infty),\qquad
L_{B,i}=\log\sum_{j=0}^{K-1}\exp(\alpha q_i^D(c_j^D)^\top).
$$

$O_{B,i}$ 是块内归一化输出，$L_{B,i}$ 是不含 sink 的 log-sum-exp。
分段计算时，$O_r$ 是段内归一化输出，$L_r$ 是该段不含 sink 的 LSE；各段必须按下式合并，而不是直接相加：

$$
L=\operatorname{logsumexp}(s,L_1,\ldots,L_R),\qquad
\operatorname{Merge}_s\{(O_r,L_r)\}_{r=1}^{R}
=\sum_{r=1}^{R}\exp(L_r-L)O_r.
$$

$R$ 为分段数；sink 等价于 value 为零、LSE 为 $s$ 的额外段，只计一次。
空历史段使用 $O_r=0,L_r=-\infty$。融合 kernel 可直接计算同一个全局 softmax，无需显式输出分段 LSE。

**现成算子组合。**

```python
# BF16 单层 demo：临时块独立计算，历史计算结果随后一并参与归一化。
block_output, block_lse = FIA(query, kv_block, causal=False, return_lse=True)
output = torch_lse_merge(history_segments + [(block_output, block_lse)], sink)

# W8A8 服务：块内计算已合入第 2.2 节的滑窗 FIA，不再重复计算。
# 最终用 npu_attention_update 合并滑窗/块、压缩历史及零 value 的 sink 段。
```

FIA 指 `torch_npu.npu_fused_infer_attention_score`。实现见
[demo 的 _native_fia_segment / forward](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py)、
[服务的 merge_attention_outputs](../src/blockspec/vllm_ascend/dsv4_hybrid.py)。

**PyPTO 已有能力与开发内容。**
[dspark_attention.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/dspark_attention.py)
已有整块双向可见和 softmax 归一化，但入口先把当前块写入 DSpark 私有 cache，再读取历史与块。
Orthrus 需要改为接收临时块 KV、使用 diffusion 参数、支持 $K=32$，不写主模型 cache。
块内计算可以内联到三条历史路径，不要求新增独立 kernel。

### 2.2 SWA：滑窗历史与双向块

**作用与算法。** 让所有 draft query 读取同一段 anchor 前的滑窗历史及整个临时块。
$p$ 是 AR 当前位置，$a$ 是 anchor 位置；$W(t)$ 是截至 $t$ 的滑窗 KV，$\Vert$ 表示按行拼接。

| AR Attention | Orthrus-SWA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert B^D;s^D)$ |

历史边界固定为 $a-1$，不随块内 query 的位置右移；块内全部 $K$ 行均可见。

**现成算子组合。**

```python
# 通过页表读取主模型滑窗；每个请求的 K 个 query 共用同一历史范围。
history = gather_paged_cache(swa_cache, block_table, window_indices(anchor))
kv = torch.cat([history, kv_block], dim=1)
# 只屏蔽历史 padding，不能遮挡块内未来位置。
output, lse = FIA(query, kv, mask=history_padding_mask, return_lse=True)
# 纯 SWA 层再与 sink 合并；HCA/CSA 层还需加入对应压缩历史段。
```

W8A8 服务使用一次联合 FIA；BF16 demo 分别计算历史和块内 FIA，再按第 2.1 节合并。
代码见 [服务 _swa_block_segment / _history_segments](../src/blockspec/vllm_ascend/dsv4.py)。

**PyPTO 已有能力与开发内容。**
[decode_sparse_attn_swa.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py)
提供滑窗 Attention，DSpark 路径提供整块可见的参考。
适配时将输入接到主模型只读 SWA cache，统一各 query 的历史边界，加入临时块并按实际长度屏蔽 padding；
不能沿用 AR 的当前行写回或逐行因果追加逻辑。

### 2.3 HCA：C128 压缩历史

**作用与算法。** 在滑窗与双向块之外，读取全部有效的 C128 压缩历史。
$H_{128}(t)$ 表示截至 $t$ 已完成的 ratio-128 压缩 KV 行。

| AR Attention | Orthrus-HCA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_{128}(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_{128}(a-1)\Vert B^D;s^D)$ |

**现成算子组合。**

```python
local = swa_and_block_FIA(query, swa_cache, kv_block, anchor)  # 第 2.2 节
if page_layout_supported_by_FIA:
    compressed = paged_FIA(query, c128_cache, block_table, valid_lengths)
else:
    # sparse 仅作为分页读取接口：枚举全部有效行，不做 Top-K 截断。
    compressed = npu_sparse_flash_attention(query, c128_cache, all_valid_row_ids)
output = merge_with_sink([local, compressed], sink)          # 第 2.1 节
```

W8A8 服务按页大小选择分页 FIA 或 `torch_npu.npu_sparse_flash_attention`；
兼容回退为 `gather_paged_cache + FIA`。BF16 demo 对模拟的连续 C128 历史直接调用 FIA。
代码见 [服务 _c128_segment / _sparse_history_segment](../src/blockspec/vllm_ascend/dsv4.py)。

**PyPTO 已有能力与开发内容。**
[decode_hca.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_hca.py)
和 [decode_sparse_attn_hca.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)
已有局部/压缩历史读取与合并，但 AR 编排包含 compressor 和 cache 更新。
适配需使用 diffusion query、跳过 draft 的压缩与写回、加入全可见临时块，并统一三部分与 sink 的归一化。
目标模型的 C128 压缩算法保持不变。

### 2.4 CSA：C4 Indexer 与稀疏 Attention

**作用与算法。** 每个 draft query 用自己的 indexer query 从 anchor 前的 C4 历史选出相关行，
再与滑窗、整个双向块共同计算 Attention。当前块不参与 Top-K。

$k_u^I$ 是主模型 indexer cache 的历史 key，$q_{i,h}^{I,D}$ 和 $w_{i,h}^D$ 来自 diffusion indexer；
$w$ 包含模型规定的缩放，$H_I$ 是 indexer head 数。
$\mathcal P_4(t)$ 是截至 $t$ 的有效 C4 行号集合，$k_I=512$ 为检索个数，与块长 $K$ 不同：

$$
A_{i,u}^{D}=\sum_{h=1}^{H_I}w_{i,h}^{D}
\operatorname{ReLU}\!\left(q_{i,h}^{I,D}(k_u^I)^\top\right),\qquad
I_i^{D}=\operatorname{TopIdx}_{k_I}
\{A_{i,u}^{D}:u\in\mathcal P_4(a-1)\}.
$$

AR 使用自己的 indexer query/weight，并将检索范围换成 $\mathcal P_4(p)$。
$H_4[I]$ 表示按行号集合 $I$ 取出的 C4 KV：

| AR Attention | Orthrus-CSA |
| --- | --- |
| $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_4[I_p^{AR}];s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_4[I_i^D]\Vert B^D;s^D)$ |

**现成算子组合。**

```python
index_query, weights = diffusion_indexer_projection(hidden, query_low_rank)
# W8A8：QLI = npu_vllm_quant_lightning_indexer，配套非因果 metadata。
ids = QLI(index_query, target_indexer_cache, weights, sparse_mode=0)
ids = keep_completed_c4_rows(ids, anchor)  # QLI 已返回压缩行号，不再次除以 4
compressed = npu_sparse_flash_attention(query, target_c4_cache, ids)
local = swa_and_block_FIA(query, swa_cache, kv_block, anchor)
output = merge_with_sink([local, compressed], sink)
```

BF16 demo 使用 `npu_lightning_indexer + torch.gather + FIA + Torch LSE`；
W8A8 服务使用量化 QLI、原生分页 sparse Attention 和 `npu_attention_update`，保留 gather 回退。
代码见 [服务 _c4_topk_readonly / _c4_segment](../src/blockspec/vllm_ascend/dsv4.py)、
[demo _native_indexer / _gather_c4](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py)。

**PyPTO 已有能力与开发内容。**
[decode_indexer.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_indexer.py)
已有量化评分/Top-K；
[decode_csa.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_csa.py)
和 [decode_sparse_attn_csa.py](https://github.com/hw-native-sys/pypto-lib/blob/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py)
已有 C4 读取及稀疏 Attention。现有 AR 路径同时含写回和逐 query 的因果边界。
适配需绑定 diffusion indexer 权重、生成 K 行查询、把检索边界固定到 anchor 前，
只读 indexer/C4 cache，并加入公共双向块。评分和 Top-K 数学不变，量化 scale、行号和有效长度需逐项对齐。

## 3. 公共计算与框架分工

| 部分 | 现成实现与适配要求 |
| --- | --- |
| diffusion Q/KV 前处理 | 服务使用动态量化、量化 matmul、RMSNorm 和 partial RoPE；入口为 `dsv4.py::_qkv_readonly`。PyPTO 可复用 `qkv_proj_rope.py` 的基础计算，适配独立权重、整块形状、TP 分片及量化 scale |
| diffusion 输出投影 | 逆 partial RoPE、分组低秩输出投影；服务调用原生 DSV4 `_forward_o_proj`，demo 使用 `npu_transpose_batchmatmul` 和 Linear。PyPTO 对照 `decode_o_proj.py` |
| 共享 backbone | embedding、mHC、外层 norm、MoE、LM head 复用目标参数；draft 的 K 行仍要执行计算，需验证对应 shape/TP 路径 |
| 引擎接入 | proposer 构造 anchor/MASK、绑定 head 和主模型 cache、传递候选概率；verifier/scheduler 负责因果验证、提交接受前缀、拒绝回滚及分页管理 |

所有 Attention 路径都须满足：**draft 运行前后，目标 KV、indexer cache 和 compressor state 不变。**
验证后保留的是主模型重新计算的 target KV，临时 draft KV 不转正。
kernel 内禁写与框架的分页/回滚管理分别负责不同层面，二者都需要实现。

主要开发量在三条 Attention 路径的输入和状态语义适配，CSA 还涉及量化 indexer 接线。
基础 matmul、norm、RoPE、Top-K 和输出投影不需要改变数学，但形状与调用兼容性仍需验证。
融合粒度、临时空间和流水线优化应基于实测确定。

## 4. Demo 与完整推理入口

| 交付项 | 代码入口 | 覆盖范围 |
| --- | --- | --- |
| BF16 单层 demo | [运行脚本](../scripts/demo_torch_npu_dsv4_layer.py)、[核心代码](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py) | DSV4-Flash 实际结构尺寸、随机权重和模拟历史；完整的 Attention 输入投影、三条路径、双向块、sink、输出投影 |
| W8A8 服务算子组合 | [dsv4.py](../src/blockspec/vllm_ascend/dsv4.py)、[dsv4_hybrid.py](../src/blockspec/vllm_ascend/dsv4_hybrid.py) | 第 2 节的分页、量化 QLI、联合 FIA、原生 LSE 合并以及共享 backbone |
| 完整推理接入 | [proposer](../src/blockspec/vllm_ascend/dsv4_proposer.py)、[固定版本补丁](../runtime/vllm_ascend_v0_26/README.md)、[部署说明](DSV4_DEPLOYMENT.md) | head 加载、运行时上下文、真实 cache、候选/概率传递及 OpenAI API 服务 |
| 验证 | [验证入口](DSV4_VALIDATION_RUNBOOK.md)、[实测记录](DSV4_VALIDATION_20261028.md) | 算子参考对照、cache 不变、采样、真实 engine 状态与完整模型验收项 |

在匹配的 Torch-NPU/CANN 环境、仓库的 `online-speculation` 目录下运行单层 demo：

```bash
PYTHONPATH=src python scripts/demo_torch_npu_dsv4_layer.py --block-size 32
```

该 demo 不依赖 PyPTO 或 vLLM metadata，但依赖 Torch-NPU；它不是纯 CPU/CUDA PyTorch 示例，
也不包含整个 backbone、verifier 和 scheduler。完整服务入口见部署说明。
当前服务契约 K=32；改变 K 需同时调整 head/调度配置并重新编译验证，而非重写 Attention 数学。

## 5. 验证状态与验收要求

已取得的证据包括真实 A2 算子对照、四层 W8A8 TP2 运行、全部候选接受长度、
概率传递、拒绝恢复及物理页复用；抢占恢复由受控分配失败触发。
初始化 head 用于验证推理接线，不代表已获得训练 head 的接受率或加速收益。

完整 43 层 TP8、graph、1M 上下文和持续负载的验收尚未完成。
PyPTO 实现应依次通过：逐模块 FP32 参考与现成算子对照、分页/空历史/不同 batch、
目标 cache 不变、整层输出对照，以及完整引擎的接受/拒绝状态检查。
