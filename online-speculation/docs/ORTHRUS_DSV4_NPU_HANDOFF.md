# Orthrus-DSV4 NPU 推理适配交接

**本材料把需求、公式、现有算子组合、demo 和验证证据对应起来。先不涉及训练和在线更新。**
对象是 DeepSeek-V4-Flash-0731 W8A8、Ascend A2、固定 v0.26 镜像，
不是未经核对的 DSV4.1。当前 head 是初始化版本，只用于验证接线。

## 1. 整体要实现什么

```text
已确认的 token / 主模型历史 cache
                  |
anchor + 31 个 MASK -> 共享 embedding
                  |
                  +-> 每层共享 mHC / norm
                  +-> 本层 diffusion Q/KV/O、sink（C4 层另有 diffusion indexer）
                  |      读取主模型历史 + 当前临时双向块
                  +-> 每层共享 mHC / norm / MoE
                  |   上面三步重复 43 层
                  v
             共享 LM head -> 31 个候选及其概率
                  v
       主模型因果验证 -> 接受前缀 / 拒绝补采样 / 更新主模型状态
```

共享的是参数和已计算的历史，不是复用本轮 draft 的全部激活；draft 仍执行各层计算。
draft 的临时 KV 不转成 target KV。接受后保留的是 **verifier 计算的 target KV**。
分页分配、接受/回滚和请求调度属于框架；kernel 内的 cache 写入必须另外禁用，
不能以“框架负责 cache”为由调用会覆盖历史的 AR 融合入口。

## 2. 算法与改造对应

**之前说的四个是逻辑模块：公共双向块、SWA、HCA、CSA，不是必须新写四个独立 kernel。**
建议沿 PyPTO 已有三条 Attention 路径改造，块内计算和投影作为公共逻辑。
NPU 现有算子已能组合表达这些计算；PyPTO 的任务是实现同样的输入/输出和状态语义，
再决定融合粒度，不是重新发明 Attention。

下面“已有”固定指 [pypto-lib 4c908f3](https://github.com/hw-native-sys/pypto-lib/tree/4c908f3449793a1f90e4297bd3decf6019a11fd8/models/deepseek_v4_flash_dspark)，不代表其后所有版本。

| 逻辑模块 | 已有入口及限制 | 要适配的内容 | 当前 NPU 服务实现 |
| --- | --- | --- | --- |
| 公共双向块 | `dspark_attention.py` 已让整个块互见，但先将当前块写入 DSpark cache，再统一读取 | 接收临时块 KV，不写主模型 cache；按 Orthrus 的 K 和 diffusion 权重执行，可内联到三条路径 | `_swa_block_segment` 将滑窗历史与临时块拼接，一次 FIA；只屏蔽历史 padding，不屏蔽块内未来位置 |
| SWA | `decode_sparse_attn_swa.py` / DSpark 已有滑窗读取，AR/DSpark 调用方的状态更新语义不同 | 所有 K 个 query 固定读取 anchor 前同一段滑窗，并加入整个临时块 | 上述联合 FIA，加一次 sink 归一化 |
| HCA | `decode_hca.py` 已调度 C128 压缩、cache 写回和历史 Attention；不是只读 draft 入口 | 跳过 draft 的 compressor 和写回；读取全部有效 C128 行，与滑窗/双向块联合归一化 | 兼容的大分页用分页 FIA；小分页用 sparse attention 枚举全部有效压缩行，不截断历史 |
| CSA（含 indexer） | `decode_csa.py` / `decode_indexer.py` 已有量化 indexer、Top-K、C4 读取，但包含当前行写回和逐 query 的 AR 历史边界 | 用 diffusion query/weight 投影；全部 query 的检索边界固定到 anchor 前；只读 indexer/C4 cache，再加入临时双向块 | QLI `sparse_mode=0` + 压缩行号过滤 + 原生分页 sparse attention；保留 gather + FIA 回退 |

公共前后处理沿用 RMSNorm、partial RoPE、量化矩阵乘和分组输出投影的数学，
但要绑定 diffusion 权重，并验证整块形状、TP 分片、量化 scale 和临时空间。
不能直接承诺所有非 Attention kernel 原样调用即兼容。

### 公式

为避免把量化误差混入语义定义，下面按反量化后的实数写公式。
对单个 head，$q$ 是 query，$c_j$ 是 DSV4 shared KV 行（同时作为 key/value），
$\alpha$ 是模型配置的缩放，$s$ 是该 head 的 sink logit：

$$
\mathcal A(q,C;s)=
\frac{\sum_{c_j\in C}\exp(\alpha q c_j^\top)c_j}
{\exp(s)+\sum_{c_j\in C}\exp(\alpha q c_j^\top)}.
$$

$p$ 为 AR 当前位置，$a$ 为 anchor 位置，$0\le i<K$ 为 draft 块位置。
$W(t)$ 为截至 $t$ 的滑窗 KV，$H_r(t)$ 为截至 $t$ 已完成的 ratio-$r$ 压缩行，
$B^D$ 为全部 K 行临时 draft KV，$\Vert$ 表示按行拼接。上标 AR/D 区分两套 Attention 参数。

| 路径 | AR 数学语义 | Orthrus 数学语义 |
| --- | --- | --- |
| SWA | $o_p^{AR}=\mathcal A(q_p^{AR},W(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert B^D;s^D)$ |
| HCA | $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_{128}(p);s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_{128}(a-1)\Vert B^D;s^D)$ |
| CSA | $o_p^{AR}=\mathcal A(q_p^{AR},W(p)\Vert H_4[I_p^{AR}];s^{AR})$ | $o_i^D=\mathcal A(q_i^D,W(a-1)\Vert H_4[I_i^D]\Vert B^D;s^D)$ |

CSA 中，$\mathcal P_4(t)$ 是 $H_4(t)$ 中有效压缩行的行号集合，
$k_u^I$ 是主模型 indexer cache 的历史 key，$q_{i,h}^{I,D}$ 和
$w_{i,h}^D$ 来自 diffusion indexer；$w$ 包含模型规定的缩放。$H_I$ 为 indexer head 数，
$k_I$ 为检索个数（当前 512，与块长 K 无关）：

$$
A_{i,u}^D=\sum_{h=1}^{H_I}w_{i,h}^D\operatorname{ReLU}
\big(q_{i,h}^{I,D}(k_u^I)^\top\big),\qquad
I_i^D=\operatorname{TopIdx}_{k_I}\{A_{i,u}^D:u\in\mathcal P_4(a-1)\}.
$$

AR 使用自己的 query/weight，并将候选行号范围换成 $\mathcal P_4(p)$；ReLU 和 Top-K 算法不变。
当前双向块全部可见，不参与历史 Top-K。QLI 输出已经是压缩行号，不再除以 4。

当前服务将“滑窗 + 双向块”算成一段，压缩历史算成另一段。
各段输出 $O_r$ 已在段内归一化，$L_r$ 为不含 sink 的 log-sum-exp，则：

$$
L=\operatorname{logsumexp}(s,L_1,\ldots,L_R),\qquad
O=\sum_{r=1}^{R}\exp(L_r-L)O_r.
$$

sink 只计一次，等价于 value 为零、LSE 为 $s$ 的额外段。该公式与整体 softmax 等价，
但不要求 PyPTO 分段执行；融合实现可以直接完成同一归一化。

## 3. 现成算子版本与 demo

| 材料 | 入口 | 用途及边界 |
| --- | --- | --- |
| BF16 单层 demo | [运行脚本](../scripts/demo_torch_npu_dsv4_layer.py)、[核心代码](../src/blockspec/vllm_ascend/torch_npu_dsv4_layer.py) | 真实 DSV4-Flash 结构尺寸、随机参数/模拟历史；覆盖投影、SWA/HCA/CSA、双向块、sink 和输出投影；不包含整个 backbone 和调度 |
| 当前 W8A8 Attention 组合 | [dsv4.py](../src/blockspec/vllm_ascend/dsv4.py)、[dsv4_hybrid.py](../src/blockspec/vllm_ascend/dsv4_hybrid.py) | 实际服务使用的优化路径，含分页、量化 QLI、QLI 历史边界、原生 LSE 合并 |
| 整体推理接入 | [proposer](../src/blockspec/vllm_ascend/dsv4_proposer.py)、[启动脚本](../scripts/serve_dsv4_orthrus.sh)、[固定版本补丁](../runtime/vllm_ascend_v0_26/README.md) | 构造 anchor/MASK、共享 backbone、读取 target metadata/cache、交给原生 verifier/scheduler；与 kernel 开发分工不同 |
| 算子和 engine 验证 | [验证入口](DSV4_VALIDATION_RUNBOOK.md)、[实测记录](DSV4_VALIDATION_20261028.md) | 对照公式、现有算子、真实少层 engine 的记录，不以 demo 能运行代替验收 |

单层 demo 在匹配的 Torch-NPU/CANN 环境中运行：

```bash
PYTHONPATH=src python scripts/demo_torch_npu_dsv4_layer.py --block-size 32
```

它不是纯 CPU/CUDA PyTorch demo，也不是训练 demo。BF16 演示版仍用分段 FIA + Torch LSE，
便于阅读公式；**当前 W8A8 服务已采用更好的现成算子组合**：

```python
# 每层的草稿 Attention；伪代码中的 cache 都属于主模型，作为只读输入。
q, qr, kv_block = diffusion_qkv(hidden, positions)  # 动态量化、量化 matmul、norm、RoPE
local = FIA(q, concat(read_swa(cache, anchor), kv_block), mask_padding_only=True)
segments = [local]                                # local 已包含整个双向块
if ratio == 128:
    segments += [read_all_c128_attention(q, cache)]  # 大页 FIA / 小页 sparse，无 Top-K 截断
elif ratio == 4:
    ids = QLI(diffusion_indexer(qr, hidden), cache.indexer, causal=False)
    ids = keep_completed_c4_rows(ids, anchor)      # 过滤，不对压缩行号再次除以 4
    segments += [sparse_attention(q, cache.c4, ids)]
out = npu_attention_update(segments + [zero_value_sink])  # 同一全局 softmax
return diffusion_output_projection(inverse_rope(out))   # 不提交 kv_block
```

接口至少包括 Q/临时 KV、anchor 历史长度、各 cache 的页表及长度、sink、diffusion 权重/scale，
输出是 Attention hidden states；历史和 compressor state 保持不变。当前产品契约 K=32，
换 K 还需调整 head/调度并重新编译验证，不等于重新设计数学算法。

## 4. 工作量与验收边界

**主要开发量在三条融合路径的输入、读写副作用和块形状适配，CSA 最复杂；不是从零重写所有基础算子。**
优先做 SWA/临时块与只读接口，再做 HCA 分页和归一化，最后做 CSA 的量化 query、边界和 Top-K 接线。
基础计算/Top-K 可复用，融合、流水线和临时内存优化需 PyPTO 实测后决定，暂不承诺 kernel 数或人天。

验收依次检查 FP32 参考、空历史/分页/不同 batch、只读 cache、整层投影结果，再进入完整 engine。
当前少卡已覆盖真实 A2 算子、四层 W8A8 TP2、全部接受长度、概率 q 对齐、拒绝恢复和物理页复用；
抢占恢复是受控分配失败注入，不是自然 OOM。完整 43 层 TP8 复验、graph、1M、持续负载仍未验收，
不能说“除八卡以外所有问题都排除了”。训练好 head 的接受率和推理加速收益另测。

八卡窗口的自动入口是 [watch_dsv4_tp8.py](../scripts/watch_dsv4_tp8.py)：只在八卡均无进程且显存回落后
运行完整模型 target/Orthrus API、长上下文及服务重启检查；结束清理自身容器。
此第一批自动测试不包含完整模型高接受率故障注入、graph、1M 或训练 head 的性能验收，报告始终单独标记。
