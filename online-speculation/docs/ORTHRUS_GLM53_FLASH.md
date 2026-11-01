# Orthrus GLM 5.3 Flash 推理适配

目标是让 draft 复用 GLM-5.3-Flash 的 backbone 和已提交历史，一轮提出多个候选，由原生 vLLM-Ascend 验证并决定接受前缀。当前目标权重为 `Eco-Tech/GLM-5.3-Flash-w8a8`；不涉及训练、在线更新和多模态输入。

## 架构与算法边界

该模型有 45 层，其中 34 层为 KDA 线性注意力，11 层为稀疏 MLA；另外使用 mHC 和 MoE。它不是 DSV4 的 SWA/C4/C128 结构。

- 稀疏 MLA：draft 的每行读取同一段真实历史，以及整个临时 draft 块。历史由主模型的 KPool indexer cache 检索；临时块不经过 Top-K，全部可见。
- KDA：历史是卷积状态和递推矩阵，不能通过一个 SDPA mask 变成双向 Attention。当前实验扩展采用与 Orthrus Qwen3.5 混合模型相近的方式：复制注意力参数，从真实历史快照出发，在临时状态上因果递推。

第二项是 **GLM 混合模型的实验定义**，不是原始普通 Qwen3 算法已提供的 GLM 实现。默认策略为 `experimental_causal_kda_bidirectional_mla_v1`，入口要求显式实验确认。

同一 proposer 也提供实验性的门控双向 KDA：正向读取 AR 状态，反向只遍历临时块、从零状态开始，两路在输出归一化之前融合。通过 `--kda-mode bidirectional` 选择，head 标记为 `experimental_residual_bidir_kda_bidirectional_mla_v1`。这不是 SDPA 换 mask，也不是已确认优于因果 KDA 的训练方案；方向性测试及三组小规模蒸馏对照见 [KDA 方向选择](GLM53_KDA_DIRECTION.md)。

## 算子需求

下表是需要适配的计算入口，不代表必须开发三个独立新 kernel。现有算子能组合完成时，优先组合。

| 模块 | 当前已有能力 | 适配内容 | 当前实现 |
| --- | --- | --- | --- |
| Diffusion KPool 检索 | PyPTO `models/glm5_3_flash/indexer.py` 已有投影、池级打分、Top-K、token 索引展开和未满池尾部处理；正常 AR 路径还会生成 key、压缩和写 cache。 | 接收 draft query 与 head weights，只对 anchor 前的完整池打分；展开为原始 token 位置，并追加未满池的真实历史。不要调用写 cache 的完整 AR 入口。 | `select_history` 用 `einsum → ReLU → 加权求和 → topk → 索引展开` 实现。历史池由引擎读取，不在 draft 中重新压缩。 |
| Diffusion 稀疏 MLA | PyPTO `attention.py` 已有低秩 query 吸收、稀疏历史 Attention 和输出投影；现有 decode 面向 AR 的查询与 cache 生命周期。 | 增加当前块的双向分支，固定历史边界。历史使用 AR 的 KV 展开权重，临时块使用 draft 权重，按共同 softmax 分母合并。 | 两次 NPU FIA 分别返回输出和 LSE；转换到相同 value 空间后用 `npu_attention_update` 合并。不展开整段历史的每头 K/V。 |
| KDA 临时状态执行 | PyPTO `kda.py` 已有投影、短卷积、因果递推和输出；这些计算会更新传入的状态。 | 在上述实验策略下，算子仍做因果 KDA，只把状态参数换成当前请求的临时副本。主模型状态的选取、复制和接受位置管理属于引擎适配。 | draft 使用现有 AscendC recurrent KDA。当前原生 GLM 路径最多处理 8 个位置，首轮接入采用 K=8；没有宣称 K=32 整模型已支持。 |

Embedding、mHC、norm、MoE、LM head 和 TP collective 仍调用主模型已有实现，不复制第二套 backbone。目标模型的 MTP 不作为 Orthrus draft 使用。

## 稀疏 MLA 的数学契约

设第 $i$ 个 draft query 为 $q_i$，主模型低秩历史为 $c_j^{AR}$，临时块低秩表示为 $c_t^D$。$W_K^{AR},W_V^{AR}$ 和 $W_K^D,W_V^D$ 分别表示主模型与 draft 的低秩展开权重；下式省略 head 下标。

历史 token 集合为 $\mathcal H_i$，临时块为 $\mathcal B$。正确的输出是：

$$
o_i=\frac{
\sum_{j\in\mathcal H_i} e^{q_i^\top W_K^{AR}c_j^{AR}/\sqrt d}\,W_V^{AR}c_j^{AR}
+\sum_{t\in\mathcal B}e^{q_i^\top W_K^Dc_t^D/\sqrt d}\,W_V^Dc_t^D}
{\sum_{j\in\mathcal H_i}e^{q_i^\top W_K^{AR}c_j^{AR}/\sqrt d}
+\sum_{t\in\mathcal B}e^{q_i^\top W_K^Dc_t^D/\sqrt d}}.
$$

不能将两种 latent 简单拼接后统一乘 draft 的 KV 展开权重，除非已经约束两套展开权重完全相同。当前实现对两段分别吸收 query，取得局部输出 $o_{i,r}$ 和自然对数 LSE $L_{i,r}$，再计算

$$
L_i=\operatorname{logaddexp}(L_{i,AR},L_{i,D}),\qquad
o_i=e^{L_{i,AR}-L_i}o_{i,AR}+e^{L_{i,D}-L_i}o_{i,D}.
$$

这与显式展开历史及块内 K/V 后做一次 Attention 等价。空历史只计算临时块。

## 推理引擎接入

`GLM53OrthrusProposer` 使用 `custom_class` 入口，复用同步 propose/verify 的请求记账和原生 rejection sampler。原生目标模型、验证时的 cache 写入和接受前缀处理保持不变。

每轮验证结束后，根据本轮实际接受位置读取历史：KDA 选择对应的递推状态及卷积尾部；MLA 读取有效分页 latent 和完整历史池。然后构造 `anchor + MASK`，执行 draft。整个 draft 不写目标 cache；候选的 draft 状态不会“转正”，正式状态始终由目标模型验证产生。

draft 只持有注意力投影和内部归一化等参数。执行时临时替换各层 Attention 的调用，仍使用原模型完成 embedding、mHC、MoE、norm 和输出头；`finally` 恢复 AR 调用。该方法仅用于 eager、同步、单请求实验，不用于并发或图模式。head 文件必须匹配基模配置、量化描述、TP 布局、K、mask token 和混合策略。

## 验证与启动

验证分为数学与适配器回归、单卡 NPU 算子检查、原生引擎闭环三层。90 项定向回归通过，覆盖 Attention 与显式 SDPA 等价、双向信息流、KPool 边界、接受位置的状态选取、分页寻址、head 契约和异常恢复。NPU 检查覆盖 MLA 的 K=2/8/32、KDA 的 K=1/2/8；MLA 对 FP32 的相对 RMS 误差约 0.3%，KDA 输出及逐位置状态的相对 RMS 误差不超过 7e-8。

原生引擎测试使用源权重的前 5 层，保留原始宽度、W8A8 权重、4 层 KDA、1 层 MLA、mHC，以及 dense/MoE 两类 FFN；不是随机缩小网络，也不是官方小模型。截层模型的回答没有语义评测价值，此处只检查推理执行和缓存协作。

| 实测范围 | 结果 |
| --- | --- |
| 单卡，因果 KDA + 双向 MLA | 3 条请求共 96 个 greedy token 与 target-only 完全相同；50 轮整块验证，53 次 draft 缓存审计通过。 |
| 单卡，非零门控双向 KDA + 双向 MLA | 同样 96 个 token 完全相同；53 轮验证，56 次缓存审计通过，反向 KDA 实际调用 224 次。 |
| 单卡，两种 head 保存与重启加载 | 重载后全部参数 SHA256 与导出值一致，greedy 输出仍与 target-only 一致。 |
| 双卡，双向版本及 head 重载 | 两个 rank 的验证计数一致；各 rank 的参数哈希重载一致，greedy 输出与同 TP 的原生基线一致。 |
| 单卡，4911-token 前缀 | 跨越 MLA 缓存页并超过 indexer 的 Top-K 上限；双向版本与对应原生基线输出相同，54 次缓存审计通过。 |
| 单卡，temperature=0.7 | 真实起草、原生 rejection sampling 和后续生成通过，57 次缓存审计通过；这不是采样分布等价的统计验收。 |

每轮审计逐字节比较当前请求的有效 MLA latent、indexer 池和尾部环形缓存、KDA 卷积及验证 checkpoint。上述测试实际经历了候选的部分接受、拒绝以及多条请求依次复用引擎。审计不覆盖未分配给当前请求的整个缓存池，也不代替并发隔离测试。

联调中修复了插件配置被后端严格校验拒绝、NoPE 空 RoPE 占位未识别，以及带 padding 的大缓存被整体物化等适配问题。后者已加入 NPU 内存回归：复制 512 KiB 的选中状态，额外峰值为 513 KiB。2K 测试配置曾触发原生 SFA 元数据容量检查；当前单卡使用实测通过的 34816 上限，双卡为 18432。一次 4 号卡原生 worker 初始化异常发生在请求执行前，换 3 号卡重试通过；该启动异常尚未定位，不归因于长上下文或 draft 计算。

结果、运行参数和代码哈希见 `docs/validation/glm53/native-5l-tp1/`、`native-5l-tp2/`、`native-5l-long/`、`operators-cache-20261101.json` 和 `runtime-native-20261101.json`。测试结束后相关容器已删除。

单卡算子检查：

```bash
PYTHONPATH=src python scripts/probe_glm53_ops.py --output artifacts/operators.json
```

原生闭环可用 `prepare_glm53_smoke_model.py` 生成截层权重，再用 `validate_glm53_native.py` 指定物理卡、模型和输出目录运行；它依次完成基线、两种起草、重载及采样。`--max-model-len` 和 `--long-prompt-repeats` 可指定上下文测试档位。日志和 JSON 留在输出目录，head 文件留在独立子目录，不提交权重到代码仓。

八卡完整模型测试由 `watch_glm53_validation.py` 组织：等完整权重通过逐文件 SHA256 校验、运行时通过前置检查、八卡都没有 NPU 进程且 HBM 低占用后，先运行 target-only，再运行两种初始化 draft，比较 greedy token 并检查真实 propose/verify 计数及缓存审计。前置检查记录整套插件源码哈希，源码改变后必须重新检查。每阶段有超时限制，结束后删除本实验容器释放卡；不运行空占卡任务，不停止其他人的任务。

各节点可独立运行 `watch_npu_resources.py --output /path/to/status.json`，每 30 秒记录一次空卡和心跳，发现完整八卡窗口后继续监控。它仅查询资源，不预留设备或启动模型；进程列表为空但 HBM 仍高占用的卡也不会被判为空闲。资源监控和模型测试是两件事，某个节点具备空卡后，仍须在该节点准备并校验权重及运行镜像。

**当前通过的是原始宽度、真实权重截层模型的 eager、同步、单请求、K=8 推理验收。完整 45 层 TP8 尚未验收，并发、图模式和训练/在线更新未接入。** 采样分布统计与真实训练 head 的性能还需单独评测；不能用上述初始化 head、重复文本或截层模型的接受计数推断实际 GLM 的加速收益。

参考：[GLM-5.3-Flash 官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)、[Ascend 部署说明](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.3-Flash.html)、[PyPTO GLM 模型目录](https://github.com/hw-native-sys/pypto-lib/tree/main/models/glm5_3_flash)。
