# GLM-5.3-Flash 推理算子复核

日期：2026-11-03。仅检查推理，不推进训练与在线更新，不开发新 kernel。

## 推理接入

主模型继续走原生 vLLM-Ascend GLM 路径。插件通过 custom proposer 调用独立 draft 投影，共享 embedding、mHC、norm、MoE 和 LM head。MLA 读取已提交的 latent/indexer 池，KDA 读取接受位置的卷积尾部和递推矩阵；这些历史先形成请求自己的快照，draft 只修改临时状态。

共享历史并不等于零拷贝。`NativeGLM53History` 复制有效前缀/状态，`select_history` 用 Torch 组合打分、选池，再收集各 query 选中的 latent。分页快照与选中行收集已分别计时并调整调用方式；仍有随前缀长度增长的复制，不能将局部改进等同于零拷贝或全场景最优。

MLA 保持块内双向；KDA 保留因果、门控反向块两种显式实验策略，分别加载匹配 head。没有因算子优化改动其数学定义。完整原生 KDA 入口仍限定 K≤8；单独 MLA 的 K=32 测试不等于放开模型限制。

## 本轮调用选择

| 部分 | 当前实现及选择依据 |
| --- | --- |
| 共享主干及投影 | 保留原生量化线性、mHC、MoE、norm 等模块，不替换主模型计算。 |
| 分页历史快照 | 至少 16 个有效页、规则 stride 且存储边界允许时，建立包含行末 padding 的连续只读视图，再批量选行；短前缀和不兼容布局仍逐页复制。 |
| 选中 latent 收集 | 连续请求快照、大索引集用展平索引的 `index_select`，再恢复 `[K,S,512]`；小索引集和非连续输入沿用普通索引。 |
| MLA 选中历史 | 每行使用自己的索引，通过 FIA 返回 latent 输出和 LSE；AR 的 K/V 展开参数用于解释 AR 历史，draft 参数只解释临时块。 |
| MLA 临时块 | 从逐 query 展开 KV 改为一份临时块、K 行 query 的 FIA；已测局部耗时约 220–259 → 168–175 μs。 |
| 联合 softmax | 各段先经过自己的 value 展开，再用 `npu_attention_update` 合并，不把两个归一化结果直接相加。 |
| KDA 递推 | 继续使用目标 GLM 的 `run_recurrent_kda`，只传独立临时状态槽。 |
| KDA 卷积 | 接入同版 GLM 原生 `causal_conv1d`；实测 32/64 heads 短块更快，8 heads 更慢。默认仅在 BF16、K≤8、32/64 heads 时采用融合路径，其余保留 `conv1d + SiLU`。 |

卷积比较包含临时状态、非连续输入、FP32 参数到激活精度的转换与元数据构造：K=8 时，32 heads 约 618 → 568 μs，64 heads 约 1067 → 560 μs；8 heads 约 264 → 566 μs，后者没有设为默认。

**少复制临时 KV 不等于总 workspace 更小。** 例如 K=8/8-head 的 FIA 调用峰值约 18.3 → 27.8 MB，64-head 约 28.3 → 105.4 MB；更快的卷积同样增加临时 workspace。它们是短暂峰值，不是持久缓存增长，仍需完整模型评估余量。计时基于合成输入，只能说明局部选择，不能推出模型吞吐提升。

## 分页读取与检索

逐页 `clone` 的成本主要来自大量小调用；不能直接对非连续目标 cache 做 `index_select`，因为 NPU 可能先物化整个 arena。新的批量路径把物理行 pitch 包含在连续视图内，按页表映射生成有效行号，只收集本请求前缀，再去掉 padding。视图不复制存储，收集结果则拥有独立存储；检查失败时回到逐页路径。

合成 arena 使用 512 个物理页、每页 128 行，算子页长 16；MLA 行为 512 维、物理 pitch 576，池 key 为连续 128 维。相同输入、同版镜像内的结果如下，所有快照和收集输出均逐元素相同：

| 操作 | 输入规模 | 原调用 | 当前调用 | 峰值临时分配 |
| --- | --- | --- | --- | --- |
| MLA 历史快照 | 32771 行 | 89.8 ms | 3.00 ms | 134.8 → 71.6 MB |
| 池 key 快照 | 32771 行 | 133.4 ms | 2.84 ms | 34.5 → 8.7 MB |
| MLA 历史快照 | 2055 行 | 5.04 ms | 0.53 ms | 21.1 → 4.5 MB |
| 每行选中 latent | K=8、每行 2051 个索引 | 132 μs | 49–51 μs | 33.6 → 16.8 MB |

这里的池 key 32771 行是独立算子压力用例，不代表本轮模型运行了四倍于此的 token 上下文。7/65/129 行的短前缀仍使用逐页路径；257 行及以上的已测布局采用批量读取。小索引集也保留原调用，没有为了使用融合/批量接口统一替换所有形状。

KPool 检索总计约 0.52–0.60 ms，其中 FP32 打分约 0.25–0.29 ms、Top-K 约 0.03–0.04 ms，其余包括类型转换和 token 索引展开。本轮不修改检索公式、分数精度或 Top-K 选择；优先处理了开销更大的复制与收集。上述为局部算子比较，不是模型吞吐收益。

代码和结果见 `scripts/benchmark_glm53_history_paths.py` 与 `validation/glm53/history-20261103/`。新增测试覆盖子页、行/页 padding、非零 storage offset、存储不足时的回退、非法页号、重复索引、输出与目标存储隔离，以及峰值分配不随整个 arena 大小增长。

## 环境与验证

GLM 固定镜像：

```text
quay.io/ascend/vllm-ascend@sha256:61584d790fa3668e7c43f0eb50a3e3a86d94b3eed15bce5ac82391ed383df229
```

对应 vLLM/vLLM-Ascend 0.30、torch-npu 2.10.0.post7、CANN 9.2.1。DSV4 的 0.26 镜像没有相同 KDA/卷积接口，不能混用。111 的 HCCL 测试需显式选择节点网卡；这些环境值由脚本参数/环境传递，不写死在算子里。

单卡检查包含：不同 AR/draft 展开权重的 MLA、空/短/超过 512 个池的历史、两种 KDA 策略、非零反向门、K=1/2/8/32 的独立块计算、卷积尾部和目标状态不变。原始结果与完整模型以外的复验记录归档于 `validation/glm53/ops-20261103/`。

全部 CPU 回归为 **646 passed、57 skipped**；单卡读取、收集、完整 Attention demo 及卷积专项共 **66 passed**，其中 31 项为 NPU 专项、35 项为通用复验。另有 14 组快照对照和 4 组检索分项计时，不与历史用例重复相加。卷积覆盖 BF16/FP32 参数，以及真实融合投影中 QKV 切片的非连续行 stride；实际模型中的参数精度转换按原生 GLM 路径处理。

```bash
# 同版镜像内运行；源目录为 online-speculation。
GLM53_DEMO_NPU=1 ORTHRUS_RUN_NPU_TESTS=1 PYTHONPATH=src \
  python -m pytest -q tests/test_glm53_orthrus_demo.py tests/test_glm53_ops_audit.py
python scripts/run_in_runtime.py scripts/benchmark_glm53_attention_paths.py --output /tmp/glm53-paths.json
ORTHRUS_RUN_NPU_TESTS=1 PYTHONPATH=src python -m pytest -q tests/test_glm53_history_read.py
python scripts/run_in_runtime.py scripts/benchmark_glm53_history_paths.py --output /tmp/glm53-history.json
```

完整 45 层 TP8 尚未通过全部验收。真实源权重截层、单卡算子及初始化 head 只能验证所覆盖的链路，不能代替正式 head 的效果或整模型生产验收。PyPTO 当前能力、算法及开发接口见 [需求报告](ORTHRUS_GLM53_PYPTO_REPORT.md)。

## 本轮原生模型复验

使用真实源权重的前五层，保留原始宽度、W8A8、4 层 KDA 和 1 层 MLA。配置为单卡 TP1、K=8、eager、同步单请求，最大上下文 34816。每阶段运行相同三个请求，各输出 32 token；其中长输入由 320 次重复构造，按模型 tokenizer 计 2251 token，超过 512 个完整池，覆盖真实 Top-K 筛选和新的大索引集读取。不是完整 45 层模型，也不是 32K 模型上下文测试。

| 阶段 | 实测结果 |
| --- | --- |
| Target-only | 生成本轮固定基线。 |
| 因果 / 门控双向 draft | 两者各 96 个 greedy token 与基线逐个一致；分别完成 52/62 次 draft/cache 审计。双向门为 0.25，实际执行反向 KDA。 |
| 两种 head 独立进程重载 | 重载哈希与各自导出一致，重新生成的 greedy token 与基线一致；不是同一进程沿用内存里的 head。 |
| 两种策略 temperature=0.7 | 各完成 96-token 采样、真实 propose/verify 和缓存审计；这是功能测试，不是分布等价验收。 |

七个阶段全部通过，六个 draft 阶段累计 **330 次目标缓存审计**通过，三个双向阶段共执行 716 次反向 KDA。各报告的推理代码哈希均与 `ded51dd` 提交一致；未提交模型/head 权重。结果、环境锁、源码哈希索引和压缩日志见 `validation/glm53/history-20261103/`，原生结果在其 `native-5l/` 子目录。

这些结果不替代九层四卡和完整 45 层的严格验收，也不包含并发或图模式。本轮测试容器已删除，设备锁已释放。

三台节点的完整八卡待验队列已更新为同一份 `ded51dd` 推理源码。队列先等待八卡空闲，执行同版镜像、驱动、worker 和 HCCL 预检，通过后才运行完整模型的七阶段验证；没有沿用旧源码的运行时哈希。当前仍在等待八卡，不能把队列已配置算作八卡验收通过。
