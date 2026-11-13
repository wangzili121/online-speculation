# Orthrus 接入 DeepSeek V4.1 Flash 的实施计划

首阶段以 **DeepSeek-V4.1-Flash-1017、Ascend、纯文本推理**为目标。先用 vLLM-Ascend 跑出可复现的原生主模型基线，同时用少卡验证 Orthrus 特有的 Attention 和缓存协作。初始化 head 用于验证接入，训练 head 用于检验接受长度和加速收益。离线训练、在线更新、视觉输入、PD 和图模式各自安排后续验收。

## 模型变化与 draft 设计

常博的实现中，draft 以一个已经确定的 anchor 加若干 MASK 构成预测块，共用主模型的真实历史 KV 和主干参数。每层独立的 diffusion Attention 一次给出整块候选，AR 主模型验证后提交接受前缀。首版沿用 K=32，也就是一个 anchor 和 31 个候选位置。K 记入 head 契约，实际执行宽度受请求剩余预算限制。 [online-speculation 主线](https://github.com/xchang1121/inference_scaling/tree/main/online-speculation) · [Orthrus 论文](https://arxiv.org/abs/2605.12825)

V4.1-1017 的 40 层文本主干分为 20 层因果 encoder 和 20 层 decoder。下面是从官方配置推导出的层角色，层号从 0 开始。实现从所选 checkpoint 的配置生成层图。 [官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) · [V4.1 技术报告 §2.2–2.4](https://arxiv.org/html/2609.19969v1)

| 层角色 | 层号和依赖 | Orthrus 的计算 |
|---|---|---|
| SWA-only | 0、1 | 读取本层滑窗历史，加入本轮临时块 KV。块内双向可见。 |
| C2A | 2–19。2、8、14 是 Full 层，其余层沿用同组选择 | Full 层用 draft indexer Q 对已提交的源 indexer K 计算 Top-K。后续层复用本轮 draft 的选择，并继续读取各自的滑窗。单独对照奇偶边界和运行时传入的残余状态。 |
| C1A | 20–39。20 是 Full 层，24、28、32、36 是 Reindex 层，其余是 Reuse 层 | 全部读取源层 20 的目标全局 KV/indexer K。20 为每个 draft query 生成候选池和 Top-K。Reindex 层在对应候选池内重排。Reuse 层沿用最近一次 draft Top-K。 |

每层的可见范围由本层 SWA 历史、该层选中的全局历史以及本轮临时块组成。历史读取以上一个已提交位置为界，临时块按自身位置应用 RoPE。全局和滑窗按原生 Attention 的输入约定保留各自的行，Top-K 和候选池使用 draft 私有缓冲。CED 全局 KV 由目标 encoder 边界的状态构建，draft 使用已发布的源层缓存。首个 FP32 Torch golden 一次完成共同的 softmax。分段实现时按 LSE 合并，并在归一化中只计一次 sink。 [官方模型代码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py) · [Ascend V4.1 Attention](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)

主干继续共享 embedding、Single-Pass mHC、外层 norm、MoE、Engram 和 LM head 的权重。每轮 draft 单独传递 mHC 的跨层 `pre_mix`。Engram 需要为 draft 构造独立的 n-gram 输入，第一版拟用已提交 token 加 anchor/MASK 序列，并把同一约定写入 head 训练契约。还要核对输入 ID 对 MoE 路由的影响。目标侧的 Engram lookback、SWA、全局 KV、indexer K、压缩残余和跨层选择缓冲都纳入状态检查。 [Ascend V4.1 模型](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/model.py) · [Ascend 缓存定义](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/cache_config.py)

## 接入路径

- **vLLM-Ascend 负责首个完整推理版本。** 以官方 V4.1 支持栈为基线，固定实际使用的 vLLM、vLLM-Ascend、镜像 digest、torch_npu、CANN 和驱动。先在相同版本、拓扑和 KV 配置上跑 target-only eager。最新版 A2 指南采用 MRV2，v0.30.0 归档中的部分配置仍用 MRV1，启动参数按所选版本配套。 [当前部署指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) · [v0.30.0 指南](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html)
- **Orthrus 放在独立插件，先实测注册。** 用最小候选验证注册配置到 Ascend MRV2 worker 的调用链，再确认 model load、cache bind、propose、draft logits 传递和校正入口。上游较新的 `SpeculatorConfig` 支持自定义类，已查看的 Ascend MRV2 工厂仍按内置方法分派。若固定版本缺入口，优先在 Ascend 加薄的注册和生命周期适配。只有上游配置或 verifier 接口也缺能力时才补 vLLM。旧 MRV1 `custom_class` 补丁作为历史参考。 [vLLM 扩展接口](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/spec_decode/speculator_config/) · [Ascend MRV2 工厂](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/__init__.py)
- **PyPTO 并行梳理算子和接口。** PyPTO Lib 已有 V4.1 算子目录，公开 Serving 的 V4.1 工作以 A5 为主。本地检视的快照也有 A5 executor 和真实模型 smoke 入口。先用同一 Torch golden 定义两处重点适配，分别是从只读源页做 K 行 Full/Reindex/Reuse 检索，以及让 SWA、选中全局历史和临时双向块共同归一化。现有 Q/KV/O 投影、RoPE、mHC、MoE 和 Engram 优先验证组合入口的兼容性。按实际 cache ABI、批次和 K 行支持判断复用、薄适配或新增融合。 [PyPTO Lib V4.1](https://www.pypto.ai/pypto-lib/models/deepseek_v4_1_flash/) · [PyPTO Serving V4.1 跟踪](https://github.com/hw-native-sys/pypto-serving/issues/240) · [Golden Harness](https://www.pypto.ai/pypto-lib/run-and-validate/golden-harness/)

## 实施和验收

| 阶段 | 要做的工作 | 完成标志 |
|---|---|---|
| 1. 冻结参考与 head 契约 | 以 1017 配置和权重索引生成 40 层角色表。记录基模哈希、tokenizer、MASK、K、源层映射、Engram 策略和张量布局。head 包含各层 diffusion Attention 权重和 8 个产出索引层的 query/打分权重。记录 BF16 交换格式和独立校准的部署格式。 | preflight 能拒绝旧 V4 head、错版本、错源层和错误分片。AR-copy 初始化和训练产物有明确标记。 |
| 2. CPU 与少卡算子验证 | 先写 Torch golden 和小拓扑样例，覆盖 SWA、C2A Full/Reuse、C1A Full/Reindex/Reuse。用已有单层权重样本核对真实权重与算子形状。逐个尝试读取原生分页 cache 的算子路径，实测可见范围、长度、缓存布局和 LSE。 | 正确覆盖奇偶、跨页、空历史、Top-K 候选池截断和请求换序。draft 前后目标缓存及状态逐项一致。 |
| 3. MRV2 推理闭环 | 在小拓扑上接通真实 proposer 和原生 verifier。先做 greedy，再校验随机采样时的 draft 分布传递和标准 rejection sampling。用受控候选覆盖接受 0、1、跨页和全接受，同时覆盖 EOS、取消、并发、页复用和 C2A 残余。 | 与同版本、同输入、同调度的原生参考对照通过。报告分别记录目标原生单步与多行验证的数值差异、Orthrus 状态差异和采样分布。 |
| 4. 全模型与性能 | 在官方拓扑上先验收 target-only，再跑初始化 Orthrus。记录 API、greedy、采样、连续请求和缓存状态。取得训练 head 后用同一请求集比较 target-only、原生 DSpark 和 Orthrus，并分解 draft、检索、verify、回放和显存开销。 | 初始化 head 的功能结果和训练 head 的性能结果各有独立证据。记录 TTFT、TPOT、吞吐、goodput、接受长度和内存。 |

阶段 2 要验证的第一个算子接口是 **C2A Full 加一个 Reuse 层**。用同一组目标源页跑两种逻辑相同的物理布局，先验证 Top-K 逻辑索引、C2A 奇偶边界和 sink，再接临时双向块。旧 V4 已有 proposer、概率传递、分页与回滚测试可沿用。旧 V4 的 C4 曾独立复现物理页顺序引起的精度差异，这项测试给 V4.1 提供回归场景。 [旧版实现和验证记录](DSV4_PAUSE_HANDOFF.md)

接下来核对原生 `select_projected` 能否独立处理 draft 的 K 行查询和候选池，以及 `npu_sparse_flash_mla` 能否同时接收 anchor 处的固定历史边界和临时块。当前原生路径读取分组历史，并关闭 LSE 返回。若需要分段计算，先确认设备是否支持返回合并所需的归一化统计量，再决定使用原生算子组合或补充融合入口。A2/A3 和 A5 分别核对实际缓存格式。 [Ascend 检索与 Attention 入口](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py#L528-L641) · [Ascend 缓存 ABI](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/cache_config.py)

## 资源与待核实项

A2 官方 W4A8 colocated 示例是两台八卡、DP2/TP8/EP16。另有单台 8×910B3 的 W4A8 推理报告，使用 TP8 + EP，并有 1017 的 Engram CPU offload 启动配置可参考。单机开发优先复现该配置的纯文本 target-only eager，并记录各 rank 显存峰值和 Engram 主存占用。双节点作为官方拓扑的完整验收路径。ModelScope 页面显示 W4A8 包约 529 GB，W8A8 包约 781 GB。显存预算按实际分片、Engram offload、专家布置、KV 和 draft head 分开测算。 [Ascend 部署拓扑](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) · [单机 8×910B3 报告](https://github.com/vllm-project/vllm-ascend/issues/21392) · [1017 单机配置](https://github.com/vllm-project/vllm-ascend/issues/20600) · [W4A8 权重](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8) · [W8A8 权重](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8)

## 单机八卡的 draft 训练

**先验证 W4A8 主模型作老师、BF16 draft 后段作学生。** 以部署用的 W4A8 主模型执行无梯度因果前向，采集监督位置的全词表 logits 和 anchor 之前的只读历史。用初始化后的双向 draft 前段生成后段输入。离线训练首选 forward KL。在线续训可按原实现用已验证的有效行和 TV。首个实验先训第 39 层，随后扩到第 36–39 层，推理仍执行 40 层双向 draft。 [Orthrus 蒸馏](https://arxiv.org/html/2605.12825v1#S3.SS2) · [常博的续训实现](https://github.com/xchang1121/inference_scaling/blob/main/online-speculation/src/blockspec/parallel/online.py)

后段重放需要保存整个 draft 块进入后段时的 residual 和 `pre_mix`，以及对应的历史 KV、源层页表和检索状态。第 39 层复用第 36 层的 draft Top-K。训练第 36–39 层时，第 36 层需要从第 20 层的 draft 候选池重做 Reindex。第一轮固定 indexer 参数，先验 diffusion Attention 的反向。要优化离散 Top-K 的检索效果时，另加可微的索引打分蒸馏，在同一 anchor 历史与候选域下对齐教师。语言模型 KL 单独使用时，梯度无法通过离散的索引选择直接更新打分权重。 [V4.1 分层检索](https://arxiv.org/html/2609.19969v1#S2.SS3) · [NeMo 的 V4.1 训练边界](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/large-language-models/deepseek-ai/DeepSeek-V4.1-Flash)

AR 教师前向可全程无梯度。draft 后段中的共享 MoE、mHC、norm 和 LM head 权重保持冻结，计算仍要能把梯度传到前面的 draft Attention。首个反向验证可以用从量化权重逐层反量化的 BF16 模块和可微 Attention 组合，并比较推理时实际使用的 W4A8 路径。W4A8 的激活量化和融合舍入需要单独做同输入对照。若同驻显存不足，可先采集后段重放数据，再卸载教师，单独加载后段训练。记录完整 logits、历史、边界张量的实际体积和采集耗时。 [QLoRA 的低精度存储与反量化计算](https://arxiv.org/html/2305.14314#S3)

按官方 1017 的投影尺寸、40 层和 8 个独立 draft indexer 估算，全量新增约 51 亿参数，BF16 权重约 9.5 GiB。若采用 BF16 参数和梯度、FP32 主权重及 Adam 的两个 FP32 状态，仅这些张量合计约 76 GiB，激活、临时区和基模另计。末四层对应约 5.1 亿参数和约 7.6 GiB 的同口径训练状态，取决于 indexer 冻结策略。全量训练的验收先测一层反向，再实测跨层梯度、峰值显存和每步耗时。 [官方模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)

2026 年的量化 DFlash 报告对 Qwen3.5-4B 的冻结 INT4 目标直接训练过 BF16 drafter，也测试了先用 BF16 目标预热再适配 INT4 的路线。报告表 4 的平均接受长度分别为 4.97 和 5.03。第二阶段用 INT4 网格反量化后计算。该实验支持以量化模型为训练目标的路线。DSV4.1 自己的技术报告也记录了冻结主干单独训练 DSpark 的阶段。Orthrus 的额外工作是让梯度通过 draft 使用的共享主干计算，并复用真实分页历史。 [量化 DFlash §3.2、表 4](https://arxiv.org/html/2607.04244v1#S3.SS2) · [V4.1 的 DSpark 训练](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS3)
