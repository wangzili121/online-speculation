# Orthrus-DSV4.1 GSM8K 在线更新

本实验回答两个不同问题：从基模 Attention 复制初始化后，在线更新是否能改善接受率；短预热后继续在线更新，能否在计入训练和发布开销后提升端到端吞吐。Qwen3.8 的有效配方只作为扫描起点，DSV4.1 的最终参数由独立 tuning 集确定。

## 数据与对照

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-prepare \
  /workspace/data/orthrus-dsv41-gsm8k
```

脚本从 GSM8K 官方仓库下载并校验固定 SHA256，生成互不重叠的 128 题预热集、128 题调参集、384 题在线评测集和官方质量测试集。标准答案只用于最后的正确率统计；draft 更新只使用主模型在真实修正前缀上的 logits，避免把答案标签混入在线教师反馈。

内网机器不能联网时，可在联网机器保存官方的 `train.jsonl`、`test.jsonl`，把目录作为第二个参数传入；脚本仍执行同一组 SHA256 校验。失效的 `127.0.0.1/localhost` 隧道代理会自动绕过，不影响普通公司代理。

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-prepare \
  /workspace/data/orthrus-dsv41-gsm8k /workspace/data/gsm8k-source
```

评测按 AR、固定起始 head、在线更新、冻结更新后的 head 四阶段执行。AR 和两个固定 head 阶段使用同一批留出题、相同题序和 greedy 生成。在线阶段用独立题目，支持从模型复制初始化的 head 或已预热的 head 开始。训练和权重发布时间计入在线总耗时。配置见 [`gsm8k-online.json`](../configs/dsv41/gsm8k-online.json)。

## 在线监督

推理仍使用 K32。一次更新默认最多监督 7 个未来位置，但拒绝后的行不能直接使用：系统保留原 anchor 的 draft 快照，沿主模型修正后的真实前缀继续收集验证分布，直到收齐 7 行或请求结束。last-N 只更新连续后 N 层的 diffusion Attention；W8A8 主模型、indexer、共享 mHC、MoE、Norm 和 LM Head 均保持冻结。首轮在相同题目顺序下比较 N=1/2/4/8，不预设 Qwen3.8 的 last-1 也是 DSV4.1 最优值。

Torch 参考可对最后一层做可微重放、TV/KL 更新和 head 导出；通用及 C1A/mHC 的 PyPTO 反向叶子、last-N 连续重放契约和双机驻留双槽发布也已准备。默认候选为 `lr=1e-5`、8 次线性 warmup、前 256 次更新每 16 个 verifier round 更新一次，之后每 128 轮一次、每 4 次更新发布。`last_layers`、`teacher_rows`、学习率和频率均须在 tuning 集扫描，不能把 Qwen3.8 的数值直接当作 DSV4.1 结论。

## 双机执行

先在两台机器各刷新一次本地镜像。该命令不访问网络，也不重编 PyPTO/CANN；它只把当前仓库的专家状态 ABI 和 Python 编排叠加到已经固定的镜像上：

```bash
PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline \
  bash online-speculation/scripts/dsv41_pypto_env.sh refresh-image
```

先在任一空闲卡运行单机 readiness gate。它会执行全部 DSV4.1 回归、在固定 serving 源码上实际应用 online patch 并做 Python 编译，然后复验 129280 维完整词表 TV/KL、投影输入与参数梯度、RMSNorm 前反向、C1A Attention、mHC、MoE、AdamW 和发布 kernel：

```bash
NPU_DEVICE=0 PYPTO_VISIBLE_DEVICE=0 PYPTO_RUNTIME_DEVICE=0 \
  bash online-speculation/scripts/dsv41_pypto_env.sh online-ready
```

首轮单机验收记录包括 158 项 DSV4.1 测试、4 项跳过和 5 组子测试通过，22 个设备数值用例通过，61 个生产形状程序完成编译；其中包含 TP4/K32 的逐 query C1A 历史重放、sink、临时块 K/V 梯度，K32×5120 的 mHC 反向，以及固定 Top-6 路由、冻结 W8A8 专家输入 VJP、EP16 编排、AdamW 和发布。inactive head bank 的设备到设备发布也已实卡通过，完整 head 不经 CPU 往返。`readiness.json` 汇总测试、serving 补丁和实卡 kernel 证据，并保留为宿主用户可读；它不替代双机 serving 中的一次完整在线更新。

先在两台机器放置相同的 `tuning.jsonl`。固定 head 推理链路已经通过双机验收；在线更新应直接使用统一入口同时启动 4 题、每题 32 token 的 smoke。它会加载在线策略，执行真实反馈收集、last-N 反向、同步更新与安全发布：

公司环境的镜像、模型、数据、八卡列表、结果目录和密钥路径统一保存在一个入口中。两边只需同时运行短命令：

```bash
# node7
bash online-speculation/scripts/run_dsv41_pypto_company_online.sh node7 smoke 1

# node8（与 node7 同时执行）
bash online-speculation/scripts/run_dsv41_pypto_company_online.sh node8 smoke 1
```

公司入口按节点选择同一份密钥的实际位置：node7 使用 `/root/.orthrus-cluster.key`，node8 使用 `/home/wangzili/.orthrus-cluster.key`。

结果目录会在启动时打印。任一节点可汇总 Orthrus 结果：

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-summary \
  /workspace/orthrus-cluster-accept.XXXXXX/acceptance/orthrus
```

在线版本必须额外满足三件事才允许扩大样本：保存真实修正前缀上的完整教师 logits；从同一 anchor 重放最后一层并完成有效梯度更新；TP4/DP4 在 verifier 轮次边界原子切换同一 head 版本。缺少其中任何一项都只能称为 fixed-head 推理。

## 性能记录

两组预热使用相同输入，并对比生成的 token。每个评测臂记录 Serving 实际执行过的 `draft_head_versions`，冻结评测只有实际使用最终发布版本才通过。

报告的 `serving_phases_by_host` 按主机分别记录 prefill、decode、draft、verify、accept 和 release 的调用数、执行时间和搬运量。预热计入单独的启动时间。在线实验同时保留计入全部更新的吞吐与扣除更新的估算值，冻结后的 head 在同一批未参与更新的题上与 AR、初始 head 对照。

- 词表归约支持同一 DP 分区一次处理 1～4 组 K32。默认 GSM8K 调度仍按每分区一个请求运行。111 单卡对照中，每个 TP 分片的回读与选 token 从 3.62 ms 降到 1.07 ms，回读量从 2,068,480 字节降到 1,024 字节。167 已实卡对照 64、96 和 128 行，128 行的每分片阶段从 21.43 ms 降到 1.73 ms。所有用例的 token ID 一致，并列按最小 ID 处理。需要训练的分区保留完整学生 logits，验证器的完整教师 logits 由本机 TP 分片按原词表顺序拼接。
- GSM8K 报告路径的验证器对无待收集教师反馈的分区也使用设备端词表归约。等待反馈的分区仍将完整教师 logits 交给 learner，诊断验收保留完整 logits 文件。167 用两张卡分两批执行四个实际大小的词表分片，K32 的词表回读和选 token 从 19.15 ms 降到 3.16 ms，设备回读从 8,273,920 字节降到 4,096 字节。32、64、128 行的 greedy token ID 全部一致。
- C1A 反向把 softmax VJP 和两份块内矩阵的提取合到一次调用，单卡从 3.51 ms 降到 1.27 ms。完整 score 梯度、sink 梯度和两份块输出与原实现逐元素一致。
- 在线更新将 Adam 的候选参数写回与 BF16 转换合在一次调用中。111 上 8192×1280 投影从 2.14 ms 降到 1.33 ms，三个 FP32 状态及 BF16 执行权重逐元素一致。本机各 rank 并发执行，所有候选写完后才切换优化器状态。发布时各 rank 并行复制到各自的备用权重槽。双机确认后，按既有步频切换版本。167 用设备写入的候选重测后，两卡发布 120 MiB 从 28.69 ms 降到 15.59 ms，四卡发布 240 MiB 从 56.04 ms 降到 16.92 ms。测试逐次核对 D2D 字节计数、零 H2D/D2H 和发布权重。四卡 Adam 调度从 11.64 ms 降到 9.16 ms。
- 在线梯度统计按 rank 并行，累计时保持原来的参数与 rank 顺序。统计和归约覆盖全部输出行，复用工作区时可省去清零。167 两张卡分批执行四个逻辑 rank 的投影梯度统计中，裁剪阶段从 68.67 ms 降到 48.07 ms，省去 1,361,600 字节工作区清零，范数和裁剪系数完全相同。
- EP16 反向按整行搬运路由载荷，目的端一次写完专家输入，源端按原 slot 顺序直接读回并累加六条路由。167 单卡 K32、192 条路由的同输入对照中，相较 `c2be3c5`，打包 1.93→1.71 ms、解包 1.77→1.12 ms、回传打包 1.41→1.32 ms、源端恢复 9.16→0.83 ms。路由与 token 输入梯度的最大绝对误差均为 0。四卡同时启用整行写入和 rank 并行调度后，32 行集中在一个源 rank 时的打包从 10.21 降到 5.97 ms，四张卡的包逐字节一致。
- 在线 MoE all-to-all 的接收缓冲由 HCCL 完整写入，成功后继续作为驻留设备数据使用。167 四卡用同尺寸张量测试，接收清零从 252,053,504 字节降为 0，耗时 17.13→15.61 ms。每张卡的接收和后续设备到设备复制结果一致。
- 每次在线更新按层读取被选中 draft 的专家 ID，用任一源 rank 发往任一目的 rank 的最大路由数决定桶宽。两台机器通过已有的更新消息共享 32、64、128 或 192 行的选择。192 行可容纳 K32 的全部 Top-6 路由。桶宽为 32、64、128 时，每 rank 的双向逻辑载荷分别约是满宽的 1/6、1/3、2/3。167 实卡用三组相同路由分别比较小桶和满桶，四种规格的恢复梯度均逐元素一致，单卡 32 行桶的打包耗时从 1.91 降到 0.96 ms。驻留路由 ID 选桶只读回 768 字节每层，双卡通信和后续设备复制通过。在线 profiling 记录每层选择和实际发送字节。
- Serving 在每轮起草中按请求解析一次历史页表。K32 的层间有限值检查在各 rank 并行执行。单卡 32 行检查从 2.23 ms 降到 0.94 ms，有限值与 NaN/Inf 判定相同。验证回滚合并相邻区间。SWA 仅备份候选会写入的行，压缩缓存与 Engram 保留原来的页级备份。111 跨两个非连续物理页的 32 行对照中，一份 SWA 缓存从备份 918,400 字节降到 32,768 字节，备份和恢复从 0.85 ms 降到 0.46 ms。实际写入后回滚，缓存逐字节一致。167 四卡已验证并行备份与回滚，96 MiB 的备份和恢复合计从 50.16 ms 降到 16.21 ms。正常结束与模拟验证失败均恢复相同缓存字节。

单卡和多卡数字均为各阶段的稳态微基准。双机报告用同一组输入核对 greedy 输出、权重发布版本、接受长度和整体吞吐。

单卡 EP16 算子可用 `dsv41_pypto_env.sh ep-vjp-check --active 32 --iterations 16` 复验，脚本将逐阶段时延与精度写入 `ep-vjp.json`。
小桶与满桶对照可在相同参数上分别加 `--routing balanced --bucket-rows 32` 和 `--routing balanced --bucket-rows 192`。

在同一双机拓扑上用相同入口与参数分别运行 `c2be3c5` 和当前版本。两次结果都保留后，在同一节点比较：

```bash
python3 online-speculation/scripts/summarize_dsv41_pypto_cluster.py \
  /workspace/候选运行/acceptance \
  --compare-gsm8k /workspace/基线运行/acceptance \
  --output dsv41-gsm8k-comparison.json
```

工具会核对硬件映射、初始 head、模型权重、题目顺序、生成结果和在线策略，再输出吞吐、发布成本和各阶段搬运量。较早的报告可从保存的 Serving 回执恢复阶段数据。新版报告单列 verifier 快照与恢复耗时，并记录每轮提交的最大 DP token 数，包含 anchor。该分布用于评估复用 16 行执行路径的覆盖率，提交仍走现有 128 行程序。

## 验收顺序

1. 固定反馈重放：loss 下降、仅配置的 last-N draft 参数变化，教师、indexer 和主模型参数不产生梯度。
2. 16～32 题 smoke：确认延迟教师行连续、版本发布原子、greedy 输出与 target-only 相同。
3. tuning 集选择配置：先看接受长度是否提高，再看计入更新后的吞吐；接受率不动时先排查反馈、梯度与初始化，不盲目增加更新频率。
4. 384 题三种顺序正式比较。分别报告 AR、固定 head、直接在线和预热在线的总 token/总时间、接受长度、训练及发布成本；训练 head 的质量与加速结论不得由初始化 head 的 smoke 替代。
