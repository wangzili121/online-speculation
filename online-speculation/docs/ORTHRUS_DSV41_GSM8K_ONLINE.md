# Orthrus-DSV4.1 GSM8K 在线更新

本实验回答两个不同问题：从基模 Attention 复制初始化后，在线更新是否能改善接受率；短预热后继续在线更新，能否在计入训练和发布开销后提升端到端吞吐。Qwen3.8 的有效配方只作为扫描起点，DSV4.1 的最终参数由独立 tuning 集确定。

## 数据与对照

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-prepare \
  /workspace/data/orthrus-dsv41-gsm8k
```

脚本从 GSM8K 官方仓库下载并校验固定 SHA256，生成互不重叠的 128 题预热集、128 题调参集、384 题在线评测集和官方质量测试集。标准答案只用于最后的正确率统计；draft 更新只使用主模型在真实修正前缀上的 logits，避免把答案标签混入在线教师反馈。

四个实验臂使用相同题目顺序和 greedy 生成：target-only、固定初始化 head、初始化 head 直接在线更新、短预热 head 继续在线更新。每个在线实验重新初始化优化器，训练和权重发布时间计入总耗时。配置见 [`gsm8k-online.json`](../configs/dsv41/gsm8k-online.json)。

## 在线监督

推理仍使用 K32。一次更新默认最多监督 7 个未来位置，但拒绝后的行不能直接使用：系统保留原 anchor 的 draft 快照，沿主模型修正后的真实前缀继续收集验证分布，直到收齐 7 行或请求结束。last-1 只更新第 39 层 diffusion Attention；W8A8 主模型、共享 mHC、MoE、Norm 和 LM Head 均保持冻结。

当前 Torch 参考可对最后一层做可微重放、TV/KL 更新和 head 导出。它用于先验收反馈对齐与学习效果；PyPTO 服务侧的高性能反向和双机版本发布仍需接入。默认候选为 `lr=1e-5`、8 次线性 warmup、前 256 次更新每 16 个 verifier round 更新一次，之后每 128 轮一次、每 4 次更新发布。`teacher_rows`、学习率和频率均须在 tuning 集扫描，不能把 Qwen3.8 的数值直接当作 DSV4.1 结论。

## 明日双机执行

先在两台机器放置相同的 `tuning.jsonl`，然后同时启动 4 题、每题 32 token 的固定 head smoke。这个入口先验证真实 GSM8K 输入、文本解码、greedy 一致性、接受率和吞吐；它不会伪装成已经在线更新。

```bash
export ORTHRUS_WORKLOAD=/workspace/data/orthrus-dsv41-gsm8k/tuning.jsonl
export CLUSTER_KEY_FILE=/root/.orthrus-cluster.key

# node7
bash online-speculation/scripts/dsv41_pypto_env.sh cluster-accept node7 \
  --max-requests 4 --tokens 32 --no-trace-prefill

# node8（与 node7 同时执行）
bash online-speculation/scripts/dsv41_pypto_env.sh cluster-accept node8 \
  --max-requests 4 --tokens 32 --no-trace-prefill
```

结果目录会在启动时打印。任一节点可汇总 Orthrus 结果：

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh gsm8k-summary \
  /workspace/orthrus-cluster-accept.XXXXXX/acceptance/orthrus
```

在线版本必须额外满足三件事才允许扩大样本：保存真实修正前缀上的完整教师 logits；从同一 anchor 重放最后一层并完成有效梯度更新；TP4/DP4 在 verifier 轮次边界原子切换同一 head 版本。缺少其中任何一项都只能称为 fixed-head 推理。

## 验收顺序

1. 固定反馈重放：loss 下降、仅最后一层 draft 参数变化，教师和主模型参数不产生梯度。
2. 16～32 题 smoke：确认延迟教师行连续、版本发布原子、greedy 输出与 target-only 相同。
3. tuning 集选择配置：先看接受长度是否提高，再看计入更新后的吞吐；接受率不动时先排查反馈、梯度与初始化，不盲目增加更新频率。
4. 384 题三种顺序正式比较。分别报告 AR、固定 head、直接在线和预热在线的总 token/总时间、接受长度、训练及发布成本；训练 head 的质量与加速结论不得由初始化 head 的 smoke 替代。
