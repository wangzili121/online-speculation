# Qwen3.8 在线更新效果实验

这个实验回答：没有离线训练 head 时，从主模型注意力复制初始化，只更新最后 1 层或 4 层，能否提高接受长度并带来净加速。它使用现有 PyPTO 推理和手写反向链路，不改变 GDN 策略和 reached-prefix TV 目标。

## 数据与对照

使用 [GSM8K 官方数据](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)。只读取 `question`，应用模型 chat template、关闭 thinking；数据集答案不作为输入或训练标签。监督来自 verifier 实际到达位置的主模型概率分布。

训练题、调参题、最终测试题按题目内容去重并分开。两个 last-N 实验固定题目顺序、初始化 head、学习率及更新预算，报告原文件 SHA256、分组和 tokenized SHA256。每次评估暂停更新，交错运行同题 target-only 和固定 head；评估数据不进入重放区。

初始化 head 也将对应末尾参数放进更新时使用的设备双缓冲，且不执行优化器步骤。这样更新前后参数驻留方式相同，不会把少上传参数带来的提速误计为学习收益。last-1 和 last-4 各自与自己的初始化及 AR 对照，不能仅根据两组原始 TPS 判断训练方法优劣。

本实验默认让冻结的 AR 复制前段直接引用主模型参数 bank，仍独立执行 draft 的计算；需要保留 FP32 的 `A_log` 单独存放。启用前先对照共享前后的首轮完整 draft logits SHA256 和输出 token，二者必须相同。这项复用只适用于未训练过的 AR 复制前段；可用 `--no-share-ar-prefix` 关闭，不能套用于参数不同的离线训练 head。参数共享前后的耗时独立记录，不计入在线学习收益。

| 设置 | 初始试验 |
| --- | --- |
| 初始化 | 所有 draft mixer 从主模型 AR 参数复制，未加载训练 head |
| 更新范围 | last-1、last-4 分别启动；其余参数冻结 |
| 总上下文 | 512；输入不超过 128 token |
| 起草宽度 | 最大 32；另测活跃宽度 4/8/16/32 的理想接受成本 |
| 在线更新 | 32 次；每 8 轮更新，重放最近 8 块 |
| 优化器 | 现有 FP32 Adam、梯度裁剪 1、峰值学习率 1e-4、前 4 步线性预热 |
| 小规模评估 | 每题输出最多 48 token，正常按 EOS 停止；0/16/32 步评估 |
| 保存 | 更新中不落盘，只在末尾保存一次完整 checkpoint |

这里有三个不同的“窗口”：512 是上下文上限；32 是起草块长；8 是重放块数。改变活跃块长不会让当前 K32 padding 的投影、GDN 等计算按比例减少，所以短块是否更快必须实测。

## 指标

同时记录 TV loss、每次更新监督位置数、梯度范数和更新耗时。效果主要看留出题的平均接受长度、候选接受率，以及相对同题 AR 的输出吞吐。

固定 head 推理和在线净吞吐分别统计。后者包含生成、反馈搬运、反向、优化器及参数发布；加载、编译、预热、评估和末尾 checkpoint 保存单独列出。吞吐为总输出 token / 总耗时。只有逐 token 与 AR 相同，才报告无损加速比。

成本标定仍执行真实 draft，再用目标答案替换候选，测全部接受时的成本上限；只有这项标定忽略 EOS。它不能作为训练效果或实际加速结果。

## 运行

将官方 `train.jsonl`、`test.jsonl` 放在数据目录，可同时提供记录仓库 commit 的 `provenance.json`。在已固定的 OneServing/PyPTO 环境中执行：

```bash
python scripts/benchmark_qwen38_pypto_online.py \
  --source /path/to/OneServing \
  --model /data/disk/models/Qwen3.8-27B \
  --data /data/disk/datasets/gsm8k \
  --device 0 --learner-device 1 \
  --last-layers 1 --steps 32 --eval-steps 0,16,32 \
  --tokens 48 --dev-count 2 --test-count 3 \
  --output /data/disk/results/last1.json
```

将 `--last-layers` 改为 `4` 并指定新的输出路径，独立运行另一组。不要传训练过的 head；本入口刻意从 AR 复制初始化。数据准备、参数加载和训练都在同一固定环境中完成，NPU 数量为主模型 1 张、learner 1 张。

## 如何解释结果

这是冷启动探索，不等同于已有离线 head 的在线续训。[源仓库](https://github.com/xchang1121/inference_scaling/blob/8d82be93e9874537fc8055a06574f8e25fdc9d39/online-speculation/docs/RUNNING.md) 将两者分开：主线 last-N TV 续训使用预训练 head；AR 复制冷启动实验则使用全量 draft attention、完整回答重放及短块 KL 训练。因此，这里的 last-1/4 若未明显加速，不能直接推断离线 head 的在线续训无效。

32 次更新与少量留出题用于观察趋势和成本，不是收敛结论或业务性能验收。根据结果再决定是否延长训练、调整块长或重放量。
