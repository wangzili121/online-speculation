# 2026-11-10：真实 27B 整块 KL 冷启动实验

**故障版本归档。** 本批运行使用的 LM-head 输入梯度存在真实尺寸长行读取错误，不能作为训练效果或反向正确性的验收。原始文件保持不变，供复现与对照；修复记录见[真实反馈数值诊断](../../../QWEN38_PYPTO_TRAINING_DIAGNOSIS.md)。

四组均从同一 AR-copy head 开始，K=8，128 次更新，每次监督 8 块、共 56 个位置。使用同一批 GSM8K 训练问题的 45 条 AR-only 已交付回答，随机顺序和 anchor 完全相同。两个独立 NPU worker 分别承担 TP1 推理与后段训练。

| 主记录 | 更新范围 | 学习率 |
| --- | --- | --- |
| [cold-last1-k8-r2.json](cold-last1-k8-r2.json) | last-1 | 1e-4 |
| [cold-last4-k8-r2.json](cold-last4-k8-r2.json) | last-4 | 1e-4 |
| [cold-last1-k8-lr1e5-r3.json](cold-last1-k8-lr1e5-r3.json) | last-1 | 1e-5 |
| [cold-last4-k8-lr1e5-r3.json](cold-last4-k8-lr1e5-r3.json) | last-4 | 1e-5 |

每个前缀对应的 `.loss.json` / `.suffix.json` 为 NPU 预检查，`.supervisor.json` 记录命令及退出码，`.checkpoint.json` 为 Adam 检查点清单。大权重文件留在服务器，未放入 Git。

四组均正常退出，版本连续为 1–128，每组 1,024 个完整监督块。逐项复核了随机 anchor、样本版本、训练与留出题目隔离、留出评估期间不更新参数；全部 104 个留出投机请求逐 token 对齐配对 AR，最终 scheduler、KV 占用与错误计数均为零。

[权重抽查](cold-weights-check.json)额外核对第 63 层 Q 投影前 8 行：四组 FP32 master 和舍入后的 BF16 权重均发生变化，Adam 一阶状态有限且非零。这是指定参数区域抽查，不代表逐一复核所有参数。

[CPU 记录](cpu.txt)为 502 项回归。NPU 预检查覆盖完整 248,320 词表 KL 及其梯度、小尺寸末层前后向。真实模型运行用于验证采集、反向、Adam、发布、后续推理和检查点的闭环。

结果说明见[实验报告](../../../QWEN38_PYPTO_COLD_KL.md)。[summary](summary) 中保留逐步 loss、全部 dev/test、首尾对照 CSV 和曲线。固定 head 的测试吞吐不含训练；回放成本包含真实前缀计算、教师、起草、反向和参数发布，诊断重算耗时另列但不从总墙钟时间中扣除。

R2 使用 `58afc31` 的训练实现；R3 使用 `2930617` 的相同训练 kernel，增加同批 loss 重算。具体软件版本、代码摘要及文件 SHA256 见 `provenance.json`。结果不使用强制接受、替换候选或理想化 oracle。

重新生成汇总，在 `online-speculation` 目录执行：

```bash
python scripts/summarize_qwen38_pypto_online.py \
  docs/validation/pypto-qwen38/cold-kl-20261110/cold-last1-k8-r2.json \
  docs/validation/pypto-qwen38/cold-kl-20261110/cold-last4-k8-r2.json \
  docs/validation/pypto-qwen38/cold-kl-20261110/cold-last1-k8-lr1e5-r3.json \
  docs/validation/pypto-qwen38/cold-kl-20261110/cold-last4-k8-lr1e5-r3.json \
  --vary-learning-rate --output-dir /tmp/cold-kl-summary
```
