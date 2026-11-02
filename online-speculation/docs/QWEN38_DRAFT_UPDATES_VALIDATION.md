# Qwen3.8 Draft 更新验证

## 范围

保留在线 last-N 与独立离线全 draft 两条路径，冻结主模型；不包含 LoRA，也不是全参数微调基模。
代码位于 `feature/orthrus-qwen38`；本轮执行代码为 `50d3bdc`，后续 README 调整不影响执行代码。

## 在线 Last-N

已有真实 27B、TP2 验收覆盖 last-1 / last-4 的更新、原生验证反馈、目标缓存只读、目标参数冻结和 head 导出重载。
8K 验收中的 last-4 累计 298 次更新，含 32 步固定反馈重放；固定反馈 TV loss 从 0.5044139 降到 0.4928726。
32K 的选定请求边界也已通过。具体范围与未通过的数值用例保留在 [验收摘要](QWEN38_ACCEPTANCE_SUMMARY.md) 和 [验证队列](VALIDATION_QUEUE.md)，不能据此宣称所有推理用例或业务加速都已验收。

## 离线全层

使用真实 Qwen3.8-27B 权重、两张 Ascend 910B3，BF16、TP2、K=32。
配置为 CPU FP32 master/AdamW、逐层 checkpoint、forward KL、学习率 1e-4，全部 64 层 draft mixer 更新。
Full Attention 块内双向，GDN 保持 `orthrus_qwen35_reference` 策略。

本次用一段 64-token 自制说明文本作功能短训，anchor 限定在位置 32，完整监督后续 31 个位置，便于对照重复更新。
训练入口可随机选取合法 anchor；本次固定块不等于论文数据集训练或泛化测试。

| 更新步 | 两个 TP rank 的 KL loss | 含教师前向的单步时间 |
| --- | --- | --- |
| 1 | 7.161620 | 120.82 s |
| 2 | 4.123861 | 95.93 s |
| 3 | 3.945479 | 94.99 s |
| 4 | 3.055085 | 94.66 s |

每步两个 rank 的 loss、梯度范数一致，梯度覆盖层 0–63。更新后 draft 哈希改变，主模型每 rank 26,896,675,584 字节参数的 SHA256 前后完全相同。
已按原生 TP head 契约导出 `version4`，进程正常结束并释放设备；原始记录见 [离线结果](validation/qwen38/full-draft-20261102/offline.json)。

离线教师使用共享原生 TP 权重的 AR 视图及自有历史，而不是在线生成反馈；因此离线记录中原生 proposer/验证/cache 审计计数为零是预期行为，不能将其当作已验证线上缓存。
线上缓存、候选验证和 last-N 更新由下面的重载测试单独检查。

## 导出后重载

正在独立进程加载离线 `version4`，运行 greedy/采样请求并续训 last-4，检查 head 哈希、主模型冻结、历史只读与 TP 计数一致性。完成前不将这一新增串联测试标为通过。

## 回归与限制

CPU 回归为 630 passed、32 skipped，包含全层梯度、双 TP 全局裁剪、checkpoint 等价性、混合参数精度与原有 last-N 路径。跳过项不能算作已通过的设备测试。
没有自研 NPU kernel。少卡 CPU offload 证明功能可运行，不代表高效预训练；本次 loss 下降不证明业务接受率、端到端延迟或吞吐提升。
