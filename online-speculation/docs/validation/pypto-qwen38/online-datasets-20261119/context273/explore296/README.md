# 代码生成探索流

两个固定探索集分别比较 head135 / 硬标签 KL / 3e-5 与 head128 / TV 分布蒸馏 / 1e-5。LongProc 使用 30 题、每题最多 128 token；SPoC 派生流使用 64 题、每题最多 192 token，均遇 EOS 停止。这是配方比较，不是单因素消融。

每条流使用两张物理 A2、TP2、rank0 共卡学习器、1,024 上下文、fresh Adam、last-1、七个候选。全程持续更新并计入全部开销，AR、固定起草与在线按相同请求对交错计时。每组均保留最终发布 head 重跑和原开发题对照，新的预留确认及留出题不在计划内。

计划和初始权重校验位于相邻 `service292/`。`watch_explore296.py` 仅收集已完整关闭的结果，核对逐窗口计时、同源实现、资源配置、实际发布权重及探索门槛；不启动模型、不调参、不派发预留题。

```bash
python watch_explore296.py --artifacts /path/to/artifacts \
  --validation-root /path/to/online-datasets-20261119
```

LongProc 官方 24 题预留和 SPoC 派生 256 题预留分别保留，只有探索配方通过并冻结后才另行派发。它们与此前已完成但未达标的原 24 题确认分开记录。
