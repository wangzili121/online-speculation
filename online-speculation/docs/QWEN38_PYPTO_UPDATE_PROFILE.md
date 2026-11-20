# Qwen3.8 TP2 在线更新计时

`blockspec.pypto_qwen38.update_profile` 用于定位一次在线更新的开销，默认不安装。它在现有 TP2 更新入口外记录同步调用时间，保留原参数、返回值和异常，不替换计算或修改发布周期。

在独立诊断进程中，执行模型 benchmark 前安装：

```python
from blockspec.pypto_qwen38.update_profile import install

profile = install("/data/disk/results/update-profile.json")
# 随后调用现有 TP2 benchmark 的 main；正常退出时自动写出结果。
```

输出中的 `train` 包含清梯度、前反向、梯度累加、有限性与范数检查、Adam、执行权重渲染，以及发布分片准备。`commit` 单独记录安全边界上的服务权重上传。LoRA 模式另外记录完整权重合并与检查。

每条事件带有 `id`、`parent`、`inclusive_ns` 和 `exclusive_ns`。父窗口已经包含子窗口；汇总不同计算阶段时使用排除子窗口后的 `exclusive_ns`，不能将父子耗时直接相加。`train` 的元数据保留更新版本、轮次、监督位置数和 loss；带 `replay_audit=true` 的更新包含额外回放，应与普通更新分开分析。

这里测得的是主机调用窗口，包含 worker 通信和等待设备完成的时间，不是纯 kernel 时间。诊断请求流不用于正式吞吐验收；正式 AR、固定起草、持续更新对照均不安装该计时器。

460 项 CPU 回归通过，其中新增三项检查覆盖嵌套窗口去重、异常与返回值保持、位置/关键字参数和更新元数据。

真实 27B、同两张 A2 的 step128 / last-1 / TV / 逻辑宽度 8 短诊断已完成八次更新并逐次发布，greedy 与 AR 一致，导出与服务权重精确一致。排除第一次带额外回放审计的更新后，剩余七次的训练调用中位数为 75.25 ms，提交上传为 16.04 ms。训练内的前反向为 33.95 ms，有限性和范数检查每次调用为 6.54 ms（一次更新调用两次），执行权重渲染为 1.70 ms，发布分片准备为 16.12 ms。子阶段已经包含在训练时间内；各阶段中位数也不要求相加等于整体中位数。

这是一组短诊断的同步调用测量，不能外推为其他层数、发布周期或完整数据集的性能。原始事件、源码哈希、服务结果和去重汇总保存在 `validation/pypto-qwen38/online-datasets-20261119/profile196/`。
