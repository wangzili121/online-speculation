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

460 项 CPU 回归通过，其中新增三项检查覆盖嵌套窗口去重、异常与返回值保持、位置/关键字参数和更新元数据。实际设备计时另以原始诊断记录为准。
