# Orthrus-Qwen3.8 GDN 递推 kernel

GDN 从主模型已提交的历史矩阵出发，按位置连续递推。draft 只修改临时状态，目标历史不变。本入口独立测量 Q/K 归一化和递推，不包含投影、卷积、门控参数生成、输出 norm、O projection 或共享 MLP；这些阶段仍需要分别检查，不能将递推收益视为整层收益。

## 计算与接口

输入 Q/K/V 为卷积后的张量，形状均为 `[32,48,128]`；Q/K 已按 value heads 展开，尚未归一化。`log_decay` 和 `beta` 为 `[32,48]` FP32，后者保留上游 sigmoid 后的 BF16 舍入。历史矩阵为 `[48,128,128]` FP32，末两维依次为 value、key；`width` 为 1–32。输出为每个位置的 BF16 特征和最终 FP32 临时矩阵。

对每个 head，先按当前模型约定归一化 Q/K，再执行：

$$
\bar{\mathbf q}_t=\frac{\mathbf q_t}{\sqrt{\|\mathbf q_t\|^2+\epsilon}\sqrt d},
\qquad
\bar{\mathbf k}_t=\frac{\mathbf k_t}{\sqrt{\|\mathbf k_t\|^2+\epsilon}},
\qquad \epsilon=10^{-6}.
$$

$$
\mathbf S_t^-=e^{g_t}\mathbf S_{t-1},\qquad
\boldsymbol\delta_t=\beta_t(\mathbf v_t-\mathbf S_t^-\bar{\mathbf k}_t),\qquad
\mathbf S_t=\mathbf S_t^-+\boldsymbol\delta_t\bar{\mathbf k}_t^\top,
\qquad
\mathbf o_t=\mathbf S_t\bar{\mathbf q}_t.
$$

其中 $g_t$ 为 log decay，$\beta_t$ 为更新门；$\mathbf S_0$ 是只读目标历史，$\mathbf S_t$ 是本轮临时状态。当前模型采用块内因果递推，执行优化不改变这些公式。

## 实现与实测

- `gdn.py::recurrence_golden`：独立 Torch 参考，不修改输入。
- `gdn_programs.py::make_recurrence`：原生两段 16 位置实现，以及两段 16 / 单段 32 的纯 Vector 候选。原生输出点积使用 Cube matmul，Vector 候选用逐行乘加归约实现同一个点积；32 位置版本直接读取历史，省去初始矩阵复制及中途状态交接。
- `scripts/validate_qwen38_pypto_gdn.py`：相同输入下交错计时，上传、回读、Torch 参考和 profiling 在墙钟计时外。对有效宽度 1/7/16/17/31/32 和零/非零初态，检查输出、最终矩阵、输入逐字节不变；无效位置填 NaN。输出误差门槛固定为 `rtol=.02, atol=.002`，矩阵为 `rtol=.002, atol=.00002`。

真实 A2 上，value 分块 16/32/64 各 36 项检查通过，共 108 项；Torch 数学测试 9 项通过。下表是 value 分块 64、K32、非零历史的同 worker 对照：每个入口预热五次，交错计时 30 次；设备窗口在计时外单独采集五次，取中位数。

| 递推实现 | Resident dispatch 墙钟中位数 | 设备任务窗口中位数 |
| --- | --- | --- |
| 原生，16+16 | 1806.5 μs | 1067.48 μs |
| Vector，16+16 | 1134.9 μs | 385.92 μs |
| Vector，32 | 1037.7 μs | 289.14 μs |

设备任务窗口按所有任务的最早开始至最晚结束计算，不把并行 block 耗时相加；它也不是纯算术耗时。value 分块 16/32/64 的 Vector32 窗口分别约 414/317/289 μs，当前选择 64 作为接回模型的候选。完整 GDN、整轮 draft 和服务收益需要接回后另测。

原始记录在 `validation/pypto-qwen38/gdn-recurrence-20261109/`，包含三组输入结果、重复时间线、CPU 记录和哈希。

## 运行

使用 [固定环境](QWEN38_PYPTO_PORT.md#固定基线)，在 `online-speculation/` 下执行：

```bash
python scripts/validate_qwen38_pypto_gdn.py \
  --source /sources/OneServing --device 0 --value-tile 64 \
  --repeats 30 --profile-repeats 5 --output /results/gdn.json
```

所有参数驻留设备后开始计时，不需要加载完整 27B 或训练好的 head。该入口显式要求 ONBOARD，不能把模拟器结果作为 NPU 性能。
