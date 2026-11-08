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
- `gdn_programs.py::make_recurrence`：原生两段 16 位置实现，以及两段 16 / 单段 32 的 Vector 实现。原生输出点积使用 Cube matmul，Vector 版本用逐行乘加归约实现同一个点积，并在递推任务内完成 Q/K 归一化；32 位置入口直接读取历史，省去初始矩阵复制及中途状态交接。
- `scripts/validate_qwen38_pypto_gdn.py`：相同输入下交错计时，上传、回读、Torch 参考和 profiling 在墙钟计时外。对有效宽度 1/7/16/17/31/32 和零/非零初态，检查输出、最终矩阵、输入逐字节不变；无效位置填 NaN。输出误差门槛固定为 `rtol=.02, atol=.002`，矩阵为 `rtol=.002, atol=.00002`。

真实 A2 上，value 分块 16/32/64 各 36 项检查通过，共 108 项；Torch 数学测试 9 项通过。下表是 value 分块 64、K32、非零历史的同 worker 对照：每个入口预热五次，交错计时 30 次；设备窗口在计时外单独采集五次，取中位数。

| 递推实现 | Resident dispatch 墙钟中位数 | 设备任务窗口中位数 |
| --- | --- | --- |
| 原生，16+16 | 1806.5 μs | 1067.48 μs |
| Vector，16+16 | 1134.9 μs | 385.92 μs |
| Vector，32 | 1037.7 μs | 289.14 μs |

设备任务窗口按所有任务的最早开始至最晚结束计算，不把并行 block 耗时相加；它也不是纯算术耗时。value 分块 16/32/64 的 Vector32 窗口分别约 414/317/289 μs，因此选择 64。继续将 Q/K 归一化并入递推，36 项检查通过，Vector32 窗口降至约 255 μs；再将 V 行转列改为 reshape，降至约 202 μs，另 36 项检查通过。value 分块 128 超出 A2 的 Vector 内存，不作为可用配置。

最终递推支持两种输入：普通 `[32,48,128]` Q/K/V，以及卷积后的 `[32,10240]` packed QKV。后者在每个 value head 内按 1:3 分组直接读取对应 Q/K，不先复制展开。两组独立检查分别为 36 / 48 项，均通过；非零历史、K32 下，原生设备窗口约 1073 μs，Vector32 约 201 μs，packed Vector32 约 200 μs。packed 的主要作用是省去整层前面的展开阶段，而非显著加快递推本身。

原始记录在 `validation/pypto-qwen38/gdn-recurrence-20261109/`，包含三组输入结果、重复时间线、CPU 记录和哈希。

## 前后处理与整网

`gdn_aux_programs.py` 提供两项配套计算：gate 生成后直接输出 48 个有效 heads，保留 beta 的原有中间舍入；gated norm 通过 reshape 直接读取连续 Z 投影，取消逐 head 复制，再调用原生 O projection。`validate_qwen38_pypto_gdn_aux.py` 对三组输入分别与 Torch 参考、原生路径比较，12 项检查通过，全部输出与原生逐字节相同，输入不变。

| 计算段 | 原生设备窗口 | 改造后设备窗口 |
| --- | --- | --- |
| decay/beta 及去 padding | 9.18 μs | 2.46 μs |
| gated norm 与 O projection | 288.32 μs | 50.64 μs |
| K32 卷积，包含原生路径的历史复制 | 137.72 μs | 37.40 μs |

每个窗口在交错计时外采集五次，取中位数。前两行为每 16 位置，第三行为 K32；第二行包含投影，不能视为单独 norm 的耗时。卷积入口在三组零/非零历史下的六项检查通过，输出与原生逐字节相同；它直接读取目标历史的最后三个输入，只生成本轮特征，不提交新的卷积状态。

完整执行通过 `--gdn-backend native|vector` 选择。`vector` 使用相同公式和参数，将只读 K32 卷积、packed 递推和前后处理接成完整 GDN 层；投影与共享 MLP 保持原生计算。目标状态只读，递推使用私有输出，因果策略不变。

最终组合使用只读卷积、packed 递推和上述前后处理。真实 27B、同一 head、同 worker 六轮交错对照的结果如下；设备时间线另采三次取中位数。

| 指标 | 原生 GDN | Vector GDN |
| --- | --- | --- |
| 首个完整 GDN 层的设备任务窗口，包含共享 MLP | 4.001 ms | 2.271 ms |
| 逐层执行：整个 K32 的同步计算调用墙钟 | 0.324 s | 0.240 s |
| 四层编排：整个 K32 的同步计算调用墙钟 | 0.274 s | 0.195 s |
| 四层编排：包含 head 上传的完整起草 | 1.401 s | 1.321 s |

前两行来自逐层对照，后两行来自四层对照；不能混合相加。`last_timings.device_seconds` 实际是同步 `worker.run` 的墙钟，包含主机调度；第一行才来自单独设备时间线。单卡上传仍占大头，逐层执行的总耗时在两组都约为 2.39 s；不能把设备窗口的减少直接当作吞吐收益，也不跨实验比较绝对耗时。

本组合通过 383 项 CPU 回归，52 项 Attention、84 项递推和 18 项前后处理独立 NPU 检查，以及最终 128 项真实输入逐层 HF 对照；后者最大相对 L2 为 0.285%，门槛固定为 2%。真实 K32 的目标缓存不变、六个历史边界、八组 verifier、接受前缀回放、非零槽位/换页、scheduler 续写均通过。非零槽位检查显式把状态从 slot 0 搬到 slot 1，清空原槽并更换物理页，同一上下文的 logits 逐字节相同；这不是并发服务验收。

HTTP completion、chat、SSE、后端配置与正常退出通过；三种 Attention 后端搭配 Vector GDN 的直接服务检查均通过。当前整网范围仍为 TP1、单活动请求、512 位置和 greedy。

取消最终矩阵回写的独立候选仅将递推窗口从 200.50 降至 198.74 μs，约 0.9%；当前保留输出完整状态的统一契约，不将此候选接入模型。原始结果、实验源码和重复时间线见 `validation/pypto-qwen38/kernel-optimization-20261109/`。

## 运行

使用 [固定环境](QWEN38_PYPTO_PORT.md#固定基线)，在 `online-speculation/` 下执行：

```bash
python scripts/validate_qwen38_pypto_gdn.py \
  --source /sources/OneServing --device 0 --value-tile 64 \
  --repeats 30 --profile-repeats 5 --output /results/gdn.json
```

所有参数驻留设备后开始计时，不需要加载完整 27B 或训练好的 head。该入口显式要求 ONBOARD，不能把模拟器结果作为 NPU 性能。
