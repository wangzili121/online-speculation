# Orthrus-Qwen3.8 GDN 与 K32 执行

GDN 从主模型已提交的历史矩阵出发，按位置连续递推。draft 只修改临时状态，目标历史不变。本文分别检查递推、完整模型层和完整起草；独立递推入口不包含投影、卷积或共享 MLP，完整层的计时则包含这些阶段。

## 计算与接口

展开接口的 Q/K/V 为卷积后的张量，形状均为 `[32,48,128]`；Q/K 已按 value heads 展开，尚未归一化。`log_decay` 和 `beta` 为 `[32,48]` FP32，后者保留上游 sigmoid 后的 BF16 舍入。packed 接口直接接收 `[32,10240]`，通道按 Q2048、K2048、V6144 排列。历史矩阵为 `[48,128,128]` FP32，末两维依次为 value、key；`width` 为 1–32。输出为每个位置的 BF16 特征和最终 FP32 临时矩阵。

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

其中 $d=128$ 为 key 维度，$g_t$ 为 log decay，$\beta_t$ 为更新门；$\mathbf S_0$ 是只读目标历史，$\mathbf S_t$ 是本轮临时状态。当前模型采用块内因果递推，执行优化不改变这些公式。

## 实现边界

`gdn_programs.py` 将 Q/K 归一化和因果递推放在同一个 Vector 任务中。它按每 64 个 value 通道分块，用逐行乘加归约实现状态与 query 的点积，直接读取卷积后的 packed QKV 和 AR 初态；每三个 value heads 共用一组 Q/K，不先复制展开。最终状态属于临时工作区。

完整 GDN 层由 `gdn_block_programs.py` 组织：

```text
K32 input norm、QKV/Z/A/B 投影
    -> 只读历史的因果卷积、decay/beta
    -> packed QKV 归一化与 K32 临时递推
    -> gated norm、K32 O projection
    -> 共享 post norm / MLP、残差
```

`gdn_convolution_programs.py` 只读取历史最后三个输入，生成本轮特征。`gdn_aux_programs.py` 直接输出 48 个有效 gate heads，通过 reshape 读取 Z，保留 sigmoid、gated norm 和 SiLU 的原有中间舍入。

`projection_programs.py` 一次投影 32 行，保持原生 K512 的 FP32 累加顺序；宽投影使用 N128，小投影使用 N64。RMSNorm 将一行分成八段后计算整行均方，分段仅用于 A2 对齐。`shared_programs.py` 以同样的 K32 分块运行共享 MLP，直接引用主模型 bank，不增加第二套 backbone 权重。主模型执行不变。

Full Attention 的 K32 准备和 O projection 也采用整块投影，Q/K norm 与 RoPE 继续调用原生逐 head 计算。`full_preparation_programs.py` 组织这些阶段和共享 MLP；独立分页 Attention 的接口不变。

## 独立算子结果

所有数据在同 worker 内交错测量；上传、Torch 计算和回读位于计时外。设备窗口在墙钟计时外另采五次取中位数，按最早任务开始到最晚结束计算，不累加并行 block。

| 计算段 | 原生 16+16 设备窗口 | K32 设备窗口 |
| --- | --- | --- |
| GDN QKV 投影 | 134.72 μs | 73.48 μs |
| GDN Z 投影 | 86.18 μs | 45.88 μs |
| 单个 A/B 投影 | 12.84 μs | 9.38 μs |
| O projection | 76.42 μs | 36.50 μs |
| Full Q 投影 | 164.80 μs | 88.34 μs |
| 单个 Full K/V 投影 | 18.10 μs | 11.52 μs |
| MLP gate/up 单个投影 | 230.50 μs | 115.70 μs |
| MLP down 投影 | 148.72 μs | 80.98 μs |
| 输入 RMSNorm | 31.82 μs | 7.06 μs |
| Q/K 归一化与 GDN 递推，非零初态 | 约 1073 μs | 约 200 μs |

前九行各预热五次、交错计时 50 次。投影与原生逐字节相同；RMSNorm 的归约顺序产生少量 BF16 差异，使用固定 `rtol=.02, atol=.002` 检查独立 Torch 参考。

投影和 norm 的两组检查共 54 项，32 行 gate 与 gated norm/O projection 另有六项通过。递推对有效宽度 1/7/16/17/31/32、零/非零初态和被 NaN 污染的无效行，分别检查输出、最终矩阵及输入不变。最终 packed 递推重测 value 分块 16/32/64 的 K32 窗口约为 327/231/201 μs，保留 64。

## 真实 27B 接回

单独替换共享 MLP 分块的对照中，两边均使用 Vector 递推和 K32 准备，四层编排的同步计算由 0.199 降至 0.136 s。

下表是正式包的完整起草对照。两列固定 head、worker、历史、分页 Full Attention 和存储方式，仅切换 `native/vector` GDN；后者组合 Vector 递推、packed 输入、K32 准备和共享 MLP。

| 指标 | `native` GDN | `vector` GDN |
| --- | --- | --- |
| 首个完整 draft GDN 层的设备窗口，包含 MLP | 4.169 ms | 1.210 ms |
| 逐层编排：整轮同步计算调用墙钟 | 0.281 s | 0.160 s |
| 四层编排：整轮同步计算调用墙钟 | 0.255 s | 0.131 s |
| 四层编排：包含参数上传的完整起草 | 1.221 s | 1.097 s |

各组交错六轮，第一行另采三次设备时间线。逐层与四层是两组独立实验；`device_seconds` 是同步 `worker.run` 的墙钟，包含主机调度。完整起草仍受 head 上传影响，不能把计算阶段的降幅直接当作服务吞吐收益。

Full Attention 单独比较两种准备方式时，固定输入与共享 MLP32，50 对调用有 49 对采用 K32 更快；调用墙钟中位数由 2.396 降至 2.289 ms。设备窗口分别约 721/721 μs；本次保留 K32 准备，不将这项约 4.4% 的调用收益解释为设备算术提速。

正式包的 64 层、两条路径共 128 项真实输入逐层 HF 对照通过，最大相对 L2 为 0.321%，门槛固定为 2%。387 项 CPU 回归通过；目标缓存不变、六个历史边界、八组 verifier、接受前缀回放、非零槽位与换页均已覆盖。正式包的独立服务检查覆盖 native/fused Attention 搭配 Vector GDN；分页路径的完整起草、校正、缓存回放、HTTP completion/chat/SSE 与正常退出也通过。逐层和四层编排的候选、logits 逐字节相同，验收后所有 NPU 实验进程已退出。记录与源码哈希一同归档。

## 参数上传与 NUMA

单卡装不下整个 head 时，`DraftWeights` 使用预分配缓冲流式上传。主机内存的 NUMA 位置影响这部分耗时：同一 NPU、同一个 512 MiB 张量、30 轮交错 H2D，本地内存约 13.26 GB/s，远端约 7.31 GB/s，回读逐字节一致。

真实模型再用两份完全相同的 head、同一计算程序和同一 NPU 交错六轮，仅切换主机参数的内存位置：

| 指标 | 远端 NUMA | NPU 本地 NUMA |
| --- | --- | --- |
| head 上传 | 1.595 s | 0.947 s |
| 同步计算调用 | 0.130 s | 0.130 s |
| 完整起草 | 1.740 s | 1.094 s |

两边 logits 逐字节相同，`numa_maps` 确认参数分别位于 N6 和 N0。生产启动可在分配 head 之前，用 `taskset` 绑定 NPU 对应的本地 CPU；具体 CPU 编号应从该服务器的 PCIe/NUMA 拓扑确认。这不增加生产 head 副本，也不消除流式上传；两份 head 仅用于对照实验。

## 配置与复现

当前完整服务为 TP1、一个活动请求、512 个总位置和 greedy。默认保留 `native` 对照，优化配置为：

```bash
--attention-backend paged --gdn-backend vector \
--draft-group-size 4 --head-transfer packed
```

在固定环境、`online-speculation/` 下运行独立检查：

```bash
python scripts/validate_qwen38_pypto_gdn.py \
  --source /sources/OneServing --device 0 --value-tile 64 \
  --backends native vector32 packed32 --repeats 30 --profile-repeats 5 \
  --output /results/gdn.json

python scripts/validate_qwen38_pypto_projection.py \
  --source /sources/OneServing --device 0 --repeats 50 \
  --output /results/projection.json
```

入口显式要求 ONBOARD。独立参考、误差门槛和只读检查不随候选改动；完整模型检查使用 `compare_qwen38_pypto_gdn.py`、`audit_qwen38_pypto_layers.py` 以及服务 smoke。

原始记录位于 `validation/pypto-qwen38/` 的 `projection-kernels-20261109/` 与 `k32-integration-20261109/`。其中保留候选源码、重复时间线、未采用的较慢卷积版本及 SHA256。更大的 GDN value tile、N256 投影超过 A2 片上内存；取消最终状态输出仅收益约 0.9%，均未采用。完整 TP 和长上下文需要另行接入，未来模型优化分支的布局与接口需逐项核对。
