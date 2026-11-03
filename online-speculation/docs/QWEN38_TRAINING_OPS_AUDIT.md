# Qwen3.8 Draft 训练算子与真实语料验证

验证目标是冻结 Qwen3.8-27B 主模型，只更新 draft 参数，并复用已有高效算子；不开发 kernel、不改变 Attention/GDN 的数学定义。
训练性能按两种配置分别记录：**参数、梯度和优化器状态驻留 NPU 是主要基线；双卡 CPU offload 是省显存的兼容路径。** CPU 优化器或搬运的改善不代表设备驻留训练的提速。下文的小算子前后向计时在 NPU 上执行，不包含 offload。
算子对照与少卡验证使用 Ascend 910B3、TP2、K=32；设备驻留验证另使用 TP4。Torch 路径起始代码为 `51d7464`，可选 FLA 接入为 `b078413`，本轮执行优化为 `fe12ee2`。
镜像固定为 `quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a`，Torch-NPU 为 `2.10.0.post4`，Triton-Ascend 为 `3.2.2`。

## 四卡设备驻留验证

真实 27B、TP4、K32、128-token 语料窗口、FLA 反向下，开启和关闭逐层 checkpoint 各完成六步全 64 层更新并导出 head，另有两轮各两步的独立短训。每轮内部四 rank 的 loss、梯度范数和层覆盖一致；draft 确实更新，冻结主模型的完整参数 SHA256 未变。

实测每 rank 的 FP32 master 和梯度各约 6.74 GiB，AdamW moment 约 13.49 GiB，均驻留 NPU；CPU 仅保留约 2 KiB 的优化器标量计数，并非参数或 moment offload。当前配置为每 rank 4 GiB KV、2K 最大上下文。

剔除首步编译，后五步中位数如下。单步包含教师前向、draft 更新和验证计时；阶段时间取 rank0，不能相加当作完整单步。

| 配置 | 单步 | 重放/反向/通信 | 范数/归并/裁剪 | 优化器 | 峰值 allocated / reserved |
| --- | --- | --- | --- | --- | --- |
| 开启 checkpoint | 3.63 s | 1.90 s | 22 ms | 124 ms | 53.79 / 54.38 GiB |
| 关闭 checkpoint | 2.91 s | 1.15 s | 22 ms | 124 ms | 53.79 / 55.30 GiB |

两轮使用相同文本窗口、anchor、初始化参数和执行配置，仅切换 checkpoint。短序列下关闭 checkpoint 减少重算，观测到单步中位耗时降低约 20%；内存峰值此时主要受模型和优化器影响，不能据此推断长序列也不增加显存。两次独立运行的 loss 不作为逐值等价性证据；本表也不代表长期吞吐或业务推理加速。原始记录见 [开启 checkpoint](validation/qwen38/execution-20261103/offline-tp4.json) 与 [关闭 checkpoint](validation/qwen38/execution-20261103/offline-no-checkpoint-tp4.json)。

同一份真实梯度上，两种配置各完成四 rank、六步共 24 次旧/新范数对照，48 次全部通过，最大相对误差约 `2.7e-7`。开启 checkpoint 时旧范数与归并的中位数约 35 ms；新路径连同梯度缩放约 22 ms。两者工作范围略有不同，不把这个比例换算为完整训练加速比。对照仅在验证 worker 中执行，额外耗时独立记录。

记录见 [设备驻留训练](validation/qwen38/execution-20261103/offline-tp4.json) 与 [独立参考运行](validation/qwen38/execution-20261103/offline-tp4-reference.json)。这些结果不与双卡 offload 的整步时间直接换算加速比。

独立 TP4 推理进程分别重载六步训练导出的 head，加载前哈希与导出完全一致；last-1、last-4 两种配置均完成 greedy、采样请求和 7 次在线更新。每种配置的四 rank 更新计数与 loss 一致，各完成 6 次历史缓存只读检查；更新后 head 改变，主模型完整参数哈希仍未变。记录见 [TP4 last-1](validation/qwen38/execution-20261103/online-last1-tp4.json) 和 [TP4 last-4](validation/qwen38/execution-20261103/online-last4-tp4.json)。这证明设备驻留的训练、导出、加载、推理和续训链路可执行，不评价接受率或 head 收敛。

## TP 通信与词表分片

本轮只组合现成 Torch 算子和 HCCL collective，不新增 NPU kernel。FP32 master、梯度与优化器驻留 NPU，双卡 offload 继续保留。

### 复制参数梯度

64 层 draft 的 80 个复制 norm 梯度不再逐个 `all_reduce`，而是拼成一个 FP32 buffer，归并一次。各参数的 `.grad` 指向该 buffer 的不重叠视图，避免再逐项发起 NPU 复制；下一轮 `zero_grad(set_to_none=True)` 仍按原方式释放梯度。CPU offload 也只搬运这个小包。

四卡真实 head 规格、预热后 30 次同步计时，rank0 中位数由 **16.35 ms 降至 0.53 ms**。最大绝对差 `4.77e-7`，相对 L2 误差 `2.46e-8`；这是 FP32 collective 归约顺序差异，不宣称逐 bit 一致。它只衡量复制参数梯度的归并，不是完整训练加速比。

### 分片 Loss

原路径汇总完整 student logits 后计算 KL/TV。可选路径仅计算本 rank 的 LM head 词表行，汇总每个位置的局部 log-sum-exp，得到完整词表的归一化量，再归并各分片的 loss。KL 反向为 `q-p`；TV 反向为 `q*(sign(q-p)-sum(q*sign(q-p)))/2`，其中求和覆盖完整词表。LM head 继续归并 hidden 梯度，教师分布保持 detach。没有截断词表或改变蒸馏目标。

下面是 TP4、31 行、248,320 词表下，BF16 logits 的 loss 前后向和临时 allocator 峰值。两条路径都包含完整词表上的精确目标；临时峰值不包含输入、模型、优化器和外部 CANN 内存。

| Loss | 完整汇总 | 词表分片 | 完整汇总临时峰值 | 词表分片临时峰值 |
| --- | --- | --- | --- | --- |
| KL | 1.43 ms | 1.73 ms | 164.68 MiB | 59.37 MiB |
| TV | 1.51 ms | 1.99 ms | 195.37 MiB | 74.69 MiB |

分片版本降低临时显存，但在当前小块规模下更慢，因此 **默认保留 `gathered`，`vocab_parallel` 显式选择**。推理 logits、候选采样和 verifier 不走这个分片 loss 接口，head 格式不变。

四个 rank 的 FP32/BF16、KL/TV 均通过数值对照，loss 最大相对 L2 误差为 `9.25e-7`，局部 logits 梯度为 `5.99e-4`。CPU 另覆盖两卡/四卡、词表 padding、空分片、极端 logits 和非均匀上游梯度；两卡小模型覆盖全层/last-1、KL/TV、checkpoint 的完整参数梯度与裁剪、更新对照。本轮回归为 660 passed、32 skipped。

原始记录见 [rank0](validation/qwen38/tp-execution-20261103/tp-training-rank0.json)、[rank1](validation/qwen38/tp-execution-20261103/tp-training-rank1.json)、[rank2](validation/qwen38/tp-execution-20261103/tp-training-rank2.json)、[rank3](validation/qwen38/tp-execution-20261103/tp-training-rank3.json)。以上是本轮算子与梯度证据；真实 27B 长窗口和在线更新复验仍待可用的四卡运行环境，不将上文旧版本已完成的整模型验证算作本轮增量验收。

## 算子选择

| 模块 | 核对与实测 | 当前采用 |
| --- | --- | --- |
| GDN | 镜像的 `npu_chunk_gated_delta_rule` 可以前向，但反向对 Q/K/V、g、beta 均未返回梯度。 | 保留 Transformers 可微 chunk 实现，短块按长度选择 1/8/32 等 chunk 大小，避免默认补到 64；长序列和显式 chunk 配置不变。 |
| 因果卷积 | `npu_masked_causal_conv1d` 有配套反向，但镜像文档限定宽度 3、昇腾 950；本模型为宽度 4、A2。推理所用自定义卷积也不能直接当训练接口。 | 训练使用已有 `conv1d + SiLU`，保留 FP32 中间计算；推理仍用原有融合路径。 |
| GDN 门控 RMSNorm | 现成 `npu_rms_norm` 支持所需输入与权重梯度，可继续连接 gate 的 SiLU。 | NPU 训练用 FP32 `npu_rms_norm + SiLU`；推理仍用既有融合门控 norm。 |

实际 TP2 的 head 数和维度下，以下时间均包括一次前向、标量 loss 和反向，不包括 CPU 参数 offload；预热后 20 次同步计时取中位数。

| K32 模块 | 原实现 | 调整后 | 数值对照 |
| --- | --- | --- | --- |
| GDN | 62.13 ms，chunk=64 | 34.50 ms，chunk=32 | 前向完全一致；各输入梯度最大相对 L2 误差 `4.8e-7`。 |
| 门控 RMSNorm | 1.57 ms | 1.15 ms | 前向完全一致；各输入及权重梯度最大相对 L2 误差 `9.0e-7`。 |

GDN 还比较了长度 1/8/17，norm 比较了长度 1/16；不是只测一个 K32 样例。
融合 RMSNorm 反向使用了约 16 MiB 额外工作空间，选择它是时间与空间的权衡，不能宣称所有指标均优或所有环境最优。
原始记录见 [算子计时与梯度](validation/qwen38/training-audit-20261102/operators.json)。

固定镜像缺少 GDN 融合反向，不代表 NPU 生态没有现成实现。[FLA 0.6.0](https://github.com/fla-org/flash-linear-attention/tree/v0.6.0) 提供 Triton-Ascend 后端。
将 `fla-core==0.6.0` 和 `flash-linear-attention==0.6.0` 隔离加载后，K32 GDN 前后向中位数为 **9.61 ms**（预热后 5 次同步计时）；对照 Torch FP32 中间计算，前向相对 L2 误差为 `0.00370`，五类输入梯度最大为 `0.00486`。
这不是逐 bit 等价，但通过当前 BF16 算子数值门槛。记录见 [FLA 与搬运小测](validation/qwen38/training-audit-20261102/fla-operators.json)。

该路径以 `--training-gdn-backend fla` 显式选择，只替换训练阶段的 chunk GDN；不改变因果递推定义、不替换主模型或 draft 推理 kernel，也不接管缓存提交。
默认 `torch` 路径仍然保留。FLA 推荐的 Torch-NPU/Triton 版本比当前镜像新，本轮测试结果只适用于记录的镜像与依赖组合，不能直接推广到所有版本。

可选 FLA 路径另验证了长度 1/8/17/32：前向最大相对 L2 误差 `0.00380`，五类输入梯度最大 `0.00561`，最终 FP32 状态最大 `0.00244`，初始状态均保持不变。
这些检查直接调用接入后的 `delta_rule(..., training_fallback=...)`，而不只是绕开适配层调用 FLA，见 [边界与状态检查](validation/qwen38/training-audit-20261102/fla-boundaries.json)。

## 真实语料

数据来自 [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext)，固定版本 `b08601e04326c79dfdd32d625aee71d232d685c3`，配置 `wikitext-2-raw-v1`、`train` split；数据许可为 CC BY-SA 3.0 / GFDL。
源 Parquet SHA256 为 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`，与发布版本的 LFS 哈希一致，不是手写或循环拼接的 smoke 文本。

完整训练 split 的 36,718 行已用基模 tokenizer 转换：保留长度不少于 128 的 8,951 条记录，共 1,885,193 个 token；短行跳过，不补造 token，也不跨记录拼接。
训练数据读取器已对完整转换结果检查索引、随机 128-token 窗口和合法 K32 anchor。

NPU 短训从其中 8 条原文抽取四个不同窗口，来源行号为 9、3、17、10；anchor 分别为 76、46、44、45。
配置为全部 64 层 draft、forward KL、学习率 `1e-4`、CPU FP32 AdamW 和逐层 checkpoint。
四步两 rank 的 loss 和梯度范数完全一致，每步均覆盖 0–63 层；主模型完整参数 SHA256 未变，draft SHA256 改变，并成功导出 TP head。
这验证了真实语料的数据与更新链路，不代表已完成完整 epoch、预训练收敛或业务接受率验收。

| 步骤 | KL loss | 含教师前向的单步耗时 |
| --- | --- | --- |
| 1 | 5.300837 | 116.92 s |
| 2 | 8.582958 | 101.95 s |
| 3 | 4.922004 | 91.35 s |
| 4 | 4.177819 | 100.31 s |

各步来自不同文本和位置，loss 不必单调下降。当前检查的是 loss/梯度有限、参数确实更新，以及主模型保持冻结。
详细结果见 [离线记录](validation/qwen38/training-audit-20261102/offline.json) 和 [数据来源](validation/qwen38/training-audit-20261102/data.json)。

独立推理进程已重载本次语料训练的 head，head 哈希与离线导出一致；6 个请求覆盖 greedy/采样及重复请求，并继续完成 22 次 last-4 更新，两 rank 一致。
主模型参数哈希未变、历史缓存只读审计通过，见 [重载与在线续训](validation/qwen38/training-audit-20261102/reload.json)。

启用 FLA 后端后，另从完整的 8,951 条数据索引中随机取样，在真实 27B 上完成两步全 64 层更新。
两 rank 的 loss、梯度范数与层覆盖一致，主模型参数哈希未变，head 实际改变并导出；运行统计明确记录 `training_gdn_backend=fla`。
两步 KL 为 5.776670、6.020746；第一步含首次 JIT 编译为 287.40 s，第二步为 91.85 s。数据与运行时负载不同，不能据此与上述 Torch 四步直接比较整步性能。
记录见 [FLA 离线全层更新](validation/qwen38/training-audit-20261102/fla-offline.json)。

随后用独立进程加载这份 FLA 训练的 head，加载哈希与导出一致，并在 `training_gdn_backend=fla` 下完成 greedy、采样两个请求及 8 次在线 last-4 更新。
两 rank 的更新计数和 loss 一致，6 次历史只读审计通过；在线后 head 再次改变，主模型完整参数哈希仍未变，见 [FLA 重载与在线更新](validation/qwen38/training-audit-20261102/fla-reload.json)。

## 更新耗时

在验证 worker 中加入同步计时，不改变正常训练入口的执行方式。以 rank0 的步骤 2–4 为例：

| 阶段 | 测得范围 |
| --- | --- |
| Draft 重放前向、反向、逐层参数/梯度搬运 | 35.57–44.70 s |
| 全局梯度范数、TP 通信与裁剪 | 8.81–10.98 s |
| CPU AdamW 更新 | 35.38–40.87 s |
| 更新后的参数发布到 NPU | 6.28–9.50 s |

第一项包含 checkpoint 重算和 CPU/NPU 搬运，不能当作纯算子计算时间；各 rank 的耗时也包含同步等待。
这个少卡配置明显受 CPU 优化器与数据搬运影响，不能将单个 GDN 的提速等比例换算成训练整步提速。

另对单个 `5120×5120` 的 TP2 投影矩阵分别小测：CPU FP32 → NPU BF16（含转换）约 33.11 ms，梯度反向搬运约 20.68 ms，发布约 32.31 ms，CPU AdamW 约 149.26 ms。
这是单矩阵、非 pinned host 内存的独立测量，不是整步训练的搬运总时间。

## 使用入口

### 设备驻留训练

`--optimizer-device model` 将 FP32 master、梯度和 AdamW 状态留在对应 NPU；`--checkpoint-layers` 仅重算激活，不把参数卸载到 CPU。四卡 128-token 配置使用 `--no-checkpoint-layers` 避免重算，长序列按显存选择是否开启。在线 last-N 默认已采用设备驻留；离线全层训练需要显式选择该选项。

设备侧梯度范数使用现成的 `_foreach_norm` 批量归约，再执行原有 TP 归并和统一裁剪。TP4 投影、卷积及 norm 的混合梯度规格下，单项归约由 0.667 ms 降至 0.245 ms，结果一致；批量缩放梯度没有明显收益，仍保留原路径。该计时不包含反向、TP 通信或完整训练步骤，见 [设备小算子对照](validation/qwen38/execution-20261103/device-primitives.json)。

验证计时使用 `optimizer_step` 通用名称，并记录每个 rank 的实际优化器设备和 NPU allocator 峰值；新增的验证路径可在同一份真实梯度上对照旧归约，额外计时单列，不进入生产更新。反向阶段包含重算与 TP 通信，不能当作纯计算时间；allocator 峰值也不包含所有 CANN 外部工作空间。

### CPU Offload 执行

Qwen 原生 learner 的 CPU FP32 master 使用 PyTorch 已有的 fused AdamW，替代逐张量更新；设备上的优化器路径保持不变。在线 last-N 和离线全层选择同一个 learner，更新层数、loss 和 head 格式均未改变。

CPU 梯度范数先按最后一维归约，再累加平方和，避免为整个投影矩阵分配同尺寸的平方临时张量。TP 仍只统计一次复制参数，再归并各 rank 的平方和并统一裁剪。离线 KL 另外省去没有参与公式的 student 概率张量，TV 路径不变。

`scripts/audit_qwen38_execution.py` 使用 TP2 的投影矩阵规格比较 AdamW 后端，核对连续五步的参数、`exp_avg` 和 `exp_avg_sq`，再计时；同时比较梯度范数归约。单元测试另覆盖变化的梯度、last-1 / 全层参数选择、复制梯度计数、非有限梯度拒绝和双 rank 更新。

NPU fused AdamW 在本镜像中可执行且更快，但 moment 对照未满足本轮 `rtol=1e-5, atol=1e-8` 的严格门槛，因此没有默认替换设备侧优化器。这不等于该算子不可用或训练算法有问题；融合 AdamW 只在已通过对应数值检查的 CPU offload 路径启用，NPU 保持原有 AdamW。

双卡路径已复验两步真实语料全层更新、独立重载及八次 last-4 更新；主模型哈希未变、只读历史检查通过，见 [双卡训练](validation/qwen38/execution-20261103/offline-tp2.json) 与 [重载续训](validation/qwen38/execution-20261103/online-last4-tp2.json)。本轮 CPU 回归为 **653 passed、32 skipped**，见 [JUnit](validation/qwen38/execution-20261103/cpu.xml.gz)。跳过项不算设备验证通过。

### 命令

`scripts/prepare_qwen38_data.py` 将本地 JSONL/Parquet 转为 token JSONL，保留来源行号并生成哈希 manifest；使用本地基模 tokenizer，不依赖在线获取模型。
`scripts/train_qwen38_draft.py` 保持原离线训练入口，增加 `--timing` 可选验证计时；默认行为不变。
`scripts/audit_qwen38_training_ops.py` 复现已有算子的前后向比较，必须在指定运行时和空闲设备上执行。
可选库用 `ORTHRUS_QWEN38_FLA_PATH` 指定目录，只在 NPU worker 中延迟加载并校验设备；验证脚本的 `--runtime-cache` 将同版本编译缓存显式挂载进容器，避免引用容器外的宿主机路径。
本轮 CPU 回归为 **653 passed、32 skipped**，跳过项不算作设备测试通过。训练、算子审计和重载均采用限时实验进程，结束后清理容器并释放实验设备。
