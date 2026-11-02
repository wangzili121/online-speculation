# Qwen3.8 Draft 训练算子与真实语料验证

验证目标是冻结 Qwen3.8-27B 主模型，只更新 draft 参数，并复用已有高效算子；不开发 kernel、不改变 Attention/GDN 的数学定义。
环境为两张 Ascend 910B3、TP2、K=32，Torch 路径执行代码为 `51d7464`，可选 FLA 接入为 `b078413`。
镜像固定为 `quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a`，Torch-NPU 为 `2.10.0.post4`，Triton-Ascend 为 `3.2.2`。

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

`scripts/prepare_qwen38_data.py` 将本地 JSONL/Parquet 转为 token JSONL，保留来源行号并生成哈希 manifest；使用本地基模 tokenizer，不依赖在线获取模型。
`scripts/train_qwen38_draft.py` 保持原离线训练入口，增加 `--timing` 可选验证计时；默认行为不变。
`scripts/audit_qwen38_training_ops.py` 复现已有算子的前后向比较，必须在指定运行时和空闲设备上执行。
CPU 回归为 **636 passed、32 skipped**，跳过项不算作设备测试通过。
