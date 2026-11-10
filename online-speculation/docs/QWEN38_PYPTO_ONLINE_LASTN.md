# Qwen3.8 PyPTO 在线 last-N 更新

PyPTO Serving 在请求验证后采集教师分布，只更新末尾指定层的 draft 参数。主模型、共享 MLP/norm/LM head 和目标历史状态不更新。推理仍使用因果 GDN、块内双向 Full Attention；更新采用已有训练路径的 TV 目标。

## 更新范围

`--last-layers N` 可选择该模型末尾 1–64 层，默认 1。可运行的 N 还受显存和固定运行时容量限制；表中层号从 0 开始：

| 配置 | 更新的层 | 参数组 |
| --- | --- | --- |
| 1 | 第 63 层 Full Attention | Q/K/V/O、Q/K norm，共 6 组 |
| 2 | 第 62 层 GDN + 第 63 层 | 15 组 |
| 3 | 第 61–63 层 | 24 组 |
| 4 | 第 60–63 层 | 33 组 |
| 5 | 第 59 层 Full Attention + 第 60–62 层 GDN + 第 63 层 Full Attention | 39 组 |
| 8 | 第 56–63 层，包含两组 GDN/GDN/GDN/Full Attention | 66 组 |

每层 GDN 更新 QKV、Z、A/B 投影、四 tap 卷积、A_log、dt_bias、gated RMSNorm 权重和输出投影。GDN norm 使用原权重，Full Attention Q/K norm 使用 `1 + weight`；A_log 保持 FP32。共享模块的参数冻结，但仍计算其输入梯度，以把梯度传回前面的 draft 层。最前一个更新层的输入和全部 AR 历史作为常量。跨过四层组边界时，每个 Full Attention 层读取自己对应的历史 KV；只在整个末段最后计算一次 LM head 和 TV loss。中间层复用相同的前向、输入梯度和参数梯度算子。

## 运行流程

```text
NeuralDraft.propose
  → 在第 64-N 层前保存输入，并复制 anchor 前的 AR 历史
  → 正常起草；verifier 采集实际到达位置的教师 logits
SuffixLearner.observe / update
  → 重放末 N 层，按全部 replay 块的有效位置总数计算 TV loss
  → TrainingWorkspace 执行显式反向，累计全部 draft 参数梯度
  → SuffixOptimizer 裁剪梯度、计算 Adam 候选并检查有限值
  → SuffixPublisher 搬运备用执行参数，DraftWeights.publish_many 整组切换版本
```

last-1 保留原 `OnlineLearner` 路径；N≥2 使用 `SuffixLearner`。后段计算分成若干 CHIP 入口，避免超过固定运行时的参数数量限制。FP32 master、moment 和梯度采用同一个扁平布局，执行时仍按每层原有 bank 形状读取。

64 GB A2 使用独立学习卡：推理卡运行主模型和 draft；学习卡保存后段所需的冻结权重及学习状态。两张卡通过有界主机缓冲传递字节，数值计算在 NPU 上完成。这是两个独立 worker，主模型仍为 TP1。N≥2 要求指定学习卡。

推理卡需要两份末段执行参数，学习卡另需 FP32 参数、Adam 状态、梯度、冻结共享权重和反向中间值。以当前真实尺寸、两个 replay 块估算，last-5 在推理卡额外占约 2.14 GiB，last-8 约 3.46 GiB，均不含预留余量。程序在分配大缓冲前检查两侧剩余显存；固定运行时每个 worker 的 CHIP 入口上限为 64，也会提前检查。配置范围不代表所有 N 都能在当前两张 64 GB A2 上运行。

## 启动、恢复与导出

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python scripts/serve_qwen38_pypto.py \
  --source /path/to/OneServing --model /models/Qwen3.8-27B \
  --head-dir /models/orthrus-pypto-head \
  --learner-head-dir /models/orthrus-raw-head \
  --device 0 --learner-device 1 \
  --attention-backend paged --gdn-backend vector \
  --draft-group-size 4 --head-transfer packed \
  --online --last-layers 4 --update-stride 16 --replay-blocks 2 \
  --learning-rate 1e-5 --checkpoint-dir /data/online-heads
```

首次初始化需要原始 BF16 Q/K norm，不能从推理 bank 的 `1 + weight` 精确逆推。若 `--head-dir` 直接指向原始训练 head，可省略 `--learner-head-dir`。

重启恢复时增加 `--resume-online /data/online-heads/step-00000001`，并保留相同初始化 head 和 `--last-layers`。last-1 使用 v1 checkpoint；多层使用 v2，记录完整参数布局、基模/head 哈希、FP32 master、Adam moment 和步数，拒绝不同层数或不同 head 的状态。

仅用于固定 head 推理时，可以导出标准 PyPTO head：

```bash
python scripts/prepare_qwen38_pypto_head.py \
  --model /models/Qwen3.8-27B --head-dir /models/orthrus-pypto-head \
  --device 0 --online-checkpoint /data/online-heads/step-00000001 \
  --output-dir /models/orthrus-updated-head
```

导出只替换更新层的 bank，保留其他层和源 head；新目录可直接作为 `--head-dir`。继续训练仍需对应 optimizer checkpoint。现有 PyTorch 在线 last-N 和离线全量训练入口保持不变。

## 验证

真实尺寸的独立 autograd 对照曾发现并修复 LM-head 输入梯度的长行读取错误。修复后 last-1 / last-4 的全部参数梯度、八块累积、Adam 和执行参数转换通过，固定反馈 loss 正常下降；见[数值诊断与修复](QWEN38_PYPTO_TRAINING_DIAGNOSIS.md)。以下早期功能记录不替代这项数值验收，也不用于判断训练效果。

组件验证覆盖真实 GDN 头数、部分块、非零历史、解析反向、完整 last-2/3/4/5/8 后段梯度、两次 Adam 更新和 checkpoint 精确恢复。488 项 CPU 回归通过，包含 last-2/3/4/5/8 的独立 autograd 对照，以及 64 层图结构、跨组输入采集和显存拒绝检查。

真实 27B 的 last-2、last-3、last-4、last-5 均通过两块反馈联合更新、下一轮生效、目标 cache 不变、greedy 对齐、scheduler 及恢复后的 HTTP 更新。四种配置分别在新进程中恢复各自的 checkpoint，再执行相同更新；所得参数和两组 Adam 状态与连续运行逐字节一致。last-3/4/5 导出后的 24/33/39 组执行参数与 checkpoint 精确一致，未更新区域保持原字节；新 head 独立启动服务后，completion、chat、SSE 均通过。

合成 last-4 用例中，BF16 前向微差使一个位置的 TV 正负号变化。逐节点使用相同输入对照后，最大局部相对误差约 `1.84e-4`；使用同一 TV 梯度做独立 autograd 对照，全部 33 组参数通过，最大相对误差约 2.3%。验证没有修改 TV 目标或放宽梯度阈值。

last-5/8 的独立 NPU 后段、Adam 和 checkpoint 对照已通过；last-5 还完成了真实 27B 的更新、冷恢复、导出及独立服务验证。last-8 尚未完成真实 27B 部署验收。跨组结果见[扩展验证记录](validation/pypto-qwen38/lastn-general-20261110/README.md)。

具体范围和结果见[验证记录](validation/pypto-qwen38/lastn-20261110/README.md)。当前服务配置为 BF16、TP1、K32、512 个总位置、greedy 和一个活动请求；学习卡独立运行，不构成 TP2。短请求功能验证不用于评价收敛或加速收益。
