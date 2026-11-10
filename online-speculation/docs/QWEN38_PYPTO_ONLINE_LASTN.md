# Qwen3.8 PyPTO 在线 last-N 更新

PyPTO Serving 在请求验证后采集教师分布，只更新末尾指定层的 draft 参数。主模型、共享 MLP/norm/LM head 和目标历史状态不更新。推理仍使用因果 GDN、块内双向 Full Attention；更新采用已有训练路径的 TV 目标。

## 更新范围

`--last-layers` 支持 1、2、3、4，默认 1：

| 配置 | 更新的层 | 参数组 |
| --- | --- | --- |
| 1 | 第 63 层 Full Attention | Q/K/V/O、Q/K norm，共 6 组 |
| 2 | 第 62 层 GDN + 第 63 层 | 15 组 |
| 3 | 第 61–63 层 | 24 组 |
| 4 | 第 60–63 层 | 33 组 |

每层 GDN 更新 QKV、Z、A/B 投影、四 tap 卷积、A_log、dt_bias、gated RMSNorm 权重和输出投影。GDN norm 使用原权重，Full Attention Q/K norm 使用 `1 + weight`；A_log 保持 FP32。共享模块的参数冻结，但仍计算其输入梯度，以把梯度传回前面的 draft 层。最前一个更新层的输入和全部 AR 历史作为常量。

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

last-1 保留原 `OnlineLearner` 路径；last-2/3/4 使用 `SuffixLearner`。后段计算分成若干 CHIP 入口，避免超过固定运行时的参数数量限制。FP32 master、moment 和梯度采用同一个扁平布局，执行时仍按每层原有 bank 形状读取。

64 GB A2 使用独立学习卡：推理卡运行主模型和 draft；学习卡保存后段所需的冻结权重及学习状态。两张卡通过有界主机缓冲传递字节，数值计算在 NPU 上完成。这是两个独立 worker，主模型仍为 TP1。last-2/3/4 要求指定学习卡。

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

组件验证覆盖真实 GDN 头数、部分块、非零历史、解析反向、完整 last-2/4 后段梯度、两次 Adam 更新和 checkpoint 精确恢复。真实 27B 的 last-2 已通过两块反馈联合更新、下一轮生效、目标 cache 不变、greedy 对齐、scheduler 及恢复后的 HTTP 更新。

合成 last-4 用例中，BF16 前向微差使一个位置的 TV 正负号变化。逐节点使用相同输入对照后，最大局部相对误差约 `1.84e-4`；使用同一 TV 梯度做独立 autograd 对照，全部 33 组参数通过，最大相对误差约 2.3%。验证没有修改 TV 目标或放宽梯度阈值。

真实 last-4、冷启动及导出后的独立服务检查继续补充。具体范围和结果见[验证记录](validation/pypto-qwen38/lastn-20261110/README.md)。当前服务配置为 K32、512 个总位置、greedy 和一个活动请求；短请求功能验证不用于评价收敛或加速收益。
