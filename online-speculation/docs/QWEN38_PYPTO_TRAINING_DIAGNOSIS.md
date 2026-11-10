# Qwen3.8 PyPTO：真实尺寸反向诊断与修复

**已确认并修复 LM-head 输入梯度的长行读取错误。** 真实 27B、相同反馈和相同初始化下，修复前固定批次训练发散，修复后 last-1 / last-4 均能降低 loss。此前冷启动实验不能用来判断在线更新或训练配方无效。

## 问题与修复

LM head 虽然冻结，反向仍需用它把 logits 梯度传回 draft：`dHidden = dLogits @ lm_head`。实际形状为 `[32, 248320] @ [248320, 5120]`。原实现直接对超长行做二维切片读取，第一行正确，后续行错误；输入中未监督的行全部为零，输出却不为零。这类错误没有被小尺寸用例及“梯度非零、参数有变化”检查发现。

逐节点使用相同输入对照后，首个错误定位在上述矩阵乘；相同梯度下的 Adam 与执行参数转换通过。修复将 logits 梯度按连续行片段重排、转置，再调用已有矩阵乘，避免超长行的二维读取。公式、完整词表、梯度归一化及训练目标均不改变。改动位于 `training_programs.py::make_linear` 和 `training_long_linear.py`，只影响 PyPTO 手写反向，不影响 vLLM 的 Torch autograd。

独立真实尺寸回归覆盖实际梯度、单个有效行、词表最后一列与未监督零行。修复后实际梯度相对 L2 为 `0.0143%`，余弦相似度 `0.99999999`，无效行输出严格为零；原实现相对 L2 为 `268%`。

## 真实反馈结果

下表使用同一批 8 个真实反馈块、同一份初始参数，学习率 `1e-5`，NPU 固定训练 128 步。共享权重、历史和教师固定；训练执行生产 learner 的融合前后向、梯度累积及 Adam。该单卡重放检查的是训练计算，跨设备发布与后续请求由完整模型用例另行检查。

| 检查 | last-1 | last-4 |
| --- | --- | --- |
| 固定批次 loss：初始 → 128 步后 | `7.16994 → 0.014185` | `7.16897 → 0.002907` |
| 首块参数梯度与独立 autograd 对照 | 6 组全部通过；最大相对 L2 `0.66%` | 33 组全部通过；最大相对 L2 `2.96%` |
| 八块累计参数梯度 | 6 组全部通过；最大相对 L2 `0.33%` | 33 组全部通过；最大相对 L2 `1.45%` |
| 首次 Adam 的 master / 一阶 / 二阶状态 | 同梯度对照全部通过 | 同梯度对照全部通过 |
| 执行参数与 master 转换 | 逐值一致 | 逐值一致 |

梯度判据预先固定为相对 L2 `<6%`、余弦相似度 `>0.99`；未因本次结果调整。Torch 只使用前向与 autograd，不调用 PyPTO 手写 VJP。513 项 CPU 回归通过。原始报告、故障版本对照及命令见[验证归档](validation/pypto-qwen38/training-diagnosis-20261110/README.md)。

这证明真实尺寸后段可以正确求导并拟合同一批反馈，不代表留出集接受率提高或已经获得加速；这些指标需要修复后的独立实验。旧 PyPTO 实验的 head 和 Adam 状态含有错误梯度的影响，重新对照应从相同的干净初始化开始。

## 完整模型复验

真实 27B、独立推理卡与学习卡上，last-1 另完成 16 次不同真实批次的 KL 更新和 12 个留出请求的 AR 配对对照。版本连续到 16，三次同批 loss 重算均下降；所有配对输出逐 token 一致，结束时请求队列、KV 占用和错误计数均为零。该短测 test 接受候选数从 `1.00` 变为 `1.03` / 轮，固定 head 吞吐为 AR 的 `0.71×`，尚未获得净加速。

last-4 使用 K32 和原有 reached-prefix TV 目标，完成两块反馈联合更新、下一轮参数生效、主模型历史不变及 greedy 对齐检查。随后经 scheduler 更新到版本 3，重建优化器并从 checkpoint 恢复，再通过 HTTP 请求继续更新到版本 4。全部检查通过。具体命令与结果见验证归档中的 `serving-fixed-last1-r1` 和 `online-tv-fixed-last4-r1`。

当前训练图的物理行缓冲为 32，K≤32 通过块长和有效行数配置；这不代表已经支持 64 行缓冲。额外的 64 行探索未通过，已在长词表 VJP 入口增加前置拒绝，避免误用未验证的扩展路径。

## 对照方法

从真实 27B 的一次训练中保存 8 个块的实际输入：冻结前缀输出、AR 历史、教师 logits、初始 draft 参数，以及 PyPTO 的前向输出、梯度和第一次 Adam 状态。两个实现使用同一份快照，避免教师、历史或 anchor 不同影响结果。

| 检查 | 参考与判据 |
| --- | --- |
| 前向 | Torch 重新计算 logits，对照服务起草及训练重算输出；记录相对 L2 |
| 反向 | Torch 只执行前向，使用 autograd 求导；逐参数核对首块梯度和 8 块累计梯度 |
| Adam | 给 Torch Adam 与 PyPTO 同一份实际梯度，独立核对裁剪、FP32 master 和两个 moment |
| 参数发布 | 有执行参数快照时，核对设备 master 按执行格式转换后的值；未采集时显式注明 |
| 固定批次 | 两边分别从相同初始化开始，反复训练相同 8 块，记录初始、逐步和最终 loss |

8 块共 56 个监督位置，每块 loss 均除以同一个 56。冻结前缀、历史与教师固定，训练只重放待更新后段。固定数据能排除抽样变化；是否泛化到新请求要在这项诊断通过后另行检验。

## 数据与执行格式

使用已记录的 GSM8K 训练问题 AR-only 回答，复用原实验的第一批随机 anchor，K=8。检查 last-1 和 last-4，学习率 1e-5，固定批次最多 128 步。

Torch 参考使用普通矩阵乘、softmax、norm、卷积和 GDN 前向，梯度全部由 autograd 计算，不调用手写 VJP。FP32 master 的前向值按 BF16 执行格式舍入；Q/K norm 先舍入原始权重再加 1，A_log 保持 FP32。CPU 仅用于独立数值参考，实机训练与 Adam 仍在 NPU 上执行。

参考只计算有效 8 行；其与物理 K32 缓冲的活动行结果、梯度和全批归一化有独立小尺寸测试。冻结大矩阵的 FP32 副本在参考进程中复用，避免每次重放重复转换。

## 运行

在固定 OneServing / PyPTO 环境中采集并执行 NPU 固定批次训练：

```bash
PYTHONPATH=src python scripts/diagnose_qwen38_pypto_training.py \
  --source /path/to/OneServing --model /models/Qwen3.8-27B \
  --data /data/gsm8k --records-report /results/online-last1-k8-128.json \
  --device 0 --learner-device 1 --last-layers 4 --steps 128 \
  --learning-rate 1e-5 --output /results/fixed-last4.json
```

快照保存在 `fixed-last4.tensors`。首批完成后生成的 `manifest.json` 不再修改，可在独立 CPU 进程中运行参考：

```bash
PYTHONPATH=src python scripts/diagnose_qwen38_pypto_training.py \
  --phase reference --capture-report /results/fixed-last4.tensors/manifest.json \
  --steps 128 --threads 16 --output /results/reference-last4.json
```

Tensor 快照包含真实尺寸冻结矩阵与优化器状态，需要数 GB 至数十 GB 磁盘和主机内存。仓库保存代码、精简报告与哈希，不提交大权重文件。

## 结果解释

数值通过与训练收敛分别记录。Torch 能降低固定批次 loss 而 PyPTO 不能时，继续检查适配计算；两边固定批次均可学习而留出集没有收益时，再研究训练覆盖、数据和泛化。小尺寸测试或参数发生变化，均不能替代真实反馈上的数值对照。
