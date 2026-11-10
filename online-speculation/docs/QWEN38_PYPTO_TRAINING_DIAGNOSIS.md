# Qwen3.8 PyPTO：真实反馈数值对照与固定批次训练

初始化 head 的短训练没有提高接受长度，同时 loss 未呈现明确下降。本诊断先检查真实数据上的训练计算，再判断训练范围、数据量和收敛问题。

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
