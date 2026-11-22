# Orthrus-DSV4.1 PyPTO 在线更新实现边界

## 当前可用部分

双机 PyPTO 已经完成 DSV4.1 W8A8 的 target 推理、K32 draft、整块 verify、接受前缀提交和拒绝回滚；初始化 head 下，双机 16 卡的 greedy 输出与 target-only 逐 token 相同。最后一层的 Torch 参考学习器也已具备：它从同一 anchor 重放 C1A Reuse 层，冻结 W8A8 主模型、mHC、MoE、Norm 和 LM Head，只更新最后一层 diffusion Attention，并支持 TV/KL、梯度裁剪和 head 导出。

GSM8K 数据固定、延迟教师反馈状态机、两阶段更新频率、学习率 warmup、发布边界和结果汇总已经独立实现并通过单元测试。Qwen3.8 的参数只作为候选起点，DSV4.1 会在独立 tuning 集扫描教师行数、学习率和更新频率。

## PyPTO 还要补的反向链路

这里缺的不是普通 Attention 前向，而是最后一层从教师 logits 回到 diffusion Attention 参数的完整反向。模型前向已经能复用现有算子；在线更新需要以下设备计算按顺序连起来：

| 计算 | 输入与输出 | 可复用部分 | DSV4.1 专属工作 |
| --- | --- | --- | --- |
| 完整词表 TV/KL | draft/teacher logits → loss、`dlogits` | Qwen 在线实现的分块 softmax、TV/KL 公式 | 改为 DSV4.1 词表和 TP4 分片，保留真实教师行掩码 |
| LM Head 与末层共享模块 VJP | `dlogits` → 最后一层 Attention 输出梯度 | 长矩阵乘、RMSNorm、SwiGLU 基础梯度 | 冻结 W8A8 LM Head、mHC 和 MoE，只计算输入梯度；Top-K 路由按前向固定 |
| C1A Reuse Attention 反向 | 输出梯度 → `dQ/dKV/dO` | matmul、softmax、RoPE、norm 基础梯度 | 同时覆盖 SWA 历史、C1A 压缩历史、sink/LSE 合并和 K32 临时双向块；历史 cache 只读 |
| 参数梯度与 AdamW | `dQ/dKV/dO` → FP32 master/moments | Qwen 的统计、clip、AdamW、BF16 发布公式 | 按 DSV4.1 TP4 的 Q/KV/O shard 布局生成参数梯度 |
| 双机版本发布 | 新 head → 16 个执行 rank | 已有双机控制链路和驻留权重句柄 | DP4 的同一 verifier 边界切换同一版本；失败时所有 rank 保留旧版本 |

因此，当前不能把 Torch/CPU 学习器称为“PyPTO 在线训练已完成”。它已经证明训练目标、梯度范围和权重格式可行；最终高性能版本仍需补 C1A/mHC/MoE 的模型专属反向，以及 TP4/DP4 原子发布。普通权重加载、数据下载和主模型推理不需要重写成训练算子。

## 明日验收门槛

先用 4～16 道 GSM8K 完成 fixed-head smoke，确认真实文本解码、greedy 一致、接受统计和吞吐。在线版本只有同时满足以下条件才算跑通：

1. 教师 logits 来自主模型沿拒绝后修正的真实前缀，不能来自错误候选分支。
2. 只有最后一层 diffusion Attention 参数变化；主模型和共享模块的参数哈希不变。
3. loss、梯度范数和版本号有限，发布前后 16 个 rank 的 head 哈希一致。
4. 开启在线更新后仍与 target-only greedy 输出逐 token 相同。
5. 报告接受长度、更新/发布时间和端到端吞吐，不能只报告 loss 下降。
