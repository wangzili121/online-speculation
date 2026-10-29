# Qwen3.8-27B 真机验证

日期：2026-10-30。使用完整 Qwen3.8-27B BF16 权重、64 个文本层、原始隐藏维度及词表，不缩小模型。执行设备为两张 Ascend 910B3 64GB，前后各放置 32 层；这是按层放置，不是 TP，也不是 vLLM OpenAI API 服务。

draft mixer 从每层 AR mixer 独立复制，embedding、层外 norm、MLP 与 LM head 共享并冻结。当前 draft 未经离线训练，MASK token 1 仅用于本轮初始化链路检查，不代表最终训练 head 的约定。实验不验证加速收益。

## 已验证

| 检查 | 真实模型结果 |
| --- | --- |
| 原生 HF 前向对照 | 同一 prompt 的 logits 逐元素相同，最大绝对差 0 |
| Greedy 生成 | AR、固定 draft、在线更新后的 8 个输出 token 完全一致 |
| 完整 K32 | 实际处理 32 个 query，生成 31 个候选，接受 1 个后在第 2 个候选处拒绝 |
| draft 历史隔离 | 完整 K32 draft 前后的历史 KV、卷积与递归状态逐元素相同 |
| 拒绝恢复 | 提交的 AR 状态与相同分块的独立纯 AR 重放逐元素相同 |
| 在线最后一层 | 只优化第 64 层 draft mixer 的 104,858,112 个参数；短请求完成 3 次更新 |
| 参数更新与冻结 | 短请求中 32,437,424 个 draft 执行元素发生变化；冻结参数的版本计数未变 |
| 采样模式 | temperature=1 完整 K32 提议与拒绝恢复通过；随后生成 8 token，另完成 2 次在线更新 |

三次在线反馈的 TV loss 为 0.52183、0.49257、0.51695，梯度范数均有限。这些反馈来自不同位置，不能据此宣称 loss 收敛、接受率提高或推理加速。它证明真实大模型的后缀反向、优化器及参数发布实际执行了，而非仅更新计数。

## 缓存检查的数值对照

首次真机尝试已成功完成 AR，但在“分块缓存必须等于整段重算缓存”的检查中失败。随后禁用 draft，只用纯 AR 对照：同样会出现分块与整段重算差异，深层状态最大绝对差达到 0.125，所列深层状态相对 L2 约 0.0037–0.0067；相同分块的纯 AR 重放则逐元素相同。

因此，缓存污染检查改为独立重放相同 AR 输入及分块顺序，仍要求 `rtol=0, atol=0`，没有放宽容差。整段重算的差异另行保留为诊断数据。新增单测主动破坏 Full Attention KV、GDN 卷积和递归状态，确认审计确实会报错。

这不保证所有 prompt 在不同低精度分块策略下都能得到相同 argmax，更不能用本轮短请求代替大规模精度验收。

CPU 回归 111 项通过、4 项 CUDA 专属测试跳过。原始结果包含完整模型维度、7,239,780,864 个 draft 参数、逐轮接受长度、更新 loss/梯度及 token ID：[完整 K32 与采样](validation/qwen38/qwen38-27b-real-1030-k32.json)、[首轮短请求](validation/qwen38/qwen38-27b-real-1030-short.json)。实验完成后容器自动删除，没有继续占卡。

## 复现

项目目录下执行；模型和设备按实际环境指定：

```bash
PYTHONPATH=src python scripts/run_qwen38_orthrus.py \
  --model /data/disk/models/Qwen3.8-27B \
  --device npu:1 --layer-devices npu:1,npu:2 \
  --reserve-memory-gb 44 --dtype bfloat16 \
  --mask-token-id 1 --block-size 32 --prompt Hello --tokens 8 \
  --online --last-layers 1 --stride 1 \
  --native-reference --diagnose-numerics --probe-block \
  --sampling-smoke-tokens 8 \
  --hybrid-policy orthrus_qwen35_reference \
  --output local/qwen38-27b-real.json
```

`--reserve-memory-gb` 在加载 CPU 权重前为实验分配显存，放置权重前释放临时张量并由同一 allocator 复用；进程结束后全部释放，不是独占锁，也不会停止其他任务。资源已不足时应报错退出。复现环境沿用实验镜像 `sha256:b55dc48044273052387facc05879e512f9b33bb6e882cafcb145e4988f73f2d5`，Transformers 5.14.1、PyTorch 2.10.0 与 torch_npu。

## 边界

本轮是独立 Torch 混合架构执行器的真机功能验证。当前 GDN 使用 Transformers 的 Torch 回退实现，严格审计还包含额外 AR 重放，结果中的 TPS 不用于性能结论。量化 head、vLLM 分页/递归状态接入、并发调度、图模式、长上下文、大样本采样分布及真实业务接受率仍需单独验收。
