# Qwen3.8 Orthrus 推理与在线更新

2026-10-30 更新：完整 27B 已在两张 NPU 上完成短请求推理和最后一层在线更新，见 [真机验证记录](QWEN38_REAL_VALIDATION.md)。下方 2026-10-29 的缩小模型及资源失败记录保留为历史；不代表当前仍未完成真实前向。

同日新增 [原生 vLLM 接入代码与启动入口](QWEN38_VLLM.md)，包括 TP 权重别名、验证回调和同步后缀更新。
它与本页的独立 Torch 路径不同：原生 TP2/K32、last-1/last-4 更新、head 重载和 OpenAI API 已通过真实 27B 功能检查，见 [验证记录](QWEN38_VLLM_VALIDATION.md)。原生服务部署优先阅读上述入口。

## 当前范围

首阶段实现文本混合架构的独立参考执行器，不覆盖已暂停的 DSV4 代码，不用 MTP 替代 Orthrus，也不依赖 PyPTO。它不是已经接好分页 cache、TP 和 OpenAI API 的 vLLM 服务。

常博当前公开主线支持普通 Qwen3，不能直接加载 Qwen3.8。这里沿用其 anchor 加 MASK 的并行起草、标准概率校正、实际到达行监督和冻结前段的后缀重放机制。Qwen3.8 的 GDN 扩展由我们负责适配，采用 Orthrus Qwen3.5 的参考定义：读取 AR 的卷积与递归状态，在临时副本上运行独立 draft mixer。**GDN 在块内仍按因果递推计算，不是把 GDN 改成双向 SDPA；双向可见作用于 Full Attention 层。** 程序用 `--hybrid-policy orthrus_qwen35_reference` 明确记录这项模型适配，不能将它描述为常博已经验证过的 Qwen3.8 实现。

参考来源：

- [常博的运行与在线更新配置](https://github.com/xchang1121/inference_scaling/blob/main/online-speculation/docs/RUNNING.md)
- [Orthrus Qwen3.5 混合架构参考](https://github.com/chiennv2000/orthrus/blob/main/src/models/modeling_orthrus_qwen3_5.py)

## 实现

`src/blockspec/parallel/qwen38.py` 复用 Transformers 的 Qwen3.5/3.8 文本层，不新增底层算子：

- embedding、层外 norm、MLP 和 LM head 共用原参数；每层 mixer 从 AR 复制一份独立 draft 参数。
- Full Attention 使用原有门控投影、Q/K norm、部分 RoPE 和 SDPA，draft 的显式 mask 允许整块读取严格 AR 前缀及整个临时块。
- GDN 使用原有卷积、门控递推与输出投影，读取 AR 前缀状态的副本。draft 临时状态全部丢弃，不提交给 AR。
- 主模型一次验证候选块。全部接受时保留 verifier 状态；发生拒绝时，从验证前快照重放实际接受的 AR 输入前缀，恢复正确的 KV、卷积和递归状态。递归状态不能像 KV 一样直接截短；这一步重放是正确性基线，尚非高效 checkpoint 回滚。
- `place_layers` 支持连续按层放到多个设备，也可显式放到 CPU。它不是 TP，不处理 Hugging Face 自动 offload hook；单设备测试通过也不能替代跨设备验证。

`parallel/generation.py` 仅增加可选的历史长度和提交接口，普通 Qwen3 保留原来的 KV 截断逻辑。候选采样、标准 rejection sampling 和在线反馈继续使用现有实现。

## 在线更新层数

**先默认最后 1 层，不把它当成最终最优值。** 常博公开示例为 `last_layers=1`、`stride=16`、`replay_blocks=1`、`learning_rate=1e-5`、`loss=tv`。Qwen3.8 第 64 层是 Full Attention，适合作为首个在线闭环；截断边界以前不需要 GDN 反向，后缀内冻结的 MLP、norm、LM head 仍需输入梯度。

直接复用 `SuffixLearner`：仅为选中的 draft 后缀保留 FP32 master 和 AdamW 状态；从 verifier 收集实际到达位置的完整词表 logits；detach 前段表示，重放后缀并更新；本轮校正结束后发布参数。不会根据未到达的候选路径训练，也不更新 AR 参数。

最后 1 层与最后 4 层的真实 27B 原生更新均已通过功能检查，性能收益比较仍待完成。最后 4 层包括三个 GDN 和一个 Full Attention，收益必须覆盖额外训练成本。反向通过不代表已获得接受率或吞吐收益。服务并发中的 head 版本切换、预算控制和候选发布门控尚未实现。

## 运行

测试环境为 Transformers 5.14.1、PyTorch 2.10.0 + torch_npu 的现有实验镜像。可先运行不依赖模型下载的缩小结构；保留 Full Attention head_dim=256、GDN head_dim=128，不能据此估计 27B 性能。

```bash
python -m pip install -e '.[dev,qwen38]'
PYTHONPATH=src python scripts/run_qwen38_orthrus.py \
  --tiny --device npu:0 --dtype bfloat16 \
  --block-size 32 --tokens 40 --online --last-layers 1 --stride 4 \
  --hybrid-policy orthrus_qwen35_reference \
  --output local/qwen38-reference.json
```

`--temperature 1` 检查采样校正路径，`--last-layers 4` 检查含 GDN 的后缀更新。不同采样执行不能要求同一随机种子输出逐 token 相同；统计分布检验需单独完成。

真实 checkpoint 使用 `--model /data/disk/models/Qwen3.8-27B` 替代 `--tiny`，显式传入训练约定的 `--mask-token-id`。模型只从本地加载，不执行远程代码。`--device npu:0 --layer-devices npu:0,npu:1` 是连续按层分配，真实 27B 已在 1、2 号卡完成该路径的短请求检查；`--layer-devices npu:0,cpu` 将后一段保留在 CPU，仍只有缩小模型的成功记录。它们不是已验收服务部署命令；当前执行器也不加载 W8A8 部署权重。

## 下一阶段

真实 27B 的原生 vLLM 服务与同步在线更新已完成首轮功能验收。原生入口现已从 target Full Attention KV/GDN 状态提取接受前缀，不再重放 AR 历史；目前通过连续布局快照连接现有 draft 算子，尚非零拷贝分页读取。后续需要训练 head 的交付与收益验证，再处理批量调度和预算控制。上述独立 Torch 参考路径仍保留重放恢复，不应与原生入口混为一谈。已有的 DSV4 proposer 仍绑定该模型的缓存结构。

## 验证记录（2026-10-29）

CPU 回归共 106 项通过、4 项 CUDA 专属测试跳过。包含普通 Qwen3 原有生成与采样回归，以及新增混合模型的 AR/Transformers 数值对齐、分块 AR、draft 历史只读、K=2/4/8/32、后缀重放对齐、拒绝后全历史重算审计、全接受提交、参数冻结、优化器恢复、固定样本 loss 下降及 RoPE 精度保护。

Ascend 910B3 上，缩小混合模型完成了以下实测，均使用真实 Full Attention/GDN head 宽度，但不使用 27B 的层宽和权重：

| 配置 | 实际结果 |
| --- | --- |
| BF16、K=32、greedy、更新最后 1 层 | 40 token 与 AR 相同；通过状态审计并发生在线更新 |
| BF16、K=32、temperature=1、更新最后 4 层 | 经历接受与拒绝，通过状态审计；完成含 GDN 的一次在线更新 |
| BF16、K=32、NPU＋CPU 分层、更新最后 1 层 | 40 token 与 AR 相同；完成 5 次更新 |

这些是功能检查，不是接受率、采样分布或吞吐验收。当前环境明确回退到 Transformers 的 Torch GDN 实现；没有将其描述为已有的高性能 NPU GDN 后端。实验使用本地镜像 `sha256:b55dc48044273052387facc05879e512f9b33bb6e882cafcb145e4988f73f2d5`。

2026-10-29，真实 Qwen3.8-27B 的 BF16 checkpoint 已成功加载，独立 draft mixer 已复制，但三次尝试均在搬权重阶段因共享设备被其他任务占用而 OOM：单卡＋CPU 时本进程仅分配约 2.37GiB 就只剩约 1.95GiB；双卡时本进程分配约 29.10GiB 后设备耗尽；进一步缩减为 NPU 前 8 层、CPU 后 56 层时，搬第一个 embedding 前设备仅剩约 907MiB。当时未完成真实前向或 vLLM 服务验收；2026-10-30 的成功结果见文首链接。实验容器以 `--rm` 运行，结束后不保留占卡进程，不停止他人任务。

缩小模型的原始结果：[NPU K32 greedy](validation/qwen38/npu-k32-greedy-last1.json)、[NPU K32 采样及 GDN 反向](validation/qwen38/npu-k32-sampling-last4.json)、[NPU＋CPU 分层](validation/qwen38/npu-cpu-k32-greedy-last1.json)。
