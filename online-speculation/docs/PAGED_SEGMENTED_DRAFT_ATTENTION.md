# Orthrus Batched Paged-Segmented Draft Attention

## 问题

Orthrus 的 draft 分支复用 AR backbone，但每层使用独立双向 attention。原实现
对 active batch 中每个请求依次执行：从 vLLM paged KV gather 完整历史、拼接
32-token 临时 draft KV、调用一次 dense attention。native GQA 消除 head
expansion 后，这个 request loop 仍会产生 `layers × active_requests` 次小 attention
调用，并重复 materialize 最长 40K 的历史 KV。

## 精确分解

每个 draft query 的 key/value 集合严格由两段组成：

```text
AR history in vLLM paged KV | local 32-token bidirectional draft KV
```

分别计算两段的 partial attention output `O_h/O_l` 和 softmax log-sum-exp
`L_h/L_l`，然后执行：

```text
L = logaddexp(L_h, L_l)
O = exp(L_h - L) * O_h + exp(L_l - L) * O_l
```

这与在拼接 KV 上执行一次 softmax attention 数学等价。历史段使用
`npu_fused_infer_attention_score` 的 paged block-table 接口；local 段使用同一
算子的 dense TND 接口。多个请求以 cumulative query lengths 和每请求 KV
lengths 合并成两个 variable-length TND 调用。

## 语义边界

- 不稀疏、不丢弃 KV page，不近似 attention。
- 不改变 K32、draft logits、target verifier 或 exact rejection sampling。
- FP32 合并 LSE，输出再转回原 attention dtype。
- `native_gqa` 保留为 B1 control；`expanded` 保留为历史 reference。
- 环境变量 `ORTHRUS_DRAFT_ATTENTION=paged_segmented` 显式启用。
- `ORTHRUS_DRAFT_SEGMENT_THRESHOLD=24576` 只控制 batch width 为 1 时的门槛。

## 验证

`scripts/benchmark_segmented_draft_attention.py` 验证单请求、非整块历史的数值
等价与长度拐点；`scripts/benchmark_batched_segmented_draft_attention.py` 验证
混合长度 block table、跨请求 TND 合批、端到端算子时间和临时显存。

正式验收使用固定 64-request workload，分别比较 `native_gqa` 与
`paged_segmented` 的 output token/s、SLO goodput、P95 TTFT/TPOT、KV
preemption，并采集独立 Torch-NPU trace，确认 `Gather/Concat`、attention call
数量与 NPU free time 同时下降。

正式 64-request 结果已经达到 34.30 output token/s，相对 native-GQA B1 的
25.00 token/s 提升 37.17%；P95 latency 下降 26.56%，并从 1 次 preemption
降为 0。Torch-NPU trace 中历史 attention 从每层平均 23 次降到 1 次，KV
materialization 从 19.22% 降到 0.67%，NPU busy 从 52.23% 升到 87.19%。
