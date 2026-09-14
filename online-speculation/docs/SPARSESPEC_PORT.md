# SparseSpec 在 Orthrus/vLLM-Ascend 上的移植与实验

## 结论先行

SparseSpec 的开源实现不能直接用于当前 910B3 服务。上游固定为 H100、CUDA、
定制 FlashInfer 和 JIT CUDA kernel；仓库自身也将其标为 research proof of
concept。可复用的是算法与系统设计，不是二进制或 CUDA 源码。

本仓库已完成第一版 Ascend 兼容移植：完整 verifier 保持不变，draft 通过
Paged KV 的 block-table 子集只读取一部分历史页。由于 Ascend Qwen GQA 的
FIA 不暴露逐 token attention score，当前 `verifier` 策略用最后一层最新
verifier query 与每页末尾 key 的相关性选择历史页，并将同一页集合复用于
32 层；`recent` 策略作为不做动态选择的对照。

这条路径保持 speculative rejection correction 不变，因此最终目标分布仍然
无损；页选择只会影响 draft 接受率与性能。

## 为什么不能逐项照搬

SparseSpec 和 Orthrus 都属于 self-speculation，但 draft 工作负载不同：

- SparseSpec 每次用同一个模型和 sparse attention 生成一个 draft token，连续
  多个 draft phase 后再执行一次 full-attention verify。
- Orthrus 用学习得到的双向 draft-attention 视图，在一次共享-backbone forward
  中并行产生完整 32-token block，随后用 AR attention 视图验证整块。

因此 PillarAttn 的原则可以直接应用：只近似 draft 的历史 attention，verifier
仍保持完整。但 5% 预算和调度节奏不能照抄。Orthrus 一次有 32 个 draft query，
而且 draft attention 是针对 full history 训练的；历史过度稀疏时，block 接受率
可能比 SparseSpec 更快下降。

## Orthrus、SparseSpec 与 Vegas

- [Orthrus](https://arxiv.org/abs/2605.12825) 为冻结的 AR backbone 训练额外的
  双向 attention view，一次并行提出 32-token block。它的并行上限高，但每换
  backbone 都需要重新训练 draft attention。
- [SparseSpec](https://arxiv.org/abs/2512.01278) 不训练额外参数，用稀疏 attention
  作为同模型 draft，并同时设计 unified scheduler、delayed verification 和 KV
  offload。它更像一套完整的长推理 serving 系统。
- [Vegas](https://arxiv.org/abs/2602.07223) 同样是 training-free self-speculation，
  但重点是让完整 verifier 顺带产出逐 token attention importance，下一轮 draft
  只读取 Top-K KV。它直接针对 standalone selector 不准、选择开销高的问题。

三者不存在脱离模型、上下文和硬件的统一胜者。对当前已经训练好的 Orthrus，
最有价值的组合是保留 K32 双向 draft，同时移植 Vegas 的精确 verifier-guided
selector；之后再评估 SparseSpec 的 mixed-phase scheduler。当前实现的单层页代理
只能验证稀疏执行路径，不能视为 Vegas 的等价复现。

## 上游核心技术与当前状态

| SparseSpec 技术 | 上游做法 | Orthrus/Ascend 状态 |
|---|---|---|
| PillarAttn | full verify 顺带导出各层、各 KV head 的 attention score；Top-K 约 5% 历史 token；draft 读取重要 token 与最近 token | 已实现页粒度原型与真实服务 A/B；不是 CUDA token-granular kernel 的逐行照搬 |
| Unified batch scheduler | NORMAL/DRAFT/VERIFY/STALL 状态机；同一 forward 混合不同请求的 draft 与 verify | 已做 delayed/staggered admission 原型；当前 vLLM custom proposer 仍是 verify 后同步调用 draft，不能形成上游的真正 mixed-phase forward |
| Fused draft/verify attention | FlashInfer persistent kernel 在同一 kernel 中分派 sparse draft 和 full verify tile | 当前无 Qwen GQA 对应 Ascend kernel；只有 `paged_segmented` 和页子集 FIA。需在 trace 证明收益后再做 Triton-Ascend/AscendC |
| Delayed verification | VERIFY 后插入 STALL 周期，使 CPU correction 与其他请求 GPU 工作重叠 | 已实测；当前实现吞吐下降 1.56%，NPU busy 从 87.19% 降至 81.21%，不保留为默认 |
| CUDA graph double buffer | 两套 graph/workspace 交替，消除 workspace 复用同步 | 依赖 fused mixed-phase graph；Ascend ACL graph 下尚无可直接复用实现，暂不独立开发 |
| Chunk-wise async KV offload | 独立 stream 分块卸载，并维护稀疏页 logical/physical index 转换 | 当前 64x128 基线无 KV preemption；仅在 40K/1024 或更高并发出现容量瓶颈后启用，避免给非瓶颈增加搬运 |

## 上游实现核对

核对版本：`sspec-project/SparseSpec@5c9616c9`。

PillarAttn 的实际预算不是固定窗口：

```text
token_budget = max(128, ceil(context_length * 0.05))
token_budget < 6144
draft KV = top-k historical tokens + newly accepted/recent tokens
```

选择在每轮 verify 后刷新。上游要求 `page_size=1`，并为每层、每 KV head 保存
不同 token index。verify attention kernel 在正常计算输出时同时 rematerialize
并累积 raw attention logits，避免额外执行一遍 QK。随后自定义 batch top-k
kernel 更新稀疏索引。

当前 Ascend 镜像虽然导出了 `torch_npu.npu_sparse_flash_attention`，实机调用
确认其安装 kernel 只支持 DeepSeek MLA 的 `attention_mode=2`；Qwen3 GQA 的
`attention_mode=0` 会被 CANN tiling 拒绝。因此本阶段使用已经验证过的
`npu_fused_infer_attention_score` Paged KV block-table 子集路径。

## 当前接口

```bash
ORTHRUS_DRAFT_ATTENTION=paged_pillar
ORTHRUS_PILLAR_POLICY=verifier       # verifier 或 recent
ORTHRUS_PILLAR_BUDGET_RATIO=0.05
ORTHRUS_PILLAR_MIN_TOKENS=128
ORTHRUS_PILLAR_MAX_TOKENS=6144
ORTHRUS_PILLAR_SELECTOR_LAYER=-1     # 最后一层
ORTHRUS_DRAFT_SEGMENT_THRESHOLD=8192
```

所有参数进入 trial/profile manifest。低于 threshold 的请求仍使用完整历史；
超过 threshold 的请求保留最新物理页，并在预算内选择旧页。最新的 partial
page 始终放在 block table 末尾，保证 `actual_seq_lengths_kv` 不会读到未写槽位。

## 微基准

形状来自正式混合负载中的 8 个请求，history 为 6.6K--39.3K，Q=32，
Q-head=32，KV-head=8，head-dim=128，page=128。低于 8K 保持 dense。

```text
dense paged-history attention: 0.845 ms / layer

5% recent pages:
  page selection                 0.642 ms / draft round
  sparse history attention      0.293 ms / layer
  32-layer amortized estimate   2.70x

5% verifier-guided pages:
  page selection                 1.178 ms / draft round
  sparse history attention      0.304 ms / layer
  32-layer amortized estimate   2.48x
```

这里的 32 层估算只覆盖 history-attention 子阶段，不代表端到端加速。正式判断
必须同时看 output token/s、P95 和 acceptance rate。

## 第一轮端到端结果

固定同一 64-request burst、输出 128、MNS64、MBT65536、memory0.94、
FULL_AND_PIECEWISE/capture1024；基线是 `paged_segmented`。

| 路径 | output tok/s | 相对 B1 | P95 latency | acceptance |
|---|---:|---:|---:|---:|
| B1 `paged_segmented` | 34.296 | - | 233.94 s | 5.80% |
| Pillar `recent`, 5%, threshold=8K | 29.157 | -14.98% | 275.17 s | 2.51% |
| Pillar `verifier`, 5%, threshold=8K | 26.830 | -21.77% | 298.63 s | 2.58% |
| Pillar `recent`, 25%/6K, threshold=8K | 31.369 | -8.53% | 255.74 s | 3.94% |
| Pillar `verifier`, 25%/6K, threshold=8K | 31.354 | -8.58% | 254.65 s | 4.00% |
| Pillar `recent`, 25%/6K, threshold=24K | 33.462 | -2.43% | 239.59 s | 4.80% |
| Pillar `verifier`, 25%/6K, threshold=24K | 31.462 | -8.26% | 253.34 s | 4.80% |

因此 5% 不能从 SparseSpec 原样搬到 Orthrus。kernel 级读取量下降是真实的，
但页级选择近似不足以维持 Orthrus draft 质量，额外 verify 轮数抵消了收益。
提高到 25% 并只对 24K 以上请求启用后，`recent` 已接近 B1，且 P95 TPOT
从 0.435 s 降至 0.378 s；不过总吞吐仍低 2.43%，尚不能作为默认优化。

当前 `verifier` 页代理几乎没有改善接受率，却额外执行 query-key page score
和 Top-K，因此比 `recent` 慢约 6%。这不否定论文里的 PillarAttn：论文直接
复用 full attention kernel 内部的 token-level score，而当前实现是在 FIA
外部用单层、单页代表 key 近似。两者的选择精度和额外开销并不等价。

补充的 16-request、每请求 1024-token 长生成 A/B：

| 路径 | output tok/s | P50/P95 TPOT | acceptance |
|---|---:|---:|---:|
| B1 `paged_segmented` | 76.884 | 0.112/0.130 s | 6.76% |
| `recent`, 25%/6K, threshold=24K | 74.790 | 0.099/0.136 s | 5.92% |
| `recent`, 50%/12K, threshold=24K | 74.076 | 0.093/0.139 s | 6.02% |
| B1 同卡复跑 | 77.657 | 0.105/0.128 s | 6.73% |
| `recent`, 75%/24K, threshold=24K，同卡复跑 | 74.356 | 0.099/0.131 s | 6.23% |

稀疏路径改善了约 12% 的 P50 TPOT，但总吞吐低 2.72%，P95 TPOT 反而高
4.52%。把预算放宽至 50%/12K 后，接受率只从 5.92% 回升到 6.02%，吞吐
反而进一步降至 74.076 tok/s。由此可以排除“只因预算过小”的解释，当前
页级选择路径停止扫参。进一步去掉小 cap、保留最多 75%/24K 后，同卡吞吐仍比
B1 低 4.25%；P50 TPOT 改善约 5.96%，但 P95 TPOT 变差约 2.00%。

并行运行的 `pillar-recent-r075-max24k-t24k-16-out1024-20260914` 和
`pillar-recent-r100-max40k-t24k-16-out1024-20260914` 不计入结果。两个容器均使用
host network 且误用了同一端口，两个客户端实际打到同一服务，导致负载和指标
相互污染。上表只使用随后在同一张 NPU 上顺序运行的有效对照。

## 下一步判据

1. 当前页级 selector 不合格，不作为默认优化，也不继续扫描预算。
2. 下一版 selector 应优先在完整 verifier FIA 内导出 page score，避免额外 QK；只有 trace 支持至少 8% 端到端空间时才开发 GQA 专用 Ascend kernel。
3. Unified scheduler 必须和 sparse/full 混合 attention runner 一起实现；单独延迟或截断 draft batch 的负结果已经说明，模仿状态名并不能复现 SparseSpec 的收益。

可复算的微基准、trial manifest 与汇总结果保存在
[`artifacts/sparse_spec`](../artifacts/sparse_spec/README.md)；逐请求结果、服务日志和
profiler trace 按其中记录的服务器路径归档。

## 参考

- Paper: <https://arxiv.org/abs/2512.01278>
- Official code: <https://github.com/sspec-project/SparseSpec>
- Ascend sparse attention API: <https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md>
