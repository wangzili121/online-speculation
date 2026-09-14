# DSV4 PIVOT Ascend microbenchmark

这些 JSON 是 `scripts/benchmark_dsv4_pivot_npu.py` 在单张 910B3 上的原始输出。
它们使用当前容器提供的 BF16 `npu_lightning_indexer` 测 DSV4 形状，不是最新
DSV4 正式采用的量化 indexer，也不包含真实模型 forward 或 serving：

- `h*-noise010.json`：top-k 1024 的首轮压缩历史扫描；
- `topk2048-h*-noise010.json`：按最新 DSV4 top-k 2048 合同重跑；
- `h65536-noise*.json`：固定 256K 原始上下文，扫描相邻 query 的相关性；
- `groups 5/7` 是首轮形状探针；正式按 DSpark 5/7 个 speculative tokens
  使用 verifier group 6/8；
- candidate ratio：PIVOT-Refine 公共候选集相对最终 top-k 的倍数。

合成 query 用共同 anchor 加可控噪声构造。`topk_recall` 因而只能描述机制和
敏感性，不能代替真实 DSV4 query、acceptance 或端到端质量结果。
