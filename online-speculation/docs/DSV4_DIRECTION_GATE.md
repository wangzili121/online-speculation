# DeepSeek-V4 方向筛选

## 结论

后续 Orthrus 实验只保留能够迁移到 DeepSeek-V4/DSpark 的机制。Qwen3 GQA 上的
VEGAS 页选择已经完成机制验证和负向 A/B，不再继续产品化：它依赖 Qwen 的
多 KV-head attention 布局，而 DSV4 使用模型原生的 learned indexer、单 KV-head
compressed sparse MLA 和 DSpark，二者的执行瓶颈不同。

## 已被现有框架覆盖

截至 2026-09-14，最新 vLLM/vLLM-Ascend 已包含以下能力：

- DSV4 原生 learned lightning indexer 与 sparse MLA；
- IndexCache，通过 Full/Shared layer pattern 跨层复用 top-k indices；
- 普通串行 MTP 的 `index_share_for_mtp_iteration`：第 0 个 draft step 计算
  top-k，后续 step 复用；这不等价于 DSpark 的并行 block proposer；
- DSpark 自适应 verification budget 和 Ascend dynamic speculative scheduler；
- indexer/compressor/projection 多 stream overlap，以及 DSA context parallel。

因此以下工作不再作为创新方向：给 DSV4 另加普通稀疏 draft attention、简单复用
上一 draft token 的 top-k、再次实现动态 draft length，或单独移植 VEGAS CUDA
kernel。

## Qwen VEGAS 负结果

同一张 910B3、16 请求、每请求 128 输出、eager、固定编码 workload：

| 实现 | output tok/s | 相对 dense | acceptance |
|---|---:|---:|---:|
| dense `paged_segmented` | 29.964 | - | 5.90% |
| 每 block、每层 VEGAS reference | 19.344 | -35.44% | 5.85% |
| request/layer 缓存的 VEGAS | 25.964 | -13.35% | 5.67% |

原生 `npu_lightning_indexer` 对 DSV4 的单 KV-head、head-dim 128 形状很快，但
拒绝 Qwen3 的 8 KV-head GQA；`npu_sparse_flash_attention` 也面向 MLA 合同。
这进一步说明 Qwen 原型不适合作为 DSV4 产品路径。

## 当前唯一优先候选：PIVOT

[PIVOT](https://arxiv.org/abs/2607.24593) 优化的是 DSA indexer 的 query 轴，和
IndexCache 的 layer 轴不同。它把一组相邻 query 合成一个 proxy，只完整扫描
prefix 一次：

- Reuse 直接让整组 query 使用 proxy top-k；
- Refine 先取稍大的公共候选集，再让每个 query 只在候选集内重评分。

这与 DSpark 的目标模型验证阶段匹配：verifier 一次处理当前 token 和整个 draft
block。官方部署示例使用 5 或 7 个 speculative tokens，因此目标验证形成 6 或
8 个相邻 query。DSpark 自己的 3 层 draft attention 只保留
128-token sliding window，不是这里要优化的长历史 DSA 扫描。论文在
DeepSeek-V3.2 与 GLM-5.1 上报告 indexer 最高 4 倍、长上下文端到端最高 1.6 倍，
但尚不能把该数字外推到 DSV4 或 Ascend。

## Ascend 算子探针结果

探针使用 910B3 上可用的 BF16 `npu_lightning_indexer`，batch 8、64 query
heads、head-dim 128。它复现 DSV4 的主要 shape 和分页访问，但不是最新 DSV4
正式使用的量化 `npu_quant_lightning_indexer_v2`，所以只能用于判断数量级和
break-even，不能作为正式 DSV4 性能结果。`Refine` 先用 proxy 扫完整压缩历史，
再把候选页 pack 起来，用第二次原生 indexer 为每个 query 重选 top-k。

以下首轮表使用 top-k 1024、候选集 `2 x top-k`、query noise 0.1；最新 DSV4
合同的 top-k 2048 结果另行记录，不混用：

| 原始上下文等价长度 | 5-query Refine | 7-query Refine | 5-query recall | 7-query recall |
|---:|---:|---:|---:|---:|
| 128K | 0.67x | 0.86x | 99.90% | 99.93% |
| 256K | 1.22x | 1.58x | 99.68% | 99.77% |
| 512K | 2.03x | 2.63x | 99.58% | 99.61% |
| 1M | 2.44x | 3.18x | 99.27% | 99.38% |

按最新 DSV4 部署示例重跑 top-k 2048，并将 5/7 个 speculative tokens 计入
当前 token，使用 verifier group 6/8。论文默认 `c=2k=4096` 的结果如下：

| 原始上下文等价长度 | 6-query Refine | 8-query Refine | 6-query recall | 8-query recall |
|---:|---:|---:|---:|---:|
| 128K | 0.62x | 0.78x | 99.99% | 99.98% |
| 256K | 0.74x | 0.95x | 99.96% | 99.90% |
| 512K | 0.83x | 1.09x | 99.85% | 99.78% |
| 1M | 0.88x | 1.13x | 99.72% | 99.59% |

因此，top-k 1024 的积极结果不能外推到 DSV4。现有算子在 candidate count
超过 2048 后进入 1024-token bucket，1M/group 8/c=4096 的 2.21 ms 中，约
1.97 ms 都花在 proxy top-4096 的完整扫描；candidate pack 和局部重评分分别
仅约 0.12 ms 和 0.13 ms。真正的机会是让一次 scan 在片上直接保留 4096 个
候选并完成局部 refine，而不是把两个现成 indexer 串起来。

所以它不是中短上下文优化。按真实 top-k 2048 合同，现有算子组合直到 1M 也只
在 group 8 得到约 13% indexer-only 收益，尚未通过 1.5x 实现门槛。纯 `Reuse`
更快，但 query noise 0.1 时 recall 只有约 86%--90%，不能直接采用。

这些 recall 只来自可控相关性的合成 query。真正决定是否继续实现的是从真实
DSV4 verifier 捕获相邻 query 后得到的重合率、acceptance 和端到端 profile；
当前两台服务器没有 DSV4 权重，因此尚未完成这项质量门槛。

本仓库使用 `scripts/benchmark_dsv4_pivot_npu.py` 判断 Ascend 算子级上限。当前
现成算子组合没有通过门槛，因此不直接接入 serving。只有同时满足以下条件才进入
真实 DSV4 实现：

1. 在 DSpark verifier group 6/8、raw context 256K 以上，融合或量化
   PIVOT-Refine 的 indexer 路径至少快 1.5 倍；
2. 最新 vLLM/vLLM-Ascend 尚无等价 query-group indexer；
3. 真实 DSV4 profile 中 indexer 占 DSpark step 至少 10%；
4. top-k recall、DSpark acceptance 和目标模型质量无不可接受回退；
5. 端到端 SLO goodput 至少提高 5%。

如果算子 probe 失败，或 DSV4 profile 表明 indexer 已被 overlap 隐藏，则停止
PIVOT，转查 DSV4 特有的 compressor/indexer/DSpark 跨阶段融合，不继续稀疏选页。

## 参考

- PIVOT: <https://arxiv.org/abs/2607.24593>
- IndexCache: <https://arxiv.org/abs/2603.12201>
- vLLM-Ascend IndexCache PR: <https://github.com/vllm-project/vllm-ascend/pull/9390>
- vLLM-Ascend DSV4 model: <https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/models/deepseek_v4>
- vLLM-Ascend DSpark proposer: <https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/spec_decode/dspark_proposer.py>
