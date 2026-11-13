# Orthrus-DSV4.1 参考实现与验证

DeepSeek-V4.1 的 SWA 使用本层历史，C2A 和 C1A 还会从源层读取稀疏全局历史。Orthrus 每轮从一个已知 anchor 和最多 31 个 MASK 起草。每个 draft 位置都能读取 anchor 前的历史和本轮的整个临时块。Attention 和 indexer 使用各自的缓存页表。Draft 的 Top-K 和 C1A 候选池只在本轮内传递。 [1017 模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) · [官方模型实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py)

## Head 与计算接口

在 `online-speculation/` 运行以下命令。只传 `--model` 即可从配置生成 40 层角色表。传入 `--head` 后还会校验基模配置、权重索引、tokenizer 的 SHA256，以及 head 分片中的张量名、形状和精度。

```bash
PYTHONPATH=src python scripts/preflight_dsv41_orthrus.py --model "$MODEL"
PYTHONPATH=src python scripts/preflight_dsv41_orthrus.py \
  --model "$MODEL" --head "$HEAD" --mask-token-id "$MASK_ID" --require-trained
```

`orthrus-dsv41-head-v1` 是 BF16 交换格式，sink 保留 FP32。40 层各自包含 Q/KV/O 投影、内部 norm 和 sink，8 个索引源层另含 indexer Q 和打分权重。历史 KV、indexer K 和 compressor 由目标模型提供。初始化测试产物应标记为 `initialized-plumbing`。 [契约源码](../src/blockspec/vllm_ascend/dsv41_contract.py)

Torch 参考接收已做 RoPE 的查询和逻辑分页视图。C2A 只检索已完成的压缩组。C1A 第 20 层以每个粗粒度块内的最高分构造候选池，保留最新可见块。第 20 层自己的 Top-K 从全部可见历史产生，后续 Reindex 层在候选池内重排。Reuse 层直接取最近的源层 Top-K。参考对同分项优先选择逻辑行号较小的一项。SWA、全局历史与临时块使用同一次 softmax，sink 只计一次。 [参考实现](../src/blockspec/vllm_ascend/dsv41_attention.py)

## 验证结果

CPU 覆盖 1017 拓扑、BF16/F32 张量校验、C2A 奇偶和跨页读取、C1A 候选池及后续层 Reindex/Reuse、空历史、块内双向与 sink。页表换序后，逻辑 Top-K 和输出保持一致。非法 head、超出已提交历史的选中行均被拒绝。 [测试](../tests/test_dsv41_orthrus.py)

在 Ascend 910B3 上，分段 BF16 FIA 对 CPU FP32 参考的最大输出绝对误差为 `0.0078213`。按 `atol=0.001, rtol=2.01/128` 的逐元素误差检查，约 157 万个值中有 118 个超限。同输入对照中，将可见 KV 合入一次 BF16 FIA 后仍有 48 个值超限。BF16 FIA 返回的输出本身已经舍入。

精度验证默认使用 FP32 路径。SWA 与临时块的 KV 在 K 个 query 之间共享，稀疏全局 KV 按 query 取页。两路分别计算 FP32 分数，和 sink 一起归一化，再用 FP32 累加输出。测试关闭 Ascend HF32 matmul，并对输出和 LSE 分别使用 `atol=1e-5, rtol=1.3e-6`。 [FP32 实现](../src/blockspec/vllm_ascend/dsv41_precise_attention.py)

| 910B3 复验 | 输出最大绝对误差 | LSE 最大绝对误差 |
|---|---:|---:|
| 原 C2A Full/Reuse、K32、连续和乱序页 | 2.63e-6 | 1.91e-6 |
| SWA、C1A、C2A，滑窗 128、Top-K 512、历史最长 8194 | 1.67e-6 | 9.54e-7 |
| 第 2 层真实权重投影样本、K32、Top-K 512 | 1.42e-7 | 9.54e-7 |

以上用例均通过精度检查。原 C2A 用例 Top-K 与 CPU 完全一致，各用例目标 KV、索引页及页表逐元素保持一致。

```bash
PYTHONPATH=src python scripts/smoke_dsv41_orthrus_npu.py \
  --config "$MODEL/config.json" --compare-fia
```

添加 `--layer2-weights /path/to/layer2-a2-w8a8.pt` 可复测第 2 层样本。该用例使用原版 Flash 的 A2 W8A8 权重转换样本和合成隐藏状态，CPU 反量化并生成 Q/KV；其余用例使用合成的分页 KV。复验使用 `torch 2.10.0+cpu` 和 `torch_npu 2.10.0.post4`。实际 prompt 激活、完整 1017 量化服务和吞吐尚待验收。

## 原生 INT8 检索

目标 index cache 已按 C2A 或 C1A 的粒度写好。Draft 调用 QLI 时直接传入可读行数 `floor(anchor / ratio)`，在这段截断后的前缀内让全部 K 行检索。调用中的 `cmp_ratio=1` 表示坐标已经是压缩行号。C1A 候选池由 Full 层返回并传给 Reindex，输出会校验候选成员资格并去重。实测当候选行少于 Top-K 时，原生算子可能返回额外的行 0。

在 `nightly-releases-v0.32.0rc` 的 Ascend 910B3 上，K32 的 C2A 和 C1A 各验证 96 个查询，另核对 64 组 C1A 候选池。覆盖空历史、奇偶压缩边界、乱序物理页、候选截断与不满 Top-K 的末块。目标 INT8 key、scale 和页表逐元素一致。C1A 样例用 Top-K 512 和 64 个候选块，部署配置的 2048 个候选块及长历史还需验收。 [检索实现](../src/blockspec/vllm_ascend/dsv41_native_index.py)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_native_index.py
```

## 原生缓存与引擎接入

下一步把检索绑定到 MRV2 提供的目标源页，并验证原生 SparseFlashMla 能否组合真实 SWA 页、选中的压缩页和临时块。目标侧原生检索仍使用因果元数据及 C2A 压缩余数。接入后对照 target-only 验收接受前缀、拒绝回滚和目标状态。 [原生检索实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/indexer.py) · [原生 Attention 调用](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)
