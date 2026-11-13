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

在 Ascend 910B3 上用真实配置的头数和维度构造测试张量，覆盖 C2A 层 2/3、K32、每 rank 8×512 的 Attention 形状、anchor 0/33/34，以及连续和打乱的物理页。Torch-NPU 组合完成逻辑取页和检索，两个 fused Attention 段分别返回 LSE，再与 sink 合并。Top-K 与 CPU 精确一致，Attention 输出最大绝对误差为 `0.0078213`，LSE 为 `1.43e-6`。输入页及页表逐元素一致。

```bash
PYTHONPATH=src python scripts/smoke_dsv41_orthrus_npu.py --config "$MODEL/config.json"
```

设备验证使用 `torch 2.10.0+cpu` 和 `torch_npu 2.10.0.post4`。此次使用合成历史页，融合 Attention 前会展开选中的历史行。167 上另有原版 Flash 的第 2 层权重样本，已核对 Attention 与 indexer Q/打分张量形状。1017 的量化权重转换和端到端推理仍待验收。

## 原生 INT8 检索

目标 index cache 已按 C2A 或 C1A 的粒度写好。Draft 调用 QLI 时直接传入可读行数 `floor(anchor / ratio)`，在这段截断后的前缀内让全部 K 行检索。调用中的 `cmp_ratio=1` 表示坐标已经是压缩行号。C1A 候选池由 Full 层返回并传给 Reindex，输出会校验候选成员资格并去重。实测当候选行少于 Top-K 时，原生算子可能返回额外的行 0。

在 `nightly-releases-v0.32.0rc` 的 Ascend 910B3 上，K32 的 C2A 和 C1A 各验证 96 个查询，另核对 64 组 C1A 候选池。覆盖空历史、奇偶压缩边界、乱序物理页、候选截断与不满 Top-K 的末块。目标 INT8 key、scale 和页表逐元素一致。C1A 样例用 Top-K 512 和 64 个候选块，部署配置的 2048 个候选块及长历史还需验收。 [检索实现](../src/blockspec/vllm_ascend/dsv41_native_index.py)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_native_index.py
```

## 原生缓存与引擎接入

下一步把检索绑定到 MRV2 提供的目标源页，并验证原生 SparseFlashMla 能否组合真实 SWA 页、选中的压缩页和临时块。目标侧原生检索仍使用因果元数据及 C2A 压缩余数。接入后对照 target-only 验收接受前缀、拒绝回滚和目标状态。 [原生检索实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/indexer.py) · [原生 Attention 调用](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)
