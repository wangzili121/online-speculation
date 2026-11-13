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

高效路径使用现有 Ascend 融合核。SWA 历史和选中的压缩历史分别调用 `npu_sparse_flash_mla`，临时 K×K 双向块调用 fused infer attention，再以 FP32 LSE 精确合并三段。这样每个 draft query 共享同一 anchor 历史边界，压缩 Top-K 仍可逐 query 不同，sink 只计一次，目标缓存始终只读。目标模型将 SWA 和压缩历史合在一次因果调用里的接口无法表达这组 Orthrus 语义，因此这里需要分段。 [原生实现](../src/blockspec/vllm_ascend/dsv41_native_attention.py)

在 `nightly-releases-v0.32.0rc`、`torch_npu 2.10.0.post7` 和 Ascend 910B3 上，K32、TP8 单 rank 维度、SWA 128、Top-K 512、anchor 为 0/1023/2047 的用例如下。融合路径按 BF16 误差门限通过，LSE 与 FP32 参考保持在 `1e-6` 内，目标页和页表逐元素不变。计时为同一张卡上的算子窗口，不含投影、调度和 verifier。 [验证记录](validation/dsv41/native-draft-attention-910b3.json)

| Attention 路径 | 输出最大绝对误差 | LSE 最大绝对误差 | 耗时 |
|---|---:|---:|---:|
| 原生融合分段 | 4.42e-3 | 9.54e-7 | 2.98 ms |
| FP32 golden 路径 | 1.55e-6 | 9.54e-7 | 5.70 ms |

Attention 后的 grouped low-rank O projection 也在同一设备栈上单独复验。算子使用主模型相同的 `npu_transpose_batchmatmul` 和 `[group, group_width, o_lora_rank]` 权重布局。覆盖 1、2、8 个本地 group 以及 32、64、96 个 token；相对 CPU FP32 公式的最大绝对误差不超过 `0.00406`，P99 不超过 `0.00196`，均通过 BF16 门限。 [验证记录](validation/dsv41/grouped-output-projection-910b3.json)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_orthrus_npu.py \
  --config "$MODEL/config.json" --compare-fia
PYTHONPATH=src python scripts/probe_dsv41_sparse_mla.py
PYTHONPATH=src python scripts/smoke_dsv41_oproj_npu.py --device npu:0
```

添加 `--layer2-weights /path/to/layer2-a2-w8a8.pt` 可复测第 2 层样本。该用例使用原版 Flash 的 A2 W8A8 权重转换样本和合成隐藏状态，CPU 反量化并生成 Q/KV；其余用例使用合成的分页 KV。复验使用 `torch 2.10.0+cpu` 和 `torch_npu 2.10.0.post4`。实际 prompt 激活、完整 1017 量化服务和吞吐尚待验收。

## 原生 INT8 检索

目标 index cache 已按 C2A 或 C1A 的粒度写好。Draft 调用 QLI 时直接传入可读行数 `floor(anchor / ratio)`，在这段截断后的前缀内让全部 K 行检索。调用中的 `cmp_ratio=1` 表示坐标已经是压缩行号。C1A 候选池由 Full 层返回并传给 Reindex，输出会校验候选成员资格并去重。实测当候选行少于 Top-K 时，原生算子可能返回额外的行 0。

在 `nightly-releases-v0.32.0rc` 的 Ascend 910B3 上，K32 的 C2A 和 C1A 各验证 96 个查询，另核对 64 组 C1A 候选池。覆盖空历史、奇偶压缩边界、乱序物理页、候选截断与不满 Top-K 的末块。目标 INT8 key、scale 和页表逐元素一致。C1A 样例用 Top-K 512 和 64 个候选块，部署配置的 2048 个候选块及长历史还需验收。 [检索实现](../src/blockspec/vllm_ascend/dsv41_native_index.py)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_native_index.py
```

## 原生缓存与引擎接入

MRV2 接入以插件形式注册 `OrthrusDSV41DraftModel`，并为该架构选择 Orthrus speculator。调度沿用并行 draft 的 anchor/MASK 布局和标准 verifier，模型语义由 Orthrus draft 实现：每轮绑定目标模型的 SWA、压缩 KV 与 indexer key 页，使用 PAD slot 和 `skip_ring_state_update` 保持目标 cache 只读；Top-K 与临时块状态只存在于本轮 draft。draft 只持有 diffusion Attention/indexer 参数，共享目标 embedding、mHC、Engram、norm、MoE/MLP 和 LM head。初始化 head 可按正式张量契约导出并严格 preflight，用于验证装载和执行链路；它不用于评估接受率。

当前 CPU 契约、head 导出/重载、分页与回滚状态、Attention/indexer/O-projection 独立 NPU 路径和 v0.32 插件导入均已覆盖。MRV2 元数据复验确认，parallel input prep 中包含临时 K32 的长度会在构造目标历史时扣除，所有 draft slot 使用 PAD，且禁用 ring 状态更新。 [验证记录](validation/dsv41/mrv2-readonly-metadata.json)

使用官方 1017 维度和 dummy BF16 权重，完整 40 层 head 已在 910B3 上完成 TP1 与 TP4 构造。TP1 包含 5,107,950,080 个参数，336 个 checkpoint 张量全部通过 loader round-trip；TP4 完成 HCCL 初始化和逐 rank 名称/本地形状检查，`wo_a` 从 TP1 的 `[8,4096,1024]` 分为每 rank `[2,4096,1024]`。热路径使用加载后常驻的三维原生布局，不在每轮转置权重。 [验证记录](validation/dsv41/head-construction-910b3.json)

## A2 两节点部署

正式 A2 路径采用官方的两节点 colocated 拓扑：全局 DP2，每个 DP rank 在一台 8 卡服务器上运行 TP8，并启用 expert parallel。两台服务器必须使用同一镜像、同一绝对模型/head 路径和可互通的 HCCL 网卡。当前固定镜像为 `quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc`，仓库 digest 是 `sha256:f040ca190fe5c5864a5491b2cbe19e687966ae8609b8a68fccc2c1ad1d260cbe`，本地镜像 ID 是 `sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2`。先启动 Node 0，再启动 Node 1；只有 Node 0 暴露 API。

```bash
# 两台服务器都在仓库根目录执行；按本机填写 NODE_RANK/LOCAL_IP。
export NODE0_IP=<node-0-ip>
export LOCAL_IP=<this-node-hccl-ip>
export NIC_NAME=<hccl-interface>
export NODE_RANK=0                    # 另一台设为 1
export ACK_EXCLUSIVE_NODE=1           # 确认本节点八卡已预留
PROFILE=smoke-16k scripts/deploy_dsv41_node.sh
```

宿主机脚本校验不可变镜像 digest，挂载代码、模型和八张 NPU，再在容器内调用服务脚本。模型和 head 默认取 `/data/disk/models` 下的约定目录，也可用 `MODEL_HOST`、`HEAD_HOST` 覆盖。`PROFILE` 可改为 `prod-128k` 或 `prod-1m`。Orthrus 脚本会安装本仓插件、校验 trained head 和基模哈希，然后以 eager 模式运行 K32；设置 `MODE=target` 是 target-only 回滚路径。部署前可设置 `DRY_RUN=1` 查看完整命令而不创建容器。 [官方 A2 拓扑](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html)

完整服务验收仍需实际 V4.1 checkpoint，验证 target-only 启动、40 层 draft 前向、greedy 一致性、拒绝回滚、并发换序与显存预算；TP8 dummy 构造也需在八卡同时空闲时补测。

原生融合路径固定依赖已验证的 post7 SparseFlashMLA ABI。较旧的 post4 栈在 A2/A3 上不支持 `ori_sparse_indices`。167 上的 vLLM-Ascend 0.30 镜像虽然带 post7 Python 包，其本地 CANN kernel 缺少该组合的 tiling key，也不能作为此路径的部署基线。部署必须固定完整镜像 digest、CANN 和驱动，并运行 `probe_dsv41_sparse_mla.py` 作为 preflight。 [原生检索实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/indexer.py) · [原生 Attention 调用](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)
