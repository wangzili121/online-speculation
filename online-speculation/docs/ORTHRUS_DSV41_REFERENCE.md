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

### 离线训练交接

训练端以 1017 的 W4A8 主模型为冻结教师，导出与部署相同的 tokenizer、mask token 和 K32 位置信息。学生使用真实 anchor 前的历史和块内双向可见的临时 KV。第 0–30 行 draft logits 对齐 anchor 后的第 1–31 个位置，训练数据应为超出答案范围的位置打上无效标记。可先用 forward KL 检查语言模型蒸馏。训练 indexer 时，需单独监督检索分数，离散 Top-K 无法把 KL 梯度传回 indexer。

训练端将 336 个 head 张量以 `layers.N.attn.*` 命名，保存到单个 BF16/FP32 safetensors 中。按训练记录选择同一个基模目录和 mask token，通过以下命令生成服务可加载的分片、配置和 manifest。导出会检查缺失或多出的张量、形状、精度和非有限值，最终保存 BF16 权重和 FP32 sink。 `head_shapes` 返回逐张量形状。 [张量契约](../src/blockspec/vllm_ascend/dsv41_contract.py)

```bash
PYTHONPATH=src python scripts/export_dsv41_orthrus_head.py \
  --model "$MODEL" --mask-token-id "$MASK_ID" \
  --from-trained-head "$TRAINED_SAFETENSORS" --output "$HEAD"
PYTHONPATH=src python scripts/preflight_dsv41_orthrus.py \
  --model "$MODEL" --head "$HEAD" --require-trained
```

在启动长训前，用真实样本确认 teacher 与 draft 的 logits 逐位置对齐，对比训练端与服务端同一 head、同一历史输入下的 draft logits，并验证一步反向只更新约定的参数。再用这一步导出的 head 启动服务，检查候选确实进入 verifier、greedy token 与 target-only 一致，以及连续拒绝后的目标 cache 状态。当前导出和预检已通过 CPU 往返验证，真实训练样本到八卡服务的往返还待跑通。

单机验收脚本增加了 `--offline-handoff`。它在运行请求时保存一轮对齐后的原生 teacher/draft logits、最后一层输入及只读历史。CPU 先对照服务输出与 FP32 replay 的 logits，再用 forward KL 更新最后一层，将保存的 head 分片交回镜像启动。输出记录 logits 最大误差、P99、TV、argmax 一致数量、梯度范数和 BF16/F32 导出后仍发生变化的张量。脚本只更新第 39 层，离线团队可沿用同一 head 契约，并单独验证所选训练层的反向。采集会记录推理时 verifier 实际到达的位置，适用于本次交接 smoke。完整离线训练的数据应来自 teacher-forced 序列并包含该样本所有有效 MASK 监督位置。

```bash
PYTHONPATH=src python scripts/train_dsv41_last_layer_smoke.py \
  --model "$MODEL" --head "$INITIAL_HEAD" --feedback "$FEEDBACK" --output "$UPDATED_HEAD"
```

在 910B3 上使用实际 1017 W4A8 末层和 40 层初始化 head，配合合成历史和教师 logits，单步 KL 由 6.0493 降至 5.8103，导出的 8 个末层张量中 7 个发生变化。按部署精度重读权重后，KL 为 5.8396，前 39 层分片的 SHA256 全部一致。vLLM safetensors loader 在单卡和 TP2 均加载完整 40 个分片、336 个张量，各 rank 的末层 sink 与导出文件对应分片逐元素一致。此用例未加载主模型服务。真实请求的训练与推理 logits 对齐、greedy 及 cache 复验由八卡脚本完成。

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

SWA 历史与临时 K×K 双向块一起用 FP32 批量矩阵乘法计算，选中的压缩历史调用 Ascend `npu_sparse_flash_mla`，两段以 FP32 LSE 合并。短于 draft 块的压缩历史走小规模 FP32 计算。混合长短请求时，融合算子的长度下限与实际可见行数分别传入，可见范围仍由显式索引和有效行数决定。这样每个 draft query 共享同一 anchor 历史边界，压缩 Top-K 仍可逐 query 不同，sink 只计一次，目标缓存始终只读。这组可见范围同时包括固定的真实前缀与双向临时块。 [原生实现](../src/blockspec/vllm_ascend/dsv41_native_attention.py)

在 `nightly-releases-v0.32.0rc`、`torch_npu 2.10.0.post7` 和 Ascend 910B3 上，K32、TP8 单 rank 维度、SWA 128、Top-K 512、anchor 为 0/1023/2047 的用例如下。目标页和页表逐元素保持一致。表中计时仅覆盖算子。 [原始全融合记录](validation/dsv41/native-draft-attention-910b3.json)

| Attention 路径 | 输出最大绝对误差 | LSE 最大绝对误差 | 耗时 |
|---|---:|---:|---:|
| SWA 与临时块 FP32，压缩历史融合（32 条有效检索历史） | 1.18e-3 | 9.54e-7 | 3.46 ms |
| SWA 与临时块 FP32，压缩历史融合（512 条有效检索历史） | 9.10e-4 | 9.54e-7 | 3.68 ms |
| 全 FP32 对照（512 条有效检索历史） | 1.55e-6 | 9.54e-7 | 4.99 ms |

切换前，全 BF16 融合路径在 32 和 512 条有效历史的用例分别有 13 和 12 个值超出逐元素门限（`atol=0.001, rtol=2.01/128`）。当前两组共 786,432 个输出元素全部通过，LSE 误差保持在 `1e-6` 内。独立短历史测试还发现，原生融合算子在 KV 长度小于 K32 时可能产生 NaN 或错误 LSE。调整后，在空、短和跨页历史用例中对照 FP32 结果，同时对目标缓存和页表做逐元素核对。混合批次包含 0、1、2、31、32、127、513-token 前缀和打乱的物理页，SWA、C1A、C2A 各有 917,504 个输出元素，三组均通过同一逐元素门限，最大绝对误差分别为 `1.28e-6`、`1.98e-3`、`1.60e-3`。真实请求上的 logits、greedy 及 cache 验收仍待八卡测试。

Attention 后的 grouped low-rank O projection 也在同一设备栈上单独复验。算子使用主模型相同的 `npu_transpose_batchmatmul` 和 `[group, group_width, o_lora_rank]` 权重布局。覆盖 1、2、8 个本地 group 以及 32、64、96 个 token；相对 CPU FP32 公式的最大绝对误差不超过 `0.00406`，P99 不超过 `0.00196`，均通过 BF16 门限。 [验证记录](validation/dsv41/grouped-output-projection-910b3.json)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_orthrus_npu.py \
  --config "$MODEL/config.json" --compare-fia
PYTHONPATH=src python scripts/probe_dsv41_sparse_mla.py --valid-topk 512
PYTHONPATH=src python scripts/probe_dsv41_sparse_mla.py --mixed-history
PYTHONPATH=src python scripts/smoke_dsv41_oproj_npu.py --device npu:0
```

添加 `--layer2-weights /path/to/layer2-a2-w8a8.pt` 可复测第 2 层样本。该用例使用原版 Flash 的 A2 W8A8 权重转换样本和合成隐藏状态，CPU 反量化并生成 Q/KV；其余用例使用合成的分页 KV。复验使用 `torch 2.10.0+cpu` 和 `torch_npu 2.10.0.post4`。实际 prompt 激活、完整 1017 量化服务和吞吐尚待验收。

## 原生 INT8 检索

目标 index cache 已按 C2A 或 C1A 的粒度写好。Draft 调用 QLI 时直接传入可读行数 `floor(anchor / ratio)`，在这段截断后的前缀内让全部 K 行检索。调用中的 `cmp_ratio=1` 表示坐标已经是压缩行号。C1A 候选池由 Full 层返回并传给 Reindex，输出会校验候选成员资格并去重。实测当候选行少于 Top-K 时，原生算子可能返回额外的行 0。

在 `nightly-releases-v0.32.0rc` 的 Ascend 910B3 上，K32 的 C2A 和 C1A 各验证 96 个查询，另核对 64 组 C1A 候选池。覆盖空历史、奇偶压缩边界、乱序物理页、候选截断与不满 Top-K 的末块。目标 INT8 key、scale 和页表逐元素一致。C1A 用 Top-K 512 复验了 64 和 2048 块候选池。2048 块用例覆盖 16,407/16,413 token 历史的候选截断和 Reindex，所有逻辑索引与参考一致。 [检索实现](../src/blockspec/vllm_ascend/dsv41_native_index.py)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_native_index.py --candidate-blocks 2048
```

## 原生缓存与引擎接入

MRV2 接入以插件形式注册 `OrthrusDSV41DraftModel`，并为该架构选择 Orthrus speculator。调度沿用并行 draft 的 anchor/MASK 布局和标准 verifier，模型语义由 Orthrus draft 实现：每轮绑定目标模型的 SWA、压缩 KV 与 indexer key 页，使用 PAD slot 和 `skip_ring_state_update` 保持目标 cache 只读；Top-K 与临时块状态只存在于本轮 draft。draft 只持有 diffusion Attention/indexer 参数，共享目标 embedding、mHC、Engram、norm、MoE/MLP 和 LM head。初始化 head 可按正式张量契约导出并严格 preflight，用于验证装载和执行链路；它不用于评估接受率。

当前 CPU 契约、head 导出/重载、分页与回滚状态、Attention/indexer/O-projection 独立 NPU 路径和 v0.32 插件导入均已覆盖。MRV2 元数据复验确认，parallel input prep 中包含临时 K32 的长度会在构造目标历史时扣除，所有 draft slot 使用 PAD，且禁用 ring 状态更新。 [验证记录](validation/dsv41/mrv2-readonly-metadata.json)

执行路径直接复用 MRV2 已在 CPU 侧计算好的 `max_seq_len`，并按压缩比例得到全局历史上界。这样 Attention 和 QLI 不再逐层对 NPU 长度张量调用 `.item()`；一次 40 层起草最多避免 40 次 Attention 和 8 次 indexer 的设备到主机同步。显式长度路径已在 910B3 上复验 C2A、C1A Full/Reindex 和目标页只读性。

40 层原生 Attention/indexer 状态链也已在 910B3 上连续执行两轮。用例覆盖 2 个 SWA、4 个 Full、30 个 Reuse、4 个 Reindex 和全部 8 个 indexer source，并同时放入空历史与 544-token 历史请求。C1A 候选池在 Full/Reindex/Reuse 间完整传递，执行前后所有目标 SWA、压缩 KV、INT8 index key、scale 和页表逐元素不变。当前实现的第二轮同进程稳态耗时为 673.19 ms；该数值只包含合成 Attention/indexer 链，不包含投影、共享主干、调度和 verifier，不能代替整机吞吐。 [验证记录](validation/dsv41/runtime-chain-910b3.json)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_runtime_chain.py \
  --config "$MODEL/config.json" --device npu:0 --tp 8
```

使用官方 1017 维度和 dummy BF16 权重，完整 40 层 head 已在 910B3 上完成 TP1、TP4 与 TP8 构造。TP1 和 TP8 均把 336 个 checkpoint 张量完整送入自定义 loader；TP8 还给 sink、Q 投影和两段 O 投影写入坐标签名，逐 rank 验证实际分片内容。TP4 和 TP8 均完成 HCCL collective 及逐 rank 名称/本地形状检查。TP8 每 rank 为 997,529,920 个参数、1,995,060,480 bytes，`wo_a` 从 TP1 的 `[8,4096,1024]` 分为每 rank `[1,4096,1024]`。热路径使用加载后常驻的三维原生布局，不在每轮转置权重。 [验证记录](validation/dsv41/head-construction-910b3.json)

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

固定镜像内已实际安装并发现 `blockspec_orthrus` entry point；两节点启动脚本使用的 headless DP、TP8、Engram、speculative config、lazy safetensors 和 SWA 参数均存在于该镜像的 `vllm serve --help=all`，不是只按文档推测。 [验证记录](validation/dsv41/deployment-cli-v032.json)

### 实验性单机 A2 smoke

W4A8 仓库约 529GB，其中 Engram 由 CPU offload 承担，因此可尝试用一台 8×64GB A2 做受限启动；但这不是官方验收拓扑，也不能替代上述两节点测试。单机入口固定 TP8/EP8、16K、batch 1、512 batched tokens、eager 和 0.88 显存利用率。先运行 target-only，确认权重、Engram 与 HBM 预算后，才以初始化 head 检查 Orthrus 链路。

```bash
export LOCAL_IP=<this-node-hccl-ip>
export NIC_NAME=<hccl-interface>
export ACK_EXCLUSIVE_NODE=1

# 第一步：仅启动目标模型。
MODE=target scripts/deploy_dsv41_single_a2.sh

# 第二步：target-only 退出并清理容器后，显式指定 head。
MODE=orthrus HEAD_HOST=/data/disk/wangzili/models/orthrus-dsv41-head-v1 \
  scripts/deploy_dsv41_single_a2.sh
```

该入口明确标为 smoke：若单机因 HBM、EP8 或运行时 workspace 失败，应回到官方两节点 DP2/TP8/EP16，不通过 CPU offload 主干或逐层换入来伪造可部署结论。

完整服务验收仍需实际 V4.1 checkpoint，验证 target-only 启动、40 层 draft 前向、greedy 一致性、拒绝回滚、并发换序与显存预算。TP8 head 构造和通信已通过，不再是剩余项。

原生融合路径固定依赖已验证的 post7 SparseFlashMLA ABI。较旧的 post4 栈在 A2/A3 上不支持 `ori_sparse_indices`。167 上的 vLLM-Ascend 0.30 镜像虽然带 post7 Python 包，其本地 CANN kernel 缺少该组合的 tiling key，也不能作为此路径的部署基线。部署必须固定完整镜像 digest、CANN 和驱动，并运行 `probe_dsv41_sparse_mla.py` 作为 preflight。 [原生检索实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/indexer.py) · [原生 Attention 调用](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)

### 单层在线更新

单机 eager 可以显式设置 `ORTHRUS_DSV41_ONLINE_STRIDE`。实验入口限定 DP1、单请求，更新零起始第 39 层的 diffusion attention。它沿用第 36 层的 Top-K，并在起草时复制本轮重放所需的 KV 行。verifier 返回后核对请求、anchor 和候选 token，只用实际验证到的位置做全词表 TV 蒸馏。第 0–30 行的 draft logits 预测位置 1–31，与研究实现的 next-token 对齐方式一致。

当前使用 rank 0 的 CPU 参考反传。最后一层冻结的 W4A8/INT8 MoE 逐专家反量化成 FP32，训练完成后将 BF16 head 和 FP32 sink 发布到各 TP rank。运行中的主模型仍使用原生 W4A8 前向。CPU 参考与 NPU 量化前向存在计算精度差异，单机真实服务的反馈对齐、权重发布及更新后请求仍需八卡验收。此入口用于验证功能，更新与请求同步。

CPU 单测验证了可微重放、目标历史不变及更新后 loss 下降。使用真实 1017 第 39 层和 LM head，配合合成边界与教师 logits，连续两步更新的 TV 为 0.84515、0.78542，Q 投影参数发生变化。这个用例验证梯度和权重更新，线上续训效果需用真实请求确认。61 上的固定镜像已通过 MRV2 Orthrus 配置检查。最新 K32 Attention 和目标缓存复验结果见上表。

`scripts/watch_dsv41_single_a2.py --online --offline-handoff` 会在下载完整、固定镜像可用且本机八卡空闲后，依次执行 target-only、Orthrus greedy 对照、离线 head 更新并重载，以及在线发布 smoke。Orthrus 验收检查 verifier 实际收到候选，并抽样检查 draft 前后目标缓存和页表一致。每一步结束后停止本次测试创建的容器。
