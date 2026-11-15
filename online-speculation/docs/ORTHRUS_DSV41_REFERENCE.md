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

1. **固定教师和初始化权重。** 使用部署对应的 1017 W4A8 checkpoint、tokenizer 和 K32，当前测试用的 mask token ID 为 `128799`。从目标 Attention 初始化 diffusion 分支，冻结主模型，包含共享的 embedding、Engram、mHC、norm、MoE、LM head，以及目标 compressor 和 indexer K 投影。优化器只收所选层的 diffusion Attention 参数，包含其内部的 Q/KV norm。若训练 diffusion indexer，给其打分另设监督，离散 Top-K 无法把语言模型 KL 的梯度传回检索分数。
2. **对齐监督位置。** 在真实 token 序列中选择位置 `a` 作为 anchor。教师在完整真实序列上跑因果前向，学生输入分别为 anchor 和 31 个 MASK，并读取 anchor 之前的目标历史。对每个 `j = 0…30`，draft 块第 `j` 行与教师在绝对位置 `a+j` 的 logits 比较，两者都预测位置 `a+j+1`。用全词表 forward KL，屏蔽样本末尾和监督范围之外的位置。
3. **先用真实样本做交接检查。** 固定同一个 head 和同一段历史，比较训练前向与服务执行的 draft logits，记录 TV 和 argmax 一致性。做一步反向，核对实际改动的张量是否都在约定训练范围，随后将部署精度下的更新权重交回服务，检查能否加载、起草和通过 verifier。

训练端保存一个 safetensors，包含 336 个 `layers.N.attn.*` 张量。冻结的 draft 层也随 head 打包。交付时同时记录教师 checkpoint、mask token ID 和更新的层号。导出会拒绝缺失或多出的张量、错误形状和非有限值，输出 BF16 权重、FP32 sink 及基模哈希。 `head_shapes` 返回逐张量形状。 [张量契约](../src/blockspec/vllm_ascend/dsv41_contract.py)

```bash
PYTHONPATH=src python scripts/export_dsv41_orthrus_head.py \
  --model "$MODEL" --mask-token-id "$MASK_ID" \
  --from-trained-head "$TRAINED_SAFETENSORS" --output "$HEAD"
PYTHONPATH=src python scripts/preflight_dsv41_orthrus.py \
  --model "$MODEL" --head "$HEAD" --require-trained
```

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

切换前，全 BF16 融合路径在 32 和 512 条有效历史的用例分别有 13 和 12 个值超出逐元素门限（`atol=0.001, rtol=2.01/128`）。当前两组共 786,432 个输出元素全部通过，LSE 误差保持在 `1e-6` 内。独立短历史测试还发现，原生融合算子在 KV 长度小于 K32 时可能产生 NaN 或错误 LSE。调整后，在空、短和跨页历史用例中对照 FP32 结果，同时对目标缓存和页表做逐元素核对。混合批次包含 0、1、2、31、32、127、513-token 前缀和打乱的物理页，SWA、C1A、C2A 各有 917,504 个输出元素，三组均通过同一逐元素门限，最大绝对误差分别为 `1.28e-6`、`1.98e-3`、`1.60e-3`。真实请求结果见[单机 A2 实测](#实验性单机-a2-smoke)。

Attention 后的 grouped low-rank O projection 也在同一设备栈上单独复验。算子使用主模型相同的 `npu_transpose_batchmatmul` 和 `[group, group_width, o_lora_rank]` 权重布局。覆盖 1、2、8 个本地 group 以及 32、64、96 个 token；相对 CPU FP32 公式的最大绝对误差不超过 `0.00406`，P99 不超过 `0.00196`，均通过 BF16 门限。 [验证记录](validation/dsv41/grouped-output-projection-910b3.json)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_orthrus_npu.py \
  --config "$MODEL/config.json" --compare-fia
PYTHONPATH=src python scripts/probe_dsv41_sparse_mla.py --valid-topk 512
PYTHONPATH=src python scripts/probe_dsv41_sparse_mla.py --mixed-history
PYTHONPATH=src python scripts/smoke_dsv41_oproj_npu.py --device npu:0
```

添加 `--layer2-weights /path/to/layer2-a2-w8a8.pt` 可复测第 2 层样本。该用例使用原版 Flash 的 A2 W8A8 权重转换样本和合成隐藏状态，由 CPU 反量化并生成 Q/KV。其余用例使用合成的分页 KV。复验使用 `torch 2.10.0+cpu` 和 `torch_npu 2.10.0.post4`。

## 原生 INT8 检索

目标 index cache 已按 C2A 或 C1A 的粒度写好。Draft 调用 QLI 时直接传入可读行数 `floor(anchor / ratio)`，在这段截断后的前缀内让全部 K 行检索。调用中的 `cmp_ratio=1` 表示坐标已经是压缩行号。C1A 候选池由 Full 层返回并传给 Reindex，输出会校验候选成员资格并去重。实测当候选行少于 Top-K 时，原生算子可能返回额外的行 0。

在 `nightly-releases-v0.32.0rc` 的 Ascend 910B3 上，K32 的 C2A 和 C1A 各验证 96 个查询，另核对 64 组 C1A 候选池。覆盖空历史、奇偶压缩边界、乱序物理页、候选截断与不满 Top-K 的末块。目标 INT8 key、scale 和页表逐元素一致。C1A 用 Top-K 512 复验了 64 和 2048 块候选池。2048 块用例覆盖 16,407/16,413 token 历史的候选截断和 Reindex，所有逻辑索引与参考一致。 [检索实现](../src/blockspec/vllm_ascend/dsv41_native_index.py)

```bash
PYTHONPATH=src python scripts/smoke_dsv41_native_index.py --candidate-blocks 2048
```

## 原生缓存与引擎接入

MRV2 接入以插件形式注册 `OrthrusDSV41DraftModel`，并为该架构选择 Orthrus speculator。调度沿用并行 draft 的 anchor/MASK 布局和标准 verifier，模型语义由 Orthrus draft 实现：每轮绑定目标模型的 SWA、压缩 KV 与 indexer key 页，使用 PAD slot 和 `skip_ring_state_update` 保持目标 cache 只读；Top-K 与临时块状态只存在于本轮 draft。draft 只持有 diffusion Attention/indexer 参数，共享目标 embedding、mHC、Engram、norm、MoE/MLP 和 LM head。初始化 head 可按正式张量契约导出并严格 preflight，用于验证装载和执行链路；它不用于评估接受率。

当前 CPU 契约、head 导出/重载、分页与回滚状态、Attention/indexer/O-projection 独立 NPU 路径和 v0.32 插件导入均已覆盖。MRV2 为 draft 使用独立的元数据构造器，只取目标页表、anchor 之前的历史长度和 RoPE，关闭目标 Attention、indexer、compressor 的算子元数据构造。所有 draft slot 使用 PAD，且禁用 ring 状态更新。块内双向可见范围由 draft Attention 前向计算。 [验证记录](validation/dsv41/mrv2-readonly-metadata.json)

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

单机运行日志显示，挂载 Engram 的主机内存每 rank 约 25.75 GiB，主模型与 draft head 的设备权重每 rank 约 38.16 GiB。按固定镜像的 INT8/FP32 表布局估算，权重全放设备约需每卡 63.91 GiB，已超过运行时可见的 60.96 GiB。这套配置使用注册主机内存，NPU 查表时只读命中行。

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

在单台 8×64 GB A2 上，实际 W4A8 checkpoint 已通过 target-only 启动和初始化 head 的 K32 投机推理。短提示词和数百 token 提示词各生成 40 个 token，greedy 输出逐 token 一致。verifier 共处理 1,116 个候选，接受 44 个，draft 前后抽样检查的目标 KV 和页表一致。用真实请求反馈微调最后一层后，导出 head 可以通过契约检查并在八卡重载。短请求仍逐 token 一致，较长请求在第 17 个输出 token 出现分叉。

回放该用例的候选时跳过 draft 前向，主模型的批量 verifier 在同一位置仍选出 token 5136，top-2 logits 为 32.75 和 32.50。target-only 在该位置选出 token 14。回放覆盖了首次分叉，后续生成因候选耗尽结束。分叉出现在 verifier 的打分阶段，具体算子仍待定位。生产场景的严格 greedy 一致性、并发、长上下文和性能仍待验收。

#### 原生 verifier 对照

同一镜像及 W4A8 checkpoint 下，使用原生 MRV2 `ngram_gpu` proposer 接口喂固定候选，服务只载入主模型。下表三次测量的前 16 个输出 token 相同。批量验证在同一轮保留前两个候选 `[21361, 41660]`，后缀对照把余下 29 个候选换成 token 263。

| 控制组 | 验证行数 | token 14 的 logit | token 5136 的 logit | 第 17 个输出 token |
|---|---:|---:|---:|---:|
| 主模型逐步解码 | 1 | 32.25 | 32.25 | 14 |
| 原生 verifier，原候选 | 32 | 32.00 | 32.25 | 5136 |
| 原生 verifier，替换候选后缀 | 32 | 32.25 | 32.25 | 14 |

独立脚本再跑一遍，三组前 16 个 token 仍相同，第 17 个依次为 5136、14、5136。两次实验均在原生批量 verifier 复现了与逐步解码的排序差异，具体获胜 token 随运行而变。替换后缀的对照也观察到分数变化。要区分批量数值路径与 Attention、缓存边界，需要同输入重复测量，并逐层比较目标行的激活。 [独立复现实测](validation/dsv41/native-verifier-repro-910b3.json)

在已挂载真实 checkpoint 和八张 A2 的固定镜像内，从仓库目录运行下列命令。脚本用临时 vLLM 插件喂候选、抓取目标 logits，并在两次启动结束后清理自身服务。输出包含两个服务日志、`trace.jsonl` 和 `report.json`。 [复现脚本](../scripts/repro_dsv41_native_verifier.py) · [候选 fixture](validation/dsv41/native-verifier-candidates.json)

```bash
python3 scripts/repro_dsv41_native_verifier.py --model "$MODEL" \
  --local-ip "$LOCAL_IP" --nic "$NIC_NAME" --output-dir local/native-verifier-repro
```

`--layer-trace` 会把每组控制重复一次，并捕获逐层激活和前两个 SWA 层的 Q、可见 KV 与原始输出。默认抓取第 434 个零起始位置，`--trace-position` 可指定前缀中的其他位置。目标模型连续两次逐步解码时，已观察到第 0 层 Attention 的输入行一致，Attention 输出最早出现差异，最大绝对差为 0.015625。

八卡窗口较短时可加 `--single-engine`。脚本复用一次模型加载，在同步调度器中把逐步组的候选清空，批量组仍走原生 `ngram_gpu`。输出会核对目标位置实际使用一行或 32 行，并记录每条响应中第一个不同的 token 位置。

在四张 A2 上，用实际第 0 层的 `wo_a` 和 `wo_b` 分片构造 BF16 输出投影，固定各 rank 的输入后，rank 0 的本地投影在单行和 32 行之间逐元素一致。差异出现在 HCCL AllReduce 之后。每种形状重复 32 次，默认模式下有 24 次单行结果和 32 次批量结果偏离各自的首次值，跨形状最大差为 0.015625。设置 `HCCL_DETERMINISTIC=strict` 后，重复值与跨形状结果均逐元素一致。 [独立输出投影探针](../scripts/probe_dsv41_oproj_hccl.py)

同一探针加 `--timing-iterations 128`，测两轮输入复制与 AllReduce 的设备时间。单行在默认模式为 0.20–0.22 ms，严格模式为 0.28–0.29 ms。32 行分别为 0.26–0.29 ms 和 0.27–0.28 ms。这组结果只反映四卡通信微基准。

单机部署入口会将宿主机显式设置的 `HCCL_DETERMINISTIC` 传给容器内的服务。实际八卡模型打开 `strict` 后，两次逐步解码逐 token 一致。原候选和替换后缀各跑两次，四次 K32 验证在同一输入行上得到相同分数，仍在第 17 个输出 token 与逐步解码分叉。该位置逐步解码的 token 14 和 5136 分数均为 32.00。K32 中 token 5136 为 32.25，token 14 仍为 32.00。

该行的首层滑窗 Q、原始 Attention 输出及第一层完整输出都逐元素一致。可见 KV 有一个元素相差 `1.19e-7`。第一处传播到层输入的差异出现在零起始第 1 层的 Engram 注入后，2 个元素有差异，最大绝对差为 `3.05e-5`。随后的 Attention 输出最大绝对差达到 `0.015625`。该测试使用原生 `ngram_gpu` 和固定候选，服务只加载实际 W4A8 主模型。 [八卡严格模式实测](validation/dsv41/native-verifier-hccl-strict-910b3.json)

进一步追踪时，两组的 Engram 输入、查表结果、mask 和 `wkv` 投影全部逐元素一致。用捕获的张量在单卡重跑，FP32 点积归约出现形状相关差异。用现有的 Ascend `npu_reduce_sum_batch_invariant` 替换这一步后，真实输入行在 1、32、65、256、419 和 512 行的门控输出逐元素一致，其他行填入独立输入。单独归约的设备时间约为 45–47 µs，原路径约 17–27 µs。 [局部归约补丁](../runtime/vllm_ascend_v0_32/dsv41_engram_batch_invariant.patch)

八卡回放使用局部补丁后，第 1 层 Engram 输出和 Attention 输入均逐元素一致。第 1 层 Attention 输出仍相差最多 `0.01171875`，第 17 个 token 仍分叉。逐步解码与原生 K32 组各自可重复，改变 K32 的未来候选保持同样结果。这个补丁用于隔离 Engram 的数值差异，单机部署入口保持原配置。 [局部归约八卡实测](validation/dsv41/native-verifier-engram-invariant-910b3.json)

在第 1 层的 SWA 算子入口，目标位置 434 的 Q 逐元素一致。128 个可见 KV 行中，位置 420、423、429、432 有 503 个元素不同，最大绝对差为 `0.015625`。单卡使用实模抓取的 Q、历史 KV 和该层实际 sink 重放，固定 KV 时单行和 K32 的输出逐元素一致。换用 K32 路径的历史 KV 后，923 个输出元素不同，最大绝对差为 `0.001953125`，两组输出与各自的八卡记录逐元素一致。

追踪最早不同的历史位置 420 时，rank 0 的第 0 层 Q、可见 KV 和原始 SWA 输出在两种形状下逐元素一致。完整 Attention 输出有 90 个元素不同，最大绝对差为 `0.0078125`。该差异随后进入第 1 层并改变位置 420 写入的 KV。复现脚本的 `--trace-projection` 可继续抓取各 TP rank 的逆 RoPE、低秩投影、本地输出和归约结果。

八个 rank 的 Q 和可见 KV 都一致。rank 5 的 SparseFlashMLA 输出有一个 BF16 元素相差 `0.000244140625`，差异在输出投影后扩大。单卡用实际输入和 sink 逐元素复现了两种八卡结果。把 32 个 query 作为各含一行的序列提交到同一次算子调用，每行仍读同一张页表，使用自己的因果长度。单卡输出与单行解码逐元素一致。 [单 query 元数据补丁](../runtime/vllm_ascend_v0_32/dsv41_swa_single_query.patch)

八卡同时启用 Engram 归约补丁、上述 SWA 补丁和 HCCL 严格模式后，位置 420 的原始 SWA 与输出投影在八个 rank 上都逐元素一致。四组原生 K32 与逐步解码在原先第 17 个 token 的用例上通过，完整 40-token 输出的首次分叉出现在第 31 个 token。补丁当前用于单请求 eager 定位，尚未达到端到端逐 token 验收要求。 [SWA 补丁八卡实测](validation/dsv41/native-verifier-swa-row-invariant-910b3.json)

第 31 个 token 对应的输入行在第 0 层 `Wq_a` 出现一个 BF16 舍入差异，经过动态量化后变成一个 INT8 输入值的变化，继续传播到数百个 Q 元素。单卡使用 ND 权重和 `npu_mm_batch_invariant` 消除了该行的投影和量化输入差异。应用到八卡目标模型的 Q 和 KV 投影后，原生 K32 的首次分叉出现在第 20 个 token。这个投影试验未启用到服务配置。

#### Draft 缓存读取

实模联调定位到三处可能复制 3.57 GiB KV 张量的操作，分别是缓存审计的展平、短历史读取的高级索引和压缩历史归一布局。审计和读取现在按实际 stride 访问原始存储，只读取采样值或选中行。SparseFlashMLA 支持页间 stride，可以直接使用每页内部连续的目标 KV。页内有 padding 时，读取选中行后使用 FP32 计算。

带 padding 的 64 MiB 缓存单卡实测中，新旧读取结果逐元素一致，128 行读取的峰值临时显存约从 64 MiB 降到 16 MiB。缓存抽检固定 65,536 个值。页间 stride 的短历史和融合算子输出均与连续页对照逐元素一致，页内 padding 的 FP32 路径对参考值最大绝对误差为 `2.38e-7`。

全局 `VLLM_BATCH_INVARIANT=1` 会强制切到 ND 权重格式。固定镜像的 W4A8 MoE 在 warmup 调用 `aclnnGroupedMatmulSwigluQuantWeightNzV2` 时要求五维 NZ 存储，实际收到一维形状，导致启动失败。局部补丁保留 W4A8 的既有权重布局，只替换 Engram 点积归约。 [vLLM-Ascend batch invariance 实现说明](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/batch_invariance.md)

原生融合路径固定依赖已验证的 post7 SparseFlashMLA ABI。较旧的 post4 栈在 A2/A3 上不支持 `ori_sparse_indices`。167 上的 vLLM-Ascend 0.30 镜像虽然带 post7 Python 包，其本地 CANN kernel 缺少该组合的 tiling key，也不能作为此路径的部署基线。部署必须固定完整镜像 digest、CANN 和驱动，并运行 `probe_dsv41_sparse_mla.py` 作为 preflight。 [原生检索实现](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v41/indexer.py) · [原生 Attention 调用](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/attention/dsa_v41.py)

#### 已更新 head 的原生 verifier 对照

在单机 8×64 GiB A2 上用 `Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8` 跑 TP8/EP8，使用 eager、同步调度、单请求、显存利用率 0.88 和 HCCL strict。测试使用原生目标模型路径，未启用上述诊断补丁。Draft head 从初始化权重出发，对最后一层做过一次离线更新，用于验证更新后权重的加载与运行。

先用 Orthrus 生成两个各 40 token 的回答并记录实际提出的候选，再让原生 `ngram_gpu` verifier 按相同位置回放这些候选。两组的 40 轮候选、80 个输出 token 和提出／接受计数均一致。38 轮进入验证，共提出 1,178 个候选，接受 41 个。Orthrus 的目标缓存审计全程通过。原生逐步解码与 K32 验证之间仍有数值分叉，逐 token 生产验收仍待解决。 [复现脚本](../scripts/probe_dsv41_matched_candidates.py) · [八卡实测](validation/dsv41/orthrus-matched-verifier-910b3.json)

### 单层在线更新

单机 eager 可以显式设置 `ORTHRUS_DSV41_ONLINE_STRIDE`。实验入口限定 DP1、单请求，更新零起始第 39 层的 diffusion attention。它沿用第 36 层的 Top-K，并在起草时复制本轮重放所需的 KV 行。verifier 返回后核对请求、anchor 和候选 token，只用实际验证到的位置做全词表 TV 蒸馏。第 0–30 行的 draft logits 预测位置 1–31，与研究实现的 next-token 对齐方式一致。

当前使用 rank 0 的 CPU 参考反传。最后一层冻结的 W4A8/INT8 MoE 逐专家反量化成 FP32，训练完成后将 BF16 head 和 FP32 sink 发布到各 TP rank。运行中的主模型使用原生 W4A8 前向。这条更新路径与请求同步，用于验证反馈、反传和发布。

CPU 单测验证了可微重放、目标历史不变及更新后 loss 下降。使用真实 1017 第 39 层和 LM head，配合合成边界与教师 logits，连续两步更新的 TV 为 0.84515、0.78542，Q 投影参数发生变化。这个用例验证梯度和权重更新，线上续训效果需用真实请求确认。61 上的固定镜像已通过 MRV2 Orthrus 配置检查。最新 K32 Attention 和目标缓存复验结果见上表。

单机八卡实测中，从真实 verifier 捕获的两行有效反馈完成了一步离线更新，KL 从 2.3541 降至 0.0378，部署精度下 7 个末层张量发生变化。CPU 重放与捕获的 NPU draft logits 平均 TV 为 0.00390，两行 argmax 一致。在线实验连续发布了 13 次单层更新，两条请求各生成 12 个 token，均与 target-only 基线对应前缀一致。这组实验只覆盖链路功能和短样例。

#### 单请求分段计时

单机 TP8/EP8、eager、Engram CPU offload，使用初始化 head，每三轮尝试一次在线更新。预热后，419 个输入 token 的请求生成 12 个 token 用时 51.02 s，其中两次 rank 0 CPU 更新共 42.63 s，两次发布共 1.46 s。CPU 更新计时包含冻结专家权重读取、反量化、前后向及优化器执行。

| 阶段 | 单次中位耗时 | 样本数 |
|---|---:|---:|
| 主模型批量 verifier 前向 | 278 ms | 7 |
| Draft 前向及采样 | 546 ms | 8 |
| Verifier 采样 | 0.19 ms | 7 |
| Engram offload 查表，主模型 | 0.23 ms | 16 |
| Engram offload 查表，draft | 0.24 ms | 16 |
| CPU 最后一层更新 | 21.32 s | 2 |
| 发布到 TP8 | 729 ms | 2 |

设备项用 rank 0 的 NPU 事件计时，Engram 项只覆盖单张表在该 rank 上的查表和反量化，部分计时区间互相嵌套。CPU 更新和发布用主机墙钟时间。该样例的在线更新开销主要在 CPU 更新。推理路径中，draft 前向和批量 verifier 的耗时量级高于 Engram 查表。 [计时记录](validation/dsv41/tp8-online-timings-910b3.json)

`scripts/watch_dsv41_single_a2.py --online --offline-handoff` 会在下载完整、固定镜像可用且本机八卡空闲后，依次执行 target-only、Orthrus greedy 对照、离线 head 更新并重载，以及在线发布 smoke。Orthrus 验收检查 verifier 实际收到候选，并抽样检查 draft 前后目标缓存和页表一致。每一步结束后停止本次测试创建的容器。
