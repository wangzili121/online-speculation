# Qwen3.8 Orthrus 原生 vLLM 接入

**状态：真实 27B 的 TP2 推理、在线更新及 head 交付链路已打通；扩展验收发现 K32 的数值一致性边界，尚不能宣称全部 greedy 用例逐 token 等价。**
原生 OpenAI API 的流式、排队、取消恢复及跨页输入已通过已有功能检查；当前不宣称生产吞吐收益。具体证据见文末验证记录。
主模型缓存复用已接入；固定 head 和最后四层在线更新已在新缓存路径重新验证，不再通过 AR 重放生成历史。
本轮结论与未通过项见 [验收摘要](QWEN38_ACCEPTANCE_SUMMARY.md)。

PyPTO 的计算路径、算子适配与在线反向边界见 [Orthrus-Qwen3.8 算子需求报告](ORTHRUS_QWEN38_PYPTO_REPORT.md)。

该入口使用 vLLM-Ascend 加载和执行真实 Qwen3.8 主模型，插件负责 Orthrus 起草和在线续训。
主模型验证候选、执行 rejection sampling、维护正式缓存，均走原生引擎；不是将 Torch 参考执行器包装成服务。

## 实现边界

当前真机验收覆盖 BF16、eager、同步调度、`max_num_seqs=1`，不支持量化、PP、DP、DCP、PCP、LoRA、图模式或多模态输入。整模型 FP32 尚未真机验收；CANN prefill 选项只接收 BF16。
TP 分片的 query/KV 头和 GDN 头数须能整除 TP；27B 首选 TP=2。

- Embedding、norm、MLP 和 LM head 引用当前 TP rank 已加载的主模型参数，不重新加载完整 checkpoint。
- 每层独立复制 draft mixer。Full Attention 块内双向；GDN 按 Orthrus Qwen3.5 扩展，在块内执行因果递推。
- 历史来源为 `shared_target_snapshot`：从原生主模型的分页 KV、GDN 卷积及递推状态读取接受前缀，**不再执行 AR 历史重放，也不维护另一套持续追加的 AR cache。** Full Attention 将有效物理页整理为融合 Attention 的连续输入；GDN 仅提取接受位置的工作状态。这里仍有布局转换和临时复制，不是分页 KV 的零拷贝直读。
- 候选及其真实生成分布 q 交给原生 rejection sampler；拒绝后的 token 不进入可读历史，draft KV 也不会成为正式 KV。

`qwen38_ops.py` 为 draft 绑定镜像已有算子：Full Attention 的无梯度分支使用原生 QKV 拆分、Q/K norm、RoPE 融合算子，再以 `npu_fusion_attention` 计算历史与双向块；共享 MLP 使用 packed gate/up 投影和 `npu_swiglu`。
GDN 使用已有 causal convolution、L2 norm、递推和门控 norm，所有工作状态均为自有临时存储。在线反向中不支持 autograd 的融合步骤改用可微分表达式，保持相同的中间精度和舍入位置；Attention 本身仍调用支持反向的融合算子。
这些适配不改变 head 的参数名或交付格式，不新增 NPU kernel，也不表示性能已经最优。

draft 默认跟随请求的 temperature、top-k 和 top-p；greedy 请求直接起草最高概率 token。
有显式 seed 的新请求会独立初始化 draft RNG，避免沿用上一个请求的起草随机状态。
固定 head 下可检查同种子请求复现，但不能要求投机与无投机使用同种子得到相同的采样序列；在线更新改变 head 后也不作这一要求。

`qwen38_history.py` 负责缓存边界：当前最后一个采样 token 是下一轮 anchor，尚未经过主模型，不属于已计算历史。
整块验证后，KV 虽可能已写入被拒绝的位置，draft 只能读到接受前缀末尾；GDN 则从本轮接受位置的 checkpoint 取递推状态，并从卷积行提取对应的历史输入，不能沿用上一轮接受数或读取块末状态。
得到的快照不与主模型缓存共享可写存储。它们既供本轮 draft 使用，也让在线后缀重放不受下一轮验证或请求切换时的 cache 复用影响。

算法插件通过 `custom_class` 加载，但普通 custom proposer 接口不提供神经网络生命周期和验证反馈。
`qwen38_runtime.py` 因此为 v0.26 NPU runner 安装范围受限的兼容 hook：保留和取得 proposer、输入准备、目标缓存元数据、起草调度、验证反馈。
其中，无候选的 prefill 将 GDN draft-token 计数清为 -1；已经进入 decode 的请求则设为 0，并继续向 GDN metadata builder 传递上轮接受位置。
没有新候选不代表没有上轮投机状态：拒绝后的最后一个普通 decode 仍须从接受前缀继续，不能退回第一个状态槽。
这些 hook 只处理本插件，其他 proposer 仍调用原方法；遇到不匹配的方法签名立即报错。
**这是插件加内部兼容适配，不是仅靠上游公开接口即可运行的纯插件。** 本入口不修改安装目录内的 vLLM/vLLM-Ascend 源文件。

当前 A2 后端的 paged FIA 短 decode 路径最多接收 16 个 query。K=32 时，`qwen38_attention.py` 将主模型每层的验证 Attention 切成两个 query 片段，分别调用同一个原生算子。
两片均读取该层正式 KV，但第一片的可见 KV 末端停在自身末端，第二片再推进，因此保留整块因果 Attention 的数学含义。
这不是将 draft 块缩成 K=16，也没有拆成两轮模型验证；GDN 和其他层计算仍处理完整验证块。

镜像的 GDN recurrent kernel 同样将单次长度限制在 16，超过时会直接返回，不能把 K32 整块直接传入。
`qwen38_gdn_backend.py` 将验证块按最多 16 行分段调用原有算子：首段从上轮接受位置的状态开始，后段接着前段的最终状态计算，并保留每个候选位置的状态供下一轮回滚选择。
即使下一轮长度缩短，也先按上一轮真实接受位置选取状态，不把接受数量错误地限制为当前段长。

当前镜像的原生 Triton GDN prefill 在关闭 Orthrus 的基线中也出现 NaN，最早见于第一层 GDN 的状态计算。
因此启动脚本默认显式选择 `--gdn-prefill-backend cann`，将 prefill 切换为镜像自带的 `torch_npu.npu_chunk_gated_delta_rule`；解码和投机验证的 GDN 仍由原生引擎处理。
适配仅处理布局、归一化和状态返回，不新增 kernel；CANN 接口使用 BF16 初始状态，数值舍入与 Triton 不能视为逐位相同，验收时基线与 Orthrus 必须选择同一后端。
`--gdn-prefill-backend native` 可恢复镜像原路径，但不应在已复现 NaN 的环境中直接作为生产配置。
启动脚本默认 `--gpu-memory-utilization 0.65`，为 CANN 工作空间和在线重放留出显存；原有 profiling 不覆盖其全部开销，不能直接把 KV 池开满。
也可使用 vLLM 已有的 `--kv-cache-memory-bytes` 固定每 rank 的 KV 预算，仍执行 profiling forward，但不采用自动预算估值。
真实 27B、TP2、4K、单请求的扩展验收使用每 rank 12 GiB；这不是其他 TP、上下文长度或并发配置的通用预算。

## 在线更新

默认建议从最后 1 层开始，`--last-layers` 可以选择后段层数，`--update-stride` 控制每隔多少个有效验证块更新一次。
训练只使用真正验证到的教师行：接受前缀和首次拒绝的位置；拒绝后的行不监督。
保存的边界表示与历史均脱离梯度，只重放后段 draft，以完整词表 TV loss 更新 FP32 master，再写回执行权重。

TP 的列投影在反向归并输入梯度，行投影在前向归并结果；复制的 norm 梯度单独归并，梯度裁剪使用全局范数。
共享 MLP/LM head 只计算输入梯度，不计算权重梯度。原生 inference tensor 通过冻结线性层适配传递输入梯度，不复制整份共享权重，也不开发新 kernel。
更新在本轮验证完成后同步发布，下轮才使用新 head。当前不是异步训练服务，不保证训练带来吞吐收益。
原生主模型必须在插件生命周期内保持不变，不允许权重热更新。

## 运行

入口针对以下已部署环境开发；其他镜像不能仅凭方法签名就视为已经验证：

```text
quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a
PyTorch 2.10.0 + torch_npu
Transformers 5.14.1
vLLM 0.26.0
```

在有模型和设备映射的容器中，从 `online-speculation` 目录运行：

```bash
python -m pip install --no-deps --no-build-isolation -e .
python scripts/run_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --tp 2 \
  --mask-token-id 1 --block-size 32 \
  --online --last-layers 1 --update-stride 16 \
  --max-model-len 4096 --kv-cache-memory-bytes 12884901888 --port 8000
```

`mask-token-id=1` 仅对应当前复制 AR 参数的功能测试初始化，真实训练 head 必须使用训练时约定的 mask ID。
服务默认监听本机；需要供内网其他机器访问时显式传 `--host 0.0.0.0` 并配置访问控制。
不传 `--online` 则为固定 head。没有 `--head-dir` 时从 AR mixer 复制初始化，只证明链路，不代表已有可加速的预训练 head。

服务使用原生 OpenAI API：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3.8-27B","prompt":"Hello","temperature":0,"max_tokens":8}'
```

可选 `--checkpoint-dir /data/heads`：每 16 次更新保存一份按版本分目录的 draft 参数，频率由 `--checkpoint-every` 控制。
在同一 TP 布局下，用 `--head-dir /data/heads/version16` 重启加载；manifest 校验基模 config 和权重索引 SHA256、TP、K、mask ID 和混合层策略。
这是 draft 权重交付，不是 AdamW 状态的跨进程精确续训；基模文件的完整内容哈希仍需部署侧另行锁定。
已经发布的版本不允许覆盖。验证脚本可将 `--checkpoint-every` 设得较大，在请求全部完成后通过 worker RPC 仅导出一次。

## 验证方式

`scripts/validate_qwen38_vllm.py` 使用原生 `vllm.LLM`，分别发送 greedy 和 temperature=1 请求，再通过具名 worker RPC 收集起草、验证、接受和更新计数。
设置 `ORTHRUS_VALIDATE_TARGET_ONLY=1` 为无投机基线，`ORTHRUS_VALIDATE_OUTPUT=/work/evidence/run.json` 保存结果。
`ORTHRUS_VALIDATE_EXTENDED=1` 增加较长输出、跨 chunk 的 prefill 和 top-k/top-p 采样；`ORTHRUS_VALIDATE_BASELINE=/work/evidence/target.json` 直接逐 token 比较同后端的 greedy 结果。
测试时用 `--update-stride 1`，保证短请求覆盖在线更新；正式配置不建议每步训练。
采样模式不能要求与 target-only 的同种子输出逐 token 一致；greedy 才按同一输入做逐 token 对照。

`ORTHRUS_VALIDATE_SOAK=1` 增加 128-token 输出、精确填满上下文的输入、停止 token 和停止后的请求隔离检查；这是一组有界边界测试，不是长时间稳定性验收。
同时设置 `ORTHRUS_VALIDATE_AUDIT=1`、`ORTHRUS_VALIDATE_REPLAY_STEPS=8`，可在在线测试后对同一份已到达反馈再更新八次，检查 loss 下降、draft 参数变化和主模型 cache 不变；它不是业务接受率或吞吐收益测试。
定位单个用例可设置 `ORTHRUS_VALIDATE_CASES=context-tail`，`ORTHRUS_VALIDATE_LOGITS=1` 保存原始目标 logits 的前四项。
`ORTHRUS_VALIDATE_NGRAM_CONTROL=1 --block-size 4` 运行不加载 Orthrus head 的原生 n-gram 对照。

服务启动后，可运行 `scripts/validate_qwen38_api.py --base-url http://127.0.0.1:8000 --model /models/Qwen3.8-27B --baseline /work/evidence/target.json --output /work/evidence/api.json`。
该脚本覆盖普通请求、流式输出、两个客户端排队、取消后恢复及 1804-token 跨页输入；服务的 `--max-model-len` 至少为 2048。

可用以下入口连续运行原生引擎验收，每个配置使用独立进程，分别保留日志和 JSON：

```bash
python scripts/accept_qwen38_vllm.py \
  --model /models/Qwen3.8-27B --mask-token-id 1 --tp 2 \
  --max-model-len 8192 --output-dir /work/evidence/acceptance
```

该入口覆盖无投机基线、K=2/8/17/32、last-1/last-4 在线更新、head 导出重载，以及受控的整块接受和连续拒绝。
每个主配置先完整运行输出长度、跨页、上下文末端、代码、JSON 和中文用例，再对五种请求边界重复检查在线状态及活跃内存。
采样检查直接记录主模型采样前的 logits，在三个受限 token 上计算参考分布，按条件概率累计期望频数；不依赖 API logprobs 作为独立真值。
last-4 还检查映射的 AR 参数与原生验证 logits 的偏差，并在固定反馈上更新八次；主模型全量权重哈希和缓存只读审计独立检查。
受控候选只用于覆盖接受/拒绝分支，不能据其接受率评价真实 head。
输出目录必须是新目录；已有目标基线可用 `--baseline` 复用，`--blocks 32` 可缩短重复验证。
API 可增加 `--recovery-rounds 5`，连续检查取消、非法请求及后续正常请求恢复。

CPU 测试覆盖权重别名与 AR 对照、原生参数类型复制、inference tensor 输入反向、双 TP 前向与梯度、全局裁剪、hook 调用顺序和 GDN 分段状态等价性。
缓存测试另外覆盖乱序物理页、跨页读取、所有接受位置的选择、全接受与拒绝后的 draft 对照、请求边界校验，以及引擎复用缓存后在线快照仍保持不变；proposer 测试明确禁止调用 AR forward。
真机验收结果另行记录，不将上述单元测试当作服务或性能验收。

本次详细边界和证据见 [2026-10-30 验证记录](QWEN38_VLLM_VALIDATION.md)。
