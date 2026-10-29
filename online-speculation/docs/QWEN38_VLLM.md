# Qwen3.8 Orthrus 原生 vLLM 接入

**状态：原生接入和在线更新代码已实现，尚未通过真实 27B 的完整端到端验收。**
TP/梯度/反馈等单元测试已通过；原生基线首先遇到 GDN prefill NaN，切换现有 CANN 后端后的完整复测因两卡被其他任务占用而中断。

该入口使用 vLLM-Ascend 加载和执行真实 Qwen3.8 主模型，插件负责 Orthrus 起草和在线续训。
主模型验证候选、执行 rejection sampling、维护正式缓存，均走原生引擎；不是将 Torch 参考执行器包装成服务。

## 实现边界

当前入口首先用于打通功能：BF16/FP32、eager、同步调度、`max_num_seqs=1`，不支持量化、PP、DP、DCP、PCP、LoRA、图模式或多模态输入。CANN prefill 选项只接收 BF16。
TP 分片的 query/KV 头和 GDN 头数须能整除 TP；27B 首选 TP=2。

- Embedding、norm、MLP 和 LM head 引用当前 TP rank 已加载的主模型参数，不重新加载完整 checkpoint。
- 每层独立复制 draft mixer。Full Attention 块内双向；GDN 按 Orthrus Qwen3.5 扩展，在块内执行因果递推。
- 现阶段历史来源为 `replay_native_weights`：用共享 AR 权重重放已提交 token，生成插件自己的只读历史快照。**尚未直接共享原生分页 KV/GDN cache，因此存在额外历史计算和存储。**
- 候选及其真实生成分布 q 交给原生 rejection sampler；拒绝后的 token 不进入插件历史，draft KV 也不会成为正式 KV。

算法插件通过 `custom_class` 加载，但普通 custom proposer 接口不提供神经网络生命周期和验证反馈。
`qwen38_runtime.py` 因此为 v0.26 NPU runner 安装范围受限的兼容 hook：保留和取得 proposer、输入准备、起草调度、验证反馈。
其中，无候选的 prefill 会将 GDN draft-token 计数清为 -1，避免沿用 profiling 或上一轮的计数、误走投机状态读取。
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
  --online --last-layers 1 --update-stride 16 --port 8000
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

## 验证方式

`scripts/validate_qwen38_vllm.py` 使用原生 `vllm.LLM`，分别发送 greedy 和 temperature=1 请求，再通过具名 worker RPC 收集起草、验证、接受和更新计数。
设置 `ORTHRUS_VALIDATE_TARGET_ONLY=1` 为无投机基线，`ORTHRUS_VALIDATE_OUTPUT=/work/evidence/run.json` 保存结果。
`ORTHRUS_VALIDATE_EXTENDED=1` 增加较长输出、跨 chunk 的 prefill 和 top-k/top-p 采样；`ORTHRUS_VALIDATE_BASELINE=/work/evidence/target.json` 直接逐 token 比较同后端的 greedy 结果。
测试时用 `--update-stride 1`，保证短请求覆盖在线更新；正式配置不建议每步训练。
采样模式不能要求与 target-only 的同种子输出逐 token 一致；greedy 才按同一输入做逐 token 对照。

CPU 测试覆盖权重别名与 AR 对照、原生参数类型复制、inference tensor 输入反向、双 TP 前向与梯度、全局裁剪、hook 调用顺序和 GDN 分段状态等价性。
真机验收结果另行记录，不将上述单元测试当作服务或性能验收。

本次详细边界和证据见 [2026-10-30 验证记录](QWEN38_VLLM_VALIDATION.md)。
