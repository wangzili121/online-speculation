# DSV4 Orthrus 验证入口

先用少卡排除问题，再用八卡验收完整模型。下面的命令在已安装本仓库、已应用固定版本补丁的容器内执行；脚本不会启动服务、自动选卡或停止别人的任务。

## 验收范围

**我们负责 Orthrus 适配正确，不负责修复主模型原有的数值差异。**

- 已在不运行 draft 的主模型路径中复现、并完成对应案例归因的差异，记录为上游已知限制，不作为适配验收的阻塞项，不再继续追查其内部根因。
- 尚未归因的差异不能一并豁免；特别是并发、请求错位、历史边界或回滚问题，仍须确认是否由适配引入。
- 原始输出、严格对照的失败结果和退出码保留。单次严格对照是否一致，与 Orthrus 适配是否通过，是两个独立结论，不能修改原始结果来代替归因。

适配验收重点是：在相同输入、历史 KV 和权重下，draft 的 SWA/C4/C128 计算与参考实现一致；历史选择和块内双向可见范围正确；draft 不写主模型 cache；在相同验证 logits 下，候选接受、拒绝补采样和输出提交符合算法。另须在真实 engine 中覆盖不同接受长度、分页边界、并发、取消及拒绝后的 KV 状态，确认接线和调度没有引入错误。

已有算子和 sampler 单测只覆盖其中一部分，不能代替真实 engine 或完整 43 层模型测试。初始化 head 可以验证链路，但低接受率不能证明整块接受路径已覆盖；训练 head 的收益另行验收。

## 不加载模型

```bash
python scripts/validate_dsv4_readiness.py --stage cpu --output local/check-cpu
python scripts/validate_dsv4_readiness.py --stage npu --output local/check-npu
```

`cpu` 检查 head 契约、部署保护、数学和接口回归；`npu` 只使用容器可见的 0 号卡，检查真实 QLI、rejection sampler、分页 FIA、8/16/32 heads、SWA/块内合并及 C4 稀疏路径。NPU 检查不加载模型权重。

每阶段保存 `report.json` 和各检查日志，包含源码哈希、runtime lock、命令及退出码。单项失败后继续收集其他结果，整体返回非零；输出目录必须是新的，避免覆盖已有证据。`--timeout` 控制单项超时，默认 1,800 秒。

## 少卡服务对照

使用 `prepare_dsv4_smoke_model.py --materialize` 从实际 W8A8 权重保留前四层，再用 `export_dsv4_orthrus_head.py` 导出初始化 head。四层保留 SWA、C4、C128，但不是完整模型，也不用于评价 head 质量。

在同样的 `TP=2`、`VLLM_BATCH_INVARIANT=1`、eager、上下文和并发设置下，依次运行 `serve_dsv4_target.sh` 与 `serve_dsv4_orthrus.sh`。Orthrus 使用 `ORTHRUS_PREFLIGHT_MODE=development`、`ORTHRUS_DRAFT_SAMPLE_METHOD=probabilistic` 和 `ORTHRUS_VERIFY_CACHE_CHECKSUM=1`。

启动脚本在 `VLLM_BATCH_INVARIANT=1` 时同时关闭固定版本的 `multistream_dsv4_dsa_overlap`，两服务必须保持一致。该多流路径的 prefill/decode 使用不同的 norm/quant 顺序，仅设置上游 batch-invariant 开关不足以消除差异；其余 additional config 保留。未开启确定性检查时，不改变原有多流配置。

```bash
# target-only 服务启动后
python scripts/validate_dsv4_readiness.py --stage service \
  --base-url http://127.0.0.1:18901 --model dsv4-target \
  --eos-token-id 1 --output local/check-target

# 停止自己启动的 target-only 服务，启动 Orthrus 后
python scripts/validate_dsv4_readiness.py --stage service \
  --base-url http://127.0.0.1:18901 --model dsv4-orthrus \
  --eos-token-id 1 --output local/check-orthrus

python scripts/validate_dsv4_service.py compare \
  --baseline local/check-target/snapshot.json \
  --candidate local/check-orthrus/snapshot.json \
  --output local/check-comparison.json
```

EOS 必须取实际模型的配置；本次 0731 权重是 `1`。不要套用其他模型的特殊 token。检查包括分页边界、长输出、串行重复、并发换序、同 seed 并发、停止、取消以及取消后重放。回放保存请求 ID 和第一个不同的 token。设置 `ORTHRUS_TRACE_SAMPLING=1` 与 `ORTHRUS_TRACE_FILE=/tmp/orthrus-{pid}.jsonl` 可追踪 draft 随机数偏移；只在诊断时开启。

固定 seed 跨引擎不要求逐 token 相同；同一服务的批次变化也可能影响量化数值。回放不一致是待分析项，不能直接等同于采样分布错误。当前检查保留严格失败结果，不通过放宽阈值获得“验收通过”。首 token 的分布仅作诊断，不能代替 rejection 分布验证。

若 greedy 不同，可在 target-only 和 Orthrus 服务上分别运行 `scripts/check_dsv4_greedy_divergence.py`，传入 `--base-url`、`--model`、`--baseline`、`--candidate`、`--output`。加 `--replay-original` 同时重放原始请求。该检查比较首处分歧的共同前缀、原生 prefill 的 top-5 logprobs 和未来 token 不可见性，帮助区分主模型路径差异与投机状态问题。对应案例确认来自主模型后即可按上述范围记录，不需要继续修复主模型；其他案例不能自动放行。

`trace_dsv4_native_layers.py` 可在独立离线 engine 中逐层记录相同位置的激活，接收 `--model`、`--baseline`、`--candidate`、`--output` 和 `--tp`。`--no-dsa-overlap` 做原生配置对照；`--benchmark-prolog` 在移除诊断 hooks 后，比较真实权重下 Q/KV 共用量化前后的结果及耗时。该脚本会加载模型、使用指定 TP 数量的卡，不能与服务同时占用同一组卡。

## 真实引擎状态检查

```bash
# 使用独占的两张可见卡；保留镜像原有 PYTHONPATH，不能覆盖 CANN 的 acl 路径。
PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}" \
VLLM_BATCH_INVARIANT=1 ASCEND_RT_VISIBLE_DEVICES=0,1 \
python scripts/check_dsv4_engine_state.py \
  --model local/models/dsv4-4l --head local/models/head-4l --tp 2 \
  --kv-cache-mib 1024 --output local/check-engine-state \
  --natural-baseline local/check-orthrus/snapshot.json
```

此脚本自行启动和退出离线 engine，不需要 API 服务；仍使用真实模型、Orthrus proposer、验证器和 scheduler。只在测试 worker 中，真实 draft 运行后将候选替换为可控序列，配合 greedy 单 token 白名单，触发 0–31 个候选被接受、连续拒绝和整块接受。四个请求使用不同 token、输入长度和输出预算，并测试换序、取消后的新请求。白名单输出不能证明任意请求的模型数值正确，因此另做两类检查：

- 逐轮核对请求 ID、候选传递、接受数量、anchor/MASK、有效历史长度和各 cache group 的分页表；两个 TP rank 的状态轨迹必须相同。
- 在同一串行调度及同一候选序列下，对照主模型完整词表 logits 的字节哈希：先运行两次完全绕过 Orthrus proposer 的 token-only 参考，再运行真实 draft，以及仅跳过 head 计算的诊断路径。首次 token-only 运行预先指定为参考，不能事后挑选匹配的一次；同时保留参考路径自身的重复性结果。正式推理代码不变。

脚本开启全量 cache guard，每层、每种 batch 大小首次全量快照，随后抽样；默认将每 rank KV 池限制为 1 GiB，避免诊断快照和比较临时张量耗尽显存。这是测试配置，不是生产性能配置。`--natural-baseline` 可选，用于再运行无 token 约束的串行/并发文本请求；其中的输出差异单独记录，不能用受控输出通过掩盖。

`state_checks_passed` 代表状态检查，`adaptation_reference_passed` 要求真实路径匹配预定参考且匹配仅跳过 head 的诊断路径；二者通过时脚本返回 0。`reference_repeat_exact`、`all_shadow_logits_exact` 和 `natural_exact` 保留其他对照原始结果，不会因为适配检查通过而改成 true。参考自身不稳定不能豁免适配与预定参考不一致的情况。

每 rank 的逐轮证据保存在 `events-rank*.jsonl`。这不覆盖真实训练 head 的接受率、随机采样高接受率的引擎路径、抢占后的换入换出、物理页强制复用或 graph；完整模型必须另跑。不要把小模型测试成功写成完整生产验收。

## 随机采样与 KV 压力

```bash
PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}" \
VLLM_BATCH_INVARIANT=1 ASCEND_RT_VISIBLE_DEVICES=0,1 \
timeout -k 30s 1800 python scripts/check_dsv4_sampling_state.py \
  --model local/models/dsv4-4l --head local/models/head-4l --tp 2 \
  --num-blocks 256 --waves 2 --tokens 512 --require-preemption \
  --inject-allocation-failure \
  --output local/check-sampling-state
```

同样只在独立测试 worker 安装钩子，正式服务不加载这些钩子。主模型和 draft head 仍实际计算，但两端采样前的 logits 被替换为已知的双 token 概率分布；真实概率采样、标准 rejection、scheduler 和 KV 操作不替换。此测试检验推理接线，不检验模型精度或训练 head 的加速效果。

- 随机轮使用四组互不重叠的 token、不同 temperature/seed；draft 概率随请求和候选位置变化。检查整块接受、拒绝后从正差分布补采样，以及请求换序后 verifier 收到的完整 q 是否对齐。
- 压力轮使用 p=q，让四个请求同时快速增长，每个输出 1536 token。限制实际 KV block 数；加 `--inject-allocation-failure` 时，测试 scheduler 对最多四个请求各返回一次分配失败，之后完全由原生逻辑释放、重算及恢复请求。报告区分故障注入和自然容量压力，不能混称为真实 OOM。
- 物理页复用依据 scheduler 新分配的 block ID，而不是 worker 可能保留的旧分页表项；null page 不计入。每轮继续检查 anchor、有效历史和各 cache group 分页表，两个 TP rank 的事件轨迹必须一致。
- 随机轮和 p=q 压力轮分别检查输出频率，逐 token 的接受前缀及补采样支持集另外校验；不能只看 API 输出 token 的频率。cache guard 仍为各层各 batch 首次全量、后续抽样。

`--require-preemption` 未观察到抢占恢复时返回失败，不能将“请求正常完成”当作抢占验证通过。256 blocks 是四层测试用压力配置，不是完整模型的部署建议；这也不是 CPU offload 或长时间稳定性测试。启动失败后须确认本轮 worker 已退出，再启动下一轮，不要让两轮测试占同一组卡。

## 八卡窗口

1. 确认完整模型和 head 的哈希、固定镜像及代码版本，先确认八卡不被其他任务使用。
2. 换回完整 43 层模型、`TP=8`、production preflight，其他正确性配置保持一致，重复上述 target/Orthrus 对照；出现分歧时仅做必要归因，已证实的主模型差异不阻塞适配验收。
3. 使用 `--cases configs/dsv4/correctness-extended.json --include-long` 增加 32K/128K 测试，并检查服务重启后的重放。
4. 另外验收高接受率下真实 scheduler 的 KV 提交/回滚、1M 上下文、graph 和长时间稳定性。初始化 head 与 sampler 单测不能替代高接受率服务验收。
5. Orthrus 适配正确性通过后再用训练 head 测吞吐和 goodput；适配仍有未解决或未归因的问题时保持 target-only 回滚路径，不启用生产/graph 验收标志。graph 必须独立验收。

八卡只用于完整模型相关测试；算子、head 契约和多数接口问题应提前在前两阶段解决。具体已通过项及未通过项见 [验证记录](DSV4_VALIDATION_20261028.md)。
