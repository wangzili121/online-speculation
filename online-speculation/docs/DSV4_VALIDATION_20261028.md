# DSV4 Orthrus 验证记录（2026-10-28）

**还不能宣称“推理已完全验收，只差训练好的 head”。此前完整模型的短 smoke 只证明链路能跑，不能覆盖 draft 语义、长输出和并发状态。**

**验收范围更新：已在纯主模型中复现并归因的数值差异，仅记录为上游限制，不要求我们修复，也不单独阻塞 Orthrus 适配验收。** 下文按时间保留严格对照结果；已归因的两个 greedy 案例不再继续追查主模型内部原因。后续已补做少卡真实 engine 的受控接受/回滚检查，并完成原并发案例归因，见“真实引擎状态验收”。完整模型、概率 draft 的随机高接受率等仍需复验，不能因上游也有差异而全部豁免。具体标准见 [验收范围](DSV4_VALIDATION_RUNBOOK.md#验收范围)。

## 本次修正

- C4 indexer 原来误用 AR 因果模式，使一个 draft 块前面的 query 少看部分历史；改为 `sparse_mode=0`，32 个 query 共用 anchor 前的历史边界。
- QLI 返回的是压缩行号，原来又除以 4 会读错行；现保留原行号，只过滤尚未完成的压缩组。
- 上游 draft 采样忽略请求 seed；Orthrus 增加按请求生成随机数的路径，覆盖该请求全部 31 个候选位置。无 seed 的请求沿用原路径。

## 已取得的证据

环境为 A2、固定 v0.26 镜像、W8A8。服务扩展测试使用 DSV4 前四层和初始化 head、TP=4，包含 SWA/C4/C128；这不是完整 43 层模型的验收。

| 验证项 | 实测结果与边界 |
|---|---|
| C4 历史可见范围、行号和 Top-512 | NPU 测试通过，覆盖长度 3/4、127/128/129、2049、4096/4097；旧因果模式的问题可复现 |
| 现有 FIA + 原生 LSE 合并 | K=32、D=512、8/16 heads，通过独立 FP32 拼接 softmax 对照，包含空压缩历史及只计一次的 sink；最大绝对误差约 0.0035/0.0038 |
| 原生 rejection sampler | greedy、随机及混合批次通过全部 31 个拒绝位置和整块接受测试；8,192 次采样符合已知目标分布的六标准差门槛。这不是 scheduler 回滚验证 |
| draft 请求 seed | NPU 上请求换序、全局 RNG 改变后仍可复现；CPU 回归测试覆盖连续多轮 |
| draft 不写 target cache | 四层服务覆盖 batch=1/2/3/4；每种 batch 首次全量快照对比通过，其余轮次抽样检查通过 |
| 长度与服务生命周期 | 32K、128K 请求跑完；EOS、显式 stop、SSE 断开及请求清空检查通过。输出一致性另外判定，不能由“不崩溃”代替 |
| 普通执行模式的输出一致性 | **未通过**：14 组 greedy 有 2 组不一致；target-only 自身重复输出也不稳定，不能将全部差异归因于 Orthrus |
| 确定性模式对照 | 两边各自三次 greedy 重复均一致，但跨服务仍有 2/14 组差异；target 固定 seed 回放 4/4 一致，Orthrus 为 1/4，一致性门槛仍未通过 |

## 算子使用

没有开发新 kernel。原生 `npu_attention_update` 合并在 D=512 的测试中约 209–225 us，原 PyTorch 组合约 519–565 us；这是局部耗时，不是服务加速比。

C128 另测了直接分页 FIA：128-token 页下约 535–630 us，gather + FIA 约 1,052–1,298 us；输出和 LSE 一致。当前 A2 FIA 拒绝 32-token 分页，因此该布局保留原 gather 路径。8/16 heads、32/128 页、空历史到 8,192 个压缩行共 16 组测试通过。不能将这一收益套用到不兼容的缓存布局。

### 后续单卡优化

完整模型 TP8 两次启动均在 NPU 驱动初始化超时，未进入权重加载。随后按要求停止重试八卡，改用空闲卡继续测试现成算子；没有停止其他用户的任务，也没有开发新 kernel。

| 路径 | 修改及实测结果 |
|---|---|
| SWA + 块内 Attention | 临时拼接 KV，一次 FIA 计算，mask 只屏蔽历史 padding。包含 sink 合并的 36 组测试（batch 1/2/4、heads 8/16/32、历史 0/1/127/128）通过 FP32 对照；约 0.67–0.86 ms 降至 0.61–0.66 ms，减少约 9%–24% |
| C4 稀疏历史 | 当前镜像的 `npu_sparse_flash_attention` 实际支持分页、非因果及 softmax max/sum 输出。六组测试覆盖空历史、短历史、Top-512、多请求和 262,144 个压缩行；通过 FP32 对照、LSE 对照和完整 cache 不变检查。生产适配函数约 0.63–0.68 ms，gather + FIA 约 2.2–4.7 ms |
| C4 兼容回退 | 保留 `ORTHRUS_C4_BACKEND=gather`。多请求时改用扁平 `index_select`；单请求和 SWA 保留原索引路径，避免实测回退 |
| 合并后的完整数学路径 | SWA/块内联合结果、压缩历史和 sink 再合并；32 heads 的独立 FP32 对照最大绝对误差约 0.00383，缓存检查通过 |

C4 native 路径需要一个全零的辅助 query RoPE，以及目标 key 的只读切片，以满足现成 MLA 接口；附加项为零，不改变 DSV4 的 Attention 公式。切片本身不复制 KV，但算子仍有 workspace：测试中其临时峰值约 72–118 MiB，gather 路径约 34–131 MiB，不能声称所有场景都省内存。

上述时间是局部 eager 路径耗时，不是整模型吞吐提升。短历史 C4 的 FP32 最大绝对误差约 0.0108，其余非空案例约 0.0012–0.0026；这是 BF16 算子容差验证，不代表服务逐 token 一致性已经通过。

优化代码随后在 DSV4 前四层、W8A8、TP=2 实际接入：10 组 greedy 请求（最长输出 160 token、输入 4K）、16 组随机请求、batch 1/2/3/4 均跑完，SWA/C4/C128 的抽样 cache guard 未报错。两次串行 greedy 重复一致；EOS、显式 stop、SSE 取消、取消后重放通过。

**服务正确性仍未通过：混合长度并发 greedy 与串行记录有差异；随机请求并发后串行回放仍只有 1/4 个 seed 一致。** 本次没有把局部算子通过或四层服务能运行解释成完整模型验收。CPU 回归为 74 passed、3 skipped（需要 CUDA graph 硬件的用例）。

## 仍需验收

1. 已测的混合长度并发案例已完成归因，不再阻塞少卡适配验收；完整模型或新增请求出现的未知差异仍需单独判断。少卡 seed 回放的修复仍需完整模型和长上下文复验。
2. 完整 43 层、TP=8 复验，包括长输出、长上下文和服务重启。四层通过不能替代这一项。
3. 受控 greedy 候选已覆盖全部接受长度；后续也通过了四层 TP2、temperature>0 的概率 draft、整块接受、KV 页复用及故障注入后的抢占恢复。完整模型和真实请求分布仍需复验，受控候选不能证明 head 质量。
4. 1M 上下文、Ascend graph、拿到训练好的 head 后的推理加速收益及长时间稳定性。当前不将这些标为通过；这里不涉及训练实现。

## 167 少卡复验与修复

后续使用 167 的两张 A2 卡、同一镜像 digest，在独立容器重建环境。该机驱动是 26.1.1，与 111 的 25.5.1 不同，结果分别保留。模型仍是实际 W8A8 权重截取的前四层，TP=2、K=32、eager、`VLLM_BATCH_INVARIANT=1`。

- CPU 回归 89 passed、3 skipped；跳过项要求 CUDA graph 硬件。head 契约和评估契约自测通过。111 上完整 43 层初始化 head 的生产 preflight 也通过，但不代表训练或生产推理验收。
- 单卡九组检查全部通过：QLI、真实 Ascend rejection sampler、分页 FIA、8/16/32 heads 混合 attention，以及 joined/gather/sparse 三条路径。没有开发新 kernel。
- 确认并修复固定版本 Ascend sampler 的一个问题：`sample_recovered_tokens` 对零候选请求仍生成并丢弃随机数，偏移从 0 变为 12。补丁跳过该请求后，零候选偏移保持 0，正常请求仍为 12；全部拒绝位置、整块接受、混合采样及 8,192 次分布检查通过。补丁保留版本保护、dry-run 和重复应用检查。
- 修正健康检查的超时设置：HTTP 检查仍为 30 秒，首个 decode 单独允许 300 秒，避免首次算子编译被误报为部署失败。
- target-only 的 greedy 回放及混合长度并发/停止/取消检查通过；随机回放的一个 seed 在并发情况下从第 11 个 token 开始不同，串行重复和并发后串行回放仍一致。
- 修复后的 Orthrus 串行、并发、换序、同 seed 并发及并发后回放共五组、每组四个请求全部一致。EOS、显式 stop、SSE 取消及取消后重放通过，抽样 cache guard 未报错。
- **严格 greedy 门槛仍未通过**：两服务 10 例中 2 例不同，首次分歧在第 4、第 18 个输出 token；Orthrus 混合长度并发与串行仍有差异。这些结果没有因 seed 修复或算子通过而被改为通过。

新增 `validate_dsv4_readiness.py` 分离 CPU、单卡和服务检查，记录源码哈希、环境锁、每项命令、日志和退出码；失败后继续收集证据，整体返回失败。复验命令及八卡窗口清单见 [验证入口](DSV4_VALIDATION_RUNBOOK.md)。测试客户端第一次生命周期检查误用了其他模型的 EOS ID，该次配置错误保留；按本模型 `generation_config.json` 的 EOS=1 重新运行后，target-only 生命周期检查通过。

### Greedy 分歧定位

用 `check_dsv4_greedy_divergence.py` 将两组输出首次分歧前的共同 token 作为新的 prompt，分别测试单独 prefill、追加 31 个未来 token 后该位置的因果 prompt logprobs。**纯 target-only 服务与 Orthrus 服务的新鲜 prefill 结果一致，但与 target-only 的逐 token decode 结果不同。** target-only 重启后原始 64/160-token 输出也逐 token 重现。

| 首处分歧 | target-only decode | target-only 新鲜 prefill |
| --- | --- | --- |
| 第 4 个 token | token 119543，领先另一候选约 0.125 logprob | token 67290，两候选并列 |
| 第 18 个 token | token 64544，两候选并列 | token 117465，领先约 0.125 logprob |

这说明上述两个反例不依赖 draft 或 Orthrus 的拒绝回滚，在原生主模型路径中就能重现；当时尚未逐层定位，后续结果见下一节。保留原失败记录。八卡测试应同时复验完整模型的 prefill/decode 一致性，不能把四层截取模型的并列 logits 当作完整模型的精度结论。

### 逐层定位与现成算子复核

继续使用 167 的 0/1 两卡。`trace_dsv4_native_layers.py` 对同一 causal token 记录输入和模块输出，纯主模型即可重现前述两例；记录时同步复制激活到 CPU，因此这些 trace 不能用于性能结论。

- 第一处差异在第 0 层 `_mla_prolog_multistream` 的 Q 输出，最大绝对差 0.0625–0.078125；此前 embedding、mHC、输入 norm 一致。
- 固定镜像的该方法在 prefill 先产生 BF16 norm 结果再量化，在 decode 使用 `npu_rms_norm_dynamic_quant`。真实权重的独立测试也确认两者产生不同的 INT8 值和 scale。这不是 Attention 可见范围或 draft cache 写入导致的差异。
- 用现有配置 `multistream_dsv4_dsa_overlap=false`，在两个案例原先分歧的位置，两个 TP rank、每例每 rank 记录的 175 项中间张量全部逐元素一致，预测 token 也一致。未修改 target 数学或开发新 kernel。此结论只覆盖这些前缀，不能自动放行更长输出和完整模型。
- 公共启动脚本只在 `VLLM_BATCH_INVARIANT=1` 时强制关闭该多流路径；普通性能配置不被静默更改。

进一步优化及候选筛选：

| 路径 | 实测与选择 |
| --- | --- |
| draft Q/KV 前处理 | 复用原生 DSV4 的做法，共用一次输入动态量化，保留各自 projection、norm 和 RoPE。4 层 × batch 1/2/4 × 2 ranks，共 24 组 Q、QR、scale、KV 全部逐元素一致。23 组耗时减少约 1%–3%，1 组约慢 1%；属于小幅局部优化，不能宣称整网提速 |
| C128 小分页历史 | 复用 C4 已采用的原生 `npu_sparse_flash_attention`，索引覆盖全部有效压缩行，不做 Top-K 截断。8 组包含空历史、1–8192 压缩行、随机物理页、多请求及 32 heads，通过独立 FP32、LSE 和完整 cache 不变检查。多数案例约 1.1–1.3 ms 降至 0.8–0.9 ms，batch=4/heads=32/8192 行约 1.96→1.94 ms，收益基本持平 |
| C128 大分页 | 仍使用直接分页 FIA；`ORTHRUS_C128_BACKEND=gather` 保留回退。没有为了复用稀疏算子而改动历史选择或 cache 布局 |
| FIA v2 内置 sink | 当前镜像要求 sink 为 BF16，不能直接接收现有 FP32 sink。转换后，首例 LSE 最大误差约 0.00942，未通过原有 `atol=0.002, rtol=0.001` 门槛；因此没有接入生产路径，仍保留独立 FP32 sink + `npu_attention_update` |

Q/KV 计时移除了所有逐层诊断 hooks，包含 host dispatch 和临时缓冲区；C128 使用合成历史、真实 DSV4 形状。均不是完整模型吞吐结果，也没有证明所有 batch/上下文下已达最优。CPU 回归为 99 passed、3 skipped，head/评估契约自测通过；更新后的单卡十组检查全部通过，包含原有九组及 C128 小分页新路径。原始结果保存在 167 的 `local/native-layer-trace-final/`、`local/c128-sparse-final.log`、`local/sink-v2-bf16-probe.log`、`local/readiness-npu-operators-final/`。

### 更新后服务结果

上述修改已在实际四层 W8A8、TP=2 服务复验。Orthrus 的五组固定 seed 回放共 20/20 一致，停止、取消、取消后重放通过，抽样 cache guard 未报错。但**严格 greedy 仍未通过**：10 个跨服务案例有 2 个不同，首次分歧由第 4/18 个输出推迟至第 43/72 个；混合长度并发也仍有差异。target-only 随机并发仍有 seed=2 在第 11 个 token 不同。

继续对新分歧执行纯主模型逐层对照：两例均无需 draft 即可重现，首次记录到的当前 token 激活差异分别出现在第 2 层 C4、第 1 层 SWA 的原生 `npu_sparse_attn_sharedkv` 输出，最大绝对差约 0.0039–0.0078。进一步抓取实际参与计算的历史后，确认当前 Q、sink 以及 C4 检索索引均一致，但算子接收的历史 KV 已经不同：C4 层局部历史最大差 0.0234375、压缩历史 0.015625；SWA 层历史最大差 0.015625，两个 TP rank 均重现。因此不能把输出差异归咎于 Attention kernel，更不能据此判断 draft 写坏了 cache。按更新后的验收范围，到此保留上游差异的证据，不继续追查主模型内部根因。证据保存在 `local/native-layer-trace-later/`、`local/native-layer-trace-history/` 和 `local/greedy-later-divergence.json`；原始严格对照结果不变。

实际服务的 NPU profiler 也确认了现成算子的使用：每 rank 记录到 80 次 FIA、16 次 SparseFlashAttention、64 次 AttentionUpdate，以及原生 QLI、量化 matmul 和融合 MoE。这次服务布局的 C128 使用直接分页 FIA，小分页 sparse 优化是另行测过的兼容路径，不能把它的约三成局部收益套到本服务。该 profile 开启了 cache guard，Slice/AsStrided 等检查开销很大，只用于确认调用路径，不用作生产耗时结论。汇总见 `local/operator-profile-prolog/operator-summary.json`。

本轮结束后，167 的测试容器只剩 `sleep infinity`，0/1 两卡无 NPU 进程；未停止其他用户的任务。原始 trace、局部 benchmark、服务结果和 profiler 汇总另行归档，不提交权重或大体积 profiler 数据到代码仓库。

### 真实引擎状态验收

新增 `check_dsv4_engine_state.py`，仍使用 167 的 0/1 卡、实际四层 W8A8 权重、初始化 head、TP2、eager。正式 proposer 和算子实现没有改动；所有注入和诊断都在独立测试 worker 内。

| 验证项 | 本轮结果 |
| --- | --- |
| 实际接受/拒绝流程 | 真实 draft 先运行，再仅在测试出口注入候选；覆盖接受 0–31 个候选、连续拒绝和整块接受，逐轮采样数量及最终输出符合预期 |
| 请求与历史对应 | batch=1/2/3/4、输入长度 127/128/511/513、请求换序；逐轮检查 anchor/MASK、有效历史长度和各 cache group 分页表，通过 |
| KV 只读 | 每层、每种 batch 首次全量 cache 快照，后续轮次抽样；未发现 draft 修改目标 cache |
| 多轮与取消 | 最长受控输出 600 token，跨历史页和压缩组；取消后新请求正确。未强制制造物理页复用或内存抢占 |
| TP 一致性 | 两个 rank 的请求、采样及状态轨迹哈希相同 |
| 对目标计算的影响 | 预先指定首次 token-only 运行作为参考，完全绕过 Orthrus proposer；真实 Orthrus 与该参考的 45 轮完整词表 logits 字节哈希全部相同，仅跳过 head 的诊断路径也与真实路径相同 |
| 原混合长度并发差异 | 4 例中 3 例逐 token 相同；剩余一例的共同前缀、分歧位置和两个候选 token，均与之前纯主模型 prefill/decode 的反例完全对应，按已知限制归档 |

参考路径自身重复两次，在第 32–44 个验证步仍出现 logits 差异；此时两次都没有执行 Orthrus proposer，不能归咎于 draft 执行。保留该不一致，不继续追查其内部原因；同时要求真实路径匹配预先固定的第一组参考，不能因为参考不稳定就放行我们的差异。`state_checks_passed`、`adaptation_reference_passed` 与参考自身是否稳定分别记录。

验证过程中的失败记录也保留：首轮启动覆盖了 CANN 的 `PYTHONPATH`，导致缺少 `acl`；全量 guard 初次使用自动分配的约 15 GiB KV 池，快照和临时比较张量导致 OOM。修正启动环境，并将此测试的每 rank KV 池固定为 1 GiB 后完成检查，没有缩短上述测试用例或放宽数值阈值。

这些结果只支持“四层 TP2、greedy 受控候选的引擎适配检查”，不等同于完整 43 层、随机高接受率、graph 或性能验收。测试命令及各结果字段见 [真实引擎状态检查](DSV4_VALIDATION_RUNBOOK.md#真实引擎状态检查)。

最终复跑保存在 167 的 `local/engine-state-v6/`，脚本退出码为 0：`state_checks_passed=true`、`adaptation_reference_passed=true`；`reference_repeat_exact=false` 和 `natural_exact=[false,true,true,true]` 如实保留。每 rank 记录 337 个受控请求验证步、429 个 draft 请求步，覆盖 batch 1–4；两个 rank 状态轨迹一致。CPU 最终回归为 135 passed、3 skipped，head/评估契约自测通过。旧轮次的失败记录没有覆盖，原始结果归档为 `artifacts/dsv4-20261028/engine-state-167.tar.gz`。

结束检查确认 167 的 0/1 卡没有 NPU 进程，测试容器只剩空闲主进程；111 的八卡仍有运行任务，本轮未做完整模型 TP8 验证。

### 概率 Draft、KV 复用与抢占恢复

新增 `check_dsv4_sampling_state.py` 和测试专用的 `dsv4_test_scheduler.py`。仍使用 167 的 0/1 卡、四层真实 W8A8、初始化 head、TP2、eager；正式推理实现、算子和运行时补丁均未修改。

主模型与 head 的 forward 正常执行，测试钩子只在采样前注入已知的 p/q logits。真实 draft 概率采样、概率传递、标准 rejection、scheduler 和 KV 操作保留。因此这是推理集成检查，不是自然模型精度、head 接受率或吞吐测试。

| 检查 | 最终结果 |
| --- | --- |
| 概率采样 | 两轮随机请求、两轮 p=q 压力请求，共 16 请求、16,384 个输出 token；随机和压力轮分别检查频率，均通过预设统计界限 |
| 接受与补采样 | 每 rank 1,451 个有效请求采样事件，419 次完整接受 31 个候选、1,012 次拒绝；逐步检查接受前缀及补采样属于正差分布支持集 |
| q 对齐 | 597 次概率传递检查；覆盖 batch 1–4、请求换序、不同 temperature/seed、随位置变化的 q，以及抢占恢复 |
| KV 页复用 | 从 scheduler 实际分配记录确认 7 个 cache group 内共 482 个 `(group, block_id)` 被不同请求复用，排除 null page；未声称覆盖第 8 个 group 的跨请求复用 |
| 抢占恢复 | 对 4 个请求各注入一次 `allocate_slots` 失败；由原生 Ascend scheduler 执行抢占、释放和重算，4/4 恢复后再次进入投机验证，最终完成输出 |
| 历史与 cache | anchor/MASK、有效历史、分页表检查通过；各层各 batch 首次全量、之后抽样的 cache guard 未报错 |
| TP 与 CPU | 两个 rank 的采样/状态轨迹哈希相同；CPU 回归 143 passed、3 skipped（CUDA graph 硬件要求），head/评估契约自测通过 |

**抢占由受控分配失败触发，不是实际 NPU OOM，也不是自然压力触发。** 测试 scheduler 继承当前镜像的 `BalanceScheduler`，只注入失败信号和记录事件；不自行实现抢占或恢复。256 blocks 是测试池容量，不是生产配置。

最终结果为 `local/sampling-state-v6/report.json`，`passed=true`，退出码 0；CPU 为 `local/readiness-cpu-sampling-v6/`。测试源码 SHA256 随报告保存，两个 rank 的状态哈希为 `728d1c730a26c31f8ec7a0eddc22651e6528acb8c95c89098ac3be98589ce1ea`。

保留此前所有失败记录：v1 的 128-block 池未通过启动容量检查；v2 在上一轮 worker 尚未完全退出时启动，被内存门槛拒绝，随后清理了本任务残留；v3/v4 未自然触发抢占，按严格门槛返回失败，不能算抢占验证通过；v5 的测试 hook 未透传 `drop_stale_output`，修正并加入回归后完成 v6。没有用本轮结果覆盖旧失败。

原始轮次、日志和 CPU 结果归档为 `artifacts/dsv4-20261028/sampling-state-167.tar.gz`。本轮补齐的是少卡概率采样和调度恢复验证；完整 43 层 TP8、长上下文、自然容量耗尽下的持续负载、graph 及训练好 head 的实际推理收益仍未验收。

结束检查确认测试容器只剩 `sleep infinity`，167 的 0/1 卡均无 NPU 进程；没有停止其他用户的任务。

## 2026-10-29 完整 TP8 主模型窗口

111 的 `window-20261029-retry1` 曾获得完整八卡窗口，运行完整 43 层 W8A8 主模型、TP8、eager。此次已进入真实 API 推理，不再停留在加载阶段；但尚未执行 Orthrus，不能算完整投机链路通过。

| 检查 | 结果 |
| --- | --- |
| 健康检查与基础快照 | 通过，`target-initial-health` 和 service snapshot 返回 0 |
| Greedy replay | 脚本通过；混合长度并发属于另一个检查，见下行 |
| 生命周期 | EOS、显式 stop、SSE 取消后排空、取消后重放通过；8/127/2047/4096 输入长度混合并发有 1 例与串行参考不同，因此整项返回 1 |
| 随机 replay | 串行重复、并发换序、同 seed 并发和并发后串行均 4/4 一致；普通并发中 seed 0、2 不一致，因此返回 1 |
| 长上下文采集 | 32768 和 131072 输入 token 均完成 4-token 输出，扩展快照命令返回 0；不等于 Orthrus 长上下文对照已通过 |
| Orthrus 切换 | 启动前端口检查报 `OSError(98, 'Address already in use')`，尚未启动 Orthrus 服务；专用容器已停止 |

这些差异出现在 target-only 路径，没有执行 draft。按既定验收范围保留证据，不修改主模型，也不将未知的 Orthrus 差异一并豁免。原始结果位于宿主机 `/data/disk/wangzili/orthrus-dsv4-validation/window-20261029-retry1/` 及容器持久目录 `validation-20261029/local/tp8-window-20261029-retry1/`。

端口检查未设置 `SO_REUSEADDR`。在 111 上用真实 TCP 连接复现：服务主动关闭连接后，无监听者的端口仍使旧检查报 EADDRINUSE；采用与服务一致的可重用绑定则通过。原始故障时没有监听者快照，不能排除当时另有端口竞争，但旧检查本身确实会误判 TIME_WAIT。已改为 `SO_REUSEADDR` 后探测 `0.0.0.0`，仍拒绝真实监听者；包括这两个回归场景在内的 7 项窗口测试通过。

完整 43 层 W8A8 初始化 head 的 TP8 production preflight 再次通过：SWA/C4/C128 分别为 2/21/20 层，head 约 6.067 GiB。该检查只验证模型/head 契约，不代替执行验证。

已使用新目录 `window-20261029-retry2` 恢复 12 小时宿主机监控，每 3 秒检查一次完整八卡无进程且 HBM 回落后启动 target/Orthrus/重启对照。等待阶段仅保留宿主机监控和容器 `sleep infinity`，不加载模型、不占 NPU；恢复时状态为等待窗口。旧失败和结果未覆盖。

## 2026-10-29 八卡结果与双卡定位

`window-20261029-retry2` 于北京时间 10:54–12:24 完成完整 43 层、TP8、eager 的 target-only、Orthrus 和 Orthrus 重启测试。三个服务均通过健康检查；主模型和 Orthrus 的 32K/128K 输入均完成 4-token 输出，EOS、显式 stop、SSE 取消后排空及取消后重放通过。Orthrus 重启前后的十组 greedy 快照相同。

**完整模型仍未验收。** target-only 与 Orthrus 的十组 greedy 快照有两组不同；Orthrus 串行和并发后串行回放均 4/4 一致，但三组并发 greedy 回放均 0/4 一致，对应 target-only 的这些 greedy 回放均 4/4 一致。随机并发回放在两种服务中都存在差异。所有原始非零退出码保留，不能把启动或重启成功写成输出正确性通过。

### 概率生命周期修复

上午服务日志出现 `Missing cached draft probabilities`。将告警请求与保存的 API 记录对应，43 个请求中确认 38 个是 greedy 请求，对应 4,248 条告警；另五个请求没有对应的完整记录，不据此断言它们是随机请求。总告警 5,328 条包含八个 TP rank 的重复记录。

Ascend 的 `propose_draft_token_ids` 重写路径没有像上游一样清空上一轮概率。随机请求结束后，下一批 greedy proposal 不产生概率，runner 因而保留了旧请求的概率及 ID，后续验证查询产生告警。补丁仅在 Orthrus proposal 开始时清空这两个字段，再接收本轮实际产生的概率；不改 Attention 或 rejection 算法。

在 111 的物理 0/1 卡上，以真实四层权重、TP2、K32、同步调度、初始化 head 交替运行 greedy、随机和混合批次，取得以下配对结果：

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| 每 rank 有候选的验证步 | 248 | 248 |
| greedy 请求遇到旧概率 | 15 次 | 0 次 |
| 随机候选验证未取得概率 | 0 次 | 0 次 |
| 批量 greedy 与串行一致 | 4/4 | 4/4 |
| 并发后串行与原串行一致 | 4/4 | 4/4 |

八个阶段的全部输出 token 在修复前后完全一致。因此这轮确认的是旧概率残留及其修复，不能将告警直接解释为随机采样已经发生分布错误。对应结果为 `baseline-sync-priv/` 和 `fixed-sync/`；修复提交为 `e9ef9f5`。

进一步从原始权重保留前 12 层（2 SWA、5 C4、5 C128），导出匹配的初始化 head，以 TP2、2 GiB/rank KV 池运行相同测试。每 rank 259 个验证步无概率缺失或旧概率残留，批量 greedy 及批量后串行均 4/4 一致，`fixed-12l-v2/report.json` 返回成功。截取模型权重约 77.46 GiB、head 约 1.69 GiB，证明更深模型可用双卡检查，但没有复现完整 43 层的上述文本差异。

### 四请求目标 logits 对照

`check_dsv4_engine_state.py --shadow-batch 4 --shadow-logit-errors` 在四层 TP2 中保持相同受控候选和调度，对照跳过 proposer、真实 draft 及跳过 head 三条路径。接受 0–31 个候选、换序、分块 prefill、跨页、多请求先后结束和取消恢复的状态检查通过；每 rank 449 个有效验证事件、306 个 draft 事件，两 rank 状态哈希相同。

严格数值检查仍返回失败：74 个 shadow 请求验证事件中，完全跳过 draft 的两次原生运行有 15 个事件的全词表 logits 不逐位相同，最大绝对差 0.890625，首行最大差 0.125，首行 argmax 有一次变化。真实 draft 与预先固定的第一组原生参考在相同 15 个事件存在差异，未因此豁免。

真实 draft 与跳过 head 直接对照，仅一个事件的其余验证行不同，最大绝对差 0.125；全部事件的首行 logits 一致，所有行的 argmax 均未变化。这支持已测路径的状态协作，但不满足“所有验证 logits 逐位一致”的门槛，更不能替代完整模型的差异归因。`shadow-batch4-v3/report.json` 中 `state_checks_passed=true`、`adaptation_reference_passed=false`，脚本退出码仍为 1。

上午完整模型的相应 greedy 反例中，串行的 `and` 与 `for` logprob 并列，并发后 `for` 领先约 0.125；这是排序改变的直接证据，尚未逐层定位其来源，不将完整模型差异全部归因于主模型或全部归因于 draft。

### 复现与资源

最终 CPU readiness 为 158 passed、3 skipped（CUDA graph 硬件要求），head/评估契约检查通过。运行时补丁 dry-run 和实际应用均通过，新增修复保留版本保护及重复应用检查。

失败轮次没有覆盖：首次显式启用 async scheduling 被当前 Orthrus 配置保护拒绝；非特权容器在驱动访问阶段失败，后改用上午已有的权限配置并限制可见设备为 0/1；12 层使用 1 GiB KV 池时未通过容量检查，改为 2 GiB；四请求 shadow 首轮的测试出口错误推进了 discarded prefill 行的候选计划，已修正并增加回归，后续才取得相同的 74 事件调度。以上均与正式推理实现的错误分别记录。

本轮仅在 111 的 0/1 卡运行，未使用 4/5，也未恢复八卡监控；结束后两个调试容器均已停止。原始输出保存在独立调试项目的 `local/debug-concurrency/`，包含修复前后、12 层、严格 shadow 对照和 CPU 检查。本地另归档为 `debug-concurrency-20261029.tar.gz`，SHA256 为 `8065f17407a0f5ef5702687d899ab620c6a67f09caa04a4ecd40ba08c2c3860f`。后续完整模型应复用这些诊断入口确认数值差异来源；graph、1M、持续负载和训练 head 的实际收益仍未验收。

## 复现入口

- `scripts/check_dsv4_indexer_npu.py`：真实 QLI 的历史边界、行号及排序。
- `scripts/smoke_dsv4_hybrid_npu.py`：真实形状 FIA、sink、LSE 合并和分页读取。
- `scripts/check_dsv4_paged_npu.py`：分页 FIA 与 gather 路径对照。
- `scripts/check_dsv4_rejection_npu.py`：真实 Ascend sampler 及请求级 seed。
- `scripts/validate_dsv4_service.py` 配合 `configs/dsv4/correctness-extended.json`：服务快照和对照。
- `scripts/check_dsv4_lifecycle.py`：混合长度并发、停止、取消和取消后的重放。
- `scripts/benchmark_dsv4_attention_paths.py --case joined|gather|sparse|c128-sparse`：现有算子路径对照；`sparse` 同时检查 FP32 参考、完整缓存和临时峰值内存。`sink-v2` 是尚未通过精度门槛的候选探测，不在 readiness 必过集合内。
- `scripts/trace_dsv4_native_layers.py`：纯主模型逐层 prefill/decode 对照；可加 `--benchmark-prolog` 测试真实权重的 Q/KV 前处理，此计时会先移除诊断 hooks。
- `scripts/check_dsv4_sampling_replay.py`：区分串行重复、并发换序和并发同 seed，结果逐阶段保存。

服务对照应同时设置相同的 `VLLM_BATCH_INVARIANT`；普通模式的不稳定结果保留，不用确定性测试覆盖或抹去。单看 API 首 token 的分布也不能验证 speculative rejection，因为它通常还没经过投机验证。
