# DSV4 Orthrus 验证记录（2026-10-28）

**还不能宣称“推理已完全验收，只差训练好的 head”。此前完整模型的短 smoke 只证明链路能跑，不能覆盖 draft 语义、长输出和并发状态。**

**验收范围更新：已在纯主模型中复现并归因的数值差异，仅记录为上游限制，不要求我们修复，也不单独阻塞 Orthrus 适配验收。** 下文的严格对照失败仍如实保留；其中已归因的两个 greedy 案例不再继续追查主模型内部原因。未归因的并发差异、真实 engine 的高接受长度提交/回滚，以及完整模型复验仍属于我们的工作，不能因上游也有差异而全部豁免。具体标准见 [验收范围](DSV4_VALIDATION_RUNBOOK.md#验收范围)。

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

1. 只继续定位尚未归因的混合长度并发等差异；下文已在纯主模型中复现并归因的两个 greedy 案例不再阻塞适配验收。少卡 seed 回放的修复仍需完整模型和长上下文复验。
2. 完整 43 层、TP=8 复验，包括长输出、长上下文和服务重启。四层通过不能替代这一项。
3. 高接受率下的真实 engine KV 提交/回滚；初始化 head 接受率低，单测覆盖所有拒绝位置不等于服务已覆盖整块接受。
4. 1M 上下文、Ascend graph、训练 head 的性能收益及长时间稳定性。当前不将这些标为通过。

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
