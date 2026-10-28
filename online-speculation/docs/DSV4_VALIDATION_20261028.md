# DSV4 Orthrus 验证记录（2026-10-28）

**还不能宣称“推理已完全验收，只差训练好的 head”。此前完整模型的短 smoke 只证明链路能跑，不能覆盖 draft 语义、长输出和并发状态。**

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

## 仍需验收

1. 继续定位确定性模式下的跨服务 greedy 差异及随机采样并发回放差异。按请求 seed 的独立测试通过，不能代替端到端复现通过。
2. 完整 43 层、TP=8 复验，包括长输出、长上下文和服务重启。四层通过不能替代这一项。
3. 高接受率下的真实 engine KV 提交/回滚；初始化 head 接受率低，单测覆盖所有拒绝位置不等于服务已覆盖整块接受。
4. 1M 上下文、Ascend graph、训练 head 的性能收益及长时间稳定性。当前不将这些标为通过。

## 复现入口

- `scripts/check_dsv4_indexer_npu.py`：真实 QLI 的历史边界、行号及排序。
- `scripts/smoke_dsv4_hybrid_npu.py`：真实形状 FIA、sink、LSE 合并和分页读取。
- `scripts/check_dsv4_paged_npu.py`：分页 FIA 与 gather 路径对照。
- `scripts/check_dsv4_rejection_npu.py`：真实 Ascend sampler 及请求级 seed。
- `scripts/validate_dsv4_service.py` 配合 `configs/dsv4/correctness-extended.json`：服务快照和对照。
- `scripts/check_dsv4_lifecycle.py`：混合长度并发、停止、取消和取消后的重放。

服务对照应同时设置相同的 `VLLM_BATCH_INVARIANT`；普通模式的不稳定结果保留，不用确定性测试覆盖或抹去。单看 API 首 token 的分布也不能验证 speculative rejection，因为它通常还没经过投机验证。
