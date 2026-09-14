# Orthrus Draft/Verify 协同调度实验

## 目标

本实验只考察系统调度，不修改 Orthrus 的 K32、温度、采样校正或模型权重。目标是验证
[SparseSpec](https://arxiv.org/abs/2512.01278) 的 unified scheduling 与 delayed
verification 思路能否直接迁移到 Orthrus-Qwen3-8B 的单卡 vLLM-Ascend 服务。

SparseSpec 的开源实现为请求维护 `NORMAL/DRAFT/VERIFY/STALL` 状态，并通过错开请求的
draft phase 让一个 batch 同时含 draft 与 verify；异步模式把完成 verify 的请求放入一个
周期的 `STALL`，用其他请求的 NPU 工作覆盖 CPU rejection 与元数据更新。参考其
[`SpecScheduler`](https://github.com/sspec-project/SparseSpec/blob/main/serve/scheduler/spec_scheduler.py)。

## Orthrus 与 SparseSpec 的关键差异

- SparseSpec 连续执行 K 次单 token 稀疏 draft；Orthrus 一次双向 attention 就并行产生
  31 个 draft 位置。
- SparseSpec 的 draft/verify attention 共享可统一调度的数据流；当前 Orthrus vLLM runner
  在一次迭代内固定执行 target verify，再调用完整的 Orthrus draft backbone。
- vLLM-Ascend 已在 NPU 上完成 rejection correction。Orthrus draft 又必须立即知道准确的
  accepted count、anchor token 和 history position，因此仅打开通用 async scheduler 不等价于
  SparseSpec 的 `VERIFY -> STALL -> DRAFT`。

所以本轮分两步验证：先移植一周期异步 correction，再实现一个不改 target 分布的
request-level staggered admission。没有把后者称为完整的 mixed-phase runner。

## 已实现

### 1. Orthrus async correction 兼容

- 放开 vLLM 0.23 对 `custom_class + async_scheduling` 的限制。
- 向 Ascend runner 回传每个请求的 accepted count 和最后一个真实 token。
- 按 `request_id` 维护精确 history，隔离 vLLM CPU 侧 31-token 乐观 placeholder。
- 原始错误会使 history 从约 7.6K 一次跳 32，acceptance 降到约 1%；修复后 trace 按
  `7664 -> 7666 -> 7668 -> 7671` 推进，单请求 acceptance 恢复到 6.5%。

### 2. Staggered draft admission

- 维护逻辑 draft-ready 队列及每个 request 的等待年龄。
- 用 8K history bucket 形成较稳定的长短上下文 draft batch。
- 每轮最多允许 `draft_max_batch` 个请求执行 K32 draft；未选请求本轮执行精确 AR，
  下一 token 重新竞争，不改变 target 分布。
- 超过等待阈值的请求优先，避免固定的一组请求长期占用 draft。
- trace 新增 `orthrus.phase.admission`，保存 ready/admitted/deferred、请求 ID、长度桶和年龄。

这一步能够限制下一轮 speculative verify 的宽度，但仍不是在同一个 layer forward 内融合
Orthrus draft 与 target verify。真正的 unified runner 需要拆开当前固定的 `verify -> draft`
调用，并在每层分别执行两种 attention 后合并共享 MLP batch。

## 实验配置

统一配置：单张 910B3、vLLM/vLLM-Ascend 0.23、Orthrus-Qwen3-8B、K32、
`MNS=64`、`MBT=65536`、memory 0.94、`FULL_AND_PIECEWISE`、paged-segmented
draft attention。固定 64 个 coding 请求一次性并发，prompt 为 6K-40K，输出 128 token。

| 实验 | async | max draft batch | output tok/s | 相对 B1 | P95 latency |
|---|---:|---:|---:|---:|---:|
| 同步 B1 | 否 | 不限制 | 34.296 | 基线 | 233.94s |
| delayed correction | 是 | 不限制 | 33.762 | -1.56% | 236.32s |
| staggered | 是 | 8 | 31.945 | -6.86% | 249.18s |
| staggered | 是 | 16 | 32.975 | -3.85% | 242.24s |
| staggered | 是 | 24 | 32.257 | -5.95% | 247.94s |
| staggered isolate | 否 | 16 | 32.344 | -5.69% | 247.57s |

所有正式 async 实验成功率 100%、无 KV preemption；同步 staggered-16 出现 1 次
preemption，因此也不满足正式 baseline 门槛。

## Profiling 结论

仅打开 async 后，NPU busy 从同步 B1 的 87.19% 降到 81.21%，device free 从 12.81%
升到 18.79%。无 profiler 的正式吞吐也下降 1.56%，说明当前路径没有足够的 CPU rejection
空洞可供通用 async scheduler 覆盖；Orthrus 为取得精确 history 所需的同步状态又抵消了
通用异步流水的收益。

限制 draft batch 也未改善吞吐。限制越强，target/draft 随机流和请求组合变化越明显，
K32 draft acceptance 从 B1 的 `5627/96844=5.81%` 降至 batch-16 的 4.64% 和
batch-8 的 3.24%，导致为完成同样 8192 个输出 token 反而执行更多 draft block。
同时，小 draft batch 失去共享 backbone 的批处理效率。因此当前 staggered admission 是
可复现的负结果，默认保持关闭。

batch-16 的独立 Torch profile 进一步确认策略按设计执行：draft batch P95/max 从
async-only 的 23/26 压到 16/16，P95 每轮有 7 个请求被 deferred；segmented draft
attention 占 kernel time 从 23.61% 降到 17.76%。但省下的计算没有与别的 phase 重叠，
NPU busy 又从 81.21% 降到 77.57%，device free 升到 22.43%。这正是“只做 admission，
没有 mixed-phase forward”的结构性限制。

## 决策与下一步

不把 `--async-scheduling` 或 staggered admission 合入 B1。若继续这一方向，应实现真正的
Orthrus mixed-phase runner：为请求显式维护 draft-ready/verify-ready 状态，将不同请求的
target attention 与 draft attention 在每层分别执行，再把二者拼入同一个 MLP batch；
bounded delay、queue age、SLO slack 和 remaining work 应在该 runner 上控制 phase 配额，
而不是通过跳过 speculative block 间接控制。

进入该重构前先用本轮原始 trace 验证两个门槛：混合后的 MLP batch 利用率预计能否补回
至少 5% 端到端吞吐，以及额外状态/KV correction 是否能完全留在 NPU。否则应优先处理
profiling 已显示占 kernel time 约 24% 的 Orthrus draft attention。

## 复现数据

远端根目录：`/data/disk/wangzili/online-speculation/artifacts`

- B1：`tuning/paged-segmented-formal-64-out128-m94`
- async：`tuning/phase-async-delayed-verify-formal64-out128-m94-v1`
- staggered：`tuning/phase-staggered-formal64-out128-b{8,16,24}-v1`
- async Torch profile：`profiling/phase-async-delayed-verify-torch64-out128-v1`
- staggered Torch profile：`profiling/phase-staggered-torch64-out128-b16-v1`
- admission smoke trace：`tuning/phase-staggered-smoke8-out16-b2-v1`
