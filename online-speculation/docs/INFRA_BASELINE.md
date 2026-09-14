# Orthrus 单卡 Infra 基线

本阶段冻结算法，只调系统。唯一固定配置见
`configs/orthrus_single_npu.toml`。其中 `block_size=32` 在 vLLM 接口中对应
`num_speculative_tokens=31`：一个已提交 anchor 加 31 个候选组成 32 行
verify，不是 32 个候选。

## 基线边界

- B0 是上游独立 PyTorch 实现，用于记录原始执行形态。
- B1 是共享物理 backbone 的 vLLM-Ascend 0.23.0 实现。
- 普通 AR 只做 greedy、随机分布、rejection correction 和 KV crop
  正确性 oracle，不参与吞吐选参。
- `runtime/vllm_ascend_v0_23` 只补齐 custom proposer、概率校正和 K32
  attention fallback，属于兼容接入，不计优化收益。
- 独立加载第二份完整 draft/backbone 的实现只能作为接线 smoke，不能命名为 B1。

## 当前完成度

- [x] 导入并记录上游 SHA。
- [x] 支持单文件和 sharded safetensors 权重。
- [x] PyTorch runner 支持 CPU/CUDA/NPU 设备抽象。
- [x] NPU tiny 模型 fixed-draft exact correction smoke。
- [x] NPU tiny 模型 online TV backward/publish smoke。
- [x] vLLM-Ascend 0.23 custom proposer 与概率校正兼容补丁。
- [x] 官方 Orthrus-Qwen3-8B B0 fixed、greedy 和 online TV NPU smoke。
- [x] vLLM shared-backbone Orthrus 模型和 paged-KV draft view。
- [x] PyTorch AR 与 vLLM AR 在相同模板下连续 64 个 greedy token 一致。
- [x] B1 eager 与 `FULL_AND_PIECEWISE` 单请求 smoke，后者确认 ACL Graph replay。
- [x] 256 条 6K--40K 混合编码 workload 和流式 TTFT/TPOT 测量入口。
- [x] B1 单卡 256-token 粗搜索收敛。
- [x] B1 1024-token 正式稳定性复验：native-GQA 在 memory 0.94 下
  64/64 成功，输出吞吐 45.68 token/s。
- [x] Torch-NPU、MS Service Profiler 和 Orthrus request-aware trace 采集入口。
- [x] 第一份饱和 Torch-NPU 原始 profile 与可重算派生摘要。
- [x] native-GQA B1 正式 Torch-NPU profiling 与有效 MS Service Profiler trace。
- [x] batched paged-segmented draft attention 数值探针、微基准和真实服务 smoke。
- [x] batched paged-segmented 64-request 正式 A/B 与复验 profiling。
- [ ] Draft/Verify 协同调度、异步续训、sparse draft KV、DDTree。

## 验收纪律

所有性能实验使用同一模型、请求 manifest、seed、算法参数和系统配置。
系统配置先按 SLO goodput 选 B1；后续每项优化只和 B1 A/B。单项收益不足
5% 不保留。自定义算子仅在 trace 同时满足计划中的 busy-time、现有算子
不可消除和预期端到端收益门槛后启动。

## Smoke 结论

- B0 K32 fixed draft、一次 online TV update、参数发布与恢复均已在 8B
  checkpoint 上通过。
- B1 使用一份 18.1 GiB 模型权重；drafter 只持有 runner 引用，没有加载第二份
  backbone。
- `FULL_AND_PIECEWISE` 可编译并 replay；短请求 smoke 的时间受编译缓存和预热
  影响，只用于 capability 判断，不作为收益数字。
- 第一条 7.7K prompt、1024-token 服务请求成功；它验证 streaming 指标采集，
  并非并发性能结论。
- vLLM 0.23 原生会关闭 `custom_class` speculative decoding 的 async scheduling；
  现已完成 Orthrus 兼容与正确性验证，但正式 A/B 为负，因此仍不在 B1 启用。

## 饱和探针与搜索裁剪

首轮 `MNS=64 / MBT=64K / memory=0.94 / FULL_AND_PIECEWISE` 的
`64 request burst x 1024 output` 是容量探针，不是正式基线。它运行
1800.75 秒，63/64 条在客户端超时前完成，输出吞吐为 35.65 token/s；最后
一条 39K prompt 的首 token 等待约 1667 秒并触发 1800 秒客户端超时。
运行期间观测到 NPU AICore 100%、KV 峰值约 98%、持续 server backlog 和
1 次 preemption，说明负载已经饱和，但该配置不满足 100% 成功门槛。

这个探针也说明当前首先受 KV 容量而不是 MNS 上限约束：`MNS=64` 时实际
running request 峰值约 25。系统粗筛因此使用完全相同 prompt 的 256-token
manifest，先比较 memory 0.90/0.94/0.98 和 eager/graph，再搜索 MBT。若最佳
memory 下 running request 仍显著小于 64，则只补一个 `MNS=128` 无效性验证，
不机械运行 256/512；只有 running request 真正触及 MNS 上界才继续扩展。
最终胜出配置必须回到原 1024-token manifest，使用 3600 秒客户端超时完成
100% 成功复验。

## 256-token 粗筛结果

以下试验均使用同一组 64 条请求，成功率均为 100%；`0.98` 除外，它在真实
K32 warmup 中因显存不足失败。吞吐单位为 output token/s。

| MNS | MBT | memory | mode/capture | 吞吐 | preemption | 结论 |
|---:|---:|---:|---|---:|---:|---|
| 64 | 64K | 0.94 | eager | 26.72 | 0 | eager 对照 |
| 64 | 64K | 0.90 | graph/512 | 26.28 | 3 | 淘汰 |
| 64 | 64K | 0.94 | graph/512 | 27.13 | 1 | graph 有效 |
| 64 | 64K | 0.98 | graph/512 | OOM | - | 淘汰 |
| 64 | 64K | 0.94 | graph/1024 | **27.93** | 1 | 当前最佳 |
| 64 | 16K | 0.94 | graph/512 | 26.92 | 1 | 淘汰 |
| 64 | 128K | 0.94 | graph/512 | 25.14 | 0 | 淘汰 |
| 128 | 64K | 0.94 | graph/1024 | 27.27 | 0 | 淘汰；running 峰值仅 25 |
| 64 | 64K | 0.96 | graph/1024 | 20.82 | 1 | 淘汰 |

粗筛胜出配置为 `MNS=64 / MBT=64K / memory=0.94 /
FULL_AND_PIECEWISE / max graph capture=1024`。上界不是未经验证的边界：
MNS 128、MBT 128K 和 memory 0.96/0.98 均已覆盖，且分别变慢或失败；因此
不再机械扩展 MNS 256/512、MBT 256K 或 memory 0.97。`async scheduling`
的 Orthrus 兼容层和性能实验见 [Draft/Verify 协同调度实验](DRAFT_VERIFY_SCHEDULING.md)。

## 1024-token 稳定性反例

将 256-token 胜出配置原样用于 64-request、1024-token 输出时，服务在完成
23/64 条后崩溃，因此 `memory=0.94` 不能成为正式 B1。崩溃前实际 running
约 20、waiting 约 21、KV 使用率约 87%，错误来自 Orthrus draft attention
的 `aclnnStack`，底层明确报告 device memory allocation failure。这里的
额外显存不是普通 KV 配额，而是当前 eager draft 路径为每个请求、每层收集
完整历史 KV，并将 GQA KV 展开到全部 query heads 所产生的临时张量。

expanded-GQA 将 KV 预留降低到 `memory=0.90` 后完成了相同正式负载：成功率
100%，吞吐 34.86 output token/s，但发生 11 次 KV preemption。它证明降低
KV 配额可以绕过临时工作区 OOM，却会把压力转成 KV 抢占，因此只保留为
expanded control，不作为最终 B1。native-GQA 在 memory 0.94 下的正式复验
用于判断能否同时保留较大 KV 容量并消除临时展开。

## 首份饱和 NPU Profile

`64 requests x 128 output` 的 Torch-NPU 采集窗口在持续 backlog 下开始：
开始时 running=10、waiting=54，完整 workload 成功率 100%，无 preemption。
10 秒窗口中 NPU computing 占 97.18%，因此仅增加 admission 或外部并发不是
首要方向。原始 `ascend_pt` 保留了 `trace_view.json`、`analysis.db`、
`operator_details.csv`、`kernel_details.csv` 和 `task_time.csv`。

按 kernel time 聚合的主要开销为：

| 算子/路径 | NPU kernel time 占比 |
|---|---:|
| Transpose | 27.27% |
| RepeatInterleaveV2 | 18.40% |
| FlashAttentionScore（draft） | 17.78% |
| GatherV3 | 6.81% |
| ConcatD | 4.73% |
| FusedInferAttentionScore（verify） | 7.85% |

`Transpose + RepeatInterleaveV2 + GatherV3 + ConcatD` 合计 57.20%；加上 draft
attention 后为 74.98%。这证明当前第一瓶颈不是 verifier、rejection sampler
或调度空洞，而是 Orthrus draft view 将 8 个 GQA KV heads 显式扩展成 32 个
query heads，并为每个请求、每层 materialize 连续历史 KV。

同一张 910B3 上的 32K-history 定点微基准显示，CANN 已有
`npu_fusion_attention_v3` 可以原生接收 32 query heads / 8 KV heads：与旧路径
逐元素完全一致，attention 从 2.326 ms 降到 0.569 ms（4.09x），单次调用的
增量显存峰值从约 1.03 GiB 降到 33.8 MiB。该项属于把硬件现有能力接入 B1，
不计作后续算法专属创新收益。正式 B1 因此保留 `expanded` control，并默认使用
`native_gqa`；两者通过 `ORTHRUS_DRAFT_ATTENTION` 显式选择并写入实验 manifest。

同一份 64-request、128-output manifest 的无 profiler 端到端 A/B 结果为：

| draft attention | output tok/s | P95 latency | P95 TTFT | P95 TPOT |
|---|---:|---:|---:|---:|
| expanded | 20.79 | 381.19 s | 326.70 s | 0.736 s |
| native GQA | **25.00** | **318.56 s** | **278.91 s** | **0.539 s** |

native GQA 使 output throughput 提升 20.25%，P95 latency 降低 16.43%，P95
TPOT 降低 26.73%，且两组成功率均为 100%。两组运行在两台同型号 910B3 上；
这是快速并行 A/B，后续正式 native profile 和 1024-output 复验在固定卡上
完成。该收益用于确定 B1 的正确实现，不计入 B1 之后的创新收益。

native-GQA 下补测 `memory=0.98` 仍然失败：KV cache 预留后设备只余约
986 MiB，backbone 的 SwiGLU 需要申请 1.50 GiB，64 条 workload 请求成功率
为 0%。因此 0.98 是经过验证的不可用边界，不再扫描 0.97/0.99。实验 runner
已增加 `success_rate == 1.0` 硬校验，失败负载不会再被错误标为 `COMPLETE`。

MS Service Profiler 首次窗口虽然完成了 64 请求，但产出的 DB 为空。根因是
26.0.0 handlers 实际使用 `Engine/Schedule/Schedule.KVCache/Execute` domain，
而公开指南中的 domain 名称不同，旧白名单静默过滤了全部事件。正式采集改为
空白名单，即保留所有 domain；这次空 DB 标记失败并保留，不作为 profiling
证据。

## native-GQA 正式 B1

`memory=0.94` 下的 64-request、1024-output 正式负载已经 64/64 完成：输出
吞吐为 45.68 token/s，P95 latency 为 1366.30 秒，P95 TTFT 为 1201.66 秒，
P95 TPOT 为 0.426 秒。相对 expanded-GQA 在唯一可稳定运行的 memory=0.90
对照 34.86 token/s，吞吐提升 31.03%。该结果说明 native GQA 不仅消除了
head expansion，还使 0.94 的较大 KV 配额成为可运行配置。运行中仍发生 3 次
KV preemption；由于总吞吐和尾延迟优于 0.90 对照，preemption 作为 SLO 与容量
信号记录，不单独作为淘汰条件。

有效的 MS Service Profiler 窗口包含 21 个 spec-decode batch，每批 22--23 个
请求，P50 draft 执行为 590.38 ms，而 P50 scheduler framework processing 只有
1.403 ms，即 0.24%。KV usage P99 为 100%，有 3 个采样点达到 99.9%以上。
原始 `profiler.db`、`chrome_tracing.json`、batch/request/KV/spec-decode CSV 和
可重算 `service-derived-summary.json` 均已保留。

native-GQA 的独立 Torch-NPU 窗口暴露出第二层瓶颈：NPU busy 只有 52.23%，
`GatherV3 + ConcatD` 仍占 19.22% kernel time；更关键的是 432 次 draft layer
调用产生了 9,936 次 native draft attention，即每层平均 23 次。当前 Python
实现仍逐 request 收集完整历史 KV，并逐 request 发 attention；head expansion
虽已消失，但 host launch fragmentation 和历史 KV materialization 成为新瓶颈。

## Batched Paged-Segmented Draft Attention

针对 Orthrus 固定的两段 KV 结构，新增候选后端 `paged_segmented`：

1. 历史 AR KV 不再 gather，直接使用 vLLM paged KV 与 block table 完成一次
   variable-length TND attention。
2. 32-token 临时 draft KV 作为第二段，以一次 TND attention 覆盖整个 active
   request batch。
3. 两段返回 FP32 softmax LSE，通过 exact online-softmax merge 合成结果；不
   改变 verifier、rejection correction 或 target distribution。
4. batch width 至少为 2 时使用合批分段路径；单请求历史达到 24K 时也使用，
   单请求短上下文继续使用 native GQA。

单请求微基准显示 8K/16K 分段路径较慢，24K/32K/40K 分别快约
1.21x/1.40x/1.86x；因此单请求门槛固定为 24K。真实长度分布的 8-request
单层探针中，合批分段路径相对逐请求 native GQA 快 3.68x，临时峰值少约
303 MiB；只有两条 7--8K 请求时仍快 1.23x。所有形状与 materialized
reference 均满足 `rtol=atol=0.02`，实测最大绝对误差不超过 4.89e-4。

第一组真实 8-request 服务 smoke 包含 7K 与 38--39K prompt，8/8 成功、无
preemption。逐请求分段版本为 25.35 output token/s，最终 batched 版本为
33.09 output token/s，端到端提升 30.54%，P95 latency 从 40.30 秒降至
30.82 秒。

同一份 64-request、128-output workload 的正式结果为 34.30 output token/s，
成功率 100%、无 preemption；native-GQA B1 对照为 25.00 token/s，因此创新
路径相对 B1 端到端提升 37.17%。P95 latency 从 318.56 秒降至 233.94 秒，
下降 26.56%。

独立 Torch-NPU trace 也与端到端结果一致：

| 指标 | native GQA B1 | batched paged-segmented |
|---|---:|---:|
| NPU busy | 52.23% | 87.19% |
| draft KV materialization kernel time | 19.22% | 0.67% |
| history-attention calls / draft layer | 23.0 | 1.0 |
| mean draft batch | 399.01 ms | 138.51 ms |

新路径中 `FusedInferAttentionScore` 合计占 41.75%，其中按 profiler span
扣除 paged-history 与 local draft attention 后，verifier FIA 约占 20%；MatMul
占 37.11%。NPU 仍有约 12.81% free time，sampler/random 路径约占 4%--5%，
它们是后续调度与采样实验的证据入口，但不能再把已消除的 KV materialization
写成当前瓶颈。

## Draft/Verify 调度尝试

已实现并验证 vLLM async correction 兼容，以及按 queue age 和 history bucket
限制 K32 draft fan-in 的 staggered admission。64-request 正式 A/B 中，async-only
相对当前 B1 吞吐下降 1.56%；admission batch 8/16/24 分别下降 6.86%/3.85%/5.95%。
Torch trace 显示 async-only NPU busy 为 81.21%，batch-16 admission 又降至 77.57%，
所以这两项均默认关闭。代码和负结果保留用于复现；详细机制、数据路径及下一步
真正 mixed-phase runner 的门槛见 [独立实验报告](DRAFT_VERIFY_SCHEDULING.md)。
