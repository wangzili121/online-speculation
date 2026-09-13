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
- [ ] B1 1024-token 正式稳定性复验：memory 0.94 已失败，0.90 正在复验。
- [x] Torch-NPU、MS Service Profiler 和 Orthrus request-aware trace 采集入口。
- [ ] B1 正式 profiling。
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
- vLLM 0.23 当前会关闭 `custom_class` speculative decoding 的 async scheduling。
  该项在兼容与正确性验证前不能写成“已启用”。

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
被 vLLM 0.23 的 `custom_class` speculative backend 明确禁用，记录为能力限制，
不伪装成性能结论。

## 1024-token 稳定性反例

将 256-token 胜出配置原样用于 64-request、1024-token 输出时，服务在完成
23/64 条后崩溃，因此 `memory=0.94` 不能成为正式 B1。崩溃前实际 running
约 20、waiting 约 21、KV 使用率约 87%，错误来自 Orthrus draft attention
的 `aclnnStack`，底层明确报告 device memory allocation failure。这里的
额外显存不是普通 KV 配额，而是当前 eager draft 路径为每个请求、每层收集
完整历史 KV，并将 GQA KV 展开到全部 query heads 所产生的临时张量。

因此正式复验先降低 KV 预留到 `memory=0.90`，为 Orthrus 临时工作区留出
空间。若仍失败，再验证 0.88；这属于让固定算法 workload 稳定运行的系统
配置选择。后续 profiling 会单独量化 KV gather、GQA materialization 和
draft attention，不能用短输出粗筛掩盖这一专属路径。
