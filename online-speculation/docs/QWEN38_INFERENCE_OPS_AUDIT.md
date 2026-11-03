# Qwen3.8 Draft 推理算子审计

验证日期：2026-11-03。真实模型使用 Qwen3.8-27B、Ascend 910B3、TP2、K32；运行环境沿用 [固定镜像与部署配置](QWEN38_VLLM.md)。本轮代码为 `31922e2`，不修改投机算法、head 格式或主模型 kernel。

## 采用的改动

单请求且所有位置有效时，draft 的 Full Attention 可以读取整段历史与整个临时块。全零加性 mask 与不传 mask 表示相同的可见性。
实现将 draft mixer 显式设为 `is_causal=False`，使用 `None` mask，省去 mask 的创建、边界捕获复制和各层的禁止访问 mask 转换；AR 仍使用原有因果 mask。
这不是去掉双向约束，而是利用已有融合 Attention 的全可见入口。未来增加 padding 或并行 batch 时，需要另行处理无效位置和请求隔离，不能直接延用这一假设。

## 算子对照

测试使用 TP2 的实际 head 规格：12 个 Q head、2 个 KV head、head_dim=256。预热后，每组 50 次、重复 5 组，以下为含 mask 准备的 Attention 调用中位数：

| Query 行数 | 总 KV 长度 | 显式零 mask | 不传 mask |
| --- | --- | --- | --- |
| 32 | 32 | 0.124 ms | 0.081 ms |
| 32 | 4096 | 0.135 ms | 0.130 ms |
| 32 | 8192 | 0.245 ms | 0.240 ms |

另覆盖 query 行数 1/2/8/17，1 仅为单 token 算子边界。七组前向均逐 bit 相同；Q/K/V 梯度最大相对 L2 差异为 `3.74e-5`，不是所有梯度逐 bit 相同。
这些结果仅说明该调用的开销减少，短上下文更明显，不代表整网或服务吞吐同比提升。原始计时、梯度及内存记录见 [算子对照](validation/qwen38/inference-ops-20261103/history-and-mask-final.json)。

还比较了两种历史读取改写，但均未采用：

- 分页 KV 改为逐 token 索引或 gather：32K 历史下，原分页路径约 0.145 ms，另两种约 0.231 / 1.311 ms，临时内存也更多。
- GDN 卷积历史先切三行、再选状态槽：未减少耗时，临时分配从约 0.33 MiB 增至 1.20 MiB。

因此仍保留现有的分页快照与 GDN 状态读取。上述比较没有证明零拷贝分页 Attention 不值得做，只说明这两种 Torch 索引顺序调整没有收益。

## 真实模型复验

固定 head 在真实 27B 上完成 6 类请求：greedy、采样、连续拒绝、分块 prefill、两 token 输出边界及跨页请求。
每个 rank 的 59 次完整 draft 前向均与显式零 mask 路径逐 bit 相同；两 rank 的起草/验证/接受计数一致，各完成 18 次缓存只读检查，主模型完整参数哈希未变。见 [固定 head](validation/qwen38/inference-ops-20261103/fixed.json)。

另从 WikiText 真实语料取两个随机窗口，完成两步全 64 层 draft 更新：所有层都有梯度，两 rank 的 loss/梯度范数一致，draft 参数改变，主模型完整参数哈希未变，并导出 head。见 [离线更新](validation/qwen38/inference-ops-20261103/offline.json)。
独立进程加载该 head，加载哈希与导出一致；greedy、采样两个请求中各 rank 完成 8 次 last-4 更新及 6 次缓存只读检查。head 再次改变，主模型仍冻结；在线阶段另有 10 次 draft 前向与零 mask 路径逐 bit 相同。见 [重载与在线更新](validation/qwen38/inference-ops-20261103/online.json)。

这验证的是优化后的推理、真实语料短训、导出重载与在线续训链路，不评价 head 收敛、接受率或业务加速收益。
完整 CPU 回归为 **639 passed、32 skipped**，见 [JUnit 记录](validation/qwen38/inference-ops-20261103/cpu.xml.gz)；跳过项不算硬件验证通过。

## 复现入口

`scripts/benchmark_qwen38_history.py --output <结果.json>` 比较现有算子的前向、梯度、延迟和临时内存。
在 `scripts/validate_qwen38_vllm.py` 的验证环境中设置 `ORTHRUS_VALIDATE_DRAFT_MASK=1`，会将每轮 draft 与显式零 mask 重放对照；该检查只在验证 worker 中启用，不进入正常服务路径。
实验进程和容器结束后释放测试卡，不保留占卡任务。

## GDN 状态与布局

本节对应核心提交 `fe12ee2`，不改变 GDN 的因果递推定义。

`recurrent_slices` 继续调用镜像的递推 kernel，保留每个候选位置的 checkpoint。`index_select` 已经返回独立存储，因此不再追加 `clone`；末段结束后也不再读取用不到的尾状态。只有一个片段时直接返回算子输出，不额外拼接。

Q/K/V 和 gate 在分段前统一整理为连续布局，后续沿 token 维切片直接传给 kernel。这样 K32 的两个片段不必分别打包同一组输入。draft 临时状态区只初始化第一行，其余行由递推 kernel 写出；不会把未初始化存储交给主模型作历史。

状态测试覆盖乱序槽位、上轮接受位置大于本轮长度、跨 16 行分段以及没有写到的状态槽。另将临时状态的后续行填成 NaN，与全零工作区比较，确认输出和全部 checkpoint 一致。主模型历史仍是只读快照，draft 状态不转为正式缓存。

`scripts/audit_qwen38_execution.py --model <本地模型> --output <结果.json>` 对照原适配逻辑和当前逻辑，包括完整 draft 调用；`scripts/probe_qwen38_execution_ops.py` 检查融合 Attention、卷积、GDN、norm 和 SwiGLU 的前向及可微路径。以上均复用已有算子，不新增 kernel。

TP2 实际规格下，完整 GDN draft 适配调用的 K8 为 1.178 → 1.045 ms，K32 为 1.553 → 1.493 ms；输出和最终状态逐 bit 一致。另覆盖 K1/K17、乱序状态槽、上轮接受边界及 NaN 工作区，全部 checkpoint 对照一致。计时包含适配层调用，不代表服务吞吐同比提升，见 [算子记录](validation/qwen38/execution-20261103/operators.json)。

真实 27B 的固定 head 另复验六类请求，greedy 与同配置 target-only 逐 token 一致；各 rank 完成 56 次历史读取和 18 次只读检查，主模型与固定 head 哈希均未变。见 [固定 head](validation/qwen38/execution-20261103/fixed-tp2.json) 和 [基线](validation/qwen38/execution-20261103/target-tp2.json)。

适配层另在 NPU 上直接检查融合卷积、GDN、门控 norm、SwiGLU、GQA 和完整 Full Attention 的前向及可微路径。GDN 的 K1/8/17/32 空历史与显式零初态逐 bit 相同；Full Attention 的因果与双向模式均通过数值门槛，历史输入保持不变。见 [设备适配层检查](validation/qwen38/execution-20261103/device-adapters.json)。这些是算子与适配层检查，不代替整模型验收。

四卡设备驻留的离线训练、head 重载和在线 last-1/last-4 更新结果见[训练审计](QWEN38_TRAINING_OPS_AUDIT.md#四卡设备驻留验证)。它们验证更新后继续起草和验证的链路，不将在线期间变化的 head 与固定 head 作逐 token 对照。
