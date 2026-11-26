# Orthrus Qwen3.8 TP2 性能验证

OneServing TP2 固定到 [b99e118](https://github.com/ndleslx/OneServing/commit/b99e1188fad98b8e071e7b0bb90320381324ea4b)，即已合入的 [PR #3](https://github.com/ndleslx/OneServing/pull/3)。截至 2026-11-27，远端 main 仍指向此提交，PR #3 分支和合并提交的文件内容相同。[PR #2](https://github.com/ndleslx/OneServing/pull/2) 的草稿提供另一条 TP1/DP1 NZ 路径。本轮复用已经验收的 TP2 环境，Orthrus 代码基线为 c69d543。

## 起草结果解析和回滚

每轮把两个 rank 的候选字段一次读为 Python 数值，保留无效行、有限值、token 所属词表分片和零填充检查。平分时仍选择 ID 较小的候选。验证结果需要回滚时才准备控制缓冲区，并按相同的快照位置恢复状态。

在 C61 的原有 Torch 2.10.0 CPU 环境，针对 OneServing 的 packed greedy 解析完成 148 个结果和错误边界对照，5 组 CPU 稳态计时如下。相关 TP2、EOS、反馈及训练更新的 113 项 CPU 回归也通过。计时不包含设备搬运和模型计算。

| Draft 形状 | 原解析 | 批量解析 |
| --- | ---: | ---: |
| 32 行，7 个候选 | 1.468 ms | 0.127 ms |
| 64 行，14 个候选 | 2.875 ms | 0.184 ms |
| 64 行，62 个候选 | 3.062 ms | 0.225 ms |

复测入口为 `scripts/benchmark_qwen38_packed_draft_decisions.py`，使用同时可导入本仓和固定 OneServing 的环境。原始计时见 [CPU 记录](packed-draft-decisions.json)。

在同两张 A2 上实跑真实 27B、8 条 GSM8K 请求，每条生成 32 token，固定起草和 last-1 在线更新均与 AR 逐 token 一致。前后各有 102 个验证轮次，接受 133 个候选，在线组完成 6 次更新和 1 次发布。非耗时更新字段逐项相同。在线组 draft 累计耗时由 3.320 秒降至 3.129 秒，吞吐由 22.23 升至 22.92 tok/s。同期 AR 由 23.18 变为 23.54 tok/s。样本用于验证链路和计时方向。见 [双卡记录](serving-host-merge.json)。

## 共享缓冲区返回候选

最后一组 draft 直接返回两个 rank 的小型 greedy 决策包，去掉两次独立的设备回读。每种 batch 形状使用自身的共享缓冲区，在启动 worker 前分配。K32 的全词表 logits 仍供采样和训练反馈使用。

在真实 rank 词表尺寸上做了两种行数的独立对照。每种覆盖四组有效行，直接写回与旧式回读逐位一致。32 行中位耗时由 3.45 降至 1.85 ms，64 行由 3.74 降至 2.27 ms。复测入口为 `scripts/validate_qwen38_pypto_shared_draft_output.py <new.json>`，见 [双卡算子记录](shared-output.json)。

在同一对 A2、同一请求和 head 上，加入共享回读后，固定起草为 24.20 tok/s，在线更新为 23.10 tok/s，AR 为 23.51 tok/s。与改动前同配置相比，在线组的 draft 累计耗时由 3.320 秒降至 3.033 秒，在线吞吐提升 3.91%。greedy token、接受数和六次更新及一次发布的非耗时字段均相同。此对照覆盖 8 条短请求，同期 AR 计时也有 1.43% 波动。见 [真实 27B 记录](serving-shared-output.json)。

最终代码在固定 OneServing 环境中通过全部 1,173 项 Qwen3.8 PyPTO CPU 回归。

## 分页双向 Attention

尝试在 L1 区间完全被历史 KV 覆盖时省掉零填充。单卡 910B3 上以 TP2 本地头数 12/2、128 token 页、512 上下文、16 query tile 验证。18 组样例覆盖空历史、页边界、K8 和 K32，均与现有 PyPTO 路径逐位一致，并检查了有意用 NaN 填充的无效区域及只读输入。

整次 resident dispatch 的延迟降幅中位数约 0.07%。504 历史加 8 个块位置的设备任务窗口由 40.94 μs 降至 37.51 μs，各取两条采样，窗口数值仅用于定位。候选留在独立验证入口，serving 仍使用原有 Attention。以 `scripts/validate_qwen38_pypto_attention.py --backend paged --compare-full-history --page-size 128 --context 512 --q-heads 12 --kv-heads 2 --query-tile 16 --output <new.json>` 复测。见 [真实设备记录](device.json) 与 [设备 profiling 摘要](profile.json)。

## 在线更新

上一轮权重发布直写候选有 CPU 结构测试，但在固定 PyPTO 编译器上未通过。last-1 直写使用了不支持的条件表达式，候选修改后的 last-4 packing 将 INT32 与 INDEX 混用，导致原分支和直写分支均不能编译。此候选未进入代码分支。还需要验证跨 rank 读取渲染参数，以及第二次及后续发布是否会复用当前 serving bank。

## GDN 递推尾部清零

将跳过递推的无效行统一清零，再覆盖有效前缀，替代每行分别写零。用真实 TP2 局部 GDN 维度和非零独立历史验证，16 物理行的 5 组和 32 物理行的 8 组输出均与原有效行递推逐位一致，历史状态不变。调用时间降幅中位数约 0.41% 和 0.37%，单组约 −1.14% 至 +1.67%。保留在独立验证入口，serving 仍采用原有效行递推。以 `scripts/validate_qwen38_pypto_gdn_rows.py --compare-tail-clear --block-rows 16 --output <new.json>` 复测。见 [16 行](gdn-rows16.json) 和 [32 行](gdn-rows32.json)。

## 在线反向中的投影权重梯度

用真实 27B 尺寸逐项测试每个并行任务处理的输出 tile 数。4 和 8 相比原 16 都更慢。增加到 32 时，Q 投影权重梯度从 1.880 ms 降至 1.720 ms，O 投影从 1.357 ms 降至 1.292 ms，K/V 变化约 1%。这些独立算子的样本均与 Torch 逐位一致。把 Q/O 的分组应用到真实尺寸 last-1 的合并训练图后，逻辑宽度 8 和 32 的整次调用分别为 33.66 → 33.52 ms 和 33.66 → 33.38 ms。两种宽度下六个参数梯度、logits 和 loss 与原图逐位一致。训练图默认仍使用 16。

复测入口为 `scripts/benchmark_qwen38_pypto_weight_grad_tasks.py <new.json> --groups 16 24 32` 和 `scripts/benchmark_qwen38_pypto_suffix_tasks.py <new.json>`。见 [小分组记录](training-dw.json)、[大分组记录](training-dw-wide.json) 和 [合并图记录](training-suffix-tasks.json)。
