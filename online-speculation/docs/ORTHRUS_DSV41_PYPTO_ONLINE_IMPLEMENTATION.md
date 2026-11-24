# Orthrus-DSV4.1 PyPTO 在线更新

## 当前状态

双机 16 卡已经跑通 DSV4.1 W8A8 的 target-only 与 Orthrus 推理：K32 draft、整块 verify、接受前缀提交、拒绝回滚均执行，greedy 输出逐 token 一致。初始化 head 只用于证明链路，不能代表投机收益。

线上反馈链路已经固定为以下顺序：

1. 两台机器在同一个 verifier round 中全局选择唯一请求。
2. 保存该请求同一 draft 版本的后缀边界、Attention 状态、mHC 状态、MoE 路由和 student logits。
3. verifier 只沿拒绝后修正的真实前缀补齐教师 logits。
4. 请求所属 DP 产生真实梯度，其他 DP 以零梯度参加同一组 collective。
5. 所有副本执行相同 optimizer step，并在 verifier 安全边界切换同一 head 版本。

`DSV41SuffixPlan` 支持 last-N；首轮实验扫描 last-1/2/4/8。当前 resident 反向只允许最后 20 层的 C1A 后缀，避免把 C1A 的反向错误套到 C2A 或 SWA。last-1/2/4/8 全部落在该范围内。

## 已实现的设备计算

以下 PyPTO 叶子已经有 Torch golden、编译测试和 910B3 数值对照：

- 完整词表 TV/KL 及 logits VJP，支持 DSV4.1 的 129280 维词表。
- 线性、RMSNorm、SiLU、梯度统计/裁剪、AdamW 和 BF16 发布。
- 含 sink、每个 query 独立的 SWA128＋Top-K C512 历史以及共享 K32 双向临时块的 C1A Attention VJP。历史保持冻结，只返回 dQ、临时块 dK/dV 和 sink 梯度；生产 TP4/K32 实卡数值已经通过。
- mHC post 数据/门控反向与 mHC pre 的 sigmoid、Sinkhorn 反向；后者严格复现原生 8×8 双 token block-diagonal 布局，避免近似重放引入跨 token 数值耦合。
- 固定 Top-6 路由下的 MoE combine 与归一化路由权重反向。
- router logits 的 sigmoid、Top-6 归一化 VJP 和 384 维稀疏回填；Top-K 成员固定使用前向记录，不在反向重新选择专家。
- EP16 固定路由编排：按专家 owner 发送六条 expert-output 梯度，回收 expert-input 梯度并恢复原 token/slot 顺序。
- 原生 EP MoE combine 在在线 draft 模式下保存六个选中专家的未加权输出，供 route-weight VJP 精确使用；生产 EP16/K32 形状编译及小形状 NPU 数值对照均通过。
- 冻结 W8A8 MoE 权重按原生分块布局在设备端反量化，供输入 VJP 使用；真实 `2304×5120` 和 `5120×4608` 尺寸已经编译通过，小形状 NPU 数值逐元素一致。
- routed expert 的 clipped-SwiGLU VJP 支持上界/双边 clamp，并明确以 STE 穿过动态 INT8 激活边界；生产 `K32×2304` 形状编译通过，小形状 NPU 结果与 Torch golden 逐元素一致。

完整的冻结 MoE Torch golden 已覆盖 routed experts、shared expert 和 router 三条输入梯度路径。当前生产规格共 61 个设备程序编译通过，包含 C1A 每 query 历史重放、`K32×384` router、`K32×5120` MoE/mHC、真实 LM head、TP4 draft 投影和 EP16 路由编排；22 组在线反向/优化器实卡数值用例全部通过。单机 readiness gate 同时在固定 serving 基线上实际应用补丁并做 Python 编译；当前证据为 157 项 DSV4.1 测试通过、4 项跳过、5 组子测试通过。

更新后的 draft 权重发布也已通过单卡实机验证：FP32 optimizer 候选状态在设备上转换为执行权重，并直接复制到 inactive resident head bank；验证记录显示 `host_round_trip=false`，不会在每次发布时把完整 head 绕回 CPU。

serving 生命周期也已补齐：online bootstrap 在首轮 prefill 创建 resident runtime、目标权重和 draft 后才构造 learner；一份 K32 capture 可以只训练当前已经到达的真实教师前缀；EOS 时不足目标宽度的短块仍可更新，但不会填充虚假行。TP4 梯度契约显式区分复制参数的求和和分片参数的拼接，并在 collective 前检查四条 lane 及八参数公开 ABI，避免 rank 顺序或权重方向错误。

`DSV41ResidentUpdateBackend` 将这些叶子组织成严格的一次更新：

```text
TV/KL
  -> frozen LM head input VJP
  -> final norm / mHC / fixed-route MoE input VJP
  -> trainable draft Attention VJP
  -> TP 梯度拼接
  -> DP all-reduce
  -> global norm / clip
  -> FP32 AdamW
  -> versioned head publication
```

服务中不允许隐式回退到 Torch autograd，也不允许用新一轮前向重建路由、Sinkhorn 或 Attention 选择。冻结 W8A8 主模型仍按原路径前向；只有反向传播到更早 draft 层时，才用反量化后的冻结权重计算输入梯度，不为这些权重创建梯度或 optimizer 状态。

## Resident 集成状态

上述叶子已经绑定到真实 resident session：W8A8 LM head 与专家权重只参与输入 VJP，不生成主模型参数梯度；EP16 固定路由、TP4/DP4 梯度归并、FP32 Adam 状态和 draft 双槽发布均已接入。单卡完整门禁验证了这些绑定能够编译、执行并满足数值契约，且发布不经过 CPU 往返。

尚未完成的是双机 16 卡真实 serving 中的一次在线更新验收。它不是新的算子开发，而是验证两个节点在同一 verifier 安全边界选择反馈、完成 EP16/TP4/DP4 同步并切换到相同 head 版本。通过后才能把“独立算子和单机 resident 更新已打通”提升为“双机在线更新端到端已验收”。

## 验收标准

1. 教师 logits 只来自主模型修正后的真实前缀。
2. 只有所选 last-N diffusion Attention 参数变化；主模型、indexer 和共享模块哈希不变。
3. loss、梯度范数和 optimizer 版本有限，16 个 rank 激活同一 head 哈希。
4. 开启更新后仍与 target-only greedy 输出逐 token 一致。
5. GSM8K 分离预热、调参和最终评测数据，并同时报告接受长度、更新/发布耗时和端到端吞吐。

冻结 MoE 专家的输入 VJP 将 INT8 权重按 128 行反量化 tile 执行，保留 BF16 边界和 FP32 累加。111 单张 910B3 上，16 行和 32 行的 routed expert 梯度核分别从 1.67 ms 降至 1.36 ms、从 1.70 ms 降至 1.38 ms；共享专家为 1.24 ms 降至 1.17 ms。四个 tile 尺寸用相同输入做驻留调度对照，梯度逐字节一致，未激活行保持零。以上耗时仅覆盖单次专家输入 VJP。
