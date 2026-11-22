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
- 含 sink、历史和 K32 双向临时块的 C1A Attention VJP。
- mHC post 数据/门控反向与 mHC pre 的 sigmoid、Sinkhorn 反向。
- 固定 Top-6 路由下的 MoE combine 与归一化路由权重反向。
- 原生 EP MoE combine 在在线 draft 模式下保存六个选中专家的未加权输出，供 route-weight VJP 精确使用；生产 EP16/K32 形状编译及小形状 NPU 数值对照均通过。
- 冻结 W8A8 MoE 权重按原生分块布局在设备端反量化，供输入 VJP 使用；真实 `2304×5120` 和 `5120×4608` 尺寸已经编译通过，小形状 NPU 数值逐元素一致。

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

## 尚未完成的最后一层集成

当前缺口不是算法或反向公式，而是将上述叶子绑定到真实 resident session：

- 将已验证的 W8A8 输入 VJP 绑定到驻留 LM head、共享专家和选中专家句柄；权重冻结，不生成主模型参数梯度。
- 六个选中专家的最终输出已经由原生前向保存；还需保存 gate/up 激活或用驻留 W8A8 句柄等价重算，以完成每个选中专家的输入 VJP。
- 把 TP4/DP4 的设备句柄接入 HCCL 梯度归并。
- 将 FP32 master/moments 与 draft 双槽权重绑定，并把 controller/learner 注入真实 `NativeSession`。

完成这些绑定后，才能称为“PyPTO 在线更新端到端打通”。Torch 参考学习器、原子 kernel 通过或推理链路通过，都不能代替这项验收。

## 验收标准

1. 教师 logits 只来自主模型修正后的真实前缀。
2. 只有所选 last-N diffusion Attention 参数变化；主模型、indexer 和共享模块哈希不变。
3. loss、梯度范数和 optimizer 版本有限，16 个 rank 激活同一 head 哈希。
4. 开启更新后仍与 target-only greedy 输出逐 token 一致。
5. GSM8K 分离预热、调参和最终评测数据，并同时报告接受长度、更新/发布耗时和端到端吞吐。
