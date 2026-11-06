# Orthrus-Qwen3.8 PyPTO 接入计划

目标是在原生 Qwen3.8 模型之外完成 Orthrus 起草、验证和 head 换入；主模型公共算子的优化跟随上游。训练和在线更新继续走现有 PyTorch 路径，PyPTO 负责加载更新后的 head 并推理。

| 顺序 | 工作 | 验收方式 |
| --- | --- | --- |
| 1 | 固定分页 Attention 接口，复用 Torch 参考，增加融合计算候选 | 真实头数/维度，整块双向、部分有效块、分页边界、无历史；目标 KV 不变；与参考及原实现对照、计时 |
| 2 | 接回完整 draft；复用投影、norm、GDN、MLP 和 LM head | 逐层输出和最终 logits 对照；同一 head、同一历史下比较整轮耗时 |
| 3 | 完善 head 与状态管理 | 离线/在线导出的 head 加载与重启换入；共享 backbone；拒绝、异常、跨页与请求释放后无状态污染 |
| 4 | 随原生模型接口接通完整 TP 和服务验收 | 局部 Attention/GDN、输出归约、各 rank 接受长度与回滚一致；真实多卡 HTTP 和吞吐对照 |

已有基础：TP1 原生服务、K32 draft、共享 backbone、目标历史只读、短请求验证/恢复及 HTTP 检查；head 的 TP2/4 布局，以及两卡驻留、输出投影和归约组件已单独验证。完整 TP 服务尚未接通。

首个 kernel 的边界为：已完成 norm/RoPE 的 Q、分页历史 KV、临时块 KV、有效长度及输出门控 → 门控后的 Attention 输出。O projection、TP 归约、residual 和 MLP 保持独立。历史与块内位置参与同一个 softmax，目标 cache 不作输出。

融合候选先保留原生数值舍入位置，使用独立 Torch 参考检查；编译通过不等于实机正确，单层提速也不等于完整请求提速。新实现经完整验证后再调整默认后端。

当前进度：第 1 项已有独立 Torch 参考、融合/分阶段对照和 14 组 A2 实测；融合后端已接回完整 64 层，真实 27B 的 K32 起草、目标 cache 只读、拒绝恢复和短请求 greedy 检查通过。352 项 CPU 回归通过，相关卡已释放。

接下来补逐层 Torch 数值对照、同一 head 的整轮性能比较及 head 重载；随后扩展请求边界和服务验收。完整 TP 的执行接入需要原生模型提供局部 Attention/GDN/MLP 和归约接口；现有 head 分片与双卡组件继续作为其对接基础。具体 kernel 的输入、参考、验证和性能口径见 [任务说明](QWEN38_PYPTO_KERNEL_TASK.md)。
