# Orthrus-DSV4.1 / Ascend

`feature/orthrus-dsv41` 面向 DeepSeek-V4.1-Flash-1017，目标是在 Ascend 上接入 Orthrus 的共享历史和块内双向起草。

- 模型保持原生因果验证。Draft 共享主干权重，读取目标 SWA、C1A 和 C2A 历史。
- 推理先验证 vLLM-Ascend MRV2 的 proposer、缓存绑定和校正，再核对 PyPTO 的检索与 Attention 接口。
- 训练计划以 W4A8 基模提供监督，先验证 BF16 draft 末层重放，再扩展到末四层。
- 已实现 1017 层角色解析、BF16 head 校验和 Torch Attention 参考。Ascend 910B3 已通过 C2A Full/Reuse 的 K32 算子验证。下一步接入原生分页检索和 MRV2。

[实施计划](online-speculation/docs/ORTHRUS_DSV41_PLAN.md) · [参考实现与验证](online-speculation/docs/ORTHRUS_DSV41_REFERENCE.md)
