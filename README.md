# Orthrus-DSV4.1 / Ascend

`feature/orthrus-dsv41` 面向 DeepSeek-V4.1-Flash-1017，目标是在 Ascend 上接入 Orthrus 的共享历史和块内双向起草。

- 模型保持原生因果验证。Draft 共享主干权重，读取目标 SWA、C1A 和 C2A 历史。
- 推理先验证 vLLM-Ascend MRV2 的 proposer、缓存绑定和校正，再核对 PyPTO 的检索与 Attention 接口。
- 训练计划以 W4A8 基模提供监督，先验证 BF16 draft 末层重放，再扩展到末四层。
- 当前阶段为设计与验证准备。单机八卡先复现原生主模型基线，再验证 Orthrus 接入。

[实施计划](online-speculation/docs/ORTHRUS_DSV41_PLAN.md)
