# Orthrus-DSV4.1 / Ascend

`feature/orthrus-dsv41` 面向 DeepSeek-V4.1-Flash-1017，目标是在 Ascend 上接入 Orthrus 的共享历史和块内双向起草。

- Draft 共享主干权重，读取目标 SWA、C1A 和 C2A 历史。vLLM-Ascend MRV2 推理已在单机八卡 A2 上完成相同候选回放对照。
- W4A8 教师反馈已用于末层 draft 更新，更新后的 BF16 head 已在单机八卡重载。离线训练使用同一套权重交换格式。
- 原生 verifier 的 K32 批量路径与逐步解码仍存在数值差异。正式 head 的吞吐、长上下文和并发验收待完成。

[训练交接](online-speculation/docs/ORTHRUS_DSV41_TRAINING_HANDOFF.md) · [参考实现与验证](online-speculation/docs/ORTHRUS_DSV41_REFERENCE.md) · [PyPTO 分支](https://github.com/wangzili121/online-speculation/tree/feature/orthrus-dsv41-pypto)
