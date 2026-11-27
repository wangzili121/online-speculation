# Orthrus-DSV4.1 / Ascend

单机 16 卡 PyPTO A3 的 smoke、GSM8K 实验和配套代码版本见 [A3 验证说明](online-speculation/docs/ORTHRUS_DSV41_PYPTO_A3.md)。

`feature/orthrus-dsv41` 面向 DeepSeek-V4.1-Flash-1017，目标是在 Ascend 上接入 Orthrus 的共享历史和块内双向起草。

- Draft 共享主干权重，读取目标 SWA、C1A 和 C2A 历史。vLLM-Ascend MRV2 推理已在单机八卡 A2 上完成相同候选回放对照。
- W4A8 教师反馈已用于末层 draft 更新，更新后的 BF16 head 已在单机八卡重载。离线训练使用同一套权重交换格式。
- PyPTO 的 K32 SWA、块内双向和压缩历史 Attention 已在单张 A2 上与 Torch 对照。整模 Serving 还需要 A2/A3 的 V4.1 W4A8 基线和 draft 检索接入。

[训练交接](online-speculation/docs/ORTHRUS_DSV41_TRAINING_HANDOFF.md) · [参考实现与验证](online-speculation/docs/ORTHRUS_DSV41_REFERENCE.md) · [PyPTO 接入](online-speculation/docs/ORTHRUS_DSV41_PYPTO.md)
